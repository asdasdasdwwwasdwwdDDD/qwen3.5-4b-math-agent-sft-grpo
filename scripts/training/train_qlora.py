# -*- coding: utf-8 -*-
"""
QLoRA(AdaLoRA) 微调 Qwen3.5-4B 增强智能体（工具调用）能力
- 4bit NF4 量化 + 双重量化，LoRA 挂载到全部线性投影（含 GatedDeltaNet 线性注意力）
- 增量渲染 chat template 计算 assistant 区间 loss mask（仅训练模型自己的回复：
  包括 <tool_call> XML 调用、tool 后的续写、最终回答）
- 8GB 显存适配：bf16 + 梯度检查点 + 8bit AdamW + 分块交叉熵 + 冻结权重压回 bf16 +
  评估只算 loss —— 避免 Windows/WDDM 显存打满后换页到内存（换页会让每步从秒级变分钟级）
用法:
  python -m scripts.training.train_qlora --model models/Qwen3.5-4B --train data/tora_sft/train.jsonl --val data/tora_sft/val.jsonl --out outputs/qlora-tora
"""
import argparse
import atexit
import gc
import json
import os
import sys

# Windows/WDDM：显存打满后驱动会把显存页换出到"共享显存"（内存），每步从秒级恶化到分钟级
# （GPU 利用率 100% 但功耗只有 30W）。expandable_segments 在 Windows 上不支持，改用
# 垃圾回收阈值：缓存块接近上限时提前归还驱动，给瞬时峰值留出空间。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import AdaLoraConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from bitsandbytes.nn import Params4bit

IGNORE_ID = -100


def make_chunked_ce_loss(chunk=256):
    """分块交叉熵，与 transformers 的 ForCausalLMLoss 数值等价。

    Qwen3.5 词表 248k：原实现 `logits.float()` 会把 [L, 248k] 整段复制成 fp32（L=900 时
    约 0.9GB），再加 softmax 输出又一份，合计约 2GB 瞬时显存——8GB 卡上这就是峰值来源，
    显存吃满后 Windows/WDDM 会把显存换出到内存，每步从几秒恶化到几分钟。
    按行分块后瞬时只需 chunk×248k×4B（256 行约 250MB），梯度与逐位置 softmax 完全一致。"""

    def _loss(logits, labels, vocab_size, num_items_in_batch=None,
              ignore_index=IGNORE_ID, shift_labels=None, **kwargs):
        if shift_labels is None:
            labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = shift_labels.reshape(-1)
        total = flat_logits.new_zeros((), dtype=torch.float32)
        n_valid = 0
        for i in range(0, flat_logits.shape[0], chunk):
            lb = flat_labels[i:i + chunk]
            nv = int((lb != ignore_index).sum())
            if nv == 0:
                continue
            total = total + torch.nn.functional.cross_entropy(
                flat_logits[i:i + chunk].float(), lb,
                ignore_index=ignore_index, reduction="sum")
            n_valid += nv
        denom = num_items_in_batch if num_items_in_batch is not None else max(n_valid, 1)
        return total / denom

    return _loss


def drop_vision_tower(model):
    """纯文本训练不需要视觉塔（Qwen3.5-4B 是 VL 架构，视觉塔 168.9M 参数、4bit 约 0.16GB）：
    从模块树摘掉并归还显存。注意必须先删局部引用再 gc，否则只是解注册、显存不会释放。"""
    vis = getattr(getattr(model, "model", None), "visual", None)
    if vis is None:
        return 0
    n = sum(p.numel() for p in vis.parameters())
    del model.model.visual
    del vis
    gc.collect()
    torch.cuda.empty_cache()
    return n


def shrink_frozen_fp32(model):
    """把 peft 升到 fp32 的冻结权重压回 bf16。

    prepare_model_for_kbit_training 会把所有非量化权重从 bf16 升到 fp32（为 8bit 优化器
    保精度）。Qwen3.5 词表 248k，词嵌入（与 lm_head 共享）一个张量就是 2.37GB，而且它
    会让 logits/梯度也变成 fp32（896 token 时各 0.89GB）——这是 8GB 卡峰值随序列长度暴涨、
    触发 WDDM 换页的根源。冻结权重不参与优化器更新，压回 bf16 只省显存不损数值
    （4bit 底座精度本来就更低）。共享的嵌入/头部（tied）只处理一次，tie 保持不变。"""
    n = 0
    for p in model.parameters():
        if not isinstance(p, Params4bit) and p.dtype == torch.float32 and not p.requires_grad:
            p.data = p.data.to(torch.bfloat16)
            n += 1
    if n:
        gc.collect()
        torch.cuda.empty_cache()
    return n


# LoRA 目标：Qwen3.5-4B 是 3:1 混合注意力（24 层 GatedDeltaNet 线性注意力 + 8 层 softmax
# 注意力），两种注意力的全部线性投影 + 稠密 SwiGLU FFN 都要挂。PEFT 按模块名 endswith
# 匹配，带父模块前缀可避免命中 vision tower 的同名层（vision 用的是 attn.qkv/linear_fc*）。
# 模型为 Qwen3_5ForConditionalGeneration（VL 架构），文本层前缀是 model.language_model.*。
TARGET_MODULES = [
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b",
    "linear_attn.in_proj_a", "linear_attn.out_proj",          # 线性注意力（24 层）
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",  # 全注意力（8 层）
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",          # 稠密 FFN（全部 32 层）
]

# Qwen3.5 是"思考/非思考"双模模型：智能体轨迹为非思考模式，chat template 会在
# assistant 轮插入空 <think></think> 块（由 enable_thinking=False 控制）。
CHAT_TMPL_KW = {"enable_thinking": False}


class AdaLoraTrainer(Trainer):
    """AdaLoRA 需要在每个优化步后更新秩预算分配（importance 排序 -> 重分配 rank）。
    peft 在最后强制掩码步（global_step == total_step - tfinal）会跳过分数更新却仍读取
    全部层的得分，触发 KeyError；该步仅出现一次，失败时沿用上一次正常分配的掩码即可。"""

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        m = model.module if hasattr(model, "module") else model
        base = getattr(m, "base_model", None)
        if base is not None and hasattr(base, "update_and_allocate"):
            try:
                base.update_and_allocate(self.state.global_step)
            except KeyError as e:
                print(f"[adalora] 跳过 step {self.state.global_step} 的强制秩重分配（peft bug: {e}）",
                      flush=True)
        return loss


class EmptyCacheCallback(TrainerCallback):
    """每个优化步结束把空闲缓存块归还驱动（cudaFree）。

    Windows/WDDM 下缓存块会一直占着专用显存；瞬时峰值到来时驱动会优先把这些
    "最近未触碰"的缓存页换出到内存，之后 PyTorch 复用它们又要换回来——这正是
    "跨显卡搬运"式灾难性降速（GPU 100% 占用但只有 30W）的起点。每步释放一次，
    让专用显存始终贴着真实用量（代价：每步多几毫秒）。"""

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class AgentTrajectoryDataset(Dataset):
    """把对话渲染为 token 序列，并仅对 assistant 生成的区间计算 loss。

    依赖 Qwen chat template 的前缀稳定性：逐条消息增量渲染，定位每段
    assistant 内容（含 <tool_call> 块与 <|im_end|>）在最终 token 序列中的区间。
    """

    def __init__(self, path, tokenizer, max_len=2048):
        self.samples = load_jsonl(path)
        self.tok = tokenizer
        self.max_len = max_len
        self._cache = [self._encode(s) for s in self.samples]

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, i):
        return self._cache[i]

    def _encode(self, sample):
        """整体渲染一次得到 input_ids；对每个 assistant 轮次，用
        「渲染到该轮之前 + 生成前缀」的 token 序列与整体序列求公共前缀长度，
        该公共前缀之后、本轮渲染结束之前即 assistant 可学习区间。
        （整体渲染与前缀渲染在 BPE 边界可能有一个 token 的合并差异，
        取公共前缀可保证 mask 永不越过边界错位。）"""
        tok = self.tok
        # 本项目只保留数学轨迹链路：ToRA 数据无 tools 字段，恒按纯对话渲染
        # （apply_chat_template 的 tools 参数传 None 即可）
        tools = None
        msgs = sample["messages"]

        full_ids = tok.apply_chat_template(msgs, tools=tools, tokenize=True,
                                           add_generation_prompt=False,
                                           return_dict=False, **CHAT_TMPL_KW)
        labels = [IGNORE_ID] * len(full_ids)

        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            prefix_ids = tok.apply_chat_template(msgs[:i], tools=tools, tokenize=True,
                                                 add_generation_prompt=True,
                                                 return_dict=False, **CHAT_TMPL_KW)
            start = 0
            for a, b in zip(prefix_ids, full_ids):
                if a != b:
                    break
                start += 1
            # 本轮渲染结束位置：渲染 msgs[:i+1]（生成提示已包含本轮开头，
            # 其长度即该轮结束处）——直接对 msgs[:i+1] 渲染取长度
            upto_ids = tok.apply_chat_template(msgs[: i + 1], tools=tools, tokenize=True,
                                               add_generation_prompt=False,
                                               return_dict=False, **CHAT_TMPL_KW)
            end = 0
            for a, b in zip(upto_ids, full_ids):
                if a != b:
                    break
                end += 1
            end = min(max(end, start), len(full_ids))
            for j in range(start, end):
                labels[j] = full_ids[j]

        input_ids = full_ids
        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            labels = labels[: self.max_len]
        return {"input_ids": input_ids, "labels": labels}


def make_collator(pad_id):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [IGNORE_ID] * pad)
            attn.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
    return collate


LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", ".train.lock")


def acquire_lock():
    """单实例锁：同一时刻只允许一个训练进程，防止双进程争抢 8GB 显存
    导致每步 250 秒的灾难性降速和随机 OOM。"""
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    if os.path.exists(LOCK_FILE):
        try:
            holder = open(LOCK_FILE, encoding="utf-8").read().strip()
        except OSError:
            holder = "unknown"
        # 锁文件残留但持有进程可能已死：检查 PID 是否存活
        # （os.kill(pid,0) 在 Windows 上对存活进程会抛未处理的 SystemError，改用 kernel32）
        alive = False
        try:
            pid = int(holder.split(":")[0])
            if sys.platform == "win32":
                import ctypes
                h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                if h:
                    ctypes.windll.kernel32.CloseHandle(h)
                    alive = True
            else:
                os.kill(pid, 0)
                alive = True
        except (ValueError, OSError, SystemError):
            alive = False
        if alive:
            sys.exit(f"[train] 已有训练进程在运行（{holder}）。"
                     f"若确认它已死，删除 {LOCK_FILE} 后重试。切勿双开训练！")
        os.remove(LOCK_FILE)
    with open(LOCK_FILE, "w", encoding="utf-8") as f:
        f.write(f"{os.getpid()}:{os.path.basename(sys.argv[0]) or 'train'}")
    atexit.register(lambda: os.path.exists(LOCK_FILE) and os.remove(LOCK_FILE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/Qwen3.5-4B")
    ap.add_argument("--train", type=str, default="data/tora_sft/train.jsonl")
    ap.add_argument("--val", type=str, default="data/tora_sft/val.jsonl")
    ap.add_argument("--out", type=str, default="outputs/qlora-tora")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=216, help="总训练步数（按步驱动训练）")
    ap.add_argument("--logging-steps", type=int, default=1, help="按步记录 loss")
    ap.add_argument("--save-steps", type=int, default=None,
                    help="保存/评估间隔；默认 max_steps//8")
    ap.add_argument("--save-total-limit", type=int, default=0, help="保留 checkpoint 数，0=全部")
    ap.add_argument("--bs", type=int, default=1, help="per-device batch size")
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--peft-method", type=str, default="adalora", choices=["adalora", "lora"])
    ap.add_argument("--init-r", type=int, default=16, help="AdaLoRA 初始秩（预算池）")
    ap.add_argument("--target-r", type=int, default=8, help="AdaLoRA 目标平均秩")
    ap.add_argument("--ce-chunk", type=int, default=256,
                    help="分块交叉熵的行数（0=用 transformers 原始整段 loss；"
                         "见 make_chunked_ce_loss，248k 词表下这是 8GB 显存的关键）")
    ap.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    ap.add_argument("--optim", type=str, default="adamw_8bit",
                    choices=["adamw_8bit", "paged_adamw_8bit", "adamw_torch"],
                    help="8bit AdamW；不要用 paged 版：Qwen3.5 的词表是 248k，"
                         "logits/交叉熵的瞬时显存把 8GB 卡顶满，paged 优化器会把状态页"
                         "在 CPU/GPU 之间来回搬运（GPU 100%% 占用但只有 30W，每步几分钟）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    acquire_lock()

    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True) 
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("加载训练/验证数据集 ...")
    train_ds = AgentTrajectoryDataset(args.train, tok, args.max_len)
    val_ds = AgentTrajectoryDataset(args.val, tok, args.max_len)
    lens = sorted(len(x["input_ids"]) for x in train_ds._cache)
    print(f"训练 {len(train_ds)} 条 | 验证 {len(val_ds)} 条 | "
          f"token 长度 p50={lens[len(lens)//2]} p95={lens[int(len(lens)*0.95)]} max={lens[-1]}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    # 关掉 kv cache（训练必须）；文本子配置也置 False，避免 forward 读到 text_config 的值
    model.config.use_cache = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = False
    if args.ce_chunk > 0:  # 分块交叉熵：省约 1GB 瞬时显存，数值等价
        model.loss_function = make_chunked_ce_loss(args.ce_chunk)

    # 纯文本训练用不到视觉塔：摘掉并归还显存
    if drop_vision_tower(model):
        print(f"已移除视觉塔 | 当前显存 {torch.cuda.memory_allocated()/2**30:.2f}GB", flush=True)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})

    # 冻结 fp32 权重压回 bf16（248k 词表下这是 8GB 卡最大的一块开销，见函数注释）
    n_conv = shrink_frozen_fp32(model)
    if n_conv:
        print(f"冻结 fp32 权重压回 bf16: {n_conv} 个张量 | 当前显存 "
              f"{torch.cuda.memory_allocated()/2**30:.2f}GB / "
              f"reserved {torch.cuda.memory_reserved()/2**30:.2f}GB", flush=True)

    import math
    total_steps = args.max_steps
    steps_per_epoch = math.ceil(len(train_ds) / (args.bs * args.accum))
    save_steps = args.save_steps or max(1, total_steps // 8)
    warmup_steps = max(5, int(0.03 * total_steps))
    print(f"按步训练: total_steps={total_steps} (≈{total_steps/steps_per_epoch:.1f} epochs) | "
          f"每步记录 loss | 每 {save_steps} 步保存+评估 | 共 {total_steps // save_steps}+ 个保存点")

    if args.peft_method == "adalora":
        peft_cfg = AdaLoraConfig(
            init_r=args.init_r,
            target_r=args.target_r,
            tinit=max(10, total_steps // 10),
            tfinal=max(10, total_steps // 10),
            deltaT=3,
            beta1=0.85,
            beta2=0.85,
            orth_reg_weight=0.5,
            total_step=total_steps,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        )
    else:
        peft_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    free, _ = torch.cuda.mem_get_info()
    print(f"模型就绪: 显存 {torch.cuda.memory_allocated()/2**30:.2f}GB / reserved "
          f"{torch.cuda.memory_reserved()/2**30:.2f}GB | 设备剩余 {free/2**30:.2f}GB", flush=True)
    torch.cuda.reset_peak_memory_stats()

    targs = TrainingArguments(
        output_dir=args.out,
        max_steps=total_steps,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        bf16=True,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit or None,
        eval_strategy="steps",
        eval_steps=save_steps,
        eval_accumulation_steps=1,
        # 以 eval_loss 追踪最佳 checkpoint（收尾打印 + trainer.state.best_model_checkpoint）
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # 评估只算 loss、不留 logits：248k 词表下验证集 45 条的 logits 累积需要十几 GB，
        # 会把整张卡顶到 WDDM 换页（跑到评估就"卡死"的根因）
        prediction_loss_only=True,
        report_to=["tensorboard"],
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = AdaLoraTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collator(tok.pad_token_id),
        callbacks=[EmptyCacheCallback()],
    )
    trainer.train(resume_from_checkpoint=args.resume)
    print(f"训练峰值显存: allocated {torch.cuda.max_memory_allocated()/2**30:.2f}GB | "
          f"reserved {torch.cuda.max_memory_reserved()/2**30:.2f}GB"
          f"（接近 7GB 即逼近 WDDM 换页阈值，可用 --ce-chunk 64 再压）")
    trainer.save_model(os.path.join(args.out, "final"))
    tok.save_pretrained(os.path.join(args.out, "final"))
    print(f"训练完成，LoRA adapter 已保存到 {os.path.join(args.out, 'final')}")
    if trainer.state.best_metric is not None:
        print(f"最佳 eval_loss={trainer.state.best_metric:.4f} @ {trainer.state.best_model_checkpoint}")
        print("提示：AdaLoRA checkpoint 保存的是按秩收缩后的权重，"
              "评估请直接指向该 checkpoint 目录（PeftModel.from_pretrained 可加载）")


if __name__ == "__main__":
    main()
