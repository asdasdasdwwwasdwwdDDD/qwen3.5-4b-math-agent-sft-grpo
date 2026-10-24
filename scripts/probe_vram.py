# -*- coding: utf-8 -*-
"""显存最坏情况探测：按 train_qlora.py 的完整路径加载模型（4bit+AdaLoRA+梯度检查点），
构造 bs × max_len 的满标签 batch 跑一次前向+反向，报告峰值显存。
用途：长训练开跑前确认最坏 batch（group_by_length 长度分组后的最长批）不会触发 OOM/WDDM 换页。
"""
import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import AdaLoraConfig, get_peft_model, prepare_model_for_kbit_training
from train_qlora import TARGET_MODULES, drop_vision_tower, shrink_frozen_fp32, make_chunked_ce_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/Qwen3.5-4B")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--ce-chunk", type=int, default=256)
    ap.add_argument("--data", type=str, default=None,
                    help="给定 jsonl 时用真实数据的最长样本组批探测（真实标签分布），"
                         "否则用随机满标签批（最坏中的最坏）")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=True)
    model.config.use_cache = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = False
    model.loss_function = make_chunked_ce_loss(args.ce_chunk)
    drop_vision_tower(model)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    shrink_frozen_fp32(model)
    model = get_peft_model(model, AdaLoraConfig(
        init_r=16, target_r=8, tinit=10, tfinal=10, deltaT=3,
        beta1=0.85, beta2=0.85, orth_reg_weight=0.5, total_step=25,
        lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=TARGET_MODULES))
    model.train()
    free, _ = torch.cuda.mem_get_info()
    print(f"模型就绪: {torch.cuda.memory_allocated()/2**30:.2f}GB | 设备剩余 {free/2**30:.2f}GB", flush=True)
    torch.cuda.reset_peak_memory_stats()

    # 最坏 batch：优先用真实数据最长样本（真实标签分布，约 70% 被 mask）；
    # 未给 --data 时用随机满标签批（比真实数据更狠——全算 loss）
    from train_qlora import AgentTrajectoryDataset, make_collator
    if args.data:
        ds = AgentTrajectoryDataset(args.data, tok, args.max_len)
        cache = sorted(ds._cache, key=lambda x: len(x["input_ids"]), reverse=True)
        batch = make_collator(tok.pad_token_id)(cache[: args.bs])
        batch = {k: v.cuda() for k, v in batch.items()}
        n_tok = batch["input_ids"].shape
        n_valid = int((batch["labels"] != -100).sum())
        print(f"真实最长 batch: shape={n_tok} 有效标签 {n_valid}", flush=True)
        out = model(**batch)
    else:
        vocab = model.get_output_embeddings().weight.shape[0]
        ids = torch.randint(1000, vocab - 1, (args.bs, args.max_len), device="cuda")
        labels = ids.clone()
        attn = torch.ones_like(ids)
        out = model(input_ids=ids, attention_mask=attn, labels=labels)
    (out.loss / 4).backward()
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_res = torch.cuda.max_memory_reserved() / 2**30
    free2, total = torch.cuda.mem_get_info()
    print(f"前向+反向完成 loss={out.loss.item():.2f}")
    print(f"峰值: allocated {peak_alloc:.2f}GB | reserved {peak_res:.2f}GB")
    print(f"nvidia-smi 口径剩余/总量: {free2/2**30:.2f} / {total/2**30:.2f}GB")
    print("判据: allocated 峰值 ≤ 7.0GB 且 nvidia-smi 剩余 > 0.6GB 视为安全"
          "（8GB 卡需给桌面/WDDM 留余量）")


if __name__ == "__main__":
    main()
