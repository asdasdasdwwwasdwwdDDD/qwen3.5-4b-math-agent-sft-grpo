# -*- coding: utf-8 -*-
"""GRPO 组内验证对齐训练（Qwen3.5-4B，整条链路 / trajectory 模式）

从用户问题出发，模型自己走完整条链路——写 Python 代码、经 math_env 受限子进程真实执行、
输出以 role="tool" 回填、继续推理，直到 \boxed 终答。同一任务采样 G 条完整轨迹成组，
score_trajectory 按「答案正确 0.8 + boxed 格式 0.1 + 执行无误 0.1」打 0~1 分，组内归一化
优势后沿轨迹全部生成 token 反传：
    PPO 裁剪策略梯度 + KL 锚定（参考策略 = 关掉 LoRA 的 merged 模型，零额外显存）

按步驱动：每步记录 reward/成功率/执行报错率到 logs/grpo_log_math.jsonl 与 tensorboard；
每步把每个任务的 G 条完整链路（含执行返回与得分明细）写入 logs/grpo_trajs_math.jsonl。

用法（仓库根运行）:
  python -m scripts.training.train_grpo --model outputs/merged-tora \
      --out outputs/grpo-tora --max-steps 60 --num-generations 4 --save-steps 10
"""
import argparse
import json
import os
import random
import time

# Windows/WDDM：显存打满后驱动会把显存页换出到"共享显存"（内存），每步从秒级恶化到分钟级；
# expandable_segments 在 Windows 上不支持，用垃圾回收阈值让缓存块提早归还驱动。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")

import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from scripts.envs.math_env import (MathEnv, format_trajectory, load_tasks,
                                   rollout, score_trajectory)
from scripts.training.train_qlora import (TARGET_MODULES, acquire_lock,
                                          drop_vision_tower, shrink_frozen_fp32)


def token_logprobs(model, input_ids, attention_mask, n_comp):
    """返回序列末尾 n_comp 个 token 的逐 token log p（next-token 移位后取尾部）。

    只对末尾 n_comp 个位置算 log_softmax（logits_to_keep）：Qwen3.5 词表 248k，
    若对整段（prompt+补全，约 1000 token）出 logits 再做 fp32 log_softmax，
    LM head 与 softmax 会多花约 1.5GB 显存且白算 prompt 段；逐位置 softmax 与
    「先算全段再切片」结果一致，梯度也只落在补全 token 上（与 GRPO 目标一致）。"""
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                   logits_to_keep=n_comp + 1).logits[0]
    lp = torch.log_softmax(logits[:-1].float(), dim=-1)
    tgt = input_ids[0, -n_comp:].unsqueeze(-1)
    return lp.gather(-1, tgt).squeeze(-1)


def make_bnf4_model(model_path):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=True)
    model.config.use_cache = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = False
    return model


def run_closed_eval(model, tok, tasks, args):
    """训练中的闭环评估：greedy 走完整链路，与 evaluation/eval_math.py 同口径
    （每任务独立 MathEnv + score_trajectory）。只读不反传；生成期间 rollout 内部
    自行切换推理态并恢复训练态。返回 (success_rate, reward_mean)。"""
    succ, rs = 0, []
    for t in tasks:
        env = MathEnv(t)
        ro = rollout(model, tok, env, max_turns=args.max_turns,
                     max_new_tokens=args.max_new_tokens,
                     temperature=1.0, top_p=1.0, max_len=args.max_len,
                     do_sample=False, grad_ckpt=bool(args.grad_ckpt))
        r, det = score_trajectory(ro, env)
        rs.append(r)
        succ += int(det["success"])
        torch.cuda.empty_cache()
    return succ / len(tasks), sum(rs) / len(tasks)


def make_task_order(pool, curriculum):
    """任务顺序：课程化=各难度层内部打乱后轮转交织（L1,L2,L3,L4,L5,L1,…），
    否则整体打乱。默认课程化：训练池 82% 是 Level 3-5 难题，SFT 模型
    采样正确率低 → 整组全错且 reward 同为 0.2（格式分全拿）→ 组内零方差零梯度
    （实测首版 run 的 step1 两组 8 轨迹全 0.2 无信号）。交织保证每步的任务里
    大概率混有低难度题（组内对错混合 → 有优势信号）。"""
    if curriculum:
        by_lv = {}
        for t in pool:
            by_lv.setdefault(t.get("level", "?"), []).append(t)
        for group in by_lv.values():
            random.shuffle(group)
        order, lvs = [], sorted(by_lv)
        while any(by_lv[lv] for lv in lvs):
            for lv in lvs:
                if by_lv[lv]:
                    order.append(by_lv[lv].pop())
        return order
    order = pool[:]
    random.shuffle(order)
    return order


def train_trajectory_loop(model, tok, pool, args, opt, sched, trainable,
                          writer, log_fp, traj_fp, save_steps, eval_tasks=None):
    """整条链路 GRPO：模型从用户问题出发自己走完多轮（代码真执行），
    按整条轨迹的结果打分、组内归一化优势，沿轨迹全部生成 token 做 PPO 裁剪 + KL。
    eval_tasks 非空时，每 args.eval_every 步跑一次 greedy 闭环评估并保存最优 adapter。"""
    order = make_task_order(pool, getattr(args, "curriculum", False))
    cursor = 0
    best = {"success": -1.0, "reward": -1.0, "step": 0}  # 闭环评估最优点 -> out/best
    for step in range(1, args.max_steps + 1):
        step_t0 = time.time()
        if cursor + args.prompts_per_step > len(order):
            order = make_task_order(pool, getattr(args, "curriculum", False))
            cursor = 0
        batch = order[cursor:cursor + args.prompts_per_step]
        cursor += args.prompts_per_step

        # ---- 1) 整条链路 rollout（组 = 同一任务 G 条完整轨迹）----
        rollouts = []   # [(env, ro, adv, reward, detail)]
        gstats = []
        for task in batch:
            grp = []
            envs = []   # 每条轨迹一个独立 env
            for _ in range(args.num_generations):
                # 每条轨迹必须新建一个 MathEnv（2026-10-24 修复后保留的口径，勿回退）：
                # env 内部有回合状态，共用会让 G 条轨迹互相干扰、组内优势被污染。
                env = MathEnv(task)
                ro = rollout(model, tok, env, max_turns=args.max_turns,
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_p=args.top_p,
                             max_len=args.max_len, do_sample=True,
                             grad_ckpt=bool(args.grad_ckpt))
                r, det = score_trajectory(ro, env)
                grp.append((ro, r, det))
                envs.append(env)
            torch.cuda.empty_cache()  # 生成阶段的 kv cache 归还驱动，给梯度阶段腾峰值
            rewards = [x[1] for x in grp]
            mean = sum(rewards) / len(rewards)
            std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
            advs = [(r - mean) / (std + 1e-4) for r in rewards] if std > 1e-6 \
                else [0.0] * len(rewards)
            for (ro, r, det), a, e in zip(grp, advs, envs):
                rollouts.append((e, ro, a, r, det))
            gstats.append({"mean_r": round(mean, 4), "std_r": round(std, 4),
                           "rewards": [round(r, 4) for r in rewards]})
            traj_fp.write(json.dumps({
                "step": step, "tid": task["id"], "category": task["category"],
                "group": [{"reward": round(r, 4), "adv": round(a, 3), "detail": det,
                           "text": format_trajectory(e, ro)}
                          for (ro, r, det), a, e in zip(grp, advs, envs)],
            }, ensure_ascii=False) + "\n")
        traj_fp.flush()

        n = len(rollouts)
        dets = [x[4] for x in rollouts]
        all_r = [x["mean_r"] for x in gstats]
        frac_valid = sum(1 for g in gstats if g["std_r"] > 1e-6) / len(gstats)

        # ---- 2) 整条链路 GRPO 更新：轨迹级优势，逐回合反传（token 数加权）----
        opt.zero_grad(set_to_none=True)
        tot_loss = pol_sum = kl_sum = 0.0
        n_turn = n_trained = 0
        clip_hits = 0
        gnorm = 0.0
        pending = 0  # 已 backward 尚未 opt.step 的轨迹数
        for i, (env, ro, adv, r, det) in enumerate(rollouts):
            if adv == 0.0:  # 组内全同分（全对/全错）无梯度
                continue
            toks = sum(len(t["gen_ids"]) for t in ro["turns"])
            if toks == 0:
                continue
            loss = None
            for t in ro["turns"]:
                L = len(t["gen_ids"])
                if L == 0:
                    continue
                ids = torch.tensor([list(t["prompt_ids"]) + list(t["gen_ids"])],
                                   device=model.device)
                attn = torch.ones_like(ids)
                with torch.no_grad():
                    old = token_logprobs(model, ids, attn, L)
                    with model.disable_adapter():
                        ref = token_logprobs(model, ids, attn, L)
                act = token_logprobs(model, ids, attn, L)
                d = (ref - act).clamp(max=8.0)
                kl_tok = (d.exp() - d - 1).clamp(max=args.kl_clip)
                ratio = (act - old).exp()
                pol = -torch.min(ratio * adv,
                                 ratio.clamp(1 - args.epsilon, 1 + args.epsilon) * adv).mean()
                piece = (pol + args.beta * kl_tok.mean()) * (L / toks)
                loss = piece if loss is None else loss + piece
                pol_sum += pol.detach().item()
                kl_sum += kl_tok.detach().mean().item()
                n_turn += 1
                clip_hits += int((ratio.detach() < 1 - args.epsilon).sum()
                                 + (ratio.detach() > 1 + args.epsilon).sum())
            if loss is None:
                continue
            (loss * n / args.grad_accum).backward()
            tot_loss += loss.detach().item()
            n_trained += 1
            pending += 1
            if pending == args.grad_accum:
                gn = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                gnorm = max(gnorm, float(gn))
                opt.step()
                opt.zero_grad(set_to_none=True)
                pending = 0
                torch.cuda.empty_cache()  # WDDM：优化步后把空闲缓存块归还驱动
        # 修复（2026-10-24）：旧版按 enumerate 下标 (i+1)%accum 触发优化步，若一步末尾
        # 的轨迹因 adv==0 被跳过，已累积的梯度会在下一步开头被 zero_grad 静默丢弃；
        # 改为按实际 backward 计数触发，循环结束后仍有残留则补一次优化步。
        # 学习率同步改为每个训练步推进一次：优化步次数随 P×G/accum 变化
        # （P=2 时每训练步 2 次优化），若仍按优化步推进，cosine 会在半程就衰减到 0。
        if pending > 0:
            gn = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            gnorm = max(gnorm, float(gn))
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        if n_trained:
            sched.step()

        # ---- 2.5) 训练中闭环评估：greedy 固定子集（与 eval_math 同口径），追踪最优 ----
        last_eval = None
        if eval_tasks and step % args.eval_every == 0:
            es, er = run_closed_eval(model, tok, eval_tasks, args)
            last_eval = (es, er)
            print(f"  [eval@{step:3d}] success={es:.3f} reward={er:.3f}", flush=True)
            writer.add_scalar("grpo/eval_success", es, step)
            writer.add_scalar("grpo/eval_reward", er, step)
            if (es, er) > (best["success"], best["reward"]):
                best.update(success=es, reward=er, step=step)
                model.save_pretrained(os.path.join(args.out, "best"))
                tok.save_pretrained(os.path.join(args.out, "best"))
                print(f"  当前评估最优，已保存 -> {os.path.join(args.out, 'best')}", flush=True)

        # ---- 3) 按步记录 ----
        n_exec = sum(d["n_exec"] for d in dets)
        rec = {"step": step, "lr": round(sched.get_last_lr()[0], 9),
               "reward_mean": round(sum(all_r) / len(all_r), 4),
               "valid_group_frac": round(frac_valid, 3),
               "success_rate": round(sum(1 for d in dets if d["success"]) / len(dets), 3),
               "avg_turns": round(sum(d["n_turns"] for d in dets) / len(dets), 2),
               "exec_err": round(sum(d["n_error"] for d in dets) / max(n_exec, 1), 3),
               "policy": round(pol_sum / max(n_turn, 1), 4),
               "kl": round(kl_sum / max(n_turn, 1), 5),
               "loss": round(tot_loss / max(n_trained, 1), 4),
               "grad_norm": round(gnorm, 3), "clip_tokens": clip_hits,
               "sec": round(time.time() - step_t0, 1),
               "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        if last_eval is not None:
            rec["eval_success"] = round(last_eval[0], 3)
            rec["eval_reward"] = round(last_eval[1], 3)
        log_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_fp.flush()
        print(f"[step {step:3d}/{args.max_steps}] reward={rec['reward_mean']:.3f} "
              f"success={rec['success_rate']:.0%} turns={rec['avg_turns']} "
              f"exec_err={rec['exec_err']:.0%} kl={rec['kl']:.4f} loss={rec['loss']:.4f} "
              f"gnorm={rec['grad_norm']:.2f} lr={rec['lr']:.2e} "
              f"mem={rec['mem_gb']}GB ({rec['sec']}s)", flush=True)
        for k in ("reward_mean", "valid_group_frac", "success_rate", "avg_turns",
                  "exec_err", "kl", "policy", "loss", "grad_norm"):
            writer.add_scalar("grpo/" + k, rec[k], step)
        writer.add_scalar("grpo/lr", rec["lr"], step)
        writer.add_scalar("grpo/mem_gb", rec["mem_gb"], step)

        if step % save_steps == 0 or step == args.max_steps:
            ck = os.path.join(args.out, f"checkpoint-{step}")
            model.save_pretrained(ck)
            print(f"  已保存 {ck}", flush=True)

    if best["step"]:
        print(f"闭环评估最优: step {best['step']} success={best['success']:.3f} "
              f"reward={best['reward']:.3f} -> {os.path.join(args.out, 'best')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="outputs/merged-tora",
                    help="SFT 合并后的 bf16 模型目录（GRPO 起点，同时作 KL 参考）")
    ap.add_argument("--data", type=str, default="data/tora_sft/grpo_train.jsonl",
                    help="训练任务文件（默认池已剔除 SFT 见过的题）")
    ap.add_argument("--out", type=str, default="outputs/grpo-tora")
    ap.add_argument("--resume", type=str, default=None, help="续训：加载此目录的 LoRA")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--num-generations", type=int, default=4,
                    help="每组候选数 G（每个任务 G 条完整轨迹）")
    ap.add_argument("--prompts-per-step", type=int, default=2,
                    help="每步任务数 P（≥2 时一次更新含多个组，显著降低单组采样的梯度方差；"
                         "时间约随 P 线性增长，约 1 分钟/任务/步）")
    ap.add_argument("--max-turns", type=int, default=4,
                    help="单条轨迹最多模型回合数（代码→出错重试→终答）")
    ap.add_argument("--max-len", type=int, default=2048,
                    help="单条轨迹 prompt+生成 总 token 上限（题目短、代码为主）")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="单回合生成上限（代码块较长）")
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="采样温度（T=1.0 实测显著压低 SFT 模型的采样正确率，"
                         "组内全错同分→零梯度步变多）")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--lr", type=float, default=3e-5,
                    help="旧默认 1e-5 偏保守（旧版实测 grad_norm 2~5.5 且被硬剪 1.0，"
                         "等效步长过小，100 步 reward 无趋势）")
    ap.add_argument("--beta", type=float, default=0.025, help="KL 系数（锚定 SFT 模型）")
    ap.add_argument("--max-grad-norm", type=float, default=2.0,
                    help="梯度裁剪上限（旧版硬编码 1.0；组内归一化优势的梯度范数"
                         "常态 2~5.5，每步硬剪到 1.0 等效把学习率再砍 2~5 倍）")
    ap.add_argument("--kl-clip", type=float, default=0.05, help="单 token KL 截断，防离群主导")
    ap.add_argument("--epsilon", type=float, default=0.2, help="PPO 裁剪")
    ap.add_argument("--save-steps", type=int, default=None, help="默认 max_steps//10")
    ap.add_argument("--eval-data", type=str, default="data/tora_sft/eval_set.jsonl",
                    help="训练中闭环评估的任务文件（与 SFT/RL 训练题均不相交的持出集）")
    ap.add_argument("--eval-every", type=int, default=20,
                    help="每 N 个训练步做一次闭环评估（greedy 固定子集），0 关闭；"
                         "每次约 2~4 分钟，最优 checkpoint 存 out/best")
    ap.add_argument("--eval-n", type=int, default=16,
                    help="每次闭环评估的任务数（固定随机子集，各评估点分数可比）")
    ap.add_argument("--max-prompt-len", type=int, default=600,
                    help="初始 prompt token 上限过滤")
    ap.add_argument("--curriculum", type=int, default=1,
                    help="任务顺序课程化：各难度层轮转交织，保证每步混有低难度题"
                         "（整组全错同分 → 零梯度步 的密度显著下降）")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="轨迹级梯度累积（每 accum 条轨迹一次 opt.step）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    acquire_lock()
    set_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("加载 NF4 量化 merged 模型 ...", flush=True)
    base = make_bnf4_model(args.model)
    if drop_vision_tower(base):
        print(f"已移除视觉塔 | 当前显存 {torch.cuda.memory_allocated()/2**30:.2f}GB", flush=True)
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=bool(args.grad_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False})
    # 冻结 fp32 权重（248k 词表的 embedding/头是最大一块）压回 bf16，避免 8GB 卡 WDDM 换页
    print(f"冻结 fp32 权重压回 bf16: {shrink_frozen_fp32(base)} 个张量 | 当前显存 "
          f"{torch.cuda.memory_allocated()/2**30:.2f}GB", flush=True)
    if args.resume:
        model = PeftModel.from_pretrained(base, args.resume, is_trainable=True)
        print(f"续训 adapter: {args.resume}", flush=True)
    else:
        cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES)
        model = get_peft_model(base, cfg)
    model.print_trainable_parameters()

    tasks = load_tasks(args.data, tok, args.max_prompt_len)
    print(f"可用任务 {len(tasks)}（初始 prompt <= {args.max_prompt_len} tokens）", flush=True)

    trainable = [q for q in model.parameters() if q.requires_grad]
    import bitsandbytes as bnb
    # 不要用 PagedAdamW8bit：8GB 卡上优化器状态页会在 CPU/GPU 间来回搬运
    # （GPU 100% 占用但只有 30W，每步几分钟），非 paged 的 8bit 状态已足够小
    opt = bnb.optim.AdamW8bit(trainable, lr=args.lr, weight_decay=0.01)
    save_steps = args.save_steps or max(1, args.max_steps // 10)
    warmup = max(5, int(0.05 * args.max_steps))
    sched = get_cosine_schedule_with_warmup(opt, warmup, args.max_steps)

    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter(os.path.join(args.out, "runs"))
    log_fp = open(os.path.join("logs", "grpo_log_math.jsonl"), "a", encoding="utf-8")

    model.train()
    eval_tasks = []
    if args.eval_every > 0:
        pool_eval = load_tasks(args.eval_data, tok, args.max_prompt_len)
        random.Random(0).shuffle(pool_eval)  # 固定子集：各评估点同一批任务，分数可比
        eval_tasks = pool_eval[: args.eval_n]
        print(f"训练中闭环评估: {len(eval_tasks)} 个任务（{args.eval_data}），"
              f"每 {args.eval_every} 步一次", flush=True)
    traj_fp = open(os.path.join("logs", "grpo_trajs_math.jsonl"), "a", encoding="utf-8")
    train_trajectory_loop(model, tok, tasks, args, opt, sched, trainable,
                          writer, log_fp, traj_fp, save_steps, eval_tasks)
    traj_fp.close()

    model.save_pretrained(os.path.join(args.out, "final"))
    tok.save_pretrained(os.path.join(args.out, "final"))
    writer.close()
    log_fp.close()
    print(f"GRPO 完成 -> {os.path.join(args.out, 'final')}", flush=True)


if __name__ == "__main__":
    main()
