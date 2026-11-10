# -*- coding: utf-8 -*-
"""GRPO 组内验证对齐训练（Qwen3.5-4B，整条链路 / trajectory 模式）

从用户问题出发，模型自己走完整条链路——写 Python 代码、经 math_env 受限子进程真实执行、
输出以 role="tool" 回填、继续推理，直到 \boxed 终答。同一任务采样 G 条完整轨迹成组，
score_trajectory 按「答案正确 0.8 + boxed 格式 0.1 + 执行无误 0.1」打 0~1 分，组内归一化
优势后沿轨迹全部生成 token 反传：
    PPO 裁剪策略梯度 + KL 锚定（参考策略 = 关掉 LoRA 的 merged 模型，零额外显存）

同一批数据默认只重放 1 遍（--epochs，μ=1 即封存口径：old_logp 现算，ratio 恒 1，裁剪不触发）；
μ>1 时 old_logp 首遍缓存供后续遍复用，ratio 偏离 1、PPO 裁剪真正上岗（每训练步参数移动量
约 ×μ）。首回合组内批量采样由 --gen-batch 控制（1=逐条封存口径，>1 只换随机数流、
统计等价）。默认参数与封存快照逐位一致，新口径全部显式开启。

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
import sys
import time

# 兼容两种运行方式：除文档用法 `python -m scripts.training.train_grpo` 外，直接以文件
# 方式运行（如 VS Code「运行 Python 文件」）时 sys.path 里没有仓库根，包导入会
# ModuleNotFoundError；此处把仓库根补进 sys.path（前者场景下该路径已存在，无副作用）
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

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
                                   rollout, rollout_group, score_trajectory)
from scripts.training.train_qlora import (TARGET_MODULES, acquire_lock,
                                          drop_vision_tower, shrink_frozen_fp32)


def token_logprobs(model, input_ids, attention_mask, n_gen_tokens):
    """返回序列末尾 n_gen_tokens 个 token 的逐 token log p（next-token 移位后取尾部）。

    只对末尾 n_gen_tokens 个位置算 log_softmax（logits_to_keep）：Qwen3.5 词表 248k，
    若对整段（prompt+补全，约 1000 token）出 logits 再做 fp32 log_softmax，
    LM head 与 softmax 会多花约 1.5GB 显存且白算 prompt 段；逐位置 softmax 与
    「先算全段再切片」结果一致，梯度也只落在补全 token 上（与 GRPO 目标一致）。"""
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                   logits_to_keep=n_gen_tokens + 1).logits[0]
    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
    target_ids = input_ids[0, -n_gen_tokens:].unsqueeze(-1)
    return log_probs.gather(-1, target_ids).squeeze(-1)


def load_model_4bit(model_path):
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


def run_closed_eval(model, tokenizer, tasks, args):
    """训练中的闭环评估：greedy 走完整链路，与 evaluation/eval_math.py 同口径
    （每任务独立 MathEnv + score_trajectory）。只读不反传；生成期间 rollout 内部
    自行切换推理态并恢复训练态。返回 (success_rate, reward_mean)。"""
    n_correct, rewards = 0, []
    for task in tasks:
        env = MathEnv(task)
        trajectory = rollout(model, tokenizer, env, max_turns=args.max_turns,
                             max_new_tokens=args.max_new_tokens,
                             temperature=1.0, top_p=1.0, max_len=args.max_len,
                             do_sample=False, grad_ckpt=bool(args.grad_ckpt))
        reward, detail = score_trajectory(trajectory, env)
        rewards.append(reward)
        n_correct += int(detail["success"])
        torch.cuda.empty_cache()
    return n_correct / len(tasks), sum(rewards) / len(tasks)


def make_task_order(task_pool, curriculum):
    """任务顺序：课程化=各难度层内部打乱后轮转交织（L1,L2,L3,L4,L5,L1,…），
    否则整体打乱。默认课程化：训练池 82% 是 Level 3-5 难题，SFT 模型
    采样正确率低 → 整组全错且 reward 同为 0.2（格式分全拿）→ 组内零方差零梯度
    （实测首版 run 的 step1 两组 8 轨迹全 0.2 无信号）。交织保证每步的任务里
    大概率混有低难度题（组内对错混合 → 有优势信号）。"""
    if curriculum:
        by_level = {}
        for task in task_pool:
            by_level.setdefault(task.get("level", "?"), []).append(task)
        for bucket in by_level.values():
            random.shuffle(bucket)
        task_order, levels = [], sorted(by_level)
        while any(by_level[level] for level in levels):
            for level in levels:
                if by_level[level]:
                    task_order.append(by_level[level].pop())
        return task_order
    task_order = task_pool[:]
    random.shuffle(task_order)
    return task_order


def group_advantages(rewards):
    """组内相对优势（GRPO 名字的来源）：以组均值/标准差把绝对分变成相对分。

    adv = (r - mean) / (std + 1e-4)；组内全同分（全对或全错，std<=1e-6）时无区分度，
    返回全 0——这些轨迹在更新阶段整条跳过（无梯度）。1e-4 防止 std 极小时除爆。
    返回 (advs, mean, std)。"""
    mean = sum(rewards) / len(rewards)
    std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
    advs = [(r - mean) / (std + 1e-4) for r in rewards] if std > 1e-6 \
        else [0.0] * len(rewards)
    return advs, mean, std


def generate_scored_group(model, tokenizer, task, args):
    """同一道题采 G 条完整轨迹并打分（GRPO 的"组"）。返回 (group, envs)，
    group = [(trajectory, reward, detail), ...]，envs 与 group 一一对应。

    gen_batch > 1：首回合 prompt 完全相同，按块批量采样（rollout_group，效率路径）；
    gen_batch = 1：逐条 rollout（封存口径，随机数流与封存一致）。
    每条轨迹一个独立 MathEnv（2026-10-24 修复口径，勿回退）：env 内部有回合状态，
    共用会让 G 条轨迹互相干扰、组内优势被污染。"""
    if args.gen_batch > 1:
        traj_envs = rollout_group(model, tokenizer, task, num_generations=args.num_generations,
                                   gen_batch=args.gen_batch, max_turns=args.max_turns,
                                   max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, top_p=args.top_p,
                                   max_len=args.max_len, do_sample=True,
                                   grad_ckpt=bool(args.grad_ckpt))
        group, envs = [], []
        for trajectory, env in traj_envs:
            reward, detail = score_trajectory(trajectory, env)
            group.append((trajectory, reward, detail))
            envs.append(env)
        return group, envs
    group, envs = [], []
    for _ in range(args.num_generations):
        env = MathEnv(task)
        trajectory = rollout(model, tokenizer, env, max_turns=args.max_turns,
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_p=args.top_p,
                             max_len=args.max_len, do_sample=True,
                             grad_ckpt=bool(args.grad_ckpt))
        reward, detail = score_trajectory(trajectory, env)
        group.append((trajectory, reward, detail))
        envs.append(env)
    return group, envs


def train_step_update(model, rollouts, trainable, opt, args):
    """更新阶段：同一批 rollout 数据重放 args.epochs 遍（PPO 的 μ），逐回合反传。

    每回合重放三次前向（token_logprobs）：old_logp/ref_logp 首次现算并以 (轨迹号, 回合号) 缓存，
    后续 epoch 直接复用——
      - μ=1（默认，封存口径）：每条轨迹只重放一次，缓存永不复用，与旧实现逐位一致；
      - μ>1：第 2 遍起权重已动而 old_logp 还是首遍读数 → ratio=exp(act_logp-old_logp)≠1，
        min(ratio*adv, 裁剪版) 两项开始分开，PPO 裁剪与 clip_tokens 真正工作。
        注意：epoch 循环在外、backward 尺度不变 → 每训练步参数移动量约 ×μ，
        想保持总移动量可比可相应减小 --max-steps（如 60→20）。
    ref_logp（摘 adapter=SFT 参考模型）与 epoch 无关（底座权重恒定），缓存绝对安全。

    梯度累积与封存一致：每条轨迹 backward 时尺度 ×n_rollouts/grad_accum（n_rollouts=len(rollouts)，
    含被跳过的轨迹），pending_backwards 计满触发 opt.step；循环结束有残留补一次。
    返回统计 dict（tot_loss/pol_sum/kl_sum/n_turn/n_trained/clip_hits/gnorm）。"""
    stats = {"tot_loss": 0.0, "pol_sum": 0.0, "kl_sum": 0.0,
             "n_turn": 0, "n_trained": 0, "clip_hits": 0, "gnorm": 0.0}
    opt.zero_grad(set_to_none=True)   # 清空残留梯度（置 None 比置 0 更省内存）
    # 循环前过滤：组内全同分（组内优势为 0，无梯度）或一条 token 都没生成的轨迹整条跳过
    active_rollouts = []
    for _env, trajectory, adv, _r, _det in rollouts:
        if adv == 0.0:
            continue
        if sum(len(turn["gen_ids"]) for turn in trajectory["turns"]) == 0:
            continue
        active_rollouts.append((_env, trajectory, adv))
    n_rollouts = len(rollouts)          # backward 尺度基准与封存一致（含被跳过的轨迹）
    old_logp_cache, ref_logp_cache = {}, {}  # (轨迹号, 回合号) -> 首次重放的读数（本训练步内有效）
    pending_backwards = 0                # 已 backward 尚未 opt.step 的轨迹数
    for _epoch in range(args.epochs):
        for traj_i, (_env, trajectory, adv) in enumerate(active_rollouts):
            n_traj_tokens = sum(len(turn["gen_ids"]) for turn in trajectory["turns"])
            loss = None
            for turn_j, turn in enumerate(trajectory["turns"]):
                n_gen_tokens = len(turn["gen_ids"])
                if n_gen_tokens == 0:
                    continue
                # 把"当时的输入 + 当时生成的内容"拼回完整序列（批量=1）。
                # 重放不做批量化：[k, L, 248k] 的 fp32 logits 一拼就是 GB 级峰值，
                # batch=1 + logits_to_keep 是 8GB 显存红线的核心防线，勿改
                full_ids = torch.tensor([list(turn["prompt_ids"]) + list(turn["gen_ids"])],
                                        device=model.device)
                attention_mask = torch.ones_like(full_ids)    # 全 1 注意力掩码：没有 padding，全位置参与
                cache_key = (traj_i, turn_j)
                if cache_key not in ref_logp_cache:
                    with torch.no_grad(), model.disable_adapter():
                        # ref_logp：摘掉 LoRA（=SFT 参考模型）时的读数，KL 锚定用，不反传。
                        # disable_adapter 只是临时摘下小矩阵，出了 with 块就装回去
                        ref_logp_cache[cache_key] = token_logprobs(model, full_ids, attention_mask, n_gen_tokens)
                ref_logp = ref_logp_cache[cache_key]
                if cache_key in old_logp_cache:
                    old_logp = old_logp_cache[cache_key]
                else:
                    with torch.no_grad():
                        # old_logp：更新前模型对这些 token 的读数（PPO 的"旧概率"，不反传）。
                        # 首次重放时现算（此刻权重=生成时权重，与封存口径逐位一致）；
                        # μ>1 时冻结供后续 epoch 复用——ratio 才会偏离 1
                        old_logp_cache[cache_key] = token_logprobs(model, full_ids, attention_mask, n_gen_tokens)
                    old_logp = old_logp_cache[cache_key]
                # act_logp：同样的前向，但带着梯度——真正会被优化的"新概率"
                # 三次读数来自同一函数，区别只有两点：带不带梯度、挂不挂 LoRA
                act_logp = token_logprobs(model, full_ids, attention_mask, n_gen_tokens)
                log_ratio = (ref_logp - act_logp).clamp(max=8.0)
                kl_per_token = (log_ratio.exp() - log_ratio - 1).clamp(max=args.kl_clip)
                ratio = (act_logp - old_logp).exp()
                # 分轮次优势：代码轮在轨迹级优势上再叠加「该轮执行是否无报错」的局部信号，
                # 终答轮只用轨迹级优势。动机（2026-11-10 实测）：整条轨迹共用一个 adv 时，
                # 答错的轨迹会把其中写对的代码一起压低 → exec_err 从 28.9% 涨到 45.2%，
                # 代码能力被终答的对错连坐拖垮。turn_exec_w=0 时与旧行为逐位一致。
                turn_adv = adv
                if turn.get("exec_ok") is not None:
                    turn_adv = adv + args.turn_exec_w * (1.0 if turn["exec_ok"] else -1.0)
                policy_loss = -torch.min(ratio * turn_adv,
                                         ratio.clamp(1 - args.epsilon, 1 + args.epsilon) * turn_adv).mean()
                turn_loss = (policy_loss + args.beta * kl_per_token.mean()) * (n_gen_tokens / n_traj_tokens)
                loss = turn_loss if loss is None else loss + turn_loss
                stats["pol_sum"] += policy_loss.detach().item()
                stats["kl_sum"] += kl_per_token.detach().mean().item()
                stats["n_turn"] += 1
                stats["clip_hits"] += int((ratio.detach() < 1 - args.epsilon).sum()
                                          + (ratio.detach() > 1 + args.epsilon).sum())
            if loss is None:
                continue
            (loss * n_rollouts / args.grad_accum).backward()
            stats["tot_loss"] += loss.detach().item()
            stats["n_trained"] += 1
            pending_backwards += 1
            if pending_backwards == args.grad_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                stats["gnorm"] = max(stats["gnorm"], float(grad_norm))
                opt.step()
                opt.zero_grad(set_to_none=True)
                pending_backwards = 0
                torch.cuda.empty_cache()  # WDDM：优化步后把空闲缓存块归还驱动
    # 修复（2026-10-24）：旧版按 enumerate 下标 (i+1)%accum 触发优化步，若一步末尾
    # 的轨迹因 adv==0 被跳过，已累积的梯度会在下一步开头被 zero_grad 静默丢弃；
    # 改为按实际 backward 计数触发，循环结束后仍有残留则补一次优化步。
    if pending_backwards > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        stats["gnorm"] = max(stats["gnorm"], float(grad_norm))
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return stats


def log_step(writer, log_fp, args, step, step_t0, details, group_stats, stats, last_eval, sched):
    """按训练步记录：构造 record（=日志一行）→ jsonl 落盘 → 终端打印 → TensorBoard 曲线。

    GRPO 日志里 loss/kl 的口径：μ=1 时 ratio 恒 1 → policy 项各组求和为 0（组内优势
    之和为 0），kl 起步也近 0，学习信号看 grad_norm 与 success_rate；μ>1 时 ratio
    偏离 1，loss 不再恒 0，clip_tokens 是裁剪是否触发的直接读数。"""
    total_exec = sum(detail["n_exec"] for detail in details)
    record = {"step": step, "lr": round(sched.get_last_lr()[0], 9),
           "reward_mean": round(sum(g["mean_r"] for g in group_stats) / len(group_stats), 4),
           "valid_group_frac": round(sum(1 for g in group_stats if g["std_r"] > 1e-6)
                                     / len(group_stats), 3),
           "success_rate": round(sum(1 for d in details if d["success"]) / len(details), 3),
           "avg_turns": round(sum(d["n_turns"] for d in details) / len(details), 2),
           "exec_err": round(sum(detail["n_error"] for detail in details) / max(total_exec, 1), 3),
           "policy": round(stats["pol_sum"] / max(stats["n_turn"], 1), 4),
           "kl": round(stats["kl_sum"] / max(stats["n_turn"], 1), 5),
           "loss": round(stats["tot_loss"] / max(stats["n_trained"], 1), 4),
           "grad_norm": round(stats["gnorm"], 3), "clip_tokens": stats["clip_hits"],
           "epochs": args.epochs,
           "sec": round(time.time() - step_t0, 1),
           "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    if last_eval is not None:
        record["eval_success"] = round(last_eval[0], 3)
        record["eval_reward"] = round(last_eval[1], 3)
    log_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    log_fp.flush()
    print(f"[step {step:3d}/{args.max_steps}] reward={record['reward_mean']:.3f} "
          f"success={record['success_rate']:.0%} turns={record['avg_turns']} "
          f"exec_err={record['exec_err']:.0%} kl={record['kl']:.4f} loss={record['loss']:.4f} "
          f"gnorm={record['grad_norm']:.2f} lr={record['lr']:.2e} "
          f"mem={record['mem_gb']}GB ({record['sec']}s)", flush=True)
    for k in ("reward_mean", "valid_group_frac", "success_rate", "avg_turns",
              "exec_err", "kl", "policy", "loss", "grad_norm"):
        writer.add_scalar("grpo/" + k, record[k], step)
    writer.add_scalar("grpo/lr", record["lr"], step)
    writer.add_scalar("grpo/mem_gb", record["mem_gb"], step)


def train_trajectory_loop(model, tokenizer, task_pool, args, opt, sched, trainable,
                          writer, log_fp, traj_audit_fp, save_steps, eval_tasks=None):
    """整条链路 GRPO 主循环。每个训练步的时间线（与日志字段一一对应）：
      取批 → 1) 生成组+打分+组内优势+审计落盘 → 2) 同批数据重放 μ 遍反传
           → 2.5) 每 eval-every 步 greedy 闭环评估存 best → 3) 记日志 → 存档。
    各环节细节见 generate_scored_group / group_advantages / train_step_update / log_step。"""
    task_order = make_task_order(task_pool, getattr(args, "curriculum", False))
    cursor = 0
    best = {"success": -1.0, "reward": -1.0, "step": 0}  # 闭环评估最优点 -> out/best
    for step in range(1, args.max_steps + 1):
        step_t0 = time.time()
        if cursor + args.prompts_per_step > len(task_order):
            task_order = make_task_order(task_pool, getattr(args, "curriculum", False))
            cursor = 0
        task_batch = task_order[cursor:cursor + args.prompts_per_step]
        cursor += args.prompts_per_step

        # ---- 1) 整条链路 rollout（组 = 同一任务 G 条完整轨迹）----
        rollouts = []   # [(env, trajectory, adv, reward, detail)]
        group_stats = []
        for task in task_batch:
            group, envs = generate_scored_group(model, tokenizer, task, args)
            torch.cuda.empty_cache()  # 生成阶段的 kv cache 归还驱动，给梯度阶段腾峰值
            rewards = [reward for _trajectory, reward, _detail in group]
            advs, mean, std = group_advantages(rewards)
            for (trajectory, reward, detail), adv, env in zip(group, advs, envs):
                rollouts.append((env, trajectory, adv, reward, detail))
            group_stats.append({"mean_r": round(mean, 4), "std_r": round(std, 4),
                           "rewards": [round(reward, 4) for reward in rewards]})
            traj_audit_fp.write(json.dumps({
                "step": step, "tid": task["id"], "category": task["category"],
                "group": [{"reward": round(reward, 4), "adv": round(adv, 3), "detail": detail,
                           "text": format_trajectory(env, trajectory)}
                          for (trajectory, reward, detail), adv, env in zip(group, advs, envs)],
            }, ensure_ascii=False) + "\n")
        traj_audit_fp.flush()
        details = [entry[4] for entry in rollouts]

        # ---- 2) 更新：同一批 rollout 数据重放 args.epochs 遍（μ=1 即封存口径）----
        stats = train_step_update(model, rollouts, trainable, opt, args)
        if stats["n_trained"]:
            # 学习率按训练步推进：优化步次数随 P×G/accum/μ 变化，若按优化步推进，
            # cosine 会在半程就衰减到 0（2026-10-24 修复口径的延续）
            sched.step()

        # ---- 2.5) 训练中闭环评估：greedy 固定子集（与 eval_math 同口径），追踪最优 ----
        last_eval = None
        if eval_tasks and step % args.eval_every == 0:
            eval_success, eval_reward = run_closed_eval(model, tokenizer, eval_tasks, args)
            last_eval = (eval_success, eval_reward)
            print(f"  [eval@{step:3d}] success={eval_success:.3f} reward={eval_reward:.3f}", flush=True)
            writer.add_scalar("grpo/eval_success", eval_success, step)
            writer.add_scalar("grpo/eval_reward", eval_reward, step)
            if (eval_success, eval_reward) > (best["success"], best["reward"]):
                best.update(success=eval_success, reward=eval_reward, step=step)
                model.save_pretrained(os.path.join(args.out, "best"))
                tokenizer.save_pretrained(os.path.join(args.out, "best"))
                print(f"  当前评估最优，已保存 -> {os.path.join(args.out, 'best')}", flush=True)

        # ---- 3) 按步记录（record 构造 / jsonl 落盘 / 打印 / TensorBoard 见 log_step）----
        log_step(writer, log_fp, args, step, step_t0, details, group_stats, stats, last_eval, sched)

        if step % save_steps == 0 or step == args.max_steps:
            ckpt_dir = os.path.join(args.out, f"checkpoint-{step}")
            model.save_pretrained(ckpt_dir)
            print(f"  已保存 {ckpt_dir}", flush=True)

    if best["step"]:
        print(f"闭环评估最优: step {best['step']} success={best['success']:.3f} "
              f"reward={best['reward']:.3f} -> {os.path.join(args.out, 'best')}", flush=True)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", type=str, default="outputs/merged-tora",
                    help="SFT 合并后的 bf16 模型目录（GRPO 起点，同时作 KL 参考）")
    arg_parser.add_argument("--data", type=str, default="data/tora_sft/grpo_train.jsonl",
                    help="训练任务文件（默认池已剔除 SFT 见过的题）")
    arg_parser.add_argument("--out", type=str, default="outputs/grpo-tora")
    arg_parser.add_argument("--resume", type=str, default=None, help="续训：加载此目录的 LoRA")
    arg_parser.add_argument("--max-steps", type=int, default=100)
    arg_parser.add_argument("--num-generations", type=int, default=4,
                    help="每组候选数 G（每个任务 G 条完整轨迹）")
    arg_parser.add_argument("--gen-batch", type=int, default=1,
                    help="首回合组内批量采样数：同题 G 条的 prompt 相同，一次批量出 N 条"
                         "（生成阶段提效，显存增量约百 MB 级）；1=逐条（封存口径）。"
                         "批量只换随机数流，与逐条统计等价")
    arg_parser.add_argument("--prompts-per-step", type=int, default=2,
                    help="每步任务数 P（≥2 时一次更新含多个组，显著降低单组采样的梯度方差；"
                         "时间约随 P 线性增长，约 1 分钟/任务/步）")
    arg_parser.add_argument("--max-turns", type=int, default=4,
                    help="单条轨迹最多模型回合数（代码→出错重试→终答）")
    arg_parser.add_argument("--max-len", type=int, default=2048,
                    help="单条轨迹 prompt+生成 总 token 上限（题目短、代码为主）")
    arg_parser.add_argument("--max-new-tokens", type=int, default=512,
                    help="单回合生成上限（代码块较长）")
    arg_parser.add_argument("--temperature", type=float, default=0.8,
                    help="采样温度（T=1.0 实测显著压低 SFT 模型的采样正确率，"
                         "组内全错同分→零梯度步变多）")
    arg_parser.add_argument("--top-p", type=float, default=0.95)
    arg_parser.add_argument("--lr", type=float, default=3e-5,
                    help="旧默认 1e-5 偏保守（旧版实测 grad_norm 2~5.5 且被硬剪 1.0，"
                         "等效步长过小，100 步 reward 无趋势）")
    arg_parser.add_argument("--beta", type=float, default=0.025, help="KL 系数（锚定 SFT 模型）")
    arg_parser.add_argument("--max-grad-norm", type=float, default=2.0,
                    help="梯度裁剪上限（旧版硬编码 1.0；组内归一化优势的梯度范数"
                         "常态 2~5.5，每步硬剪到 1.0 等效把学习率再砍 2~5 倍）")
    arg_parser.add_argument("--kl-clip", type=float, default=0.05, help="单 token KL 截断，防离群主导")
    arg_parser.add_argument("--epsilon", type=float, default=0.2, help="PPO 裁剪")
    arg_parser.add_argument("--save-steps", type=int, default=None, help="默认 max_steps//10")
    arg_parser.add_argument("--eval-data", type=str, default="data/tora_sft/eval_set.jsonl",
                    help="训练中闭环评估的任务文件（与 SFT/RL 训练题均不相交的持出集）")
    arg_parser.add_argument("--eval-every", type=int, default=20,
                    help="每 N 个训练步做一次闭环评估（greedy 固定子集），0 关闭；"
                         "每次约 2~4 分钟，最优 checkpoint 存 out/best")
    arg_parser.add_argument("--eval-n", type=int, default=16,
                    help="每次闭环评估的任务数（固定随机子集，各评估点分数可比）")
    arg_parser.add_argument("--max-prompt-len", type=int, default=600,
                    help="初始 prompt token 上限过滤")
    arg_parser.add_argument("--curriculum", type=int, default=1,
                    help="任务顺序课程化：各难度层轮转交织，保证每步混有低难度题"
                         "（整组全错同分 → 零梯度步 的密度显著下降）")
    arg_parser.add_argument("--lora-r", type=int, default=16)
    arg_parser.add_argument("--lora-alpha", type=int, default=32)
    arg_parser.add_argument("--grad-ckpt", type=int, default=1)
    arg_parser.add_argument("--grad-accum", type=int, default=4,
                    help="轨迹级梯度累积（每 accum 条轨迹一次 opt.step）")
    arg_parser.add_argument("--log-dir", type=str, default="logs",
                    help="步级/轨迹审计 JSONL 的落盘目录（默认 logs）。换一个目录可让新 run 的"
                         "审计记录与封存快照的 logs/grpo_*_math.jsonl 隔离，避免趋势曲线混入多套口径")
    arg_parser.add_argument("--epochs", type=int, default=1,
                    help="同一批 rollout 数据的重放遍数（PPO 的 μ）。默认 1=封存口径"
                         "（old_logp 现算，ratio 恒 1，裁剪不触发）；>1 时 old_logp 首遍缓存复用，"
                         "ratio 偏离 1、PPO 裁剪与 clip_tokens 真正工作。注意每训练步"
                         "参数移动量约 ×μ，想保持总移动可比可减 --max-steps（如 60→20）")
    arg_parser.add_argument("--turn-exec-w", type=float, default=0.5,
                    help="分轮次优势的执行信号权重：代码轮 adv += w×(+1 执行无错 / -1 报错)，"
                         "终答轮不加。0=旧行为（整条轨迹同一优势，代码被终答连坐）")
    arg_parser.add_argument("--seed", type=int, default=42)
    args = arg_parser.parse_args()
    # 前置检查：本地没有该目录时直接报中文错并给修复提示。否则 transformers 会把这个
    # 路径当成 HuggingFace 仓库名去联网查询，抛一长串难懂的 404 RepositoryNotFoundError
    if not os.path.isdir(args.model):
        sys.exit(f"模型目录不存在: {args.model}\n"
                 "  封存包不含 merged-tora（8.5GB，本地可再生），先执行合并：\n"
                 "    python -m scripts.training.merge_lora --model models/Qwen3.5-4B "
                 "--adapter outputs/qlora-tora/checkpoint-2000 --out outputs/merged-tora\n"
                 "  或用 --model 指向一个已存在的本地模型目录")
    acquire_lock()
    set_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载 NF4 量化 merged 模型 ...", flush=True)
    base_model = load_model_4bit(args.model)
    if drop_vision_tower(base_model):
        print(f"已移除视觉塔 | 当前显存 {torch.cuda.memory_allocated()/2**30:.2f}GB", flush=True)
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=bool(args.grad_ckpt),
        gradient_checkpointing_kwargs={"use_reentrant": False})
    # 冻结 fp32 权重（248k 词表的 embedding/头是最大一块）压回 bf16，避免 8GB 卡 WDDM 换页
    print(f"冻结 fp32 权重压回 bf16: {shrink_frozen_fp32(base_model)} 个张量 | 当前显存 "
          f"{torch.cuda.memory_allocated()/2**30:.2f}GB", flush=True)
    if args.resume:
        model = PeftModel.from_pretrained(base_model, args.resume, is_trainable=True)
        print(f"续训 adapter: {args.resume}", flush=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES)
        model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    train_tasks = load_tasks(args.data, tokenizer, args.max_prompt_len)
    print(f"可用任务 {len(train_tasks)}（初始 prompt <= {args.max_prompt_len} tokens）", flush=True)
    print(f"重放遍数 μ={args.epochs}（>1 时同批数据重放多遍，PPO 裁剪上岗）| "
          f"首回合批量 gen_batch={args.gen_batch}", flush=True)

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
    # 审计日志落到 --log-dir（默认 logs）：换目录可让新 run 与封存快照的记录互不覆盖
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.out, "runs"))
    log_fp = open(os.path.join(args.log_dir, "grpo_log_math.jsonl"), "a", encoding="utf-8")

    model.train()
    eval_tasks = []
    if args.eval_every > 0:
        eval_pool = load_tasks(args.eval_data, tokenizer, args.max_prompt_len)
        random.Random(0).shuffle(eval_pool)  # 固定子集：各评估点同一批任务，分数可比
        eval_tasks = eval_pool[: args.eval_n]
        print(f"训练中闭环评估: {len(eval_tasks)} 个任务（{args.eval_data}），"
              f"每 {args.eval_every} 步一次", flush=True)
    traj_audit_fp = open(os.path.join(args.log_dir, "grpo_trajs_math.jsonl"), "a", encoding="utf-8")
    train_trajectory_loop(model, tokenizer, train_tasks, args, opt, sched, trainable,
                          writer, log_fp, traj_audit_fp, save_steps, eval_tasks)
    traj_audit_fp.close()

    model.save_pretrained(os.path.join(args.out, "final"))
    tokenizer.save_pretrained(os.path.join(args.out, "final"))
    writer.close()
    log_fp.close()
    print(f"GRPO 完成 -> {os.path.join(args.out, 'final')}", flush=True)


if __name__ == "__main__":
    main()
