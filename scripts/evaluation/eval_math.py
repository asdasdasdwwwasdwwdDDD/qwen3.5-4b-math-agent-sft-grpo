# -*- coding: utf-8 -*-
"""闭环数学评测：模型自己做题（```python → 受限执行 → ```output 回填 → \boxed 终答），
按 boxed 答案与 gt 等价性判正确率。base_model / SFT / GRPO 用同一把尺子（math_env.py）。

用法（仓库根运行）:
  python -m scripts.evaluation.eval_math --model models/Qwen3.5-4B --tag base_model
  python -m scripts.evaluation.eval_math --model outputs/merged-tora --adapter outputs/grpo-tora/final --tag grpo
模型加载口径：4bit NF4 量化底座 + adapter 直挂推理（不合并，量化权重不可 merge）。
"""
import argparse
import json
import os
import random
import sys
import time

import torch
from transformers import AutoTokenizer

# 兼容两种运行方式：除文档用法 `python -m scripts.evaluation.eval_math` 外，直接以文件
# 方式运行（如 VS Code「运行 Python 文件」）时 sys.path 里没有仓库根，包导入会
# ModuleNotFoundError；此处把仓库根补进 sys.path（前者场景下该路径已存在，无副作用）
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.envs.math_env import MathEnv, rollout, score_trajectory, load_tasks


def load_model(model_path, adapter=None):
    """4bit 量化加载底座模型；adapter 可选直挂（与训练/评测全程同一口径）。"""
    from transformers import BitsAndBytesConfig
    # 加载必须用 VL 架构类：AutoModelForCausalLM 会静默随机初始化文本层（见 AGENTS.md）
    # 与训练/合并脚本同走完整模块路径，不依赖顶层命名空间是否导出该类
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=True)
    if adapter:
        from peft import PeftModel
        # 4bit 量化底座上直接挂载 adapter 推理（不合并，量化权重不可 merge）
        model = PeftModel.from_pretrained(model, adapter)
        print(f"已加载 adapter: {adapter}")
    model.eval()
    return model, tokenizer


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", type=str, default="models/Qwen3.5-4B")
    arg_parser.add_argument("--adapter", type=str, default=None)
    arg_parser.add_argument("--data", type=str, default="data/tora_sft/eval_set.jsonl",
                    help="评测集（GRPO 训练池之外的持出题，固定不变保证各模型可比）")
    arg_parser.add_argument("--n", type=int, default=100)
    arg_parser.add_argument("--max-turns", type=int, default=4)
    arg_parser.add_argument("--max-new-tokens", type=int, default=640)
    arg_parser.add_argument("--max-len", type=int, default=2048)
    arg_parser.add_argument("--out", type=str, default="outputs/eval")
    arg_parser.add_argument("--tag", type=str, default="model")
    arg_parser.add_argument("--subset-order", type=str, default="shuffle",
                    choices=["shuffle", "file"],
                    help="--n 小于全集时的取题口径：shuffle=固定种子打乱（分层子集，默认，"
                         "正确率可无偏估计全集）；file=按文件顺序取前 N（旧口径，"
                         "因 eval_set 按难度排序会系统性高估，仅用于与历史分数对齐）")
    args = arg_parser.parse_args()
    # 前置检查：本地没有该目录时直接报中文错并给修复提示。否则 transformers 会把这个
    # 路径当成 HuggingFace 仓库名去联网查询，抛一长串难懂的 404 RepositoryNotFoundError
    if not os.path.isdir(args.model):
        sys.exit(f"模型目录不存在: {args.model}\n"
                 "  用 --model 指向一个已存在的本地模型目录（如 models/Qwen3.5-4B 或 "
                 "outputs/merged-tora；merged-tora 未入封存包，先用 merge_lora.py 合并）")
    if args.adapter and not os.path.isdir(args.adapter):
        sys.exit(f"adapter 目录不存在: {args.adapter}")

    model, tokenizer = load_model(args.model, args.adapter)
    eval_pool = load_tasks(args.data, tokenizer, 600)
    # 口径修复（2026-11-10）：eval_set.jsonl 是按难度排序的（L1→L5），直接取前 N 题
    # 会变成"只考简单题"的系统性高估（跑 40 题时虚高到 72.5%，全集只有 ~40%）。
    # --n 小于全集时先按固定种子打乱，保证子集是分层的；取全集时顺序不变，与历史分数可比。
    if args.n < len(eval_pool) and args.subset_order == "shuffle":
        random.Random(0).shuffle(eval_pool)
    eval_tasks = eval_pool[: args.n]
    print(f"闭环数学评测: {len(eval_tasks)} 题（{args.data}，取题口径={args.subset_order}）"
          f"tag={args.tag}", flush=True)

    details = []
    start_time = time.time()
    for task_i, task in enumerate(eval_tasks):
        env = MathEnv(task)
        trajectory = rollout(model, tokenizer, env, max_turns=args.max_turns,
                             max_new_tokens=args.max_new_tokens, temperature=1.0, top_p=1.0,
                             max_len=args.max_len, do_sample=False, grad_ckpt=False)
        reward, detail = score_trajectory(trajectory, env)
        detail["reward"] = round(reward, 4)
        detail["problem"] = task["problem"][:160]
        details.append(detail)
        torch.cuda.empty_cache()
        if (task_i + 1) % 10 == 0:
            acc = sum(detail["correct"] for detail in details) / len(details)
            print(f"  [{task_i+1}/{len(eval_tasks)}] 累计正确率 {acc:.1%} ({time.time()-start_time:.0f}s)", flush=True)

    n_total = len(details)
    acc = sum(detail["correct"] for detail in details) / n_total
    # level 从任务里带出（detail 无 level，回填）
    for detail, task in zip(details, eval_tasks):
        detail["level"] = task["level"]
    by_level = {}
    for level in sorted({task["level"] for task in eval_tasks}):
        level_details = [detail for detail in details if detail["level"] == level]
        by_level[level] = {"n": len(level_details), "acc": round(sum(detail["correct"] for detail in level_details) / len(level_details), 3)}
    result = {
        "tag": args.tag, "model": args.model, "adapter": args.adapter, "n": n_total,
        "accuracy": round(acc, 4),
        "avg_turns": round(sum(detail["n_turns"] for detail in details) / n_total, 2),
        "avg_exec": round(sum(detail["n_exec"] for detail in details) / n_total, 2),
        "exec_error_rate": round(sum(detail["n_error"] for detail in details)
                                 / max(1, sum(detail["n_exec"] for detail in details)), 3),
        "premature_rate": round(sum(1 for detail in details if detail["ended"] != "final") / n_total, 3),
        "avg_reward": round(sum(detail["reward"] for detail in details) / n_total, 4),
        "by_level": by_level, "details": details, "sec": round(time.time() - start_time, 1),
    }

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, f"math_{args.tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n==== 数学闭环评测 [{args.tag}] n={n_total} ({result['sec']}s) ====")
    print(f"正确率 {acc:.1%} | 平均回合 {result['avg_turns']} | 执行 {result['avg_exec']} 次/题 "
          f"(报错率 {result['exec_error_rate']:.1%}) | 未终答 {result['premature_rate']:.1%}")
    for level, stat in by_level.items():
        print(f"  {level}: {stat['acc']:.1%} ({stat['n']} 题)")
    print(f"结果 -> {out_json}", flush=True)

    # 汇总表追加一行（多个 tag 对比用）
    summary = os.path.join(args.out, "math_summary.md")
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write("# 数学闭环评测汇总（eval_set 100 题持出集）\n\n"
                    "| tag | accuracy | avg_turns | exec/题 | 报错率 | 未终答 | by_level |\n"
                    "|---|---|---|---|---|---|---|\n")
    with open(summary, "a", encoding="utf-8") as f:
        f.write(f"| {args.tag} | {acc:.1%} | {result['avg_turns']} | {result['avg_exec']} | "
                f"{result['exec_error_rate']:.1%} | {result['premature_rate']:.1%} | "
                f"{json.dumps({k: v['acc'] for k, v in by_level.items()}, ensure_ascii=False)} |\n")


if __name__ == "__main__":
    main()
