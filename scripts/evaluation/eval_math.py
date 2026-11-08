# -*- coding: utf-8 -*-
"""闭环数学评测：模型自己做题（```python → 受限执行 → ```output 回填 → \boxed 终答），
按 boxed 答案与 gt 等价性判正确率。base / SFT / GRPO 用同一把尺子（math_env.py）。

用法（仓库根运行）:
  python -m scripts.evaluation.eval_math --model models/Qwen3.5-4B --tag base
  python -m scripts.evaluation.eval_math --model outputs/merged-tora --adapter outputs/grpo-tora/final --tag grpo
模型加载口径：4bit NF4 量化底座 + adapter 直挂推理（不合并，量化权重不可 merge）。
"""
import argparse
import json
import time

import torch
from transformers import AutoTokenizer

from scripts.envs.math_env import MathEnv, rollout, score_trajectory, load_tasks


def load_model(model_path, adapter=None):
    """4bit 量化加载底座模型；adapter 可选直挂（与训练/评测全程同一口径）。"""
    from transformers import BitsAndBytesConfig
    # 加载必须用 VL 架构类：AutoModelForCausalLM 会静默随机初始化文本层（见 AGENTS.md）
    from transformers import Qwen3_5ForConditionalGeneration
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=True)
    if adapter:
        from peft import PeftModel
        # 4bit 量化底座上直接挂载 adapter 推理（不合并，量化权重不可 merge）
        model = PeftModel.from_pretrained(model, adapter)
        print(f"已加载 adapter: {adapter}")
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/Qwen3.5-4B")
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--data", type=str, default="data/tora_sft/eval_set.jsonl",
                    help="评测集（GRPO 训练池之外的持出题，固定不变保证各模型可比）")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--out", type=str, default="outputs/eval")
    ap.add_argument("--tag", type=str, default="model")
    args = ap.parse_args()

    model, tok = load_model(args.model, args.adapter)
    tasks = load_tasks(args.data, tok, 600)[: args.n]
    print(f"闭环数学评测: {len(tasks)} 题（{args.data}）tag={args.tag}", flush=True)

    details = []
    t0 = time.time()
    for i, t in enumerate(tasks):
        env = MathEnv(t)
        ro = rollout(model, tok, env, max_turns=args.max_turns,
                     max_new_tokens=args.max_new_tokens, temperature=1.0, top_p=1.0,
                     max_len=args.max_len, do_sample=False, grad_ckpt=False)
        r, det = score_trajectory(ro, env)
        det["reward"] = round(r, 4)
        det["problem"] = t["problem"][:160]
        details.append(det)
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            acc = sum(d["correct"] for d in details) / len(details)
            print(f"  [{i+1}/{len(tasks)}] 累计正确率 {acc:.1%} ({time.time()-t0:.0f}s)", flush=True)

    n = len(details)
    acc = sum(d["correct"] for d in details) / n
    # level 从任务里带出（detail 无 level，回填）
    for d, t in zip(details, tasks):
        d["level"] = t["level"]
    by_level = {}
    for lv in sorted({t["level"] for t in tasks}):
        sub = [d for d in details if d["level"] == lv]
        by_level[lv] = {"n": len(sub), "acc": round(sum(d["correct"] for d in sub) / len(sub), 3)}
    res = {
        "tag": args.tag, "model": args.model, "adapter": args.adapter, "n": n,
        "accuracy": round(acc, 4),
        "avg_turns": round(sum(d["n_turns"] for d in details) / n, 2),
        "avg_exec": round(sum(d["n_exec"] for d in details) / n, 2),
        "exec_error_rate": round(sum(d["n_error"] for d in details)
                                 / max(1, sum(d["n_exec"] for d in details)), 3),
        "premature_rate": round(sum(1 for d in details if d["ended"] != "final") / n, 3),
        "avg_reward": round(sum(d["reward"] for d in details) / n, 4),
        "by_level": by_level, "details": details, "sec": round(time.time() - t0, 1),
    }

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, f"math_{args.tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n==== 数学闭环评测 [{args.tag}] n={n} ({res['sec']}s) ====")
    print(f"正确率 {acc:.1%} | 平均回合 {res['avg_turns']} | 执行 {res['avg_exec']} 次/题 "
          f"(报错率 {res['exec_error_rate']:.1%}) | 未终答 {res['premature_rate']:.1%}")
    for lv, v in by_level.items():
        print(f"  {lv}: {v['acc']:.1%} ({v['n']} 题)")
    print(f"结果 -> {out_json}", flush=True)

    # 汇总表追加一行（多个 tag 对比用）
    summary = os.path.join(args.out, "math_summary.md")
    if not os.path.exists(summary):
        with open(summary, "w", encoding="utf-8") as f:
            f.write("# 数学闭环评测汇总（eval_set 100 题持出集）\n\n"
                    "| tag | accuracy | avg_turns | exec/题 | 报错率 | 未终答 | by_level |\n"
                    "|---|---|---|---|---|---|---|\n")
    with open(summary, "a", encoding="utf-8") as f:
        f.write(f"| {args.tag} | {acc:.1%} | {res['avg_turns']} | {res['avg_exec']} | "
                f"{res['exec_error_rate']:.1%} | {res['premature_rate']:.1%} | "
                f"{json.dumps({k: v['acc'] for k, v in by_level.items()}, ensure_ascii=False)} |\n")


if __name__ == "__main__":
    main()
