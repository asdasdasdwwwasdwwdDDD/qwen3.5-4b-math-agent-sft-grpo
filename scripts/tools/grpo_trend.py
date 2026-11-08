# -*- coding: utf-8 -*-
"""训练中途看趋势：读 logs/grpo_log_math.jsonl（每步）与 logs/grpo_trajs_math.jsonl（每组明细），
输出窗口均值、前后对比、健康检查与任务类型拆分，回答一个问题——"模型在有效学习吗？"

单步日志只有 1 个任务 × G 个样本，噪声极大，必须看窗口/前后段的均值方向。
判读要点：
  - success_rate 的窗口均值逐窗上移（至少看 20-30 步）；
  - kl 均值应远小于 kl_clip=0.05 且不为恒 0（恒 0 = 一直没更新过）；
  - 空步率（gnorm=0，组内全同分）不要长期 >1/3；
  - 分难度层看：高难度层的 success 涨了，才叫"整条链路在学"。

用法（在仓库根目录）:
  python -m scripts.tools.grpo_trend
  python -m scripts.tools.grpo_trend --window 20 --max-steps 100
"""
import argparse
import json
import os


def load_jsonl(path):
    """逐行读 JSONL，容忍正在写入的最后一行（解析失败即跳过）。"""
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


METRICS = [("reward_mean", "reward", 2), ("success_rate", "success", 2), ("coverage", "cov", 2),
           ("call_f1", "call_f1", 2), ("arg_avg", "arg_avg", 2), ("answer", "answer", 2),
           ("kl", "kl", 5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default="logs/grpo_log_math.jsonl")
    ap.add_argument("--trajs", type=str, default="logs/grpo_trajs_math.jsonl")
    ap.add_argument("--window", type=int, default=10, help="窗口大小（步）")
    ap.add_argument("--max-steps", type=int, default=None, help="与训练命令一致，用于估剩余时间")
    args = ap.parse_args()

    steps = load_jsonl(args.log)
    if not steps:
        print(f"还没有训练日志（{args.log} 为空或不存在）。训练至少跑完 1 步后再看。")
        return
    n = len(steps)
    avg_sec = mean([s["sec"] for s in steps])
    print(f"共 {n} 步（最新 step {steps[-1]['step']}）| 平均 {avg_sec:.0f}s/步 | "
          f"最后 lr {steps[-1]['lr']:.2e}")
    if args.max_steps and steps[-1]["step"] < args.max_steps:
        rem = args.max_steps - steps[-1]["step"]
        print(f"还剩约 {rem} 步 ≈ {rem * avg_sec / 60:.0f} 分钟")

    # —— 窗口趋势 ——
    print(f"\n窗口趋势（每 {args.window} 步一窗；单步噪声大，只看方向）:")
    print(f"{'steps':>10} " + " ".join(f"{name:>8}" for _, name, _ in METRICS) +
          f"{'空步率':>8}{'有效组':>8}")
    for lo in range(0, n, args.window):
        win = steps[lo:lo + args.window]
        tag = f"{win[0]['step']}-{win[-1]['step']}"
        row = f"{tag:>10} " + " ".join(f"{mean([w[k] for w in win]):8.{p}f}"
                                       for k, _, p in METRICS)
        empty = mean([1.0 if w["grad_norm"] == 0 else 0.0 for w in win])
        vgf = mean([w.get("valid_group_frac", 1.0) for w in win])
        print(row + f"{empty:8.0%}{vgf:8.2f}")

    # —— 前 1/3 vs 后 1/3 ——
    if n >= 9:
        k = max(n // 3, 3)
        old, new = steps[:k], steps[-k:]
        print(f"\n前 {k} 步 vs 后 {k} 步（是否上移）:")
        for key, name in [("reward_mean", "reward"), ("success_rate", "success"),
                          ("coverage", "coverage"), ("call_f1", "call_f1"),
                          ("arg_avg", "arg_avg"), ("answer", "answer")]:
            a, b = mean([s[key] for s in old]), mean([s[key] for s in new])
            arrow = "↑" if b > a + 0.02 else ("↓" if b < a - 0.02 else "→")
            print(f"  {name:9}{a:.3f}  {arrow}  {b:.3f}")
    else:
        print(f"\n（只有 {n} 步，等 20-30 步后前后对比才有意义）")

    # —— 健康检查 ——
    kls = [s["kl"] for s in steps]
    empty_rate = mean([1.0 if s["grad_norm"] == 0 else 0.0 for s in steps])
    vgf = mean([s.get("valid_group_frac", 1.0) for s in steps])
    kl_ok = "✓" if max(kls) < 0.01 else "⚠ 偏大，检查是否过冲"
    print("\n健康检查:")
    print(f"  kl: 均值 {mean(kls):.5f} 最大 {max(kls):.5f}（应远小于 kl_clip=0.05）{kl_ok}")
    print(f"  空步率: {empty_rate:.0%}（gnorm=0 的步，组内全同分无信号）"
          + ("⚠ 长期 >1/3 考虑 --temperature 1.0 或 --prompts-per-step 2" if empty_rate > 1/3 else ""))
    print(f"  有效组占比: {vgf:.2f}（组内有分差的占比，越高学习信号越多）")

    # —— 分类别（from trajs）——
    groups = load_jsonl(args.trajs)
    if groups:
        half = groups[len(groups) // 2]["step"]
        cats = {}
        for g in groups:
            d = cats.setdefault(g.get("category", "?"),
                                {"n": 0, "s": 0.0, "cov": 0.0, "r": 0.0,
                                 "n2": 0, "s2": 0.0, "cov2": 0.0, "r2": 0.0})
            late = g["step"] > half
            for m in g["group"]:
                det = m["detail"]
                ok = float(bool(det["success"]))
                d["n"] += 1; d["s"] += ok; d["cov"] += det["coverage"]; d["r"] += m["reward"]
                if late:
                    d["n2"] += 1; d["s2"] += ok
                    d["cov2"] += det["coverage"]; d["r2"] += m["reward"]
        print(f"\n分类别（rollout 口径；后半 = step>{half}）:")
        print(f"{'类别':<12}{'n':>4}{'success':>9}{'前半':>7}{'后半':>7}{'cov':>7}{'reward':>8}")
        for c, d in sorted(cats.items()):
            s_all = d["s"] / d["n"]
            cov = d["cov"] / d["n"]
            r = d["r"] / d["n"]
            s2 = f"{d['s2'] / d['n2']:.0%}" if d["n2"] else "-"
            n1 = d["n"] - d["n2"]
            s1 = f"{(d['s'] - d['s2']) / n1:.0%}" if n1 else "-"
            print(f"{c:<12}{d['n']:>4}{s_all:>9.0%}{s1:>7}{s2:>7}{cov:>7.2f}{r:>8.3f}")
        print("  （cov/reward 为全程口径；多步类别 chained/error_retry 的前后半分差最值得盯）")
    else:
        print(f"\n（无 {args.trajs}，跳过分类别）")


if __name__ == "__main__":
    main()
