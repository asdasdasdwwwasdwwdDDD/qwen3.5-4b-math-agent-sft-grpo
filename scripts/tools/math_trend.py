# -*- coding: utf-8 -*-
"""数学 GRPO 的事后可分析视图（TensorBoard 原生曲线的替代品）。

问题：train_grpo 的原生指标对本 run 几乎不可读——
  coverage/call_f1/arg_avg/answer 是旧智能体链路字段，math 模式恒 0；
  loss/kl 组内优势抵消恒 0；reward/success 每步难度不同（课程化轮转），
  原始曲线是难度噪声不是学习信号。
做法：从 logs/grpo_trajs_math.jsonl 的轨迹明细重建可读趋势——
  1) 按难度层的采样正确率/得分（同层自身前后对比，消除难度混合干扰）
  2) 代码执行报错率（代码质量，应随训练下降）
  3) [eval@N] 闭卷考试序列（eval_set 前缀切片，注明难度构成再解读）
  --tb：把逐层曲线写成独立 TensorBoard run（outputs/grpo-tora/runs_derived），
  与原 run 并存，浏览器刷新即可按层折叠查看。
用法（仓库根运行）: python -m scripts.tools.math_trend [--tb]
"""
import argparse
import collections
import json
import os
from statistics import mean

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def short(cat):
    return (cat or "?").replace("MATH-Level ", "L").replace("GSM8K-", "G-")


def pct(x):
    return f"{x:.0%}" if x is not None else "  - "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajs", type=str, default="logs/grpo_trajs_math.jsonl")
    ap.add_argument("--log", type=str, default="logs/grpo_log_math.jsonl")
    ap.add_argument("--eval-data", type=str, default="data/tora_sft/eval_set.jsonl")
    ap.add_argument("--eval-n", type=int, default=16, help="训练中评测的题数（与 train_grpo 一致）")
    ap.add_argument("--tb", action="store_true", help="把逐层曲线写成独立 TensorBoard run")
    args = ap.parse_args()

    trajs = load_jsonl(os.path.join(ROOT, args.trajs))
    steps = load_jsonl(os.path.join(ROOT, args.log))
    if not trajs:
        print("还没有轨迹明细（logs/grpo_trajs_math.jsonl 为空），先跑几步再看。")
        return

    # 轨迹 → (层, 步) 聚合：采样正确率 / reward / 执行报错
    by_cat = collections.defaultdict(list)   # 层 -> [(step, succ, reward)]
    by_step = {}                             # 步 -> {n_exec, n_error, n_traj, succ}
    for t in trajs:
        lv = short(t.get("category"))
        for g in t["group"]:
            d = g["detail"]
            by_cat[lv].append((t["step"], 1.0 if d["success"] else 0.0, g["reward"]))
            s = by_step.setdefault(t["step"], {"n_exec": 0, "n_error": 0, "succ": []})
            s["n_exec"] += d.get("n_exec", 0)
            s["n_error"] += d.get("n_error", 0)
            s["succ"].append(1.0 if d["success"] else 0.0)

    all_steps = sorted(by_step)
    n = len(all_steps)
    thirds = [all_steps[:n // 3], all_steps[n // 3: 2 * n // 3], all_steps[2 * n // 3:]]
    thirds = [t for t in thirds if t]

    def win_succ(lv, steps_in):
        rows = [s for (st, s, _) in by_cat[lv] if st in steps_in]
        return mean(rows) if rows else None

    print(f"共 {n} 步（step {all_steps[0]}-{all_steps[-1]}），{len(trajs)} 组 / "
          f"{sum(len(t['group']) for t in trajs)} 条轨迹\n")

    print("== 按难度层的采样正确率（同层前后对比，箭头=首末窗变化）==")
    hdr = "层      n   " + "".join(f"  {t[0]}-{t[-1]}步" for t in thirds)
    print(hdr)
    for lv in sorted(by_cat):
        rows = by_cat[lv]
        cells, vals = [], []
        for t in thirds:
            v = win_succ(lv, set(t))
            vals.append(v)
            cells.append(pct(v))
        first, last = vals[0], vals[-1]
        arrow = "↑" if first is not None and last is not None and last > first + 0.03 else \
                ("↓" if first is not None and last is not None and last < first - 0.03 else "→")
        print(f"{lv:<6} {len(rows):>3}  " + "     ".join(cells) + f"   {arrow}")

    print("\n== 全层汇总与代码执行报错率 ==")
    for t in thirds:
        ex = sum(by_step[s]["n_exec"] for s in t)
        er = sum(by_step[s]["n_error"] for s in t)
        succ = mean([x for s in t for x in by_step[s]["succ"]])
        print(f"  {t[0]:>2}-{t[-1]:>2} 步: 采样正确率 {succ:.0%} | 执行报错 "
              f"{er}/{ex} = {er / max(ex, 1):.0%}")

    # [eval@N] 序列 + 难度构成（切片偏易，不能与 30/101 题口径横比）
    evals = [(s["step"], s["eval_success"]) for s in steps if "eval_success" in s]
    if evals:
        et = load_jsonl(os.path.join(ROOT, args.eval_data))
        comp = collections.Counter(short(t.get("level", "?")) for t in et[:args.eval_n])
        comp_s = " + ".join(f"{k}×{v}" for k, v in sorted(comp.items()))
        print(f"\n== 闭卷考试 [eval@N]（greedy，前 {args.eval_n} 题切片，构成: {comp_s}）==")
        for st, v in evals:
            print(f"  eval@{st}: 正确率 {v:.0%}")

    if args.tb:
        from torch.utils.tensorboard import SummaryWriter
        out = os.path.join(ROOT, "outputs", "grpo-tora", "runs_derived")
        w = SummaryWriter(out)
        for lv in sorted(by_cat):
            for st, s, r in by_cat[lv]:
                w.add_scalar(f"按层正确率/{lv}", s, st)
                w.add_scalar(f"按层得分/{lv}", r, st)
        for st, s in sorted(by_step.items()):
            ex, er = s["n_exec"], s["n_error"]
            if ex:
                w.add_scalar("代码执行报错率", er / ex, st)
            w.add_scalar("本步采样正确率", mean(s["succ"]), st)
        for st, v in evals:
            w.add_scalar("闭卷考试正确率", v, st)
        w.close()
        print(f"\n已写入 TensorBoard run: outputs/grpo-tora/runs_derived "
              f"（浏览器刷新后按「按层正确率/按层得分/代码执行报错率」折叠查看）")


if __name__ == "__main__":
    main()
