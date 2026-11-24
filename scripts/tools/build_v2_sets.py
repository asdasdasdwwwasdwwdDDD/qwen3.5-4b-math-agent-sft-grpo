# -*- coding: utf-8 -*-
"""构建 grpo v2 题集四件套（与旧题集完全分开，写入 data/grpo_v2/）：

  difficulty_v2.json   全池测量汇总 {tid: [correct, gens]}（train_grpo --difficulty 直接可用）
  pool_train_v2.jsonl  训练池 = grpo_train_nodev - dev_v2 - eval_v2
  dev_v2.jsonl         训练中评估尺子（半会带，100 题）
  eval_v2.jsonl        最终测试尺子（半会带，80 题，与 dev_v2 不相交）

选题口径（用户指定 2026-11-21）：尺子题必须是「半会」类型（不太难不太易，p 有分辨率）、
模型没训练过的。没训练过由池保证：grpo_train_nodev 已剔除 SFT 见过 / 旧 dev / 旧 eval。
半会带优先聚合 p∈[0.25,0.75]，不足时用边缘半会补齐。

测量来源（全部按 tid 聚合，标来源）：旧难度表、outputs/probe/*.jsonl、
data/grpo_v2/probe_results.jsonl（新探针）、logs/grpo_g8 与 logs/grpo_mid100 的
轨迹审计（merged-tora 干净口径）。

用法（仓库根）: python -m scripts.tools.build_v2_sets [--out-dir data/grpo_v2]
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import Counter, defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEV_N = 100
EVAL_N = 80


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tid_of(idx):
    return f"math-test-{idx}"


def collect_measurements(pool_idx):
    """聚合全部来源的每题测量 -> {tid: {"c","n","src":Counter}}。只保留池内题。"""
    agg = defaultdict(lambda: {"c": 0, "n": 0, "src": Counter()})

    def add(tid, c, n, src):
        if int(tid.split("-")[-1]) not in pool_idx or n <= 0:
            return
        a = agg[tid]
        a["c"] += c
        a["n"] += n
        a["src"][src] += 1

    old = json.load(open("data/tora_sft/difficulty.json", encoding="utf-8"))
    for k, (c, n) in old.items():
        add(k, c, n, "old")

    for f in sorted(glob.glob("outputs/probe/*.jsonl")):
        name = "probe:" + os.path.basename(f)
        for r in load_jsonl(f):
            tid = r.get("id") or tid_of(r["idx"])
            add(tid, r["correct"], r["gens"], name)

    v2 = "data/grpo_v2/probe_results.jsonl"
    if os.path.exists(v2):
        for r in load_jsonl(v2):
            add(r["id"], r["correct"], r["gens"], "probe_v2")

    for tag, path in (("g8", "logs/grpo_g8/grpo_trajs_math.jsonl"),
                      ("mid100", "logs/grpo_mid100/grpo_trajs_math.jsonl")):
        if os.path.exists(path):
            for row in load_jsonl(path):
                k = sum(1 for t in row["group"] if t["detail"]["correct"])
                add(row["tid"], k, len(row["group"]), tag)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/grpo_v2")
    ap.add_argument("--dev-n", type=int, default=DEV_N)
    ap.add_argument("--eval-n", type=int, default=EVAL_N)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    pool = load_jsonl("data/tora_sft/grpo_train_nodev.jsonl")
    pool_idx = {t["idx"] for t in pool}
    task_of = {tid_of(t["idx"]): t for t in pool}
    print(f"训练池 grpo_train_nodev: {len(pool)} 题")

    agg = collect_measurements(pool_idx)
    measured = {tid: a for tid, a in agg.items() if a["n"] > 0}
    mixed = {tid: a for tid, a in measured.items() if 0 < a["c"] < a["n"]}
    trusted = lambda a: any(s != "old" for s in a["src"])
    band_a = {tid: a for tid, a in mixed.items()
              if 0.25 <= a["c"] / a["n"] <= 0.75}
    print(f"已测量 {len(measured)} | 半会 {len(mixed)}（中带 p∈[0.25,0.75] {len(band_a)}，"
          f"其中新口径可信 {sum(1 for a in band_a.values() if trusted(a))}）")

    # ---- 选 dev_v2 / eval_v2：中带优先、可信来源优先，同优先级内随机洗后稳定排序 ----
    # Level 1 不入选：T=0.8 采样下显得「半会」多为采样噪声，greedy 近天花板（旧 dev-40
    # 实测 L1 acc 1.0），对 greedy 评测尺没有区分度。
    def pick(n_needed, taken):
        cands = [tid for tid in mixed if tid not in taken
                 and task_of[tid]["level"] != "Level 1"]
        rng.shuffle(cands)
        # 排序键：中带优先(0) > 边缘(1)；可信优先(0) > 仅旧表(1)
        cands.sort(key=lambda tid: (0 if tid in band_a else 1,
                                    0 if trusted(mixed[tid]) else 1))
        chosen = cands[:n_needed]
        if len(chosen) < n_needed:
            print(f"警告: 半会题不足 {n_needed}（现有 {len(chosen)}），"
                  f"尺子规模将缩小——考虑补探或放宽分带")
        return chosen

    dev_ids = pick(args.dev_n, set())
    eval_ids = pick(args.eval_n, set(dev_ids))
    print(f"dev_v2 {len(dev_ids)} 题 | eval_v2 {len(eval_ids)} 题（两集不相交，均已排除 L1）")

    dev_tasks = [task_of[t] for t in dev_ids]
    eval_tasks = [task_of[t] for t in eval_ids]

    # ---- 训练池 = 池 - 两把尺子 ----
    ruler_idx = {t["idx"] for t in dev_tasks + eval_tasks}
    train_pool = [t for t in pool if t["idx"] not in ruler_idx]

    # ---- 校验（硬断言，不过即失败）----
    old_dev = load_jsonl("data/tora_sft/dev_set.jsonl")
    old_eval = load_jsonl("data/tora_sft/eval_set.jsonl")
    old_dev_idx = {r["idx"] for r in old_dev if "idx" in r}
    old_eval_idx = {r["idx"] for r in old_eval if "idx" in r}
    dev_idx = {t["idx"] for t in dev_tasks}
    eval_idx = {t["idx"] for t in eval_tasks}
    train_idx = {t["idx"] for t in train_pool}
    assert not (dev_idx & eval_idx), "dev_v2 与 eval_v2 相交"
    assert not (ruler_idx & train_idx), "尺子题漏进了训练池"
    assert not (dev_idx & old_dev_idx) and not (dev_idx & old_eval_idx), \
        "dev_v2 撞上旧 dev/eval"
    assert not (eval_idx & old_dev_idx) and not (eval_idx & old_eval_idx), \
        "eval_v2 撞上旧 dev/eval"
    assert ruler_idx <= pool_idx and len(train_pool) + len(ruler_idx) == len(pool)
    miss = [t for t in dev_tasks + eval_tasks if tid_of(t["idx"]) not in measured]
    assert not miss, f"尺子里有未测量题 {len(miss)} 道"
    # 尺子里不允许出现 SFT 级别「全对」题（半会口径自检：全部 0<p<1）
    assert all(0 < measured[tid_of(t["idx"])]["c"] < measured[tid_of(t["idx"])]["n"]
               for t in dev_tasks + eval_tasks)

    # ---- 落盘 ----
    diff_v2 = {tid: [a["c"], a["n"]] for tid, a in measured.items()}
    report = {
        "pool": len(pool), "measured": len(measured),
        "mixed_total": len(mixed), "band_a": len(band_a),
        "mixed_in_train_pool": sum(1 for tid in mixed
                                   if int(tid.split("-")[-1]) in train_idx),
        "dev_n": len(dev_tasks), "eval_n": len(eval_tasks),
        "dev_level": dict(Counter(t["level"] for t in dev_tasks)),
        "eval_level": dict(Counter(t["level"] for t in eval_tasks)),
        "dev_p": [round(measured[tid_of(t["idx"])]["c"] / measured[tid_of(t["idx"])]["n"], 3)
                  for t in dev_tasks],
        "eval_p": [round(measured[tid_of(t["idx"])]["c"] / measured[tid_of(t["idx"])]["n"], 3)
                   for t in eval_tasks],
        "sources": {tid: dict(a["src"]) for tid, a in sorted(measured.items())},
    }
    with open(os.path.join(args.out_dir, "difficulty_v2.json"), "w", encoding="utf-8") as f:
        json.dump(diff_v2, f, ensure_ascii=False, indent=1)
    for name, tasks in (("pool_train_v2", train_pool), ("dev_v2", dev_tasks),
                        ("eval_v2", eval_tasks)):
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "selection_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print(f"\n== 落盘 {args.out_dir} ==")
    print(f"difficulty_v2.json {len(diff_v2)} 题 | pool_train_v2 {len(train_pool)} | "
          f"dev_v2 {len(dev_tasks)} | eval_v2 {len(eval_tasks)}")
    print(f"训练池内半会题: {report['mixed_in_train_pool']}（抽题主战场）")
    print(f"dev_v2 level 分布 {report['dev_level']}，p 均值 "
          f"{sum(report['dev_p'])/max(1,len(report['dev_p'])):.2f}")
    print(f"eval_v2 level 分布 {report['eval_level']}，p 均值 "
          f"{sum(report['eval_p'])/max(1,len(report['eval_p'])):.2f}")
    print("校验全部通过：两尺子不相交、均∈池、与旧 dev/eval 全不相交、全部带测量且为半会")


if __name__ == "__main__":
    main()
