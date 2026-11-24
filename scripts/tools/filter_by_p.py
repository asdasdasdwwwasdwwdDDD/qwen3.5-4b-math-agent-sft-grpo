# -*- coding: utf-8 -*-
"""按探针实测 p 从题库里筛出"有梯度"的题。

GRPO 只看组内相对优势：全对（p→1）与全错（p→0）的组优势恒为 0，整组零梯度。
所以必须用**当前模型**实测的每题 p 来筛，不能用别的模型留下的旧难度表。

用法:
  python -m scripts.tools.filter_by_p \
      --probe outputs/probe_sft200.jsonl \
      --pool  data/tora_sft/band_l4_sft.jsonl \
      --lo 0.2 --hi 0.8 \
      --out   data/tora_sft/grpo_sft_filtered.jsonl
"""
import argparse
import json
import os
import sys


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="probe_band.py 产出的逐题 jsonl")
    ap.add_argument("--pool", required=True, help="题库 jsonl（含 idx/problem/gt/level）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--min-gens", type=int, default=4, help="采样条数不足的题不采信")
    args = ap.parse_args()

    probe = load_jsonl(args.probe)
    pool = load_jsonl(args.pool)
    by_idx = {int(r["idx"]): r for r in pool}

    keep, hard, easy, missing = [], 0, 0, 0
    for r in probe:
        if r.get("gens", 0) < args.min_gens:
            continue
        idx = int(r["idx"])
        if idx not in by_idx:
            missing += 1
            continue
        p = float(r["p"])
        if p < args.lo:
            hard += 1
        elif p > args.hi:
            easy += 1
        else:
            keep.append(dict(by_idx[idx], p_probe=p))

    seen, uniq = set(), []
    for r in keep:
        key = r["problem"].strip()
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(probe)
    print(f"探针题数 {n}")
    print(f"  太难怪不着 p<{args.lo} : {hard} ({hard / max(n, 1):.0%})")
    print(f"  太简单 p>{args.hi}     : {easy} ({easy / max(n, 1):.0%})")
    print(f"  可用 {args.lo}<=p<={args.hi}  : {len(uniq)} ({len(uniq) / max(n, 1):.0%})")
    if missing:
        print(f"  警告: {missing} 条探针记录在 pool 中找不到对应 idx")
    print(f"已写出 {args.out}")


if __name__ == "__main__":
    sys.exit(main())
