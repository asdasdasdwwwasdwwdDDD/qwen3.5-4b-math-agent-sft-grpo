# -*- coding: utf-8 -*-
"""从现有数据切出 L4 的两批题（同难度带、不相交），供「SFT → GRPO」两阶段使用。

输出（data/tora_sft/）：
  band_l4_sft.jsonl   来自 train.jsonl（456 道，带 ToRA 官方轨迹）→ 二轮 SFT 素材
  band_l4_grpo.jsonl  来自 grpo_pool（726 道，无轨迹）→ GRPO 素材，已剔除 eval/dev

两批交集必然为 0：convert_tora.py 在生成 pool 时已用 seen_math_idx 剔除 SFT 见过的题。
本脚本只做一次显式校验，不再依赖上游隐含保证。

字段统一为 {idx, problem, gt, level, type}，与 math_env.load_tasks 对齐；
SFT 批额外带 traj_len（原轨迹消息数），便于人工抽查。

用法: python -m scripts.tools.build_band_sets
"""
import argparse
import json
import os
import re
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DATA = os.path.join(_ROOT, "data", "tora_sft")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="Level 4", help="题目难度层")
    ap.add_argument("--sft-src", default=os.path.join(DATA, "train.jsonl"))
    ap.add_argument("--pool-src", default=os.path.join(DATA, "grpo_pool.jsonl"))
    ap.add_argument("--eval", default=os.path.join(DATA, "eval_set.jsonl"))
    ap.add_argument("--dev", default=os.path.join(DATA, "dev_set.jsonl"))
    ap.add_argument("--out-dir", default=DATA)
    args = ap.parse_args()

    # ---- 1) SFT 批：train.jsonl（多轮轨迹，idx 藏在 id 里）----
    sft_rows = []
    for row in read_jsonl(args.sft_src):
        if row.get("level") != args.level:
            continue
        m = re.search(r"(\d+)$", str(row.get("id", "")))
        if not m:
            continue
        problem = next((msg["content"] for msg in row.get("messages", [])
                        if msg.get("role") == "user"), None)
        if not problem or not row.get("gt"):
            continue
        sft_rows.append({"idx": int(m.group(1)), "problem": problem, "gt": row["gt"],
                         "level": row["level"], "type": row.get("category", ""),
                         "traj_len": len(row.get("messages", []))})

    # ---- 2) GRPO 批：grpo_pool 剔除 eval/dev ----
    forbid = set()
    for path in (args.eval, args.dev):
        if os.path.exists(path):
            forbid |= {r["idx"] for r in read_jsonl(path)}
    grpo_rows = []
    for row in read_jsonl(args.pool_src):
        if row.get("level") != args.level or row["idx"] in forbid:
            continue
        grpo_rows.append({"idx": row["idx"], "problem": row["problem"], "gt": row["gt"],
                          "level": row["level"], "type": row.get("type", "")})

    # ---- 3) 校验与落盘 ----
    a = {r["idx"] for r in sft_rows}
    b = {r["idx"] for r in grpo_rows}
    print(f"{args.level}: SFT 批 {len(sft_rows)} 道 / GRPO 批 {len(grpo_rows)} 道")
    print(f"交集 = {len(a & b)}（必须为 0）| eval+dev 已排除 {len(forbid)} 题")
    if a & b:
        sys.exit("交集非 0，拒绝写出")

    for name, rows in (("band_l4_sft.jsonl", sft_rows), ("band_l4_grpo.jsonl", grpo_rows)):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"写入 {path}")


if __name__ == "__main__":
    main()
