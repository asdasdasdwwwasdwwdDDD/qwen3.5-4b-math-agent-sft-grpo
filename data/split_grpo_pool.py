# -*- coding: utf-8 -*-
"""把 grpo_pool.jsonl（3315 题）切成持出评测集 + GRPO 训练池：
eval_set.jsonl 100 题（按 Level 分层，seed=7 固定）——base/SFT/GRPO 共用同一把尺子；
grpo_train.jsonl 其余——GRPO 采样池。SFT/评测/RL 三方数据互不相交。"""
import json
import os
import random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "tora_sft", "grpo_pool.jsonl")
EVAL_N = 100


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    rng = random.Random(7)
    by_level = {}
    for r in rows:
        by_level.setdefault(r["level"], []).append(r)
    eval_rows = []
    for lv, group in sorted(by_level.items()):
        group = group[:]
        rng.shuffle(group)
        n = max(1, round(EVAL_N * len(group) / len(rows)))
        eval_rows.extend(group[:n])
    eval_idx = {r["idx"] for r in eval_rows}
    train_rows = [r for r in rows if r["idx"] not in eval_idx]
    print(f"eval_set {len(eval_rows)} / grpo_train {len(train_rows)}")
    print("eval 分层:", dict(sorted(Counter(r["level"] for r in eval_rows).items())))
    for name, data in [("eval_set.jsonl", eval_rows), ("grpo_train.jsonl", train_rows)]:
        path = os.path.join(HERE, "tora_sft", name)
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"写入 {path}")


if __name__ == "__main__":
    main()
