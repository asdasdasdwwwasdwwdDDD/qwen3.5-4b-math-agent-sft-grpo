"""从 GRPO 训练池里分层留出一份 dev 集，供训练中评估/选 best 使用。

为什么必须这么做（2026-11-11 实测）：训练内评估原本从 eval_set.jsonl（仅 101 题）
抽 16 题，而这 16 题**全部落在 101 题测试集内**——等于用测试集选 best，再在同一
测试集上报数。mu2-150 的 39.0% 里，"+2pp 超过 SFT" 全部来自这 16 题（9 vs 7，
两题之差），其余 84 题与 SFT 完全相同（30/84）。

做法：按 level 分层从 grpo_train.jsonl 抽 120 题作为 dev（与 eval_set 天然不相交，
已验证 eval ∩ grpo_train = 0），训练池同步剔除这 120 题。

用法（仓库根运行）：
    python -m scripts.tools.split_dev --n 120
产物：
    data/tora_sft/dev_set.jsonl          训练中评估 / 选 best 用（dev）
    data/tora_sft/grpo_train_nodev.jsonl 实际训练用（3094 题）
"""
import argparse
import json
import os
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/tora_sft/grpo_train.jsonl")
    ap.add_argument("--eval", default="data/tora_sft/eval_set.jsonl")
    ap.add_argument("--out-dev", default="data/tora_sft/dev_set.jsonl")
    ap.add_argument("--out-train", default="data/tora_sft/grpo_train_nodev.jsonl")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    test = {json.loads(l)["problem"] for l in open(args.eval, encoding="utf-8") if l.strip()}
    leak = [r for r in rows if r["problem"] in test]
    if leak:
        raise SystemExit(f"训练池里有 {len(leak)} 道题与测试集重复，先修数据再跑")

    by_level = defaultdict(list)
    for r in rows:
        by_level[r.get("level", "?")].append(r)

    total = len(rows)
    rng = random.Random(args.seed)
    dev, keep = [], []
    for level in sorted(by_level):
        bucket = by_level[level][:]
        rng.shuffle(bucket)
        k = round(args.n * len(bucket) / total)
        dev += bucket[:k]
        keep += bucket[k:]

    for path, data in ((args.out_dev, dev), (args.out_train, keep)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"训练池 {total} 题 -> dev {len(dev)} 题 / 训练 {len(keep)} 题")
    print("dev  level 分布:", dict(sorted(Counter(r.get('level') for r in dev).items())))
    print("test level 分布:", dict(sorted(Counter(
        json.loads(l).get('level') for l in open(args.eval, encoding='utf-8') if l.strip()).items())))
    print(f"已写入 {args.out_dev} / {args.out_train}")


if __name__ == "__main__":
    main()
