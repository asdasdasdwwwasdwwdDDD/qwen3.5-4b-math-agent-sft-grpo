"""从历史 GRPO 轨迹审计日志统计每道题的正确率，生成难度表。

用途：给 `train_grpo.py --difficulty` 用。难度定向采样把「已知部分会做」的题
（正确率在 0 与 1 之间）放进优先池 —— 只有这类题能提供正确性方差。实测均匀抽题时
74.4% 的组是全对或全错，其中全错组靠格式分挤出的梯度占总梯度质量的 38%。

用法（仓库根运行）：
    python -m scripts.tools.build_difficulty                       # 扫描全部历史日志
    python -m scripts.tools.build_difficulty --out data/tora_sft/difficulty.json
"""
import argparse
import glob
import json
import os
import sys


def default_logs():
    """归档日志 + 各 run 的独立日志（--log-dir 隔离后的产物）。"""
    paths = []
    if os.path.exists("logs/grpo_trajs_math.jsonl"):
        paths.append("logs/grpo_trajs_math.jsonl")
    paths += sorted(glob.glob("outputs/grpo-*/logs/grpo_trajs_math.jsonl"))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=None, help="轨迹日志路径，默认自动扫描")
    ap.add_argument("--out", default="data/tora_sft/difficulty.json")
    ap.add_argument("--min-samples", type=int, default=1,
                    help="至少被采样过几次才写进表（默认 1）")
    args = ap.parse_args()

    paths = args.logs if args.logs else default_logs()
    if not paths:
        sys.exit("没找到任何 grpo_trajs_math.jsonl")

    stats, seen = {}, 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                group = row.get("group") or []
                if not group:
                    continue
                tid = row["tid"]
                slot = stats.setdefault(tid, [0, 0])
                slot[0] += sum(1 for x in group if x["detail"]["correct"])
                slot[1] += len(group)
                seen += 1

    table = {k: v for k, v in stats.items() if v[1] >= args.min_samples}
    mixed = [k for k, v in table.items() if 0 < v[0] < v[1]]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)

    print(f"扫描日志 {len(paths)} 个 / {seen} 个组")
    print(f"覆盖题目 {len(table)} 道 | 其中「部分会做」{len(mixed)} 道 "
          f"（{100 * len(mixed) / max(len(table), 1):.1f}%）| 全对/全错 {len(table) - len(mixed)} 道")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
