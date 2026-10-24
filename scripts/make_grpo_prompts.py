# -*- coding: utf-8 -*-
"""从 agent 轨迹提取 GRPO 的步级决策点（每个 assistant 轮 = 一个决策点）。

每条输出 JSONL：
  {"pid", "sample_id", "turn_idx", "category", "tools", "prefix"(该轮之前的消息),
   "gt"(该轮真值 assistant 消息)}
只从训练集提取（验证集保持留出，供 eval_agent 做前后对比不受污染）。
用法:
  python scripts/make_grpo_prompts.py --in data/train.jsonl --out data/grpo_prompts.jsonl
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default="data/train.jsonl")
    ap.add_argument("--out", type=str, default="data/grpo_prompts.jsonl")
    args = ap.parse_args()

    n_out = 0
    cats = {}
    kinds = {"tool": 0, "final": 0}
    with open(args.inp, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            for i, m in enumerate(s["messages"]):
                if m["role"] != "assistant":
                    continue
                rec = {
                    "pid": f"{s['id']}#{i}",
                    "sample_id": s["id"],
                    "turn_idx": i,
                    "category": s["category"],
                    "tools": s["tools"],
                    "prefix": s["messages"][:i],
                    "gt": m,
                }
                g.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
                cats[s["category"]] = cats.get(s["category"], 0) + 1
                kinds["tool" if m.get("tool_calls") else "final"] += 1
    print(f"提取决策点 {n_out} 个 -> {args.out}")
    print(f"类型分布: 工具调用步 {kinds['tool']} | 回答/澄清步 {kinds['final']}")
    print(f"类别分布: {cats}")


if __name__ == "__main__":
    main()
