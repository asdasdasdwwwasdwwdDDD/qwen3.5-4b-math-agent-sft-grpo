# -*- coding: utf-8 -*-
"""ToRA 轨迹数据转换：内联单轮 → 智能体多轮 messages + 训练/验证划分 + GRPO 提示池。

输入（D:/NLP/ft-qwen3.5-4b-agent-cot-data/data/tora/）：
  - tora_sft_2k_stratified.jsonl  2000 条官方模型轨迹（user 题目 + assistant 内联
    "```python 代码 → ```output 真实执行结果 → 收尾 \boxed 答案"）
  - math_test.jsonl               MATH 5000 题（problem/level/type/solution）

输出（data/tora_sft/）：
  - train.jsonl / val.jsonl  多轮轨迹：user(题) → assistant(```python) →
    tool(```output 执行结果) → assistant(收尾+boxed)。loss 只算两个 assistant 轮
    （代码生成 + 最终作答），题目与执行结果被 mask——模型学"写代码"与"读结果作答"，
    不学"背题目"与"猜执行输出"。
    注意执行结果必须用 role="tool"（模板渲染成 <tool_response> 包裹的 user 轮，
    且被 last_query 回扫豁免）：若用裸 user 轮，Qwen3.5 模板会把中间的 assistant
    代码轮剥掉 <think> 块、与生成提示渲染分叉，公共前缀对齐算法会让代码轮整个被
    mask（只训收尾答案不训代码）。
  - grpo_pool.jsonl           GRPO 提示池：math_test 中 SFT 未见过且可提取 gt 的题
    （idx 与 SFT 的 math-test-N id 对齐，已验证 20/20 题面精确一致）。

用法: python data/convert_tora.py  （seed 固定，可复现）
"""
import argparse
import json
import os
import random
import re
from collections import Counter

SRC_DIR = r"D:/NLP/ft-qwen3.5-4b-agent-cot-data/data/tora"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tora_sft")

BLOCK_PAT = re.compile(r"```python\n(.*?)```\s*```output\n(.*?)```", re.S)


def extract_boxed(sol):
    """从 MATH 官方解答里取最后一个 \\boxed{...} 的内容（支持一层嵌套花括号）。"""
    m = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", sol)
    return m[-1] if m else None


def split_trace(content):
    """内联轨迹 → (代码, 执行输出, 收尾文本)；结构不符返回 None。"""
    m = BLOCK_PAT.search(content)
    if not m:
        return None
    code = m.group(1).strip("\n")
    out = m.group(2).strip("\n")
    tail = content[m.end():].strip()
    if not code or not out or not tail:
        return None
    return code, out, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(SRC_DIR, "tora_sft_2k_stratified.jsonl"),
                                        encoding="utf-8") if l.strip()]
    math_rows = [json.loads(l) for l in open(os.path.join(SRC_DIR, "math_test.jsonl"),
                                             encoding="utf-8") if l.strip()]

    # ---- 1) 拆分内联轨迹为多轮 messages ----
    samples, failed = [], 0
    seen_math_idx = set()
    for r in rows:
        parts = split_trace(r["messages"][1]["content"])
        if parts is None:
            failed += 1
            continue
        code, out, tail = parts
        samples.append({
            "id": r["id"],
            "category": "math-tora",
            "strat": r["strat"],
            "level": r["level"],
            "gt": r["gt"],
            "messages": [
                {"role": "user", "content": r["messages"][0]["content"]},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
                {"role": "tool", "content": f"```output\n{out}\n```"},
                {"role": "assistant", "content": tail},
            ],
        })
        m = re.match(r"math-test-(\d+)$", r["id"])
        if m:
            seen_math_idx.add(int(m.group(1)))
    print(f"拆分成功 {len(samples)} / {len(rows)}（失败 {failed}）")

    # ---- 2) 按难度层分层切验证集（保持各层比例） ----
    rng = random.Random(args.seed)
    by_strat = {}
    for s in samples:
        by_strat.setdefault(s["strat"], []).append(s)
    val, train = [], []
    for strat, group in sorted(by_strat.items()):
        group = group[:]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * args.val_ratio))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    print(f"train {len(train)} 条 / val {len(val)} 条")
    print("val 分层:", Counter(s["strat"] for s in val))

    # ---- 3) GRPO 提示池：math_test 去掉 SFT 见过的题，保留可提取 gt 的 ----
    pool = []
    for r in math_rows:
        if r["idx"] in seen_math_idx:
            continue
        gt = extract_boxed(r["solution"])
        if gt is None:
            continue
        pool.append({"idx": r["idx"], "problem": r["problem"], "gt": gt,
                     "level": r["level"], "type": r["type"]})
    print(f"GRPO 提示池 {len(pool)} 条（已剔除 SFT 见过的 {len(seen_math_idx)} 题）")
    print("难度分布:", dict(sorted(Counter(p["level"] for p in pool).items())))

    # ---- 4) 落盘 ----
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, data in [("train.jsonl", train), ("val.jsonl", val), ("grpo_pool.jsonl", pool)]:
        path = os.path.join(OUT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"写入 {path}（{len(data)} 条）")


if __name__ == "__main__":
    main()
