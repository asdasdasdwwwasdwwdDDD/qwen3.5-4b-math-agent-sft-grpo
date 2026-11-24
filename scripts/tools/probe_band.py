# -*- coding: utf-8 -*-
"""题目难度探针：同一题采 G 条轨迹，统计采样正确率 p 与「混合组率」。

为什么需要它
------------
GRPO 只用组内相对优势：全对（p→1）与全错（p→0）的组优势恒为 0，整组零梯度。
真正能学的只有「有对有错」的组，其概率为

    混合率 = 1 − p^G − (1−p)^G

G=4 时 p=0.5 混合率 88%、p=0.33 为 79%、p=0.8 只剩 59%、p=0.9 仅 34%。
所以选题必须按实测 p 分带：p 太低两阶段都无信号，p 太高 GRPO 空转。

用法
----
python -m scripts.tools.probe_band --data data/tora_sft/band_l4_sft.jsonl \
    --n 40 --out outputs/probe/l4_sft_40.jsonl
每题一行增量落盘，中断后重跑自动续测（按 id 去重）。
"""
import argparse
import json
import os
import random
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from scripts.envs.math_env import rollout_group, score_trajectory, load_tasks
from scripts.evaluation.eval_math import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="含 idx/problem/gt/level 的 jsonl")
    ap.add_argument("--n", type=int, default=40, help="抽多少题（固定种子，可复现）")
    ap.add_argument("--gens", type=int, default=4, help="每题采样条数 G")
    ap.add_argument("--gen-batch", type=int, default=4, help="首回合批量采样块大小")
    ap.add_argument("--model", default="outputs/merged-tora")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True, help="逐题结果 jsonl（增量写）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()

    model, tokenizer = load_model(args.model, args.adapter)
    pool = load_tasks(args.data, tokenizer, 600)
    random.Random(args.seed).shuffle(pool)
    tasks = pool[: args.n]
    print(f"探针: {len(tasks)} 题 × G={args.gens}（{args.data}）", flush=True)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
        if done:
            print(f"已完成 {len(done)} 题，续跑", flush=True)

    fp = open(args.out, "a", encoding="utf-8")
    start = time.time()
    for i, task in enumerate(tasks):
        if task["id"] in done:
            continue
        t0 = time.time()
        group = rollout_group(model, tokenizer, task, num_generations=args.gens,
                              gen_batch=min(args.gen_batch, args.gens),
                              max_turns=args.max_turns, max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature, top_p=args.top_p,
                              max_len=args.max_len, do_sample=True, grad_ckpt=False)
        ok, execs = [], []
        for traj, env in group:
            _reward, detail = score_trajectory(traj, env)
            ok.append(int(detail["correct"]))
            execs.append(detail["n_exec"])
        rec = {"id": task["id"], "idx": int(str(task["id"]).split("-")[-1]),
               "level": task["level"], "correct": sum(ok), "gens": len(ok),
               "p": round(sum(ok) / len(ok), 3), "n_exec": execs,
               "sec": round(time.time() - t0, 1)}
        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fp.flush()
        torch.cuda.empty_cache()
        print(f"  [{i+1}/{len(tasks)}] {task['id']} {rec['correct']}/{rec['gens']}"
              f" ({rec['sec']}s)", flush=True)
    fp.close()

    with open(args.out, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ps = [r["p"] for r in rows]
    mixed = sum(1 for r in rows if 0 < r["correct"] < r["gens"])
    print("\n=== 汇总 ===", flush=True)
    print(f"题数 {len(rows)} | 平均 p = {sum(ps)/len(ps):.3f}", flush=True)
    print(f"混合组 0<k<G : {mixed}/{len(rows)} = {mixed/len(rows):.1%}", flush=True)
    print(f"全错 p=0     : {sum(1 for r in rows if r['correct'] == 0)/len(rows):.1%}", flush=True)
    print(f"全对 p=1     : {sum(1 for r in rows if r['correct'] == r['gens'])/len(rows):.1%}", flush=True)
    print(f"总耗时 {(time.time() - start)/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
