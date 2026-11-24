# -*- coding: utf-8 -*-
"""单题诊断：打印一条完整轨迹的文本与判分细节，用于排查「格式对但答错」。
用法:
  python -m scripts.tools.debug_one --data data/tora_sft/band_l4_sft.jsonl \
      --id math-test-1485 --model outputs/merged-sft-l4
"""
import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.envs.math_env import (MathEnv, rollout, score_trajectory,
                                   format_trajectory, load_tasks, extract_boxed)
from scripts.evaluation.eval_math import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--model", default="outputs/merged-sft-l4")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--gens", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=2048)
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    tail = args.id.split("-")[-1]
    row = next(r for r in rows if r.get("id") == args.id or str(r.get("idx")) == tail)
    model, tokenizer = load_model(args.model, args.adapter)
    tasks = load_tasks(args.data, tokenizer, 600)
    task = next(t for t in tasks if str(t["id"]) == args.id or str(t["id"]).endswith(tail))

    print(f"题目: {task['id']} | gt={task['gt']!r}", flush=True)
    for g in range(args.gens):
        env = MathEnv(task)
        traj = rollout(model, tokenizer, env, max_turns=args.max_turns,
                       max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature, top_p=1.0,
                       max_len=args.max_len, do_sample=True, grad_ckpt=False)
        r, d = score_trajectory(traj, env)
        final_text = format_trajectory(env, traj, max_chars=4000)
        print(f"\n===== 采样 {g+1} | reward={r:.2f} | ended={traj['ended']} | {d} =====", flush=True)
        print(final_text, flush=True)
        last = traj["turns"][-1]["text"] if traj["turns"] else ""
        print(f"--- 末轮 boxed 抽取: {extract_boxed(last)!r} vs gt {task['gt']!r}", flush=True)


if __name__ == "__main__":
    main()
