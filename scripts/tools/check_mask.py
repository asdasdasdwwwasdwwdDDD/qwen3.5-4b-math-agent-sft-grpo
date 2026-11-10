# -*- coding: utf-8 -*-
"""验证 loss mask：解码出所有被训练的 token，人工检查是否恰好覆盖 assistant 回复区间
用法（仓库根运行）: python -m scripts.tools.check_mask [--data data/tora_sft/val.jsonl]"""
import json
import argparse
import os
import sys

from transformers import AutoTokenizer

# 兼容两种运行方式：除文档用法 `python -m scripts.tools.check_mask` 外，直接以文件
# 方式运行（如 VS Code「运行 Python 文件」）时 sys.path 里没有仓库根，包导入会
# ModuleNotFoundError；此处把仓库根补进 sys.path（前者场景下该路径已存在，无副作用）
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.training.train_qlora import AgentTrajectoryDataset

MODEL = "models/Qwen3.5-4B"

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=str, default="data/tora_sft/val.jsonl")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--max-len", type=int, default=2048)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

samples = [json.loads(l) for l in open(args.data, encoding="utf-8")][: args.n]
ds = AgentTrajectoryDataset(args.data, tok, max_len=args.max_len)

for s, enc in zip(samples, list(ds)[: args.n]):
    ids = enc["input_ids"]
    labels = enc["labels"]
    trained = [t for t, l in zip(ids, labels) if l != -100]
    tag = s.get("category", s.get("strat", "?"))
    print(f"\n===== {s.get('id','?')} ({tag}) 总长 {len(ids)} 训练token {len(trained)} =====")
    print("【被训练的内容】")
    print(tok.decode(trained))
    print("【被 mask 的内容】")
    masked = [t for t, l in zip(ids, labels) if l == -100]
    print(tok.decode(masked)[:600])
