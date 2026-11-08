# -*- coding: utf-8 -*-
"""打印所有 checkpoint 的 eval_loss 曲线，辅助挑选最佳 epoch"""
import glob
import json
import sys

for p in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1
                          else "outputs/qlora-adapter/checkpoint-*/trainer_state.json")):
    st = json.load(open(p, encoding="utf-8"))
    ev = [h for h in st["log_history"] if "eval_loss" in h]
    curve = ", ".join(f"e{h['epoch']:.0f}:{h['eval_loss']:.3f}" for h in ev)
    print(f"{p.split('/')[-2]:16s} step {st['global_step']:3d} | {curve}")
