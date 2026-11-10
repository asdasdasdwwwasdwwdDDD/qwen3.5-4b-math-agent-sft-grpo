# -*- coding: utf-8 -*-
"""打印所有 checkpoint 的 eval_loss 曲线，辅助挑选最佳 epoch"""
import glob
import json
import os
import sys

for p in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1
                          else "outputs/qlora-tora/checkpoint-*/trainer_state.json")):
    st = json.load(open(p, encoding="utf-8"))
    ev = [h for h in st["log_history"] if "eval_loss" in h]
    curve = ", ".join(f"e{h['epoch']:.0f}:{h['eval_loss']:.3f}" for h in ev)
    # normpath 统一分隔符：Windows 下 glob 返回反斜杠路径，直接 split('/') 会取错段
    name = os.path.normpath(p).split(os.sep)[-2]
    print(f"{name:16s} step {st['global_step']:3d} | {curve}")
