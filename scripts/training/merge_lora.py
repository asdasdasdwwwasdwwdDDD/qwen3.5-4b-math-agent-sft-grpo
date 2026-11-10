# -*- coding: utf-8 -*-
"""
将训练好的 LoRA adapter 合并回底座模型，输出可独立部署的完整模型
用法:
  python -m scripts.training.merge_lora --model models/Qwen3.5-4B \
      --adapter outputs/qlora-tora/checkpoint-2000 --out outputs/merged-tora
"""
import argparse
import os

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/Qwen3.5-4B")
    ap.add_argument("--adapter", type=str, default="outputs/qlora-tora/checkpoint-2000")
    ap.add_argument("--out", type=str, default="outputs/merged-tora")
    args = ap.parse_args()

    print("加载底座模型（bf16, CPU）...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    print("加载并合并 LoRA adapter ...")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    print(f"保存合并后的模型到 {args.out} ...")
    model.save_pretrained(args.out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.save_pretrained(args.out)
    print("合并完成。")


if __name__ == "__main__":
    main()
