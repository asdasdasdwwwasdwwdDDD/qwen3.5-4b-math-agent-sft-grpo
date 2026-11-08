# -*- coding: utf-8 -*-
"""从 ModelScope 下载 Qwen3.5-4B 底座模型到本地 models/ 目录

注：本项目当前使用的模型已就位于 models/Qwen3.5-4B（指向 D:\\NLP\\Qwen3.5-4 B 的目录联接），
本脚本仅用于在新机器上复现底座下载。
"""
import argparse
import os

from modelscope import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", type=str, default="models")
    args = ap.parse_args()

    path = snapshot_download(args.model, local_dir=os.path.join(args.out, args.model.split("/")[-1]))
    print(f"模型已下载到: {path}")


if __name__ == "__main__":
    main()
