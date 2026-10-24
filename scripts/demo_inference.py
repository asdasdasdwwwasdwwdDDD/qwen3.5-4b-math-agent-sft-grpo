# -*- coding: utf-8 -*-
"""微调后模型对话演示：加载底座 + LoRA adapter，完整智能体闭环（多轮工具调用）

与训练/评估共用 agent_env 的工具沙箱：确定性 mock（同一调用永远返回同一结果，
run_python 出于安全一律拒绝执行）。工具调用按 Qwen3.5 原生 XML 格式解析
（<tool_call><function=名><parameter=参数>值</parameter></function></tool_call>）。
"""
import argparse
import os
import sys

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_env import generate_once, render_ids, sandbox_exec  # noqa: E402
from build_dataset import TOOL_DEFS  # noqa: E402
from rewards import parse_all_calls  # noqa: E402

SYS = "你是一个乐于助人的智能助手，可以调用工具来回答问题。"
# 演示暴露全部工具（run_python 在沙箱中禁用，调用会返回 sandbox_disabled）
DEMO_TOOLS = list(TOOL_DEFS.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/Qwen3.5-4B")
    ap.add_argument("--adapter", type=str, default=None, help="不传则用底座模型")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=6)
    args = ap.parse_args()

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=True)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"已加载 adapter: {args.adapter}")
    model.eval()

    msgs = [{"role": "system", "content": SYS}]
    print("输入你的问题（直接回车退出）。工具为确定性 mock 沙箱（run_python 已禁用）。")
    while True:
        try:
            user = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            break
        msgs.append({"role": "user", "content": user})

        for _ in range(args.max_turns):
            ids = render_ids(tok, msgs, DEMO_TOOLS)
            enc = torch.tensor([ids], device=model.device)
            out = generate_once(model, tok, enc, args.max_new_tokens, 1.0, 1.0,
                                do_sample=False, grad_ckpt=False)
            text = tok.decode(out[0][len(ids):], skip_special_tokens=False)
            text = text.split("<|im_end|>")[0].strip()
            calls = [(n, a) for n, a, ok in parse_all_calls(text) if ok]
            if not calls:
                print(f"助手: {text}")
                msgs.append({"role": "assistant", "content": text})
                break
            # 有工具调用：展示、真执行（沙箱）、把返回喂回去继续
            print(f"助手(工具调用): {text}")
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function",
                                         "function": {"name": n, "arguments": a}}
                                        for n, a in calls]})
            for n, a in calls:
                result = sandbox_exec(n, a)
                print(f"[工具返回] {result}")
                msgs.append({"role": "tool", "name": n, "content": result})
        else:
            print("（达到最大回合数，结束本轮）")


if __name__ == "__main__":
    main()
