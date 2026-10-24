# -*- coding: utf-8 -*-
"""分析 ToRA 轨迹：token 长度分布（渲染后）、拆分结构统计。仅用于数据转换前的摸底。"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer

SRC = r"D:/NLP/ft-qwen3.5-4b-agent-cot-data/data/tora/tora_sft_2k_stratified.jsonl"
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.5-4B")

# 与 agent 轨迹一致的渲染参数：无工具定义、非思考模式
TMPL_KW = {"enable_thinking": False}


def split_content(content):
    """把 ToRA 内联轨迹拆成 (代码段, 输出段, 收尾文本)；不匹配则返回 None。"""
    pat = re.compile(r"```python\n(.*?)```\s*```output\n(.*?)```", re.S)
    m = pat.search(content)
    if not m:
        return None
    return m.group(1), m.group(2), content[m.end():]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    lens, bad = [], 0
    for r in rows:
        code, out, tail = split_content(r["messages"][1]["content"]) or (None, None, None)
        if code is None:
            bad += 1
            continue
        msgs = [
            {"role": "user", "content": r["messages"][0]["content"]},
            {"role": "assistant", "content": "```python\n" + code + "\n```"},
            {"role": "user", "content": "```output\n" + out + "\n```"},
            {"role": "assistant", "content": tail},
        ]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                      return_dict=False, **TMPL_KW)
        lens.append(len(ids))
    lens.sort()
    n = len(lens)
    print(f"可拆分 {n} / {len(rows)}（失败 {bad}）")
    print(f"token 长度: min={lens[0]} p50={lens[n//2]} p90={lens[int(n*0.9)]} "
          f"p95={lens[int(n*0.95)]} p99={lens[int(n*0.99)]} max={lens[-1]}")
    for cap in (1024, 1536, 1792, 2048, 2560):
        keep = sum(1 for x in lens if x <= cap)
        print(f"  max_len={cap}: 覆盖 {keep}/{n} = {keep/n:.1%}")


if __name__ == "__main__":
    main()
