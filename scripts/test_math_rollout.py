# -*- coding: utf-8 -*-
"""math_env.rollout 离线集成测试：用 mock 模型按剧本吐 assistant 文本，
验证多轮消息流（代码→受限执行→tool 回填→终答）、模板渲染与打分。不需要 GPU。"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoTokenizer

from agent_env import render_ids
from math_env import MathEnv, rollout, score_trajectory


class MockModel:
    """按剧本顺序返回 assistant 文本对应的 token 序列。"""
    device = "cpu"

    def __init__(self, tok, scripts):
        self.tok = tok
        self.scripts = list(scripts)
        self.i = 0
        self.training = False
        self.config = SimpleNamespace(use_cache=False)

    def eval(self):
        return self

    def train(self):
        return self

    def gradient_checkpointing_disable(self):
        pass

    def gradient_checkpointing_enable(self, **kw):
        pass

    def generate(self, enc, max_new_tokens=512, **kw):
        text = self.scripts[min(self.i, len(self.scripts) - 1)]
        self.i += 1
        ids = self.tok(text, add_special_tokens=False)["input_ids"][:max_new_tokens]
        return torch.tensor([enc[0].tolist() + ids])


def main():
    tok = AutoTokenizer.from_pretrained("models/Qwen3.5-4B", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    task = {"id": "math-test-0", "category": "MATH-Test", "problem": "What is 6*7?",
            "gt": "42", "level": "Level 1", "type": "Test"}

    # ---- 场景 1：代码 → 执行 → 终答（正确）----
    env = MathEnv(task)
    model = MockModel(tok, ["```python\nprint(6*7)\n```",
                            "The answer is $\\boxed{42}$."])
    ro = rollout(model, tok, env, max_turns=4, max_new_tokens=512, temperature=1.0,
                 top_p=1.0, max_len=2048, do_sample=False, grad_ckpt=False)
    assert ro["ended"] == "final" and len(ro["turns"]) == 2, ro["ended"]
    assert [t["kind"] for t in ro["turns"]] == ["code", "final"]
    assert env.n_exec == 1 and env.n_error == 0
    assert ro["turns"][0]["exec_output"] and "42" in ro["turns"][0]["exec_output"]
    r, det = score_trajectory(ro, env)
    assert det["correct"] and det["success"] and det["pred"] == "42", det
    assert abs(r - 1.0) < 1e-6, r
    print("场景1（代码→执行→终答正确）OK reward=%.2f" % r)

    # 消息渲染自检：assistant 两轮都带空 think、执行结果是 tool_response 包裹的 user 轮
    msgs = [{"role": "user", "content": task["problem"]},
            {"role": "assistant", "content": ro["turns"][0]["text"]},
            {"role": "tool", "content": "```output\n42\n```"},
            {"role": "assistant", "content": ro["turns"][1]["text"]}]
    s = tok.decode(render_ids(tok, msgs, None)[: len(render_ids(tok, msgs, None))])
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                   enable_thinking=False)
    assert full.count("<think>") == 2, full[:400]          # 两个 assistant 轮都在 last_query 之后
    assert "<tool_response>" in full and "```output" in full
    print("场景1 渲染自检 OK（双 think + tool_response）")

    # ---- 场景 2：报错 → 修正重试 → 终答 ----
    env = MathEnv(task)
    model = MockModel(tok, ["```python\nx=1/0\n```",
                            "```python\nprint(6*7)\n```",
                            "It is $\\boxed{42}$."])
    ro = rollout(model, tok, env, max_turns=4, max_new_tokens=512, temperature=1.0,
                 top_p=1.0, max_len=2048, do_sample=False, grad_ckpt=False)
    assert env.n_exec == 2 and env.n_error == 1, (env.n_exec, env.n_error)
    assert ro["ended"] == "final"
    r, det = score_trajectory(ro, env)
    assert det["correct"], det
    print("场景2（报错→修正→终答）OK reward=%.2f n_error=%d" % (r, env.n_error))

    # ---- 场景 3：一直写代码不给终答（回合上限）----
    env = MathEnv(task)
    model = MockModel(tok, ["```python\nprint(1)\n```"] * 6)
    ro = rollout(model, tok, env, max_turns=3, max_new_tokens=512, temperature=1.0,
                 top_p=1.0, max_len=2048, do_sample=False, grad_ckpt=False)
    assert ro["ended"] == "limit" and env.n_exec == 2  # 3 回合上限：前 2 回合执行、第 3 回合收尾判定
    r, det = score_trajectory(ro, env)
    assert not det["correct"] and det["pred"] is None and det["ended"] == "limit", det
    assert abs(r - 0.1) < 1e-6, r  # 只拿"用过代码"的 0.1
    print("场景3（回合上限未终答）OK reward=%.2f" % r)

    # ---- 场景 4：答案错误 ----
    env = MathEnv(task)
    model = MockModel(tok, ["```python\nprint(6*7)\n```", "The answer is $\\boxed{41}$."])
    ro = rollout(model, tok, env, max_turns=4, max_new_tokens=512, temperature=1.0,
                 top_p=1.0, max_len=2048, do_sample=False, grad_ckpt=False)
    r, det = score_trajectory(ro, env)
    assert not det["correct"] and abs(r - 0.2) < 1e-6, (det, r)  # boxed 有 + 代码有，答案错
    print("场景4（答案错误）OK reward=%.2f" % r)

    print("ALL PASS")


if __name__ == "__main__":
    main()
