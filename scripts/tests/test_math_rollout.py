# -*- coding: utf-8 -*-
"""math_env.rollout 离线集成测试：用 mock 模型按剧本吐 assistant 文本，
验证多轮消息流（代码→受限执行→tool 回填→终答）、模板渲染与打分。不需要 GPU。
用法（仓库根运行）: python -m scripts.tests.test_math_rollout 或 python scripts/tests/test_math_rollout.py"""
import json
import os
import sys
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

# 兼容两种运行方式：`python -m` 之外直接以文件方式运行时 sys.path 里没有仓库根，
# 包导入会 ModuleNotFoundError；补进仓库根后两种方式都可用（前者该路径已存在，无副作用）
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.envs.math_env import (MathEnv, render_ids, rollout, rollout_group,
                                   score_trajectory)


class MockModel:
    """按剧本顺序返回 assistant 文本对应的 token 序列。"""
    device = "cpu"

    def __init__(self, tokenizer, scripts):
        self.tokenizer = tokenizer
        self.scripts = list(scripts)
        self.script_i = 0
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

    def generate(self, input_ids, max_new_tokens=512, **kw):
        text = self.scripts[min(self.script_i, len(self.scripts) - 1)]
        self.script_i += 1
        gen_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:max_new_tokens]
        return torch.tensor([input_ids[0].tolist() + gen_ids])


class MockGroupModel(MockModel):
    """批量首回合：按行各自取剧本生成，行尾补真实 eos（模拟 generate 的产出），
    短行用 eos（=训练时的 pad_token）填齐——rollout_group 必须按 eos 截断、
    丢掉 eos 之后的填充，否则 pad 会混进 gen_ids 污染重放。"""

    def generate(self, input_ids, max_new_tokens=512, num_return_sequences=1, **kw):
        rows = []
        for _ in range(num_return_sequences):
            text = self.scripts[min(self.script_i, len(self.scripts) - 1)]
            self.script_i += 1
            gen_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:max_new_tokens]
            rows.append(input_ids[0].tolist() + gen_ids + [self.tokenizer.eos_token_id])
        width = max(len(r) for r in rows)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        return torch.tensor([r + [pad] * (width - len(r)) for r in rows])


def main():
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3.5-4B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    task = {"id": "math-test-0", "category": "MATH-Test", "problem": "What is 6*7?",
            "gt": "42", "level": "Level 1", "type": "Test"}

    # ---- 场景 1：代码 → 执行 → 终答（正确）----
    env = MathEnv(task)
    model = MockModel(tokenizer, ["```python\nprint(6*7)\n```",
                            "The answer is $\\boxed{42}$."])
    trajectory = rollout(model, tokenizer, env, max_turns=4, max_new_tokens=512, temperature=1.0,
                         top_p=1.0, max_len=2048, do_sample=False, grad_ckpt=False)
    assert trajectory["ended"] == "final" and len(trajectory["turns"]) == 2, trajectory["ended"]
    assert [turn["kind"] for turn in trajectory["turns"]] == ["code", "final"]
    assert env.n_exec == 1 and env.n_error == 0
    assert trajectory["turns"][0]["exec_output"] and "42" in trajectory["turns"][0]["exec_output"]
    reward, detail = score_trajectory(trajectory, env)
    assert detail["correct"] and detail["success"] and detail["pred"] == "42", detail
    assert abs(reward - 1.0) < 1e-6, reward
    print("场景1（代码→执行→终答正确）OK reward=%.2f" % reward)

    # 消息渲染自检：assistant 两轮都带空 think、执行结果是 tool_response 包裹的 user 轮
    messages = [{"role": "user", "content": task["problem"]},
            {"role": "assistant", "content": trajectory["turns"][0]["text"]},
            {"role": "tool", "content": "```output\n42\n```"},
            {"role": "assistant", "content": trajectory["turns"][1]["text"]}]
    decoded = tokenizer.decode(render_ids(tokenizer, messages, None)[: len(render_ids(tokenizer, messages, None))])
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                   enable_thinking=False)
    assert full.count("<think>") == 2, full[:400]          # 两个 assistant 轮都在 last_query 之后
    assert "<tool_response>" in full and "```output" in full
    print("场景1 渲染自检 OK（双 think + tool_response）")

    # ---- 场景 2：报错 → 修正重试 → 终答 ----
    env = MathEnv(task)
    model = MockModel(tokenizer, ["```python\nx=1/0\n```",
                            "```python\nprint(6*7)\n```",
                            "It is $\\boxed{42}$."])
    trajectory = rollout(model, tokenizer, env, max_turns=4, max_new_tokens=512, temperature=1.0,
                         top_p=1.0, max_len=2048, do_sample=False, grad_ckpt=False)
    assert env.n_exec == 2 and env.n_error == 1, (env.n_exec, env.n_error)
    assert trajectory["ended"] == "final"
    reward, detail = score_trajectory(trajectory, env)
    assert detail["correct"], detail
    print("场景2（报错→修正→终答）OK reward=%.2f n_error=%d" % (reward, env.n_error))

    # ---- 场景 3：一直写代码不给终答（回合上限）----
    env = MathEnv(task)
    model = MockModel(tokenizer, ["```python\nprint(1)\n```"] * 6)
    trajectory = rollout(model, tokenizer, env, max_turns=3, max_new_tokens=512, temperature=1.0,
                         top_p=1.0, max_len=2048, do_sample=False, grad_ckpt=False)
    assert trajectory["ended"] == "limit" and env.n_exec == 2  # 3 回合上限：前 2 回合执行、第 3 回合收尾判定
    reward, detail = score_trajectory(trajectory, env)
    assert not detail["correct"] and detail["pred"] is None and detail["ended"] == "limit", detail
    # 新打分：无终答、无部分分，只剩"执行全程无报错"的 0.05
    assert abs(reward - 0.05) < 1e-6, reward
    print("场景3（回合上限未终答）OK reward=%.2f" % reward)

    # ---- 场景 4：代码算对但收尾写错（部分分生效）----
    env = MathEnv(task)
    model = MockModel(tokenizer, ["```python\nprint(6*7)\n```", "The answer is $\\boxed{41}$."])
    trajectory = rollout(model, tokenizer, env, max_turns=4, max_new_tokens=512, temperature=1.0,
                         top_p=1.0, max_len=2048, do_sample=False, grad_ckpt=False)
    reward, detail = score_trajectory(trajectory, env)
    # 0.15（执行输出命中 gt "42"）+ 0.10（有 boxed）+ 0.05（执行无报错）；正确性 0
    assert not detail["correct"] and detail["code_hit"] and detail["exec_clean"], detail
    assert abs(reward - 0.30) < 1e-6, (detail, reward)
    print("场景4（答案错误但代码算对）OK reward=%.2f（含 0.15 部分分）" % reward)

    # ---- 场景 5：rollout_group 组批量采样（首回合一次出 G 条 + eos 截断去 pad）----
    model = MockGroupModel(tokenizer, [
        "```python\nprint(6*7)\n```",        # 行0：代码 → 续写
        "```python\nprint(6*7)\n```",        # 行1：代码 → 续写
        "The answer is $\\boxed{42}$.",      # 行2：首回合直接终答（该轨迹 1 回合收工）
        "```python\nprint(6*7)\n```",        # 行3：代码 → 续写
        "The answer is $\\boxed{42}$.",      # 续写剧本 ×3（逐条分叉后各取一份）
        "The answer is $\\boxed{42}$.",
        "The answer is $\\boxed{42}$.",
    ])
    traj_envs = rollout_group(model, tokenizer, task, num_generations=4, gen_batch=4,
                           max_turns=4, max_new_tokens=512, temperature=1.0,
                           top_p=1.0, max_len=2048, do_sample=False, grad_ckpt=False)
    assert len(traj_envs) == 4 and len({id(env) for _, env in traj_envs}) == 4, len(traj_envs)  # G 条独立 env
    trajectory0, env0 = traj_envs[0]
    assert trajectory0["ended"] == "final" and [turn["kind"] for turn in trajectory0["turns"]] == ["code", "final"]
    assert env0.n_exec == 1 and "42" in trajectory0["turns"][0]["exec_output"]
    trajectory2, env2 = traj_envs[2]
    assert trajectory2["ended"] == "final" and len(trajectory2["turns"]) == 1 and env2.n_exec == 0
    eos_id = tokenizer.eos_token_id
    for trajectory, _env in traj_envs:   # 首回合 gen_ids 必须按 eos 截断：恰好 1 个 eos 在末尾，pad 不残留
        first_gen = trajectory["turns"][0]["gen_ids"]
        assert first_gen[-1] == eos_id and first_gen.count(eos_id) == 1, first_gen[-5:]
    print("场景5（rollout_group：首回合批量出 G 条、首回合终答不再续写、eos 截断去 pad）OK")

    # ---- 场景 6：rollout_group 分块批量（gen_batch=2 → 两次批量调用）----
    # 剧本按实际消耗顺序排：块1首回合×2 → 块1续写×2 → 块2首回合×2 → 块2续写×2
    # （分块语义：第 1 块轨迹完整续写完，才轮到第 2 块的首回合生成）
    model = MockGroupModel(tokenizer, ["```python\nprint(6*7)\n```"] * 2
                           + ["The answer is $\\boxed{42}$."] * 2
                           + ["```python\nprint(6*7)\n```"] * 2
                           + ["The answer is $\\boxed{42}$."] * 2)
    traj_envs = rollout_group(model, tokenizer, task, num_generations=4, gen_batch=2,
                           max_turns=4, max_new_tokens=512, temperature=1.0,
                           top_p=1.0, max_len=2048, do_sample=False, grad_ckpt=False)
    assert len(traj_envs) == 4, len(traj_envs)
    for trajectory, env in traj_envs:
        assert trajectory["ended"] == "final" and [turn["kind"] for turn in trajectory["turns"]] == ["code", "final"]
        assert env.n_exec == 1 and "42" in trajectory["turns"][0]["exec_output"]
    assert model.script_i == 8  # 2 次批量首回合（各 2 行）+ 4 次逐条续写 = 8 份剧本
    print("场景6（rollout_group 分块 gen_batch=2）OK")

    print("ALL PASS")


if __name__ == "__main__":
    main()
