# -*- coding: utf-8 -*-
"""train_grpo.train_step_update 离线测试：用可控 logits 的 mock 策略验证
μ（--epochs）重放语义、old_logp/ref_logp 缓存行为、梯度累积时机与 PPO 裁剪触发。不需要 GPU。

覆盖场景：
  1) μ=1（封存口径）：每回合 3 次前向（ref_logp/old_logp/act_logp）、ratio 恒 1 → clip_tokens=0、
     W=W0 时 kl=0 → loss=-adv 的解析值逐位吻合（默认路径与封存代码逐位一致的证据）；
  2) μ=4：前向计数 = 首遍 3/回合 + 后续遍 1/回合（old_logp/ref_logp 缓存复用），
     opt.step 按 grad_accum 跨 epoch 连续计数触发；
  3) μ=4 + 大学习率：权重移动后 ratio 偏离 1，clip_hits>0（裁剪真正上岗）；
  4) 跳过过滤：adv==0（组内全同分）与零 token 轨迹整条跳过，backward 尺度仍按全部轨迹。
用法（仓库根运行）: python -m scripts.tests.test_grpo_update
（或直接以文件方式运行：本文件已把仓库根补进 sys.path）
"""
import contextlib
import os
import sys
from types import SimpleNamespace

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.training.train_grpo import train_step_update


class MockPolicy(torch.nn.Module):
    """可控 logits 的最小策略：logits[b, pos, v] = W[input_ids[b, pos], v]。

    adapter 挂载时用可训练的 W（带梯度），disable_adapter() 期间用冻结的 W0
    （=SFT 参考模型）——镜像真模型 PeftModel.disable_adapter 的语义。"""

    def __init__(self, vocab=64):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.W0 = torch.randn(vocab, vocab, generator=g) * 0.1
        self.W = torch.nn.Parameter(self.W0.clone())   # 起点 W=W0：kl=0、ratio=1
        self.calls = 0
        self._adapter_off = False
        self.device = torch.device("cpu")

    @contextlib.contextmanager
    def disable_adapter(self):
        self._adapter_off = True
        try:
            yield
        finally:
            self._adapter_off = False

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0):
        self.calls += 1
        W = self.W0 if self._adapter_off else self.W
        logits = W[input_ids[0]].unsqueeze(0)[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


class CountingOpt:
    """包一层 opt.step 计数，断言优化步的触发时机。"""

    def __init__(self, opt):
        self.opt = opt
        self.n_step = 0

    def step(self):
        self.n_step += 1
        self.opt.step()

    def zero_grad(self, set_to_none=True):
        self.opt.zero_grad(set_to_none=set_to_none)


def make_rollout(turn_lens):
    """假轨迹：每回合 prompt_ids=5 个、gen_ids=turn_lens[i] 个互不相同的合法 id。"""
    turns, next_id = [], 1
    for turn_len in turn_lens:
        prompt_ids = list(range(next_id, next_id + 5))
        next_id += 5
        gen_ids = list(range(next_id, next_id + turn_len))
        next_id += turn_len
        turns.append({"prompt_ids": prompt_ids, "gen_ids": gen_ids,
                      "kind": "code", "text": "", "code": "x", "exec_output": ""})
    return {"turns": turns, "ended": "final"}


def make_args(epochs, grad_accum, turn_exec_w=0.0):
    return SimpleNamespace(epochs=epochs, grad_accum=grad_accum, epsilon=0.2,
                           beta=0.025, kl_clip=0.05, max_grad_norm=1e9,
                           turn_exec_w=turn_exec_w)


def main():
    # ---- 场景 1：μ=1（封存口径）----
    model = MockPolicy()
    rollouts = [(SimpleNamespace(), make_rollout([4, 3]), 1.0, 1.0, SimpleNamespace())]
    opt = CountingOpt(torch.optim.SGD([model.W], lr=0.0))
    stats = train_step_update(model, rollouts, [model.W], opt, make_args(1, 1))
    assert model.calls == 6, model.calls          # 每回合 ref_logp/old_logp/act_logp 共 3 次前向 × 2 回合
    assert opt.n_step == 1, opt.n_step            # accum=1：backward 后立即 opt.step
    assert stats["n_trained"] == 1 and stats["n_turn"] == 2, stats
    assert stats["clip_hits"] == 0, stats         # μ=1：ratio 恒 1，裁剪永不触发
    # W=W0 → log_ratio=0 → kl=0；ratio=1 → policy_loss=-adv；turn_loss 按token数加权后 loss=-adv（解析值）
    assert abs(stats["tot_loss"] + 1.0) < 1e-5, stats
    assert stats["kl_sum"] < 1e-6 and abs(stats["pol_sum"] + 2.0) < 1e-5, stats
    # （不在此处断言 grad：每个 opt.step 后 zero_grad(set_to_none=True) 会把 grad 置 None，
    #  梯度真实流通由场景 3 的权重移动证明）
    print("场景1（μ=1：3 前向/回合、ratio 恒 1、loss=-adv 解析值吻合）OK")

    # ---- 场景 2：μ=4 前向计数与优化步时机（lr=0 权重不动，ratio 仍恒 1）----
    model = MockPolicy()
    rollouts = [(SimpleNamespace(), make_rollout([4, 3]), 0.5, 1.0, SimpleNamespace()),
                (SimpleNamespace(), make_rollout([5, 2]), -0.5, 0.2, SimpleNamespace())]
    opt = CountingOpt(torch.optim.SGD([model.W], lr=0.0))
    stats = train_step_update(model, rollouts, [model.W], opt, make_args(4, 4))
    # 前向：每轨迹每回合 = 首遍 3（ref_logp/old_logp/act_logp）+ 后 3 遍各 1（缓存复用，仅 act_logp）= 6
    # 2 轨迹 × 2 回合 × 6 = 24
    assert model.calls == 24, model.calls
    assert opt.n_step == 2, opt.n_step            # 8 次 backward，每 4 次一个 opt.step
    assert stats["n_trained"] == 8 and stats["n_turn"] == 16, stats
    assert stats["clip_hits"] == 0, stats         # lr=0：权重不动，缓存 old_logp 也不产生偏移
    print("场景2（μ=4：首遍 3/回合 + 后续 1/回合，opt.step 跨 epoch 连续计数）OK")

    # ---- 场景 3：μ=4 + 大学习率 → ratio≠1，裁剪真正触发 ----
    model = MockPolicy()
    w_before = model.W.detach().clone()
    rollouts = [(SimpleNamespace(), make_rollout([6]), 2.0, 1.0, SimpleNamespace())]
    opt = CountingOpt(torch.optim.SGD([model.W], lr=100.0))
    stats = train_step_update(model, rollouts, [model.W], opt, make_args(4, 1))
    assert not torch.allclose(w_before, model.W.detach()), "权重未移动：梯度没有流回 W"
    assert opt.n_step == 4, opt.n_step            # accum=1：每遍 backward 后立即 step
    assert stats["clip_hits"] > 0, stats          # 权重大幅移动 → ratio 冲出信任带
    assert abs(stats["tot_loss"] + 8.0) > 0.1, stats  # 第 2 遍起 loss ≠ 纯 -adv（ratio≠1）
    print("场景3（μ=4 大步长：ratio 偏离 1、clip_tokens>0，裁剪上岗）OK")

    # ---- 场景 4：跳过过滤（adv==0 / 零 token），尺度基准仍含被跳过轨迹 ----
    model = MockPolicy()
    rollouts = [(SimpleNamespace(), make_rollout([4]), 1.0, 1.0, SimpleNamespace()),
                (SimpleNamespace(), make_rollout([4]), 0.0, 0.5, SimpleNamespace()),
                (SimpleNamespace(), {"turns": [], "ended": "limit"}, 1.5, 0.3,
                 SimpleNamespace())]
    opt = CountingOpt(torch.optim.SGD([model.W], lr=0.0))
    stats = train_step_update(model, rollouts, [model.W], opt, make_args(1, 8))
    assert model.calls == 3, model.calls          # 只有第一条轨迹被重放（ref_logp/old_logp/act_logp）
    assert stats["n_trained"] == 1, stats
    assert opt.n_step == 1, opt.n_step            # pending_backwards=1 < accum=8 → 循环末残留 flush
    print("场景4（adv==0 与零 token 轨迹整条跳过）OK")

    # ---- 场景 5：分轮次优势（turn_exec_w）——代码轮按执行成败局部加减，终答轮不受影响 ----
    model = MockPolicy()
    trajectory = make_rollout([6, 5])
    trajectory["turns"][0]["exec_ok"] = False      # 第 1 轮代码报错
    trajectory["turns"][1]["exec_ok"] = None       # 终答轮不执行，None = 不加局部信号
    rollouts = [(SimpleNamespace(), trajectory, 1.0, 1.0, SimpleNamespace())]
    opt = CountingOpt(torch.optim.SGD([model.W], lr=0.0))
    stats = train_step_update(model, rollouts, [model.W], opt, make_args(1, 1, turn_exec_w=0.5))
    # adv=1.0；代码轮 1.0-0.5=0.5、终答轮 1.0，按 token 数加权：(0.5*6 + 1.0*5)/11
    expected = -(0.5 * 6 + 1.0 * 5) / 11
    assert abs(stats["tot_loss"] - expected) < 1e-5, (stats["tot_loss"], expected)
    print("场景5（分轮次优势：报错代码轮 adv 1.0→0.5，终答轮不变）OK")

    # 同一轨迹 turn_exec_w=0 → 与旧行为一致（loss=-adv）
    model = MockPolicy()
    opt = CountingOpt(torch.optim.SGD([model.W], lr=0.0))
    stats = train_step_update(model, rollouts, [model.W], opt, make_args(1, 1, turn_exec_w=0.0))
    assert abs(stats["tot_loss"] + 1.0) < 1e-5, stats
    print("场景5b（turn_exec_w=0 与旧行为逐位一致）OK")

    print("ALL PASS")


if __name__ == "__main__":
    main()
