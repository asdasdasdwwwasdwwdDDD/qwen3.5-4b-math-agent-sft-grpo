# -*- coding: utf-8 -*-
"""train_grpo.make_sampler 离线测试：探索比例衰减、全错冷却、低放回上限、
mixed 池进出与统计口径。不需要 GPU。用法（仓库根运行）: python -m scripts.tests.test_sampler"""
import os
import sys
from collections import Counter

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.training.train_grpo import make_sampler

POOL_N = 4


def mkpool(n):
    return [{"id": f"math-test-{i}", "category": "MATH", "problem": f"题{i}", "gt": "1",
             "level": "Level 3", "type": ""} for i in range(n)]


def mkdiff_half_mixed():
    """前 2 题 mixed（[1,4]），后 2 题全错（[0,4]）。"""
    return {f"math-test-{i}": ([1, 4] if i < 2 else [0, 4]) for i in range(POOL_N)}


def main():
    pool = mkpool(POOL_N)
    diff = mkdiff_half_mixed()
    all_ids = {f"math-test-{i}" for i in range(POOL_N)}
    mixed_ids = {"math-test-0", "math-test-1"}

    # ---- 场景1：explore_ratio=0 → 只从 mixed 池抽 ----
    draw, update, dump, n_mixed = make_sampler(pool, diff, explore_ratio=0.0, seed=0)
    assert n_mixed() == 2
    ids = {draw()["id"] for _ in range(200)}
    assert ids == mixed_ids, ids
    print("场景1（explore=0 只抽 mixed 池）OK")

    # ---- 场景2：explore_ratio=1 → 全池抽（含非 mixed）----
    draw, update, dump, _ = make_sampler(pool, diff, explore_ratio=1.0, seed=0)
    ids = {draw()["id"] for _ in range(500)}
    assert ids == all_ids, ids
    print("场景2（explore=1 全池抽）OK")

    # ---- 场景3：探索比例线性衰减 1→0（horizon=100）----
    draw, update, dump, _ = make_sampler(pool, diff, explore_ratio=1.0, seed=0,
                                         explore_end=0.0, explore_horizon=100)
    early = {draw()["id"] for _ in range(30)}     # 前期 ratio≈1 → 全池
    for _ in range(80):                            # 推进抽签数越过 horizon
        draw()
    late = {draw()["id"] for _ in range(100)}     # 后期 ratio=0 → 只 mixed
    assert early - mixed_ids, "前期应能抽到非 mixed 题"
    assert late == mixed_ids, late
    print("场景3（探索比例 1→0 线性衰减）OK")

    # ---- 场景4：全错冷却 —— 冷却期内抽不到，期满解冻 ----
    draw, update, dump, _ = make_sampler(pool, diff, explore_ratio=1.0, seed=0,
                                         cooldown_draws=5)
    update(pool[0], 0, 4)                          # 全错 → 冷却接下来 5 次抽签
    seen = [draw()["id"] for _ in range(5)]
    assert "math-test-0" not in seen, seen
    seen2 = {draw()["id"] for _ in range(80)}
    assert "math-test-0" in seen2, "解冻后应恢复可抽"
    print("场景4（全错冷却 5 次抽签后解冻）OK")

    # ---- 场景5：冷却中的题出现方差（进入 mixed）立即解冻 ----
    draw, update, dump, _ = make_sampler(pool, diff, explore_ratio=1.0, seed=0,
                                         cooldown_draws=100)
    update(pool[0], 0, 4)                          # 全错 → 长冷却
    update(pool[0], 2, 4)                          # 变 mixed → 立即解冻
    seen = {draw()["id"] for _ in range(30)}
    assert "math-test-0" in seen
    print("场景5（mixed 解冻）OK")

    # ---- 场景6：dump 统计口径 —— 只含采样过的题，增量正确（与抽中分布无关）----
    draw, update, dump, _ = make_sampler(pool, diff, explore_ratio=0.0, seed=0)
    for _ in range(3):
        update(draw(), 2, 4)
    d = dump()
    assert d["math-test-2"] == [0, 4], "未被抽到的题只保留种子计数"
    # 种子总量 4 题 × 4 次 + 3 次更新 × 4 条 = 28；正确数种子 1+1+0+0=2，更新 3×2=6 → 8
    assert sum(v[1] for v in d.values()) == 28, d
    assert sum(v[0] for v in d.values()) == 8, d
    print("场景6（dump 统计口径）OK")

    # ---- 场景7：默认参数（无衰减/无冷却）不触发新分支，可正常大量抽签 ----
    draw, update, dump, _ = make_sampler(pool, diff, explore_ratio=0.3, seed=0)
    for _ in range(300):
        update(draw(), 1, 4)
    print("场景7（默认参数与旧口径一致）OK")

    # ---- 场景8：低放回上限 —— 单题至多 2 次，8 次抽签恰好均匀覆盖 4 题 ----
    draw, update, dump, _ = make_sampler(pool, diff, explore_ratio=1.0, seed=0,
                                         max_task_draws=2)
    got = Counter(draw()["id"] for _ in range(8))
    assert set(got) == all_ids and set(got.values()) == {2}, got
    print("场景8（低放回上限=2 强制均匀覆盖）OK")

    # ---- 场景9：mixed 池被上限滤空 → 自动落到全池找没抽过的题 ----
    draw, update, dump, _ = make_sampler(pool, diff, explore_ratio=0.0, seed=0,
                                         max_task_draws=1)
    first_two = {draw()["id"] for _ in range(2)}
    assert first_two == mixed_ids, first_two          # 前两次抽完两个 mixed 题
    third = draw()["id"]
    assert third not in mixed_ids, third              # mixed 全部到限 → 全池兜底
    print("场景9（mixed 到限自动落全池）OK")

    print("\n全部场景通过")


if __name__ == "__main__":
    main()
