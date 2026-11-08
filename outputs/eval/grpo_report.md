# GRPO 对齐 · 前后对比报告

## 1. GRPO 训练过程（整条链路 trajectory 模式）

- 步数: 60 | 平均 reward 前12步 0.584 → 后12步 0.646
- 端到端成功率: 前12步 49% → 后12步 56% | 平均回合 2.2
- KL(对 SFT 参考模型): 均值 0.0013 / 最大 0.0044（beta 锚定，偏离极小 → 通用能力不被破坏）
- 显存峰值 6.75GB | 未出现 OOM/重启

## 2. 整条链路明细样例（组内比较让模型自我优化）

- 任务 `math-test-4810`（MATH-Level 5），G=4 条轨迹:

  轨迹0: reward=0.20 adv=+0.58 | 成功=False 预测=x - 2y + z = 0 真值=20x + 23y + 26z - 69 = 0 回合=2 执行 1 次（报错 0）
    [题] Let line $L$ be the intersection of the planes $x + y + z - 6 = 0$ and $2x + 3y + 4z + 5 = 0.$  Find the equation of
    [代码] from sympy import symbols, Eq, solve
    
    def equation_of_plane():
        """Let line $L$ be the intersection of the planes $x + y + z - 6 = 0$ and $2x + 3y + 4z + 5 = 0.$  Find the equation
      ⇐ x - 2*y + z

  轨迹1: reward=0.10 adv=-1.73 | 成功=False 预测=None 真值=20x + 23y + 26z - 69 = 0 回合=3 执行 3 次（报错 3）
    [题] Let line $L$ be the intersection of the planes $x + y + z - 6 = 0$ and $2x + 3y + 4z + 5 = 0.$  Find the equation of
    [代码] from sympy import symbols, Eq, solve
    
    def find_plane_equation():
        """Let line $L$ be the intersection of the planes $x + y + z - 6 = 0$ and $2x + 3y + 4z + 5 = 0.$  Find the equation
      ⇐ C:\Users\dyc\AppData\Local\Temp\tora_sbx_90xoey6c\solution.py:5: SyntaxWarning: invalid escape sequence '\['

  轨迹2: reward=0.20 adv=+0.58 | 成功=False 预测=3x + 2y + z - 6 = 0 真值=20x + 23y + 26z - 69 = 0 回合=2 执行 1 次（报错 0）
    [题] Let line $L$ be the intersection of the planes $x + y + z - 6 = 0$ and $2x + 3y + 4z + 5 = 0.$  Find the equation of
    [代码] from sympy import symbols, Eq, simplify
    from sympy.solvers.inequalities import solve_univariate_inequality
    from sympy.calculus.util import continuous_domain
    
    def plane_containing_line():

  轨迹3: reward=0.20 adv=+0.58 | 成功=False 预测=3x + 4y + 5z - 1 = 0 真值=20x + 23y + 26z - 69 = 0 回合=2 执行 1 次（报错 0）
    [题] Let line $L$ be the intersection of the planes $x + y + z - 6 = 0$ and $2x + 3y + 4z + 5 = 0.$  Find the equation of
    [代码] from sympy import symbols, Eq, simplify
    
    def plane_equation():
        """Let line $L$ be the intersection of the planes $x + y + z - 6 = 0$ and $2x + 3y + 4z + 5 = 0.$  Find the equation
      ⇐ 3*x + 4*y + 5*z - 1

## 3. 闭环评测对比（同一持出集，boxed 正确率）

| 指标 | math_grpo | math_sel-base | math_sel-ck1500 | math_sel-ck2000 | math_sel-ck2500 | math_sft |
|---|---|---|---|---|---|---|
| 总正确率 | 44.0% | 0.0% | 66.7% | 73.3% | 66.7% | 37.0% |
| 平均回合 | 2.07 | 1.00 | 2.00 | 2.03 | 1.97 | 2.26 |
| 执行报错率 | 24.3% | 0.0% | 3.3% | 9.7% | 3.4% | 37.7% |
| 未终答率 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 6.0% |

分难度层正确率:

| 难度层 | math_grpo | math_sel-base | math_sel-ck1500 | math_sel-ck2000 | math_sel-ck2500 | math_sft |
|---|---|---|---|---|---|---|
| Level 1 | 90.9% | 0.0% | 90.9% | 90.9% | 90.9% | 90.9% |
| Level 2 | 50.0% | 0.0% | 52.6% | 63.2% | 52.6% | 55.0% |
| Level 3 | 66.7% | - | - | - | - | 57.1% |
| Level 4 | 18.2% | - | - | - | - | 4.5% |
| Level 5 | 23.1% | - | - | - | - | 11.5% |

