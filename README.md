# Qwen3.5-4B 数学智能体微调：QLoRA(AdaLoRA) SFT + 整条链路 GRPO

用 ToRA 数学推理轨迹对 **Qwen3.5-4B** 做两阶段对齐：先 SFT 学会「写 Python → 看执行结果 → \boxed 作答」的解题链路，再用**整条链路 GRPO**（模型自己走完整条链路，代码经受限子进程**真实执行**，按整条轨迹的答案正确性做组内验证对齐）继续提升。全部流程在消费级 **8GB 显存笔记本 GPU** 上完成，自研训练循环（不依赖 TRL）。

本项目是 2026-10-24 的阶段封存快照：100 题持出集上 base ~0% → **SFT 37.0%** → **GRPO 44.0%**。2026-11 用 probe 任务池 + 难度感知采样器重训 **GRPO v2** 至 200 步（见下文"v2 重训"）。

## 成果（100 题持出集，4bit 部署口径）

| 模型 | 总正确率 | L1 | L2 | L3 | L4 | L5 | 执行报错率 | 未终答 |
|---|---|---|---|---|---|---|---|---|
| base Qwen3.5-4B | ~0% | - | - | - | - | - | - | - |
| SFT (ckpt-2000) | 37.0% | 90.9% | 55.0% | 57.1% | 4.5% | 11.5% | 37.7% | 6% |
| GRPO (best=step-20) | **44.0%** | 90.9% | 50.0% | 66.7% | 18.2% | 23.1% | **24.3%** | **0%** |

GRPO 的收益集中在高难度层（L4 4.5%→18.2%、L5 11.5%→23.1%）与行为质量（报错率 −13.4pp、未终答清零）。
完整评测汇总见 [`outputs/eval/math_summary.md`](outputs/eval/math_summary.md)。

## v2 重训（2026-11，200 步）

从 merged-tora 出发的第二轮 GRPO：**probe 任务池**（2914 题摸底筛 p∈[0.05,0.95]，只训有梯度信号的中段题）+ **难度感知采样器**（按每题成功率估计加权抽题，冷却 40 步、同题≤3 次、全零组重抽，难度表每步回写）+ G=8 深组。P=2×G=8，lr 1e-5 cosine，KL β=0.05，T=0.8，reward = 0.8×正确 + 0.1×boxed + 0.1×执行无误，约 9.5 分钟/步，峰值显存 7.03GB。

- **训练池 success 36%→52%**（20 步窗口），执行报错率 26.8%→23.3%
- **训练中 40 题固定尺子**（每 20 步 greedy）：42.5 → 67.5 → 60 → **75.0（best@80）** → 后段 62.5–70 平台，final@200 = 67.5
- 全程 KL 累计 0.41，无跑飞；偷懒审计干净（做对题里"真算"91%，无跳代码/刷格式分倾向）
- 工程配套：续训三件套（`--resume` + `--init-step` + `--init-sched-steps`）保证 TB 曲线三段拼成同一条；本地看门狗 `scripts/tools/train_watchdog.py` 每 5 分钟自愈，不占模型额度
- **新鲜 80 题终评**（eval_v2，与训练池零重叠）：**best@80 = 67.5%、final@200 = 66.2%，vs base 45.0%（+22.5pp）**——泛化提升坐实；执行报错率 45.0%→15.9%，未终答 7.5%→0%。训练中尺子的 0.750 在新鲜尺子上读数 67.5%，赢者诅咒被挤出但提升真实

## 方案

- **SFT**：官方 ToRA 轨迹（题目 → ```python → 真实执行结果 → \boxed 终答）1901 条，QLoRA(AdaLoRA init_r=16→8) + NF4 4bit + 分块交叉熵，assistant 区间 loss mask（公共前缀对齐算法）；6000 步计划，按 eval_loss 谷底 + 30 题生成评测双指标选定 checkpoint-2000。
- **GRPO**：从 merged 模型出发，每任务采样 G=4 条完整轨迹成组，reward = 0.8×答案正确 + 0.1×boxed 格式 + 0.1×执行无误，组内归一化优势 + PPO 裁剪 + KL 锚定（参考策略 = 关掉 LoRA，零额外显存）；60 步课程化（难度层轮转交织），训练中每 20 步 greedy 闭环评估存 best。`--epochs`（μ）控制同批数据重放遍数：默认 1=封存口径（ratio 恒 1，裁剪为保险丝），`--epochs 4` 即 PPO μ=4（第 2 遍起 ratio 偏离 1、裁剪真正工作，每训练步移动量 ≈×4）；`--gen-batch` 支持首回合组内批量采样（同题 prompt 相同，提效不加批）。
- **环境与评测同一把尺子**：`scripts/envs/math_env.py` 同时被训练与评测调用（受限执行：隔离子进程 `-I` + 临时目录 + 超时 20s + 输出截断；执行报错文本回填给模型，鼓励「报错→读 Traceback→修正」）。

## 硬件与训练配置

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 5070 Laptop 8GB（8151 MiB 可分配） |
| 操作系统 | Windows 11（WDDM 显存模型，见下文"显存红线"） |
| 底座模型 | Qwen3.5-4B（VL 架构，仅用文本部分） |
| 量化 | bitsandbytes NF4 + 双重量化，计算精度 bf16 |
| 优化 | adamw_8bit（非 paged），cosine + warmup |
| 注意力 | PyTorch SDPA（Windows 无官方 flash-attn 轮子） |
| 运行环境 | conda `pytorch`（Python 3.12，torch 2.10.0+cu128，transformers 5.12.1，peft 0.19.1，bitsandbytes 0.50.0，见 `requirements.txt`） |

**Qwen3.5-4B 两个特别之处**：① VL 架构，必须用 `Qwen3_5ForConditionalGeneration` 加载（`AutoModelForCausalLM` 会静默随机初始化文本层），加载后摘除视觉塔；② 3:1 混合注意力（24 层 GatedDeltaNet 线性注意力 + 8 层全注意力），LoRA 目标模块按带父前缀的 endswith 匹配 12 类。

## 目录结构

```
data/
  convert_tora.py        # ToRA 原始轨迹 → 多轮对话数据（train/val + grpo_pool）
  split_grpo_pool.py     # grpo_pool → eval_set(100 持出) + grpo_train
  analyze_tora.py        # 转换前数据摸底
  tora_sft/              # 生成的数据：train(1901)/val(99)/grpo_train(3214)/eval_set(101)
scripts/
  envs/math_env.py       # 数学闭环环境：受限执行 + rollout + boxed 打分（训练/评测共用）
  training/
    train_qlora.py       # QLoRA(AdaLoRA) SFT（含 loss mask、8GB 显存适配全套）
    train_grpo.py        # 整条链路 GRPO（组内优势 + μ 轮重放的 PPO 裁剪 + KL）
    merge_lora.py        # adapter 合并回底座
  evaluation/eval_math.py # 闭环做题评测（100 题 boxed 正确率）
  tests/                 # 离线自测（不占 GPU）：纯函数 + mock rollout
  tools/                 # 显存探测、mask 校验、趋势分析、报告、下模型、冒烟测试
logs/                    # SFT/GRPO 训练日志 + 步级/轨迹审计 JSONL
outputs/                 # 训练产物（checkpoint 权重不入库，见下）
```

## 复现

```bash
conda activate pytorch
# 0. 底座模型：Qwen3.5-4B 放到 models/Qwen3.5-4B（或 python -m scripts.tools.download_model）
python -m scripts.tools.smoke_test                       # GPU 栈冒烟测试

# 1. 数据（可复现，seed 固定）
python data/convert_tora.py && python data/split_grpo_pool.py
python -m scripts.tools.check_mask --data data/tora_sft/val.jsonl   # 校验 loss mask
python -m scripts.tests.test_math_env && python -m scripts.tests.test_math_rollout

# 2. SFT（6000 步 ≈ 7.5h，峰值 ~5.6GB）
python -m scripts.training.train_qlora --model models/Qwen3.5-4B \
    --train data/tora_sft/train.jsonl --val data/tora_sft/val.jsonl \
    --out outputs/qlora-tora --max-steps 6000 --save-steps 500 --accum 4 --max-len 1024

# 3. 选优合并（eval_loss 谷底 ckpt-2000）+ 闭环评测（base/SFT 同一把尺子）
python -m scripts.training.merge_lora --model models/Qwen3.5-4B \
    --adapter outputs/qlora-tora/checkpoint-2000 --out outputs/merged-tora
python -m scripts.evaluation.eval_math --model models/Qwen3.5-4B --tag base
python -m scripts.evaluation.eval_math --model outputs/merged-tora --tag sft

# 4. 整条链路 GRPO（P=2×G=4，~3-4 分钟/步；每 20 步 greedy 闭环评估存 best）
python -m scripts.training.train_grpo --model outputs/merged-tora \
    --out outputs/grpo-tora --max-steps 60 --num-generations 4 --save-steps 10
# （可选）PPO μ=4 重放 + 首回合批量采样：--epochs 4 --gen-batch 4，写新目录勿覆写封存结果
python -m scripts.evaluation.eval_math --model outputs/merged-tora \
    --adapter outputs/grpo-tora/best --tag grpo

# 5. GRPO v2（probe 任务池 + 难度感知采样器，200 步 ≈ 32h）
python -m scripts.training.train_grpo --model outputs/merged-tora \
    --out outputs/grpo-v2 --data data/grpo_v2/pool_train_v2.jsonl \
    --difficulty data/grpo_v2/difficulty_v2.json --eval-data data/grpo_v2/dev_v2.jsonl \
    --max-steps 200 --num-generations 8 --gen-batch 8 --prompts-per-step 2 \
    --lr 1e-5 --beta 0.05 --eval-every 20 --eval-n 40 --save-steps 20
# 意外中断后续训（自动扫日志找断点，TB 曲线保持同一条）：
python -m scripts.tools.train_resume        # 本地看门狗：python -m scripts.tools.train_watchdog
python -m scripts.evaluation.eval_math --model outputs/merged-tora \
    --adapter outputs/grpo-v2/best --data data/grpo_v2/eval_v2.jsonl --n 80 --tag v2-best
cat outputs/eval/math_summary.md                          # 全部评测对比表
```

**不想跑训练？** 关键 adapter 权重在本仓库 [Release](../../releases/tag/archive-2026-10-24)：SFT ckpt-2000 与 GRPO best 两个 adapter（含 config），下载放回对应目录后从上面第 3 步开始即可（merge_lora 十分钟复现完整模型）。

## 8GB 显存红线（要点）

- **同一时刻只跑一个训练/评测进程**（`outputs/.train.lock` 单实例锁）；训练期间勿起评测/演示进程。
- WDDM 假死识别：GPU 利用率 100% 但功耗 ~30W、每步几分钟 = 显存页被换出到内存。自查 `nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv -l 2`。
- 内置六项修复，缺一不可：冻结权重压回 bf16、分块交叉熵、摘视觉塔、评估只算 loss、分配器 `garbage_collection_threshold:0.6` + 每步 `empty_cache()`（`expandable_segments` Windows 不支持）、非 paged `adamw_8bit`。GRPO 侧：`logits_to_keep` 只对补全段出 logits、KL 用 `disable_adapter()` 现场算。
- bs=1×1024 实测峰值 5.0GB 安全；bs=2×1024 触顶（7.2~7.9GB）。扩 batch 前先用 `python -m scripts.tools.probe_vram` 探测。

## 安全说明

`math_env.run_python_code` **真实执行**模型生成的 Python——代码解释器是本链路的训练目标（TIR 范式）。执行在隔离子进程中运行：`python -I`（隔离模式）+ 独立临时目录 + 环境变量精简 + 超时 20s 强杀 + 输出截断。请勿在存放敏感数据的环境里跑训练/评测。

## 文档

- `AGENTS.md` — 面向开发/维护的工程约定：显存红线细节、模板坑、已知坑、代码约定
- `ARCHIVE-INFO.txt` — 封存时点的成果与复现要点
- `outputs/eval/math_summary.md` — 全部评测汇总
