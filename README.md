# Qwen3.5-4B QLoRA(AdaLoRA) + GRPO 智能体能力微调

基于 4bit QLoRA(AdaLoRA) 微调 **Qwen3.5-4B**，增强其工具调用（function calling）与多轮智能体行为能力；
之后用**整条链路 GRPO**（模型从用户问题出发自己走完整条链路：调用工具→真的拿到返回→继续→最终回答，
按整条轨迹的结果做组内验证对齐）继续优化。全部流程在消费级 8GB 显存笔记本 GPU 上完成。

## 硬件与方案

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 5070 Laptop 8GB（8151 MiB 可分配） |
| CPU / 内存 | AMD Ryzen 9 8945HX（含 Radeon 集显）/ 16GB |
| 操作系统 | Windows 11（WDDM 显存模型，见下文"显存红线"一节） |
| 底座模型 | Qwen3.5-4B（VL 架构权重，本项目仅用文本部分） |
| 量化 | bitsandbytes NF4 4bit + 双重量化，计算精度 bf16 |
| AdaLoRA | init_r=16 → target_r=8（自适应秩重分配），alpha=32，dropout=0.05，12 类线性层 |
| 训练 | **按步驱动**：max_steps=216（≈2 epochs），每步记录 loss，每 27 步保存 checkpoint + 评估 |
| GRPO | **整条链路 rollout**：每个任务采样 G=4 条完整轨迹（工具真执行），轨迹级优势 + PPO 裁剪 + KL 锚定；旧版逐决策点模式保留为 `--mode step` |
| 优化 | adamw_8bit（8bit 状态，非 paged——原因见显存一节），lr=1e-4 cosine + warmup，bs=1 × accum=8 |
| 注意力 | PyTorch SDPA（Windows 无官方 flash-attn 轮子）；线性注意力走 torch 回退实现 |
| 运行环境 | conda `pytorch`（Python 3.12，torch 2.10.0+cu128，transformers 5.12.1，peft 0.19.1，bitsandbytes 0.50.0） |

### 模型：Qwen3.5-4B 的两个特别之处

1. **它是 VL（视觉-语言）架构**：配置类为 `Qwen3_5ForConditionalGeneration`（checkpoint 键名是
   `model.language_model.*`）。用 `AutoModelForCausalLM` 加载会因键名不匹配而**静默随机初始化**文本层，
   必须用 VL 类加载；本项目只做文本训练，加载后立即把视觉塔（168.9M 参数）从模块树摘除并归还显存。
2. **3:1 混合注意力**：32 层 = 24 层 GatedDeltaNet 线性注意力（`linear_attn.*`）+ 8 层全注意力
   （`self_attn.*`）。LoRA 按模块名 endswith 匹配，12 类目标模块：

   ```
   linear_attn.in_proj_qkv / in_proj_z / in_proj_b / in_proj_a / out_proj   （线性注意力）
   self_attn.q_proj / k_proj / v_proj / o_proj                              （全注意力）
   mlp.gate_proj / up_proj / down_proj                                      （稠密 FFN）
   ```

   带父模块前缀（`linear_attn.`、`self_attn.`）可避免误命中视觉塔里的同名层。
   可训练参数 32.5M（占 0.77%）。

### 工具调用格式：Qwen3.5 原生 XML

数据里的 assistant 轮仍是结构化 `tool_calls`（chat template 会自动渲染成 XML，**数据集文件无需改动**）：

```text
<tool_call>
<function=get_current_weather>
<parameter=city>
北京
</parameter>
</function>
</tool_call>
```

工具返回通过 `role="tool"` 渲染成下一轮 user 消息里的 `<tool_response>…</tool_response>`。
推理/评估时解析必须 `skip_special_tokens=False`（`rewards.py` 的解析器即按此实现）。
模型是思考/非思考双模，本项目的智能体轨迹全部走**非思考模式**（`enable_thinking=False`，
chat template 会给 assistant 轮插入空 `<think></think>` 块）。

## ⚠ 8GB 显存红线：如何避免 Windows/WDDM "跨显卡搬运"

这是本项目在 8GB 笔记本卡上最需要注意的坑，也是全部实现里最关键的工程约束。

**症状**：GPU 利用率 100% 但功耗只有 ~30W；每步从几秒恶化到几分钟；任务管理器里
NVIDIA 的"专用显存"打满（8.0GB），"共享显存"开始增长——驱动把显存页换出到内存（走 PCIe），
PyTorch 再复用时又要换回来，实测 896 token 的单步 fwd+bwd 峰值 7.59GB 时它**连第 0 步都跑不完**。

**本项目的六项修复**（每项都实测过收益）：

| # | 修复 | 机理与实测 |
|---|---|---|
| 1 | `shrink_frozen_fp32()`：把 peft 升到 fp32 的冻结权重压回 bf16 | `prepare_model_for_kbit_training` 会把所有非量化权重升 fp32；Qwen3.5 词表 248k，**词嵌入（与 lm_head 共享）一个张量就是 2.37GB**，而且让 logits/梯度也变 fp32（896 token 各 0.89GB）。压回 bf16 省 **≈1.9GB**（冻结权重无优化器精度问题） |
| 2 | 分块交叉熵 `--ce-chunk`（默认 256 行） | 原实现 `logits.float()` 整段 fp32 = 890MB + softmax 再一份；分块后瞬时只需 ~250MB，数值等价 |
| 3 | 加载后摘除视觉塔 | 168.9M 参数（4bit ≈ 0.16GB）文本训练用不到；**注意必须先删局部引用再 gc**，否则只是解注册、显存不释放 |
| 4 | 评估只算 loss（`prediction_loss_only=True`） | 248k 词表下验证集 45 条的 logits 若累积要十几 GB——"跑到评估就卡死"的根因 |
| 5 | 分配器 `garbage_collection_threshold:0.6` + 每步 `torch.cuda.empty_cache()` | `expandable_segments` 在 Windows 上**不支持**（日志会 warning）；WDDM 下缓存块会一直占着专用显存并被驱动优先换出，每步归还让专用显存贴着真实用量 |
| 6 | 非 paged 的 `adamw_8bit`（不要 `paged_adamw_8bit`） | paged 优化器在显存吃紧时把状态页在 CPU/GPU 之间搬运，正是 30W 假死的一大来源 |

GRPO 侧另有第 7 项：`token_logprobs` 用 `logits_to_keep` 只对补全段出 logits（省 ~1.5GB），
KL 参考策略用 `disable_adapter()` 现场计算（零额外显存）。

**修复后实测**（30 步 smoke，同机同数据，2 秒粒度采样显存）：

- 模型就绪 3.02GB / reserved 3.19GB；**训练峰值 allocated 5.57GB**（修复前同配置 7.59GB）
- 分配器 reserved 峰值 **10.78GB → 6.89GB**（每步 `empty_cache` 生效的直接证据）
- 30 步 3 分 23 秒（≈6.8 秒/步）；全程功耗 70-107W（真实计算）
- **共享显存全程 ≤283MB、训练稳态 ~95MB（=桌面临界基线），零换页**（修复前峰值 1.71GB）
- 216 步预计 ≈25-30 分钟

**自查方法**：训练脚本结尾会打印 `训练峰值显存: allocated/reserved`；运行中盯
`nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv -l 2`——利用率高但功耗 30W 左右
就是换页；或者看任务管理器"共享显存"是否增长。

**可选终极保险**：NVIDIA 控制面板 → 管理 3D 设置 → 程序设置里给 `python.exe` 设
**CUDA - Sysmem Fallback Policy = Prefer No Sysmem Fallback**。这样驱动宁可报显存不足（显式 OOM）
也不再静默换页——把"假死"变成"看得见的错误"。

另外两条纪律：

1. **必须单实例**：同一时刻只能有一个训练进程（脚本内置 `outputs/.train.lock` 单实例锁）。
   双进程争抢同一张 8GB 卡会互相把对方挤到共享显存上，必然出现上述假死。
2. **关掉占显卡的桌面应用**（浏览器/微信/Overlay 等）：实测桌面常驻占用 ~660MB 专用显存，
   训练峰值 5.6GB + 系统占用后总预算 8GB 仍有余量；若同时开大型 3D 应用则可能再次触顶。

## 目录结构

```
Qwen-QLora/
├── data/
│   ├── build_dataset.py      # 合成 agent 轨迹生成器（6 类场景，可复现 seed=42）
│   ├── train.jsonl           # 855 条训练集
│   ├── val.jsonl             # 45 条验证集
│   └── grpo_prompts.jsonl    # make_grpo_prompts.py 提取的 2394 个决策点
├── scripts/
│   ├── download_model.py     # ModelScope 下载底座模型（默认 Qwen/Qwen3.5-4B）
│   ├── smoke_test.py         # GPU 训练栈冒烟测试（不依赖模型）
│   ├── check_mask.py         # loss mask 人工校验
│   ├── train_qlora.py        # QLoRA(AdaLoRA) 训练主脚本
│   ├── merge_lora.py         # 合并 LoRA 为完整模型（CPU bf16）
│   ├── agent_env.py          # 智能体闭环环境：确定性工具沙箱 + 真值回放 + rollout + 链路打分
│   ├── eval_agent.py         # 评估：闭环整条链路（默认）/ 逐轮步进（--mode step）
│   ├── make_report.py        # 汇总 eval_*.json 为对比报告
│   ├── show_eval_curve.py    # 打印各 checkpoint 的 eval_loss 曲线
│   ├── rewards.py            # GRPO 打分器（细粒度规则+执行验证，XML 解析）
│   ├── make_grpo_prompts.py  # 从轨迹提取步级决策点（--mode step 用）
│   ├── train_grpo.py         # GRPO 对齐训练：整条链路（默认）/ 逐决策点（--mode step）
│   ├── grpo_report.py        # GRPO 前后对比报告汇总
│   ├── grpo_trend.py         # 训练中途看趋势：窗口均值 + 健康检查 + 分类别拆分
│   ├── check_forgetting.py   # 灾难性遗忘检测
│   └── demo_inference.py     # 交互演示（完整闭环，工具沙箱真执行）
├── models/
│   └── Qwen3.5-4B            # junction -> D:\NLP\ft-qwen3.5-4b-aviation\models\Qwen3.5-4B（不复制 9.3GB）
├── logs/                     # GRPO 训练日志（grpo_log / grpo_trajs / grpo_groups）
└── outputs/                  # adapter、合并模型、评估结果、日志
```

## 数据集

脚本合成 900 条工具调用轨迹（可复现，随机种子 42），6 类场景配比：

| 类别 | 数量 | 说明 |
|---|---|---|
| single_call | 240 | 单轮单工具调用（天气/计算/汇率/股票/IP/餐厅） |
| chained | 180 | 多步链式调用（上一步结果决定下一步参数，如 查航班→订酒店） |
| parallel | 150 | 并行调用（一轮内多个相互独立的工具调用） |
| error_retry | 120 | 工具报错后分析原因、修正参数重试 |
| no_tool | 120 | 负样本：无需工具直接回答 / 口算 / 拒绝危险请求 |
| clarify | 90 | 必填参数缺失时先向用户澄清，不瞎猜 |

渲染后 token 长度：p50=619，p95=850，max=896（`--max-len 1024` 不截断）。20 个内置工具定义见
`build_dataset.py::TOOL_DEFS`。

**Loss mask**：训练只对 assistant 自己生成的内容（`<tool_call>` XML 块、`<|im_end|>`、最终回答）
计算损失，系统提示、工具定义、用户消息、工具返回值全部 mask 为 -100。mask 区间用
"公共前缀对齐"算法精确定位（`train_qlora.py::AgentTrajectoryDataset._encode`），
可通过 `python scripts/check_mask.py` 人工校验。

## 快速开始

```bash
# 0. 环境（conda pytorch 环境，Python 3.12）
conda activate pytorch
pip install -r requirements.txt

# 1. （可选）下载底座模型——本机已就绪：models/Qwen3.5-4B 是指向
#    "D:\NLP\ft-qwen3.5-4b-aviation\models\Qwen3.5-4B" 的 junction
python scripts/download_model.py

# 2. 冒烟测试 GPU 训练栈
python scripts/smoke_test.py

# 3. 重新生成/调整数据集（可选）
python data/build_dataset.py --total 900

# 4. 校验 loss mask（可选）
python scripts/check_mask.py

# 5. AdaLoRA 训练（按步驱动，约 25-30 分钟）
python scripts/train_qlora.py \
    --model models/Qwen3.5-4B \
    --train data/train.jsonl --val data/val.jsonl \
    --out outputs/qlora-adapter \
    --max-steps 216 --save-steps 27 --accum 8

# 6. 基线 vs 微调后评估（默认同时跑闭环+步进两种口径）
python scripts/eval_agent.py --model models/Qwen3.5-4B --tag base
python scripts/eval_agent.py --model models/Qwen3.5-4B \
    --adapter outputs/qlora-adapter/final --tag finetuned
python scripts/make_report.py     # -> outputs/eval/report.md

# 7. 合并为可独立部署的完整模型（GRPO 的训练底座）
python scripts/merge_lora.py --model models/Qwen3.5-4B \
    --adapter outputs/qlora-adapter/final --out outputs/merged

# 8. 整条链路 GRPO（模型自己走完多轮工具调用，约 1-2 分钟/步）
python scripts/train_grpo.py --model outputs/merged --data data/train.jsonl \
    --out outputs/grpo-adapter --max-steps 100 --num-generations 4 --save-steps 10

# 9. GRPO 后评估 + 遗忘检测 + 汇总报告
python scripts/eval_agent.py --model outputs/merged \
    --adapter outputs/grpo-adapter/final --tag grpo
python scripts/check_forgetting.py --model outputs/merged \
    --adapter outputs/grpo-adapter/final --out outputs/forgetting_grpo
python scripts/grpo_report.py

# 10. 对话演示（完整闭环：真解析、真执行沙箱、多轮）
python scripts/demo_inference.py --model outputs/merged --adapter outputs/grpo-adapter/final
```

训练监控：`tensorboard --logdir outputs/qlora-adapter`

> `train_qlora.py` 关键参数：`--peft-method adalora|lora`、`--init-r/--target-r`、
> `--ce-chunk`（分块交叉熵，显存不够可降到 64）、`--optim`（默认 adamw_8bit）、`--max-len`。

## 评估口径

`eval_agent.py` 提供两种口径（`--mode both` 默认都跑，同一把尺子 = `rewards.py` + `agent_env.py`）：

**1) 端到端闭环（默认）**：模型从首条 user 消息出发，自己走完整条链路——调用工具
（沙箱/真值回放，真执行）→ 拿到返回 → 继续 → 最终回答；按整条轨迹判定任务成功。

| 指标 | 含义 |
|---|---|
| task_success_rate | 端到端任务成功率：真值调用全覆盖且参数等价 + 最终回答事实达标 + 无违规 |
| avg_turns | 平均回合数（模型出手几次，越少越利落） |
| call_f1 / arg_avg | 调用序列覆盖 F1 / 命中对参数软匹配均分 |
| answer_score | 最终回答得分（关键数字事实命中） |
| fmt_valid_rate | 可解析回合占比 |
| premature_rate | 没给出最终回答就中断（轮数/长度用尽）的比例 |

**2) 逐轮步进（`--mode step`，旧口径）**：把真值历史喂给模型只生成一轮：

| 指标 | 含义 |
|---|---|
| turn_acc | 所有轮次全对（工具选择+参数）的比例 |
| tool_choice_acc | 该调用时选对工具 + 不该调用时不调用的比例 |
| fmt_valid_rate | `<tool_call>` XML 可解析率 |
| arg_acc_when_chosen | 选对工具时参数完全匹配率（可选参数缺失可容忍） |
| final_answer_acc | 最终回答不含误调用且关键数字事实命中率 ≥60% |

## 训练结果

> **诚实声明**：SFT 已完整跑完（216 步，2.02 epochs）；GRPO 整条链路模式已通过 4 步冒烟
> （工具链端到端验证），**完整 GRPO 训练尚未运行**。下面数字均为实测，但冒烟步数很少，
> **不代表收敛后的效果**。

**SFT（AdaLoRA，216 步，已跑完）**

- 训练 loss：3.18（step 1）→ ~0.66–0.96（step 214–216）
- eval_loss（45 条留出轨迹，每 27 步评估一次）：1.866 → 1.481 → 1.193 → 1.011 →
  0.903 → 0.848 → 0.828 → **0.823**（收尾仍在下探）
- 训练峰值显存 5.57GB，全程零 WDDM 换页（见上文"显存红线"）
- 产物：`outputs/qlora-adapter/`（checkpoint-27…216 + final）

**GRPO 整条链路（4 步冒烟：基座模型直跑，只为验证工具链与显存）**

- 每步 8–66 秒（1 回合的短任务 8s；3 回合的长任务 ~65s），峰值显存恒稳 5.66GB、零 OOM
- 审计日志可见完整链路：`calculator((53+90)*9)` → 真值回放返回 `{"result": 1287}` →
  最终回答 1287 ✓（第 2 步 4 条轨迹 reward 1.0/1.0/1.0/0.80，组内方差驱动更新）
- 闭环评估冒烟（4 条验证轨迹，基座模型）：任务成功 2/4，平均回合 2.25，格式有效率 100%

跑完快速开始第 8-9 步后，`make_report.py` / `grpo_report.py` 会自动生成
base / SFT / SFT+GRPO 三方对比表（`outputs/eval/report.md`、`outputs/eval/grpo_report.md`）。

## GRPO 整条链路对齐（模型自己走完整条链路）

SFT 之后再做一轮 **整条链路 GRPO**：对每个任务，让模型**从用户问题出发自己走完**
（调用工具 → 真的拿到返回 → 继续推理 → 最终回答），同一任务采样 G=4 条完整轨迹成组，
由程序化验证器按**整条轨迹的结果**打 0~1 分，组内归一化优势 `(r-mean)/std` 沿轨迹全部
生成 token 做 PPO 裁剪策略梯度，KL 锚定 SFT 模型防漂移。不依赖人工偏好、不依赖外部奖励模型。

**工具环境（`scripts/agent_env.py`，训练/评估/演示三处共用）**

- **真值回放**：模型调用与真值轨迹中的调用等价（工具名一致、参数按"执行验证"等价，
  如 `35**3` ≡ `pow(35,3)`）→ 返回数据里记录的返回，事实性数值与真值终答严格对齐；
- **确定性沙箱兜底**：未命中回放的调用走 mock 沙箱（同一调用永远同一结果），错误语义与
  合成数据一致（`city_not_found` / `file_not_found` / `invalid_parameter` …）；
- **用户模拟器**：需要澄清/补充信息的任务，模型开口澄清后按顺序回放 GT 里的用户追问；
- **安全红线**：`run_python` 一律拒绝执行——模型生成的代码绝不落地运行。

**链路打分（`agent_env.score_trajectory`）**

```
base = 0.55 × 调用覆盖 + 0.45 × 最终回答
  调用覆盖（乘性）= 命中真值调用占比 × (0.4×名称F1 + 0.6×参数软匹配均分)   ← 漏链路直接按比例砍
  最终回答 = 0.4 + 0.6×关键数字事实命中率（无数字真值看长度合理性；误发调用记 0）
乘性纪律项：该澄清未澄清 ×0.5 ｜ 无需工具却误发调用 ×0.5 ｜ 格式门槛 ×(0.5+0.5×可解析回合占比)
```

**起点与超参**：`merge_lora.py` 合并出的 `outputs/merged`（量化后既作训练底座，也作 KL 参考策略，
`disable_adapter()` 零额外显存）；默认 `lr=1e-5, β=0.04, T=0.9, G=4, max_turns=4`，
每步 1 个任务（`--prompts-per-step` 可调，越大越费时）。

```bash
# 8. 整条链路 GRPO（100 步 ≈ 覆盖 100 个任务；训练集共 855 个任务，步数可按需加大）
python scripts/train_grpo.py --model outputs/merged --data data/train.jsonl \
    --out outputs/grpo-adapter --max-steps 100 --num-generations 4 --save-steps 10

# 旧模式（逐决策点，真值前缀钉死只学单步动作）仍在，加 --mode step：
# python scripts/train_grpo.py --mode step --model outputs/merged \
#     --prompts data/grpo_prompts.jsonl --max-steps 150 --num-generations 4 --prompts-per-step 2

# 9. 评估（闭环整条链路）+ 遗忘检测 + 汇总报告
python scripts/eval_agent.py --model outputs/merged \
    --adapter outputs/grpo-adapter/final --tag grpo
python scripts/check_forgetting.py --model outputs/merged \
    --adapter outputs/grpo-adapter/final --out outputs/forgetting_grpo
python scripts/grpo_report.py
```

**审计日志（每步留痕）**：

- `logs/grpo_log.jsonl`：每步 reward 均值 / 端到端成功率 / 平均回合 / 调用覆盖 / KL /
  grad_norm / 显存 / **单步秒数**（`loss` 恒接近 0 是 GRPO 的正常现象：组内优势之和为 0，
  更新点 ratio=1 时策略损失均值恰好抵消——学习信号看 `grad_norm` 与 `success_rate`）
- `logs/grpo_trajs.jsonl`：每步每任务的 G 条**完整链路全文**（工具调用、工具返回、最终回答）
  与逐条得分分解——可直接审计"模型这一轮怎么走、错在哪一步、被组内比较推向哪"
- **训练中途看趋势**：`python scripts/grpo_trend.py`——读上面两份日志，输出窗口均值 /
  前后段对比 / 健康检查（kl、空步率、有效组占比）/ 分类别拆分。单步日志只有 1 个任务
  ×G 个样本，噪声极大，判"是否在学"至少看 20-30 步的窗口方向，别被相邻两步的涨跌带跑
- `tensorboard --logdir outputs/grpo-adapter`

调参提示：默认偏保守。若 `success_rate` 长期不动、`coverage` 卡住，可放宽到
`--lr 3e-5 --beta 0.025 --temperature 1.0`；长任务多可加 `--max-turns 5 --max-len 3200`。
显存吃紧则降 `--num-generations 2` 或 `--prompts-per-step`（默认 1）。

## 常见问题

- **显存打满 / 训练"假死"（利用率 100% 但 30W、每步几分钟）**：见上文"8GB 显存红线"一节。
  脚本已内置六项修复并自动设置分配器参数（`PYTORCH_CUDA_ALLOC_CONF` 无需手动导出）；
  若仍触顶：降 `--ce-chunk 64`、确认没有第二个训练进程（`outputs/.train.lock`）、
  关闭占显卡的桌面应用，或按该节设置 NVIDIA 控制面板的 Sysmem Fallback Policy。
- **peft 0.19 AdaLoRA 收尾崩溃**：最后强制掩码步（step == total_step - tfinal）会因内部字典被
  `reset_ipt()` 清空而抛 KeyError，训练脚本已捕获并沿用上一次秩分配，无损训练完整性。
  （注意 AdaLoRA 要求 `tinit < total_step - tfinal`，所以 `--max-steps` 至少要 ~25。）
- **日志出现 "The fast path is not available … falling back to torch implementation"**：线性注意力
  （GatedDeltaNet）的快路径需要 `flash-linear-attention` + `causal-conv1d`，本机未装，走 torch 回退，
  结果一致、速度略慢；`pip install causal-conv1d` 后可提速。
- **换模型**：`--model` 指向其他 Qwen3.5 系列本地目录即可；若换回非 VL 的 Qwen3 系模型，
  需把 `Qwen3_5ForConditionalGeneration` 换回 `AutoModelForCausalLM` 并按需调整 `TARGET_MODULES`。
- **Windows 与 flash-attn**：无官方 Windows 轮子，本方案用 PyTorch SDPA（内置 FlashAttention 内核），
  性能差距在本次规模下可忽略。
- **视觉塔被摘除**：纯文本训练/推理不受影响；若要恢复多模态能力，需用原始 checkpoint 重新加载完整模型。
- **GRPO 日志里 `loss`/`kl` 一直是 0？** 正常现象：组内归一化优势之和为 0，更新点 ratio=1 时
  策略损失均值恰好抵消；KL 在 LoRA 增量还接近零时自然也是 0。学习信号看 `grad_norm`（>0 即有梯度）
  与 `success_rate` / `coverage` 曲线。
- **error_retry 任务"答对也不成功"？** 该类任务的成功要求真正走一遍"报错→修正"路径
  （调用序列覆盖真值）：绕过报错直接答对拿 reward 0.7–0.95，但不计 success——这正是该类别
  的训练目标。clarify 类同理：不先澄清直接猜会打半折（×0.5）。
- **闭环评估/GRPO 很慢？** 轨迹模式逐条生成（回合之间序列长度不同，无法批处理），3 回合长任务
  约 1 分钟/步（8GB 卡）；`--num-generations 2` 或缩小 `--max-new-tokens` 可提速。
- **沙箱里的工具返回值是"假的"吗？** 是确定性 mock：命中真值调用时回放数据里记录的返回，
  其余按参数哈希生成稳定返回。训练目标（格式/名称/参数/事实）全部程序化可验证，不受影响；
  同时避免训练时真的发邮件、订酒店——`run_python` 更是直接拒绝执行。
