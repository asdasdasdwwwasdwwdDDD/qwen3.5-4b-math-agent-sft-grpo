# AGENTS.md

## 项目概述

Qwen3.5-4B 智能体微调：QLoRA(AdaLoRA) SFT + 整条链路 GRPO。全部流程针对 **8GB RTX 5070 Laptop GPU + Windows 11 (WDDM)** 设计——这是所有代码修改的第一约束。

- **本目录是完整封存包**（冻结于 2026-10-24、TRL 框架迁移动工之前），用于复现阶段工作；后续开发在别的目录进行。动手前先确认用户意图是"复现/查阅"还是"在此续训"，封存说明与最终成果表见 `ARCHIVE-INFO.txt`。
- **主线（唯一保留的链路）：ToRA 数学推理**——用 `D:\NLP\ft-qwen3.5-4b-agent-cot-data` 的官方 ToRA 轨迹（题目 → ```python → 真实执行结果 → \boxed 终答）做 SFT，再用 `math_env.py`（受限执行 + boxed 正确性奖励）做整条链路 GRPO。**旧 XML 工具调用管线（agent_env/rewards/eval_agent/step 模式等）已于 2026-11-08 删除**，只保留轨迹层级训练模式。
- 已开源：https://github.com/asdasdasdwwwasdwwdDDD/qwen3.5-4b-math-agent-sft-grpo（权重在 Release 附件）。
- 运行环境：conda env `pytorch`（`D:\anaconda\envs\pytorch`，Python 3.12；torch 2.10.0+cu128 / transformers 5.12.1 / peft 0.19.1 / bitsandbytes 0.50.0）。所有脚本先 `conda activate pytorch`，**在仓库根目录以 `python -m scripts.<子包>.<模块>` 运行**（scripts 是包，勿用 `python scripts/xxx.py` 旧方式）。

## 目录结构

- `data/tora_sft/` — **数学链路数据**（由 `data/convert_tora.py` 生成，seed=42 可复现）：
  `train.jsonl`(1901)/`val.jsonl`(99)（多轮轨迹：user 题 → assistant ```python → tool ```output → assistant \boxed 终答）；
  `grpo_train.jsonl`(3214)/`eval_set.jsonl`(101 条记录，评测汇总口径 100 题)（由 `split_grpo_pool.py` 从 math_test 池切分，SFT 见过的 1640 题已剔除；eval_set 为 base/SFT/GRPO 三方共用持出尺子）
- `data/` — 处理脚本平铺：`convert_tora.py`（内联轨迹→多轮 messages，含 role="tool" 回填的模板坑详注）、`split_grpo_pool.py`、`analyze_tora.py`（转换前摸底）
- `scripts/` — **包结构**（各子包有 `__init__.py`，跨模块 import 一律 `from scripts.xxx import`）：
  - `envs/math_env.py` — 唯一闭环环境（受限 Python 执行 + rollout + boxed 打分 + `render_ids`/`generate_once` 生成原语），被训练与评测共用
  - `training/train_qlora.py` — SFT 主脚本（同时是工具函数枢纽：CHAT_TMPL_KW/TARGET_MODULES/训练锁/摘视觉塔/冻结权重压缩/公共前缀 loss mask）
  - `training/train_grpo.py` — 整条链路 GRPO（只有 trajectory 模式；μ 轮重放的 PPO 裁剪 + KL，`--epochs` 默认 1=封存口径；主干拆为 generate_scored_group/group_advantages/train_step_update/log_step 四个函数）
  - `training/merge_lora.py` — adapter 合并
  - `evaluation/eval_math.py` — 闭环做题评测（内联 4bit 加载）
  - `tests/` — 离线自测（不占 GPU）：`test_math_env.py`（纯函数）、`test_math_rollout.py`（mock 模型走多轮）
  - `tools/` — `check_mask.py`（mask 校验）、`probe_vram.py`（显存探测）、`show_eval_curve.py`（SFT 选点）、`math_trend.py`/`grpo_trend.py`/`grpo_report.py`（趋势与报告，默认读 math 日志）、`check_forgetting.py`、`download_model.py`、`smoke_test.py`
- `models/Qwen3.5-4B` — **junction（封存未包含，`models/` 目录不存在）**，指向 `D:\NLP\ft-qwen3.5-4b-aviation\models\Qwen3.5-4B`（不复制 9.3GB 权重）。**训练/评估前先 `ls models/Qwen3.5-4B` 验证能列出 config.json**；不存在/失效则重建
  `cmd /c rmdir models\Qwen3.5-4B` + `cmd /c mklink /J models\Qwen3.5-4B D:\NLP\ft-qwen3.5-4b-aviation\models\Qwen3.5-4B`（rmdir 只删链接不删模型）
- `logs/` — `sft_tora.log`（数学 SFT）、`grpo_tora_run.log`（数学 GRPO 全程）、`sel_*.log`/`merge_tora.log`/`final_evals.log`（选优/合并/终评）；GRPO 步级/轨迹审计 `grpo_log_math.jsonl`/`grpo_trajs_math.jsonl`
- `outputs/` — 封存只含 `qlora-tora/`（SFT）、`grpo-tora/`（RL）、`eval/`；`outputs/.train.lock` 为单实例训练锁（未入包）。`merged-tora/`（8.5GB）**未入包**，merged 模型用 `scripts/training/merge_lora.py` 从 checkpoint 十分钟复现

## 当前状态（封存时点 2026-10-24：全流程已完成，无待办）

- **最终成果**（eval_set 持出集 100 题，4bit 部署口径，汇总在 `outputs/eval/math_summary.md`）：
  base ~0% ｜ **SFT 37.0%**（ckpt-2000；L1 90.9 / L2 55.0 / L3 57.1 / L4 4.5 / L5 11.5）｜ **GRPO 44.0%**（best=step-20；L1 90.9 / L2 50.0 / L3 66.7 / L4 18.2 / L5 23.1）。
  GRPO 附加收益：执行报错率 37.7%→24.3%，未终答 6%→0%。
- **SFT**：6000 步计划跑到 ~4900 步，用户确认过拟合后提前终止；eval_loss 曲线 0.279(500)→0.198(1000)→0.106(1500)→**0.104(2000, 谷底)**→回升至 0.151(4500)。checkpoint-500…4500 共 9 个存档点在 `outputs/qlora-tora`，日志 `logs/sft_tora.log`。
- **选优**：eval_loss 只看 token loss，最终按 30 题快速生成评测（`outputs/eval/math_sel-*.json`：base 0% / ck1500 66.7% / **ck2000 73.3%** / ck2500 66.7%）双指标选定 **checkpoint-2000**，合并产物（未入包）即 `outputs/merged-tora`。
- **GRPO 已跑完**：60 步课程化（难度层轮转交织）+ T=0.8，组内优势 + PPO 裁剪 + KL(beta=0.025)；训练中每 20 步 greedy 闭环评估存 best，**best=step-20**，checkpoint-10…60 与 final 在 `outputs/grpo-tora`，日志 `logs/grpo_tora_run.log` + `grpo_log_math.jsonl`/`grpo_trajs_math.jsonl`。
- 复现顺序：`convert_tora.py → split_grpo_pool.py → check_mask.py → train_qlora.py → merge_lora.py → train_grpo.py → eval_math.py`（`ARCHIVE-INFO.txt` 里的 `--mode math` 是笔误，train_grpo 无 --mode/--env 旗标，默认即数学轨迹模式）。

## 常用命令（复现数学链路；数据已生成，可直接从任意一步开始；一律仓库根运行）

```bash
# 数据（可复现，源在 D:\NLP\ft-qwen3.5-4b-agent-cot-data\data\tora）
python data/convert_tora.py && python data/split_grpo_pool.py
python -m scripts.tools.check_mask --data data/tora_sft/val.jsonl  # 校验 loss mask（两个 assistant 轮都要被训练）
python -m scripts.tests.test_math_env && python -m scripts.tests.test_math_rollout  # 离线自测（不占 GPU）
python -m scripts.tools.math_trend --tb     # GRPO 分层趋势（按难度层对齐）+ 派生 TensorBoard 曲线

# SFT（6000 步 ≈ 7.5h；峰值 ~5.6GB。bs=2×1024 实测 7.2GB 触顶，勿加）
python -m scripts.training.train_qlora --model models/Qwen3.5-4B \
    --train data/tora_sft/train.jsonl --val data/tora_sft/val.jsonl \
    --out outputs/qlora-tora --max-steps 6000 --save-steps 500 --accum 4 --max-len 1024

# 合并（选 eval_loss 最低的 checkpoint）+ 闭环评测（三方同一把尺子）
python -m scripts.training.merge_lora --model models/Qwen3.5-4B --adapter outputs/qlora-tora/checkpoint-N --out outputs/merged-tora
python -m scripts.evaluation.eval_math --model models/Qwen3.5-4B --tag base
python -m scripts.evaluation.eval_math --model outputs/merged-tora --tag sft

# 整条链路 GRPO（P=2×G=4，~3-4 分钟/步；训练中每 20 步 greedy 闭环评估存 best）
python -m scripts.training.train_grpo --model outputs/merged-tora \
    --out outputs/grpo-tora --max-steps 60 --num-generations 4 --save-steps 10
# PPO μ=4 重放 + 首回合组批量采样（新口径：同批数据重放 4 遍，ratio 偏离 1、裁剪上岗；
# 每训练步参数移动量 ≈×4，想保持总量可比可用 --max-steps 20。勿覆写封存的 grpo-tora，写新目录）
python -m scripts.training.train_grpo --model outputs/merged-tora \
    --out outputs/grpo-rerun --max-steps 60 --num-generations 4 --epochs 4 --gen-batch 4
python -m scripts.evaluation.eval_math --model outputs/merged-tora --adapter outputs/grpo-tora/best --tag grpo
cat outputs/eval/math_summary.md                                  # 三方对比表
tensorboard --logdir outputs/qlora-tora                           # 或 outputs/grpo-tora
```

## 硬约束：8GB 显存红线（改训练代码必读 README「显存红线」一节）

- **同一时刻只能有一个训练进程**（`outputs/.train.lock` 单实例锁，`train_qlora.py` 与 `train_grpo.py` 共用）；训练期间也别起评测/演示进程（同卡）。
- 假死症状：GPU 利用率 100% 但功耗 ~30W、每步几分钟 → WDDM 把显存页换出到内存。自查：
  `nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv -l 2`。
- 已内置六项修复，**不要回退**：`shrink_frozen_fp32()`（冻结权重压回 bf16）、分块交叉熵 `--ce-chunk`、
  摘除视觉塔、评估只算 loss（`prediction_loss_only=True`）、分配器 `garbage_collection_threshold:0.6` +
  每步 `empty_cache()`（`expandable_segments` 在 Windows 不支持）、**非 paged** 的 `adamw_8bit`。GRPO 侧：`logits_to_keep` 只对补全段出 logits、KL 用 `disable_adapter()` 现场算。
- **bs 与 max-len 的实测边界**（probe_vram.py，数学数据）：bs1×1024 峰值 5.0GB 安全；bs2×1024 真实标签 7.2GB、满标签 7.9GB **触顶**。加 batch 前必须先探测。
- 新增显存开销的功能先想清楚峰值预算（SFT 峰值 ~5.6GB allocated / 6.9GB reserved）。

## 模型与代码约定

- 加载必须用 `Qwen3_5ForConditionalGeneration`（VL 架构类）；`AutoModelForCausalLM` 会**静默随机初始化**文本层。加载后立即摘除视觉塔（先删局部引用再 gc，否则显存不释放）。
- 3:1 混合注意力（24 层 GatedDeltaNet 线性注意力 + 8 层全注意力）；LoRA 目标模块按**带父前缀的 endswith** 匹配（12 类），前缀用于避免误命中视觉塔同名层。
- 智能体轨迹全部走非思考模式（`enable_thinking=False`，即 `CHAT_TMPL_KW`；math_env 本地定义一份避免拖入训练栈）；解析必须 `skip_special_tokens=False`；`apply_chat_template` 必须传 `return_dict=False`。
- **数学链路的模板约定（关键）**：执行结果必须以 `role="tool"` 回填（渲染成 `<tool_response>` 包裹的 user 轮）。Qwen3.5 模板只给"最后一条真实 user 消息之后"的 assistant 轮插 `<think>`：若用裸 user 轮回填 ```output，中间的 assistant 代码轮会被剥 think，与生成提示渲染分叉，公共前缀 mask 算法会把**代码轮整个 mask 掉**（只训终答不训代码）。ToRA 数据转换时踩过（2026-10-24 修复，`convert_tora.py` 有详注）。
- `math_env.py` 被 `train_grpo.py` 与 `eval_math.py` 共用（同名同签名接口：`MathEnv`/`rollout`/`rollout_group`/`score_trajectory`/`format_trajectory`/`load_tasks`），改打分逻辑同时影响训练与评测口径。
- **安全线**：`math_env.run_python_code` **真实执行**模型生成的 Python（隔离子进程 `-I` + 临时目录 + 超时 20s + 输出截断）——代码解释器是数学链路的训练目标，这是用户指定的方向，残余风险为 TIR 训练通行做法。
- Loss mask 只对 assistant 生成内容计算（公共前缀对齐算法，`train_qlora.py::AgentTrajectoryDataset._encode`；数学数据无 tools 字段，恒按纯对话渲染）。

## 已知坑

- peft 0.19.1 AdaLoRA 收尾会因 `reset_ipt()` 抛 KeyError，训练脚本已捕获并沿用上次秩分配——勿删该 try/except；AdaLoRA 要求 `tinit < total_step - tfinal`，`--max-steps` 至少 ~25。
- GRPO 日志里 `loss`/`kl` 恒为 0 是正常现象（组内优势之和为 0；`--epochs` 默认 1 时 ratio 恒 1）；学习信号看 `grad_norm` 与 `success_rate`，判断趋势用 20-30 步窗口，别看相邻两步。μ>1 时 ratio 偏离 1，loss 不再恒 0，`clip_tokens` 非零即裁剪在工作。
- 数学链路同理鼓励"报错→读 Traceback→修正"（执行错误文本会回填给模型）。
- 闭环评估/GRPO 轨迹的**第 2 回合起**逐条生成（回合间长度不同无法批处理）；首回合可批量（`--gen-batch`，同题 G 条 prompt 相同），默认 1=逐条封存口径。数学链路 ~20-30s/条属正常。
- GRPO 的环境对象必须**每条 rollout 新建一个**（`MathEnv` 同理，2026-10-24 修复，勿回退）。
- **bash 传参层会折叠反斜杠**：经 `bash -c "python -c '...\\boxed...'"` 或（个别情况下）heredoc 传 python 代码时 `\\` 会变 `\`，导致正则/字符串悄悄变形——复杂分析一律写成 .py 文件再运行，别用内联 python。
- 正则匹配字面 `\boxed` 必须写 `r"\\boxed"`：单个 `\b` 是词边界断言，在 `\boxed` 文本里永不匹配（math_env 踩过）。
- transformers 5.12 的 `TrainingArguments` 无 `group_by_length`。
- 脚本是包内模块：`python -m scripts.tools.check_mask` 这类命令必须在**仓库根**运行（CWD 相对路径 data/logs/outputs 都以仓库根为基准）。
