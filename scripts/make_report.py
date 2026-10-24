# -*- coding: utf-8 -*-
"""汇总 outputs/eval/*.json 生成对比报告 outputs/eval/report.md

支持两种评估口径（eval_agent.py --mode both 会同时写入）：
  - loop 端到端闭环：整条链路由模型自己走完（task_success_rate 等）
  - step 逐轮步进：真值前缀 + 单轮生成（turn_acc 等，旧口径）
"""
import glob
import json
import os

TAGS = {"base": "基座模型", "finetuned": "AdaLoRA 微调", "grpo": "GRPO 对齐"}
LOOP_METRICS = [
    ("task_success_rate", "端到端任务成功率"),
    ("avg_turns", "平均回合数"),
    ("avg_reward", "平均链路奖励"),
    ("call_f1", "调用序列 F1"),
    ("arg_avg", "参数匹配均分"),
    ("answer_score", "最终回答得分"),
    ("fmt_valid_rate", "格式有效率"),
    ("premature_rate", "未完成率（无最终回答）"),
]
STEP_METRICS = [
    ("turn_acc", "轮次全对率"),
    ("tool_choice_acc", "工具选择正确率"),
    ("fmt_valid_rate", "调用格式有效率"),
    ("arg_acc_when_chosen", "参数完全匹配率"),
    ("final_answer_acc", "最终回答正确率"),
]
CATS = ["single_call", "chained", "parallel", "error_retry", "no_tool", "clarify"]

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "eval")
data = {}
for path in glob.glob(os.path.join(out_dir, "eval_*.json")):
    d = json.load(open(path, encoding="utf-8"))
    data[d["tag"]] = d

if not data:
    raise SystemExit("outputs/eval/ 下没有 eval_*.json，请先运行 scripts/eval_agent.py")

_first = list(data.values())[0]
_n = (_first.get("loop") or {}).get("samples") or (_first.get("results") or {}).get("samples")
lines = ["# 智能体能力评估报告", "",
         f"评估集：`data/val.jsonl`（{_n} 条）| 闭环=模型自己走完整条链路，步进=真值前缀逐轮", ""]

loop_tags = [t for t in data if data[t].get("loop")]
if loop_tags:
    lines += ["## 1. 端到端闭环（模型从用户问题出发自己走完整条链路）", "",
              "| 指标 | " + " | ".join(TAGS.get(t, t) for t in loop_tags) + " |",
              "|" + "---|" * (len(loop_tags) + 1)]
    for key, name in LOOP_METRICS:
        row = [name]
        for t in loop_tags:
            v = data[t]["loop"].get(key)
            row.append(f"{v:.1%}" if isinstance(v, float) and key != "avg_turns" else str(v))
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "### 分类别任务成功率（成功数/总数）", "",
              "| 类别 | " + " | ".join(TAGS.get(t, t) for t in loop_tags) + " |",
              "|" + "---|" * (len(loop_tags) + 1)]
    for cat in CATS:
        row = [cat]
        for t in loop_tags:
            v = data[t]["loop"].get("by_category", {}).get(cat)
            row.append(f"{v['success']}/{v['n']}" if v else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

step_tags = [t for t in data if data[t].get("results")]
if step_tags:
    n_turn = data[step_tags[0]]["results"].get("raw", {}).get("turns", "-")
    lines += [f"## 2. 逐轮步进（真值前缀，共 {n_turn} 轮，贪心解码）", "",
              "| 指标 | " + " | ".join(TAGS.get(t, t) for t in step_tags) + " |",
              "|" + "---|" * (len(step_tags) + 1)]
    for key, name in STEP_METRICS:
        row = [name]
        for t in step_tags:
            v = data[t]["results"].get(key)
            row.append(f"{v:.1%}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "### 逐样本轮次全通过率（按类别）", "",
              "| 类别 | " + " | ".join(TAGS.get(t, t) for t in step_tags) + " |",
              "|" + "---|" * (len(step_tags) + 1)]
    for cat in CATS:
        row = [cat]
        for t in step_tags:
            det = [x for x in data[t].get("details", []) if x["category"] == cat]
            row.append(f"{sum(x['all_turns_ok'] for x in det)}/{len(det)}" if det else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

text = "\n".join(lines) + "\n"
path = os.path.join(out_dir, "report.md")
with open(path, "w", encoding="utf-8") as f:
    f.write(text)
print(text)
print("已写入", path)
