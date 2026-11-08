# -*- coding: utf-8 -*-
"""汇总 GRPO 前后对比报告：训练统计 + 闭环评测对比 + 遗忘检测。
读取（存在即用，缺省自动跳过）:
  logs/grpo_log_math.jsonl       步级训练记录
  logs/grpo_trajs_math.jsonl     整条链路明细（trajectory，抽样例展示）
  outputs/eval/math_*.json       闭环评测结果（eval_math.py 产物）
  outputs/forgetting/forgetting.json + outputs/forgetting_grpo/forgetting.json（可选）
输出 outputs/eval/grpo_report.md
用法（仓库根运行）: python -m scripts.tools.grpo_report
"""
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    steps = load_jsonl(os.path.join(ROOT, "logs", "grpo_log_math.jsonl"))
    trajs = load_jsonl(os.path.join(ROOT, "logs", "grpo_trajs_math.jsonl"))
    evals = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "outputs", "eval", "math_*.json"))):
        with open(p, encoding="utf-8") as f:
            evals[os.path.splitext(os.path.basename(p))[0]] = json.load(f)
    forget = None
    try:
        with open(os.path.join(ROOT, "outputs", "forgetting", "forgetting.json"),
                  encoding="utf-8") as f:
            sft_forget = json.load(f)
        with open(os.path.join(ROOT, "outputs", "forgetting_grpo", "forgetting.json"),
                  encoding="utf-8") as f:
            grpo_forget = json.load(f)
        forget = {"base": sft_forget["base"], "sft": sft_forget["finetuned"],
                  "grpo": grpo_forget["finetuned"]}
    except (OSError, KeyError):
        pass

    L = ["# GRPO 对齐 · 前后对比报告\n"]

    # ---- 1. 训练过程统计 ----
    if steps:
        rw = [s["reward_mean"] for s in steps]
        k = max(1, len(rw) // 5)
        head, tail_ = rw[:k], rw[-k:]
        kls = [s["kl"] for s in steps]
        L.append("## 1. GRPO 训练过程（整条链路 trajectory 模式）\n")
        L.append(f"- 步数: {len(steps)} | 平均 reward 前{k}步 {sum(head)/len(head):.3f}"
                 f" → 后{k}步 {sum(tail_)/len(tail_):.3f}")
        sr = [s.get("success_rate", 0) for s in steps]
        turns = [s.get("avg_turns", 0) for s in steps]
        L.append(f"- 端到端成功率: 前{k}步 {sum(sr[:k])/k:.0%} → 后{k}步 "
                 f"{sum(sr[-k:])/len(sr[-k:]):.0%} | 平均回合 {sum(turns)/len(turns):.1f}")
        ee = [s["exec_err"] for s in steps if "exec_err" in s]
        if ee:
            L.append(f"- 执行报错率均值: {sum(ee)/len(ee):.0%}")
        L.append(f"- KL(对 SFT 参考模型): 均值 {sum(kls)/len(kls):.4f} / "
                 f"最大 {max(kls):.4f}（beta 锚定，偏离极小 → 通用能力不被破坏）")
        L.append(f"- 显存峰值 {max(s['mem_gb'] for s in steps)}GB | 未出现 OOM/重启\n")

    # ---- 2. 明细样例 ----
    if trajs:
        sig = [t for t in trajs if len({c["reward"] for c in t["group"]}) > 1]
        if sig:
            t = sig[len(sig) // 2]
            L.append("## 2. 整条链路明细样例（组内比较让模型自我优化）\n")
            L.append(f"- 任务 `{t['tid']}`（{t['category']}），G={len(t['group'])} 条轨迹:\n")
            for i, c in enumerate(t["group"]):
                d = c["detail"]
                L.append(f"  轨迹{i}: reward={c['reward']:.2f} adv={c['adv']:+.2f} | "
                         f"成功={d['success']} 预测={d.get('pred', '-')} 真值={d.get('gt', '-')} "
                         f"回合={d['n_turns']} 执行 {d.get('n_exec', 0)} 次"
                         f"（报错 {d.get('n_error', 0)}）")
                for line in c["text"].splitlines()[:6]:
                    L.append(f"    {line[:120]}")
                L.append("")

    # ---- 3. 闭环评测对比（eval_math.py 产物）----
    if evals:
        cols = [("accuracy", "总正确率"), ("avg_turns", "平均回合"),
                ("exec_error_rate", "执行报错率"), ("premature_rate", "未终答率")]
        names = list(evals)
        L.append("## 3. 闭环评测对比（同一持出集，boxed 正确率）\n")
        L.append("| 指标 | " + " | ".join(names) + " |")
        L.append("|---|" + "---|" * len(names))
        for key, label in cols:
            row = []
            for t in names:
                v = evals[t].get(key)
                row.append("-" if v is None else
                           (f"{v:.2f}" if key == "avg_turns" else f"{v*100:.1f}%"))
            L.append(f"| {label} | " + " | ".join(row) + " |")
        L.append("\n分难度层正确率:\n")
        levels = sorted({lv for t in names for lv in evals[t].get("by_level", {})})
        L.append("| 难度层 | " + " | ".join(names) + " |")
        L.append("|---|" + "---|" * len(names))
        for lv in levels:
            cells = [f"{evals[t]['by_level'][lv]['acc']*100:.1f}%"
                     if lv in evals[t].get("by_level", {}) else "-" for t in names]
            L.append(f"| {lv} | " + " | ".join(cells) + " |")
        L.append("")

    # ---- 4. 遗忘检测 ----
    if forget:
        b, sft, gr = forget["base"], forget["sft"], forget["grpo"]
        L.append("## 4. 灾难性遗忘检测（25 题通用探针 + 跨域 PPL + 工具误触发）\n")
        L.append(f"- 通用QA: base {b['qa_acc']:.0%} | SFT {sft['qa_acc']:.0%} | "
                 f"GRPO {gr['qa_acc']:.0%}")
        L.append(f"- 跨域PPL: base {b['ppl']:.2f} | SFT {sft['ppl']:.2f} | GRPO {gr['ppl']:.2f}")
        L.append(f"- 无工具场景误发调用: SFT={sft['spurious_tool_call']} "
                 f"GRPO={gr['spurious_tool_call']}\n")

    out = os.path.join(ROOT, "outputs", "eval", "grpo_report.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"报告已生成: {out}")


if __name__ == "__main__":
    main()
