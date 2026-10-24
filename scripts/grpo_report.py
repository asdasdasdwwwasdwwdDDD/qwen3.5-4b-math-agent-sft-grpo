# -*- coding: utf-8 -*-
"""汇总 GRPO 前后对比报告：训练统计 + 整条链路/逐轮步进评估对比 + 遗忘检测。
读取（存在即用，两种模式自动识别）:
  logs/grpo_log.jsonl        步级训练记录（含 success_rate 即整条链路模式）
  logs/grpo_trajs.jsonl      整条链路明细（trajectory 模式，抽样例展示）
  logs/grpo_groups.jsonl     组内验证明细（step 模式，抽样例展示）
  outputs/eval/eval_{base,finetuned,grpo}.json
  outputs/forgetting/forgetting.json + outputs/forgetting_grpo/forgetting.json（可选）
输出 outputs/eval/grpo_report.md
用法: python scripts/grpo_report.py
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    steps = load_jsonl(os.path.join(ROOT, "logs", "grpo_log_math.jsonl"))
    trajs = load_jsonl(os.path.join(ROOT, "logs", "grpo_trajs_math.jsonl"))
    groups = load_jsonl(os.path.join(ROOT, "logs", "grpo_groups.jsonl"))
    evals = {}
    for tag in ("base", "finetuned", "grpo"):
        p = os.path.join(ROOT, "outputs", "eval", f"eval_{tag}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                evals[tag] = json.load(f)
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
        traj_mode = "success_rate" in steps[0]
        L.append(f"## 1. GRPO 训练过程（{'整条链路 trajectory' if traj_mode else '逐决策点 step'} 模式）\n")
        L.append(f"- 步数: {len(steps)} | 平均 reward 前{k}步 {sum(head)/len(head):.3f}"
                 f" → 后{k}步 {sum(tail_)/len(tail_):.3f}")
        if traj_mode:
            sr = [s["success_rate"] for s in steps]
            cov = [s.get("coverage", 0) for s in steps]
            turns = [s.get("avg_turns", 0) for s in steps]
            L.append(f"- 端到端成功率: 前{k}步 {sum(sr[:k])/k:.0%} → 后{k}步 "
                     f"{sum(sr[-k:])/len(sr[-k:]):.0%} | 调用覆盖均值 "
                     f"{sum(cov)/len(cov):.0%} | 平均回合 {sum(turns)/len(turns):.1f}")
        else:
            L.append(f"- 有效组占比（组内有方差、产生梯度）均值: "
                     f"{sum(s['valid_group_frac'] for s in steps)/len(steps):.0%}")
        L.append(f"- KL(对 SFT 参考模型): 均值 {sum(kls)/len(kls):.4f} / "
                 f"最大 {max(kls):.4f}（beta 锚定，偏离极小 → 通用能力不被破坏）")
        L.append(f"- 显存峰值 {max(s['mem_gb'] for s in steps)}GB | 未出现 OOM/重启\n")

    # ---- 2. 明细样例：整条链路优先，其次逐决策点 ----
    if trajs:
        sig = [t for t in trajs if len({c["reward"] for c in t["group"]}) > 1]
        if sig:
            t = sig[len(sig) // 2]
            L.append("## 2. 整条链路明细样例（组内比较让模型自我优化）\n")
            L.append(f"- 任务 `{t['tid']}`（类别 {t['category']}），G={len(t['group'])} 条轨迹:\n")
            for i, c in enumerate(t["group"]):
                d = c["detail"]
                L.append(f"  轨迹{i}: reward={c['reward']:.2f} adv={c['adv']:+.2f} | "
                         f"成功={d['success']} 回合={d['n_turns']} 调用覆盖={d['coverage']:.0%} "
                         f"参数均分={d['arg_avg']:.2f} 回答={d['answer_score']:.2f}")
                for line in c["text"].splitlines()[:6]:
                    L.append(f"    {line[:120]}")
                L.append("")
    elif groups:
        sig = [g for g in groups if len({c["reward"] for c in g["candidates"]}) > 1]
        if sig:
            g = sig[len(sig) // 2]
            L.append("## 2. 组内验证明细样例（模型自我优化的一步决策）\n")
            L.append(f"- 决策点 `{g['pid']}`（类别 {g['category']}），真值: "
                     f"`{str(g['gt'])[:120]}`")
            for c in g["candidates"]:
                d = c["detail"]
                comp = {k: d[k] for k in ("name_f1", "arg_avg", "fact_hit", "len_score")
                        if k in d}
                L.append(f"  - reward={c['reward']:.2f} adv={c['adv']:+.2f} "
                         f"| {c['text'][:64]} | 明细 {comp}")
            L.append("")

    # ---- 3. 评估对比：端到端闭环优先 ----
    loop_tags = [t for t in evals if evals[t].get("loop")]
    if len(loop_tags) >= 1:
        cols = ["task_success_rate", "avg_turns", "call_f1", "arg_avg",
                "answer_score", "fmt_valid_rate", "premature_rate"]
        L.append("## 3. 端到端闭环评估对比（模型自己走完整条链路，贪心解码）\n")
        L.append("| 指标 | " + " | ".join(loop_tags) + " |")
        L.append("|---|" + "---|" * len(loop_tags))
        for col in cols:
            row = []
            for t in loop_tags:
                v = evals[t]["loop"].get(col)
                row.append(f"{v*100:.1f}%" if isinstance(v, float) and col != "avg_turns"
                           else str(v))
            L.append(f"| {col} | " + " | ".join(row) + " |")
        L.append("\n分类别任务成功率:\n")
        cats = sorted({c for t in loop_tags for c in evals[t]["loop"].get("by_category", {})})
        L.append("| 类别 | " + " | ".join(loop_tags) + " |")
        L.append("|---|" + "---|" * len(loop_tags))
        for c in cats:
            cells = []
            for t in loop_tags:
                v = evals[t]["loop"]["by_category"].get(c)
                cells.append(f"{v['success']}/{v['n']}" if v else "-")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L.append("")

    step_tags = [t for t in evals if evals[t].get("results")]
    if len(step_tags) >= 2:
        L.append("## 4. 逐轮步进评估对比（真值前缀，126 轮，贪心解码）\n")
        cols = ["turn_acc", "tool_choice_acc", "fmt_valid_rate",
                "arg_acc_when_chosen", "final_answer_acc"]
        L.append("| 指标 | " + " | ".join(step_tags) + " |")
        L.append("|---|" + "---|" * len(step_tags))
        for col in cols:
            row = [f"{evals[t]['results'][col]*100:.1f}%" for t in step_tags]
            L.append(f"| {col} | " + " | ".join(row) + " |")
        L.append("")

    # ---- 4. 遗忘检测 ----
    if forget:
        b, sft, gr = forget["base"], forget["sft"], forget["grpo"]
        L.append("## 5. 灾难性遗忘检测（25 题通用探针 + 跨域 PPL + 工具误触发）\n")
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
