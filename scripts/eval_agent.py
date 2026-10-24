# -*- coding: utf-8 -*-
"""
智能体能力评估（两种口径，同一把尺子）

--mode loop（默认联测口径）端到端闭环：模型从首条 user 消息出发自己走完整条链路
  （调用工具→真的拿到返回→继续→最终回答，工具环境与训练同源 agent_env.TaskEnv），
  按整条轨迹判定任务成功：
    - task_success_rate  端到端任务成功率（调用全覆盖且参数匹配+最终回答事实达标+无违规）
    - avg_turns          平均回合数（模型出手次数）
    - call_f1 / arg_avg  调用序列覆盖 F1 / 命中对参数软匹配均分
    - answer_score       最终回答得分（事实命中）
    - fmt_valid_rate     可解析回合占比（格式门槛）
    - premature_rate     没给出最终回答就中断（轮数/长度用尽）的比例
--mode step（旧口径）对每条轨迹逐轮步进：每次把真值历史喂给模型只生成一轮：
    - turn_acc      轮次级正确率（工具选择+参数全对，或无需工具时不调用）
    - tool_choice   工具选择正确率（含"该调用时调用、不该时不调用"）
    - arg_acc       在工具选择正确的前提下参数完全匹配率
    - fmt_valid     工具调用 XML 结构可解析率
    - fact_hit      最终回答关键事实命中率

用法:
  python scripts/eval_agent.py --model models/Qwen3.5-4B --mode both
  python scripts/eval_agent.py --model models/Qwen3.5-4B --adapter outputs/qlora-adapter/final
"""
import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "data"))
from build_dataset import TOOL_DEFS  # noqa: E402
# 与 GRPO 训练用同一套解析/归一化（rewards.py）与同一套闭环环境（agent_env.py），
# 保证训练与离线评估口径一致
from rewards import has_call, norm, parse_all_calls  # noqa: E402
from agent_env import TaskEnv, format_trajectory, load_tasks, rollout, score_trajectory  # noqa: E402
from train_qlora import CHAT_TMPL_KW  # noqa: E402

NUM_RE = re.compile(r"-?\d+\.?\d*")


def check_args(gt_args, pred_args, tool_name):
    """预测参数必须覆盖所有必填参数且值匹配；可选参数缺失可容忍"""
    required = TOOL_DEFS.get(tool_name, {}).get("function", {}).get("parameters", {}).get("required", [])
    if not isinstance(pred_args, dict):
        return False
    for k, v in gt_args.items():
        if k not in pred_args:
            if k in required:
                return False
            continue  # 可选参数缺失：工具会使用默认值，视为可接受
        if norm(pred_args[k]) != norm(v):
            return False
    return True


def parse_first_call(text):
    """解析第一个 <tool_call> 块（Qwen3.5 XML 格式）。
    返回 (name, args)；结构不可解析时返回 ("INVALID", {})；无调用返回 None。"""
    calls = parse_all_calls(text)
    if not calls:
        return None
    name, args, ok = calls[0]
    return (name, args) if ok else ("INVALID", {})


@torch.no_grad()
def evaluate(model, tok, samples, max_new_tokens=256, max_samples=None, verbose=False):
    n_turn = n_turn_ok = 0
    n_choice = n_choice_ok = 0
    n_with_tool = n_fmt_valid = 0
    n_args = n_args_ok = 0
    n_final = n_final_ok = 0
    details = []

    if max_samples:
        samples = samples[:max_samples]

    for si, s in enumerate(samples):
        tools = [TOOL_DEFS[t] for t in s["tools"]]
        msgs = s["messages"]
        sample_ok = True
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            prompt_ids = tok.apply_chat_template(msgs[:i], tools=tools, tokenize=True,
                                                 add_generation_prompt=True,
                                                 return_dict=False, **CHAT_TMPL_KW)
            input_ids = torch.tensor([prompt_ids], device=model.device)
            out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=False)
            gen = gen.split("<|im_end|>")[0].strip()

            if m.get("tool_calls"):  # 该轮应调用工具
                gt_tc = m["tool_calls"][0]["function"]
                n_with_tool += 1
                parsed = parse_first_call(gen)
                fmt_ok = parsed is not None and parsed[0] != "INVALID"
                if fmt_ok:
                    n_fmt_valid += 1
                pred_name = parsed[0] if parsed else None
                choice_ok = pred_name == gt_tc["name"]
                if choice_ok:
                    n_choice_ok += 1
                args_ok = choice_ok and check_args(gt_tc["arguments"], parsed[1], gt_tc["name"])
                if choice_ok:
                    n_args += 1
                    if args_ok:
                        n_args_ok += 1
                turn_ok = choice_ok and args_ok
                if verbose and not turn_ok:
                    print(f"  [miss {s['id']} turn{i}] gt={gt_tc['name']} {gt_tc['arguments']}"
                          f" | pred={str(parsed)[:200]}")
            else:  # 最终回答轮：不应调用工具
                n_choice += 1
                no_call = not has_call(gen)
                if no_call:
                    n_choice_ok += 1
                gt_facts = set(NUM_RE.findall(m["content"]))
                gen_facts = set(NUM_RE.findall(gen))
                hits = sum(1 for f in gt_facts if any(g.endswith(f) or f.endswith(g) for g in gen_facts))
                fact_ok = no_call and (not gt_facts or hits / len(gt_facts) >= 0.6)
                if fact_ok:
                    n_final_ok += 1
                n_final += 1
                turn_ok = fact_ok
                if verbose and not turn_ok:
                    print(f"  [miss {s['id']} turn{i}] gt={m['content'][:80]} | gen={gen[:120]}")
            n_turn += 1
            if turn_ok:
                n_turn_ok += 1
            else:
                sample_ok = False
        details.append({"id": s["id"], "category": s["category"], "all_turns_ok": sample_ok})

    res = {
        "samples": len(samples),
        "turn_acc": round(n_turn_ok / max(n_turn, 1), 4),
        # 工具选择正确率统一口径：该调用时选对工具 + 不该调用时未调用
        "tool_choice_acc": round(n_choice_ok / max(n_with_tool + n_choice, 1), 4),
        "fmt_valid_rate": round(n_fmt_valid / max(n_with_tool, 1), 4),
        "arg_acc_when_chosen": round(n_args_ok / max(n_args, 1), 4),
        "final_answer_acc": round(n_final_ok / max(n_final, 1), 4),
        "raw": {"turns": n_turn, "turns_ok": n_turn_ok, "with_tool": n_with_tool,
                "final": n_final, "final_ok": n_final_ok},
    }
    return res, details


def evaluate_loop(model, tok, tasks, max_turns=4, max_new_tokens=256, max_len=2560,
                  max_samples=None, verbose=False):
    """端到端闭环评估：模型从首条 user 出发自己走完整条链路（贪心解码），
    环境/打分与 GRPO 训练完全同源（agent_env.TaskEnv + score_trajectory）。"""
    if max_samples:
        tasks = tasks[:max_samples]
    per_cat = {}
    details = []
    for t in tasks:
        env = TaskEnv(t)
        ro = rollout(model, tok, env, max_turns=max_turns, max_new_tokens=max_new_tokens,
                     temperature=1.0, top_p=1.0, max_len=max_len,
                     do_sample=False, grad_ckpt=False)
        r, det = score_trajectory(ro, env)
        details.append({"id": t["id"], "category": t["category"],
                        "reward": round(r, 4), "turns": det["n_turns"], **det})
        c = per_cat.setdefault(t["category"], {"n": 0, "ok": 0})
        c["n"] += 1
        c["ok"] += int(det["success"])
        if verbose and not det["success"]:
            print(f"  [fail {t['id']} ({t['category']})] reward={r:.2f} "
                  f"cov={det['coverage']:.0%} ans={det['answer_score']:.2f} "
                  f"ended={det['ended']} turns={det['n_turns']}")
            print("   " + format_trajectory(env, ro, 700).replace("\n", "\n   "))

    n = max(len(details), 1)
    res = {
        "samples": len(details),
        "task_success_rate": round(sum(d["success"] for d in details) / n, 4),
        "avg_turns": round(sum(d["turns"] for d in details) / n, 2),
        "avg_reward": round(sum(d["reward"] for d in details) / n, 4),
        "call_f1": round(sum(d["name_f1"] for d in details) / n, 4),
        "arg_avg": round(sum(d["arg_avg"] for d in details) / n, 4),
        "answer_score": round(sum(d["answer_score"] for d in details) / n, 4),
        "fmt_valid_rate": round(sum(d["fmt_rate"] for d in details) / n, 4),
        "premature_rate": round(sum(1 for d in details if d["ended"] != "final") / n, 4),
        "by_category": {c: {"n": v["n"], "success": v["ok"],
                            "rate": round(v["ok"] / v["n"], 3)}
                        for c, v in sorted(per_cat.items())},
    }
    return res, details


def load_model(model_path, adapter=None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=True)
    if adapter:
        from peft import PeftModel
        # 4bit 量化底座上直接挂载 adapter 推理（不合并，量化权重不可 merge）
        model = PeftModel.from_pretrained(model, adapter)
        print(f"已加载 adapter: {adapter}")
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/Qwen3.5-4B")
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--data", type=str, default="data/val.jsonl")
    ap.add_argument("--mode", type=str, default="both", choices=["both", "loop", "step"],
                    help="loop=端到端闭环；step=逐轮步进（旧口径）；both=都跑（默认）")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=4, help="loop 模式单条轨迹最多回合数")
    ap.add_argument("--max-len", type=int, default=2560, help="loop 模式单条轨迹 token 上限")
    ap.add_argument("--out", type=str, default="outputs/eval")
    ap.add_argument("--tag", type=str, default="model")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]

    model, tok = load_model(args.model, args.adapter)

    loop_res = loop_details = None
    if args.mode in ("both", "loop"):
        tasks = load_tasks(args.data, tok, 900)
        if args.max_samples:
            tasks = tasks[:args.max_samples]
        print(f"端到端闭环评估：{len(tasks)} 条验证轨迹（初始 prompt <= 900 tokens）", flush=True)
        loop_res, loop_details = evaluate_loop(model, tok, tasks, args.max_turns,
                                               args.max_new_tokens, args.max_len,
                                               None, args.verbose)

    step_res = step_details = None
    if args.mode in ("both", "step"):
        step_res, step_details = evaluate(model, tok, samples, args.max_new_tokens,
                                          args.max_samples, args.verbose)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"eval_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "mode": args.mode,
                   "results": step_res, "loop": loop_res,
                   "details": step_details or [], "loop_details": loop_details or []},
                  f, ensure_ascii=False, indent=2)

    print(f"\n===== 评估结果 [{args.tag}] =====")
    if loop_res:
        print("-- 端到端闭环（整条链路）--")
        for k, v in loop_res.items():
            if k != "by_category":
                print(f"  {k:22s}: {v}")
        print("  分类别成功率: " + " | ".join(
            f"{c} {v['success']}/{v['n']}" for c, v in loop_res["by_category"].items()))
    if step_res:
        print("-- 逐轮步进（旧口径）--")
        for k, v in step_res.items():
            if k not in ("raw", "samples"):
                print(f"  {k:22s}: {v}")


if __name__ == "__main__":
    main()
