# -*- coding: utf-8 -*-
"""智能体闭环：工具沙箱 + 真值回放环境 + 多轮 rollout + 整条链路打分

GRPO 训练（train_grpo.py --mode trajectory）、端到端评估（eval_agent.py --mode loop）
与交互演示（demo_inference.py）共用本模块，保证三处"环境行为 + 打分口径"一致。

分层设计：
  1. sandbox_exec(name, args)：无真值的确定性 mock 工具。同一 (工具名, 参数) 永远返回
     同一结果（哈希播种），错误语义与 build_dataset.py 的合成数据一致
     （city_not_found / file_not_found / invalid_parameter / invalid_expression ...）。
     安全红线：run_python 一律拒绝执行——模型生成的代码绝不落地运行。
  2. TaskEnv(task)：回放式环境。模型调用 (工具名, 参数) 命中真值轨迹里尚未使用的调用时，
     返回数据里记录的返回（保证事实性数值与真值终答可对齐）；否则退回 sandbox_exec。
     GT 中首条 user 之后的所有 user 消息按顺序回放，充当用户模拟器（澄清/补充信息场景）。
  3. rollout(...)：从首条 user 消息出发，模型自己走完整条链路——调用工具、真的拿到返回、
     继续推理，直到给出最终回答（或轮数/长度预算用尽）。
  4. score_trajectory(...)：整条链路打分 = 调用序列覆盖(F1) + 参数软匹配 + 最终回答事实
     命中 + 澄清回合 + 格式门槛，全部复用 rewards.py 的同一把尺子。
"""
import datetime
import fnmatch
import hashlib
import json
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "data"))

from build_dataset import (CITIES, CURRENCIES, RESTAURANTS, TOOL_DEFS,  # noqa: E402
                           WEATHER_CONDITIONS)
# 与训练/评估同一套解析、归一化、参数执行验证（下划线函数是同口径复用的内部工具）
from rewards import (_fact_hit_rate, _len_score, _match_arg,  # noqa: E402
                     arg_soft_score, has_call, parse_all_calls, safe_eval)

# 与 train_qlora.CHAT_TMPL_KW 一致；此处定义而不是 import，避免把训练栈拖进推理路径
CHAT_TMPL_KW = {"enable_thinking": False}
PIPE = chr(124)  # 成对竖线的特殊 token 一律拼接构造，字面量会破坏本环境的序列化
IM_END = "<" + PIPE + "im_end" + PIPE + ">"

_KNOWN_CITIES = list(CITIES) + ["大理"]
_RATES = {code: rate for _, code, rate in CURRENCIES}
_RATES["CNY"] = 1.0
_ISPS = ["中国电信", "中国联通", "中国移动"]
_VFS = {
    "~/documents": ["report_2026.md", "todo.md", "notes.md"],
    "~/workspace": ["data_2026.csv", "README.md", "main.py"],
}
_TRANSLATIONS = {
    "Hello, how can I help you today?": "你好，今天我能为你做些什么？",
    "The meeting is scheduled for next Monday.": "会议定于下周一举行。",
    "Thank you for your support.": "感谢您的支持。",
    "Please review the attached report.": "请查收附件报告。",
}


def _seed(*parts):
    raw = "|".join(p if isinstance(p, str) else json.dumps(p, ensure_ascii=False, sort_keys=True)
                   for p in parts)
    return int(hashlib.md5(raw.encode("utf-8")).hexdigest()[:8], 16)


def _ok(obj):
    return json.dumps(obj, ensure_ascii=False)


def _err(msg):
    return json.dumps({"error": msg}, ensure_ascii=False)


def _missing_args(name, args):
    req = TOOL_DEFS[name]["function"]["parameters"].get("required", [])
    return [k for k in req if k not in args or args[k] in ("", None)]


def sandbox_exec(name, args):
    """确定性 mock 工具：同一 (name, args) 永远返回同一结果。
    用于真值回放未命中时的兜底，以及交互演示（无真值场景）。"""
    if not isinstance(args, dict):
        return _err("invalid_parameter: arguments 必须是 JSON 对象")
    if name not in TOOL_DEFS:
        return _err(f"unknown_tool: 未注册的工具「{name}」")
    if name == "run_python":
        return _err("sandbox_disabled: 本地 mock 环境不执行模型生成的代码")
    miss = _missing_args(name, args)
    if miss:
        return _err(f"invalid_parameter: {'、'.join(miss)} 不能为空"
                    + ("，格式应为 YYYY-MM-DD" if "date" in miss else ""))
    s = _seed(name, args)

    if name == "get_current_weather":
        city = str(args["city"]).strip()
        if city not in _KNOWN_CITIES:
            return _err(f"city_not_found: 未找到城市「{city}」，请检查城市名称")
        temp = s % 40 - 5
        unit = args.get("unit", "celsius")
        if unit == "fahrenheit":
            temp = round(temp * 9 / 5 + 32)
        return _ok({"city": city, "temperature": temp, "unit": unit,
                    "condition": WEATHER_CONDITIONS[s % len(WEATHER_CONDITIONS)],
                    "humidity": 30 + s % 61})

    if name == "get_weather_forecast":
        city = str(args["city"]).strip()
        if city not in _KNOWN_CITIES:
            return _err(f"city_not_found: 未找到城市「{city}」，请检查城市名称")
        days = max(1, min(int(args.get("days") or 7), 7))
        base = datetime.date(2026, 9, 19)
        fc = [{"date": (base + datetime.timedelta(days=i)).strftime("%Y-%m-%d"),
               "condition": WEATHER_CONDITIONS[(s + i) % len(WEATHER_CONDITIONS)],
               "temp_low": 8 + (s + i) % 12, "temp_high": 16 + (s + i) % 15}
              for i in range(days)]
        return _ok({"city": city, "forecast": fc})

    if name == "calculator":
        v = safe_eval(str(args["expression"]))
        if v is None:
            return _err("invalid_expression: 表达式无法求值，请使用四则运算/pow(a,b) 或 **")
        if v == int(v):
            v = int(v)
        return _ok({"expression": args["expression"], "result": v})

    if name == "get_exchange_rate":
        b, t = str(args["base_currency"]).upper(), str(args["target_currency"]).upper()
        if b not in _RATES or t not in _RATES:
            return _err(f"currency_not_found: 不支持的货币代码 {b}/{t}")
        return _ok({"base": b, "target": t, "rate": round(_RATES[b] / _RATES[t], 6),
                    "update_time": "2026-09-18 10:00"})

    if name == "get_stock_price":
        return _ok({"symbol": args["symbol"], "name": args["symbol"],
                    "price": round(10 + (s % 170000) / 100, 2),
                    "change_percent": round((s % 1000) / 100 - 5, 2)})

    if name == "get_current_time":
        return _ok({"datetime": "2026-09-18 16:42:10",
                    "timezone": args.get("timezone", "Asia/Shanghai")})

    if name == "search_web":
        q = args["query"]
        return _ok({"results": [{"title": f"{q}——相关资讯{i}", "url": f"https://example.com/{s % 9999}/{i}",
                                 "snippet": f"关于「{q}」的第 {i} 条摘要，内容稳定可复现。"}
                                for i in range(1, int(args.get("top_k") or 3) + 1)]})

    if name == "send_email":
        return _ok({"success": True, "message_id": f"mail_{s % 90000 + 10000}"})

    if name == "create_calendar_event":
        return _ok({"success": True, "event_id": f"evt_{s % 90000 + 10000}"})

    if name == "query_flight":
        return _ok({"flights": [{"flight_no": f"CA{s % 9000 + 1000}",
                                 "departure_time": f"{6 + s % 15:02d}:{['00', '20', '30', '40', '50'][s % 5]}",
                                 "arrival_time": "12:30", "price": 500 + s % 2000,
                                 "punctuality": "准点率92%"}]})

    if name == "book_hotel":
        nights = int(args.get("nights") or 1)
        return _ok({"success": True,
                    "hotel_name": RESTAURANTS[s % len(RESTAURANTS)].replace("餐厅", "") + "大酒店",
                    "order_id": f"BK{s % 900000 + 100000}",
                    "total_price": (300 + s % 400) * max(1, nights)})

    if name == "translate_text":
        text = str(args["text"])
        return _ok({"translated_text": _TRANSLATIONS.get(text, text)})

    if name == "read_file":
        path = str(args["path"]).strip()
        for f in _VFS.get(os.path.dirname(path), []):
            if path == f"{os.path.dirname(path)}/{f}":
                return _ok({"path": path, "content": f"# 文件 {f}\n（mock 内容，行号校验用）\n"
                                                      f"第 2 行：本文件由确定性沙箱提供，内容与调用参数绑定。"})
        if fnmatch.fnmatch(os.path.basename(path), "report_*.md"):
            return _ok({"path": path, "content": f"# 2026年{s % 8 + 1}月销售报告\n"
                                                 f"本月销售额：{120 + s % 860} 万元\n"
                                                 f"同比增长：{5 + s % 26}%"})
        return _err(f"file_not_found: {path} 不存在")

    if name == "write_file":
        return _ok({"success": True, "path": args["path"],
                    "bytes": len(str(args["content"]).encode("utf-8"))})

    if name == "list_directory":
        path = str(args["path"]).strip()
        files = _VFS.get(path)
        if files is None:
            return _err(f"dir_not_found: 目录 {path} 不存在")
        pat = args.get("pattern")
        if pat:
            files = [f for f in files if fnmatch.fnmatch(f, str(pat))]
        return _ok({"path": path, "files": files})

    if name == "get_ip_geo":
        return _ok({"ip": args["ip"], "country": "中国",
                    "city": CITIES[s % len(CITIES)], "isp": _ISPS[s % len(_ISPS)]})

    if name == "recommend_restaurant":
        return _ok({"city": args["city"], "cuisine": args["cuisine"],
                    "name": RESTAURANTS[s % len(RESTAURANTS)],
                    "rating": round(4.0 + (s % 10) / 10, 1),
                    "address": f"{args['city']}高新区天府大道{s % 999 + 1}号"})

    if name == "get_traffic_route":
        return _ok({"origin": args["origin"], "destination": args["destination"],
                    "mode": args["mode"], "duration_minutes": 10 + s % 50,
                    "distance_km": round(2 + (s % 300) / 10, 1)})

    if name == "get_news":
        cat = args["category"]
        return _ok({"category": cat, "items": [
            {"title": f"{cat}要闻{i}：市场动态平稳", "time": "2026-09-18"}
            for i in range(1, int(args.get("count") or 3) + 1)]})

    return _err(f"not_implemented: 工具「{name}」暂无 mock 实现")


def _replay_match(name, gt_args, pred_args):
    """模型调用是否与真值调用等价：GT 给出的参数全部对上（必填必须出现；可选参数
    缺失按 tools 语义可容忍），数值做执行验证（与 rewards._match_arg 同口径）。"""
    if not isinstance(pred_args, dict):
        return False
    req = TOOL_DEFS.get(name, {}).get("function", {}).get("parameters", {}).get("required", [])
    for k, v in gt_args.items():
        if k not in pred_args:
            if k in req:
                return False
            continue
        if _match_arg(v, pred_args[k]) != 1.0:
            return False
    return True


class TaskEnv:
    """回放式任务环境。GT 调用命中 → 返回记录的返回；未命中 → sandbox_exec。"""

    def __init__(self, task):
        self.task = task
        self.tools = [TOOL_DEFS[t] for t in task["tools"]]
        self.msgs = task["messages"]
        # 把 GT 里 assistant.tool_calls 与其后的 tool 返回配对成调用池（并行调用按序配对）
        self.pool = []
        pending = []
        for m in self.msgs:
            if m["role"] == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    f = tc["function"]
                    pending.append({"name": f["name"], "args": f.get("arguments") or {}})
            elif m["role"] == "tool" and pending:
                rec = pending.pop(0)
                rec["name"] = m.get("name") or rec["name"]
                rec["result"] = m["content"]
                rec["used"] = False
                self.pool.append(rec)
        self.replayed = 0  # 命中真值回放的调用数（= 走了 GT 路径多少步）
        # 用户模拟器台词：首条 user 之后的所有 user 消息
        self.user_followups = []
        seen_user = False
        for m in self.msgs:
            if m["role"] == "user":
                if seen_user:
                    self.user_followups.append(m)
                seen_user = True

    def initial_msgs(self):
        """任务初始对话：system + 首条 user。"""
        out = []
        for m in self.msgs:
            out.append({"role": m["role"], "content": m.get("content", "")})
            if m["role"] == "user":
                break
        return out

    def execute(self, name, args):
        for rec in self.pool:
            if rec["used"] or rec["name"] != name:
                continue
            if _replay_match(name, rec["args"], args):
                rec["used"] = True
                self.replayed += 1
                return rec["result"]
        return sandbox_exec(name, args)

    def gt_calls(self):
        return [(r["name"], r["args"]) for r in self.pool]

    def gt_final(self):
        for m in reversed(self.msgs):
            if m["role"] == "assistant" and not m.get("tool_calls"):
                return m.get("content", "")
        return ""


def render_ids(tok, msgs, tools):
    return tok.apply_chat_template(msgs, tools=tools, tokenize=True,
                                   add_generation_prompt=True, return_dict=False,
                                   **CHAT_TMPL_KW)


@torch.no_grad()
def generate_once(model, tok, enc, max_new_tokens, temperature, top_p, do_sample, grad_ckpt):
    """生成期间临时切到推理模式（采样必须开 kv cache / 关梯度检查点），结束恢复。"""
    was_training = model.training
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    try:
        return model.generate(
            enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=True)
    finally:
        model.config.use_cache = False
        if was_training:
            model.train()
            if grad_ckpt:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})


def rollout(model, tok, env, *, max_turns, max_new_tokens, temperature, top_p,
            max_len, do_sample=True, grad_ckpt=True):
    """完整链路 rollout：模型自己走（调用→真实返回→继续→最终回答）。

    返回 {"turns": [...], "ended": final|limit|budget}；每个 turn 记录生成时的
    prompt_ids/gen_ids（loss 用）与解析出的调用。
    说话（无调用）且 GT 还有未回放的用户追问时，交付追问继续（用户模拟器）。
    """
    msgs = env.initial_msgs()
    init = env.task.get("_init_ids")
    prompt = list(init) if (init is not None and len(msgs) == 2) else render_ids(tok, msgs, env.tools)
    turns, ended, fu = [], "limit", 0

    for _ in range(max_turns):
        if len(prompt) + 8 > max_len:
            ended = "budget"
            break
        enc = torch.tensor([prompt], device=model.device)
        out = generate_once(model, tok, enc, min(max_new_tokens, max_len - len(prompt)),
                            temperature, top_p, do_sample, grad_ckpt)
        gen_ids = out[0][len(prompt):].tolist()
        text = tok.decode(gen_ids, skip_special_tokens=False).split(IM_END)[0]
        raw = parse_all_calls(text)
        valid = [(n, a) for n, a, ok in raw if ok]
        bad = sum(1 for _, _, ok in raw if not ok) \
            + (1 if (has_call(text) and not raw) else 0)

        if valid:
            results = [env.execute(n, a) for n, a in valid]
            turns.append({"kind": "call", "prompt_ids": prompt, "gen_ids": gen_ids,
                          "text": text, "call_list": valid, "results": results,
                          "fmt_ok": bad == 0, "followup": False})
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function",
                                         "function": {"name": n, "arguments": a}}
                                        for n, a in valid]})
            for n, res in zip([n for n, _ in valid], results):
                msgs.append({"role": "tool", "name": n, "content": res})
            prompt = render_ids(tok, msgs, env.tools)
            continue

        terminal = fu >= len(env.user_followups)
        turns.append({"kind": "speak", "prompt_ids": prompt, "gen_ids": gen_ids,
                      "text": text, "call_list": [], "results": [],
                      "fmt_ok": bad == 0, "followup": not terminal})
        msgs.append({"role": "assistant", "content": text})
        if terminal:
            ended = "final"
            break
        msgs.append({"role": "user", "content": env.user_followups[fu].get("content", "")})
        fu += 1
        prompt = render_ids(tok, msgs, env.tools)

    return {"turns": turns, "ended": ended, "n_followup_used": fu}


def format_trajectory(env, ro, max_chars=1600):
    """把一条 rollout 打印成可读的链路文本（审计日志用）。"""
    lines = []
    for t in ro["turns"]:
        if t["kind"] == "call":
            for (n, a), res in zip(t["call_list"], t["results"]):
                lines.append(f"→ {n}({json.dumps(a, ensure_ascii=False)})")
                lines.append(f"  ⇐ {res[:220]}")
        else:
            tag = "澄清" if t["followup"] else "最终回答"
            lines.append(f"[{tag}] {t['text'][:300]}")
    txt = "\n".join(lines)
    return txt[:max_chars]


def score_trajectory(ro, env):
    """整条链路打分（0~1）与明细。

    base = 0.55×调用覆盖 + 0.45×最终回答
      - 调用覆盖（乘性）：命中真值调用数占比 × (0.4×名称集合F1 + 0.6×参数软匹配均分)
        ——漏掉一半链路直接砍半，不是只扣 F1 的分
      - 最终回答：事实数字命中率（无数字真值则看长度合理性）；误发调用记 0
    乘性惩罚（链路上的关键纪律项，不满足直接打折）：
      - 该澄清未澄清（GT 有用户追问但模型从未开口）：×0.5
      - 无需工具的任务却误发调用：×0.5
      - 格式门槛：× (0.5 + 0.5×可解析回合占比)
    返回 (reward, detail)；detail.success 为端到端任务成功（评测口径）。"""
    gt_calls = env.gt_calls()
    pred_calls = [c for t in ro["turns"] for c in t["call_list"]]
    used = [False] * len(gt_calls)
    matched = []
    for pname, pargs in pred_calls:
        best_j, best_s = -1, -1.0
        for j, (gname, gargs) in enumerate(gt_calls):
            if used[j] or gname != pname:
                continue
            s = arg_soft_score(pname, gargs, pargs, TOOL_DEFS)
            if s > best_s:
                best_j, best_s = j, s
        if best_j >= 0:
            used[best_j] = True
            matched.append({"name": pname, "arg": round(best_s, 3)})
    n_gt, n_pred = len(gt_calls), len(pred_calls)
    name_f1 = 2 * len(matched) / (n_gt + n_pred) if (n_gt + n_pred) else 1.0
    arg_avg = sum(m["arg"] for m in matched) / len(matched) if matched else 0.0
    if n_gt:
        coverage = len(matched) / n_gt
        call_score = coverage * (0.4 * name_f1 + 0.6 * arg_avg)
    else:
        call_score = 1.0 if n_pred == 0 else 0.0

    term = ro["turns"][-1] if (ro["ended"] == "final" and ro["turns"]) else None
    hit = None
    if term is None or has_call(term["text"]):
        answer_score = 0.0
    else:
        hit = _fact_hit_rate(env.gt_final(), term["text"])
        answer_score = (0.4 + 0.6 * hit) if hit is not None \
            else (0.8 + 0.2 * _len_score(term["text"]))

    clarify_turns = [t for t in ro["turns"] if t["kind"] == "speak" and t is not term]
    clarify_scores = [0.0 if has_call(t["text"]) else 0.8 + 0.2 * _len_score(t["text"])
                      for t in clarify_turns]
    expects_clarify = len(env.user_followups) > 0
    if expects_clarify:
        clarify_score = sum(clarify_scores) / len(clarify_scores) if clarify_scores else 0.0
    else:
        clarify_score = 1.0

    turn_ok = [t["fmt_ok"] for t in ro["turns"]]
    fmt_rate = sum(turn_ok) / len(turn_ok) if turn_ok else 0.0
    reward = (0.55 * call_score + 0.45 * answer_score)
    if expects_clarify and not clarify_scores:
        reward *= 0.5            # 参数不足应先澄清，model 直接猜/干完活也打折
    if n_gt == 0 and n_pred > 0:
        reward *= 0.5            # 无需工具的任务误发调用（工具滥用）
    reward *= (0.5 + 0.5 * fmt_rate)

    answer_ok = term is not None and not has_call(term["text"]) \
        and (hit is None or hit >= 0.6)
    success = bool(env.replayed == n_gt and n_pred == n_gt and answer_ok
                   and fmt_rate == 1.0 and (not expects_clarify or bool(clarify_scores)))
    detail = {"success": success, "ended": ro["ended"], "n_turns": len(ro["turns"]),
              "n_gt_calls": n_gt, "n_pred_calls": n_pred, "replayed": env.replayed,
              "coverage": round(len(matched) / n_gt, 3) if n_gt else 1.0,
              "name_f1": round(name_f1, 3), "arg_avg": round(arg_avg, 3),
              "call_score": round(call_score, 3), "answer_score": round(answer_score, 3),
              "clarify_score": round(clarify_score, 3), "fmt_rate": round(fmt_rate, 3),
              "pairs": matched}
    return reward, detail


def load_tasks(path, tok, max_prompt_len):
    """读取完整任务（train.jsonl 格式），预渲染初始 prompt 并缓存（行尾写入 _init_ids）。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            env = TaskEnv(t)
            ids = render_ids(tok, env.initial_msgs(), env.tools)
            if len(ids) <= max_prompt_len:
                t["_init_ids"] = ids
                out.append(t)
    return out
