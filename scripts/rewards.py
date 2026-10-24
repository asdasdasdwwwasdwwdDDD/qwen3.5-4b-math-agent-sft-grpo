# -*- coding: utf-8 -*-
"""GRPO 组内验证打分器（细粒度规则 + 执行验证）

对一个决策步，给定模型的一个候选补全文本 + 该步真值(assistant 消息)，输出 0~1 连续 reward
及其分解明细。验证信号完全程序化，不依赖外部模型，保证 GRPO 的组内比较客观可复现。

工具调用格式为 Qwen3.5 原生 XML（chat template 渲染，非 Qwen3 的 JSON-in-tool_call）：
    <tool_call>
    <function=get_current_weather>
    <parameter=city>
    北京
    </parameter>
    </function>
    </tool_call>

- 工具调用步(真值含 tool_calls)：
    reward = 格式门槛(0/1) × ( 0.4 × 工具名集合匹配F1 + 0.6 × 命中对的参数软匹配均分 )
    参数软匹配：必填参数命中比例(0.8权重) + 可选参数命中(0.2权重)，逐参数比较时：
      先归一化字符串/数值相等，否则做"执行验证"——把参数当表达式/数值求值，
      等价(如 calculator 的 '2*(30+12)' 与 '84')视为命中。
    多个调用(漏调/多调)按名称贪心配对，集合 F1 = 2*命中/(真值数+预测数)。
- 最终回答/澄清步(真值纯 content)：
    误发  -> reward=0(硬失败)
    否则  有数字真值: 0.4 + 0.6×事实命中率(连续, 不再卡≥60%阈值)
          无数字真值(澄清/口述): 0.8 + 0.2×回答长度合理性

供 train_grpo.py 与调试脚本共用，保证训练与离线评估同一把尺子。
"""
import ast
import json
import operator
import re

# Qwen3.5 XML 工具调用：<tool_call> 内是 <function=名> 包若干 <parameter=名>值</parameter>
CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"
FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)(?:</function>|\Z)", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>\n?(.*?)\n?\s*</parameter>", re.S)
NUM_RE = re.compile(r"-?\d+\.?\d*")

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": lambda x: x ** 0.5, "pow": operator.pow, "abs": abs,
          "round": lambda x, n=0: round(x, int(n)), "min": min, "max": max}


def norm(v):
    """参数值归一化：布尔→小写串，数值(含数字串)→4 位浮点，其余→排序 JSON 串。"""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    if isinstance(v, str):
        s = v.strip().lower()
        try:
            return round(float(s), 4)
        except ValueError:
            return s
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def safe_eval(expr):
    """AST 白名单计算器：只允许四则/幂/取余/一元/括号与少量函数，禁属性/下标/未知名字。
    成功返回 float，否则返回 None（不抛异常）。"""
    if not isinstance(expr, str):
        return None

    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            r = _ev(node.right)
            if node.op.__class__ in (ast.Pow,) and abs(r) > 12:  # 防 9**9**9 DoS
                raise ValueError("exp_overflow")
            return _BIN[type(node.op)](_ev(node.left), r)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](_ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            return _FUNCS[node.func.id](*[_ev(a) for a in node.args])
        raise ValueError("disallowed_node")

    try:
        tree = ast.parse(expr.strip(), mode="eval")
        val = _ev(tree)
        return float(val) if isinstance(val, (int, float)) else None
    except Exception:
        return None


def _num_of(v):
    """把一个参数值尽量解析为数值(字符串含表达式则 safe_eval)，失败返回 None。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        try:
            return float(s)
        except ValueError:
            return safe_eval(s)
    return None


def _match_arg(gt, pred):
    """单参数匹配：0 不中 / 1 命中。归一化相等，或两侧可数值化且近似相等(执行验证)。"""
    if pred is None:
        return 0.0
    if norm(pred) == norm(gt):
        return 1.0
    g, p = _num_of(gt), _num_of(pred)
    if g is not None and p is not None and abs(g - p) < 1e-4:
        return 1.0
    return 0.0


def _parse_value(raw):
    """参数值反序列化：能 json 解析的（数字/布尔/数组/对象）转成对应类型，
    否则保持字符串（模板对字符串参数直接原样输出）。"""
    s = raw.strip()
    try:
        return json.loads(s)
    except Exception:
        return s


def has_call(text):
    """文本中是否出现工具调用标记（不论结构是否合法）——用于"误发调用"判定。"""
    return CALL_OPEN in (text or "")


def parse_all_calls(text):
    """解析生成文本中的全部 <tool_call> 块（Qwen3.5 XML 格式）。

    返回 [(name, arguments, ok)]：结构完整的块给 (函数名, 参数字典, True)；
    出现 <tool_call> 标记但内部不符合 <function=...>/<parameter=...> 结构的块
    给 (None, {}, False)，由调用方的"格式门槛"判 0。"""
    text = text or ""
    calls = []
    pos = 0
    while True:
        i = text.find(CALL_OPEN, pos)
        if i < 0:
            break
        j = text.find(CALL_CLOSE, i)
        seg = text[i + len(CALL_OPEN): j if j >= 0 else len(text)]
        m = FUNC_RE.search(seg)
        if m:
            args = {}
            for k, v in PARAM_RE.findall(m.group(2) or ""):
                args[k] = _parse_value(v)
            calls.append((m.group(1), args, True))
        else:
            calls.append((None, {}, False))
        pos = j + len(CALL_CLOSE) if j >= 0 else len(text)
    return calls


def _gt_calls(gt_msg):
    out = []
    for tc in gt_msg.get("tool_calls") or []:
        f = tc.get("function", {})
        out.append((f.get("name"), f.get("arguments") or {}))
    return out


def arg_soft_score(tool_name, gt_args, pred_args, tool_defs):
    """参数软匹配 0~1：必填命中比例×0.8 + 可选命中×0.2（仅对真值给过的参数要求）。
    无参工具（required 与 gt_args 均空）给 0.85 封顶：调用对但不奖励猜测多余参数，
    多出的未知参数按超出数轻扣。"""
    spec = tool_defs.get(tool_name, {}).get("function", {})
    params = spec.get("parameters", {})
    required = set(params.get("required", []))
    props = params.get("properties", {})
    if not isinstance(pred_args, dict):
        return 0.0
    if not isinstance(gt_args, dict) or not gt_args:
        extra = sum(1 for k in pred_args if k not in props)
        return max(0.85 - 0.15 * extra, 0.4)
    req = [k for k in gt_args if k in required]
    opt = [k for k in gt_args if k not in required]
    req_s = (sum(_match_arg(gt_args[k], pred_args.get(k)) for k in req) / len(req)) if req else 1.0
    if opt:
        opt_s = sum(_match_arg(gt_args[k], pred_args[k]) for k in opt if k in pred_args) / len(opt)
        return 0.8 * req_s + 0.2 * opt_s
    return req_s


def _match_set(gt_calls, pred_calls, tool_defs):
    """名称一致的贪心配对（优先参数分高的），返回 (name_f1, arg_avg, 明细)。"""
    used = [False] * len(pred_calls)
    pairs = []
    for gname, gargs in gt_calls:
        best_j, best_s = -1, -1.0
        for j, (pname, pargs, ok) in enumerate(pred_calls):
            if used[j] or not ok or pname != gname:
                continue
            s = arg_soft_score(gname, gargs, pargs, tool_defs)
            if s > best_s:
                best_j, best_s = j, s
        if best_j >= 0:
            used[best_j] = True
            pairs.append({"name": gname, "arg": round(best_s, 3)})
    n, p = len(gt_calls), len(pred_calls)
    name_f1 = 2 * len(pairs) / (n + p) if (n + p) else 1.0
    arg_avg = sum(x["arg"] for x in pairs) / len(pairs) if pairs else 0.0
    return name_f1, arg_avg, pairs


def _len_score(text):
    """回答长度合理性：8~120 字为满分，4~160 半分，其余 0（防截断残句与敷衍短答）。"""
    L = len(text or "")
    if 8 <= L <= 120:
        return 1.0
    if 4 <= L <= 160:
        return 0.5
    return 0.0


def _fact_hit_rate(gt_content, gen_text):
    gt_facts = set(NUM_RE.findall(gt_content or ""))
    if not gt_facts:
        return None  # 无数字可验证
    gen_facts = set(NUM_RE.findall(gen_text or ""))
    hits = sum(1 for fr in gt_facts if any(g.endswith(fr) or fr.endswith(g) for g in gen_facts))
    return hits / len(gt_facts)


def score_step(completion, gt_msg, tool_defs):
    """给一个候选补全打分。completion 为去掉 <|im_end|> 后的纯文本；gt_msg 为该步真值
    assistant 消息；返回 {"reward":float, "kind", ...明细}。"""
    text = (completion or "").strip()
    is_tool_gt = bool(gt_msg.get("tool_calls"))

    if is_tool_gt:
        calls = parse_all_calls(text)
        valid = [c for c in calls if c[2]]
        detail = {"kind": "tool", "n_pred": len(calls),
                  "gt": [{"name": n} for n, _ in _gt_calls(gt_msg)]}
        if not valid:  # 没发调用或 XML 结构全坏 → 格式门槛直接 0
            detail.update(reward=0.0, format_ok=False, name_f1=0.0, arg_avg=0.0)
            return detail
        name_f1, arg_avg, pairs = _match_set(_gt_calls(gt_msg), calls, tool_defs)
        reward = 0.4 * name_f1 + 0.6 * arg_avg
        detail.update(reward=round(reward, 4), format_ok=True,
                      name_f1=round(name_f1, 3), arg_avg=round(arg_avg, 3), pairs=pairs)
        return detail

    # —— 真值是纯文本回答（final / no_tool / clarify 的提问轮）——
    if has_call(text):  # 不该调用却调用了（含结构坏的调用）：硬失败
        return {"kind": "final", "reward": 0.0, "no_call": False, "reason": "误发工具调用"}
    hit = _fact_hit_rate(gt_msg.get("content", ""), text)
    ls = _len_score(text)
    if hit is not None:
        reward = 0.4 + 0.6 * hit
    else:
        reward = 0.8 + 0.2 * ls
    return {"kind": "final", "reward": round(reward, 4), "no_call": True,
            "fact_hit": None if hit is None else round(hit, 3), "len_score": ls}


def score_group(completions, gt_msg, tool_defs):
    """一组候选打分 + 组内归一化优势(GRPO 核心)。
    返回 (rewards, advantages, details)；std=0 的组优势全 0（已全对/全错，无学习信号）。"""
    details = [score_step(c, gt_msg, tool_defs) for c in completions]
    rewards = [d["reward"] for d in details]
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = var ** 0.5
    if std > 1e-6:
        advs = [(r - mean) / (std + 1e-4) for r in rewards]
    else:
        advs = [0.0] * len(rewards)
    return rewards, advs, details