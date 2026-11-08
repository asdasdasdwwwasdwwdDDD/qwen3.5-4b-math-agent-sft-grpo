# -*- coding: utf-8 -*-
"""数学智能体闭环环境（ToRA 链路）：题目 → 写 Python → 受限执行 → 回填输出 → 最终 boxed 作答。

本项目唯一的闭环环境，供 train_grpo.py（整条链路 GRPO）与 eval_math.py（闭环评测）共用，
两者走同一套接口（鸭子类型）：
  MathEnv(task) / rollout(model, tok, env, ...) / score_trajectory(ro, env) /
  format_trajectory(env, ro) / load_tasks(path, tok, max_prompt_len)

链路约定（与 data/convert_tora.py 生成的 SFT 数据严格一致）：
  user(题目) → assistant(```python 代码块```) → tool(```output 执行结果```) → assistant(收尾 + \boxed{})
执行结果必须用 role="tool" 回填（模板渲染为 <tool_response> 且被 last_query 回扫豁免，
中间 assistant 轮才能带上与生成提示一致的空 <think> 块——详见 convert_tora.py 注释）。

安全说明：这里真实执行模型生成的 Python（隔离子进程 + 超时 + 截断），因为"代码解释器"
就是本链路的训练目标，这是用户指定的方向；残余风险为 TIR 训练通行做法。
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import torch

# 与 training/train_qlora.py 的 CHAT_TMPL_KW 一致；此处本地定义而不是 import，
# 避免把训练栈（transformers/bnb/peft）拖进推理路径
CHAT_TMPL_KW = {"enable_thinking": False}


def render_ids(tok, msgs, tools):
    """消息列表 → 生成提示 token 序列（非思考模式；return_dict=False 拿纯 id 列表）。"""
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


# 成对竖线的特殊 token 一律拼接构造（与 train_grpo.py 同理，防工具序列化截断）
PIPE = chr(124)
IM_END = "<" + PIPE + "im_end" + PIPE + ">"
BOXED = "\\" + "boxed"          # \boxed{...}

_CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
# 注意：正则里匹配字面 \boxed 必须写 \\boxed（单个 \b 会被当成词边界断言）
_BOXED_PAT = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


# ---------------------------------------------------------------- 受限执行 ----
def run_python_code(code, timeout=20):
    """在隔离子进程中执行模型生成的 Python，返回 (回填文本, 是否无报错)。

    隔离措施：独立临时目录 + `python -I`（隔离模式：忽略 PYTHON* 环境变量与用户
    site）+ 环境变量精简 + 超时强杀 + 输出截断 1500 字符。报错时把 stderr 拼进
    回填文本——模型看到 Traceback 可以修正重试（error_retry 行为的来源）。"""
    tmp = tempfile.mkdtemp(prefix="tora_sbx_")
    script = os.path.join(tmp, "solution.py")
    try:
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        env["TEMP"] = env["TMP"] = tmp
        try:
            cp = subprocess.run([sys.executable, "-I", script], capture_output=True,
                                text=True, timeout=timeout, cwd=tmp, env=env,
                                encoding="utf-8", errors="replace")
            out, ok = cp.stdout or "", cp.returncode == 0
            if not ok:
                err = (cp.stderr or "").strip() or f"exit code {cp.returncode}"
                out = (out + "\n" if out.strip() else "") + err
        except subprocess.TimeoutExpired as e:
            out = (e.stdout if isinstance(e.stdout, str) else None) or ""
            out = (out.strip() + "\n" if out.strip() else "") + \
                  f"TimeoutError: execution exceeded {timeout}s"
            ok = False
        out = out.strip("\n")
        if len(out) > 1500:
            out = out[:1500] + "\n... (truncated)"
        return out, ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 答案比对 ----
def extract_boxed(text):
    """取最后一个 \\boxed{...} 的内容（支持一层嵌套花括号）。"""
    m = _BOXED_PAT.findall(text)
    return m[-1] if m else None


def normalize_answer(s):
    """LaTeX 答案归一化：去 $/\\left/空白宏，\\frac→除法，\\sqrt→sqrt，^→**，去尾零。"""
    if s is None:
        return None
    s = s.strip()
    for a, b in (("$", ""), ("\\(", ""), ("\\)", ""), ("\\[", ""), ("\\]", ""),
                 ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"),
                 ("\\left", ""), ("\\right", ""),
                 ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""),
                 ("\\%", ""), ("^\\circ", ""), ("\\degree", ""),
                 ("\\pi", "pi"), ("\\sqrt", "sqrt"), ("^", "**")):
        s = s.replace(a, b)
    s = re.sub(r"\\(?:mbox|text|mathrm|mathbf|mathit)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    # 剩余花括号组（如 sqrt{3}）转圆括号，与 Python/sympy 写法对齐
    s = re.sub(r"\{([^{}]*)\}", r"(\1)", s)
    s = s.replace(",", "") if re.fullmatch(r"-?[\d,]+(\.\d+)?", s) else s
    # 纯数字去尾零：4.0 -> 4（gt 常为整数，模型常输出 x.0）
    if re.fullmatch(r"-?\d+\.[0-9]*0+", s):
        s = s.rstrip("0").rstrip(".")
    # sympify 友好：隐式乘法 2sqrt(3)->2*sqrt(3)、3pi->3*pi、2(...)->2*(...)
    s = re.sub(r"(\d)(sqrt|pi|\()", r"\1*\2", s)
    # 剥离 "x\in" / "x=" 类前缀（模型 boxed 常带主语，gt 通常只有值）
    s = re.sub(r"^[a-zA-Z]\s*\\in\s*", "", s)
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)
    return s.strip()


def answers_equal(pred, gt):
    """答案等价判断：归一化字符串相等 → sympy 符号等价 → 数值近似。"""
    if pred is None:
        return False
    p, g = normalize_answer(pred), normalize_answer(gt)
    if p is None or g is None:
        return False
    if p == g:
        return True
    try:
        import sympy as sp
        a = sp.sympify(p, rational=True)
        b = sp.sympify(g, rational=True)
        if a == b or sp.simplify(a - b) == 0:
            return True
        # 符号近似（如 2*pi vs 6.283185…）：evalf 数值比对
        try:
            fa, fb = float(a.evalf()), float(b.evalf())
            if abs(fa - fb) <= 1e-6 * max(1.0, abs(fb)):
                return True
        except (TypeError, ValueError):
            pass
    except Exception:
        pass
    try:
        fa, fb = float(p), float(g)
        return abs(fa - fb) <= 1e-6 * max(1.0, abs(fb))
    except Exception:
        return False


# -------------------------------------------------------------------- 环境 ----
class MathEnv:
    """一道数学题的环境实例。每条 rollout 必须新建一个（统计互不污染）。"""

    def __init__(self, task):
        self.task = task          # {"id","problem","gt","level","type"}
        self.tools = None         # 数学链路无工具定义（区别于旧智能体数据）
        self.n_exec = 0
        self.n_error = 0
        self.exec_outputs = []

    def initial_msgs(self):
        return [{"role": "user", "content": self.task["problem"]}]


def extract_last_python_block(text):
    """取最后一个完整闭合的 ```python 代码块；未闭合（生成截断）返回 None。"""
    blocks = _CODE_BLOCK.findall(text)
    return blocks[-1].strip("\n") if blocks else None


def rollout(model, tok, env, *, max_turns, max_new_tokens, temperature, top_p,
            max_len, do_sample=True, grad_ckpt=True):
    """数学链路 rollout：生成 → 含完整代码块且未给 boxed 则执行回填 → 继续，直到最终作答。

    返回 {"turns": [...], "ended": final|limit|budget}；每个 turn 记录 prompt_ids/gen_ids
    （GRPO 反传用）与文本。终止：无代码 / 已含 boxed / 回合上限 / 长度预算。"""
    msgs = env.initial_msgs()
    prompt = render_ids(tok, msgs, env.tools)
    turns, ended = [], "limit"

    for i in range(max_turns):
        if len(prompt) + 8 > max_len:
            ended = "budget"
            break
        enc = torch.tensor([prompt], device=model.device)
        out = generate_once(model, tok, enc, min(max_new_tokens, max_len - len(prompt)),
                            temperature, top_p, do_sample, grad_ckpt)
        gen_ids = out[0][len(prompt):].tolist()
        text = tok.decode(gen_ids, skip_special_tokens=False).split(IM_END)[0]
        code = extract_last_python_block(text)
        has_answer = BOXED in text
        msgs.append({"role": "assistant", "content": text})
        turns.append({"kind": "code" if (code is not None and not has_answer) else "final",
                      "prompt_ids": prompt, "gen_ids": gen_ids, "text": text,
                      "code": code, "exec_output": None})

        if code is None or has_answer:
            ended = "final"
            break
        if i == max_turns - 1:
            break  # 已是最后一回合：执行结果没有后续轮可消费，省一次执行
        # 执行代码并把结果作为 tool 消息回填（模型可据此修正重试）
        outtext, ok = run_python_code(code)
        env.n_exec += 1
        env.n_error += int(not ok)
        env.exec_outputs.append(outtext[:200])
        turns[-1]["exec_output"] = outtext[:400]
        msgs.append({"role": "tool", "content": "```output\n" + outtext + "\n```"})
        prompt = render_ids(tok, msgs, env.tools)

    return {"turns": turns, "ended": ended}


def score_trajectory(ro, env):
    """整条轨迹打分（0~1）与明细。

    reward = 0.8×答案正确 + 0.1×给出 boxed 终答 + 0.1×真正用过代码执行
    （正确性主导；后两项给"格式/链路纪律"提供组内区分度，全错组仍有梯度信号）。
    detail.success = 端到端做题正确（评测口径，等价于 correct）。"""
    final_text = ro["turns"][-1]["text"] if ro["turns"] else ""
    pred = extract_boxed(final_text)
    correct = answers_equal(pred, env.task["gt"])
    reward = 0.8 * float(correct) + 0.1 * float(pred is not None) \
        + 0.1 * float(env.n_exec > 0)
    detail = {"success": bool(correct), "correct": bool(correct),
              "pred": pred, "gt": env.task["gt"],
              "ended": ro["ended"], "n_turns": len(ro["turns"]),
              "n_exec": env.n_exec, "n_error": env.n_error}
    return reward, detail


def format_trajectory(env, ro, max_chars=1600):
    """可读链路文本（审计日志用）。"""
    lines = [f"[题] {env.task['problem'][:120]}"]
    for t in ro["turns"]:
        if t["kind"] == "code":
            lines.append("[代码] " + (t["code"] or "")[:220])
            lines.append("  ⇐ " + (t["exec_output"] or "")[:180])
        else:
            lines.append(f"[终答] {t['text'][:260]}")
    lines.append(f"[结束] {ro['ended']} | 执行 {env.n_exec} 次（报错 {env.n_error}）")
    return "\n".join(lines)[:max_chars]


def load_tasks(path, tok, max_prompt_len):
    """加载数学题任务并按初始 prompt 长度过滤。"""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    tasks = []
    for r in rows:
        t = {"id": f"math-test-{r['idx']}", "category": f"MATH-{r.get('level', '?')}",
             "problem": r["problem"], "gt": r["gt"], "level": r.get("level", "?"),
             "type": r.get("type", "")}
        n = len(render_ids(tok, [{"role": "user", "content": t["problem"]}], None))
        if n <= max_prompt_len:
            tasks.append(t)
    return tasks
