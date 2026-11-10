# -*- coding: utf-8 -*-
"""数学智能体闭环环境（ToRA 链路）：题目 → 写 Python → 受限执行 → 回填输出 → 最终 boxed 作答。

本项目唯一的闭环环境，供 train_grpo.py（整条链路 GRPO）与 eval_math.py（闭环评测）共用，
两者走同一套接口（鸭子类型）：
  MathEnv(task) / rollout(model, tokenizer, env, ...) / rollout_group(model, tokenizer, task, ...) /
  score_trajectory(trajectory, env) / format_trajectory(env, trajectory) / load_tasks(path, tokenizer, max_prompt_len)
（rollout 逐条采样，评测用；rollout_group 同题 G 条首回合批量采样，GRPO 训练用）

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


def render_ids(tokenizer, messages, tools):
    """消息列表 → 生成提示 token 序列（非思考模式；return_dict=False 拿纯 id 列表）。"""
    return tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                                   add_generation_prompt=True, return_dict=False,
                                   **CHAT_TMPL_KW)


@torch.no_grad()
def generate_once(model, tokenizer, input_ids, max_new_tokens, temperature, top_p, do_sample,
                  grad_ckpt, n_sequences=1):
    """生成期间临时切到推理模式（采样必须开 kv cache / 关梯度检查点），结束恢复。

    n_sequences > 1 时一次批量采 n_sequences 条（同 prompt 的组内首回合用）：输出 [n_sequences, T+新长]，
    先结束的行由 generate 用 pad 填齐到最长——调用方须按 eos 截断（_trim_at_eos）。"""
    was_training = model.training
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    try:
        return model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature, top_p=top_p, num_return_sequences=n_sequences,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True)
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
            output_text, ok = cp.stdout or "", cp.returncode == 0
            if not ok:
                err = (cp.stderr or "").strip() or f"exit code {cp.returncode}"
                output_text = (output_text + "\n" if output_text.strip() else "") + err
        except subprocess.TimeoutExpired as e:
            output_text = (e.stdout if isinstance(e.stdout, str) else None) or ""
            output_text = (output_text.strip() + "\n" if output_text.strip() else "") + \
                  f"TimeoutError: execution exceeded {timeout}s"
            ok = False
        output_text = output_text.strip("\n")
        if len(output_text) > 1500:
            output_text = output_text[:1500] + "\n... (truncated)"
        return output_text, ok
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


def _eos_ids(tokenizer):
    """生成终止 token 的 id 集合（eos_token_id 可能是 int / 列表 / None）。"""
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.convert_tokens_to_ids(IM_END)
    return list(eos) if isinstance(eos, (list, tuple)) else [eos]


def _trim_at_eos(seq_ids, eos_ids):
    """批量生成时先结束的行会被 generate 用 pad 填齐到最长：截到第一个 eos（含），
    与逐条生成的产出对齐——pad 混进 gen_ids 会污染 GRPO 重放的反传段。"""
    for pos, tok_id in enumerate(seq_ids):
        if tok_id in eos_ids:
            return seq_ids[:pos + 1]
    return seq_ids


def _record_turn(tokenizer, env, messages, turns, prompt, gen_ids):
    """一条生成结果登记为一个回合（decode、im_end 截断、抽代码、判终答），就地入档。

    prompt_ids（当时的输入）/ gen_ids（当时生成的内容）原样保存——GRPO 更新时要用
    它们"重放"这段生成。返回 (code, has_answer)；执行回填由调用方负责。"""
    text = tokenizer.decode(gen_ids, skip_special_tokens=False).split(IM_END)[0]
    code = extract_last_python_block(text)
    has_answer = BOXED in text
    messages.append({"role": "assistant", "content": text})
    turns.append({"kind": "code" if (code is not None and not has_answer) else "final",
                  "prompt_ids": prompt, "gen_ids": gen_ids, "text": text,
                  "code": code, "exec_output": None, "exec_ok": None})
    return code, has_answer


def _exec_and_refill(tokenizer, env, messages, turns, code):
    """真实执行最新回合的代码，把结果以 role="tool" 回填（模型可据此修正重试），
    返回下一回合的生成提示。role="tool" 是关键约定：模板会渲染成 <tool_response>
    包裹的 user 轮，与 SFT 数据严格一致（见模块 docstring）。"""
    outtext, ok = run_python_code(code)
    env.n_exec += 1
    env.n_error += int(not ok)
    env.exec_outputs.append(outtext[:200])
    turns[-1]["exec_output"] = outtext[:400]
    turns[-1]["exec_ok"] = bool(ok)   # 分轮次执行状态：GRPO 给代码轮独立优势（不连坐终答）
    messages.append({"role": "tool", "content": "```output\n" + outtext + "\n```"})
    return render_ids(tokenizer, messages, env.tools)


def _rollout_loop(model, tokenizer, env, messages, prompt, turns, first_turn_idx, *,
                  max_turns, max_new_tokens, temperature, top_p, max_len,
                  do_sample, grad_ckpt):
    """rollout 的续写主体（rollout 与 rollout_group 共用）：从第 first_turn_idx 回合起
    生成 → 含完整代码块且未给 boxed 则执行回填 → 继续，直到最终作答/回合上限/长度预算。

    turns 就地追加；返回结束态 final|limit|budget。"""
    ended = "limit"
    for turn_idx in range(first_turn_idx, max_turns):
        if len(prompt) + 8 > max_len:
            ended = "budget"
            break
        input_ids = torch.tensor([prompt], device=model.device)
        gen_out = generate_once(model, tokenizer, input_ids, min(max_new_tokens, max_len - len(prompt)),
                                temperature, top_p, do_sample, grad_ckpt)
        gen_ids = gen_out[0][len(prompt):].tolist()
        code, has_answer = _record_turn(tokenizer, env, messages, turns, prompt, gen_ids)

        if code is None or has_answer:
            ended = "final"
            break
        if turn_idx == max_turns - 1:
            break  # 已是最后一回合：执行结果没有后续轮可消费，省一次执行
        prompt = _exec_and_refill(tokenizer, env, messages, turns, code)
    return ended


def rollout(model, tokenizer, env, *, max_turns, max_new_tokens, temperature, top_p,
            max_len, do_sample=True, grad_ckpt=True):
    """数学链路 rollout：生成 → 含完整代码块且未给 boxed 则执行回填 → 继续，直到最终作答。

    返回 {"turns": [...], "ended": final|limit|budget}；每个 turn 记录 prompt_ids/gen_ids
    （GRPO 反传用）与文本。终止：无代码 / 已含 boxed / 回合上限 / 长度预算。"""
    messages = env.initial_msgs()
    prompt = render_ids(tokenizer, messages, env.tools)
    turns = []
    ended = _rollout_loop(model, tokenizer, env, messages, prompt, turns, 0,
                          max_turns=max_turns, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_p=top_p, max_len=max_len,
                          do_sample=do_sample, grad_ckpt=grad_ckpt)
    return {"turns": turns, "ended": ended}


def rollout_group(model, tokenizer, task, *, num_generations, gen_batch, max_turns,
                  max_new_tokens, temperature, top_p, max_len, do_sample=True,
                  grad_ckpt=True):
    """同题 G 条轨迹的组采样：首回合 prompt 完全相同，按 gen_batch 分块一次批量采 n 条
    （generate 的 num_return_sequences），第 2 回合起各条分叉（代码/执行结果不同）
    逐条续写。返回 [(trajectory, env), ...]，G 个 env 各自独立（组内互不污染）。

    显存账：批量只增加首回合的 KV cache 与线性注意力状态（全注意力层仅 8 层，
    KV 每 token KB 级；DeltaNet 状态定长），n=4 时增量约百 MB 级，8GB 卡内安全；
    第 2 回合起 prompt 已分叉，无法再批，保持逐条。"""
    env_template = MathEnv(task)
    prompt = render_ids(tokenizer, env_template.initial_msgs(), env_template.tools)
    budget = min(max_new_tokens, max_len - len(prompt))
    eos = _eos_ids(tokenizer)
    traj_envs = []
    for _chunk_start in range(0, num_generations, gen_batch):
        n_chunk = min(gen_batch, num_generations - _chunk_start)
        envs = [MathEnv(task) for _ in range(n_chunk)]
        input_ids = torch.tensor([prompt], device=model.device)
        gen_out = generate_once(model, tokenizer, input_ids, budget, temperature, top_p,
                                do_sample, grad_ckpt, n_sequences=n_chunk)
        # 首回合逐行登记：终答/回合上限的到此为止，写代码的执行回填后待续写
        slots = []   # (turns, 结束态或 None=待续写, env, messages, 续写提示)
        for row_idx in range(n_chunk):
            env = envs[row_idx]
            messages = env.initial_msgs()
            turns = []
            gen_ids = _trim_at_eos(gen_out[row_idx][len(prompt):].tolist(), eos)
            code, has_answer = _record_turn(tokenizer, env, messages, turns, prompt, gen_ids)
            if code is None or has_answer:
                slots.append((turns, "final", env, None, None))
            elif max_turns == 1:
                slots.append((turns, "limit", env, None, None))  # 同封存口径：末回合不执行
            else:
                cont_prompt = _exec_and_refill(tokenizer, env, messages, turns, code)
                slots.append((turns, None, env, messages, cont_prompt))
        # 第 2 回合起逐条续写（保持 slots 顺序 = 轨迹编号，审计日志可读）
        for turns, ended, env, messages, cont_prompt in slots:
            if ended is None:
                ended = _rollout_loop(model, tokenizer, env, messages, cont_prompt, turns, 1,
                                      max_turns=max_turns, max_new_tokens=max_new_tokens,
                                      temperature=temperature, top_p=top_p, max_len=max_len,
                                      do_sample=do_sample, grad_ckpt=grad_ckpt)
            traj_envs.append(({"turns": turns, "ended": ended}, env))
    return traj_envs


def _gt_in_exec_output(env, gt):
    """中间成果部分分：某次执行输出里出现了与 gt 等价的答案。

    纯数字 gt 用词边界正则（避免 42 命中 420）；其它形态退化为去空白子串匹配。
    这是给「代码算对了但收尾写错」的轨迹留的正向信号——L4/L5 全错组里，
    唯一的组内方差就来自这里（格式分只能区分"有没有交卷"，区分不了"算得对不对"）。"""
    gt_norm = normalize_answer(gt)
    if not gt_norm:
        return False
    if re.fullmatch(r"-?\d+(\.\d+)?", gt_norm):
        pattern = r"(?<![\d.])" + re.escape(gt_norm) + r"(?![\d.])"
        return any(re.search(pattern, out) for out in env.exec_outputs)
    return any(gt_norm in out.replace(" ", "") for out in env.exec_outputs)


def score_trajectory(trajectory, env):
    """整条轨迹打分（0~1）与明细。

    reward = 0.70×答案正确 + 0.15×执行输出命中 gt + 0.10×给出 boxed 终答
             + 0.05×执行全程无报错
    设计要点（2026-11-10 修订）：
      - 正确性仍主导，但不再是唯一的高分来源；"算对但收尾写错"能拿到 0.15 部分分，
        给难题组制造与正确性正相关的组内方差（旧版只有格式分，等于教模型"快而自信地错"）。
      - 旧的 0.1「执行过一次就加分」改为「执行全程无报错才加 0.05」：不再奖励乱试错；
        但仍保留一点无条件可得的格式分（正确性为 0 的组里必须有方差，否则整步零梯度）。
    detail.success = 端到端做题正确（评测口径，等价于 correct）。"""
    final_text = trajectory["turns"][-1]["text"] if trajectory["turns"] else ""
    pred = extract_boxed(final_text)
    correct = answers_equal(pred, env.task["gt"])
    code_hit = _gt_in_exec_output(env, env.task["gt"])
    exec_clean = env.n_exec > 0 and env.n_error == 0
    reward = 0.70 * float(correct) + 0.15 * float(code_hit) \
        + 0.10 * float(pred is not None) + 0.05 * float(exec_clean)
    detail = {"success": bool(correct), "correct": bool(correct),
              "code_hit": bool(code_hit), "exec_clean": bool(exec_clean),
              "pred": pred, "gt": env.task["gt"],
              "ended": trajectory["ended"], "n_turns": len(trajectory["turns"]),
              "n_exec": env.n_exec, "n_error": env.n_error}
    return reward, detail


def format_trajectory(env, trajectory, max_chars=1600):
    """可读链路文本（审计日志用）。"""
    lines = [f"[题] {env.task['problem'][:120]}"]
    for turn in trajectory["turns"]:
        if turn["kind"] == "code":
            lines.append("[代码] " + (turn["code"] or "")[:220])
            lines.append("  ⇐ " + (turn["exec_output"] or "")[:180])
        else:
            lines.append(f"[终答] {turn['text'][:260]}")
    lines.append(f"[结束] {trajectory['ended']} | 执行 {env.n_exec} 次（报错 {env.n_error}）")
    return "\n".join(lines)[:max_chars]


def load_tasks(path, tokenizer, max_prompt_len):
    """加载数学题任务并按初始 prompt 长度过滤。"""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    tasks = []
    for row in rows:
        task = {"id": f"math-test-{row['idx']}", "category": f"MATH-{row.get('level', '?')}",
                "problem": row["problem"], "gt": row["gt"], "level": row.get("level", "?"),
                "type": row.get("type", "")}
        prompt_len = len(render_ids(tokenizer, [{"role": "user", "content": task["problem"]}], None))
        if prompt_len <= max_prompt_len:
            tasks.append(task)
    return tasks
