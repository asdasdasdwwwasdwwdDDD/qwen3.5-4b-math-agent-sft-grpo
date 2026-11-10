# GRPO 核心训练代码精讲（抄写学习版）

> 对应源码：`scripts/training/train_grpo.py`（GRPO 主循环）+ `scripts/envs/math_env.py`（数学闭环环境）
> 写给 Python 功底不深、想把核心训练逻辑彻底搞懂的读者。
> 源码本身只保留简洁注释；**本文件是"放大版"**——带详细注释的代码 + 大白话讲解，适合对照源码抄写学习。如有出入，以源码为准。

## 目录

- 0. 大局观：这个项目到底在训练什么
- 1. 预备知识：5 个必须先懂的概念
- 2. 打分：`score_trajectory` —— 裁判怎么打分
- 3. 做题：`rollout` 与组批量采样 —— 模型怎么走完一条解题链路
- 4. 概率读数仪：`token_logprobs`
- 5. 主循环·上半场：组生成与组内优势（generate_scored_group / group_advantages）
- 6. 主循环·下半场：μ 轮重放 + PPO 裁剪 + KL 锚定（train_step_update）
- 7. 收尾三件事：梯度累积的坑、训练中考试、记日志（run_closed_eval / log_step）
- 8. 8GB 显存生存术（代码里的省显存手段清单）
- 9. 一次训练步完整时间线（串起来）
- 10. 名词小词典
- 11. 常见疑问（用封存日志里的真实数据回答）

---

## 0. 大局观：这个项目到底在训练什么

一句话：**教一个 40 亿参数的语言模型做数学题——先临摹标准答案（SFT），再自己做题、按成绩受奖惩（GRPO）。**

用"刷题班"打比方：

| 阶段 | 类比 | 代码 |
|---|---|---|
| SFT 监督微调 | 发标准答案让学生临摹：1901 道题，每道带完整的「写代码 → 执行 → 作答」过程 | `scripts/training/train_qlora.py` |
| GRPO 强化学习 | 不发答案。学生自己做题，老师只打分不讲评；**同一道题做 4 遍**，做得比另外几遍好的做法被强化，差的被弱化 | `scripts/training/train_grpo.py` + `scripts/envs/math_env.py` |

GRPO 一次训练步的流水线（P=2 道题 × G=4 条轨迹 = 8 条）：

```
抽 2 道数学题
 ├─ 题 A ──采样 4 条完整解题轨迹──→ 逐条打分 r1..r4 ──→ 组内归一化 → 优势 a1..a4
 └─ 题 B ──同样 4 条───────────→ r1..r4 ──→ 组内归一化 → 优势 a1..a4
                 ▼
   8 条轨迹一起做 PPO 更新（好轨迹概率↑ 差轨迹↓）+ KL 锚定（不许跑太远）
                 ▼
   写日志 / TensorBoard；每 20 步 greedy 考试一次，考得最好就存档 out/best
```

μ=--epochs（默认 1）控制"这批轨迹重放几遍"：默认 1 = 封存口径，8 条轨迹各反传 1 遍就丢弃；设 4（PPO 的 μ=4）则同批数据重放 4 遍再丢弃——第 2 遍起权重已动而 old_logp 是首遍缓存读数，ratio 偏离 1、PPO 裁剪真正上岗（第 6 章细讲）。

最终训练的**不是整个模型**：底座全程冻结，只训插在旁边的一对对"小矩阵"（LoRA，约占参数 0.5%）——这是 8GB 显卡玩得动的根本原因。

---

## 1. 预备知识：5 个必须先懂的概念

### 1.1 token：模型眼里的"字"

模型不按"字"或"词"读文本，按 **token**（词元）——可能是一个汉字、半个英文单词、一个符号。模型从头到尾只做一件事：

> 给定前面所有 token，给词表里**每一个**候选 token 打一个"下一个是我"的分数。
> （Qwen3.5 的词表有 24.8 万个候选，代码里写作 248k。）

### 1.2 概率、logits、log_softmax

- **logits**：上面的原始分数，可以是任何数（-3.2、12.7……），不归一。
- **softmax**：把一排 logits 变成一排概率，加起来恰好 = 1。
- **log_softmax**：取完概率再取 log。为什么用 log？① 连乘会下溢，log 把乘变加；② 两个 log 概率**相减** = 概率**相除**——PPO 的"新旧概率比"就是这么算的。
- log 概率的直观读法：0 = 100% 确定；-3 ≈ 5%；越负越不确定。

### 1.3 LoRA：只训 0.5% 的参数

大模型每层都有大权重矩阵 W（如 2560×2560）。LoRA 的做法：**W 冻结不动**，旁边挂一对小矩阵 A、B（如 2560×16 和 16×2560），实际使用的权重变成 `W + B×A`。要训的只有 A、B。

`model.disable_adapter()` = 临时把 B×A 摘掉 = 模型退回 SFT 原样。GRPO 拿它当 **KL 锚定的参考模型**——零额外显存，因为根本不用加载第二个模型。

### 1.4 策略（policy）与奖励（reward）

- **策略**：就是"模型当前的做题习惯"——它生成每个 token 的概率分布。训练 = 调整这些概率。
- **奖励**：一条完整解题轨迹最后得到的一个 0~1 分数（第 2 章）。只看结果、不给中间过程打分——这叫"结果监督"。

### 1.5 训练三步流水

PyTorch 里每次参数更新都是三步，GRPO 主循环就是反复做这三步（讲究的全在 loss 怎么构造）：

```
loss = 按第 2/5/6 章构造的损失    # ① 算"损失"：衡量模型当前行为离目标有多远
loss.backward()     # ② 反向传播：算出"每个可训练参数该往哪边挪多少"（梯度）
optimizer.step()    # ③ 优化器：真的把参数挪一小步
```

---

## 2. 打分：`score_trajectory` —— 裁判怎么打分

源码位置：`scripts/envs/math_env.py` 的 `score_trajectory()`（L235）。

```python
def score_trajectory(trajectory, env):
    """整条轨迹打分（0~1）与明细。

    reward = 0.8×答案正确 + 0.1×给出 boxed 终答 + 0.1×真正用过代码执行
    （正确性主导；后两项给"格式/链路纪律"提供组内区分度，全错组仍有梯度信号）。
    detail.success = 端到端做题正确（评测口径，等价于 correct）。"""
    final_text = trajectory["turns"][-1]["text"] if trajectory["turns"] else ""
    pred = extract_boxed(final_text)
    correct = answers_equal(pred, env.task["gt"])
    reward = 0.8 * float(correct) + 0.1 * float(pred is not None) \
        + 0.1 * float(env.n_exec > 0)
    detail = {"success": bool(correct), "correct": bool(correct),
              "pred": pred, "gt": env.task["gt"],
              "ended": trajectory["ended"], "n_turns": len(trajectory["turns"]),
              "n_exec": env.n_exec, "n_error": env.n_error}
    return reward, detail
```

逐行人话：

| 代码 | 意思 |
|---|---|
| `trajectory["turns"][-1]["text"]` | 轨迹**最后一个回合**的文本（约定：终答在最后）；`trajectory["turns"]` 是这条解题过程的逐回合档案（第 3 章讲） |
| `extract_boxed(final_text)` | 从文本里抠出 `\boxed{...}` 里的答案；没写 boxed → 返回 None |
| `answers_equal(pred, gt)` | 模型答案 vs 标准答案。**不是字符串比较，是数学等价**：`1/2` ≡ `0.5` ≡ `2/4`，`2√3` ≡ `3.464…`。实现分三层：先归一化写法（去 `$`、`\frac` 变除法、去尾零…），再用 sympy 做符号等价，最后数值近似兜底 |
| `reward = 0.8×correct + 0.1×boxed + 0.1×n_exec>0` | 三项加权打分，满分 1.0 |
| `detail` | 打分明细（对错、预测值、真值、执行几次、报错几次），训练时写进审计日志，评测时当统计口径 |

**为什么正确率只占 0.8，还要掺两项"小分"？**

设想一道很难的题，4 条轨迹全错：正确分全为 0。如果 reward 只有正确分，4 条轨迹分数完全一样 → 组内没有分差 → 没有梯度（第 5 章）→ 这一组白跑。掺了 0.1 的"有没有按规矩写 boxed"和 0.1 的"有没有真的执行过代码"后，全错的组里通常也能分出高下（有的轨迹写了 boxed、有的没写）→ 有分差 → 有梯度。**这两项小分是给"组内比较"留的信号通道。**

---

## 3. 做题：`rollout` 与组批量采样 —— 模型怎么走完一条解题链路

源码位置：`scripts/envs/math_env.py`。这是 GRPO 的**数据来源**：一条 rollout = 一条完整解题轨迹。
本章五个函数按源码顺序：`rollout()`（L269）→ `_record_turn`（L214）→ `_exec_and_refill`（L229）→ `_rollout_loop`（L242）→ `rollout_group`（L285），外加 `generate_once`（L41）。

**为什么拆成小函数？** 逐条采样 `rollout` 与组批量采样 `rollout_group` 共用同一套续写逻辑（`_rollout_loop` + 两个回合级助手）——回合怎么登记、执行结果怎么回填只有一份实现，改动一处两条路径同时生效；抄写时每块也都能独立成段。

### 3.1 入口：rollout —— 一条轨迹的壳

```python
def rollout(model, tokenizer, env, *, max_turns, max_new_tokens, temperature, top_p,
            max_len, do_sample=True, grad_ckpt=True):
    # 一条 rollout = 一条完整解题轨迹：起手摆好题目，剩下的全交给 _rollout_loop
    messages = env.initial_msgs()                      # 消息列表 = 一条 user 消息（题目）
    prompt = render_ids(tokenizer, messages, env.tools)      # 把消息渲染成 chat 模板的 token 序列
    turns = []                                     # turns=逐回合档案（逐步填）
    # 续写主体：从第 0 回合开始循环——生成 → 代码？→ 真执行 → 回填 → 继续，直到终答
    ended = _rollout_loop(model, tokenizer, env, messages, prompt, turns, 0,
                          max_turns=max_turns, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_p=top_p, max_len=max_len,
                          do_sample=do_sample, grad_ckpt=grad_ckpt)
    return {"turns": turns, "ended": ended}        # ended: final|limit|budget（含义见 3.4）
```

### 3.2 生成原语：generate_once（n_sequences 批量开关）

```python
@torch.no_grad()
def generate_once(model, tokenizer, input_ids, max_new_tokens, temperature, top_p, do_sample,
                  grad_ckpt, n_sequences=1):
    # 生成期间临时切到推理模式（采样必须开 kv cache / 关梯度检查点），结束恢复。
    # n_sequences > 1 时一次批量采 n_sequences 条（同 prompt 的组内首回合用）：输出 [n_sequences, T+新长]，
    # 先结束的行由 generate 用 pad 填齐到最长——调用方须按 eos 截断（_trim_at_eos）
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
```

（`@torch.no_grad()`：生成只为采样，不需要梯度图。训练态 ↔ 推理态的切换在 `finally` 里保证无论生成是否报错都能恢复。）

### 3.3 回合级助手：登记一个回合 / 执行并回填

```python
def _record_turn(tokenizer, env, messages, turns, prompt, gen_ids):
    # 一条生成结果登记为一个回合（decode、im_end 截断、抽代码、判终答），就地入档。
    # prompt_ids（当时的输入）/ gen_ids（当时生成的内容）原样保存——GRPO 更新时要用
    # 它们"重放"这段生成（第 6 章全靠它们逐 token 算概率）
    text = tokenizer.decode(gen_ids, skip_special_tokens=False).split(IM_END)[0]
    # IM_END（<|im_end|>）是模型"我说完了"的结束符，截掉它之后的残余
    code = extract_last_python_block(text)         # 本回合写了什么：完整的 ```python 代码块？
    has_answer = BOXED in text                     # \boxed 终答给出来了吗？
    messages.append({"role": "assistant", "content": text})
    turns.append({"kind": "code" if (code is not None and not has_answer) else "final",
                  "prompt_ids": prompt, "gen_ids": gen_ids, "text": text,
                  "code": code, "exec_output": None})
    return code, has_answer
```

```python
def _exec_and_refill(tokenizer, env, messages, turns, code):
    # 真执行最新回合的代码（隔离子进程），结果以 role="tool" 回填——模型下一轮能看到并修正重试。
    # role="tool" 是关键约定：模板渲染成 <tool_response> 包裹的 user 轮，与 SFT 数据严格一致
    # （中间 assistant 代码轮的 <think> 块才不会被剥掉——convert_tora.py 有长注）
    outtext, ok = run_python_code(code)
    env.n_exec += 1                # 统计：执行了几次（打分 0.1 的"执行无误"项要用）
    env.n_error += int(not ok)     # 统计：其中几次报错
    env.exec_outputs.append(outtext[:200])
    turns[-1]["exec_output"] = outtext[:400]
    messages.append({"role": "tool", "content": "```output\n" + outtext + "\n```"})
    return render_ids(tokenizer, messages, env.tools)   # 全部历史重新渲染 = 下一回合的输入
```

### 3.4 续写主体：_rollout_loop

```python
def _rollout_loop(model, tokenizer, env, messages, prompt, turns, first_turn_idx, *,
                  max_turns, max_new_tokens, temperature, top_p, max_len,
                  do_sample, grad_ckpt):
    # rollout 的续写主体（rollout 与 rollout_group 共用）：从第 first_turn_idx 回合起循环。
    # 就地往 turns 追加；返回结束态 final（作答）| limit（回合用完）| budget（长度用完）
    ended = "limit"
    for turn_idx in range(first_turn_idx, max_turns):            # 最多 4 个回合（写代码→执行→再想→终答）
        # 长度预算：题目+已生成内容快顶到 max_len（2048）就收手——8GB 显存红线
        if len(prompt) + 8 > max_len:
            ended = "budget"
            break
        input_ids = torch.tensor([prompt], device=model.device)
        # 采样生成本回合内容（一段代码或一个终答）。temperature=0.8 的随机采样保证
        # 同一道题 G=4 条轨迹各不相同——有差异才有"组内比较"
        gen_out = generate_once(model, tokenizer, input_ids, min(max_new_tokens, max_len - len(prompt)),
                                temperature, top_p, do_sample, grad_ckpt)
        # generate 返回"完整序列"（输入+新内容），掐头取新增部分=本回合生成的 token
        gen_ids = gen_out[0][len(prompt):].tolist()
        code, has_answer = _record_turn(tokenizer, env, messages, turns, prompt, gen_ids)

        # 没写代码（直接作答/跑题）、或已给出终答 → 解题结束
        if code is None or has_answer:
            ended = "final"
            break
        if turn_idx == max_turns - 1:
            break  # 已是最后一回合：执行结果没有后续轮可消费，省一次执行
        prompt = _exec_and_refill(tokenizer, env, messages, turns, code)
    return ended
```

一个回合里发生的事，按顺序：

1. **生成**：`generate_once` 让模型接着当前对话写一段（temperature=0.8 随机采样，所以同一道题 4 次写的不一样）。
2. **切出新生成部分**：`out[0][len(prompt):]`——generate 返回"输入+新内容"，掐头去尾取新增。
3. **登记入档**：`_record_turn` 变回文本、判断写了什么，把 `prompt_ids` 和 `gen_ids` 原样存进 `turns`——更新时要用它们"重放"这段生成。
4. **要不要继续？** 没写代码或已终答 → 结束；最后一回合 → 直接结束；否则 → **真执行代码**（`_exec_and_refill`），结果回填，渲染下一轮输入。

三个容易忽略的点：

- **`run_python_code` 是真执行**（TIR 范式，代码解释器就是训练目标）。隔离措施：独立子进程 `python -I` + 独立临时目录 + 环境变量精简 + 20 秒超时强杀 + 输出截断 1500 字。报错时 Traceback 会回填给模型——模型由此学会"读报错 → 改代码"（这是执行报错率 37.7%→24.3% 的直接原因）。
- **为什么结果必须用 `role="tool"` 回填？** Qwen3.5 的聊天模板只给"最后一条 user 消息之后"的 assistant 轮插 `<think>` 块。tool 消息渲染后有豁免标记，中间的 assistant 代码轮才能和 SFT 数据的渲染完全一致；用裸 user 轮回填会把代码轮剥 think，训练直接走样（这个坑 `data/convert_tora.py` 里有长注）。
- **结束原因三种**：`final`（正常作答）/ `limit`（回合用完）/ `budget`（长度用完）。

### 3.5 组批量采样：rollout_group（效率路径，--gen-batch > 1 时启用）

同组 G=4 条轨迹的**首回合 prompt 完全相同**（同一道题、同一模板）——逐条采样时同一个 prompt 要完整前向 4 次。`rollout_group` 把首回合拼成一次批量采样（`generate` 的 `num_return_sequences`），生成阶段提效约 30-40%；第 2 回合起各条轨迹的代码/执行结果已经分叉，prompt 各不相同，只能回到逐条续写。

**显存账**：批量只增加首回合的 KV cache（全注意力层仅 8 层，每 token KB 级）与线性注意力定长状态，n=4 时增量约百 MB 级——8GB 卡内安全。**与批量生成配套的关键细节是 eos 截断**：`generate` 批量输出时先结束的行会被 pad 填齐到最长，pad 若混进 `gen_ids` 会污染第 6 章重放的反传段，所以逐行按第一个 eos 截断（`_trim_at_eos`），截完与逐条生成的产出对齐。

```python
def rollout_group(model, tokenizer, task, *, num_generations, gen_batch, max_turns,
                  max_new_tokens, temperature, top_p, max_len, do_sample=True,
                  grad_ckpt=True):
    # 同题 G 条轨迹的组采样：首回合 prompt 完全相同，按 gen_batch 分块一次批量采 n 条，
    # 第 2 回合起各条分叉（代码/执行结果不同）逐条续写。返回 [(trajectory, env), ...]
    env_template = MathEnv(task)
    prompt = render_ids(tokenizer, env_template.initial_msgs(), env_template.tools)   # 同题：渲染一次共用
    budget = min(max_new_tokens, max_len - len(prompt))
    eos = _eos_ids(tokenizer)          # 生成终止 token id（可能是列表，如 im_end + endoftext）
    traj_envs = []
    for _chunk_start in range(0, num_generations, gen_batch):   # 按 gen_batch 分块
        n_chunk = min(gen_batch, num_generations - _chunk_start)
        envs = [MathEnv(task) for _ in range(n_chunk)]       # 每条轨迹独立 env（统计互不污染）
        input_ids = torch.tensor([prompt], device=model.device)
        # 首回合一次批量采 n 条：与逐条 n 次统计等价，只换随机数流
        gen_out = generate_once(model, tokenizer, input_ids, budget, temperature, top_p,
                                do_sample, grad_ckpt, n_sequences=n_chunk)
        # 首回合逐行登记：终答/回合上限的到此为止，写代码的执行回填后待续写
        slots = []   # (turns, 结束态或 None=待续写, env, messages, 续写提示)
        for row_idx in range(n_chunk):
            env = envs[row_idx]
            messages = env.initial_msgs()
            turns = []
            gen_ids = _trim_at_eos(gen_out[row_idx][len(prompt):].tolist(), eos)
            # ↑ 批量行按第一个 eos 截断（含），pad 不许混进 gen_ids（见上文）
            code, has_answer = _record_turn(tokenizer, env, messages, turns, prompt, gen_ids)
            if code is None or has_answer:
                slots.append((turns, "final", env, None, None))
            elif max_turns == 1:
                slots.append((turns, "limit", env, None, None))  # 同封存口径：末回合不执行
            else:
                cont_prompt = _exec_and_refill(tokenizer, env, messages, turns, code)
                slots.append((turns, None, env, messages, cont_prompt))
        # 第 2 回合起逐条续写（保持 slots 顺序 = 轨迹编号，审计日志可读）；
        # 注意分块语义：第 1 块的轨迹完整续写完，才轮到第 2 块的首回合生成
        for turns, ended, env, messages, cont_prompt in slots:
            if ended is None:
                ended = _rollout_loop(model, tokenizer, env, messages, cont_prompt, turns, 1,
                                      max_turns=max_turns, max_new_tokens=max_new_tokens,
                                      temperature=temperature, top_p=top_p, max_len=max_len,
                                      do_sample=do_sample, grad_ckpt=grad_ckpt)
            traj_envs.append(({"turns": turns, "ended": ended}, env))
    return traj_envs
```

（批量配套的两个小工具，源码 L197/L205，随本节一起抄：）

```python
def _eos_ids(tokenizer):
    """生成终止 token 的 id 集合（eos_token_id 可能是 int / 列表 / None）。"""
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.convert_tokens_to_ids(IM_END)   # eos 未设置时退回 im_end 的 id
    return list(eos) if isinstance(eos, (list, tuple)) else [eos]


def _trim_at_eos(seq_ids, eos_ids):
    """批量生成时先结束的行会被 generate 用 pad 填齐到最长：截到第一个 eos（含），
    与逐条生成的产出对齐——pad 混进 gen_ids 会污染 GRPO 重放的反传段。"""
    for pos, tok_id in enumerate(seq_ids):
        if tok_id in eos_ids:
            return seq_ids[:pos + 1]
    return seq_ids
```

**默认 `--gen-batch 1` 不走这条路径**：训练主循环直接逐条 `rollout`（随机数流与封存一致）；`--gen-batch 4` 才启用组批量（统计等价、非逐位相同）。谁在调它见 5.2。

---

## 4. 概率读数仪：`token_logprobs`

源码位置：`scripts/training/train_grpo.py`（L56）。

GRPO 要回答一个问题：**"模型生成自己那条回答里的每个 token，各给了多大概率？"** 这个函数就是读数仪，主循环会以三种身份调用它（old_logp / ref_logp / act_logp，第 6 章讲）。

```python
def token_logprobs(model, input_ids, attention_mask, n_gen_tokens):
    # ① 前向跑一遍模型，拿 logits——模型对"下一个 token 是谁"的原始打分。
    #    形状 [n_gen_tokens+1, 248k]：第 i 行 = 看完前 i 个 token 后，对下一个 token 的打分。
    #    logits_to_keep=n_gen_tokens+1：只对末尾 n_gen_tokens+1 个位置出 logits，
    #    题目段的不算——词表太大，这一开关省下约 1.5GB 显存。
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                   logits_to_keep=n_gen_tokens + 1).logits[0]
    # ② log_softmax：把原始打分归一化成"对数概率"（每行 24.8 万个数的概率和为 1）。
    #    [:-1] 是 next-token 错位对齐：第 i 行预测的是第 i+1 个 token，
    #    丢掉最后一行后，第 i 行正好对上第③步取出的第 i 个真实 token。
    #    .float()：24.8 万个数求和，bf16 精度不够，log_softmax 用 32 位算才不失真。
    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
    # ③ 取出序列末尾 n_gen_tokens 个"真实生成出来的 token"编号作为目标。
    #    unsqueeze(-1) 把 [n_gen_tokens] 变 [n_gen_tokens, 1]，凑成 gather 要求的索引形状。
    target_ids = input_ids[0, -n_gen_tokens:].unsqueeze(-1)
    # ④ gather：从每行 24.8 万个对数概率里，按 target_ids 挑出"真实 token 那一项"。
    #    squeeze(-1) 把 [n_gen_tokens, 1] 压回 [n_gen_tokens] —— 即每个生成 token 的 log p。
    return log_probs.gather(-1, target_ids).squeeze(-1)
```

设轨迹最后一个回合生成了 3 个 token（n_gen_tokens=3），逐步看形状（词表真有 24.8 万列，这里画成 5 列）：

```
input_ids = [题, 题, …, 生1, 生2, 生3]        长度 = 题目 + 3

① logits 形状 [4, 5]（logits_to_keep=4：只算末尾 4 个位置）
     第0行 = 读完题目，对"生1"的打分
     第1行 = 读完"生1"，对"生2"的打分
     第2行 = 读完"生1生2"，对"生3"的打分
     第3行 = 读完全部，对"再下一个"的打分 ← 用不上，下面丢掉

② log_probs = log_softmax(logits[:-1])   形状 [3, 5]
     [:-1] 丢掉第3行 —— "错位对齐"：第 i 行预测的正好是第 i+1 个真实 token

③ target_ids = [生1, 生2, 生3] 的编号    形状 [3] → unsqueeze → [3, 1]

④ log_probs.gather(-1, target_ids)              形状 [3, 1]：每行挑出"真实 token"那一格
     squeeze(-1) → [3]            就是模型给 生1/生2/生3 的 log 概率
```

`gather` 单独演示（它就是"按索引挑数"）：

```
log_probs    = [[-0.1, -3.2, -0.9],        target_ids   = [[1],
         [-2.0, -0.2, -1.7],                 [0],
         [-1.1, -2.8, -0.3]]                 [2]]
结果  = [[-3.2], [-2.0], [-0.3]]     ← 每行挑出"真实 token"那一列
```

---

## 5. 主循环·上半场：组生成与组内优势

源码位置：`scripts/training/train_grpo.py`——主干 `train_trajectory_loop()`（L309）、组生成 `generate_scored_group()`（L137）、组内优势 `group_advantages()`（L124）。
本章代码块按顺序连起来就是"上半场"的完整原文（任务排队 → 生成打分 → 组内优势 → 审计落盘），可整段抄写；主干开头在 5.1，生成的两条路径在 5.2，组内优势在 5.3。

### 5.1 函数开头：任务队列 make_task_order 与初始化

先看任务顺序怎么排（难度层交织就是它实现的）：

```python
def make_task_order(task_pool, curriculum):
    """任务顺序：课程化=各难度层内部打乱后轮转交织（L1,L2,L3,L4,L5,L1,…），
    否则整体打乱。默认课程化：训练池 82% 是 Level 3-5 难题，SFT 模型
    采样正确率低 → 整组全错且 reward 同为 0.2（格式分全拿）→ 组内零方差零梯度
    （实测首版 run 的 step1 两组 8 轨迹全 0.2 无信号）。交织保证每步的任务里
    大概率混有低难度题（组内对错混合 → 有优势信号）。"""
    if curriculum:
        by_level = {}
        for task in task_pool:
            by_level.setdefault(task.get("level", "?"), []).append(task)   # 按难度层分桶
        for bucket in by_level.values():
            random.shuffle(bucket)                                 # 桶内部打乱
        task_order, levels = [], sorted(by_level)
        while any(by_level[level] for level in levels):
            for level in levels:                        # 各层轮转依次抽一道（L1 到 L5 依次循环）
                if by_level[level]:
                    task_order.append(by_level[level].pop())
        return task_order
    task_order = task_pool[:]                               # 非课程化：整体打乱一份副本
    random.shuffle(task_order)
    return task_order
```

再看主干函数的开头（初始化 + 每步先补任务队列）。重构后主干只做"时间线骨架"：每个环节是一个独立函数（5.2 生成 / 5.3 优势 / 6.x 更新 / 7.2 考试 / 7.3 记日志），主干读起来就是第 9 章那张时间线：

```python
def train_trajectory_loop(model, tokenizer, task_pool, args, opt, sched, trainable,
                          writer, log_fp, traj_audit_fp, save_steps, eval_tasks=None):
    # 主干一句话：每个训练步 = 取批 → 生成组+优势+审计 → μ 轮重放反传
    #   → 每 20 步 greedy 考试存 best → 记日志 → 存档。各环节细节见各自函数。
    task_order = make_task_order(task_pool, getattr(args, "curriculum", False))
    # 任务出场顺序：课程化 = 各难度层轮转交织（L1 到 L5 依次循环），保证每步混有简单题。
    # 训练池 82% 是 Level 3-5 难题，不交织会大量出现"整组全错同分"的零梯度步。
    cursor = 0                      # 任务队列读到哪了（一个队列可以连续吃多步）
    best = {"success": -1.0, "reward": -1.0, "step": 0}  # 闭环评估历史最优 -> out/best
    for step in range(1, args.max_steps + 1):        # ← 训练主循环，一个 step 见第 9 章
        step_t0 = time.time()                        # 本步起始时刻（日志 sec 字段用）
        if cursor + args.prompts_per_step > len(task_order):
            # 任务队列吃到头：按同样的规则重新排一轮，再从头开始
            task_order = make_task_order(task_pool, getattr(args, "curriculum", False))
            cursor = 0
        task_batch = task_order[cursor:cursor + args.prompts_per_step]   # 本步要做的 P=2 道题
        cursor += args.prompts_per_step
```

（下面进入本步第一件事：采样。）

### 5.2 生成一组：generate_scored_group

生成与打分从主干里独立成了函数——同一道题采 G=4 条完整轨迹并打分，返回 `group = [(trajectory, reward, detail), ...]` 与一一对应的 `envs`：

```python
def generate_scored_group(model, tokenizer, task, args):
    # 同一道题采 G 条完整轨迹并打分（GRPO 的"组"）。返回 (group, envs)，
    # group = [(trajectory, reward, detail), ...]，envs 与 group 一一对应。
    if args.gen_batch > 1:
        # 效率路径：首回合 prompt 完全相同 → 按块批量采样（rollout_group，见 3.5；
        # 批量只换随机数流，统计等价）
        traj_envs = rollout_group(model, tokenizer, task, num_generations=args.num_generations,
                                   gen_batch=args.gen_batch, max_turns=args.max_turns,
                                   max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, top_p=args.top_p,
                                   max_len=args.max_len, do_sample=True,
                                   grad_ckpt=bool(args.grad_ckpt))
        group, envs = [], []
        for trajectory, env in traj_envs:
            reward, detail = score_trajectory(trajectory, env)
            group.append((trajectory, reward, detail))
            envs.append(env)
        return group, envs
    # 封存口径（默认 gen_batch=1）：逐条采样，随机数流与封存训练一致
    group, envs = [], []
    for _ in range(args.num_generations):
        # 每条轨迹必须新建一个 MathEnv：env 里有执行计数等状态，
        # 共用会让 4 条轨迹互相污染统计（2026-10-24 修复的坑，勿回退）
        env = MathEnv(task)
        # 让模型独立做这道题：写 python → 子进程真执行 → 看结果 → 继续推理 → \boxed 终答
        trajectory = rollout(model, tokenizer, env, max_turns=args.max_turns,
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_p=args.top_p,
                             max_len=args.max_len, do_sample=True,
                             grad_ckpt=bool(args.grad_ckpt))
        reward, detail = score_trajectory(trajectory, env)   # 裁判打分：0~1 分 + 对错明细
        group.append((trajectory, reward, detail))
        envs.append(env)
    return group, envs
```

主干里每道题就剩一行调用 + 审计落盘：

```python
        # ---- 1) 整条链路 rollout（组 = 同一任务 G 条完整轨迹）----
        rollouts = []   # 收集本步所有轨迹：(env, 轨迹, 优势, 得分, 打分明细)
        group_stats = []     # 每组的 mean/std/各条得分（写审计日志用）
        for task in task_batch:
            group, envs = generate_scored_group(model, tokenizer, task, args)
            # ↑ 生成 G 条完整轨迹并打完分（gen_batch 决定批量/逐条，见 3.5）
            torch.cuda.empty_cache()   # 生成阶段的 kv cache 归还驱动，给梯度阶段腾峰值
```

### 5.3 组内优势：GRPO 名字的由来

组内优势也独立成了纯函数（只做数学，不碰模型）：

```python
def group_advantages(rewards):
    # 组内相对优势（GRPO 名字的来源）：以组均值/标准差把绝对分变成相对分
    mean = sum(rewards) / len(rewards)     # 基线 = 组平均分
    std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
    # 优势 = 比组平均好多少个标准差（+1e-4 防除零）。
    # std≈0（G 条轨迹分数全同，全对或全错）→ 没有"谁更好"的信息 → 全部置 0，
    # 这组轨迹在更新阶段会被整体跳过（零梯度步）。
    advs = [(r - mean) / (std + 1e-4) for r in rewards] if std > 1e-6 \
        else [0.0] * len(rewards)
    return advs, mean, std
```

主干里调用它并把轨迹（带着各自优势）入列、写审计日志：

```python
            rewards = [reward for _trajectory, reward, _detail in group]              # G 条轨迹各自的得分
            advs, mean, std = group_advantages(rewards)  # 组内相对优势（上面刚定义）
            for (trajectory, reward, detail), adv, env in zip(group, advs, envs):
                rollouts.append((env, trajectory, adv, reward, detail))    # 轨迹带着各自的优势入列
            group_stats.append({"mean_r": round(mean, 4), "std_r": round(std, 4),
                           "rewards": [round(reward, 4) for reward in rewards]})
            # 轨迹审计日志：本组每条轨迹的得分/优势/完整可读文本（事后复盘用，训练不读它）
            traj_audit_fp.write(json.dumps({
                "step": step, "tid": task["id"], "category": task["category"],
                "group": [{"reward": round(reward, 4), "adv": round(adv, 3), "detail": detail,
                           "text": format_trajectory(env, trajectory)}
                          for (trajectory, reward, detail), adv, env in zip(group, advs, envs)],
            }, ensure_ascii=False) + "\n")
        traj_audit_fp.flush()                              # 逐组写完即落盘（训练中断不丢）
        details = [entry[4] for entry in rollouts]              # 全部打分明细（记日志用）
```

（每组平均分与"有分差组占比"原来在主干算好再传——重构后直接在 7.3 的 `log_step` 里从 `group_stats` 现算，字段含义不变。）

用具体数字走一遍。设某题 4 条轨迹得分 `[1.0, 1.0, 0.2, 0.1]`：

```
mean = 0.575                  ← 基线（这道题的"平均水平"）
std  = 0.427                  ← 波动大小
adv  = (r − mean) / std  =  [+1.00, +1.00, −0.88, −1.11]
```

读法：前两条"显著好于平均"，更新时**提升**它们整条链路的生成概率；后两条被**压低**。注意四个数加起来 ≈ 0——"组内相对"的含义就在这：**没有绝对好坏，只有相对同组的好坏。**

两个重要推论：

- **如果 4 条全对（reward 全 1.0）**：std=0 → 优势全 0 → 这题已经会了，不用学，整组跳过。这是特性不是 bug：GRPO 只学"相对差异"，太简单/太难的题都白跑——所以任务顺序要"难度层交织"，让每步大概率混进有分差的组。
- **减去均值的意义**："做对了"本身不值钱，"比同组其他人好"才值钱。这就是 GRPO 省掉"价值网络"（教科书 PPO 里预估每道题能得几分辅助网络）的办法：**用组平均分当基线**。

---

## 6. 主循环·下半场：μ 轮重放 + PPO 裁剪 + KL 锚定

这是全项目数学上最深的一段，慢慢来。背景：现在手里有 8 条轨迹，每条带一个优势值（正=该鼓励，负=该抑制）。要做的事：把"鼓励/抑制"落实到"每个 token 的生成概率"上，同时装两道保险。
更新阶段独立成了函数 `train_step_update()`（L172）。相对封存版最大的变化是支持 **μ 轮重放**（`--epochs`）：同一批 rollout 数据连续重放反传 μ 遍（PPO 的经典做法，InstructGPT 用 μ=4）——第 2 遍起权重已动而 old_logp 还是首遍读数，ratio 偏离 1、裁剪真正工作。这全靠把 old_logp/ref_logp **按 (轨迹号, 回合号) 缓存**。

### 6.1 函数入口：过滤、统计与 μ 轮循环

```python
def train_step_update(model, rollouts, trainable, opt, args):
    # 更新阶段：同一批 rollout 数据重放 args.epochs 遍（PPO 的 μ），逐回合反传
    stats = {"tot_loss": 0.0, "pol_sum": 0.0, "kl_sum": 0.0,
             "n_turn": 0, "n_trained": 0, "clip_hits": 0, "gnorm": 0.0}
    opt.zero_grad(set_to_none=True)   # 清空残留梯度（置 None 比置 0 更省内存）
    # 循环前过滤：组内全同分（组内优势为 0，无梯度）或一条 token 都没生成的轨迹整条跳过
    active_rollouts = []
    for _env, trajectory, adv, _r, _det in rollouts:
        if adv == 0.0:
            continue
        if sum(len(turn["gen_ids"]) for turn in trajectory["turns"]) == 0:
            continue
        active_rollouts.append((_env, trajectory, adv))
    n_rollouts = len(rollouts)          # backward 尺度基准与封存一致（含被跳过的轨迹）
    old_logp_cache, ref_logp_cache = {}, {}  # (轨迹号, 回合号) -> 首次重放的读数（本训练步内有效）
    pending_backwards = 0                # 已 backward 尚未 opt.step 的轨迹数（见 7.1）
    for _epoch in range(args.epochs):                  # ← μ 轮外循环：同一批数据重放 μ 遍
        for traj_i, (_env, trajectory, adv) in enumerate(active_rollouts):
            n_traj_tokens = sum(len(turn["gen_ids"]) for turn in trajectory["turns"])
            loss = None
```

**μ 的两种口径**（理解新旧代码差异的钥匙）：

| | μ=1（默认 = 封存口径） | μ>1（PPO 口径） |
|---|---|---|
| old_logp 怎么来 | 每回合重放时现算 | 首遍现算后**缓存**，后续遍复用 |
| ratio | 恒等于 1（act_logp−old_logp=0） | 第 2 遍起偏离 1（权重已动） |
| PPO 裁剪 | 永不触发（保险丝设计） | 真正工作，clip_tokens>0 |
| loss 日志 | 恒 ≈0（组内优势相消） | 不再恒 0 |
| 每训练步参数移动量 | 基准 | ≈ ×μ（想保持可比可减 --max-steps） |
| 每回合前向次数 | 3（ref_logp/old_logp/act_logp） | 首遍 3 + 后续遍 1（读缓存） |

μ=1 时缓存"写入但永不读出"，整条路径与封存代码**逐位一致**——这就是默认值定成 1 的原因。

### 6.2 逐回合重放与三个读数 old_logp / ref_logp / act_logp

```python
            for turn_j, turn in enumerate(trajectory["turns"]):    # 逐回合重放（一条轨迹常有多个回合）
                n_gen_tokens = len(turn["gen_ids"])
                if n_gen_tokens == 0:
                    continue
                # 把"当时的输入 + 当时生成的内容"拼回完整序列（批量=1）。
                # 重放不做批量化：[k, L, 248k] 的 fp32 logits 一拼就是 GB 级峰值，
                # batch=1 + logits_to_keep 是 8GB 显存红线的核心防线，勿改
                full_ids = torch.tensor([list(turn["prompt_ids"]) + list(turn["gen_ids"])],
                                        device=model.device)
                attention_mask = torch.ones_like(full_ids)    # 全 1 注意力掩码：没有 padding，全位置参与
                cache_key = (traj_i, turn_j)                   # 缓存键：第 i 条轨迹的第 j 个回合
                if cache_key not in ref_logp_cache:
                    with torch.no_grad(), model.disable_adapter():
                        # ref_logp：摘掉 LoRA（=SFT 参考模型）时的读数，KL 锚定用，不反传。
                        # disable_adapter 只是临时摘下小矩阵，出了 with 块就装回去
                        ref_logp_cache[cache_key] = token_logprobs(model, full_ids, attention_mask, n_gen_tokens)
                ref_logp = ref_logp_cache[cache_key]
                if cache_key in old_logp_cache:
                    old_logp = old_logp_cache[cache_key]         # μ>1：后续遍直接复用首遍读数
                else:
                    with torch.no_grad():
                        # old_logp：更新前模型对这些 token 的读数（PPO 的"旧概率"，不反传）。
                        # 首次重放时现算（此刻权重=生成时权重，与封存口径逐位一致）；
                        # μ>1 时冻结供后续遍复用——ratio 才会偏离 1
                        old_logp_cache[cache_key] = token_logprobs(model, full_ids, attention_mask, n_gen_tokens)
                    old_logp = old_logp_cache[cache_key]
                # act_logp：同样的前向，但带着梯度——真正会被优化的"新概率"
                # 三次读数来自同一函数，区别只有两点：带不带梯度、挂不挂 LoRA
                act_logp = token_logprobs(model, full_ids, attention_mask, n_gen_tokens)
```

| 名字 | 挂 LoRA？ | 带梯度？ | 缓存 | 角色 |
|---|---|---|---|---|
| old_logp | ✓ | ✗ | 首遍后缓存 | PPO 的"旧概率"——首遍重放时的读数 |
| ref_logp | ✗（=SFT 原模型） | ✗ | 首遍后缓存 | KL 锚定的参考——"你原来的样子"（底座恒定，缓存绝对安全） |
| act_logp | ✓ | ✓ | 不缓存（每遍都要重算） | "新概率"——真正被优化的量 |

（"不反传"= 包在 `torch.no_grad()` 里：只读数、不算梯度，省显存。为什么 old_logp 取"首遍重放读数"而不是"生成时读数"？首遍重放发生在本步第一次反传之前，此刻权重=生成时权重——两者就是同一个策略；按轨迹冻结首遍读数，还顺带保证了 μ=1 路径与封存代码逐位一致。）

### 6.3 KL 锚定：exp(d) − d − 1

```python
                # ---- KL 锚定（橡皮筋）：别离 SFT 参考模型太远 ----
                # d = 参考与当前的对数概率之差；exp(d)−d−1 是 KL 的 k3 估计量：
                #   恒 ≥0，两模型一致时=0，偏离越大惩罚越大（小偏离时 ≈ d²/2）。
                #   全程只用轨迹上的几个标量，不必对 24.8 万词表求和（省显存的关键活法）
                log_ratio = (ref_logp - act_logp).clamp(max=8.0)                       # 封顶防 exp 溢出
                kl_per_token = (log_ratio.exp() - log_ratio - 1).clamp(max=args.kl_clip)   # 单 token 封顶防离群
```

想要的效果：新模型别和 SFT 模型差别太大——不然 SFT 学的做题格式、通用语言能力都会被 RL 冲掉。度量"两个分布差多远"的标准量是 **KL 散度**，但真算 KL 要对 24.8 万词表求和，太贵。这里用 **k3 估计量**：只用轨迹里实际生成的那几个 token 来估。

- `d = log p_ref − log p_act`：参考模型与当前模型对同一个 token 的对数概率之差。
- `exp(d) − d − 1` 的性质（画图更直观）：**恒 ≥ 0**；d=0 时 =0；小偏离时 ≈ d²/2（温和），d 大时指数级增长（重罚）。
- 两道数值保险：`clamp(max=8)` 防 exp 上溢（e⁸≈3000）；`clamp(max=0.05)` 给单个 token 的 KL 封顶，防个别离群 token 主导整步梯度。
- `β=0.025`：橡皮筋劲度。每次想跑远都被往回拉一点——GRPO 能在 60 步里提升正确率又不毁掉 SFT 能力，靠的就是它。

### 6.4 PPO 裁剪：ratio 与信任区间

```python
                # ---- PPO 裁剪 ----
                # ratio = 新概率相对旧概率的倍数：exp(log p_new − log p_old)。
                #   μ=1（封存口径）：old_logp 背靠背现算，ratio 恒等于 1（LoRA dropout 会带
                #   来个位数 token 的轻微波动）——clip 是保险丝，从不真正触发；
                #   μ>1：old_logp 是首遍缓存读数，第 2 遍起权重已动、ratio 真正偏离 1——
                #   ratio 冲出 [1−ε, 1+ε]（默认 0.8~1.2）后，继续推的梯度被掐断，
                #   防止多遍重放累积的漂移把某 token 的概率改得太狠
                #   （日志 clip_tokens 即冲出信任带的 token 数：μ=1 通常 0，μ>1 是
                #   裁剪在工作的直接读数）。
                ratio = (act_logp - old_logp).exp()
                # 优化器只会"降损失"，所以把"提升优势"写成最小化 −ratio×adv；
                #   min(未裁剪目标, 裁剪后目标) 取较悲观者：变好过头就停手，变坏则照罚。
                policy_loss = -torch.min(ratio * adv,
                                         ratio.clamp(1 - args.epsilon, 1 + args.epsilon) * adv).mean()
```

- `ratio`：某个 token 的概率被更新改成了原来的几倍。ratio=1.3 = 概率涨了 30%。
- 想做的事很朴素：**优势为正的轨迹 → ratio 往大推；优势为负 → 往小推。** 写成损失就是 `-ratio×adv`（优化器只会降损失，加负号变"最大化奖励"）。
- **clip（裁剪）**：只许 ratio 在 `[0.8, 1.2]` 的"信任区间"里活动。跑出区间后，`min(未裁剪, 裁剪后)` 选出的目标变成常数，继续推的梯度被掐断——**防止单步更新把某个 token 的概率改得太狠**，把模型改坏。μ>1 时它从"保险丝"升级为"必需品"：多遍重放意味着数据是 off-policy 的，没有裁剪约束漂移就会失控。
- `.mean()`：对这个回合的所有 token 取平均。
- `clip_tokens` 读数：μ=1 大多是 0（偶见个位数来自 LoRA dropout 噪声）；μ>1 第 2 遍起应能看到非零——离线测试（场景 3）与 GPU 冒烟都能验证。

### 6.5 组装与反传

```python
                # 本回合损失 = PPO 项 + β×KL 项；×(L/n_traj_tokens) 按 token 数加权——长回合
                #   话语权大、短回合小，把整条轨迹的优势均匀摊到每个 token 上
                turn_loss = (policy_loss + args.beta * kl_per_token.mean()) * (n_gen_tokens / n_traj_tokens)
                loss = turn_loss if loss is None else loss + turn_loss
                stats["pol_sum"] += policy_loss.detach().item()            # 统计：本回合 PPO 项数值
                stats["kl_sum"] += kl_per_token.detach().mean().item()   # 统计：本回合 KL 数值
                stats["n_turn"] += 1                               # 统计：又重放了一个回合
                stats["clip_hits"] += int((ratio.detach() < 1 - args.epsilon).sum()
                                          + (ratio.detach() > 1 + args.epsilon).sum())
                # ↑ 统计：本回合冲出信任带 [1−ε, 1+ε] 的 token 数（日志 clip_tokens）
            if loss is None:
                continue
            # 反向传播：算出"每个 LoRA 参数该往哪边挪多少"。
            #   ×n：把"单条轨迹的平均损失"放大到整批（P×G 条）尺度；
            #   ÷grad_accum：攒 accum 条才 step 一次，每条只出 1/accum 的力——
            #   两者配合，有效步长与轨迹数/累积数无关，不会忽大忽小。
            (loss * n_rollouts / args.grad_accum).backward()
            stats["tot_loss"] += loss.detach().item()   # 统计：本条轨迹的总损失
            stats["n_trained"] += 1                     # 统计：真正反传的轨迹数（跳过的不算）
            pending_backwards += 1                                # 已反传、还没优化过：又攒了一条 → 见 7.1
```

- **`×(L/n_traj_tokens)` 加权**：一条轨迹的多个回合长短不一，按 token 数加权，长回合影响大、短回合小。
- **`×n/grad_accum`**：见下面 7.1 梯度累积。
- **统计口径**：统计量收进 `stats` 字典返回（记日志用）。μ>1 时 `n_turn`/`n_trained` 是 μ 倍（每遍都累计），`loss`/`policy`/`kl` 是按遍平均后的值——同一批数据第 2 遍起 ratio≠1，各遍数值自然不同。

---

## 7. 收尾三件事：梯度累积的坑、训练中考试、记日志

### 7.1 梯度累积：8GB 显存装不下一次大批量

```python
            if pending_backwards == args.grad_accum:   # 攒够 4 条（pending_backwards 在 6.5 末尾 +1）
                # 先把梯度总范数裁到 max_grad_norm（2.0）以内（防个别步梯度爆炸），
                # 再真正更新参数
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                stats["gnorm"] = max(stats["gnorm"], float(grad_norm))   # 记录本步最大范数（日志 grad_norm）
                opt.step()
                opt.zero_grad(set_to_none=True)
                pending_backwards = 0
                torch.cuda.empty_cache()     # WDDM：优化步后把空闲缓存块归还驱动
```

- **为什么攒？** 8GB 显存装不下一次反传 P×G=8 条长序列。一条一条来，`backward()` 会自动把每条的梯度**累加**到参数的 `.grad` 里，攒够 4 条做一次 `opt.step()`，效果≈大批量。μ>1 时 pending_backwards **跨 epoch 连续计数**——第 1 遍攒了 2 条、第 2 遍再攒 2 条就凑满一次 step，优化步时机与遍数边界无关。
- **`clip_grad_norm_`**：所有梯度的总"长度"（范数）超过 2.0 就整体等比缩小。封存日志里实测 grad_norm 常态 2~13，不裁剪会步子过大。

**2026-10-24 修过的两个坑**（源码里有注释，抄写时注意）：

1. 旧版按轨迹下标 `(i+1) % accum` 触发优化步。若一步末尾的轨迹因"优势为 0"被跳过，凑不满 accum，已攒的梯度会拖到下一步、被下一步开头的 `zero_grad` **静默丢掉**。修复：按"实际反传了几条"（`pending_backwards` 计数）触发，循环结束后有残留也补一次。
2. 学习率日程（cosine）原来按优化步推进，但优化步次数 = P×G/accum，超参数一变日程就变形（cosine 半程就衰减到 0）。修复：**每个训练步推进一次** `sched.step()`。

坑 1 修复后的收尾代码（在 `train_step_update` 循环结束后，原文照抄；返回 stats 收官）：

```python
    # 修复（2026-10-24）：旧版按 enumerate 下标 (i+1)%accum 触发优化步，若一步末尾
    # 的轨迹因 adv==0 被跳过，已累积的梯度会在下一步开头被 zero_grad 静默丢弃；
    # 改为按实际 backward 计数触发，循环结束后仍有残留则补一次优化步。
    if pending_backwards > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        stats["gnorm"] = max(stats["gnorm"], float(grad_norm))
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    return stats
```

坑 2 对应的学习率推进则挪回了主干（`sched` 是主干传进来的对象，日程策略留在骨架层）：

```python
        # ---- 2) 更新：同一批 rollout 数据重放 args.epochs 遍（μ=1 即封存口径）----
        stats = train_step_update(model, rollouts, trainable, opt, args)
        if stats["n_trained"]:
            # 学习率按训练步推进：优化步次数随 P×G/accum/μ 变化，若按优化步推进，
            # cosine 会在半程就衰减到 0（2026-10-24 修复口径的延续）
            sched.step()
```

### 7.2 训练中考试：run_closed_eval

先看考试函数本体（源码 `train_grpo.py` L83）：

```python
def run_closed_eval(model, tokenizer, tasks, args):
    """训练中的闭环评估：greedy 走完整链路，与 evaluation/eval_math.py 同口径
    （每任务独立 MathEnv + score_trajectory）。只读不反传；生成期间 rollout 内部
    自行切换推理态并恢复训练态。返回 (success_rate, reward_mean)。"""
    n_correct, rewards = 0, []
    for task in tasks:
        env = MathEnv(task)
        trajectory = rollout(model, tokenizer, env, max_turns=args.max_turns,
                             max_new_tokens=args.max_new_tokens,
                             temperature=1.0, top_p=1.0, max_len=args.max_len,
                             do_sample=False, grad_ckpt=bool(args.grad_ckpt))
        #                       ↑ do_sample=False 就是 greedy：每步选概率最高的 token
        reward, detail = score_trajectory(trajectory, env)
        rewards.append(reward)
        n_correct += int(detail["success"])
        torch.cuda.empty_cache()      # 每题考完就还显存
    return n_correct / len(tasks), sum(rewards) / len(tasks)
```

再看重放主循环里怎么调用它：

```python
        # ---- 2.5) 训练中闭环评估：greedy 固定子集（与 eval_math 同口径），追踪最优 ----
        last_eval = None                                    # 本步有没有考过试
        if eval_tasks and step % args.eval_every == 0:      # 每 20 步
            eval_success, eval_reward = run_closed_eval(model, tokenizer, eval_tasks, args)  # greedy 考 16 道持出题
            last_eval = (eval_success, eval_reward)
            print(f"  [eval@{step:3d}] success={eval_success:.3f} reward={eval_reward:.3f}", flush=True)
            writer.add_scalar("grpo/eval_success", eval_success, step)   # 考试成绩也画进 TensorBoard
            writer.add_scalar("grpo/eval_reward", eval_reward, step)
            if (eval_success, eval_reward) > (best["success"], best["reward"]):   # 元组比较：先比 success，并列再比 reward
                best.update(success=eval_success, reward=eval_reward, step=step)
                model.save_pretrained(os.path.join(args.out, "best"))   # 考出新纪录就存档
                tokenizer.save_pretrained(os.path.join(args.out, "best"))
                print(f"  当前评估最优，已保存 -> {os.path.join(args.out, 'best')}", flush=True)
```

训练的 reward 是"采样分"（带随机性），不能直接当模型水平。所以每 20 步用 **greedy**（不随机，每步都选概率最高的 token）在 16 道**持出题**上考一次，刷新纪录就把 adapter 存进 `out/best`。封存那次训练：step20 考 0.5、step40 考 0.5、step60 考 0.438 → **best=step-20**（并列取先）。最终 100 题评测 44.0% 用的就是它。

### 7.3 记日志

记日志也独立成了函数 `log_step()`（源码 L271）：构造 `record`（=日志一行）→ jsonl 落盘 → 打印 → TensorBoard。每组平均分与"有分差组占比"直接从 `group_stats` 现算（见 5.3 末尾的说明）：

```python
def log_step(writer, log_fp, args, step, step_t0, details, group_stats, stats, last_eval, sched):
    # 按训练步记录。μ=1 时 ratio 恒 1 → policy 项各组求和为 0（组内优势之和为 0），
    # kl 起步也近 0，学习信号看 grad_norm 与 success_rate；μ>1 时 ratio 偏离 1，
    # loss 不再恒 0，clip_tokens 是裁剪是否触发的直接读数
    total_exec = sum(detail["n_exec"] for detail in details)      # 全部轨迹的执行总次数
    record = {"step": step, "lr": round(sched.get_last_lr()[0], 9),
           "reward_mean": round(sum(g["mean_r"] for g in group_stats) / len(group_stats), 4),
           "valid_group_frac": round(sum(1 for g in group_stats if g["std_r"] > 1e-6)
                                     / len(group_stats), 3),
           "success_rate": round(sum(1 for d in details if d["success"]) / len(details), 3),
           "avg_turns": round(sum(d["n_turns"] for d in details) / len(details), 2),
           "exec_err": round(sum(detail["n_error"] for detail in details) / max(total_exec, 1), 3),
           "policy": round(stats["pol_sum"] / max(stats["n_turn"], 1), 4),
           "kl": round(stats["kl_sum"] / max(stats["n_turn"], 1), 5),
           "loss": round(stats["tot_loss"] / max(stats["n_trained"], 1), 4),
           "grad_norm": round(stats["gnorm"], 3), "clip_tokens": stats["clip_hits"],
           "epochs": args.epochs,
           "sec": round(time.time() - step_t0, 1),
           "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    # ↑ 全是聚合统计：除法分母用 max(计数, 1) 防除零（一条轨迹都没反传时也不会崩）
    if last_eval is not None:                    # 这步考过试，就把考试成绩也记进去
        record["eval_success"] = round(last_eval[0], 3)
        record["eval_reward"] = round(last_eval[1], 3)
    log_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    log_fp.flush()                               # 立即落盘：训练中断日志不丢
    print(f"[step {step:3d}/{args.max_steps}] reward={record['reward_mean']:.3f} "
          f"success={record['success_rate']:.0%} turns={record['avg_turns']} "
          f"exec_err={record['exec_err']:.0%} kl={record['kl']:.4f} loss={record['loss']:.4f} "
          f"gnorm={record['grad_norm']:.2f} lr={record['lr']:.2e} "
          f"mem={record['mem_gb']}GB ({record['sec']}s)", flush=True)
    for k in ("reward_mean", "valid_group_frac", "success_rate", "avg_turns",
              "exec_err", "kl", "policy", "loss", "grad_norm"):
        writer.add_scalar("grpo/" + k, record[k], step)      # 指标逐个画进 TensorBoard
    writer.add_scalar("grpo/lr", record["lr"], step)
    writer.add_scalar("grpo/mem_gb", record["mem_gb"], step)
```

主干里调用它，后面紧跟存档与收尾（主干原文照抄）：

```python
        # ---- 3) 按步记录（record 构造 / jsonl 落盘 / 打印 / TensorBoard 见 log_step）----
        log_step(writer, log_fp, args, step, step_t0, details, group_stats, stats, last_eval, sched)

        if step % save_steps == 0 or step == args.max_steps:  # 每 10 步 / 最后一步存档
            ckpt_dir = os.path.join(args.out, f"checkpoint-{step}")
            model.save_pretrained(ckpt_dir)
            print(f"  已保存 {ckpt_dir}", flush=True)

    # for 循环（所有训练步）到此结束。跑完报一声 best 是哪一步
    if best["step"]:
        print(f"闭环评估最优: step {best['step']} success={best['success']:.3f} "
              f"reward={best['reward']:.3f} -> {os.path.join(args.out, 'best')}", flush=True)
```

字段速查：

| 字段 | 意思 | 怎么用 |
|---|---|---|
| `reward_mean` / `success_rate` | 每组平均分 / 端到端做对率 | 看 20-30 步窗口的趋势 |
| `valid_group_frac` | 有分差的组占比 | 越低说明信号越少（全对/全错的组多） |
| `exec_err` | 执行报错率 | 行为质量指标 |
| `grad_norm` | **学习信号主指标** | 恒 0 = 这步没更新（零梯度步） |
| `loss` / `kl` | μ=1 接近 0 属正常（组内优势相消）；μ>1 不再恒 0 | μ=1 时**不要**拿它们判断学习 |
| `clip_tokens` | 冲出 PPO 信任带的 token 数 | μ=1 通常 0（保险丝）；μ>1 非零=裁剪在工作 |
| `epochs` | 本 run 的 μ（每次写入，日志自带口径标记） | 区分封存数据与 μ>1 的新数据 |
| `mem_gb` / `sec` | 显存峰值 / 耗时 | 盯 8GB 红线 |

`logs/grpo_trajs_math.jsonl` 同时记录每组 4 条轨迹的完整明细（得分/优势/可读链路文本），事后用 `python -m scripts.tools.grpo_report` 出复盘报告。

---

## 8. 8GB 显存生存术（代码里的省显存手段清单）

| 手段 | 代码位置 | 省的是什么 |
|---|---|---|
| LoRA + 底座冻结 | main() | 只训 0.5% 参数，梯度和优化器状态只给小矩阵 |
| NF4 4bit 量化（双重量化） | `load_model_4bit()` | 底座权重 9.3GB → ~2.6GB |
| 摘除视觉塔 | `drop_vision_tower()` | Qwen3.5 是 VL 模型，视觉塔用不上白占显存 |
| 冻结权重压回 bf16 | `shrink_frozen_fp32()` | 加载时偷偷升 fp32 的冻结权重压回去（248k 词表 embedding 是最大一块） |
| `logits_to_keep` | `token_logprobs()` | 只对补全段出 logits（省 ~1.5GB） |
| 重放恒 batch=1 | `train_step_update()` | [k, L, 248k] 的 fp32 logits 一批就是 GB 级峰值——生成可批、重放不批，这是 8GB 下的效率分界线 |
| 首回合组批量采样 `--gen-batch`（默认关） | `rollout_group()` | 同题 G 条 prompt 相同，一次批量出 N 条：提效不加批（KV 仅 8 层全注意力，增量百 MB 级） |
| KL 用 `disable_adapter()` 现算 | 更新循环 | 不用加载第二个参考模型 |
| `empty_cache()`（每步/每阶段） | 多处 | WDDM 下把空闲缓存块还给驱动——否则显存页被换出到内存，每步从秒级恶化到分钟级（"假死"） |
| 梯度检查点 | `--grad-ckpt` | 用"反传时重算前向"换显存 |
| 非 paged `adamw_8bit` | main() | 优化器状态 8bit；不用 paged 版（会在 CPU/GPU 间搬页，假死元凶） |
| 单实例锁 `outputs/.train.lock` | `acquire_lock()` | 同一时刻只许一个训练/评测进程 |

---

## 9. 一次训练步完整时间线（串起来）

```
train_grpo.py main()
 ├─ acquire_lock() 锁显卡
 ├─ load_model_4bit() 加载 NF4 量化模型 → drop_vision_tower() 摘视觉塔
 ├─ shrink_frozen_fp32() 冻结权重压 bf16 → 挂 LoRA（get_peft_model）
 ├─ AdamW8bit 优化器 + cosine 学习率日程（先热身 5% 再缓缓降）
 └─ train_trajectory_loop()  ←——————— 每个训练步 step ———————
     ├─ ① make_task_order：难度层交织抽 P=2 道题
     ├─ ② 上半场：generate_scored_group——每题 G=4 条完整轨迹并打分
     │     ├─ gen_batch>1：rollout_group 首回合批量采样（3.5）；=1：逐条 rollout（3.1）
     │     └─ group_advantages 组内优势 → 轨迹带优势入列 → 审计落盘
     ├─ ③ 下半场：train_step_update——同批数据重放 μ 遍（--epochs，默认 1）
     │     └─ 每遍逐轨迹逐回合：ref_logp/old_logp（首遍缓存）→ KL(k3) → PPO(ratio+clip)
     │        → 损失 ×n/accum → backward 攒梯度 → 每 4 条 clip_grad_norm + opt.step()
     ├─ ④ 每 20 步：run_closed_eval（greedy 考试 16 题）→ 刷新纪录存 out/best
     └─ ⑤ log_step 写 logs/grpo_log_math.jsonl + TensorBoard；每 10 步存 checkpoint-N
```

---

## 10. 名词小词典

| 词 | 一句话解释 |
|---|---|
| token | 模型处理文本的最小单位（字/词片/符号） |
| logits | 模型对"下一个 token"的原始打分（未归一化） |
| softmax / log_softmax | 打分 → 概率（和为 1）；取 log 便于加减 |
| log p | 对数概率；越接近 0 越确定 |
| LoRA | 冻结大矩阵，旁边挂小矩阵 A×B 只训它们 |
| disable_adapter | 临时摘掉 LoRA = 退回 SFT 原模型 |
| rollout | 让模型完整走一遍任务（这里 = 做完一道题） |
| 轨迹 trajectory | 一条完整的解题过程（多回合） |
| reward | 一条轨迹的 0~1 得分 |
| 优势 advantage | 比组平均好多少个标准差（正=鼓励，负=抑制） |
| GRPO | 组内相对策略优化：用组内平均分当基线，省掉价值网络 |
| μ（--epochs） | 同一批 rollout 数据的重放遍数；μ>1 时 ratio 偏离 1、PPO 裁剪真正工作 |
| gen-batch | 首回合组内批量采样数（同题 prompt 相同可批量）；1=逐条封存口径 |
| on-policy / off-policy | 数据由"当前策略"产出=on-policy；重放 μ>1 遍时权重已动，数据变 off-policy——裁剪就是为此准备的 |
| PPO clip | ratio 限制在 [0.8, 1.2]，防单步改太狠 |
| KL 散度 | 两个分布差多远的度量；这里当"橡皮筋" |
| k3 估计量 | exp(d)−d−1，KL 的廉价无偏估计 |
| backward | 反向传播：算每个参数的梯度 |
| opt.step() | 按梯度真的更新参数 |
| 梯度累积 | 多条轨迹的梯度攒一批再 step（小显存跑大批） |
| clip_grad_norm | 梯度总范数超限就等比缩小，防爆炸 |
| temperature | 采样随机度；0.8 = 比完全贪心多样、比 1.0 稳 |
| greedy | 每步都选概率最高的 token（评测用，可复现） |
| detach() | 取数值不取梯度（统计用） |
| no_grad | 不记录梯度图，省显存 |
| kv cache | 生成时缓存历史注意力结果；开=快、占显存 |
| WDDM 假死 | 显存打满 → Windows 把显存页换到内存 → 每步从秒变分钟 |

---

## 11. 常见疑问（用封存日志 `logs/grpo_log_math.jsonl` 的真实数据回答）

**Q1：日志里 loss / kl 接近 0，训练坏了吗？**
没坏（μ=1 口径）。优势是组内归一化的，一组内正负相加 = 0，loss 平均后 ≈ 0；kl 有信号时也只有 0.001~0.004 的量级；零梯度步（`valid_group_frac=0`）里两者恰为 0。**学习信号看 `grad_norm`（封存日志实测 2~13）和 `success_rate`**，趋势用 20-30 步窗口看。μ>1 时 ratio 偏离 1，loss 不再恒 0，但判断趋势仍然看 grad_norm 与 success_rate。

**Q2：全对的组为什么不更新？**
组内人人同分 → 优势全 0 → 没有梯度。GRPO 只学"相对差异"，已经会了的题自动退出训练。代价是太简单/太难的题都白跑——所以任务顺序做了难度层交织。

**Q3：没有"输出层打标签"，"鼓励这条轨迹"到底是怎么发生的？**
靠概率移动。优势 +1 的轨迹，它所有生成 token 的 log p 被推高；优势 −1 的被压低。下一次采样时，"好做法"里的 token 序列更容易被采到——仅此而已，没有别的机制。

**Q4：temperature=0.8 是干嘛的？**
采样随机度。1.0 = 完全按模型自己的分布抽（实测显著压低 SFT 模型的采样正确率，组内全错同分的零梯度步变多）；越小越确定。训练用 0.8（要多样性做组内比较），评测用 greedy（可复现）。

**Q5：clip_tokens 为什么大多是 0？**
分口径。μ=1（默认/封存口径）：old_logp 是重放时现算的，与 act_logp 背靠背、权重相同，ratio 出发点恒为 1——偶见 1~4 个是 LoRA dropout 的随机噪声把个别 token 推出了 ±20% 信任带，clip 是保险丝设计。μ>1：第 2 遍起 old_logp 是首遍缓存读数而权重已动，ratio 真正偏离 1，clip_tokens 非零就是裁剪在工作的直接证据（离线测试 `test_grpo_update.py` 场景 3 与 GPU 冒烟都能复现）。

**Q6：训练中还要"考试"（run_closed_eval），为什么？**
训练 reward 是采样分（带随机性），不能直接当模型水平。每 20 步在 16 道持出题上 greedy 考一次，刷新纪录存 `out/best`——最终 best=step-20 就是这么选出来的（step40 并列 0.5 但不覆盖前者）。

---

*本文与源码一一对应：`train_grpo.py`（522 行）的 `token_logprobs` / `group_advantages` / `generate_scored_group` / `train_step_update` / `log_step` / `train_trajectory_loop` / `main`，`math_env.py`（375 行）的 `generate_once` / `rollout` / `rollout_group` / `score_trajectory` / `run_python_code`。有出入以源码为准。*
