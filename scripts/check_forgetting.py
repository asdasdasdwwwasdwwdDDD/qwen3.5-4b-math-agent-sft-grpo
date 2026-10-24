# -*- coding: utf-8 -*-
"""灾难性遗忘检测：微调前后对比
1) 25 题通用探针（数学/逻辑/常识/代码/翻译/古诗，均与训练域无关），贪心生成后判分
2) 6 段跨领域文本的困惑度 PPL（teacher-forcing 全 token）
3) 过度特化探测：prompt 中不含任何工具定义时问天气，统计模型是否仍输出工具调用标记

用法:
  python scripts/check_forgetting.py --model models/Qwen3.5-4B \
      --adapter outputs/qlora-adapter/final
"""
import argparse
import json
import os
import re

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import PeftModel

TOOL_MARK = "<" + "tool_call>"   # 拼接以避免被内容管道处理

SYS = "你是一个乐于助人的助手。"

# ---------------- 通用知识探针 ----------------
# (问题, 判定函数名, 参数)
PROBES = [
    # 数学
    ("小明有5个苹果，又买了3袋，每袋12个，现在一共有多少个苹果？", "num", "41"),
    ("一件衣服原价200元，打八折后是多少元？", "num", "160"),
    ("计算 1/2 + 1/3 等于多少？用分数表示。", "any", ["5/6", "0.833", "六分之五"]),
    ("一个长方形长8厘米、宽5厘米，周长是多少厘米？", "num", "26"),
    ("3的4次方等于多少？", "num", "81"),
    ("计算 3 + 5 * 2 的结果是多少？", "num", "13"),
    ("5个人两两握手一次，一共要握多少次手？", "num", "10"),
    ("1千米等于多少米？", "num", "1000"),
    # 逻辑
    ("所有的玫瑰都是花，有些花很快凋谢。能否必然推出\u201c有些玫瑰很快凋谢\u201d？回答能或不能。",
     "any", ["不能", "无法", "不一定"]),
    ("甲比乙高，乙比丙高，三人中谁最矮？", "any", ["丙"]),
    ("找规律：2, 4, 8, 16, 下一个数是多少？", "num", "32"),
    # 常识
    ("中国的首都是哪座城市？", "any", ["北京"]),
    ("《红楼梦》的作者是谁？", "any", ["曹雪芹"]),
    ("水的化学式是什么？", "any", ["H2O", "H₂O"]),
    ("光在真空中的速度约为每秒多少千米？", "any", ["30万", "3×10^8", "3亿", "299792", "3*10"]),
    ("一年中除春分外，另一个昼夜几乎等长的节气是什么？", "any", ["秋分"]),
    ("地球绕太阳公转一周大约需要多少天？", "num", "365"),
    ("太阳从哪个方向升起？", "any", ["东"]),
    ("古诗\u201c床前明月光\u201d的下一句是什么？", "any", ["疑是地上霜"]),
    ("成语\u201c胸有成竹\u201d最初与画哪种植物有关？", "any", ["竹"]),
    # 代码
    ("用Python写一个函数 reverse_string(s)，返回字符串s的反转。只给代码。", "all",
     ["def", "reverse_string", ["[::-1]", "reversed", "reverse"]]),
    ("Python中用什么关键字定义匿名函数？", "any", ["lambda"]),
    ("写一条SQL语句，查询 users 表的所有记录。", "all", ["select", "users", "*"]),
    # 翻译/语言
    ("把英文\u201cI look forward to your reply.\u201d翻译成中文。", "any", ["期待", "回复", "盼望"]),
    ("简述牛顿第一定律的内容（一句话即可）。", "any", ["惯性", "静止", "匀速"]),
]

# ---------------- PPL 探针文本（跨领域） ----------------
PPL_TEXTS = [
    "数字经济时代，数据已成为继土地、劳动力、资本、技术之后的第五大生产要素。"
    "数据要素的市场化配置，能够显著提升全要素生产率，推动产业结构向中高端迈进。"
    "同时也要看到，数据确权、定价与流通机制尚不健全，隐私保护与安全治理面临新的挑战。",
    "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n"
    "    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n"
    "    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)",
    "Two things are infinite: the universe and human stupidity; and I'm not sure about the universe. "
    "Science without religion is lame, religion without science is blind.",
    "患者男，45岁，因反复上腹痛3个月就诊。胃镜示十二指肠球部溃疡，幽门螺杆菌检测阳性。"
    "建议给予质子泵抑制剂联合两种抗生素的四联疗法根除治疗，疗程十四天，治疗后复查。",
    "落霞与孤鹜齐飞，秋水共长天一色。渔舟唱晚，响穷彭蠡之滨；雁阵惊寒，声断衡阳之浦。"
    "遥襟甫畅，逸兴遄飞。爽籁发而清风生，纤歌凝而白云遏。",
    "根据《中华人民共和国民法典》第五百七十七条，当事人一方不履行合同义务或者履行合同义务不符合约定的，"
    "应当承担继续履行、采取补救措施或者赔偿损失等违约责任。",
]

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def load_model(model_path, adapter=None):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=True)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def gen_answer(model, tok, question, max_new=96):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": question}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_dict=False, enable_thinking=False)
    with torch.no_grad():
        out = model.generate(torch.tensor([ids], device=model.device),
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][len(ids):], skip_special_tokens=True)


def score(kind, param, text):
    t = text.replace(",", "").replace("，", "")
    if kind == "num":
        return any(v == param for v in NUM_RE.findall(t))
    if kind == "any":
        return any(k in text for k in param)
    if kind == "all":
        return all(any(k2 in text for k2 in ([k] if isinstance(k, str) else k))
                   for k in param)
    return False


@torch.no_grad()
def ppl_of(model, tok, texts):
    total_nll, total_n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        out = model(input_ids=ids, labels=ids)
        total_nll += out.loss.float().item() * (ids.shape[1] - 1)
        total_n += ids.shape[1] - 1
    import math
    return math.exp(total_nll / total_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/Qwen3.5-4B")
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--out", type=str, default="outputs/forgetting")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    weather_q = "明天北京天气怎么样？"
    # 无任何工具定义的通用对话 prompt
    weath_msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": weather_q}]

    results = {}
    for tag, adapter in [("base", None), ("finetuned", args.adapter)]:
        print(f"\n########## 测试模型: {tag} ##########", flush=True)
        model, tok = load_model(args.model, adapter)
        correct, answers = 0, []
        for q, kind, param in PROBES:
            ans = gen_answer(model, tok, q)
            ok = score(kind, param, ans)
            correct += ok
            answers.append({"q": q[:30], "ok": ok, "ans": ans[:80]})
            print(f"  [{'✓' if ok else '✗'}] {q[:28]}... -> {ans[:48].strip()}")
        qa_acc = correct / len(PROBES)
        ppl = ppl_of(model, tok, PPL_TEXTS)
        ids = tok.apply_chat_template(weath_msgs, tokenize=True, add_generation_prompt=True,
                                      return_dict=False, enable_thinking=False)
        out = model.generate(torch.tensor([ids], device=model.device), max_new_tokens=80,
                             do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        spillover = tok.decode(out[0][len(ids):], skip_special_tokens=True)
        spurious_call = TOOL_MARK in spillover
        results[tag] = {"qa_acc": round(qa_acc, 4), "correct": correct, "total": len(PROBES),
                        "ppl": round(ppl, 3), "spurious_tool_call": spurious_call,
                        "spill_sample": spillover[:100], "answers": answers}
        print(f"  => QA准确 {correct}/{len(PROBES)} ({qa_acc:.0%}) | PPL {ppl:.2f} | "
              f"无工具场景误发调用: {spurious_call}")
        del model
        torch.cuda.empty_cache()

    with open(os.path.join(args.out, "forgetting.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    b, ft = results["base"], results["finetuned"]
    print("\n===== 灾难性遗忘检测结论 =====")
    print(f"通用QA准确率: base {b['qa_acc']:.0%} -> ft {ft['qa_acc']:.0%} "
          f"(Δ {ft['qa_acc'] - b['qa_acc']:+.0%})")
    print(f"通用文本PPL  : base {b['ppl']:.2f} -> ft {ft['ppl']:.2f} "
          f"(升幅 {(ft['ppl']/b['ppl']-1):+.1%})")
    print(f"无工具场景误发工具调用: base={b['spurious_tool_call']} ft={ft['spurious_tool_call']}")
    drop = b["qa_acc"] - ft["qa_acc"]
    ppl_rise = ft["ppl"] / b["ppl"] - 1
    if drop <= 0.04 and ppl_rise <= 0.30 and not ft["spurious_tool_call"]:
        print("判定: 未见灾难性遗忘（通用能力基本保持）")
    elif drop <= 0.12 and ppl_rise <= 0.60:
        print("判定: 轻度能力漂移，建议混入通用指令数据继续训练")
    else:
        print("判定: 存在明显灾难性遗忘，需降低lr/混入通用数据/早停")


if __name__ == "__main__":
    main()
