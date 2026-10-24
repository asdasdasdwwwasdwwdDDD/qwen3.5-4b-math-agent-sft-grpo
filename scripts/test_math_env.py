# -*- coding: utf-8 -*-
"""math_env 纯函数自测：受限执行 / boxed 提取 / 答案等价 / 代码块提取。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math_env import (run_python_code, extract_boxed, normalize_answer,
                      answers_equal, extract_last_python_block)

# ---- run_python_code ----
out, ok = run_python_code("from sympy import Rational\nprint(Rational(4,5)+Rational(1,5))")
assert ok and out.strip() == "1", (ok, out)
out, ok = run_python_code("1/0")
assert not ok and "ZeroDivisionError" in out, (ok, out)
out, ok = run_python_code("import time\ntime.sleep(60)")
assert not ok and "Timeout" in out, (ok, out)
out, ok = run_python_code("x=1")  # 无输出
assert ok and out == "", (ok, repr(out))
print("run_python_code OK")

# ---- extract_boxed / extract_last_python_block ----
t = "The answer is $\\boxed{\\frac{1}{2}}$."
assert extract_boxed(t) == "\\frac{1}{2}"
assert extract_boxed("no box") is None
assert extract_last_python_block("```python\nprint(1)\n```") == "print(1)"
assert extract_last_python_block("```python\nprint(1)") is None  # 未闭合
assert extract_last_python_block("text```python\na=1\n``` mid ```python\nb=2\n```") == "b=2"
print("extract OK")

# ---- normalize / answers_equal ----
cases = [
    ("4.0", "4", True),
    ("\\frac{1}{2}", "0.5", True),
    ("2\\sqrt{3}", "2*sqrt(3)", True),
    ("\\frac{\\pi}{2}", "\\frac{\\pi}{2}", True),
    ("1,000", "1000", True),
    ("3", "4", False),
    ("\\text{even}", "even", True),
    ("x\\in[0,3)", "[0,3)", True),
    ("2\\pi", "6.283185307179586", True),
]
for pred, gt, want in cases:
    got = answers_equal(pred, gt)
    assert got == want, (pred, gt, got, normalize_answer(pred), normalize_answer(gt))
print("answers_equal OK")

print("ALL PASS")
