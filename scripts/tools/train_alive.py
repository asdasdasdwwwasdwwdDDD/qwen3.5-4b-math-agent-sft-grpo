# -*- coding: utf-8 -*-
"""报告 GRPO 训练进程是否存活（只查不杀）。退出码 0=在跑 1=不在。
用法（仓库根）: python -m scripts.tools.train_alive"""
import subprocess
import sys

PS = ("Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
      "Where-Object {$_.CommandLine -match 'train_grpo'} | "
      "ForEach-Object {$_.ProcessId}")
r = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                   capture_output=True, text=True, timeout=90,
                   creationflags=0x08000000)  # CREATE_NO_WINDOW
pids = [x.strip() for x in r.stdout.split() if x.strip().isdigit()]
print("train_alive:", pids or "none")
sys.exit(0 if pids else 1)
