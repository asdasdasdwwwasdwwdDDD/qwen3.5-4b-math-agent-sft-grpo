# -*- coding: utf-8 -*-
"""杀掉 GRPO 训练进程（假死自愈/用户叫停用）。用法（仓库根）:
python -m scripts.tools.train_kill"""
import subprocess
import sys
import time

PS = ("Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
      "Where-Object {$_.CommandLine -match 'train_grpo'} | "
      "ForEach-Object {$_.ProcessId}")
NOWIN = 0x08000000


def live_pids():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                       capture_output=True, text=True, timeout=90,
                       creationflags=NOWIN)
    return [int(x) for x in r.stdout.split() if x.strip().isdigit()]


pids = live_pids()
print("found:", pids or "none", flush=True)
for p in pids:
    subprocess.run(["taskkill", "/PID", str(p), "/F"], capture_output=True,
                   timeout=60, creationflags=NOWIN)
    print("killed", p, flush=True)
time.sleep(4)
left = live_pids()
print("left after kill:", left or "none", flush=True)
sys.exit(1 if left else 0)
