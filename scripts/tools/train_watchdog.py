# -*- coding: utf-8 -*-
"""GRPO 训练本地看门狗：不占任何模型额度，纯本机自愈（替代 LLM 巡逻）。
每 5 分钟查一次训练进程；进程死了且日志末步 < 目标就自动 train_resume 接回。
状态追加写 logs/watchdog_v2.log；训练完成（>=目标步）后自动退出。
用法（仓库根，分离启动）:
  Start-Process -WindowStyle Hidden -FilePath D:\\anaconda\\envs\\pytorch\\python.exe -ArgumentList '-m','scripts.tools.train_watchdog' -WorkingDirectory <仓库根>
注意：用户手动暂停训练时需先停看门狗，否则会被它拉起来。
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGD = os.path.join(ROOT, "logs")
JSONL = os.path.join(LOGD, "grpo_v2", "grpo_log_math.jsonl")
WD_LOG = os.path.join(LOGD, "watchdog_v2.log")
TARGET = 200          # 日志末步达到即视为训练完成，看门狗退出
INTERVAL = 300        # 巡检间隔（秒）
NOWIN = 0x08000000    # CREATE_NO_WINDOW

PS_ALIVE = ("Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
            "Where-Object {$_.CommandLine -match 'train_grpo'} | "
            "ForEach-Object {$_.ProcessId}")


def now():
    return time.strftime("%m-%d %H:%M:%S")


def log(msg):
    with open(WD_LOG, "a", encoding="utf-8") as fp:
        fp.write(f"[{now()}] {msg}\n")


def last_step():
    try:
        last = 0
        with open(JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = max(last, json.loads(line)["step"])
        return last
    except FileNotFoundError:
        return 0


def trainer_alive():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", PS_ALIVE],
                       capture_output=True, text=True, timeout=90, creationflags=NOWIN)
    return [x.strip() for x in r.stdout.split() if x.strip().isdigit()]


def resume():
    r = subprocess.run([sys.executable, "-m", "scripts.tools.train_resume"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300,
                       creationflags=NOWIN)
    return (r.stdout + r.stderr).strip()


def main():
    log(f"看门狗启动 pid={os.getpid()} 目标 {TARGET} 步 每 {INTERVAL}s 巡检")
    while True:
        time.sleep(INTERVAL)
        try:
            alive = trainer_alive()
        except Exception as e:
            log(f"进程查询失败（下轮重试）: {e}")
            continue
        step = last_step()
        if alive:
            continue                      # 活着不打扰，不写日志
        if step >= TARGET:
            log(f"训练已完成（末步 {step} >= {TARGET}），看门狗退出")
            return
        out = resume()
        log(f"训练进程死亡(末步 {step} < {TARGET})，已自动续训 -> {out.splitlines()[-1] if out else '无输出'}")


if __name__ == "__main__":
    main()
