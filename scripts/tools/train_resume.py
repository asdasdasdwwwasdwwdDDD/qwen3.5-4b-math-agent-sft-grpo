# -*- coding: utf-8 -*-
"""GRPO v2 续训自愈：训练进程死后从断点无缝续跑，TB 曲线保持同一条不重叠。
用法（仓库根）: python -m scripts.tools.train_resume [--dry]

原理：
- 权重回退到最后一个 checkpoint-N（N = 存档步号，<= 最后日志步 E）；
- 步号从 E+1 起（--init-step E）：N..E 之间未存档的权重更新作废，但步号直接
  跳到死前的位置，保证新事件文件的步号与旧段不重叠 -> TensorBoard 拼成一条线；
- 调度器快进次数 = jsonl 里 step<=E 且 grad_norm>0 的行数（代码里每个有梯度步
  恰好 sched.step() 一次），lr 与死前曲线逐位续接；
- 不动 outputs/grpo-v2/runs 下任何事件文件（曲线连续性红线）。
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs", "grpo-v2")
LOGD = os.path.join(ROOT, "logs")
JSONL = os.path.join(LOGD, "grpo_v2", "grpo_log_math.jsonl")
PY = r"D:\anaconda\envs\pytorch\python.exe"
NOWIN = 0x08000000

PS_ALIVE = ("Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
            "Where-Object {$_.CommandLine -match 'train_grpo'} | "
            "ForEach-Object {$_.ProcessId}")

BASE_FLAGS = ["--model", "outputs/merged-tora", "--out", "outputs/grpo-v2",
              "--data", "data/grpo_v2/pool_train_v2.jsonl",
              # 采样器难度表用训练过程自己持续更新的那份（probe 初版会让续训段 93% 重抽熟题，
              # 2026-11-23 修正）；cooldown/max-draws 等内存态无法恢复，重抽不会完全为零
              "--difficulty", "outputs/grpo-v2/difficulty_updated.json",
              "--eval-data", "data/grpo_v2/dev_v2.jsonl",
              "--max-steps", "200", "--num-generations", "8", "--gen-batch", "8",
              "--prompts-per-step", "2", "--lr", "1e-5", "--beta", "0.05",
              "--explore-ratio", "0.5", "--cooldown-draws", "40", "--max-task-draws", "3",
              "--reroll-dead", "1", "--eval-every", "20", "--eval-n", "40",
              # 保险丝已关（用户要求 2026-11-23）：0.05 在 40 题尺子（σ≈8pp）上三次噪声误杀
              # （60/100 步两次只差 3 题）。best 存档机制仍在，最终以 best + 终评为准
              "--stop-drop", "0.0", "--save-steps", "20", "--log-dir", "logs/grpo_v2",
              "--epochs", "1"]


def trainer_alive():
    r = subprocess.run(["powershell", "-NoProfile", "-Command", PS_ALIVE],
                       capture_output=True, text=True, timeout=90, creationflags=NOWIN)
    return [x.strip() for x in r.stdout.split() if x.strip().isdigit()]


def scan_state():
    """返回 (最后日志步 E, 调度器快进次数, 继承 best success/reward, checkpoint 步号)。"""
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]
    # 按 "step 回退到 1" 切出谱系：最后一次从零重开之后的行才贡献参数更新，
    # 更早被杀段（如 v2 首次试跑的 step 1-10）不能计入调度器快进
    start = 0
    for i, r in enumerate(rows):
        if r["step"] == 1 and i > 0:
            start = i
    lineage = rows[start:]
    last = max(r["step"] for r in lineage)
    ff = sum(1 for r in lineage if r["step"] <= last and r["grad_norm"] > 0)
    evals = [(r["eval_success"], r["eval_reward"]) for r in lineage if "eval_success" in r]
    best_s, best_r = max(evals) if evals else (-1.0, -1.0)
    ckpts = []
    for name in os.listdir(OUT):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if m and os.path.exists(os.path.join(OUT, name, "adapter_config.json")):
            ckpts.append(int(m.group(1)))
    ckpt = max((c for c in ckpts if c <= last), default=None)
    return last, ff, best_s, best_r, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只打印将采用的参数，不启动")
    a = ap.parse_args()

    alive = trainer_alive()
    if alive:
        print(f"训练仍在运行 (pid {alive})，拒绝重复启动", flush=True)
        sys.exit(1)
    last, ff, best_s, best_r, ckpt = scan_state()
    if ckpt is None:
        print("无可用 checkpoint，无法续训", flush=True)
        sys.exit(1)
    print(f"最后日志步 E={last}  回退 checkpoint-{ckpt}  调度器快进 {ff} 次 "
          f"继承 best success={best_s:.3f} reward={best_r:.3f}", flush=True)
    if a.dry:
        return

    nums = [int(m.group(1)) for f in os.listdir(LOGD)
            if (m := re.fullmatch(r"grpo_v2_run(\d+)\.log", f))]
    log_path = os.path.join(LOGD, f"grpo_v2_run{max(nums, default=1) + 1}.log")
    cmd = [PY, "-m", "scripts.training.train_grpo", "--resume",
           f"outputs/grpo-v2/checkpoint-{ckpt}",
           "--best-init-success", str(round(best_s, 4)),
           "--best-init-reward", str(round(best_r, 4)),
           "--init-step", str(last), "--init-sched-steps", str(ff)] + BASE_FLAGS
    fp = open(log_path, "w", encoding="utf-8")
    subprocess.Popen(cmd, cwd=ROOT, stdout=fp, stderr=subprocess.STDOUT,
                     creationflags=0x00000008 | NOWIN)  # DETACHED_PROCESS | CREATE_NO_WINDOW
    print(f"已续训启动 -> {log_path}", flush=True)


if __name__ == "__main__":
    main()
