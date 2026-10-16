from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workloads" / "train_ddp.py"
SP, END = 150, 300
EVAL_EVERY, EVAL_SIZE = 10, 4096
CHECK = [150, 200, 250, 300]


def torchrun(out: Path, world: int, mode: str, max_steps: int, resume: str = "", sp_at: int = 0):
    env = {**os.environ, "GRP_RUN_DIR": str(out), "GRP_CKPT_DIR": str(out / "ckpt"),
           "GRP_RESUME": resume, "GRP_SAFEPOINT_AT_STEP": str(sp_at), "OMP_NUM_THREADS": "1"}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}",
           str(SCRIPT), "--max-steps", str(max_steps), "--batch-mode", mode,
           "--loss-log", str(out / "loss.csv"), "--eval-log", str(out / "eval.csv"),
           "--eval-every", str(EVAL_EVERY), "--eval-size", str(EVAL_SIZE),
           "--save-final", str(out / "final.pt")]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    (out / f"torchrun-w{world}.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"torchrun failed ({out.name}, world {world}):\n{r.stderr[-2000:]}")
    return time.perf_counter() - t0


def read_csv(path: Path) -> dict[int, float]:
    with open(path) as f:
        return {int(s): float(l) for s, _w, l in csv.reader(f)}


def weight_rel_diff(p: Path, ref: Path) -> float:
    import torch
    w, wa = torch.load(p), torch.load(ref)
    num = sum(((w[k] - wa[k]).double() ** 2).sum() for k in wa) ** 0.5
    den = sum((wa[k].double() ** 2).sum() for k in wa) ** 0.5
    return float(num / den)


def main():
    base = ROOT / "runs" / "experiments" / f"ddp-resize-{datetime.now():%Y%m%d-%H%M%S}"
    scen = {"A": (None, "global"), "A2": (None, "global"), "B": (1, "global"), "C": (4, "global"),
            "D": (1, "per-rank"), "E": (4, "per-rank")}
    train, evals = {}, {}
    for name, (w2, mode) in scen.items():
        out = base / name
        out.mkdir(parents=True)
        if w2 is None:
            torchrun(out, 2, mode, END)
        else:
            torchrun(out, 2, mode, END, sp_at=SP)
            ck = out / "ckpt" / f"ckpt-step{SP}.pt"
            torchrun(out, w2, mode, END, resume=str(ck))
        train[name], evals[name] = read_csv(out / "loss.csv"), read_csv(out / "eval.csv")
        print(f"{name} done", flush=True)

    desc = {"A": "baseline, world 2 kept", "A2": "baseline re-run (noise check)",
            "B": "2->1, global batch fixed", "C": "2->4, global batch fixed",
            "D": "2->1, per-rank batch fixed (control)", "E": "2->4, per-rank batch fixed (control)"}
    ea, ta = evals["A"], train["A"]
    after = [s for s in ea if s > SP]
    lines = ["| Scenario | " + " | ".join(f"step {s}" for s in CHECK)
             + " | Max diff vs baseline | Relative diff at step 300 | Weight relative diff (step 300) |",
             "|---|" + "---:|" * (len(CHECK) + 3)]
    ref_lines = ["| Scenario | Max training-batch loss diff vs baseline (151~300) |", "|---|---:|"]
    for n in scen:
        e, t = evals[n], train[n]
        diff = max(abs(e[s] - ea[s]) for s in after)
        rel = abs(e[END] - ea[END]) / ea[END] * 100
        wd = weight_rel_diff(base / n / "final.pt", base / "A" / "final.pt")
        lines.append(f"| {n} {desc[n]} | " + " | ".join(f"{e[s]:.6f}" for s in CHECK)
                     + f" | {diff:.2e} | {rel:.4f}% | {wd:.2e} |")
        tdiff = max(abs(t[s] - ta[s]) for s in range(SP + 1, END + 1))
        ref_lines.append(f"| {n} {desc[n]} | {tdiff:.2e} |")
    table, ref_table = "\n".join(lines), "\n".join(ref_lines)
    import torch
    report = f"""# Training result before/after DDP reconfiguration ({datetime.now():%Y-%m-%d %H:%M})

- Environment: WSL2 Ubuntu 24.04, 1x {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}, torchrun DDP (gloo, one process per rank sharing one GPU)
- Model: MLP 512-2048-2048-1, AdamW lr 1e-4, fixed synthetic training data per step, seed 0
- Procedure: train with world 2 -> Safe Point at step {SP} (all ranks stop together, checkpoint saved) -> relaunch torchrun with new world size -> resume from checkpoint -> step {END}
- Evaluation: MSE every {EVAL_EVERY} steps on **{EVAL_SIZE} fixed held-out samples** (same for all scenarios and steps)
- Weight relative diff: ||W - W_A|| / ||W_A|| on final weights at step {END}
- Command: `python experiments/ddp_resize_compare.py`

## Eval loss (fixed held-out data)

{table}

## Reference: training-batch loss (previous method)

{ref_table}

- Interpretation: A2 (same-condition re-run) gives the run-to-run noise unrelated to reconfiguration. If B and C are at that level, reconfiguration with a fixed global batch is indistinguishable from training without reconfiguration. D and E are measured on the same eval data, so their difference reflects the training result itself (effect of changing the global batch).
- Results: `{base.relative_to(ROOT)}` (per-scenario loss.csv, eval.csv, final.pt, torchrun logs, checkpoints)
- Limitations: one shared GPU with gloo, so communication performance and physical GPU reclaim are not evaluated. Small MLP, synthetic data, 300 steps; not generalizable to other models, optimizers or data.
"""
    (base / "report.md").write_text(report, encoding="utf-8")
    print("\n" + table + "\n\n" + ref_table + f"\n\n→ {base / 'report.md'}")


if __name__ == "__main__":
    main()
