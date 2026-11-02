"""
Kernel-shape sweep for the 2D token conv (works on Windows / Linux / macOS).

  off : no conv (baseline)
  1x1 : same extra params, no spatial mixing (control)
  3x3 : square in cells -> tall in physical space
  3x5 : ~square in physical space (cells are ~2x taller than wide)
  5x5 : bigger square in cells
  1x5 : horizontal only
  3x1 : vertical only

Examples:
  python sweep_kernel.py                                   # mnist mixed, 10000 steps, seed 0
  python sweep_kernel.py --source shapes --style edge --steps 2000
  python sweep_kernel.py --seeds 0 1 2
  python sweep_kernel.py --kernels off 3x3 3x5 --extra --space_weight 0.3

Results are appended to results.csv; samples go to ckpt/*_samples.txt.
"""
import argparse
import csv
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="mnist")
ap.add_argument("--style", default="mixed")
ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--kernels", nargs="+", default=["off", "1x1", "3x3", "3x5", "5x5", "1x5", "3x1"])
ap.add_argument("--results", default="results.csv")
ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra args passed to train_dllm.py")
a = ap.parse_args()

for seed in a.seeds:
    for k in a.kernels:
        cmd = [sys.executable, "train_dllm.py", "--source", a.source, "--style", a.style,
               "--steps", str(a.steps), "--seed", str(seed), "--results", a.results]
        if k != "off":
            cmd += ["--conv", "--kernel", k]
        cmd += a.extra
        print("\n>>>", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

print("\n=== results ===")
with open(a.results, newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))
widths = [max(len(r[i]) if i < len(r) else 0 for r in rows) for i in range(len(rows[0]))]
for r in rows:
    print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
