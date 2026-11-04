# sweep.py
# finished runs are skipped, so an interrupted sweep resumes where it stopped.
# python sweep.py E1 E2 E3 E4 [--reps 1] [--dry-run]
import argparse, time, traceback
from pathlib import Path
from experiments import EXPERIMENTS, run_dir
from run import run_one

ap = argparse.ArgumentParser()
ap.add_argument("experiments", nargs="+", choices=EXPERIMENTS)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--policy", default="fifo")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

todo = [(e, s) for e in a.experiments for s in EXPERIMENTS[e](a.reps)
        if not (Path(run_dir(e, s, a.policy)) / "meta.json").exists()]
print(f"{len(todo)} runs to do")
if a.dry_run:
    for e, s in todo:
        print(" ", run_dir(e, s, a.policy))
    raise SystemExit

started = time.time()
for i, (e, s) in enumerate(todo, 1):
    out = Path(run_dir(e, s, a.policy))
    print(f"[{i}/{len(todo)}] {out}")
    try:
        run_one(s["workload"], s["k"], s["profile"], s["users"], a.policy, out)
    except Exception:
        out.mkdir(parents=True, exist_ok=True)
        (out / "error.txt").write_text(traceback.format_exc())
        print("  failed; see", out / "error.txt")
    per_run = (time.time() - started) / i
    print(f"  elapsed {(time.time() - started) / 60:.0f} min, about {per_run * (len(todo) - i) / 60:.0f} min left")
