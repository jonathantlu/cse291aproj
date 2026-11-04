# run.py 
# one run: fresh server, one workload, one scheduler setting -> results folder.
# python run.py ..\data\workloads\mixed_670.json --k 4 [--profile default] [--users 16] [--policy fifo]
import argparse, asyncio, csv, json, platform, subprocess, time
from pathlib import Path
import httpx
from inference import server
from inference.scheduler import POLICIES
from inference.agents import run_users
from inference.telemetry import poll
from experiments import RESULTS


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


async def execute(tasks, policy, k, users):
    t0 = time.perf_counter()
    clock = lambda: time.perf_counter() - t0
    telemetry, stop = [], asyncio.Event()
    async with httpx.AsyncClient(timeout=600) as client:
        sched = POLICIES[policy](client, clock, k)
        poller = asyncio.create_task(poll(client, clock, telemetry, stop))
        outcomes = await run_users(tasks, sched, clock, users)
        wall = clock()
        stop.set()
        await poller
    return sched.records, outcomes, telemetry, wall


def run_one(workload, k, profile="default", users=16, policy="fifo", out_dir=None):
    workload = Path(workload)
    out_dir = Path(out_dir or RESULTS / "adhoc" / f"{workload.stem}_K{k}_{profile}_{policy}_U{users}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(workload.read_text(encoding="utf-8"))

    proc = server.start(profile, out_dir / "server.log")
    try:
        records, outcomes, telemetry, wall = asyncio.run(execute(tasks, policy, k, users))
    finally:
        server.stop(proc)

    with open(out_dir / "requests.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    save_csv(out_dir / "tasks.csv", outcomes)
    if telemetry:
        save_csv(out_dir / "telemetry.csv", telemetry)

    done = sum(o["status"] == "ok" for o in outcomes)
    meta = {"workload": str(workload), "k": k, "profile": profile, "users": users, "policy": policy,
            "tasks": len(tasks), "completed": done, "failed": len(outcomes) - done,
            "wall_s": round(wall, 2), "server_args": server.server_args(profile),
            "git_commit": git_commit(), "python": platform.python_version(),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))   # written last = run complete
    print(f"{out_dir}: {done}/{len(tasks)} completed in {wall:.0f} s")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("workload")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--profile", default="default", choices=server.PROFILES)
    ap.add_argument("--users", type=int, default=16)
    ap.add_argument("--policy", default="fifo", choices=POLICIES)
    ap.add_argument("--out")
    a = ap.parse_args()
    run_one(a.workload, a.k, a.profile, a.users, a.policy, a.out)
