# experiments.py
# K=1 points get one replica only. REF runs each task alone (1 user, K=1): the isolated
# latency that slowdown is measured against, and the reference outputs for the quality check.

from pathlib import Path

ROOT = Path(__file__).parent.parent
WORKLOADS = ROOT / "data" / "workloads"
RESULTS = ROOT / "results"

def runs(workloads, ks, reps, profiles=("default",), users=16):
    return [dict(workload=f"{WORKLOADS}/{w}_67{r}.json", k=k, profile=p, users=users)
            for w in workloads for p in profiles for k in ks
            for r in range(1 if k == 1 else reps)]


EXPERIMENTS = {
    # isolated references
    "REF": lambda reps: runs(["light", "mixed", "agent_heavy", "unique_system"], [1], reps, users=1),
    # concurrency
    "E1": lambda reps: runs(["light", "mixed", "agent_heavy"], [1, 2, 3, 4, 8], reps),
    # context growth
    "E2": lambda reps: runs([f"steps{n}" for n in (2, 4, 8, 12, 16)], [2], reps),
    # idle periods between agent steps
    "E3": lambda reps: runs([f"delay{d}" for d in (0, 1, 4, 16)], [2, 4], reps),
    # reusable prefixes (shared vs unique system prompt)
    "E4": lambda reps: runs(["agent_heavy", "unique_system"], [2, 4], reps),
}


def run_dir(experiment, spec, policy="fifo"):
    name = Path(spec["workload"]).stem
    return RESULTS / experiment / f"{name}_K{spec['k']}_{spec['profile']}_{policy}_U{spec['users']}"