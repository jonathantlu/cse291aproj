# arrivals.py
# open-loop copies of the E1 workloads, with Poisson arrivals
import json, random
import pandas as pd
from experiments import WORKLOADS, RESULTS

LOADS = [50, 80, 100, 120]      # % of FIFO's best throughput
N_TASKS = 40

runs = pd.read_csv(RESULTS / "runs.csv")
plan = []
for mix in ["light", "mixed", "agent_heavy"]:
    e1 = runs[(runs.experiment == "E1") & (runs.replica == f"{mix}_670") & (runs.users > 1)]
    best = e1.sort_values("tasks_per_min").iloc[-1]  # FIFO's best K and its throughput
    tasks = json.loads((WORKLOADS / f"{mix}_670.json").read_text(encoding="utf-8"))[:N_TASKS]
    for load in LOADS:
        rate = best.tasks_per_min * load / 100 / 60  # tasks per second
        r, t = random.Random(load), 0.0
        for task in tasks:
            t += r.expovariate(rate)
            task["arrival_s"] = round(t, 2)
        name = f"{mix}_load{load}_670.json"
        (WORKLOADS / name).write_text(json.dumps(tasks))
        plan.append({"workload": name, "best_k": int(best.k)})
        print(f"{name}: FIFO K={int(best.k)}, last arrival at {t / 60:.1f} min")
(WORKLOADS / "open_plan.json").write_text(json.dumps(plan, indent=2))