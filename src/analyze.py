# analyze.py
# per-run metrics from results/ -> runs.csv, per_type.csv, summary.csv
# Steady window: from WARMUP of the way to the last task start, up to the last task start
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent.parent / "results"
WARMUP = 0.1
GROUP = ["experiment", "workload", "profile", "policy", "users", "k"]
SPREAD = ["tasks_per_min", "lat_p95", "fail_rate"]


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else np.nan


def load_runs():
    # loads all runs in results folder, ignoring broken ones
    runs = []
    for meta_path in sorted(RESULTS.rglob("meta.json")):
        d = meta_path.parent
        try:
            meta = json.loads(meta_path.read_text())
            tasks = pd.read_csv(d / "tasks.csv")
            reqs = pd.read_json(d / "requests.jsonl", lines=True)
            tel = pd.read_csv(d / "telemetry.csv") if (d / "telemetry.csv").exists() else pd.DataFrame()
        except Exception as e:
            print(f"skipping {d}: {e}")
            continue
        replica = Path(meta["workload"]).stem
        key = dict(experiment=d.parent.name, workload=re.sub(r"_\d+$", "", replica), replica=replica,
                   profile=meta["profile"], policy=meta["policy"], k=meta["k"], users=meta["users"],
                   run=str(d))
        runs.append(dict(key=key, meta=meta, tasks=tasks, reqs=reqs, tel=tel))
    return runs


def window(tasks):
    w1 = tasks.start.max()
    return WARMUP * w1, w1


def ok_requests(reqs):
    return reqs[reqs["error"].isna()] if "error" in reqs.columns else reqs


def peak_active_kv(reqs):
    # most KV cache used at once by requests
    r = reqs.dropna(subset=["prompt_tokens"]) if "prompt_tokens" in reqs.columns else reqs.iloc[0:0]
    if r.empty:
        return np.nan
    s, e = r.t_start.to_numpy(), r.t_end.to_numpy()
    p, d = r.prompt_tokens.to_numpy(), r.decode_n.to_numpy()
    held = []
    for t in e:                                      # the peak falls at some request's end
        on = (s <= t) & (e >= t)
        frac = np.clip((t - s[on]) / np.maximum(e[on] - s[on], 1e-9), 0, 1)
        held.append((p[on] + d[on] * frac).sum())
    return max(held)


def run_metrics(run):
    tasks, meta, tel = run["tasks"], run["meta"], run["tel"]
    reqs = run["reqs"].assign(queue_s=run["reqs"].t_start - run["reqs"].t_enq)
    good = ok_requests(reqs)
    w0, w1 = window(tasks)
    span = max(w1 - w0, 1e-9)
    ok = tasks[tasks.start.between(w0, w1) & (tasks.status == "ok")]
    good_win = good[good.t_start.between(w0, w1)]

    # per-task latency breakdown: queue, prefill, decode, tool, rest
    per_task = good.groupby("task_id").agg(queue=("queue_s", "sum"), prefill=("prefill_ms", "sum"),
                                           decode=("decode_ms", "sum"))
    b = ok.set_index("task_id").join(per_task)
    prompt_total = good["prompt_tokens"].sum() if "prompt_tokens" in good else 0

    m = {
        "wall_s": meta.get("wall_s"), "total_s": meta.get("total_s"),
        "tasks": len(tasks), "failed": int((tasks.status != "ok").sum()),
        "fail_rate": float((tasks.status != "ok").mean()),
        "tasks_per_min": ((tasks.status == "ok") & tasks.end.between(w0, w1)).sum() / span * 60,
        "decode_tok_s": good[good.t_end.between(w0, w1)].decode_n.sum() / span,
        "lat_p50": pct(ok.latency, 50), "lat_p95": pct(ok.latency, 95), "lat_p99": pct(ok.latency, 99),
        "queue_mean": good_win.queue_s.mean(), "queue_p95": pct(good_win.queue_s, 95),
        "t_queue": b.queue.mean(), "t_prefill": b.prefill.mean() / 1000,
        "t_decode": b.decode.mean() / 1000, "t_tool": b.tool_s.mean(),
        "peak_kv_active": peak_active_kv(good),
        "cache_hit": good["cached_tokens"].sum() / prompt_total if prompt_total else np.nan,
        "batch_per_decode": np.nan, "peak_kv_slots": np.nan,
    }
    m["t_other"] = b.latency.mean() - m["t_queue"] - m["t_prefill"] - m["t_decode"] - m["t_tool"]
    if not tel.empty:
        m["batch_per_decode"] = tel.iloc[-1].get("n_busy_slots_per_decode", np.nan)
        if "kv_tokens_slots" in tel:
            m["peak_kv_slots"] = tel["kv_tokens_slots"].max()
    return m


def reference_metrics(run, ref):
    # compares it with the reference run
    if ref is None:
        return pd.Series(dtype=float), np.nan
    lat = lambda t: t[t.status == "ok"].set_index("task_id").latency
    slow = (lat(run["tasks"]) / lat(ref["tasks"])).dropna()
    if "output" not in run["reqs"] or "output" not in ref["reqs"]:
        return slow, np.nan
    out = lambda r: r.dropna(subset=["output"]).set_index(["task_id", "step"]).output
    a, r = out(run["reqs"]), out(ref["reqs"])
    both = a.index.intersection(r.index)
    return slow, (a[both] == r[both]).mean() if len(both) else np.nan


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit("no finished runs under results/")
    # reference = the same replica and profile run with 1 user (each task alone)
    refs = {(r["key"]["replica"], r["key"]["profile"]): r for r in runs if r["key"]["users"] == 1}

    rows, types = [], []
    for run in runs:
        key = run["key"]
        slow, match = reference_metrics(run, refs.get((key["replica"], key["profile"])))
        rows.append(key | run_metrics(run) | {"slowdown_median": slow.median(), "output_match": match})
        for ttype, g in run["tasks"].groupby("type"):
            okg = g[g.status == "ok"]
            types.append({k: key[k] for k in GROUP + ["replica"]} | {
                "type": ttype, "tasks": len(g), "failed": int((g.status != "ok").sum()),
                "lat_p50": pct(okg.latency, 50), "lat_p95": pct(okg.latency, 95),
                "slowdown_median": slow.reindex(okg.task_id).median() if len(slow) else np.nan})

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "runs.csv", index=False)
    pd.DataFrame(types).to_csv(RESULTS / "per_type.csv", index=False)

    grouped = table.groupby(GROUP)
    spread = grouped[SPREAD].agg(["min", "max"])
    spread.columns = [f"{c}_{s}" for c, s in spread.columns]
    summary = grouped.median(numeric_only=True).join(spread)
    summary.insert(0, "replicas", grouped.size())
    summary.to_csv(RESULTS / "summary.csv")
    print(f"{len(table)} runs -> {RESULTS / 'summary.csv'}")


if __name__ == "__main__":
    main()
