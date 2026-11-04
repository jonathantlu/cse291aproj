# analyze.py
# produces results/runs.csv, summary.csv, per_type.csv, plots/<experiment>.png
# takes steady window from 10% of the way to the last task start, up to the last task start
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent.parent / "results"


def load(run_dir):
    meta = json.loads((run_dir / "meta.json").read_text())
    tasks = pd.read_csv(run_dir / "tasks.csv")
    reqs = pd.read_json(run_dir / "requests.jsonl", lines=True)
    tel = pd.read_csv(run_dir / "telemetry.csv") if (run_dir / "telemetry.csv").exists() else pd.DataFrame()
    return meta, tasks, reqs, tel


def peak_active_kv(reqs):
    # Most KV held at once by running requests, prompt + tokens generated so far
    r = reqs.dropna(subset=["prompt_tokens"])
    if r.empty:
        return np.nan
    s, e = r.t_start.to_numpy(), r.t_end.to_numpy()
    p, d = r.prompt_tokens.to_numpy(), r.decode_n.to_numpy()
    held = []
    for t in e:  # the peak falls at some request's end
        on = (s <= t) & (e >= t)
        frac = np.clip((t - s[on]) / np.maximum(e[on] - s[on], 1e-9), 0, 1)
        held.append((p[on] + d[on] * frac).sum())
    return max(held)


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else np.nan


def run_metrics(meta, tasks, reqs, tel):
    w1 = tasks.start.max()
    w0 = 0.1 * w1
    win = tasks[(tasks.start >= w0) & (tasks.start <= w1)]
    ok = win[win.status == "ok"]
    span = max(w1 - w0, 1e-9)
    done_in_win = ((tasks.status == "ok") & tasks.end.between(w0, w1)).sum()
    reqs = reqs.assign(queue_s=reqs.t_start - reqs.t_enq)
    good = reqs[reqs.get("error").isna()] if "error" in reqs else reqs

    # per task breakdown: queue, prefill, decode, tool, rest
    per_task = good.groupby("task_id").agg(queue=("queue_s", "sum"), prefill=("prefill_ms", "sum"),
                                           decode=("decode_ms", "sum"))
    b = ok.set_index("task_id").join(per_task)
    m = {
        "tasks": len(tasks), "failed": int((tasks.status != "ok").sum()),
        "fail_rate": float((tasks.status != "ok").mean()),
        "tasks_per_min": done_in_win / span * 60,
        "decode_tok_s": good[good.t_end.between(w0, w1)].decode_n.sum() / span,
        "lat_p50": pct(ok.latency, 50), "lat_p95": pct(ok.latency, 95), "lat_p99": pct(ok.latency, 99),
        "queue_mean": good.queue_s.mean(), "queue_p95": pct(good.queue_s, 95),
        "t_queue": b.queue.mean(), "t_prefill": b.prefill.mean() / 1000,
        "t_decode": b.decode.mean() / 1000, "t_tool": b.tool_s.mean(),
        "peak_kv_active": peak_active_kv(good),
        "cache_hit": good.cached_tokens.sum() / good.prompt_tokens.sum()
                     if good.get("prompt_tokens") is not None and good.prompt_tokens.sum() else np.nan,
    }
    m["t_other"] = b.latency.mean() - m["t_queue"] - m["t_prefill"] - m["t_decode"] - m["t_tool"]
    if not tel.empty:
        last = tel.iloc[-1]
        m["batch_per_decode"] = last.get("n_busy_slots_per_decode", np.nan)
        m["peak_kv_slots"] = tel["kv_tokens_slots"].max() if "kv_tokens_slots" in tel else np.nan
    return m


def natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def main():
    runs, types, loaded = [], [], {}
    for meta_path in sorted(RESULTS.rglob("meta.json")):
        d = meta_path.parent
        meta, tasks, reqs, tel = load(d)
        base = re.sub(r"_\d+$", "", Path(meta["workload"]).stem)
        key = dict(experiment=d.parent.name, workload=base, replica=Path(meta["workload"]).stem,
                   profile=meta["profile"], policy=meta["policy"], k=meta["k"], users=meta["users"],
                   run=str(d))
        loaded[str(d)] = (key, tasks, reqs)
        runs.append(key | run_metrics(meta, tasks, reqs, tel))
    if not runs:
        raise SystemExit("no finished runs under results/")
    runs = pd.DataFrame(runs)

    # reference, the same replica and profile run with 1 user
    for i, row in runs.iterrows():
        ref = runs[(runs.replica == row.replica) & (runs.profile == row.profile) & (runs.users == 1)]
        key, tasks, reqs = loaded[row.run]
        slow = pd.Series(dtype=float)
        if len(ref):
            _, rtasks, rreqs = loaded[ref.iloc[0].run]
            r_lat = rtasks[rtasks.status == "ok"].set_index("task_id").latency
            t_ok = tasks[tasks.status == "ok"].set_index("task_id")
            slow = (t_ok.latency / r_lat).dropna()
            a = reqs.dropna(subset=["output"]).set_index(["task_id", "step"]).output
            r = rreqs.dropna(subset=["output"]).set_index(["task_id", "step"]).output
            both = a.index.intersection(r.index)
            runs.loc[i, "output_match"] = (a[both] == r[both]).mean() if len(both) else np.nan
            runs.loc[i, "slowdown_median"] = slow.median()
        for ttype, g in tasks.groupby("type"):
            okg = g[g.status == "ok"]
            types.append(dict(row[["experiment", "workload", "replica", "profile", "users", "k"]]) | {
                "type": ttype, "tasks": len(g), "failed": int((g.status != "ok").sum()),
                "lat_p50": pct(okg.latency, 50), "lat_p95": pct(okg.latency, 95),
                "slowdown_median": slow.reindex(okg.task_id).median() if len(slow) else np.nan})

    runs.to_csv(RESULTS / "runs.csv", index=False)
    pd.DataFrame(types).to_csv(RESULTS / "per_type.csv", index=False)
    group = ["experiment", "workload", "profile", "policy", "users", "k"]
    summary = runs.groupby(group).median(numeric_only=True)
    summary["replicas"] = runs.groupby(group).size()
    summary.to_csv(RESULTS / "summary.csv")

    # plots
    (RESULTS / "plots").mkdir(exist_ok=True)
    metrics = [("tasks_per_min", "completed tasks / min"), ("lat_p95", "p95 task latency (s)"),
               ("fail_rate", "failed fraction"), ("peak_kv_active", "peak active KV (tokens)")]
    # experiments whose x-axis is a workload parameter instead of K
    XAXIS = {"E2": ("steps", "agent steps (context growth)"),
            "E3": ("delay", "tool delay (s)")}

    for exp, g in summary.reset_index().query("users > 1").groupby("experiment"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes = axes.flatten()
        if exp in XAXIS:
            prefix, xlabel = XAXIS[exp]
            g = g.assign(x=g.workload.str.removeprefix(prefix).astype(float))
            series = sorted(g.groupby(["k", "profile"]))
            name = lambda key: f"K={key[0]} ({key[1]})"
        else:
            xlabel = "concurrency K"
            g = g.assign(x=g.k)
            series = sorted(g.groupby(["workload", "profile"]), key=lambda kv: (natural(kv[0][0]), kv[0][1]))
            name = lambda key: f"{key[0]} ({key[1]})"

        for key, line in series:
            line = line.sort_values("x")
            for ax, (col, label) in zip(axes, metrics):
                ax.plot(line.x, line[col], "o-", label=name(key))
                ax.set(xlabel=xlabel, ylabel=label)
        for ax, (col, _) in zip(axes, metrics):
            top = g[col].max()
            ax.set_ylim(0, top * 1.1 if top > 0 else 1)
        axes[0].legend(fontsize=8)
        fig.suptitle(exp)
        fig.tight_layout()
        fig.savefig(RESULTS / "plots" / f"{exp}.png", dpi=130)
        plt.close(fig)
    # print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
