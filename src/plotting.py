# plotting.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
WIDTH = 3.45            # inches, one column of a 7 in two-column page
KV_CAPACITY = 20480
plt.rcParams.update({"font.size": 7, "lines.markersize": 3.5,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.3, "axes.axisbelow": True})


def errorbars(g, col, scale):
    # min/max across replicas as asymmetric error bars, if analyze.py wrote them
    if f"{col}_min" not in g:
        return None
    y = g[col] * scale
    return [(y - g[f"{col}_min"] * scale).clip(lower=0), (g[f"{col}_max"] * scale - y).clip(lower=0)]


def fig_e1(summary, out):
    e1 = summary[(summary.experiment == "E1") & (summary.users > 1)]
    panels = [("tasks_per_min", 1, "Throughput (tasks/min)"), ("lat_p95", 1, "p95 latency (s)"),
              ("fail_rate", 100, "Failed tasks (%)"), ("peak_kv_active", 1 / 1000, "Peak KV (k tokens)")]
    workloads = [("light", "Light"), ("mixed", "Mixed"), ("agent_heavy", "Agent-heavy")]

    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 3.2), sharex=True)
    for ax, (col, scale, title) in zip(axs.flat, panels):
        for w, name in workloads:
            g = e1[e1.workload == w].sort_values("k")
            ax.errorbar(g.k, g[col] * scale, yerr=errorbars(g, col, scale), fmt="o-", capsize=2, label=name)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    axs[1, 1].axhline(KV_CAPACITY / 1000, color="gray", linestyle="--", linewidth=1, label="Capacity")
    axs[1, 0].set_ylim(0, 100)
    axs[1, 0].legend(loc="upper left", fontsize=6)   # the empty corner of the failure panel
    for ax in axs[1]:
        ax.set_xlabel("Concurrency K")
        ax.set_xticks(sorted(e1.k.unique()))
    fig.tight_layout()
    fig.savefig(out / "fig_e1.png", dpi=300)
    plt.close(fig)


def by_param(summary, experiment, k, prefix):
    g = summary[(summary.experiment == experiment) & (summary.k == k) & (summary.users > 1)].copy()
    g["x"] = g.workload.str.removeprefix(prefix).astype(int)
    return g.sort_values("x")


def fig_mechanisms(summary, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 1.9))

    # context growth
    e2 = by_param(summary, "E2", 1, "steps")
    a.plot(e2.x, e2.t_prefill, "o-", label="Prefill")
    a.plot(e2.x, e2.t_decode, "o-", label="Decode")
    a.set_xticks(e2.x)
    a.set_xlabel("Agent steps")
    a.set_ylabel("Seconds per task")
    a.set_title("(a) Context growth")
    a.set_ylim(0, max(e2.t_decode.max(), e2.t_prefill.max()) * 1.25)   # headroom for the legend
    a.legend(loc="upper left", fontsize=6)

    # idle periods
    e3 = by_param(summary, "E3", 2, "delay")
    x, bottom = np.arange(len(e3)), np.zeros(len(e3))
    for col, name in [("t_queue", "Queue"), ("t_prefill", "Prefill"), ("t_decode", "Decode"), ("t_tool", "Tool")]:
        b.bar(x, e3[col], bottom=bottom, label=name)
        bottom += e3[col].to_numpy()
    b.set_xticks(x, [f"{d} s" for d in e3.x])
    b.set_xlabel("Tool delay")
    b.set_title("(b) Idle periods")
    b.set_ylim(0, bottom.max() * 1.55)       # headroom for the legend
    b.legend(ncol=2, fontsize=6, loc="upper left", handlelength=1, columnspacing=0.6)

    fig.tight_layout()
    fig.savefig(out / "fig_mechanisms.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=ROOT / "results" / "summary.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "plots")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(a.summary)
    fig_e1(summary, a.out)
    fig_mechanisms(summary, a.out)
    print(f"wrote {a.out / 'fig_e1.png'} and {a.out / 'fig_mechanisms.png'}")


if __name__ == "__main__":
    main()