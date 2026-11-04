# CSE291A Project: Memory Aware Agent Edge Serving
Team 12: Jonathan Lu

## Setup

- **Hardware:** Windows 10, RTX 3060 Ti (8 GB), i7-12700K
- **Runtime:** llama.cpp b11669, CUDA 13.4
- **Model:** Llama-3.1-8B-Instruct, Q4_K_M, f16 KV cache
- **Server configuration**: `-c 20480 -np 16 --kv-unified -ngl 99 -fa on --metrics --cache-ram 0`

## Layout

```
CSE291a/
  data/                corpus and workloads
    corpus/            corpus data
    workloads/         workloads
  llama-b11669/        llama.cpp binaries
  models/              GGUF model
  results/             one folder per run and analysis output
  src/                 uv project
    inference/
      agents.py        user and task simulation
      client.py        one /completion request
      scheduler.py     scheduler
      server.py        start/stop llama-server
      telemetry.py     background polling of /metrics and /slots
    analyze.py         tables and plots from all results
    corpus.py          generate the corpus
    experiments.py     experiment definitions
    generate.py        define and generate workloads
    run.py             run a single workload
    sweep.py           run experiments
```

## Usage

From `src/`:

```powershell
uv sync                                     # install dependencies
uv run generate.py                          # generate workloads
uv run sweep.py REF E1 E2 E3 E5 --dry-run   # list planned runs
uv run sweep.py REF E1 E2 E3 E5             # run experiments
uv run analyze.py                           # results\*.csv and results\plots\
```

## Workloads

Six task types (sizes in tokens):

| Type | Steps | Input | Output |
|---|---|---|---|
| `short_short` | 1 | 400–600 | 60–90 |
| `short_long` | 1 | 400–600 | 300-500 |
| `long_short` | 1 | 4,000–6,000 | 60–90 |
| `long_long` | 1 | 4,000–6,000 | 300-500 |
| `agent_short` | 3–5 | 200–400 | 100-160 |
| `agent_long` | 10–12 | 200-400 | 100-160 |

Agents sleep for a tool delay after each step, then append a tool result to their context.
All agents share a 1,500 token system prompt, and all 1 step prompts share a 100 token system prompt.

Each workload file is a shuffled list of tasks, generated in three replicas
(`<name>_670.json` … `_672.json`). During a run, 16 simulated users each repeatedly take the
next task and run it to completion. The fixed concurrency FIFO scheduler admits at most K
requests to the server at once.

## Experiments

| | Question | Varies |
|---|---|---|
| REF | Isolated task latency and reference outputs | Control |
| E1 | Concurrency | `K = 1, 2, 3, 4, 8` |
| E2 | Context growth | Agent step count |
| E3 | Idle periods between agent steps | Tool delay |
| E5 | Retained contexts and reusable prefixes | shared vs. unique system prompt |

## Output

Each run folder contains `requests.jsonl` (every request: queue/start/end times, server
timings, prompt and cached token counts, model output), `tasks.csv`, `telemetry.csv`
(`/metrics` and per-slot KV usage), `server.log`, and `meta.json` (settings and summary).

`analyze.py` computes (per run, over steady state): completed and failed tasks,
throughput, p50/p95/p99 task latency, queueing time, a latency breakdown (queue, prefill,
decode, tool), peak KV usage, average batch size, cache hit rate, slowdown versus REF, and
output agreement with REF. It writes `runs.csv`, `summary.csv`, `per_type.csv`, and one 
plot per experiment.
