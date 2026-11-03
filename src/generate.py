# generate.py
import json, random
from pathlib import Path

SEED = 67
REPLICAS = 3
R = random.Random()

TYPES = {
    "short_short": dict(steps=1,        user=(400, 600),   n_out=(60, 90),   delay=0,          result=0,          system=100),
    "short_long":  dict(steps=1,        user=(400, 600),   n_out=(300, 500), delay=0,          result=0,          system=100),
    "long_short":  dict(steps=1,        user=(4000, 6000), n_out=(60, 90),   delay=0,          result=0,          system=100),
    "long_long":   dict(steps=1,        user=(4000, 6000), n_out=(300, 500), delay=0,          result=0,          system=100),
    "agent_short": dict(steps=(3, 5),   user=(200, 400),   n_out=(100, 160), final_out=(200, 400), delay=(0.5, 4.0), result=(200, 600), system=1500),
    "agent_long":  dict(steps=(10, 12), user=(200, 400),   n_out=(100, 160), final_out=(200, 400), delay=(0.5, 4.0), result=(200, 600), system=1500),
}

# Relative weights
MIXES = {
    "light":       dict(short_short=4, short_long=1, long_short=1, agent_short=2),
    "mixed":       dict(short_short=4, short_long=2, long_short=2, long_long=1, agent_short=4, agent_long=3),
    "agent_heavy": dict(short_short=1, short_long=1, long_short=1, agent_short=7, agent_long=6),
    "agent_only":  dict(agent_short=1, agent_long=1),
    "agent_long":  dict(agent_long=1),
}

# (mix, number of tasks, overrides for agent types)
WORKLOADS = {
    "light": ("light", 60, {}),                                              # E1
    "mixed": ("mixed", 60, {}),
    "agent_heavy": ("agent_heavy", 60, {}),
    **{f"steps{n}": ("agent_long", 48, dict(steps=n, delay=2.0)) for n in (2, 4, 8, 12, 16)},
    **{f"delay{d}": ("agent_only", 48, dict(delay=float(d))) for d in (0, 1, 4, 16)},
    "unique_system": ("agent_heavy", 60, dict(shared=False)),
}


ROOT = Path(__file__).parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
WORKLOADS_DIR = ROOT / "data" / "workloads"

CORPORA = [(CORPUS_DIR / f"corpus_{c}.txt").read_text(encoding="utf-8") for c in ("code", "prose")]
CHARS_PER_TOKEN = 4
CHUNK = (200, 500)


def draw(v):
    if not isinstance(v, tuple):
        return v
    return R.randint(*v) if isinstance(v[0], int) else round(R.uniform(*v), 2)


# return about n_tokens of text, chunked from code and python docs
def text(n_tokens):
    left = int(draw(n_tokens) * CHARS_PER_TOKEN)
    chunks = []
    while left > 0:
        n, corpus = min(left, int(draw(CHUNK) * CHARS_PER_TOKEN)), R.choice(CORPORA)
        start = R.randrange(len(corpus) - n)
        chunks.append(corpus[start:start + n])
        left -= n
    return "\n".join(chunks)


def make_task(task_id, ttype, spec, shared):
    steps = [{"user": text(spec["user"]) if i == 0 else "",
              "n_out": draw(spec["n_out"]),
              "tool_delay_s": draw(spec["delay"]),
              "tool_result": text(spec["result"])}
             for i in range(draw(spec["steps"]))]
    steps[-1].update(tool_delay_s=0, tool_result="")          # no tool after the final answer
    if "final_out" in spec:
        steps[-1]["n_out"] = draw(spec["final_out"])          # the agent's final answer is longer
    system = shared[spec["system"]] if spec.get("shared", True) else text(spec["system"])
    return {"task_id": task_id, "type": ttype, "arrival_s": 0, "system": system, "steps": steps}


if __name__ == "__main__":
    WORKLOADS_DIR.mkdir(parents=True, exist_ok=True)
    for name, (mix, n, overrides) in WORKLOADS.items():
        weights = MIXES[mix]
        types = [t for t, w in weights.items() for _ in range(round(w * n / sum(weights.values())))]
        for rep in range(REPLICAS):
            seed = SEED * 10 + rep
            R.seed(seed)                               # each file reproducible on its own
            shared = {size: text(size) for size in (100, 1500)}
            tasks = []
            for i, t in enumerate(types):
                spec = {**TYPES[t], **(overrides if t.startswith("agent") else {})}
                tasks.append(make_task(f"{t}_{i:03d}", t, spec, shared))
            R.shuffle(tasks)
            out = WORKLOADS_DIR / f"{name}_{seed}.json"
            out.write_text(json.dumps(tasks))
            print(f"Wrote {out}: {len(tasks)} tasks")
