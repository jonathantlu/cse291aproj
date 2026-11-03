# agents.py
# each simulated user repeatedly takes the next task and runs it,
# so the mix of task types stays steady for the whole run (closed loop; arrival_s is ignored).
import asyncio


async def simulate_task(task, sched, clock, outcomes):
    out = {"task_id": task["task_id"],
           "type": task["type"],
           "start": clock(),
           "steps": len(task["steps"]),
           "steps_done": 0,
           "tool_s": 0.0,
           "status": "ok",
           "error": ""}
    prompt = task["system"]
    try:
        for i, step in enumerate(task["steps"]):
            prompt += "\n" + step["user"]
            res = await sched.submit(task["task_id"], i, prompt, step["n_out"])
            prompt += "\n" + res["output"] + "\n" + step["tool_result"]
            out["steps_done"] = i + 1
            await asyncio.sleep(step["tool_delay_s"])
            out["tool_s"] += step["tool_delay_s"]
    except Exception as e:                     # the scheduler already recorded the request
        out["status"], out["error"] = "failed", repr(e)
    finally:
        out["end"] = clock()
        out["latency"] = out["end"] - out["start"]
        outcomes.append(out)


async def run_users(tasks, sched, clock, n_users):
    # n_users users share one iterator over the tasks; each runs one task at a time.
    remaining, outcomes = iter(tasks), []

    async def user():
        for task in remaining:
            await simulate_task(task, sched, clock, outcomes)

    await asyncio.gather(*(user() for _ in range(n_users)))
    return outcomes
