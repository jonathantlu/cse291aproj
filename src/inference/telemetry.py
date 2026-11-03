# telemetry.py
# sample the server's /metrics and /slots in the background.
import asyncio
from .server import BASE_URL


def slot_tokens(slot):
    # Tokens a slot holds in the KV cache
    n = slot.get("n_prompt_tokens")
    return n if isinstance(n, int) else None


async def poll(client, clock, rows, stop, every=0.5):
    while not stop.is_set():
        row = {"t": clock()}
        try:
            r = await client.get(BASE_URL + "/metrics", timeout=2)
            for line in r.text.splitlines():
                if line.startswith("llamacpp:"):
                    name, value = line.split()[:2]
                    row[name.removeprefix("llamacpp:")] = float(value)
        except Exception:
            pass
        try:
            r = await client.get(BASE_URL + "/slots", timeout=2)
            if r.status_code == 200:
                slots = r.json()
                row["slots_processing"] = sum(bool(s.get("is_processing")) for s in slots)
                held = [slot_tokens(s) for s in slots]
                if all(h is not None for h in held):
                    row["kv_tokens_slots"] = sum(held)
        except Exception:
            pass
        rows.append(row)
        try:
            await asyncio.wait_for(stop.wait(), every)
        except asyncio.TimeoutError:
            pass
