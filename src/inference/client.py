# client.py
import time
from .server import BASE_URL

async def send_request(client, prompt, n_out, slot=None, cache_prompt=True):
    body = {"prompt": prompt,
            "n_predict": n_out,
            "ignore_eos": True,
            "temperature": 0,
            "cache_prompt": cache_prompt}
    if slot is not None:
        body["id_slot"] = slot
    start = time.perf_counter()
    r = await client.post(BASE_URL + "/completion", json=body)
    if r.is_error:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")  # keep the server's reason
    res = r.json()
    t = res["timings"]
    return {
        "output": res["content"],
        "wall_ms": (time.perf_counter() - start) * 1000,
        "slot": res.get("id_slot"),
        "prompt_tokens": res.get("tokens_evaluated"),  # whole prompt
        "cached_tokens": res.get("tokens_evaluated", 0) - t["prompt_n"],  # prompt tokens reused
        "prefill_n": t["prompt_n"],
        "prefill_ms": t["prompt_ms"],  # actually processed
        "decode_n": t["predicted_n"],
        "decode_ms": t["predicted_ms"],
    }


async def tokenize(client, text):
    r = await client.post(BASE_URL + "/tokenize", json={"content": text})
    r.raise_for_status()
    return len(r.json()["tokens"])


async def erase_slot(client, slot):
    r = await client.post(BASE_URL + f"/slots/{slot}?action=erase")
    r.raise_for_status()