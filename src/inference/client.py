# client.py
import time
from .server import BASE_URL

async def send_request(client, prompt, n_out, cache_prompt=True):
    body = {"prompt": prompt,
            "n_predict": n_out,
            "ignore_eos": True,
            "temperature": 0,
            "cache_prompt": cache_prompt}
    start = time.perf_counter()
    r = await client.post(BASE_URL + "/completion", json=body)
    if r.is_error:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")   # keep the server's reason
    res = r.json()
    t = res["timings"]
    return {
        "output": res["content"],
        "wall_ms": (time.perf_counter() - start) * 1000,
        "slot": res.get("id_slot"),
        "prompt_tokens": res.get("tokens_evaluated"),          # whole prompt
        "cached_tokens": res.get("tokens_cached", t.get("cache_n")),  # reused from cache
        "prefill_n": t["prompt_n"],
        "prefill_ms": t["prompt_ms"],     # actually processed
        "decode_n": t["predicted_n"],
        "decode_ms": t["predicted_ms"],
    }
