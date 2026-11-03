# server.py
import subprocess, time
from pathlib import Path
import httpx

ROOT = Path(__file__).parent.parent.parent
EXE = ROOT / "llama-b11669" / "llama-server.exe"
MODEL = ROOT / "models" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
HOST = "127.0.0.1"
PORT = 8080
BASE_URL = f"http://{HOST}:{PORT}"

BASE = dict(ctx=20480, slots=16, cache_ram=0)   # main configuration
PROFILES = {
    "default": {},
    "nocache": dict(cache_ram=0),                  # no RAM prompt cache
    **{f"c{c // 1024}k": dict(ctx=c) for c in (8192, 12288, 16384, 20480)},   # pool size
}


def server_args(profile):
    p = {**BASE, **PROFILES[profile]}
    return [str(EXE),
            "-m", str(MODEL),
            "-ngl", "99",
            "-fa", "on",
            "-c", str(p["ctx"]),
            "-np", str(p["slots"]),
            "--kv-unified",
            "--cache-ram", str(p["cache_ram"]),
            "--metrics",
            "--host", HOST,
            "--port", str(PORT)]


def is_up():
    try:
        return httpx.get(BASE_URL + "/health", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


# starts a server and waits until its up
def start(profile, log_path, timeout_s=180):
    if is_up():
        raise RuntimeError(f"a server is already running at {BASE_URL}")
    proc = subprocess.Popen(server_args(profile) + ["--log-file", str(log_path)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.returncode}; see {log_path}")
        if is_up():
            return proc
        time.sleep(0.5)
    stop(proc)
    raise RuntimeError("llama-server did not become ready in time")


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    while is_up():
        time.sleep(0.5)
