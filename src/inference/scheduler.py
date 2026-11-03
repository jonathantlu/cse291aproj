# scheduler.py
import asyncio
from collections import deque
from .client import send_request


class Scheduler:
    def __init__(self, client, clock):
        self.client = client
        self.clock = clock
        self.records = []

    # add a request
    def push(self, req):
        raise NotImplementedError

    # if the queue is not empty and we decide we want to run a task
    def pop(self):
        raise NotImplementedError

    # for additional bookkeeping on task finish
    def finish(self, req):
        pass

    async def submit(self, task_id, step, prompt, n_out):
        req = {"task_id": task_id,
               "step": step,
               "n_out": n_out,
               "prompt_chars": len(prompt),
               "t_enq": self.clock(),
               "_go": asyncio.Event()}
        self.push(req)
        self._dispatch()
        await req["_go"].wait()
        del req["_go"]

        req["t_start"] = self.clock()
        res = {"error": "cancelled"}
        try:
            res = await send_request(self.client, prompt, n_out)
        except Exception as e:
            res = {"error": repr(e)}
            raise
        finally:
            req["t_end"] = self.clock()
            self.records.append({**req, **res})
            self.finish(req)
            self._dispatch()
        return res

    def _dispatch(self):
        while (req := self.pop()) is not None:
            req["_go"].set()


class FIFO(Scheduler):
    def __init__(self, client, clock, k):
        super().__init__(client, clock)
        self.k = k
        self.running = 0
        self.q = deque()

    def push(self, req):
        self.q.append(req)

    def pop(self):
        if self.q and self.running < self.k:
            self.running += 1
            return self.q.popleft()
        return None

    def finish(self, req):
        self.running -= 1

POLICIES = {"fifo": FIFO}