# scheduler.py
import asyncio
from collections import deque
from .client import send_request, tokenize, erase_slot
from .server import BASE


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

    # async work before queueing, need for mem aware
    async def before_queue(self, req, prompt):
        pass

    # async work after queueing
    async def before_send(self, req):
        pass

    # async work after request finishes
    async def after_request(self, req, res):
        pass

    # for when an entire request is done
    def task_done(self, task_id):
        pass

    async def submit(self, task_id, step, prompt, n_out):
        req = {"task_id": task_id,
               "step": step,
               "n_out": n_out,
               "prompt_chars": len(prompt),
               "_go": asyncio.Event()}
        await self.before_queue(req, prompt)
        req["t_enq"] = self.clock()
        self.push(req)

        self._dispatch()
        await req["_go"].wait()
        del req["_go"]

        req["t_start"] = self.clock()
        res = {"error": "cancelled"}
        try:
            if req.get("reject"):  # if request is too big
                raise RuntimeError(f"request needs {req['need']} tokens, more than the KV pool")
            
            await self.before_send(req)
            res = await send_request(self.client, prompt, n_out, slot=req.get("slot"))
        except Exception as e:  # HERE
            res = {"error": repr(e)}
            raise
        finally:
            req["t_end"] = self.clock()
            self.records.append({**req, **res})
            try:
                await self.after_request(req, res)
            finally:
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

    async def after_request(self, req, res):
        self.running -= 1


class SlotAware(Scheduler):
    # Scheduler is aware of which slot each agent is in, and attempts to place
    # agent requests in the same slot. 
    # It is also prefix aware, and will estimate memory usage of a request 
    # based on the prefix saved in a slot.
    # Requests are admitted in FIFO order if the KV pool has room, and later requests
    # can enter before an older one if there is room for that request with a tolerance (TODO).
    # The slot is assigned based on this order:
    # 1. Its previous slot
    # 2. Free slot with same prefix
    # 3. Empty slot
    # 4. LRU used slot
    # Additional room is created by erasing free slots leftover memory, then 
    # idle agent task slots (LRU)

    def __init__(self, client, clock, k=BASE["slots"], retain=True, n_slots=BASE["slots"], pool=BASE["ctx"]):
        super().__init__(client, clock)
        self.retain = retain
        self.budget = pool - 256
        self.k = min(k, n_slots)
        self.slots = [dict(tokens=0, owner=None, state="free", key=None, last=0.0) for _ in range(n_slots)]
        self.owner = {}  # owner -> slot
        self.q = deque()
        self.running = 0

    def push(self, req):
        self.q.append(req)

    # returns (chosen slot, slots to erase)
    def _plan(self, req):
        if req["need"] > self.budget:  # request is too big
            return (None, [])

        free = [i for i, s in enumerate(self.slots) if s["state"] == "free"]
        idle = [i for i, s in enumerate(self.slots) if s["state"] == "idle"]

        slot = self.owner.get(req["task_id"])
        if slot is None:
            same = [i for i in free if self.slots[i]["key"] == req["key"]]
            empty = [i for i in free if self.slots[i]["tokens"] == 0]
            pick = same or empty or free or idle
            if not pick:
                return None
            slot = min(pick, key = lambda x: self.slots[x]["last"])
        
        used = sum(s["tokens"] for s in self.slots)
        needed = req["need"] - self.slots[slot]["tokens"]
        erase = []
        if used + needed > self.budget:
            candidates = sorted(free, key = lambda x: self.slots[x]["last"]) + sorted(idle, key = lambda x: self.slots[x]["last"])
            for i in candidates:
                if i == slot or self.slots[i]["tokens"] == 0:
                    continue
                erase.append(i)
                used -= self.slots[i]["tokens"]
                if used + needed <= self.budget:
                    break
            else:
                return None
        
        return slot, erase

    def _release(self, j):
        owner = self.slots[j]["owner"]
        if owner is not None and self.owner.get(owner) == j:
            del self.owner[owner]
        self.slots[j]["owner"] = None
    
    def _apply(self, req, slot, erase):
        if slot is None:
            req["reject"] = True
            req.pop("key", None)
            return
        evicted = [self.slots[j]["owner"] for j in erase + [slot]
                   if self.slots[j]["state"] == "idle" and self.slots[j]["owner"] != req["task_id"]]
        
        for j in erase:  # tokens stay counted until the erase has happened
            self._release(j)
            self.slots[j].update(state="erasing", key=None)
        if self.slots[slot]["owner"] != req["task_id"]:
            self._release(slot)

        s = self.slots[slot]
        s.update(state="running", owner=req["task_id"], tokens=req["need"], key=req.pop("key"), last=self.clock())
        self.owner[req["task_id"]] = slot
        req.update(slot=slot, erase=erase, evicted=evicted,
                   pool_est=sum(x["tokens"] for x in self.slots))
            
    def pop(self):
        if not self.q or self.running >= self.k:
            return None
        
        plan = self._plan(self.q[0])
        if plan is not None:
            self.running += 1
            req = self.q.popleft()
            self._apply(req, *plan)
            return req
        return None

    # calculate how many tokens the request needs in total
    async def before_queue(self, req, prompt):
        t0 = self.clock()
        req["need"] = await tokenize(self.client, prompt) + req["n_out"]
        req["key"] = prompt[:400]
        req["t_before_queue"] = self.clock() - t0

    # async work after queueing
    async def before_send(self, req):
        for j in req.get("erase", []):
            await erase_slot(self.client, j)
            self.slots[j].update(state="free", tokens=0)
        if req.get("erase"):  # erasing memory can let another request in
            self._dispatch()

    # async work after request finishes
    async def after_request(self, req, res):
        self.running -= 1
        slot = req.get("slot")
        if slot is None:
            return
        s = self.slots[slot]

        if "error" in res or not self.retain:  # if there is an error, delete its context
            self._release(slot)
            s.update(state="erasing", key=None)
            try:
                await erase_slot(self.client, slot)
            finally:
                s.update(state="free", tokens=0)
        elif s["state"] == "running": # success, context kept
            s["tokens"] = (res.get("prompt_tokens") or 0) + res.get("decode_n", 0)
            s["state"] = "idle" if s["owner"] == req["task_id"] else "free"
            s["last"] = self.clock()    

    # for when an entire request is done
    def task_done(self, task_id):
        slot = self.owner.pop(task_id, None)
        if slot is not None and self.slots[slot]["state"] == "idle":
            self.slots[slot].update(state="free", owner=None)
            self._dispatch()

class SlotDiscard(SlotAware):
    def __init__(self, client, clock, k=BASE["slots"], **kw):
        super().__init__(client, clock, k, retain=False, **kw)

POLICIES = {"fifo": FIFO, "slot": SlotAware, "slot_discard": SlotDiscard}