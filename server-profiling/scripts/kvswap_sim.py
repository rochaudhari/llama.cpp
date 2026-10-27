"""Simulate the unified KV pool under pressure and compare how the server chooses what to suspend.

No GPU: a step-time model fitted to llama-server measurements on an A40 (Qwen3-30B-A3B Q4_K_M, f16 KV, shared KV) stands in
for the decode loop. Copies cost what pinbench measured: bytes at the save rate, plus one call per contiguous run of
cells, per layer, for K and V. The point is the comparison between policies on the same load; absolute times are only
as good as the model. Restore costs come from restorebench.

The server follows the kv-swap-mvp branch (f7d4450c6): one unified pool; each step one token per
generating request plus prompt chunks up to n_batch; --cache-idle-slots saves and clears every other idle slot when a
task starts; the RAM prompt cache is FIFO with a byte limit and hands an entry back on load; suspended snapshots are
protected entries; at most one restore per step, followed by a step for the restored request alone.
Not modelled: prefix sharing between slots, hybrid/recurrent memory, speculative decoding, CUDA graph churn.

Policies:
  fail     master: purge idle slots, then error every processing request
  largest  the branch: save idle slots, suspend the largest requests, restore FIFO; an episode that cannot reserve
           RAM falls back to master's path
  bestfit  the branch, with the cheapest victim to copy instead of the largest
  tiered   trim prompt chunks, then idle slots (LRU, saved only without evicting), then the cheapest set of swaps and
           recomputes within RAM, then fail one request; new tasks wait while a victim has waited too long
  newest   vLLM-like: recompute the most recently admitted request; no new tasks while one waits
  admit    no swapping: start a task only if every running task's prompt + max_tokens fits in the pool

Scoring: goodput (completed workflows, or requests, per hour); failures, pauses and waits next to it.
usage: kvswap_sim.py --workload agents --pool 65536 --slots 8 --workflows 10 [--policies fail,largest,tiered]
       kvswap_sim.py --workload generic --rate 0.5 --requests 300 --prompt-tokens 8000 --gen-tokens 512
       kvswap_sim.py --selftest
"""
import argparse
import heapq
import itertools
import json
import math
import random
import sys
import time
from collections import deque

GIB = 1024 ** 3


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


MODELS = {
    # Qwen3-30B-A3B Q4_K_M, f16 KV (48 layers x 4 KV heads x 128): server decode steps of 6.2 ms at 1 client, 32.4 ms
    # at 16 x 512+1024, 4 x 4K and 4 x 16K with shared KV; prompt processing at 4 x 4K and 4 x 16K, per-slot vs shared
    "moe30b": dict(cell_bytes=98304, attn_layers=48, tok=0.237e-3, att=10.1e-9, dec_att=0.185e-6),
    # Qwen3.6-27B Q4_K_M (dense; full attention every 4th of 64 layers, 4 KV heads x 256, f16): llama-batched-bench
    # -kvu at 4K and 16K, 1-16 sequences (commands in README.md), plus the server's CPU time per step over the engine
    "dense27b": dict(cell_bytes=65536, attn_layers=16, tok=0.676e-3, att=6.8e-9, dec_att=0.115e-6,
                     dec=[(1, 32.3e-3), (4, 40.0e-3), (8, 59.7e-3), (16, 67.5e-3)], srv=(0.3e-3, 0.32e-3)),
}


class Model:
    """Step and copy times on the A40; --model picks the calibration."""

    def __init__(self, a):
        self.p = MODELS[a.model]
        self.cell_bytes = a.cell_bytes or self.p["cell_bytes"]
        self.layers = a.attn_layers or self.p["attn_layers"]
        self.save_bw = a.save_gbps * 1e9
        self.restore_bw = a.restore_gbps * 1e9
        self.save_call = a.copy_us * 1e-6
        self.restore_call = a.restore_copy_us * 1e-6
        self.shared = a.attn == "shared"
        self.gather = a.gather

    def decode(self, n, attended):
        # attended = cells the step's attention reads (all used cells once per 16 queries with shared KV)
        p = self.p
        if "dec" not in p:
            return 2.375e-3 + 3.545e-3 * n ** 0.733 + p["dec_att"] * attended
        pts = p["dec"]
        lo, hi = pts[0], pts[-1]
        for a, b in zip(pts, pts[1:]):
            if n <= b[0]:
                lo, hi = a, b
                break
        base = lo[1] + (hi[1] - lo[1]) * (n - lo[0]) / (hi[0] - lo[0])
        return base + p["srv"][0] + p["srv"][1] * n + p["dec_att"] * attended

    def prefill_token(self, attended):
        # each prompt token reads the cells it attends to
        return self.p["tok"] + self.p["att"] * attended

    def step(self, items, used):
        """items: (kind, tokens, the request's own cells before the step)."""
        n_dec = sum(1 for kind, _, _ in items if kind == "gen")
        if self.shared:
            att = used * math.ceil(n_dec / 16)
            pre = sum(k * self.prefill_token(used + k / 2) for kind, k, _ in items if kind == "pre")
        else:
            att = sum(c for kind, _, c in items if kind == "gen")
            pre = sum(k * self.prefill_token(c + k / 2) for kind, k, c in items if kind == "pre")
        return (self.decode(n_dec, att) if n_dec else 2.375e-3) + pre

    def recompute(self, cells, used):
        return cells * self.prefill_token(used if self.shared else cells / 2)

    def save(self, cells, runs):
        # a fresh pageable buffer (page faults), plus one copy call per run, per layer, for K and V
        runs = 1 if self.gather else runs
        return cells * self.cell_bytes / self.save_bw + 2 * self.layers * runs * self.save_call

    def restore(self, cells, runs):
        # restorebench: one tensor_set per destination run, per layer, for K and V, from the snapshot's buffer
        runs = 1 if self.gather else runs
        return cells * self.cell_bytes / self.restore_bw + 2 * self.layers * runs * self.restore_call


class Req:
    def __init__(self, rid, arrival, prompt, gen, max_tokens, sess=None):
        self.id = rid
        self.arrival = arrival
        self.prompt = prompt          # tokens sent (the whole conversation for an agent turn)
        self.gen = gen                # tokens it will generate
        self.max_tokens = max_tokens  # what the client allows; admission reservations use it
        self.sess = sess
        self.slot = None
        self.state = "queued"         # queued, prefill, gen, suspended, dropped, done, failed
        self.done_prompt = 0
        self.done_gen = 0
        self.recompute = 0            # tokens to evaluate again after a drop
        self.seq = 0                  # admission order
        self.t_first = None
        self.t_last = None
        self.t_end = None
        self.resume_state = None
        self.next_work = 1
        self.snap = None              # its protected cache entry while suspended
        self.t_susp = None
        self.n_susp = 0
        self.restored_step = -(1 << 30)
        self.trim_since = None

    def prompt_left(self):
        return self.recompute + self.prompt - self.done_prompt


class Slot:
    def __init__(self, i):
        self.id = i
        self.req = None
        self.owner = None   # the session (or request) whose KV the slot holds
        self.cells = 0
        self.runs = 0       # contiguous runs of cells: one per prompt chunk, one per token decoded alongside others
        self.t_used = 0.0


class Entry:
    def __init__(self, kind, key, cells, runs, nbytes):
        self.kind = kind    # "reuse": a saved conversation; "prot": a suspended request's snapshot
        self.key = key
        self.cells = cells
        self.runs = runs
        self.bytes = nbytes
        self.pinned = False   # about to be loaded: not evicted to make room for others


class Cache:
    """The RAM prompt cache: FIFO, a byte limit (0 = none); making room evicts reusable entries, oldest first."""

    def __init__(self, limit, enabled=True):
        self.limit = limit
        self.enabled = enabled
        self.entries = []
        self.size = 0
        self.peak = 0
        self.evicted_cells = 0

    def add(self, e):
        self.entries.append(e)
        self.size += e.bytes
        self.peak = max(self.peak, self.size)

    def remove(self, e):
        self.entries.remove(e)
        self.size -= e.bytes

    def find(self, key):
        return next((e for e in self.entries if e.kind == "reuse" and e.key == key), None)

    def room(self):
        if not self.enabled:
            return 0
        return math.inf if not self.limit else self.limit - self.size

    def make_room(self, nbytes):
        if not self.enabled or (self.limit and nbytes > self.limit):
            return False
        while self.limit and self.size + nbytes > self.limit:
            e = next((e for e in self.entries if e.kind == "reuse" and not e.pinned), None)
            if e is None:
                return False
            self.remove(e)
            self.evicted_cells += e.cells
        return True


class Batch:
    def __init__(self):
        self.items = []     # [req, tokens, "gen" or "pre"]
        self.trimmed = set()

    def tokens(self):
        return sum(it[1] for it in self.items)

    def owners(self):
        return {it[0] for it in self.items}

    def work(self, r):
        return sum(it[1] for it in self.items if it[0] is r)

    def remove(self, r):
        k = self.work(r)
        self.items = [it for it in self.items if it[0] is not r]
        return k


class Sim:
    def __init__(self, a, policy, workload, check=False):
        self.a = a
        self.m = Model(a)
        self.policy = policy
        self.wl = workload
        self.check = check
        self.t = 0.0
        self.step_no = 0
        self.pool = a.pool
        self.used = 0
        self.slots = [Slot(i) for i in range(a.slots)]
        self.cache = Cache(int(max(a.cache_ram, 0) * GIB), enabled=a.cache_ram != 0)
        self.queue = deque()
        self.events = []
        self.waiters = deque()
        self.promise = None
        self.seq = itertools.count(1)
        self.ev_seq = itertools.count()
        self.reqs = []
        self.gaps = []
        self.waits = []
        self.stuck = 0
        self.out_avg = a.out_estimate   # moving average of completed outputs, for hybrid admission
        self.st = dict.fromkeys(["copy_idle", "copy_load", "copy_swap", "copy_restore", "step_s"], 0.0)
        self.st.update(dict.fromkeys(["steps", "episodes", "victims", "drops", "tok_recompute", "trimmed",
                                      "fail_all", "failed_one", "tok_gen", "tok_prompt", "idle_saves",
                                      "idle_purges", "loads", "load_misses", "stuck_fails"], 0))

    def free(self):
        return self.pool - self.used

    def copy(self, secs, kind):
        self.t += secs   # synchronous on the inference thread: every request waits
        self.st[kind] += secs

    def schedule(self, t, r):
        heapq.heappush(self.events, (t, next(self.ev_seq), r))
        self.reqs.append(r)

    def running(self):
        return [s.req for s in self.slots if s.req is not None and s.req.state in ("prefill", "gen")]

    def idle(self):
        return [s for s in self.slots if s.req is None and s.cells]

    @staticmethod
    def key(r):
        return ("s", r.sess.id) if r.sess else ("r", r.id)

    def clear(self, s):
        self.used -= s.cells
        s.cells = s.runs = 0
        s.owner = None

    def placed_runs(self, e):
        # where a loaded or restored state lands: as scattered as it was saved, or in one free stretch
        return e.runs if self.a.restore_layout == "same" else 1

    def save_idle(self, s):
        """prompt_save(): an ordinary entry, unless the cache holds this conversation already or can't make room."""
        old = self.cache.find(s.owner)
        if old is not None:
            if old.cells >= s.cells:
                return False
            self.cache.remove(old)
        nbytes = s.cells * self.m.cell_bytes
        if not self.cache.make_room(nbytes):
            return False
        self.cache.add(Entry("reuse", s.owner, s.cells, s.runs, nbytes))
        self.copy(self.m.save(s.cells, s.runs), "copy_idle")
        self.st["idle_saves"] += 1
        return True

    # tasks

    def launch(self):
        while self.queue and self.policy.can_launch(self, self.queue[0]):
            s = self.pick_slot(self.queue[0])
            if s is None:
                return
            self.start(self.queue.popleft(), s)

    def pick_slot(self, r):
        free = [s for s in self.slots if s.req is None]
        if not free:
            return None
        own = [s for s in free if s.cells and s.owner == self.key(r)]
        empty = [s for s in free if not s.cells]
        return (own or empty or [min(free, key=lambda s: s.t_used)])[0]

    def start(self, r, s):
        key = self.key(r)
        # get_available_slot(): no cache save or load while a restored request waits for its promised step
        update_cache = self.promise is None
        if s.cells and s.owner != key:
            if update_cache:
                self.save_idle(s)
            self.clear(s)
        s.req = r
        if self.a.idle_before_load and self.a.cache_idle_slots:
            # the --cache-idle-slots pass ahead of the load instead of after it, so the load finds the cells. The entry
            # to load is pinned, or saving the idle slots could evict it, and its bytes count as already handed back
            # (as when the load comes first), so the cache briefly holds one entry over its limit
            pin = self.cache.find(key) if not s.cells and update_cache else None
            if pin is not None:
                pin.pinned = True
                self.cache.size -= pin.bytes
            for o in self.idle():
                self.save_idle(o)
                self.clear(o)
            if pin is not None:
                pin.pinned = False
                self.cache.size += pin.bytes
        if s.cells:
            r.done_prompt = min(s.cells, r.prompt - 1)
        elif update_cache:
            e = self.cache.find(key)
            if e is not None and e.cells <= self.free():
                self.cache.remove(e)
                self.copy(self.m.restore(e.cells, e.runs), "copy_load")
                s.cells, s.runs = e.cells, self.placed_runs(e)
                self.used += e.cells
                r.done_prompt = min(e.cells, r.prompt - 1)
                self.st["loads"] += 1
            elif e is not None:
                self.st["load_misses"] += 1
                self.cache.make_room(0)   # a pinned entry that did not load may leave the cache over its limit
        if s.cells > r.done_prompt:
            self.used -= s.cells - r.done_prompt
            s.cells = r.done_prompt
        s.owner, r.slot = key, s
        r.state = "prefill"
        r.seq = next(self.seq)
        if self.a.cache_idle_slots and not self.a.idle_before_load:
            for o in self.idle():
                self.save_idle(o)
                self.clear(o)

    def finish(self, r, state):
        if state == "done":
            self.out_avg += 0.1 * (r.gen - self.out_avg)
        r.state = state
        r.t_end = self.t
        s = r.slot
        if s is not None:
            s.req = None
            s.t_used = self.t
            if state != "done":
                self.clear(s)
        if r.snap is not None:
            self.cache.remove(r.snap)
            r.snap = None
        if r in self.waiters:
            self.waiters.remove(r)
        if self.promise is not None and self.promise[0] is r:
            self.promise = None
        r.slot = None
        self.wl.on_end(self, r)

    def fail_all(self, b):
        """Master's last resort: every processing request gets an error, suspended ones included."""
        for s in self.slots:
            if s.req is not None:
                self.finish(s.req, "failed")
        b.items = []
        self.st["fail_all"] += 1

    # suspension

    def reserve(self, r):
        """prepare_suspend_slot(): a protected entry (evicting reusable ones if needed) and the copy into it."""
        s = r.slot
        nbytes = s.cells * self.m.cell_bytes
        if not self.cache.make_room(nbytes):
            return None
        e = Entry("prot", r.id, s.cells, s.runs, nbytes)
        self.cache.add(e)
        self.copy(self.m.save(s.cells, s.runs), "copy_swap")
        return e

    def clear_cells(self, s):
        self.used -= s.cells
        s.cells = s.runs = 0

    def suspend(self, r, e, b, next_work=None):
        """commit_prepared_suspension(): take its work out of the batch, free its cells, wait for a restore."""
        k = b.remove(r)
        r.next_work = max(1, k) if next_work is None else next_work
        r.resume_state, r.state = r.state, "suspended"
        r.snap = e
        r.t_susp = self.t
        r.n_susp += 1
        self.clear_cells(r.slot)
        self.waiters.append(r)
        self.st["victims"] += 1

    def restore(self, r):
        e, r.snap = r.snap, None
        self.cache.remove(e)
        self.copy(self.m.restore(e.cells, e.runs), "copy_restore")
        s = r.slot
        s.cells, s.runs = e.cells, self.placed_runs(e)
        self.used += e.cells
        r.state = r.resume_state
        r.restored_step = self.step_no
        self.waits.append(self.t - r.t_susp)
        self.promise = (r, r.next_work)

    def drop(self, r, b, front=False):
        """Discard its KV; it keeps its tokens and re-evaluates them when it resumes."""
        k = b.remove(r)
        r.next_work = max(1, k)
        r.recompute += r.slot.cells
        r.resume_state, r.state = r.state, "dropped"
        r.t_susp = self.t
        r.n_susp += 1
        self.clear_cells(r.slot)
        (self.waiters.appendleft if front else self.waiters.append)(r)
        self.st["drops"] += 1

    def resume_dropped(self, r, newest):
        r.state = "prefill"
        r.restored_step = self.step_no
        self.waits.append(self.t - r.t_susp)
        if newest:
            r.seq = next(self.seq)

    # steps

    def build_batch(self):
        b = Batch()
        if self.promise is not None:
            r, cells = self.promise
            if r.state == "gen":
                b.items.append([r, 1, "gen"])
                return b
            if r.state == "prefill":
                b.items.append([r, min(self.a.n_batch, cells, r.prompt_left()), "pre"])
                return b
            self.promise = None
        for s in self.slots:
            if s.req is not None and s.req.state == "gen":
                b.items.append([s.req, 1, "gen"])
        room = self.a.n_batch - b.tokens()
        for s in self.policy.prefill_order(self):
            r = s.req
            if r is not None and r.state == "prefill" and room > 0:
                k = min(r.prompt_left(), room)
                b.items.append([r, k, "pre"])
                room -= k
        return b

    def token(self, r):
        r.done_gen += 1
        self.st["tok_gen"] += 1
        if r.t_last is None:
            r.t_first = self.t
        else:
            self.gaps.append(self.t - r.t_last)
        r.t_last = self.t
        if r.done_gen >= r.gen:
            self.finish(r, "done")
        else:
            r.state = "gen"

    def execute(self, b):
        dt = self.m.step([(kind, k, r.slot.cells) for r, k, kind in b.items], self.used)
        self.t += dt
        self.st["steps"] += 1
        self.st["step_s"] += dt
        promised = self.promise is not None and any(it[0] is self.promise[0] for it in b.items)
        for r, k, kind in b.items:
            s = r.slot
            s.cells += k
            self.used += k
            if kind == "pre":
                s.runs += 1
                x = min(k, r.recompute)
                r.recompute -= x
                r.done_prompt += k - x
                self.st["tok_prompt"] += k
                self.st["tok_recompute"] += x
                if r not in b.trimmed:
                    r.trim_since = None
                if r.prompt_left() == 0:
                    if r.done_gen == 0:
                        self.token(r)       # the first token is sampled at the end of the prompt
                    else:
                        r.state = "gen"     # back from a drop: the pending token goes in next step
            else:
                if len(b.items) > 1:
                    s.runs += 1             # cells of concurrent requests interleave
                self.token(r)
        if promised:
            self.promise = None
        if self.check:
            self.verify()

    def verify(self):
        assert self.used == sum(s.cells for s in self.slots), "cell count"
        assert 0 <= self.used <= self.pool, "pool overflow"
        assert not self.cache.limit or self.cache.size <= self.cache.limit, "cache over its limit"
        assert self.cache.size == sum(e.bytes for e in self.cache.entries), "cache size"
        for s in self.slots:
            if s.req is not None:
                assert s.req.slot is s, "slot link"
                if s.req.state in ("suspended", "dropped"):
                    assert s.cells == 0, "a waiting request holds cells"

    def idle_advance(self):
        if self.events:
            self.t = max(self.t, self.events[0][0])
            return True
        if not (self.queue or self.waiters):
            return False
        self.stuck += 1
        if self.stuck > 2:   # nothing can move: fail the oldest blocked request
            r = self.waiters[0] if self.waiters else self.queue.popleft()
            self.finish(r, "failed")
            self.st["stuck_fails"] += 1
            self.stuck = 0
        return True

    def run(self, max_steps=20_000_000):
        t_wall = time.monotonic()
        self.wl.start(self)
        while not self.wl.finished():
            if self.a.wall_limit and self.step_no % 4096 == 0 and time.monotonic() - t_wall > self.a.wall_limit:
                raise TimeoutError(f"over {self.a.wall_limit} s of wall time at simulated {self.t:.0f} s")
            while self.events and self.events[0][0] <= self.t:
                self.queue.append(heapq.heappop(self.events)[2])
            self.launch()
            self.policy.restore(self)
            b = self.build_batch()
            if b.items and b.tokens() > self.free():
                self.policy.pressure(self, b)
                if b.tokens() > self.free():
                    self.fail_all(b)
            if b.items:
                self.execute(b)
                self.step_no += 1
                self.stuck = 0
            elif not self.idle_advance():
                break
            if self.step_no > max_steps:
                raise RuntimeError("too many steps")
        return self


class Policy:
    name = "none"

    def __init__(self, a):
        self.a = a

    def can_launch(self, sim, r):
        return True

    def prefill_order(self, sim):
        return sim.slots   # llama-server fills the batch's prompt chunks in slot order

    def restore(self, sim):
        pass

    def pressure(self, sim, b):
        pass


class Fail(Policy):
    name = "fail"

    def pressure(self, sim, b):
        # try_clear_idle_slots(), then the "Context size has been exceeded" path
        for s in sim.idle():
            if b.tokens() <= sim.free():
                return
            sim.clear(s)
            sim.st["idle_purges"] += 1


class Largest(Policy):
    name = "largest"

    @staticmethod
    def reclaim_idle(sim):
        # try_reclaim_idle_slot(): the first idle slot by id, saved as an ordinary entry, then cleared
        idle = sim.idle()
        if not idle:
            return False
        sim.save_idle(idle[0])
        sim.clear(idle[0])
        return True

    def restore(self, sim):
        # try_restore_pressure_waiter(): FIFO, the first whose snapshot plus next work fits; one per step
        if sim.promise is not None:
            return
        for r in list(sim.waiters):
            need = r.snap.cells + r.next_work
            if need > sim.pool:
                sim.finish(r, "failed")
                continue
            while need > sim.free() and self.reclaim_idle(sim):
                pass
            if need <= sim.free():
                sim.waiters.remove(r)
                sim.restore(r)
                return

    def pick(self, sim, b, deficit):
        owners = b.owners()
        cands = sorted((r for r in sim.running() if r.slot.cells), key=lambda r: (-r.slot.cells, r.slot.id))
        if cands and cands[0].slot.cells >= deficit and (cands[0] not in owners or len(owners) > 1):
            return [cands[0]]
        sel, planned, n_own = [], 0, 0
        for r in cands:
            if r in owners and n_own + 1 >= len(owners):
                continue
            sel.append(r)
            planned += r.slot.cells
            n_own += r in owners
            if planned >= deficit:
                return sel
        return None

    def pressure(self, sim, b):
        while b.tokens() > sim.free() and self.reclaim_idle(sim):
            pass
        if b.tokens() <= sim.free():
            return
        sel = self.pick(sim, b, b.tokens() - sim.free())
        if not sel:
            return
        prepared = []
        for r in sel:
            e = sim.reserve(r)
            if e is None:
                # the episode is dropped; master's path then errors everyone, the waiting victims included
                for _, pe in prepared:
                    sim.cache.remove(pe)
                return
            prepared.append((r, e))
        for r, e in prepared:
            sim.suspend(r, e, b)
        sim.st["episodes"] += 1


class Bestfit(Largest):
    name = "bestfit"

    def pick(self, sim, b, deficit):
        owners = b.owners()
        cands = [r for r in sim.running() if r.slot.cells]
        cost = {r: sim.m.save(r.slot.cells, r.slot.runs) + sim.m.restore(r.slot.cells, r.slot.runs) for r in cands}
        best = None
        for n in range(1, min(3, len(cands)) + 1):
            for sel in itertools.combinations(cands, n):
                if sum(r.slot.cells for r in sel) < deficit or owners <= set(sel):
                    continue
                c = sum(cost[r] for r in sel)
                if best is None or c < best[0]:
                    best = (c, list(sel))
        if best is not None:
            return best[1]
        sel, planned = [], 0
        for r in sorted(cands, key=lambda r: cost[r] / r.slot.cells):
            if owners <= set(sel + [r]):
                continue
            sel.append(r)
            planned += r.slot.cells
            if planned >= deficit:
                return sel
        return None


class Tiered(Policy):
    name = "tiered"
    allow_swap = True

    def can_launch(self, sim, r):
        return not sim.waiters or sim.t - sim.waiters[0].t_susp <= self.a.age_max

    @staticmethod
    def reclaim_idle(sim):
        idle = sim.idle()
        if not idle:
            return False
        s = min(idle, key=lambda s: s.t_used)
        if s.cells * sim.m.cell_bytes <= sim.cache.room():
            sim.save_idle(s)
        else:
            sim.st["idle_purges"] += 1
        sim.clear(s)
        return True

    def restore(self, sim):
        if sim.promise is not None:
            return
        n_gen = sum(1 for x in sim.running() if x.state == "gen")
        for r in list(sim.waiters):
            need = (r.snap.cells if r.state == "suspended" else r.recompute) + r.next_work
            if need > sim.pool:
                sim.finish(r, "failed")
                continue
            # come back only with room to stay for a while, unless it has waited too long already
            if sim.t - r.t_susp <= self.a.age_max:
                need = min(sim.pool, need + self.a.restore_headroom * n_gen)
            while need > sim.free() and self.reclaim_idle(sim):
                pass
            if need <= sim.free():
                sim.waiters.remove(r)
                if r.state == "suspended":
                    sim.restore(r)
                else:
                    sim.resume_dropped(r, newest=False)
                return

    def pressure(self, sim, b):
        # 1. trim prompt chunks, newest-admitted request first
        for it in sorted((it for it in b.items if it[2] == "pre"), key=lambda it: -it[0].seq):
            over = b.tokens() - sim.free()
            if over <= 0:
                break
            r = it[0]
            if r.trim_since is not None and sim.t - r.trim_since > self.a.trim_max:
                continue
            cut = min(it[1], over, b.tokens() - 1)
            if cut > 0:
                it[1] -= cut
                b.trimmed.add(r)
                sim.st["trimmed"] += cut
                if r.trim_since is None:
                    r.trim_since = sim.t
        b.items = [it for it in b.items if it[1] > 0]
        # 2. idle slots, least recently used first
        while b.tokens() > sim.free() and self.reclaim_idle(sim):
            pass
        if b.tokens() <= sim.free():
            return
        # 3-4. the cheapest set of swaps and recomputes; a request just restored only if nothing else covers it
        if self.evict(sim, b, self.a.min_resident) or self.evict(sim, b, 0):
            sim.st["episodes"] += 1
            return
        # 5. fail one request at a time, the newest; never a waiting victim
        while b.tokens() > sim.free():
            owners = b.owners()
            cands = [r for r in sim.running() if r.slot.cells or r in owners]
            if not cands:
                return
            r = max(cands, key=lambda r: r.seq)
            b.remove(r)
            sim.finish(r, "failed")
            sim.st["failed_one"] += 1

    def victim_pool(self, sim, cands):
        return cands

    def evict(self, sim, b, min_resident):
        a, m = self.a, sim.m
        deficit = b.tokens() - sim.free()
        owners = b.owners()
        n_dec = sum(1 for it in b.items if it[2] == "gen")
        cands = [r for r in sim.running() if r.slot.cells and sim.step_no - r.restored_step >= min_resident]
        cands = self.victim_pool(sim, cands)
        if not cands:
            return False
        per_tok = m.prefill_token(sim.used)
        promise_step = m.decode(1, sim.used)
        info = {}
        for r in cands:
            c, runs = r.slot.cells, r.slot.runs
            swap = m.save(c, runs) + m.restore(c, runs) + promise_step if self.allow_swap else math.inf
            drop = m.recompute(c, sim.used) if a.recompute else math.inf
            info[r] = (c + b.work(r), swap, drop, c * m.cell_bytes)
        room = sim.cache.room()
        if a.mem_available:
            room = min(room, a.mem_available * GIB - sim.cache.size)
        evictable = [(e.bytes, a.p_hit * e.cells * per_tok) for e in sim.cache.entries if e.kind == "reuse"]

        def ram_cost(nbytes):   # expected re-prefill of the reusable entries evicted to fit nbytes; None if it can't
            need, c = nbytes - room, 0.0
            for eb, ec in evictable:
                if need <= 0:
                    break
                need -= eb
                c += ec
            return c if need <= 0 else None

        def evaluate(sel):
            benefit = sum(info[r][0] for r in sel)
            swaps, nbytes, cost = [], 0, 0.0
            for r in sorted(sel, key=lambda r: info[r][1] - info[r][2]):
                _, swap, drop, nb = info[r]
                extra = ram_cost(nbytes + nb)
                if extra is not None and swap + extra - (ram_cost(nbytes) or 0.0) < drop:
                    swaps.append(r)
                    nbytes += nb
                elif drop == math.inf:
                    return None
            cost = sum(info[r][1] for r in swaps) + sum(info[r][2] for r in sel if r not in swaps)
            return benefit, cost + (ram_cost(nbytes) or 0.0), swaps

        best = None
        for target in (deficit + a.headroom * n_dec, deficit):
            for n in range(1, min(3, len(cands)) + 1):
                for sel in itertools.combinations(cands, n):
                    if owners <= set(sel):
                        continue
                    ev = evaluate(sel)
                    if ev is not None and ev[0] >= target and (best is None or ev[1] < best[1]):
                        best = (sel, ev[1], ev[2])
            if best is None:   # bigger sets: cheapest per cell first
                sel = []
                for r in sorted(cands, key=lambda r: min(info[r][1], info[r][2]) / info[r][0]):
                    if owners <= set(sel + [r]):
                        continue
                    sel.append(r)
                    ev = evaluate(sel)
                    if ev is not None and ev[0] >= target:
                        best = (sel, ev[1], ev[2])
                        break
            if best is not None:
                break
        if best is None:
            return False
        sel, _, swaps = best
        for r in sel:
            e = sim.reserve(r) if r in swaps else None
            if e is not None:
                sim.suspend(r, e, b, next_work=1)
            elif a.recompute:
                sim.drop(r, b)
        return b.tokens() <= sim.free()


class Hybrid(Tiered):
    """Optimistic admission: start a task if every request's prompt plus the output we expect fits (a multiple of
    the recent average, not max_tokens); tiered handles the requests that outgrow the guess."""
    name = "hybrid"

    def can_launch(self, sim, r):
        return super().can_launch(sim, r) and admits(sim, self.a, r)


def admits(sim, a, r):
    """Optimistic admission: every running or waiting request's prompt plus its expected output, and the new one's."""
    run = sim.running()
    est = min(a.max_tokens, a.out_margin * sim.out_avg)
    held = sum(x.prompt + max(est, x.done_gen + 1) for x in itertools.chain(run, sim.waiters))
    return not run or held + r.prompt + est <= sim.pool


class BestfitDrop(Bestfit):
    """bestfit, but victims whose snapshots do not fit in RAM are dropped and recomputed later, instead of the
    episode falling back to master's path."""
    name = "bestfit-d"

    def restore(self, sim):
        if sim.promise is not None:
            return
        for r in list(sim.waiters):
            need = (r.snap.cells if r.state == "suspended" else r.recompute) + r.next_work
            if need > sim.pool:
                sim.finish(r, "failed")
                continue
            while need > sim.free() and self.reclaim_idle(sim):
                pass
            if need <= sim.free():
                sim.waiters.remove(r)
                if r.state == "suspended":
                    sim.restore(r)
                else:
                    sim.resume_dropped(r, newest=False)
                return

    def pressure(self, sim, b):
        while b.tokens() > sim.free() and self.reclaim_idle(sim):
            pass
        if b.tokens() <= sim.free():
            return
        sel = self.pick(sim, b, b.tokens() - sim.free())
        if not sel:
            return
        prepared = []
        for r in sel:
            e = sim.reserve(r)
            if e is None:
                break
            prepared.append((r, e))
        for r, e in prepared:
            sim.suspend(r, e, b)
        for r in sel[len(prepared):]:
            sim.drop(r, b)
        sim.st["episodes"] += 1


class HybridBestfit(BestfitDrop):
    """hybrid's admission, with bestfit-d handling the requests that outgrow the guess."""
    name = "hybrid-b"

    def can_launch(self, sim, r):
        if sim.waiters and sim.t - sim.waiters[0].t_susp > self.a.age_max:
            return False
        return admits(sim, self.a, r)


class HybridFail(Hybrid):
    """hybrid's admission alone: an overflow takes master's path (purge idle slots, then error every request)."""
    name = "hybrid-f"

    def restore(self, sim):
        pass

    def pressure(self, sim, b):
        Fail.pressure(self, sim, b)


class HybridRecompute(Hybrid):
    """hybrid's admission with recompute only: overflow victims are dropped, nothing is kept in RAM."""
    name = "hybrid-r"
    allow_swap = False


class HybridOverrun(Hybrid):
    """hybrid, but the victims are the requests that outgrew their reservation, if any (largest overrun first);
    the cost function still picks among them and decides swap or recompute."""
    name = "hybrid-o"

    def victim_pool(self, sim, cands):
        est = min(self.a.max_tokens, self.a.out_margin * sim.out_avg)
        over = [r for r in cands if r.done_gen > est]
        return over or cands


class Newest(Policy):
    name = "newest"

    def can_launch(self, sim, r):
        return not sim.waiters   # preempted requests sit at the front of the waiting queue

    def prefill_order(self, sim):
        # vLLM gives the token budget to running requests oldest first; in slot order the newest request can take
        # the whole chunk budget, be preempted for it, resume and repeat forever
        return sorted(sim.slots, key=lambda s: s.req.seq if s.req is not None else 0)

    @staticmethod
    def purge_idle(sim):
        idle = sim.idle()
        if not idle:
            return False
        sim.clear(min(idle, key=lambda s: s.t_used))
        sim.st["idle_purges"] += 1
        return True

    def restore(self, sim):
        if not sim.waiters:
            return
        r = sim.waiters[0]
        while r.recompute + 1 > sim.free() and self.purge_idle(sim):
            pass
        if r.recompute + 1 <= sim.free():
            sim.waiters.popleft()
            sim.resume_dropped(r, newest=True)

    def pressure(self, sim, b):
        dropped = False
        while b.tokens() > sim.free():
            if self.purge_idle(sim):
                continue
            owners = b.owners()
            cands = [r for r in sim.running() if r.slot.cells or r in owners]
            if not cands:
                break
            r = max(cands, key=lambda r: r.seq)
            if r.slot.cells:
                sim.drop(r, b, front=True)
                dropped = True
            else:
                b.remove(r)   # nothing to free: it just waits this step
        sim.st["episodes"] += dropped


class Admit(Policy):
    name = "admit"

    def can_launch(self, sim, r):
        run = sim.running()
        held = sum(x.prompt + x.max_tokens for x in run)
        return not run or held + r.prompt + r.max_tokens <= sim.pool

    def pressure(self, sim, b):
        while b.tokens() > sim.free() and Tiered.reclaim_idle(sim):
            pass


POLICIES = {p.name: p for p in (Fail, Largest, Bestfit, BestfitDrop, Tiered, Hybrid, HybridOverrun, HybridBestfit,
                                HybridFail, HybridRecompute, Newest, Admit)}


class Session:
    def __init__(self, sid, wf, ctx, turns):
        self.id = sid
        self.wf = wf
        self.ctx = ctx
        self.turns = deque(turns)
        self.gap = 0.0
        self.failed = False


class Workflow:
    def __init__(self, wid, t, n):
        self.id = wid
        self.t_start = t
        self.t_end = None
        self.left = n
        self.failed = False


class Agents:
    """Workflows of parallel subagents. A subagent is a multi-turn session: a system prompt, then turns of a tool
    result in and a reply out with a tool-call gap between them; every turn sends the whole conversation. A workflow
    is done when all its subagents are; the next one starts then (--concurrent workflows at a time)."""

    def __init__(self, a, rng):
        self.a = a
        self.rng = rng
        self.ids = itertools.count()
        self.started = 0
        self.ended = 0
        self.wfs = []

    def lognorm(self, median, sigma, lo, hi):
        return int(min(hi, max(lo, self.rng.lognormvariate(math.log(median), sigma))))

    def start(self, sim):
        for _ in range(self.a.concurrent):
            self.new_workflow(sim)

    def new_workflow(self, sim):
        a = self.a
        if self.started >= a.workflows:
            return
        wf = Workflow(self.started, sim.t, a.subagents)
        self.started += 1
        self.wfs.append(wf)
        cap = int(a.pool * 0.9)   # every request fits the pool on its own
        for k in range(a.subagents):
            ctx, turns = a.sys_tokens, []
            for _ in range(self.rng.randint(a.turns_min, a.turns_max)):
                u = self.lognorm(a.tool_tokens, 0.8, 50, 8 * a.tool_tokens)
                g = self.lognorm(a.gen_tokens, 0.6, 16, min(6 * a.gen_tokens, a.max_tokens))
                gap = self.rng.lognormvariate(math.log(a.gap), 0.7) if a.gap > 0 else 0.0
                if a.fixed:   # returning.py's shape: every follow-up, reply and gap the same size
                    u, g, gap = a.tool_tokens, a.gen_tokens, a.gap
                if ctx + u + g > cap:
                    break
                turns.append((u, g, gap))
                ctx += u + g
            if not turns:
                turns.append((50, 16, 0.0))
            sess = Session((wf.id, k), wf, a.sys_tokens, turns)
            self.next_turn(sim, sess, sim.t + self.rng.uniform(0, a.stagger))

    def next_turn(self, sim, sess, t):
        u, g, sess.gap = sess.turns.popleft()
        sess.ctx += u
        sim.schedule(t, Req(next(self.ids), t, sess.ctx, g, self.a.max_tokens, sess))

    def on_end(self, sim, r):
        sess = r.sess
        if r.state == "done":
            sess.ctx += r.gen
            if sess.turns:
                return self.next_turn(sim, sess, sim.t + sess.gap)
        else:
            sess.failed = True
        wf = sess.wf
        wf.left -= 1
        wf.failed |= sess.failed
        if wf.left == 0:
            wf.t_end = sim.t
            self.ended += 1
            self.new_workflow(sim)

    def finished(self):
        return self.ended >= self.a.workflows


class Returning:
    """returning.py: each client owns sessions/clients sessions and takes their turns round-robin, one request at a
    time; a session's first turn is --prompt-tokens, each later one the whole conversation plus --tool-tokens."""

    def __init__(self, a, rng):
        self.a = a
        self.ended = 0
        self.wfs = None
        self.ids = itertools.count()
        self.total = a.sessions * a.turns_max

    def start(self, sim):
        a = self.a
        self.ctx = [0] * a.sessions
        self.turn = [0] * a.sessions
        for c in range(a.clients):
            self.issue(sim, c, list(range(c, a.sessions, a.clients)), 0)

    def issue(self, sim, client, own, i):
        if i >= len(own) * self.a.turns_max:
            return
        s = own[i % len(own)]
        self.ctx[s] += self.a.prompt_tokens if self.turn[s] == 0 else self.a.tool_tokens
        r = Req(next(self.ids), sim.t, self.ctx[s], self.a.gen_tokens, self.a.max_tokens)
        r.sess = Session(s, None, 0, [])
        r.client = (client, own, i)
        sim.schedule(sim.t, r)

    def on_end(self, sim, r):
        self.ended += 1
        s = r.sess.id
        self.ctx[s] += r.done_gen if r.state != "done" else r.gen
        self.turn[s] += 1
        client, own, i = r.client
        self.issue(sim, client, own, i + 1)

    def finished(self):
        return self.ended >= self.total


class Generic:
    """Independent single requests arriving at random (Poisson)."""

    def __init__(self, a, rng):
        self.a = a
        self.rng = rng
        self.ended = 0
        self.wfs = None

    def start(self, sim):
        a, rng, t = self.a, self.rng, 0.0
        cap = int(a.pool * 0.9)
        for i in range(a.requests):
            t += rng.expovariate(a.rate)
            p = int(min(cap // 2, max(16, rng.lognormvariate(math.log(a.prompt_tokens), 0.8))))
            g = int(min(cap - p, a.max_tokens, max(16, rng.lognormvariate(math.log(a.gen_tokens), 0.6))))
            sim.schedule(t, Req(i, t, p, g, a.max_tokens))

    def on_end(self, sim, r):
        self.ended += 1

    def finished(self):
        return self.ended >= self.a.requests


def run_one(a, name, check=False):
    rng = random.Random(a.seed)
    wl = {"agents": Agents, "generic": Generic, "returning": Returning}[a.workload](a, rng)
    sim = Sim(a, POLICIES[name](a), wl, check).run()
    reqs = [r for r in sim.reqs if r.t_end is not None]
    done = [r for r in reqs if r.state == "done"]
    span = max((r.t_end for r in reqs), default=sim.t) or 1e-9
    ttft = [r.t_first - r.arrival for r in done]
    out = dict(policy=name, ended=len(reqs), done=len(done), failed=len(reqs) - len(done), makespan_s=span,
               goodput_req_h=len(done) / span * 3600, gen_tok_s=sim.st["tok_gen"] / span,
               ttft_p50_s=pct(ttft, 50), ttft_p99_s=pct(ttft, 99), gap_p99_s=pct(sim.gaps, 99),
               gap_max_s=max(sim.gaps, default=0.0), wait_max_s=max(sim.waits, default=0.0),
               peak_ram_gib=sim.cache.peak / GIB, evicted_cells=sim.cache.evicted_cells, **sim.st)
    if wl.wfs is not None:
        ok = [w for w in wl.wfs if w.t_end is not None and not w.failed]
        out.update(wf_started=len(wl.wfs), wf_ok=len(ok), goodput_wf_h=len(ok) / span * 3600,
                   wf_p50_s=pct([w.t_end - w.t_start for w in ok], 50),
                   wf_max_s=max((w.t_end - w.t_start for w in ok), default=0.0))
    return out


COLUMNS = [
    ("policy", "policy", "{}"),
    ("wf ok", "wf_ok", "{}"),
    ("failed req", "failed", "{}"),
    ("goodput wf/h", "goodput_wf_h", "{:.1f}"),
    ("req/h", "goodput_req_h", "{:.0f}"),
    ("gen tok/s", "gen_tok_s", "{:.0f}"),
    ("TTFT p50/p99 s", ("ttft_p50_s", "ttft_p99_s"), "{:.1f}/{:.1f}"),
    ("gap p99/max s", ("gap_p99_s", "gap_max_s"), "{:.2f}/{:.1f}"),
    ("wait max s", "wait_max_s", "{:.0f}"),
    ("episodes", "episodes", "{}"),
    ("swapped", "victims", "{}"),
    ("dropped", "drops", "{}"),
    ("trimmed tok", "trimmed", "{}"),
    ("recompute tok", "tok_recompute", "{}"),
    ("swap copies s", ("copy_swap", "copy_restore"), "{:.1f}+{:.1f}"),
    ("cache copies s", ("copy_idle", "copy_load"), "{:.1f}+{:.1f}"),
    ("peak RAM GiB", "peak_ram_gib", "{:.1f}"),
]


def table(rows):
    cols = [c for c in COLUMNS if all(not isinstance(c[1], str) or c[1] in r for r in rows)]
    cells = [[c[0] for c in cols]]
    for r in rows:
        cells.append([fmt.format(*(r[k] for k in key)) if not isinstance(key, str) else fmt.format(r[key])
                      for _, key, fmt in cols])
    w = [max(len(row[i]) for row in cells) for i in range(len(cols))]
    return "\n".join("  ".join(v.rjust(w[i]) if i else v.ljust(w[i]) for i, v in enumerate(row)) for row in cells)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--workload", choices=["agents", "generic", "returning"], default="agents")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="write the results as JSON")
    ap.add_argument("--wall-limit", type=float, default=0, help="give up on a policy after this many seconds")
    g = ap.add_argument_group("server")
    g.add_argument("--pool", type=int, default=65536, help="KV cells in the unified pool (-c)")
    g.add_argument("--slots", type=int, default=8)
    g.add_argument("--n-batch", type=int, default=2048)
    g.add_argument("--cache-ram", type=float, default=8, help="GiB; -1 = no limit, 0 = no cache (as llama-server)")
    g.add_argument("--no-cache-idle-slots", dest="cache_idle_slots", action="store_false")
    g.add_argument("--idle-before-load", action="store_true",
                   help="save and clear idle slots before a task loads its prompt from the cache, not after")
    g.add_argument("--mem-available", type=float, default=0, help="GiB of free memory tiered respects; 0 = unchecked")
    g = ap.add_argument_group("model and copies")
    g.add_argument("--model", choices=list(MODELS), default="moe30b", help="step-time calibration")
    g.add_argument("--cell-bytes", type=int, default=0, help="KV bytes per cell (default: the model's)")
    g.add_argument("--attn-layers", type=int, default=0, help="(default: the model's)")
    g.add_argument("--save-gbps", type=float, default=2.0, help="into fresh pages (pinbench; also measured on server saves)")
    g.add_argument("--restore-gbps", type=float, default=12.0, help="from the snapshot's buffer (restorebench: 12-18)")
    g.add_argument("--copy-us", type=float, default=6.5, help="per save copy call (pinbench: 98,500 calls in 716 ms)")
    g.add_argument("--restore-copy-us", type=float, default=5.2, help="per restore copy call (restorebench)")
    g.add_argument("--attn", choices=["shared", "own"], default="shared",
                   help="what attention reads: every used cell (unified KV today) or each request's own (per-slot KV)")
    g.add_argument("--restore-layout", choices=["same", "contiguous"], default="same",
                   help="runs of a loaded or restored state: as saved, or one free stretch")
    g.add_argument("--gather", action="store_true", help="one copy per layer for K and V however scattered (a gathered save)")
    g = ap.add_argument_group("tiered")
    g.add_argument("--headroom", type=int, default=16, help="steps of decode growth to cover beyond the deficit")
    g.add_argument("--trim-max", type=float, default=5.0, help="seconds a prompt may be trimmed before it is exempt")
    g.add_argument("--age-max", type=float, default=30.0, help="seconds a victim may wait before new tasks wait")
    g.add_argument("--min-resident", type=int, default=32, help="steps after a restore before it is a victim again")
    g.add_argument("--restore-headroom", type=int, default=64, help="steps of decode growth free before a restore")
    g.add_argument("--out-estimate", type=float, default=512, help="hybrid: expected output before any completes")
    g.add_argument("--out-margin", type=float, default=2.0, help="hybrid: reserve this multiple of the average output")
    g.add_argument("--p-hit", type=float, default=0.5, help="chance an evicted cache entry would have been used")
    g.add_argument("--no-recompute", dest="recompute", action="store_false", help="as for hybrid models")
    g = ap.add_argument_group("agents workload")
    g.add_argument("--workflows", type=int, default=10)
    g.add_argument("--concurrent", type=int, default=1, help="workflows at a time")
    g.add_argument("--subagents", type=int, default=6)
    g.add_argument("--sys-tokens", type=int, default=3000)
    g.add_argument("--tool-tokens", type=int, default=1500, help="median tool result per turn")
    g.add_argument("--gen-tokens", type=int, default=300, help="median reply (agents) or output (generic)")
    g.add_argument("--turns-min", type=int, default=8)
    g.add_argument("--turns-max", type=int, default=20)
    g.add_argument("--gap", type=float, default=3.0, help="median tool-call seconds between turns")
    g.add_argument("--stagger", type=float, default=1.0)
    g.add_argument("--max-tokens", type=int, default=4096)
    g.add_argument("--fixed", action="store_true", help="fixed sizes and gaps instead of random ones")
    g = ap.add_argument_group("returning workload (returning.py; also --prompt-tokens, --tool-tokens, --gen-tokens, "
                              "--turns-max)")
    g.add_argument("--sessions", type=int, default=12)
    g.add_argument("--clients", type=int, default=4)
    g = ap.add_argument_group("generic workload")
    g.add_argument("--requests", type=int, default=200)
    g.add_argument("--rate", type=float, default=0.5, help="requests per second")
    g.add_argument("--prompt-tokens", type=int, default=4000, help="median prompt")
    return ap.parse_args(argv)


def selftest():
    base = ["--workflows", "2", "--subagents", "3", "--turns-min", "3", "--turns-max", "6", "--seed", "3"]
    a = parse(base + ["--pool", "2000000"])
    res = {p: run_one(a, p, check=True) for p in POLICIES}
    # newest orders prompt chunks by admission, so only the slot-order policies must match exactly
    spans = {round(r["makespan_s"], 9) for p, r in res.items() if p != "newest"}
    assert len(spans) == 1, f"without pressure every policy must behave the same: {spans}"
    assert all(r["victims"] == r["drops"] == r["failed"] == r["trimmed"] == 0 for r in res.values())

    a = parse(base + ["--pool", "24000", "--cache-ram", "-1", "--subagents", "6", "--turns-min", "8", "--turns-max", "12"])
    res = {p: run_one(a, p, check=True) for p in POLICIES}
    for p in ("largest", "bestfit", "tiered", "newest", "admit"):
        assert res[p]["failed"] == 0, (p, res[p]["failed"])
    assert res["fail"]["failed"] > 0 and res["largest"]["victims"] > 0

    # victim choice on a hand-made state: three generating requests, a deficit of 100 cells
    a = parse(["--pool", "5000"])
    sim = Sim(a, Largest(a), Generic(a, random.Random(0)))
    b = Batch()
    for i, (cells, runs) in enumerate([(3000, 2), (1000, 900), (500, 1)]):
        r = Req(i, 0.0, 100, 10, 10)
        r.state, r.slot = "gen", sim.slots[i]
        sim.slots[i].req, sim.slots[i].cells, sim.slots[i].runs = r, cells, runs
        sim.used += cells
        b.items.append([r, 1, "gen"])
    assert [r.slot.cells for r in Largest(a).pick(sim, b, 100)] == [3000]
    assert [r.slot.cells for r in Bestfit(a).pick(sim, b, 100)] == [500]
    print("selftest ok")


def main():
    a = parse()
    if a.selftest:
        return selftest()
    rows = []
    for name in a.policies.split(","):
        rows.append(run_one(a, name))
        print(f"{name}: done", file=sys.stderr)
    print(table(rows))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(args=vars(a), results=rows), f, indent=1)


if __name__ == "__main__":
    main()
