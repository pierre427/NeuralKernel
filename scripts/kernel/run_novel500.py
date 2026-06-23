#!/usr/bin/env python3
"""run_novel500.py — run the novel 500-challenge stress set THROUGH THE KERNEL on North (MLX), with web access.

Each challenge is answered by North in a bounded ReAct-style tool loop: the model may research with web_search /
fetch_page, which are routed THROUGH THE SCHEDULER (admit-gated op:web_search / op:fetch_page) so the kernel's
capability gates + telemetry are genuinely exercised. Per-challenge we record the full transcript, every tool
call, timing, and a telemetry event stream; results are appended to a JSONL so the run is RESUMABLE (re-running
skips finished ids and picks up at the next one — i.e. resume-from-the-failed-one).

Usage:
  python kernel/run_novel500.py --challenges kernel/novel500/challenges_500.json \
       [--out kernel/novel500/results.jsonl] [--telemetry reports/telemetry/novel500] \
       [--start 0] [--limit 500] [--max-tools 3] [--no-web] [--only <id>]
"""
from __future__ import annotations
import os, sys, re, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                       # scripts/ -> `kernel` package + north_adapter

from kernel.task_graph import Scheduler, Task
from kernel.web_tools import register_web_tools, web_available
from kernel.memory_store import MemoryStore, register_memory_tool
from kernel.web_distiller import distill_web_output
from kernel.m5_budget import M5Budget
from kernel.tools_builtin import sysinfo_handler

_ACTION = re.compile(r"^[ \t]*(SEARCH|FETCH|ANSWER)[ \t]*:[ \t]*(.*)$", re.I | re.M)
# North frames every answer with structural tokens (<|START_TEXT|>, <|CHATBOT_TOKEN|>, ...). The decoder leaks the
# leading <|START_TEXT|> into the text, which would push the action keyword off the line start and defeat _ACTION
# (no SEARCH/FETCH/ANSWER ever recognized -> web never fires, answer never parsed). Strip all such framing tokens.
_FRAMING = re.compile(r"<\|[A-Z_]+\|>")
# Small models often PARROT the preamble back. Its template lines ("SEARCH: <query> — ...", "ANSWER: <answer> — ...")
# match _ACTION, so a naive parser fires a garbage web search on "<query> ..." or, worse, returns the echoed
# "<answer> — give your FINAL answer" AS the final answer. The "<query>/<url>/<answer>" placeholder is the tell.
_PLACEHOLDER = re.compile(r"^<(query|url|answer)>", re.I)


def _clean(text: str) -> str:
    return _FRAMING.sub("", text or "").strip()
_TOOL_PREAMBLE = (
    "You are a careful assistant with optional web access. Think, then act with ONE of these on its own line:\n"
    "  SEARCH: <query>     — search the web (use only if you need facts you are unsure of)\n"
    "  FETCH: <url>        — fetch a page found via SEARCH\n"
    "  ANSWER: <answer>    — give your FINAL answer (do this as soon as you can answer well)\n"
    "Use at most {k} SEARCH/FETCH steps. If you do not need the web, go straight to ANSWER.\n\n")


def _first_action(text: str):
    t = _clean(text)
    for m in _ACTION.finditer(t):                              # first REAL action; skip echoed preamble templates
        line = m.group(2).strip()
        if _PLACEHOLDER.match(line):                           # e.g. "SEARCH: <query>  — search the web ..." (an echo)
            continue
        act = m.group(1).upper()
        if act == "ANSWER":                                   # ANSWER is the FINAL answer -> everything after the
            return act, t[m.start(2):].strip()                # marker (multi-line: code, essays, ...)
        return act, line                                      # SEARCH/FETCH payload is a single-line query/URL
    return None, None


class KernelWeb:
    """Routes web_search / fetch_page through the scheduler's admit gate; records telemetry events. When a memory
    path is given it also registers op:memory on the SAME scheduler, so full tool outputs spill to the kernel's
    persistent store (capability-gated, addressable) and can be retrieved locally."""
    def __init__(self, grant=True, mem_path=None):
        self.sched = Scheduler()
        register_web_tools(self.sched, grant=grant)
        self.events = []
        self.mem = None
        self._seq = 0
        if mem_path:
            self.mem = MemoryStore(mem_path)
            register_memory_tool(self.sched, self.mem, grant=grant)   # op:memory on the same admit gate

    def store_full(self, cid, value):
        """Spill a full tool output to the kernel memory store THROUGH THE SCHEDULER (op:memory put). Returns the
        slot ADDRESS (the memory key) so the asking task can retrieve the full content later, or None if no store."""
        if not self.mem:
            return None
        self._seq += 1
        addr = f"web:{cid}:{self._seq}"
        t0 = time.time()
        ev = {"kind": "mem_put", "cid": cid, "addr": addr, "t": t0}
        try:
            task = Task(task_id=f"{cid}:memory:{self._seq}", kind="memory",
                        validator={"fn": "memory_ok", "level": "SCHEMA"}, capabilities=["op:memory"])
            self.sched.admit(task)
            res, ok, _ = self.sched.executors["memory"](self.sched, task,
                                                        {"action": "put", "key": addr, "value": value, "kind": "web_raw"})
            stored = bool(ok and (res or {}).get("ok"))
            ev.update({"ok": stored, "bytes": (res or {}).get("bytes"), "ms": int((time.time() - t0) * 1000)})
            self.events.append(ev)
            return addr if stored else None
        except Exception as e:
            ev.update({"ok": False, "error": f"{type(e).__name__}: {e}", "ms": int((time.time() - t0) * 1000)})
            self.events.append(ev)
            return None

    def mem_get(self, addr):
        """Retrieve full content from the memory store THROUGH THE SCHEDULER (op:memory get). Local retrieval."""
        if not self.mem:
            return None
        task = Task(task_id=f"get:{addr}:{len(self.events)}", kind="memory",
                    validator={"fn": "memory_ok", "level": "SCHEMA"}, capabilities=["op:memory"])
        self.sched.admit(task)
        res, _, _ = self.sched.executors["memory"](self.sched, task, {"action": "get", "key": addr})
        return (res or {}).get("value")

    def call(self, tool, args, cid):
        t0 = time.time()
        ev = {"kind": "tool_call", "cid": cid, "tool": tool, "args": args, "t": t0}
        try:
            task = Task(task_id=f"{cid}:{tool}:{len(self.events)}", kind=tool,
                        validator={"fn": f"{tool}_ok", "level": "SCHEMA"}, capabilities=[f"op:{tool}"])
            self.sched.admit(task)                              # 3-gate admission (op exists / cap / validator)
            res, ok, _ = self.sched.executors[tool](self.sched, task, args)
            ev.update({"ok": bool(ok), "ms": int((time.time() - t0) * 1000),
                       "out_len": len(str((res or {}).get("output", "")))})
            self.events.append(ev)
            return res or {}
        except Exception as e:
            ev.update({"ok": False, "error": f"{type(e).__name__}: {e}", "ms": int((time.time() - t0) * 1000)})
            self.events.append(ev)
            return {"ok": False, "error": str(e)}


def _query_from_prompt(prompt: str) -> str:
    s = re.sub(r"\s+", " ", prompt or "").strip()
    m = re.search(r"^(.*?[?.])(\s|$)", s)                      # first question/sentence -> cleanest search query
    return (m.group(1) if m else s)[:200]


def _observe(web, model, tool, args, need, cid, task_prompt, distill):
    """Run ONE kernel-routed web call and turn its result into an OBSERVATION for the asking task. With distill on:
    spill the FULL output to the kernel memory store (op:memory) and replace the raw blob in-context with a clean-
    context DISTILLER AGENT's task-relevant summary + the memory slot address. Returns (observation_text, step)."""
    r = web.call(tool, args, cid)
    raw = str(r.get("output", r.get("error", "")))
    if distill and getattr(web, "mem", None):
        addr = web.store_full(cid, raw)                       # spill full content -> slot address (through scheduler)
        summary = distill_web_output(model, raw, need, task_prompt)   # the distiller agent (clean context)
        web.events.append({"kind": "distill", "cid": cid, "tool": tool, "addr": addr,
                           "in_len": len(raw), "out_len": len(summary)})
        loc = f"memory://{addr}" if addr else "(spill failed)"
        return (f"OBSERVATION (distilled; full content at {loc}):\n{summary}",
                {"gen": f"[distill {tool} -> {loc}]\n{summary[:1500]}", "ntok": 0, "action": "DISTILL",
                 "mem_addr": addr, "full_bytes": len(raw)})
    obs = raw[:1500]                                          # legacy path: truncated raw blob into context
    return f"OBSERVATION: {obs}", {"gen": f"[obs {tool}] {obs[:1500]}", "ntok": 0, "action": "OBS"}


def run_one(model, web, ch, max_tools=3, maxn=1200, force_web=False, distill=False):
    cid = ch["id"]
    web and web.events.clear()
    transcript = _TOOL_PREAMBLE.format(k=max_tools) + "TASK:\n" + ch["prompt"] + "\n\n"
    steps, tool_calls = [], 0
    answer = None
    t0 = time.time()
    if force_web and web:                                      # web_required: the model is over-confident and
        q = _query_from_prompt(ch["prompt"])                  # answers cold -> inject ONE mandatory kernel-routed
        tool_calls += 1                                        # search so it answers on fresh data
        steps.append({"gen": f"[forced] SEARCH: {q}", "ntok": 0, "action": "SEARCH", "forced": True})
        obs_text, ostep = _observe(web, model, "web_search", {"query": q, "max_results": 4}, q, cid,
                                   ch["prompt"], distill)
        steps.append(ostep)
        transcript += (f"SEARCH: {q}\n{obs_text}\n\nThe OBSERVATION above is current and authoritative; base your "
                       f"answer on it. SEARCH again only if it is insufficient, else give ANSWER.\n\n")
    for _ in range(max_tools + 1):
        out, ntok = model.gen_fast(transcript, maxn=maxn)
        out = _clean(out)                                      # strip North framing tokens before parse/record
        act, payload = _first_action(out)
        steps.append({"gen": out[:4000], "ntok": ntok, "action": act})
        if act == "ANSWER" or act is None:
            answer = payload if act == "ANSWER" else out.strip()
            break
        if act in ("SEARCH", "FETCH") and web and tool_calls < max_tools:
            tool_calls += 1
            tool = "web_search" if act == "SEARCH" else "fetch_page"
            args = {"query": payload, "max_results": 4} if act == "SEARCH" else {"url": payload}
            obs_text, ostep = _observe(web, model, tool, args, payload, cid, ch["prompt"], distill)
            steps.append(ostep)
            transcript += f"{act}: {payload}\n{obs_text}\n\n"
        else:                                                  # ran out of tool budget -> force a final answer
            transcript += "You have used your tool budget. Now give ANSWER: <final answer>.\n\n"
    if answer is None:                                         # never emitted ANSWER -> last generation is it
        answer = (steps[-1]["gen"] if steps else "").strip()
    return {
        "id": cid, "domain": ch.get("domain"), "subtype": ch.get("subtype"),
        "web_required": ch.get("web_required", False), "tool_calls": tool_calls,
        "answer": answer, "steps": steps, "tool_events": list(web.events) if web else [],
        "elapsed_s": round(time.time() - t0, 1), "ts": t0, "maxn": maxn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", default=os.path.join(HERE, "novel500", "challenges_500.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "novel500", "results.jsonl"))
    ap.add_argument("--telemetry", default=os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                                        "reports", "telemetry", "novel500"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--max-tools", type=int, default=3)
    ap.add_argument("--maxn", type=int, default=1200, help="max new tokens per generation (per-step token budget)")
    ap.add_argument("--force-web-required", action="store_true",
                    help="inject a mandatory first kernel-routed SEARCH on challenges flagged web_required")
    ap.add_argument("--distill", action="store_true",
                    help="route web results through the distiller agent: spill full output to op:memory and inject "
                         "only the task-relevant summary + slot address into the task context")
    ap.add_argument("--mem-path", default=os.path.join(HERE, "novel500", "kernel_mem.sqlite"),
                    help="path to the kernel memory store (op:memory) used by --distill to spill full web content")
    ap.add_argument("--m5", action="store_true",
                    help="M5 scheduler-owned per-task token budget: set maxn per challenge from length_class x "
                         "difficulty, cost-braked by live sysinfo, hard-capped (anti-runaway). Overrides --maxn.")
    ap.add_argument("--no-web", action="store_true")
    ap.add_argument("--only", default=None, help="run a single challenge id")
    args = ap.parse_args()

    challenges = json.load(open(args.challenges))
    if args.only:
        challenges = [c for c in challenges if c["id"] == args.only]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.telemetry, exist_ok=True)
    tele = open(os.path.join(args.telemetry, "run.jsonl"), "a")

    done = set()
    if os.path.exists(args.out):                               # RESUME: skip already-finished ids
        for line in open(args.out):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass

    use_web = (not args.no_web) and web_available()
    print(f"[novel500] {len(challenges)} challenges | web={'ON' if use_web else 'OFF'} | "
          f"distill={'ON' if (use_web and args.distill) else 'OFF'} | already done={len(done)}", flush=True)
    from north_adapter import NorthAdapter
    print("[load] north ...", flush=True)
    model = NorthAdapter()
    web = KernelWeb(grant=True, mem_path=(args.mem_path if args.distill else None)) if use_web else None
    m5 = M5Budget() if args.m5 else None
    if m5:
        print(f"[m5] scheduler-owned per-task token budget ON (length x difficulty, cost-braked, cap={m5.ceil})",
              flush=True)

    todo = [c for c in challenges[args.start:args.start + args.limit] if c["id"] not in done]
    out = open(args.out, "a")
    for i, ch in enumerate(todo):
        t0 = time.time()
        maxn, budget_audit = args.maxn, None
        if m5:                                                 # scheduler decides this task's budget from its signals
            budget_audit = m5.budget_for(ch, sysinfo_handler(getattr(web, "sched", None), {"facet": "all"}))
            maxn = budget_audit["maxn"]
        tele.write(json.dumps({"kind": "intake", "cid": ch["id"], "domain": ch.get("domain"),
                               "web_required": ch.get("web_required", False), "maxn": maxn, "t": t0}) + "\n"); tele.flush()
        try:
            res = run_one(model, web, ch, max_tools=args.max_tools, maxn=maxn,
                          force_web=(args.force_web_required and ch.get("web_required", False)),
                          distill=args.distill)
        except Exception as e:
            res = {"id": ch["id"], "domain": ch.get("domain"), "error": f"{type(e).__name__}: {e}",
                   "answer": None, "elapsed_s": round(time.time() - t0, 1)}
        if budget_audit:
            res["m5_budget"] = budget_audit
        out.write(json.dumps(res) + "\n"); out.flush()
        for ev in res.get("tool_events", []):
            tele.write(json.dumps(ev) + "\n")
        tele.write(json.dumps({"kind": "answer", "cid": ch["id"], "tool_calls": res.get("tool_calls", 0),
                               "ans_len": len(res.get("answer") or ""), "elapsed_s": res.get("elapsed_s"),
                               "error": res.get("error")}) + "\n"); tele.flush()
        print(f"  [{i+1}/{len(todo)}] {ch['id']:28} tools={res.get('tool_calls',0)} "
              f"{res.get('elapsed_s')}s ans={len(res.get('answer') or '')}c"
              + (f" ERR {res['error']}" if res.get('error') else ""), flush=True)
    out.close(); tele.close()
    print(f"[novel500] done: wrote {len(todo)} results to {args.out}", flush=True)


if __name__ == "__main__":
    main()
