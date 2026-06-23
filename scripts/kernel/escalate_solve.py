#!/usr/bin/env python3
"""kernel/escalate_solve.py — an ESCALATION LADDER for solving a coding task without cheating (every tier is
gated by the hold-out methodology: a candidate must pass thousands of NOVEL instances vs an independent oracle,
so memorizing the visible cases never wins).

  Tier 1  greedy (T=0)                      single deterministic attempt.
  Tier 2  3 diverse attempts @ temp 0.8     (T=0 retries are identical -> raise temperature for real diversity).
  Tier 3  each of the 3 REPAIRS             with feedback from its own held-out failure.
  Tier 4  RESEARCH (if web tools enabled)   web_search + fetch_page the problem, build from the references.
  Tier 5  ABSTAIN                           "I'm sorry, I don't know how to do that."

Every gate-proven solution is PERSISTED (learned_primitives.json) + registered, so good helpers are reused.
Shares SelfExtender.gate_candidate (sample-test -> hold-out gate -> sandboxed-check -> 2nd-seed re-gate ->
placebo) so the ladder judges exactly like the pipeline."""
from __future__ import annotations
import os, re, json, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts/ -> `kernel` is a package

from kernel.self_extend import Gap, SelfExtender
from kernel.task_graph import Scheduler
from kernel.code_fixups import auto_import
from kernel.web_tools import web_available, web_search, fetch_page
from kernel.k_policy import KPolicy, KSignals
from kernel.k_cost import CostModel, charge_budget

_PERSIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_primitives.json")


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n([\s\S]*?)```", text or "", re.I)
    if m:
        return m.group(1)
    i = (text or "").find("def ")
    return text[i:] if i >= 0 else (text or "")


def load_learned(path: str = _PERSIST) -> dict:
    """The persisted registry of gate-proven helpers {entry: {code, gated_n, tier}} the kernel can reuse.
    Type-checked: a malformed/corrupt file (non-dict top level, partial write) returns {} rather than crashing."""
    if os.path.exists(path):
        try:
            r = json.load(open(path))
            return r if isinstance(r, dict) else {}
        except Exception:
            return {}
    return {}


class EscalatingSolver:
    def __init__(self, model, *, web=None, n=2000, placebo_n=300, persist_path=_PERSIST, maxn=1600,
                 k_policy=None, importance=0.5, trap_budget=None, cost_source=None, fixed_k=3):
        self.model = model
        self.n = n
        self.maxn = maxn
        self.ext = SelfExtender(Scheduler(), synthesize=None, placebo_n=placebo_n)   # gate_candidate + _register
        self.web = web_available() if web is None else web
        self.persist_path = persist_path
        # M5 rung-5: scheduler-owned dynamic-k for the Tier-2 width. k_policy=None -> the original fixed_k (3),
        # so existing behavior is byte-unchanged unless a policy is supplied.
        self.k_policy = k_policy
        self.importance = importance
        self.trap_budget = trap_budget          # a remaining budget (float) -> charge_budget caps k; None == untracked
        self.cost_source = cost_source          # callable()->cost in [0,1]; None -> live sysinfo (the real sensor)
        self.fixed_k = fixed_k

    def _cost(self) -> float:
        """The k-policy's cost input: the kernel's own load. Live sysinfo by default; injectable for tests/demos."""
        if self.cost_source is not None:
            return float(self.cost_source())
        try:
            from kernel.tools_builtin import sysinfo_handler
            return CostModel().cost_from_sysinfo(sysinfo_handler(None, {}))["cost"]
        except Exception:
            return 0.0

    @staticmethod
    def _t1_severity(r) -> float:
        """Independent (validate-the-branch) uncertainty from Tier-1's hold-out gate: the fraction of held-out
        instances T1 got WRONG (gate ran), else 1.0 (failed before the gate = maximally uncertain). Ground-truth-
        measured, not the model's self-report -> a trustworthy lever for KPolicy (unlike the rung-3 raw entropy)."""
        g = r.get("gate")
        if g and g.get("n"):
            return max(0.0, min(1.0, 1.0 - g.get("passed", 0) / g["n"]))
        return 1.0

    def _t2_width(self, r, on):
        """Scheduler-owned k for the Tier-2 diverse stage — replaces the fixed 3. Returns (k, decision_log_entry).
        Uncertainty = T1 gate severity (independent); cost = sysinfo; importance = task. Budget caps k."""
        if self.k_policy is None:
            return self.fixed_k, None
        sev = self._t1_severity(r)
        cost = self._cost()
        dec = self.k_policy.decide(KSignals(disagreement=sev, cost=cost, importance=self.importance))
        charge = charge_budget(dec, self.trap_budget)
        k = max(1, charge.admitted_k)            # >=1: having decided to escalate, try at least one diverse attempt
        entry = {"tier": "t2-kpolicy", "k": k, "severity": round(sev, 3), "cost": round(cost, 3),
                 "used": dec.used_signal, "reason": dec.reason, "budget_note": charge.note}
        on(f"  [k-policy] T2 width k={k} (severity={sev:.2f} cost={cost:.2f} | {dec.used_signal}: {dec.reason})")
        return k, entry

    # ── synthesizers ──
    def _prompt(self, gap, feedback=None, research=None):
        p = gap.description
        if research:
            p += ("\n\nReference material researched from the web (use it to write a CORRECT GENERAL solution; do "
                  "NOT copy verbatim, implement the algorithm):\n" + research[:3500])
        if feedback:
            if feedback.get("fails"):                          # a VISIBLE example failed
                f = feedback["fails"][0]
                args = gap.sample_cases[f["case"]]["args"] if f.get("case") is not None else "?"
                p += (f"\n\nYour previous function was WRONG on this example: input {args!r} -> your output "
                      f"{f.get('got')!r}, expected {f.get('expected')!r}. Fix it and handle EVERY case consistently.")
            else:                                              # a HELD-OUT (novel, unseen) instance failed
                p += (f"\n\nYour previous attempt was INCORRECT on a held-out input you were NOT shown: "
                      f"{feedback.get('first_fail')}. Write a GENERAL algorithm correct for ALL inputs of this kind.")
        return p + "\n\nReturn ONLY the function definition in a ```python block."

    def _gen(self, gap, *, feedback=None, research=None, temp=0.0, seed=0):
        p = self._prompt(gap, feedback, research)
        if temp and temp > 0:
            out, _ = self.model.gen_sample(p, temp=temp, seed=seed, maxn=self.maxn)
        else:
            out, _ = self.model.gen_fast(p, maxn=self.maxn)
        return auto_import(_extract_code(out))

    def _research(self, gap) -> str:
        """web_search the problem + fetch_page the top hits -> reference text the model builds from."""
        q = gap.entry.replace("_", " ") + " algorithm specification python"
        s = web_search(q, 4)
        urls = re.findall(r"https?://[^\s\"']+", str(s.get("output", "")))
        refs = []
        for u in urls[:2]:
            f = fetch_page(u.rstrip(".,)"))
            if f.get("ok") and f.get("output"):
                refs.append(f"[{u}]\n" + str(f["output"])[:2500])
            if len(refs) >= 2:
                break
        return "\n\n".join(refs)

    # ── the ladder ──
    def solve(self, gap, *, on=print) -> dict:
        log = []

        def step(label, code):
            r = self.ext.gate_candidate(gap, code)
            log.append({"tier": label, "result": r["reason"], "gate": r.get("gate"), "regate": r.get("regate")})
            on(f"   {label:12} -> {r['reason']}"
               + (f"  (gate {r['gate']['passed']}/{r['gate']['n']})" if r.get("gate") else ""))
            return r

        on("  [tier 1] greedy T=0")
        r = step("t1-greedy", self._gen(gap))
        if r["passed"]:
            return self._win(gap, r["code"], 1, log)
        if r["reason"] in ("unsandboxed", "placebo_unsound"):
            return {"solved": False, "entry": gap.entry, "gap_unsound": r["reason"], "log": log}

        k2, kdec = self._t2_width(r, on)                # scheduler-owned k (default 3 when no policy)
        if kdec is not None:
            log.append(kdec)
        on(f"  [tier 2] {k2} diverse @ temp 0.8")
        diverse = []
        for i in range(k2):
            r = step(f"t2-div{i}", self._gen(gap, temp=0.8, seed=i))
            diverse.append(r)
            if r["passed"]:
                return self._win(gap, r["code"], 2, log)

        on("  [tier 3] each of the 3 repairs")
        for i, dr in enumerate(diverse):
            r = step(f"t3-repair{i}", self._gen(gap, feedback=dr.get("feedback"), temp=0.8, seed=300 + i))
            if r["passed"]:
                return self._win(gap, r["code"], 3, log)

        if self.web:
            on("  [tier 4] research (web_search + fetch_page)")
            research = self._research(gap)
            on(f"   researched {len(research)} chars of references")
            for seed in range(2):
                r = step(f"t4-research{seed}", self._gen(gap, research=research, temp=(0.6 if seed else 0.0), seed=seed))
                if r["passed"]:
                    return self._win(gap, r["code"], 4, log)
        else:
            on("  [tier 4] skipped (web tools unavailable)")

        on("  [tier 5] ABSTAIN")
        return {"solved": False, "entry": gap.entry, "tier": None, "web_used": self.web, "log": log,
                "abstain": "I'm sorry, I don't know how to do that."}

    def _win(self, gap, code, tier, log) -> dict:
        self._persist(gap, code, tier)
        return {"solved": True, "tier": tier, "entry": gap.entry, "code": code, "log": log}

    def _persist(self, gap, code, tier):
        reg = load_learned(self.persist_path)
        reg[gap.entry] = {"code": code, "gated_n": gap.n, "tier": tier}
        json.dump(reg, open(self.persist_path, "w"), indent=2)
        try:
            self.ext._register(gap, code)                      # also callable in this process
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="both", choices=["w47", "c156", "both"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--no-web", action="store_true")
    args = ap.parse_args()
    from kernel.solve_failed_challenges import TASKS           # reuse the trusted oracles/generators
    from north_adapter import NorthAdapter

    todo = ["w47", "c156"] if args.task == "both" else [args.task]
    print("[load] north ...", flush=True)
    model = NorthAdapter()
    solver = EscalatingSolver(model, web=(not args.no_web), n=args.n)
    print(f"[web] research tier {'ENABLED' if solver.web else 'disabled'}", flush=True)

    for tid in todo:
        s = TASKS[tid]
        gap = Gap(entry=s["entry"], description=s["prompt"], sample_cases=s["sample_cases"],
                  generate_src=s["generate_src"], oracle_src=s["oracle_src"], n=args.n, seed=0)
        print(f"\n===== escalation solve: {tid} ({s['entry']}) =====", flush=True)
        res = solver.solve(gap)
        if res.get("solved"):
            print(f"  ✅ SOLVED at tier {res['tier']} (hold-out-proven, general) + PERSISTED to learned_primitives.json", flush=True)
        else:
            print(f"  ❌ {res.get('abstain') or res.get('gap_unsound')}", flush=True)
    print(f"\n[persisted helpers] {sorted(load_learned().keys())}", flush=True)


if __name__ == "__main__":
    main()
