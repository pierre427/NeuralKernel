#!/usr/bin/env python3
"""kernel/lexical_proposer.py — Step-4 LEXICAL PROPOSER: a scored top-k router over the routable capability registry.

The decomposer (and the self-extending plane) needs to answer "is there an ALREADY-TRUSTED capability for this code
span?" before generating. This proposer indexes two tiers and RANKS them by lexical match to a query:
  • PROVEN det primitives — kernel/../det_primitives.py (hold-out-gateable instruction set: c_gcd, count_byte, ...)
    + any runtime-synthesized primitives registered via register_proven() (self_extend). tier = "proven".
  • SYSCALL tools — kernel.tools.TOOL_CATALOG (datetime, web_search, memory, venv_exec, ...). tier = "checked".

It PROPOSES (scores), the caller DISPOSES (decides whether the top score clears a threshold). The score is an
UNCALIBRATED lexical signal, so the honest use is: route on a STRONG match only, and ideally validate-the-branch
(run the candidate on the span's examples) before committing — never trust a lexical hunch the way you'd trust a
hold-out gate. Dependency-free (difflib + token overlap; no embeddings)."""
from __future__ import annotations
import os, re, sys, inspect
from dataclasses import dataclass
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                       # scripts/ -> det_primitives + kernel package

ROUTE_THRESHOLD = 0.55                                          # min score to route to a registered capability

# runtime-synthesized proven primitives register here (self_extend hook); {name: (description, signature)}
PROVEN_CATALOG: dict[str, tuple] = {}


def register_proven(name: str, description: str = "", signature: str = ""):
    PROVEN_CATALOG[name] = (description or "", signature or "")


def _toks(s: str) -> set:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s or "")         # split camelCase (RollingStats -> rolling stats)
    return {w for w in re.findall(r"[a-z0-9]{2,}", s.lower())}


@dataclass
class Candidate:
    name: str
    tier: str            # "proven" | "checked"
    description: str
    signature: str


def _builtin_proven() -> list:
    """Index det_primitives' public functions (PROVEN instruction set) using docstring + real signature."""
    out = []
    try:
        import det_primitives as dp
    except Exception:
        return out
    for name, fn in inspect.getmembers(dp, inspect.isfunction):
        if name.startswith("_") or getattr(fn, "__module__", None) != "det_primitives":
            continue
        doc = (inspect.getdoc(fn) or "").splitlines()
        try:
            sig = f"{name}{inspect.signature(fn)}"
        except (ValueError, TypeError):
            sig = name
        out.append(Candidate(name, "proven", doc[0] if doc else "", sig))
    return out


def _registry() -> list:
    cands = _builtin_proven()
    for name, (desc, sig) in PROVEN_CATALOG.items():            # runtime-synthesized primitives
        cands.append(Candidate(name, "proven", desc, sig or name))
    try:                                                        # syscall tools (CHECKED tier)
        from kernel.tools import TOOL_CATALOG
    except ImportError:
        from tools import TOOL_CATALOG
    for name, spec in TOOL_CATALOG.items():
        cands.append(Candidate(name, "checked", getattr(spec, "description", ""), getattr(spec, "signature", "") or name))
    return cands


def _score(q_entry: str, q_text: str, c: Candidate) -> float:
    q = _toks(q_entry) | _toks(q_text)
    cn = _toks(c.name)
    call = cn | _toks(c.description) | _toks(c.signature)
    if not q or not call:
        return 0.0
    name_sim = SequenceMatcher(None, " ".join(sorted(_toks(q_entry) or q)), " ".join(sorted(cn))).ratio()
    overlap = len(q & call) / max(1, min(len(q), len(call)))    # overlap coefficient (good for short queries)
    boost = 0.2 if (q & cn) else 0.0                            # a query keyword IS a candidate name token
    return min(1.0, 0.4 * name_sim + 0.6 * overlap + boost)


def propose(entry: str, text: str = "", top_k: int = 5) -> list:
    """Return up to top_k {name, tier, score, signature, description} candidates, best first."""
    scored = [(c, _score(entry, text, c)) for c in _registry()]
    scored.sort(key=lambda t: -t[1])
    return [{"name": c.name, "tier": c.tier, "score": round(s, 3), "signature": c.signature, "description": c.description}
            for c, s in scored[:top_k] if s > 0]


def best_match(entry: str, text: str = "", threshold: float = ROUTE_THRESHOLD):
    """The DISPOSE side for the decomposer: the single best candidate IFF it clears the threshold, else None."""
    cands = propose(entry, text, top_k=1)
    return cands[0] if (cands and cands[0]["score"] >= threshold) else None


def governed_propose(entry, text="", *, policy, floor=ROUTE_THRESHOLD, cost_source=None, sysinfo=None,
                     budget=None, importance=0.5, top_n=8) -> dict:
    """M5 scheduler-owned-k disposal over the lexical proposer's scores — the LAST continuation-k seam.

    propose() RANKS (scores) the routable registry; this is the DISPOSE side, but unlike best_match (fixed
    threshold + top-1) it lets ONE KPolicy decide HOW MANY candidates to try (project_scheduler_owned_k: "the
    proposer emits SCORES; let one scheduler k-policy dispose how many to try — don't bake a fixed k into each
    proposer"). It unifies disposal with GAP-DETECT (nothing fits -> the self-extending plane should synthesize).

    DOCTRINE (thin/heavy verdict, project_grounding_result):
      * a CLEAR lexical winner -> k=1 (route to the best, fast — a heavy fan-out HURTS when one answer is obvious);
      * a CLOSE race -> WIDEN k (try the top-k), because the lexical score is UNCALIBRATED: never trust a lexical
        hunch the way you'd trust a hold-out gate. The widened set is meant to be filtered by validate-the-branch
        (validate_proven / the hold-out gate) — the caller runs the candidates and keeps the one that actually
        works, so a close race buys CANDIDATES to verify, not a blind multi-route;
      * NOTHING clears the floor -> action 'synthesize' (Step-3 trigger: no trusted capability fits).
    The widening signal is the proposer's OWN top-1/top-2 score MARGIN, an INDEPENDENT signal (not a model
    self-report) -> fed as KSignals.disagreement (the trusted lever, immune to the calibration trap). cost brakes
    the discretionary widening (stay thin when busy); budget caps k (anti-runaway-via-k at the budget level).

    Args:
      entry, text   the query (function/class name + instruction), as for propose().
      policy        a KPolicy. The caller owns its shape (k_min/k_max/cost_brake). decide() disposes k.
      floor         min top score to route at all; below it -> 'synthesize'. Defaults to ROUTE_THRESHOLD (so the
                    gap-detect boundary matches best_match's route/no-route boundary).
      cost_source   callable()->cost in [0,1] (injectable; e.g. tests/demos). Overrides sysinfo when present.
      sysinfo       a sysinfo dict for CostModel().cost_from_sysinfo when no cost_source is given.
      budget        remaining trap budget (float) for charge_budget; None == untracked (admit the policy's k).
      importance    task importance -> scales the discretionary widen ceiling in KPolicy.
      top_n         how many candidates to score/consider (the widen pool ceiling).

    Returns one of:
      {'action': 'route', 'candidates': [top-k dicts], 'k': k, 'score_uncertainty': u, 'audit': {...}}
      {'action': 'synthesize', 'candidates': [], 'k': 0, 'reason': 'gap-detect: ...', 'audit': {...}}
    The route candidates are propose()'s dicts (best first); the caller still applies validate-the-branch.
    """
    from kernel.k_policy import KSignals
    from kernel.k_cost import charge_budget

    # 1. RANK: the proposer scores the routable registry.
    cands = propose(entry, text, top_k=top_n)

    # 2. GAP-DETECT: nothing scored, or even the BEST candidate is below the route floor -> no trusted capability
    #    fits. Hand off to the self-extending plane to SYNTHESIZE one (Step-3 trigger). We do NOT widen-and-pray
    #    over sub-floor junk; an uncalibrated lexical near-miss is not a route.
    top_score = cands[0]["score"] if cands else 0.0
    if not cands or top_score < floor:
        audit = {
            "used_signal": "gap_detect",
            "reason": f"gap-detect: top lexical score {top_score:.3f} < floor {floor:.3f} -> synthesize",
            "top_score": round(top_score, 3),
            "floor": round(float(floor), 3),
            "n_considered": len(cands),
        }
        return {"action": "synthesize", "candidates": [], "k": 0,
                "reason": "gap-detect: nothing clears the floor", "audit": audit}

    # 3. ROUTE: the floor is cleared. Decide HOW MANY to try from the score MARGIN (an INDEPENDENT signal).
    #    margin = (s0 - s1)/s0 over the top two scores; score_uncertainty = clamp01(1 - margin):
    #    a clear winner (big margin) -> low uncertainty -> thin (k=1); a close race (tiny margin) -> high
    #    uncertainty -> widen. Single candidate -> margin 1.0 -> uncertainty 0.0 (a lone match is a clear winner).
    s0 = top_score
    s1 = cands[1]["score"] if len(cands) > 1 else 0.0
    margin = (s0 - s1) / max(s0, 1e-9)
    score_uncertainty = min(1.0, max(0.0, 1.0 - margin))

    # cost (kernel proprioception): explicit source > sysinfo snapshot > none (0). Mirrors governed_verify/fact_loop.
    if cost_source is not None:
        cost = cost_source()
    elif sysinfo is not None:
        from kernel.k_cost import CostModel
        cost = CostModel().cost_from_sysinfo(sysinfo)["cost"]
    else:
        cost = 0.0
    try:
        cost = min(1.0, max(0.0, float(cost)))
    except (TypeError, ValueError):
        cost = 0.0

    # The margin is the proposer's OWN disagreement (not a model self-report) -> feed as the trusted disagreement
    # lever, immune to the calibration trap. risk stays 0 here (validate-the-branch is the caller's safety net).
    dec = policy.decide(KSignals(disagreement=score_uncertainty, cost=cost, importance=importance))
    charge = charge_budget(dec, budget, unit_cost=policy.unit_cost)

    # We cleared the floor, so we always route at least the best (k>=1); cap k to what candidates exist.
    k = max(1, min(charge.admitted_k, len(cands)))

    audit = {
        "used_signal": dec.used_signal,
        "reason": dec.reason,
        "cost": round(cost, 3),
        "score_uncertainty": round(score_uncertainty, 3),
        "margin": round(margin, 3),
        "top_score": round(s0, 3),
        "runner_up_score": round(s1, 3),
        "disposed_k": dec.k,
        "explore_k": dec.explore_k,
        "budget_note": charge.note,
        "budget_remaining": charge.remaining,
        "requested_k": charge.requested_k,
        "n_considered": len(cands),
    }
    return {"action": "route", "candidates": cands[:k], "k": k,
            "score_uncertainty": round(score_uncertainty, 3), "audit": audit}


def validate_proven(name: str, cases: list) -> bool:
    """VALIDATE-THE-BRANCH: actually RUN a proposed builtin PROVEN primitive on the span's example cases before
    reusing it. A proven det primitive is trusted + pure, so calling it in-process is safe. Returns True iff it runs
    and matches EVERY case — so a lexical false-positive (e.g. count_leading_zeros proposed for 'count letters',
    which TypeErrors on string args) is REJECTED rather than blindly reused. Turns an uncalibrated lexical proposal
    into a verified route. (Runtime-synthesized primitives would validate via call_in_venv; this is the builtin path.)"""
    if not cases:
        return False
    try:
        import det_primitives as dp
    except Exception:
        return False
    fn = getattr(dp, name, None)
    if not callable(fn):
        return False
    for c in cases:
        args = c.get("args", [c["input"]] if "input" in c else [])
        try:
            got = fn(*args)
        except Exception:
            return False
        if got != c.get("expected"):
            return False
    return True


__all__ = ["propose", "best_match", "governed_propose", "validate_proven", "register_proven",
           "PROVEN_CATALOG", "ROUTE_THRESHOLD", "Candidate"]
