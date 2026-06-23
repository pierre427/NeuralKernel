#!/usr/bin/env python3
"""M5 rung-1: KPolicy — the scheduler-owned DYNAMIC-k policy (the design's "fuse the policy, not a constant k").

The branching factor k (how many CONTINUATIONS the scheduler executes at a decision point — diverse drafts,
tool candidates, verifier passes, sub-agents, memories, synthesize attempts) is the SCHEDULER'S decision, not a
constant baked into each proposer. The model / MoE router / syscall head PROPOSES route scores; the scheduler
DISPOSES k. This is "learned-proposes / kernel-disposes" applied to the BRANCHING FACTOR — same architecture
spine (the disposal seam = Scheduler.admit / cap gate), new axis. See memory project_scheduler_owned_k.

THREE top-k SURFACES, with a trust boundary (kept distinct on purpose):
  * Output top-k (decoding) — what the model SAYS. NOT our lever.
  * MoE expert top-k — North cohere2_moe = 128 experts, top-8. INTERNAL. Its gate-softmax ENTROPY is an
    already-computed, ~free uncertainty signal. We OBSERVE it (entropy-IN); we never DRIVE it (that is the
    Mode-B residual/router intervention: research-grade, brittle, 129th-expert hazard).
  * Continuation top-k — routes/tools/agents/verifiers/memories/synthesize. The scheduler's OWN decision space,
    OUTSIDE the weights. We OWN it: model proposes scores, scheduler disposes k. k-OUT. Safe + high-value.
  => RULE: OBSERVE the model's internal top-k (MoE entropy as a SIGNAL); OWN the continuation top-k (the DECISION).

This module is the formal mechanism for the thin/heavy verdict (project_grounding_result, project_kernel_intent_
reconciliation): a heavy protocol HURTS a capable model; thin det-grounding wins. So: be THIN by default
(k≈k_min), and SPEND (widen k -> verify/branch/tool/synthesize) ONLY where HIGH *calibrated* uncertainty earns it,
and ONLY when cost/latency allow it.

THE CALIBRATION TRAP (must-honor): entropy == "uncertainty" ONLY if the route scores are CALIBRATED. Our own
evidence shows model self-confidence MIScalibrates (gpt-oss conf=1.0 on a FAILED grade; North reasoning-runaways).
So the policy MUST NOT widen k on a raw argmax-softmax signal it has no reason to trust. The trap is honored three
ways, exactly as the design demands ("use INDEPENDENT uncertainty: expert disagreement, a held-out-calibrated route
head, or validate the branch"):
  (1) an INDEPENDENT disagreement signal (expert/ensemble/seed disagreement) is trusted-by-construction and MAY
      drive widening even when the raw signal is uncalibrated;
  (2) a raw uncertainty signal drives widening ONLY when flagged calibrated (a held-out-calibrated route head);
  (3) otherwise the policy REFUSES to widen speculatively — it stays thin and flags the decision so the caller
      relies on the commit trap to VALIDATE the single branch (validate-the-branch, not speculate-on-a-hunch).
"fail->pass is not proof" applied to routing: don't branch on a hunch you can't trust.

SAFETY (anti-runaway-via-k, and the confidently-wrong guard): k is hard-clamped to [k_min, k_max] (the model
cannot enlarge its own branching budget), and RISK floors a MANDATORY verification count that COST cannot brake
away — a miscalibrated-confident model cannot talk the scheduler out of verifying a risky branch. Cost brakes only
the DISCRETIONARY (uncertainty-driven) exploration; the risk-mandated verification is safety, not discretion.

Pure Python (math only, no MLX) so it stays importable/testable without a model. The MoE-entropy producer is
rung-2 (an inert on-North tap); `normalized_entropy` here is the shared signal-normalisation both rungs use.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


def normalized_entropy(probs, eps: float = 1e-12) -> float:
    """Shannon entropy of a discrete distribution, normalised to [0,1] by dividing by log(n).

    This is the single definition of the MoE-gate uncertainty signal (rung-2's tap feeds raw gate probs here).
    0.0 == a one-hot gate (one expert owns the token: maximally confident routing); 1.0 == uniform over all n
    experts (maximally uncertain routing). Robust to unnormalised inputs (renormalises) and to n<=1 (-> 0.0).
    """
    p = [float(x) for x in probs if float(x) > 0.0]
    n = len(probs) if hasattr(probs, "__len__") else len(p)
    if n <= 1 or not p:
        return 0.0
    s = sum(p)
    if s <= eps:
        return 0.0
    p = [x / s for x in p]
    h = -sum(x * math.log(x) for x in p)
    return max(0.0, min(1.0, h / math.log(n)))


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


@dataclass
class KSignals:
    """Inputs to the k-decision at one continuation point. All scalars normalised to [0,1].

    uncertainty           the primary uncertainty (e.g. normalized MoE gate entropy, or a route-head score gap, or
                          model self-confidence inverted). Drives widening ONLY if `uncertainty_calibrated`.
    uncertainty_calibrated whether `uncertainty` comes from a calibrated source (a held-out-calibrated route head).
                          FALSE for a raw gate-softmax / self-reported confidence — the calibration trap.
    disagreement          an INDEPENDENT uncertainty (expert/ensemble/seed disagreement). Trusted-by-construction:
                          MAY drive widening even when `uncertainty_calibrated` is False. None == not measured.
    cost                  normalised resource pressure (host load / queue depth / memory pressure; rung-4 sysinfo).
                          High -> brake DISCRETIONARY widening (stay thin when busy). Never brakes verification.
    risk                  task risk. Floors a MANDATORY verify count that cost cannot remove (safety, not discretion).
    importance            task importance. Scales the discretionary spend ceiling (important work earns more k).
    """
    uncertainty: float = 0.0
    uncertainty_calibrated: bool = False
    disagreement: Optional[float] = None
    cost: float = 0.0
    risk: float = 0.0
    importance: float = 0.5


@dataclass
class KDecision:
    """The disposed branching factor + its accounting (so the scheduler can charge the budget and audit it)."""
    k: int                     # continuations to execute (the disposed branch count)
    explore_k: int             # discretionary part (uncertainty-driven, cost-braked)
    verify_k: int              # mandatory part (risk-driven, NOT cost-braked) — the safety floor above k_min
    used_signal: str           # audit: which signal governed (thin_default | calibrated_uncertainty |
                               #        independent_disagreement | uncalibrated_held | risk_floor | cost_braked)
    reason: str                # human-readable justification (audit / proof-ledger)
    est_cost: float            # accounted cost (monotonic in k) for the trap budget

    def __post_init__(self):
        # invariant the scheduler relies on: k is the realised total, never below k of its parts' floor.
        assert self.k >= 0 and self.explore_k >= 0 and self.verify_k >= 0


@dataclass
class KPolicy:
    """Maps KSignals -> KDecision. Thin by default; spend only where CALIBRATED uncertainty earns it and cost allows.

    k_min            the thin default / floor. 1 == always run the one chosen continuation; 0 == may stay in-model
                     (the degenerate syscall-head policy: below threshold k=0 continue, above k=1 dispatch).
    k_max            hard ceiling — anti-runaway-via-k; the model cannot enlarge its own branching budget.
    widen_threshold  effective-uncertainty above which discretionary widening begins (generalises the syscall
                     head's conf>=0.5 gate). Below it -> thin.
    require_calibrated  honor the calibration trap: a raw uncertainty signal widens k ONLY if calibrated. When
                     False, the policy trusts `uncertainty` directly (use only with a trusted source).
    cost_brake       cost above which discretionary widening is linearly braked toward 0 at cost==1.
    risk_per_k       risk mapped to mandatory verify branches: verify_k = ceil(risk / risk_per_k)-ish (see below).
    unit_cost        per-continuation cost unit for est_cost.
    """
    k_min: int = 1
    k_max: int = 4
    widen_threshold: float = 0.5
    require_calibrated: bool = True
    cost_brake: float = 0.6
    risk_floor_at: float = 0.5     # risk above this starts mandating verification
    unit_cost: float = 1.0

    def __post_init__(self):
        assert 0 <= self.k_min <= self.k_max, "need 0 <= k_min <= k_max"
        assert 0.0 <= self.widen_threshold < 1.0

    # --- the calibration trap, isolated and testable ---
    def _effective_uncertainty(self, sig: KSignals):
        """The uncertainty the policy is ALLOWED to widen on, + which signal governed (audit).

        Independent disagreement is always trustworthy. A raw uncertainty is trusted only if calibrated (or
        require_calibrated=False). If neither is trustworthy but the raw signal is high, return (0.0,
        'uncalibrated_held') — the policy will REFUSE to widen and stay thin (validate-the-branch).
        """
        dis = None if sig.disagreement is None else _clamp01(sig.disagreement)
        raw = _clamp01(sig.uncertainty)
        raw_trusted = raw if (sig.uncertainty_calibrated or not self.require_calibrated) else None

        if dis is not None and raw_trusted is not None:
            return (max(dis, raw_trusted), "independent_disagreement" if dis >= raw_trusted else "calibrated_uncertainty")
        if dis is not None:
            return (dis, "independent_disagreement")
        if raw_trusted is not None:
            return (raw_trusted, "calibrated_uncertainty")
        # only an untrusted raw signal exists
        if raw > self.widen_threshold:
            return (0.0, "uncalibrated_held")     # high hunch we won't act on -> stay thin, validate the branch
        return (0.0, "thin_default")

    def _cost_brake_factor(self, cost: float) -> float:
        """1.0 below cost_brake, falling linearly to 0.0 at cost==1.0 — brakes DISCRETIONARY widening only."""
        c = _clamp01(cost)
        if c <= self.cost_brake:
            return 1.0
        span = 1.0 - self.cost_brake
        return 0.0 if span <= 0 else max(0.0, (1.0 - c) / span)

    def decide(self, sig: KSignals) -> KDecision:
        span = self.k_max - self.k_min                       # the widening headroom above the thin floor
        eff, gov = self._effective_uncertainty(sig)

        # --- discretionary exploration: uncertainty above threshold, scaled by importance, then cost-braked ---
        if span <= 0 or eff <= self.widen_threshold:
            explore_raw = 0.0
        else:
            drive = (eff - self.widen_threshold) / (1.0 - self.widen_threshold)   # 0..1 over the actionable band
            explore_raw = drive * span * _clamp01(sig.importance)
        brake = self._cost_brake_factor(sig.cost)
        explore_k = int(math.ceil(explore_raw * brake - 1e-9))
        explore_k = max(0, min(span, explore_k))
        cost_braked = explore_raw > 0 and brake < 1.0 and explore_k < int(math.ceil(explore_raw - 1e-9))

        # --- mandatory verification: risk floors a verify count COST CANNOT remove (safety, not discretion) ---
        if sig.risk <= self.risk_floor_at or span <= 0:
            verify_k = 0
        else:
            rdrive = (_clamp01(sig.risk) - self.risk_floor_at) / (1.0 - self.risk_floor_at)
            verify_k = max(1, int(math.ceil(rdrive * span - 1e-9)))   # >=1 verify branch once over the risk floor
            verify_k = min(span, verify_k)

        # --- compose: discretionary + mandatory, hard-clamped. Safety invariant: k >= min(k_max, k_min+verify_k),
        #     because adding explore_k (>=0) only raises the pre-clamp total and min(k_max, .) is monotonic. ---
        k = min(self.k_max, self.k_min + explore_k + verify_k)

        # --- audit: report the ACTIVE cause, independent of whether k collapsed to the floor. Priority:
        #     safety (risk floor) > cost braking > a trusted widen > the calibration-trap hold > genuine thin. ---
        if verify_k > 0 and verify_k >= explore_k:
            used = "risk_floor"
            reason = f"risk {sig.risk:.2f} mandates {verify_k} verify branch(es) (cost cannot brake this)"
        elif cost_braked:
            used = "cost_braked"
            reason = (f"{gov} {eff:.2f} would widen but cost {sig.cost:.2f} braked exploration to explore_k={explore_k}")
        elif explore_k > 0:
            used = gov
            reason = f"{gov} {eff:.2f} > {self.widen_threshold:.2f} -> explore_k={explore_k} (importance {sig.importance:.2f})"
        elif gov == "uncalibrated_held":
            used = "uncalibrated_held"
            reason = (f"thin (calibration trap): raw uncertainty {sig.uncertainty:.2f} uncalibrated and no "
                      f"independent signal -> refuse to widen, validate the branch")
        else:
            used = "thin_default"
            reason = f"thin: effective uncertainty {eff:.2f} <= threshold {self.widen_threshold:.2f}"

        return KDecision(k=k, explore_k=explore_k, verify_k=verify_k, used_signal=used,
                         reason=reason, est_cost=k * self.unit_cost)


__all__ = ["normalized_entropy", "KSignals", "KDecision", "KPolicy"]
