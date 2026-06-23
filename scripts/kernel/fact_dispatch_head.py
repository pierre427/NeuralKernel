#!/usr/bin/env python3
"""LEARNED-head 'propose which fact' — the trained syscall head replaces the lexical matcher at the dispatch seam.

This is the learned upgrade to kernel/fact_dispatch.py's ROUTER half of the propose/dispose seam. The lexical
FactRouter proposes a verified fact by surface keywords ("how many ... letter", "gcd", "prime"); HeadFactRouter
proposes it from the MODEL'S OWN RESIDUAL — the syscall head reads North's L24 last-token state at the prompt-end
decision point, classifies it into a primitive class, and that class maps onto the SAME FactSpec the lexical
default_router() already builds. "Learned proposes (this head), kernel disposes (the governed fact-loop)."

It is a DROP-IN router: it exposes the exact same ABI the lexical router does — an object with
`.route(prompt:str) -> FactSpec | None` — so it plugs into fact_dispatch.dispatch() as `router=...` unchanged. The
returned object is one of default_router()'s own FactSpecs (carrying its proven det-primitive provider+validate and
its capability token), so the whole downstream path — capability gate, governed thin/spend, proof ledger — is
identical regardless of which router proposed.

The confidence threshold is the DEGENERATE scheduler-owned-k: the head emits a class + a softmax confidence, and
`conf < conf_threshold` collapses the proposal to None (pass-through, or a lexical fallback if one is wired). This
is the cheap version of the entropy/cost-gated continuation-k — one learned proposer, gated by its own confidence,
with a deterministic fallback underneath.

CLASS -> FactSpec mapping (north_syscall_head.CLASSES -> default_router() spec names):
    count_byte  -> letter_count
    c_gcd       -> gcd
    is_prime    -> is_prime
    none        -> None   (no trap: answer in-model -> pass-through / fallback)
    shortest_dist -> None  (head can route it, but the fact-loop ships no shortest_dist primitive -> honest None)

CRITICAL — NO HOME-GROWN FEATURIZER. The model-bound path uses north_syscall_e2e.load_head + head_decide EXACTLY,
so the residual featurization (chat-template -> run_to_layer at the head's trained layer -> z-score with the head's
own mu/sd) matches the head's training distribution byte-for-byte. Writing a separate featurizer here caused an
out-of-distribution bug before; we do not repeat it.

CPU-TESTABLE without a model: `decide_fn(prompt) -> (class_str, conf)` is injectable. When provided, the router
never touches a model or any safetensors — it just maps the injected (class, conf) through the threshold + class
map. When `decide_fn is None`, the constructor loads the real head and binds head_decide to the given model/tok.
See kernel/test_fact_dispatch_head.py (CPU, no model) and kernel/run_fact_dispatch_head_north.py (real North)."""
from __future__ import annotations
import os
from typing import Callable, Optional

from kernel.fact_dispatch import FactSpec, default_router

# class label emitted by the syscall head  ->  the FactSpec name default_router() registers.
# A class absent from this map (e.g. "none", "shortest_dist") proposes no fact -> None.
HEAD_CLASS_TO_SPEC = {
    "count_byte": "letter_count",
    "c_gcd": "gcd",
    "is_prime": "is_prime",
}


def _specs_by_name(router) -> dict:
    """Index a FactRouter's specs by name so a head class can pull the SAME spec the lexical router would build.

    We reuse default_router()'s specs verbatim (their proven det-primitive provider+validate + capability token);
    the head only changes WHICH spec is proposed, never WHAT a proposed spec does downstream."""
    return {s.name: s for s in router.specs}


class HeadFactRouter:
    """Learned 'propose which fact': classify the prompt's residual with the trained syscall head -> a primitive
    class -> the matching FactSpec, gated by a confidence threshold.

    Same ABI as kernel.fact_dispatch.FactRouter: `.route(prompt) -> FactSpec | None`. Drops into
    fact_dispatch.dispatch() as `router=...` unchanged.

    Decision (the degenerate scheduler-owned-k):
        cls, conf = self._decide(prompt)
        if conf >= conf_threshold and cls maps to a registered spec  -> that FactSpec
        elif lexical_fallback is not None                            -> lexical_fallback.route(prompt)
        else                                                         -> None

    A below-threshold OR unmapped class (none / shortest_dist / anything not in HEAD_CLASS_TO_SPEC) is NOT a fact
    proposal from the head. With a lexical_fallback wired, such a prompt still gets a deterministic second opinion
    (so a confident-but-wrong-class or a low-confidence head can't suppress an obviously-lexical match); without
    one, it is a clean pass-through (routed None in dispatch -> byte-identical to the bare model).

    Args:
      model, tok      North model + tokenizer; required only for the real-head path (decide_fn is None).
      head_path       checkpoint the real head is loaded from. NOTE: north_syscall_e2e.load_head reads the head
                      from north_syscall_adversarial.HEAD_PATH internally; head_path here is the EXISTENCE CHECK
                      (so a missing checkpoint fails clearly at construction, not deep inside load) and an audit
                      record. It defaults to that same /tmp path, so the two always agree out of the box.
      conf_threshold  the scheduler-owned-k gate; below it the head's proposal collapses to None/fallback.
      lexical_fallback an object with .route(prompt) (e.g. default_router()) consulted when the head proposes no
                      fact. None -> no fallback (pure head; pass-through on a non-proposal).
      decide_fn       INJECTION POINT. decide_fn(prompt) -> (class_str, conf_float). When given, the router is
                      fully model-free (CPU-testable) and head_path is NOT required to exist. When None, the
                      constructor loads the real head and binds head_decide(model, tok, ...) at the head's layer.
    """

    def __init__(self, model=None, tok=None, head_path="/tmp/north_syscall_head.safetensors",
                 conf_threshold=0.5, lexical_fallback=None, decide_fn=None,
                 spec_source: Optional[object] = None):
        self.conf_threshold = float(conf_threshold)
        self.head_path = head_path
        self.lexical_fallback = lexical_fallback
        # The pool of specs a head class maps onto. Reuse default_router()'s specs by default so the head proposes
        # the SAME proven (provider, validate, cap) the lexical router would. A caller may pass spec_source to
        # share an already-built router (e.g. the same instance used as lexical_fallback).
        src = spec_source if spec_source is not None else (lexical_fallback or default_router())
        self._specs = _specs_by_name(src)

        if decide_fn is not None:
            # CPU / injected path: no model, no checkpoint required. decide_fn fully determines (class, conf).
            self._decide = decide_fn
            self._head = self._mu = self._sd = self._layer = None
            return

        # Model-bound path: load the REAL head and reuse north_syscall_e2e.head_decide EXACTLY (matched
        # featurization). Imported lazily so the CPU/injected path never imports mlx / a model file.
        if not os.path.exists(head_path):
            raise FileNotFoundError(
                f"syscall head checkpoint not found: {head_path} — train north_syscall_head.py --layer 24 first "
                f"(or pass decide_fn=... for the CPU/model-free path).")
        if model is None or tok is None:
            raise ValueError("HeadFactRouter needs (model, tok) for the real-head path; pass decide_fn=... to run "
                             "model-free.")
        import north_syscall_e2e as e2e  # load_head + head_decide (the trained head's loader + decision)

        d = int(model.args.hidden_size)
        nclasses = len(e2e.CLASSES)
        head, mu, sd = e2e.load_head(d, nclasses)           # reads north_syscall_adversarial.HEAD_PATH
        # Honor the layer the head was trained at (stored in the checkpoint). head_decide defaults L=24, matching
        # `north_syscall_head.py --layer 24`; we read it back so a differently-trained head still featurizes right.
        layer = self._read_head_layer(head_path, default=24)
        self._head, self._mu, self._sd, self._layer = head, mu, sd, layer
        self._model, self._tok, self._e2e = model, tok, e2e

        def _decide(prompt: str):
            # head_decide(model, tok, head, mu, sd, prompt, L) -> (class_str, conf_float, ids). Drop ids: the
            # router only proposes which fact; the dispatch loop owns generation. Featurization is head_decide's
            # (chat-template -> run_to_layer at L -> z-score with this head's mu/sd) — do NOT reimplement it here.
            cls, conf, _ids = self._e2e.head_decide(self._model, self._tok, self._head,
                                                    self._mu, self._sd, prompt, L=self._layer)
            return cls, conf

        self._decide = _decide

    @staticmethod
    def _read_head_layer(head_path: str, default: int = 24) -> int:
        """Read the trained layer index the checkpoint records under key 'layer' (north_syscall_head saves it).
        Falls back to `default` (24, the gateway layer head_decide assumes) if the key is absent or unreadable."""
        try:
            import mlx.core as mx
            flat = mx.load(head_path)
            if "layer" in flat:
                import numpy as np
                return int(np.array(flat["layer"]).reshape(-1)[0])
        except Exception:
            pass
        return int(default)

    def route(self, prompt: str) -> Optional[FactSpec]:
        """Propose the verified fact for `prompt`, or None. Same return contract as FactRouter.route."""
        try:
            cls, conf = self._decide(prompt)
        except Exception:
            # a flaky decision must never crash routing — treat as a non-proposal (fail toward fallback/pass-through),
            # mirroring FactRouter.route's matcher-exception handling.
            cls, conf = "none", 0.0

        spec_name = HEAD_CLASS_TO_SPEC.get(cls)
        if spec_name is not None and float(conf) >= self.conf_threshold:
            spec = self._specs.get(spec_name)
            if spec is not None:
                return spec
            # head proposed a known class but this spec pool lacks it -> treat as a non-proposal (fallback/None).

        # head proposed no fact (none / shortest_dist / sub-threshold / unknown class) -> deterministic fallback,
        # else a clean pass-through.
        if self.lexical_fallback is not None:
            return self.lexical_fallback.route(prompt)
        return None


__all__ = ["HeadFactRouter", "HEAD_CLASS_TO_SPEC"]
