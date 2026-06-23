#!/usr/bin/env python3
"""M5 rung-2: an INERT read-only tap on North's MoE gate-softmax entropy — the OBSERVE half of scheduler-owned-k.

The cohere2_moe gate already computes a full softmax over all 128 experts before taking top-8
(cohere2_moe.py:193-194: `gates = self.gate_act(self.gate(x).astype(float32))`). Its Shannon entropy is a
~free, already-computed uncertainty signal: a one-hot gate (one expert owns the token) == confident routing;
a near-uniform gate == uncertain routing. This module taps that distribution per decode step WITHOUT touching
generation.

INERT BY CONSTRUCTION (the design's `==0` discipline, project_neural_microkernel_endstate): `_GateTap` wraps a
MoE block's `gate_act` and returns the IDENTICAL softmax array the model would have used; the entropy it records
is a SEPARATE graph branch that never feeds back into the logits. So decode is bit-for-bit identical to the
untapped model — proven token-for-token vs production gen_fast in inv_moe_entropy.py.

CALIBRATION CAVEAT (carried into rung-3): the gate-softmax is an INTERNAL routing distribution; whether its
entropy actually PREDICTS task hardness/correctness is unproven and must be measured before KPolicy trusts it
(KSignals.uncertainty_calibrated=False until rung-3 says otherwise). We OBSERVE the expert-k entropy; we never
DRIVE the expert-k (that is the Mode-B router intervention — research-grade). See project_scheduler_owned_k.

Install is removable + exact (mlx nn.Module is a dict subclass: a plain-object gate_act is stored as an instance
attr with the dict child popped; restoring the original nn.Softmax re-installs the child) — verified by the
removal-restores-oracle check in the inv.
"""
from __future__ import annotations
import math
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache


class _GateTap:
    """Inert wrapper around one MoE block's gate_act. Returns orig(x) verbatim; side-records normalized routing
    entropy (over the 128-expert axis) into a shared sink as a lazy mx scalar per (batch,token)."""
    def __init__(self, orig, layer_idx, sink):
        self.orig = orig
        self.layer_idx = layer_idx
        self.sink = sink

    def __call__(self, x):
        gates = self.orig(x)                                            # EXACT distribution the model uses
        n = gates.shape[-1]
        p = gates / mx.maximum(gates.sum(axis=-1, keepdims=True), 1e-12)   # renorm (no-op for softmax); robust
        ent = -(p * mx.log(mx.maximum(p, 1e-12))).sum(axis=-1) / math.log(n)   # [B,T] normalized to [0,1]
        self.sink.append((self.layer_idx, ent))                        # separate branch — never re-enters `gates`
        return gates


class MoEEntropyTap:
    """Installs _GateTap on every MoE block (layers with a gate+gate_act; the dense layer-0 MLP is skipped).
    Use as a context manager or install()/remove() explicitly. drain_step() pulls one decode step's signal."""

    def __init__(self, model):
        self.model = model
        self._sink = []
        self._orig = {}                       # layer_idx -> original gate_act (for exact restore)

    @property
    def layers(self):
        return self.model.model.layers

    @property
    def n_tapped(self):
        return len(self._orig)

    def install(self):
        for li, layer in enumerate(self.layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp, "gate_act"):
                self._orig[li] = mlp.gate_act
                mlp.gate_act = _GateTap(mlp.gate_act, li, self._sink)   # plain obj -> instance attr, child popped
        return self

    def remove(self):
        for li, orig in self._orig.items():
            self.layers[li].mlp.gate_act = orig                        # Module/dict -> child re-installed exactly
        self._orig.clear()
        self._sink[:] = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
        return False

    def drain_step(self, agg="mean"):
        """Eval + clear the entropies recorded since the last drain (== one model() forward = one decode step).
        Returns a per-step signal at the LAST (decision) token position: per-layer entropies + aggregate, all in
        [0,1]. None if nothing was recorded. Draining per step realizes + releases the lazy graph each token."""
        sink, self._sink[:] = list(self._sink), []
        if not sink:
            return None
        lasts = mx.stack([e[..., -1].reshape(-1)[-1] for _, e in sink])   # last-token entropy per layer, one array
        mx.eval(lasts)                                                    # ONE sync/step (not one per layer)
        vals = lasts.tolist()
        per = [(li, v) for (li, _), v in zip(sink, vals)]
        m = sum(vals) / len(vals)
        return {"agg": (m if agg == "mean" else max(vals)),
                "mean": m, "max": max(vals), "min": min(vals), "n_layers": len(per), "per_layer": per}


def gen_with_entropy(adapter, tap, prompt, maxn=96, agg="mean"):
    """gen_fast with the entropy tap LIVE. Tokens are IDENTICAL to adapter.gen_fast (the tap is inert); also
    returns a per-token uncertainty trace `sigs` where sigs[i] is the routing signal of the forward that produced
    out[i]. Uses adapter._step verbatim (so the token stream cannot diverge) and drains per step."""
    ids = adapter.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True, reasoning=False)
    cache = make_prompt_cache(adapter.model)
    nxt = adapter._step(list(ids), cache)
    sigs = []
    s0 = tap.drain_step(agg)
    if s0 is not None:
        sigs.append(s0)                       # prefill forward -> decided out[0]
    out = []
    for _ in range(maxn):
        if nxt == adapter.eos:
            break
        out.append(nxt)
        nxt = adapter._step([nxt], cache)
        s = tap.drain_step(agg)
        if s is not None:
            sigs.append(s)
    return adapter.tok.decode(out), len(out), sigs


__all__ = ["MoEEntropyTap", "gen_with_entropy"]
