#!/usr/bin/env python3
"""Phase-4 M3-foundation: a NO-OP sidecar TRAP PLANE wired into the real North forward pass.

Reuses the PROVEN ResidualGateway (north_gateway_invariants) VERBATIM — zero-init so each sidecar's delta is
exactly 0, and the residual add keeps `x.dtype` (the bf16/MXFP8 upcast discipline that the 2.943 hazard taught).
Installed at the trap sites {0,8,16,24,32,40,48} in identity/disarmed mode, the plane is bit-identical to the
native model (CohereModel.__call__) by construction — the safe substrate for the M1 commit trap (site 48), the
M3 entry trap (site 0), and M4 mid-layer interrupts.

No monkeypatch, no model-graph edit: `run_stack_plane` is a faithful manual re-implementation of
CohereModel.__call__ (cohere2_moe.py:258-279) + the Model head (as_linear * logit_scale), with a post-block hook.
It threads the REAL per-layer cache so it is correct for BOTH prefill (cache=None) and token-by-token decode —
the cache-threaded step is the integration seam M1/M4 will hook (read the residual at a site, optionally mutate
before the next layer). All 7 sites are full_attention layers (KVCache => exact trim()), so later commit/rollback
is reliable (no RotatingKVCache ring-wrap hazard).
"""
from __future__ import annotations
import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from north_gateway_invariants import ResidualGateway   # reuse the proven identity primitive verbatim

# all full_attention layers (config layer_types is period-4); 48 = post-final-layer commit point, 0 = entry trap
SITES = (0, 8, 16, 24, 32, 40, 48)


class TrapPlane:
    """Multi-site no-op sidecar plane. Identity by construction (each gateway up.weight == 0).

    Two independent guards both yield ==0: (a) disarmed -> apply() is a pure pass-through (the gateway is never
    even called); (b) armed but every up.weight==0 -> gateway returns x + 0.astype(x.dtype) == x. The double
    guard lets the parity test prove BOTH "plane absent" and "plane present-but-inert" are bit-identical.
    """
    def __init__(self, model, sites=SITES):
        nL = int(model.args.num_hidden_layers)
        bad = [s for s in sites if not (0 <= s < nL)]
        assert not bad, f"trap sites {bad} out of range for num_hidden_layers={nL}"
        d = int(model.args.hidden_size)
        self.sites = set(sites)
        self.gateways = {s: ResidualGateway(d) for s in sites}   # zero-init -> identity
        for g in self.gateways.values():
            mx.eval(g.parameters())
        self.armed = False    # master switch; disarmed -> apply() is a pass-through

    def apply(self, li, x):
        """Post-block hook. Identity when disarmed; still identity when armed while up.weight==0 (no-op plane)."""
        if self.armed and li in self.sites:
            return self.gateways[li](x)
        return x


def run_stack_plane(model, inputs, plane=None, cache=None):
    """Manual cohere2_moe forward (faithful to CohereModel.__call__ + Model head) with a post-block TrapPlane hook.

    Threads the REAL per-layer cache: prefill -> cache=None (==[None]*L, exactly native's prefill path); decode ->
    cache = model.make_cache()/make_prompt_cache(model). With plane=None or plane.armed=False the result is
    bit-identical to model(inputs, cache).
    """
    inner = model.model
    h = inner.embed_tokens(inputs)
    layers = inner.layers
    if cache is None:
        cache = [None] * len(layers)
    for li, (layer, c) in enumerate(zip(layers, cache)):
        win = inner.window_size if layer.self_attn.use_sliding_window else None
        mask = create_attention_mask(h, c, window_size=win)
        h = layer(h, mask, c)
        if plane is not None:
            h = plane.apply(li, h)                         # identity when disarmed / no-op
    return inner.embed_tokens.as_linear(inner.norm(h)) * model.args.logit_scale


__all__ = ["TrapPlane", "run_stack_plane", "SITES"]
