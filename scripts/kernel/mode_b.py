#!/usr/bin/env python3
"""Mode-B (the design's open research gap): inject a verified result as a HIDDEN-STATE vector at a layer boundary,
instead of as tokens (Mode A's trampoline). The substrate is the proven zero-init ResidualGateway pattern, made
RESULT-CONDITIONED so the injected vector encodes the kernel's verified value.

  delta = scale * up(silu(down( norm(x) + result_proj(r) )))           up zero-init -> delta == 0 -> IDENTITY
  y     = x + delta.astype(x.dtype)                                    (.astype: the bf16/MXFP8 upcast discipline)

DISABLED == BASE, exactly: with up.weight == 0 (untrained) OR armed == False, delta is exactly 0 for ANY result
vector, so the model is bit-identical to base (the design's non-negotiable: parity gate is ==0, never <eps). Only
once `up` is TRAINED does the injector steer downstream — and that steering must then clear the SAME hold-out
oracle the deterministic handlers pass + a placebo control (B2), not the model's own confidence. This module is
the mechanism + the disabled-identity invariant (B0); the trained steering is B2, where it may land partial or
negative — that is a legitimate result for an open research gap.

This is the WRITE half of "learned proposes, kernel disposes": the kernel computed a verified result; Mode B
writes it into the residual stream so downstream layers read it WITHOUT a token round-trip. We OWN the write
point (the zero-init sidecar, never a 129th MoE expert); whether the write reliably steers is the open question.
"""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask


class ModeBInjector(nn.Module):
    """Result-conditioned residual injector. y = x + up(silu(down(norm(x) + result_proj(r)))); up zero-init.

    Identity by construction when disabled (up==0 or armed==False) for ANY result vector -> the B0 ==0 invariant.
    `result_dim` is the width of the encoded result (task-defined in B2, e.g. a one-hot/embedding of an integer
    answer); `bottleneck` keeps the trainable write low-rank (like the proven gateway)."""
    def __init__(self, d, bottleneck=64, result_dim=16, scale=1.0):
        super().__init__()
        self.d = d
        self.result_dim = result_dim
        self.norm = nn.RMSNorm(d, eps=1e-6)
        self.result_proj = nn.Linear(result_dim, d, bias=False)
        self.down = nn.Linear(d, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, d, bias=False)
        self.up.weight = mx.zeros_like(self.up.weight)        # zero-init -> exact identity until trained
        self.scale = scale
        self.armed = True                                     # master switch; False -> pure pass-through

    def delta(self, x, result_vec=None):
        """The residual write (exactly 0 while up.weight==0). Separated so tests can inspect it."""
        cond = self.norm(x)
        if result_vec is not None:
            rv = result_vec if isinstance(result_vec, mx.array) else mx.array(result_vec, dtype=mx.float32)
            proj = self.result_proj(rv)
            if proj.ndim == x.ndim - 1 and x.ndim >= 3:   # [B,d] vs x [B,T,d] -> add a token axis to broadcast over T
                proj = mx.expand_dims(proj, axis=-2)
            cond = cond + proj
        return self.scale * self.up(nn.silu(self.down(cond)))

    def __call__(self, x, result_vec=None):
        if not self.armed:
            return x
        return x + self.delta(x, result_vec).astype(x.dtype)


def run_stack_modeb(model, inputs, injector=None, site=None, result_vec=None, cache=None):
    """Faithful cohere2_moe forward (== CohereModel.__call__ + head) with the Mode-B injector applied AFTER block
    `site`. Threads the real per-layer cache (prefill: cache=None; decode: make_prompt_cache). With injector=None
    or up.weight==0 the result is bit-identical to model(inputs, cache)."""
    inner = model.model
    h = inner.embed_tokens(inputs)
    layers = inner.layers
    if cache is None:
        cache = [None] * len(layers)
    for li, (layer, c) in enumerate(zip(layers, cache)):
        win = inner.window_size if layer.self_attn.use_sliding_window else None
        mask = create_attention_mask(h, c, window_size=win)
        h = layer(h, mask, c)
        if injector is not None and li == site:
            h = injector(h, result_vec)
    return inner.embed_tokens.as_linear(inner.norm(h)) * model.args.logit_scale


__all__ = ["ModeBInjector", "run_stack_modeb"]
