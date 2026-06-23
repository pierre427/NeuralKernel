#!/usr/bin/env python3
"""Neural-scheduler milestone-1 invariants, realized on REAL North (cohere2_moe).

The concurrent neural-scheduler session proved, on a toy transformer, the two
mechanism invariants a per-layer "gateway scheduler" needs:
  1. Identity-init noop: a zero-init gateway grafted between layers is bit-identical
     to the base model (max|gated - base| = 0).
  2. Exact park/resume: run layers 0..L, stash the residual (the suspended
     continuation), resume L..N from the stash -> bit-identical to straight-through.

This script reproduces both on North-Mini-Code's 49-layer cohere2_moe stack — the
foundation for an adaptive-depth halting gateway (the speedup track).
"""
from __future__ import annotations
import os
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load

MODEL = os.environ.get("NORTH_MODEL", "/Users/pierrelamy/.lmstudio/models/bsisduck/North-Mini-Code-1.0-MLX-MXFP8")
PROMPT = "Write a Python function `add(a: int, b: int) -> int` that returns a + b."


class ResidualGateway(nn.Module):
    """Zero-init residual patch == exact identity until trained (the shared-expert /
    Mixture-of-Depths gateway pattern). y = x + scale * up(silu(down(norm(x)))), up=0."""
    def __init__(self, d, bottleneck=64):
        super().__init__()
        self.norm = nn.RMSNorm(d, eps=1e-6)
        self.down = nn.Linear(d, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, d, bias=False)
        self.up.weight = mx.zeros_like(self.up.weight)  # zero-init -> identity noop
        self.scale = 1.0

    def __call__(self, x):
        # delta is exactly 0 at init (up.weight=0); cast to x.dtype so the residual
        # is not silently upcast (bf16/quantized base) — preserves bit-identity.
        delta = self.scale * self.up(nn.silu(self.down(self.norm(x))))
        return x + delta.astype(x.dtype)


def run_stack(model, inputs, gateway=None, gate_at=None, park_at=None):
    """Manual cohere2_moe forward. Optionally insert `gateway` after layer `gate_at`,
    or park (stash + resume) the residual after layer `park_at`."""
    inner = model.model
    x = inner.embed_tokens(inputs)
    parked = None
    layers = inner.layers
    for li, layer in enumerate(layers):
        win = inner.window_size if layer.self_attn.use_sliding_window else None
        mask = create_attention_mask(x, None, window_size=win)
        x = layer(x, mask, None)
        if gateway is not None and li == gate_at:
            x = gateway(x)
        if park_at is not None and li == park_at:
            # serialize the suspended continuation to DISK (safetensors preserves dtype),
            # then resume from the reloaded tensor — the strongest form of park/resume.
            mx.save_safetensors("/tmp/north_parked.safetensors", {"h": x})
            x = mx.load("/tmp/north_parked.safetensors")["h"]
    return inner.embed_tokens.as_linear(inner.norm(x)) * model.args.logit_scale


def maxabs(a, b):
    mx.eval(a, b)
    return float(mx.max(mx.abs(a - b)).item())


def main():
    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    d = int(model.args.hidden_size); nL = int(model.args.num_hidden_layers)
    ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True)
    inputs = mx.array(np.array([ids], dtype=np.int32))

    base = model(inputs); mx.eval(base)
    manual = run_stack(model, inputs)
    p_manual = maxabs(manual, base)
    print(f"[parity] manual cohere2_moe forward vs model(): max abs diff = {p_manual:.3e}", flush=True)

    L = nL // 2
    gw = ResidualGateway(d)
    gated = run_stack(model, inputs, gateway=gw, gate_at=L)
    p_ident = maxabs(gated, base)
    print(f"[invariant 1] zero-init gateway after layer {L}: max|gated - base| = {p_ident:.3e}  ({'PASS' if p_ident == 0 else 'FAIL'})", flush=True)

    resumed = run_stack(model, inputs, park_at=L)
    p_park = maxabs(resumed, base)
    print(f"[invariant 2] park@{L} -> serialize residual -> resume {L}..{nL}: max|resumed - base| = {p_park:.3e}  ({'PASS' if p_park == 0 else 'FAIL'})", flush=True)

    import json
    print(json.dumps({"hidden_size": d, "layers": nL,
                      "manual_parity": p_manual, "identity_init_invariant": p_ident, "park_resume_invariant": p_park,
                      "RESULT": "PASS" if (p_ident == 0 and p_park == 0) else "FAIL"}), flush=True)


if __name__ == "__main__":
    main()
