#!/usr/bin/env python3
"""CPU-only unit test for the Mode-B injector (no model). Proves the disabled-identity invariant locally + that
the injector CAN write once trained + is result-conditioned + dtype-preserving.
Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/test_mode_b.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
from kernel.mode_b import ModeBInjector

CHECKS = []
def ok(c, l): CHECKS.append((bool(c), l)); print(f"  {'OK ' if c else 'XX '} {l}")
def _maxabs(a, b): mx.eval(a, b); return float(mx.max(mx.abs(a - b)).item())


def test_identity_when_disabled():
    print("\n[1] disabled-identity: up zero-init OR armed=False -> exact pass-through for ANY result vec")
    inj = ModeBInjector(d=8, bottleneck=4, result_dim=3)
    mx.eval(inj.parameters())
    x = mx.random.normal((2, 5, 8)); r = mx.array([1.0, 2.0, 3.0])
    ok(_maxabs(inj(x, r), x) == 0.0, "up zero-init -> y==x (identity) with a NONZERO result vec")
    ok(float(mx.max(mx.abs(inj.delta(x, r))).item()) == 0.0, "delta exactly 0 when up==0")
    inj.up.weight = mx.random.normal(inj.up.weight.shape) * 0.1     # "train" up nonzero
    inj.armed = False
    ok(_maxabs(inj(x, r), x) == 0.0, "armed=False -> pass-through (==0) EVEN with a trained up")


def test_writes_when_trained():
    print("\n[2] trained up -> the injector writes, and the write is result-conditioned")
    inj = ModeBInjector(d=8, bottleneck=4, result_dim=3)
    inj.up.weight = mx.random.normal(inj.up.weight.shape) * 0.1
    mx.eval(inj.parameters())
    x = mx.random.normal((1, 3, 8))
    ok(_maxabs(inj(x, mx.array([1.0, 0.0, 0.0])), x) > 0.0, "trained up -> nonzero write (steers x)")
    da = inj.delta(x, mx.array([1.0, 0.0, 0.0])); db = inj.delta(x, mx.array([0.0, 0.0, 1.0]))
    ok(float(mx.max(mx.abs(da - db)).item()) > 0.0, "different result vec -> different write (result-conditioned)")
    ok(float(mx.max(mx.abs(inj.delta(x, None))).item()) > 0.0, "also writes with no result vec (unconditioned)")
    # batched result vec [B,K] must broadcast over T (the B2 training shape)
    xb = mx.random.normal((2, 5, 8)); rb = mx.eye(3)[mx.array([0, 2])]      # [2,3]
    db = inj.delta(xb, rb); mx.eval(db)
    ok(db.shape == (2, 5, 8), f"batched result vec [B,K] broadcasts over T (delta {db.shape})")


def test_dtype_preserved():
    print("\n[3] dtype discipline: output matches x.dtype (bf16/MXFP8 upcast hazard)")
    inj = ModeBInjector(d=8, bottleneck=4, result_dim=3)
    inj.up.weight = mx.random.normal(inj.up.weight.shape) * 0.1
    mx.eval(inj.parameters())
    x = mx.random.normal((1, 2, 8)).astype(mx.bfloat16)
    y = inj(x, mx.array([1.0, 0.0, 0.0])); mx.eval(y)
    ok(y.dtype == mx.bfloat16, "y.dtype == x.dtype (bf16) after the residual add")


if __name__ == "__main__":
    test_identity_when_disabled(); test_writes_when_trained(); test_dtype_preserved()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} mode_b unit checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
