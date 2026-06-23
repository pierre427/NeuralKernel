#!/usr/bin/env python3
"""CPU-only test for the M4 deterministic-interrupt LOGIT-MASK op (no model). Proves the mask forces argmax away
from forbidden ids and that an empty mask is the identity. Run with the mlx venv."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx

CHECKS = []
def ok(c, l): CHECKS.append((c, l)); print(f"  {'OK ' if c else 'XX '} {l}")


def _mask(logits, forbidden):
    neg = mx.array(-1e9, dtype=mx.float32)
    if forbidden.size == 0:
        return logits
    return mx.where(mx.zeros_like(logits).at[forbidden].add(1.0) > 0, neg, logits)


def test_mask_forces_argmax():
    print("\n[1] the deterministic interrupt masks forbidden ids before argmax")
    logits = mx.array([1.0, 5.0, 3.0, 9.0, 2.0])                          # unmasked argmax = id 3 (9.0)
    ok(int(mx.argmax(_mask(logits, mx.array([], dtype=mx.int32))).item()) == 3,
       "empty mask -> argmax unchanged (identity / inert control)")
    ok(int(mx.argmax(_mask(logits, mx.array([3], dtype=mx.int32))).item()) == 1,
       "forbidding the top id -> argmax falls to the next-best allowed id (1, 5.0)")
    ok(int(mx.argmax(_mask(logits, mx.array([3, 1], dtype=mx.int32))).item()) == 2,
       "forbidding the top two -> argmax falls to id 2 (3.0); the constraint is exact")


if __name__ == "__main__":
    test_mask_forces_argmax()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} det-interrupt checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
