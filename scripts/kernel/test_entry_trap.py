#!/usr/bin/env python3
"""CPU-only tests for the M3 entry-trap capability-WRITE logic (no LLM — classify needs the model, the
class->capability frame-stamp + honest-boundary threshold do not). Run with the mlx venv (imports mlx via the
chain; loads NO model):
  /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/test_entry_trap.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from north_syscall_head import CLASSES
from kernel.frame import ContinuationFrame
from kernel.entry_trap import capability_for, stamp_from_class

CHECKS = []
def ok(c, l): CHECKS.append((c, l)); print(f"  {'OK ' if c else 'XX '} {l}")
def _frame(): return ContinuationFrame(request_id=1, site_idx=0)


def test_capability_for():
    print("\n[1] class -> capability (CLASSES[0]=='none' authorizes no trap)")
    ok(capability_for(0) is None, "class 'none' -> no capability")
    ok(capability_for(2) == f"trap:{CLASSES[2]}", f"class '{CLASSES[2]}' -> trap:{CLASSES[2]}")
    ok(capability_for(len(CLASSES) - 1) == f"trap:{CLASSES[-1]}", f"last class -> trap:{CLASSES[-1]}")


def test_stamp_confident_trap():
    print("\n[2] a confident trap class STAMPS the capability + 1 trap of budget into the frame")
    f = _frame(); info = stamp_from_class(f, 2, conf=0.97, conf_threshold=0.5)
    ok(f.capabilities == frozenset({"trap:c_gcd"}), f"frame.capabilities == {{trap:c_gcd}} (got {set(f.capabilities)})")
    ok(f.traps_left == 1, "frame.traps_left == 1 (one trap authorized)")
    ok(info["capability"] == "trap:c_gcd", "report names the authorized capability")


def test_none_class_stamps_nothing():
    print("\n[3] the 'none' class authorizes NO trap (answer in-model)")
    f = _frame(); stamp_from_class(f, 0, conf=0.99)
    ok(not f.capabilities and f.traps_left == 0, "frame stays empty (no capability, no trap budget)")


def test_honest_boundary_below_threshold():
    print("\n[4] honest boundary: a low-confidence classification stamps NOTHING")
    f = _frame(); info = stamp_from_class(f, 2, conf=0.30, conf_threshold=0.5)
    ok(not f.capabilities and f.traps_left == 0, "below threshold -> no capability stamped (refuse, don't guess)")
    ok(info["capability"] is None, "report shows no capability authorized")


if __name__ == "__main__":
    test_capability_for(); test_stamp_confident_trap(); test_none_class_stamps_nothing(); test_honest_boundary_below_threshold()
    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} entry-trap checks passed")
    sys.exit(0 if n == len(CHECKS) else 1)
