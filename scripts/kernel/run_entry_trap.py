#!/usr/bin/env python3
"""M3 verification: the entry trap on North classifies real prompts on ingress + WRITES the authorized
capability into the ContinuationFrame, and the read is INERT (production gen_fast is unchanged before/after a
classify -> the forward pass stays bit-identical). Loads North; run solo. Default head = the L0 entry head."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from north_adapter import NorthAdapter
from kernel.frame import ContinuationFrame
from kernel.entry_trap import EntryTrap

PROBES = [
    ("What is the greatest common divisor of 48 and 36?", "c_gcd"),
    ("Is 91 a prime number?", "is_prime"),
    ("How many times does the letter 'r' appear in 'strawberry'?", "count_byte"),
    ("Shortest path distance from node 0 to node 5 in my weighted graph?", "shortest_dist"),
    ("Write a short poem about the sea.", "none"),
    ("Explain how photosynthesis works in simple terms.", "none"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="/tmp/nsh_L0.safetensors")
    a = ap.parse_args()
    print(f"[load] North + EntryTrap({a.head})", flush=True)
    base = NorthAdapter()
    et = EntryTrap(base.model, base.tok, a.head)
    print(f"[entry trap] reading the L{et.layer} residual\n", flush=True)
    correct = 0
    for prompt, expect in PROBES:
        f = ContinuationFrame(request_id=1, site_idx=et.layer)
        info = et.stamp(f, prompt)
        hit = info["class"] == expect
        correct += int(hit)
        print(f"  {'OK ' if hit else 'XX '} '{prompt[:46]:46s}' -> {info['class']:13s} "
              f"conf={info['conf']:.2f} caps={info['capabilities']}", flush=True)
    print(f"\n  entry routing on probes: {correct}/{len(PROBES)}", flush=True)

    # PARITY: the entry classification is a READ -> production gen_fast output is unchanged before/after
    p = PROBES[0][0]
    before, _ = base.gen_fast(p, maxn=40)
    et.stamp(ContinuationFrame(request_id=2, site_idx=et.layer), p)
    after, _ = base.gen_fast(p, maxn=40)
    print(f"  [parity] gen_fast identical before/after entry classify (read is inert): {before == after}", flush=True)
    print(f"\n===== M3 entry trap: routing {correct}/{len(PROBES)}, read-inert={before == after} =====", flush=True)


if __name__ == "__main__":
    main()
