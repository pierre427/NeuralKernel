#!/usr/bin/env python3
"""Phase-4 M2 rung-3 (deferred half): ROLLBACK across the RotatingKVCache RING WRAP on the real North model.

inv_speculate_rollback proved the PRE-WRAP trim fast-path (==0). This proves the POST-WRAP snapshot-RESTORE
branch of kv_park.rollback_or_restore — the branch kv_park.py itself flags DORMANT ("gate its first USE behind
its own North ==0 battery"). We decode PAST 4096 tokens so North's 36 sliding RotatingKVCache(max_size=4096)
layers WRAP (is_trimmable -> False, so can_trim_prompt_cache -> False), then at a park point park -> speculate k
-> rollback. Because the ring has evicted tokens, rollback_or_restore MUST discard the live cache and rebuild
from the deep-copied snapshot (restore_cache). The re-decoded continuation must STILL equal a straight-through
decode token-for-token (==0). This closes the last concrete deferred kernel item.

EOS is IGNORED so the wrap is guaranteed; ~MAXN tokens are decoded TWICE, so this runs several minutes. The
post-context-length tokens may be degenerate but are DETERMINISTIC, so oracle==test still holds iff restore is
bit-exact. Loads North; run solo."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache
from north_gateway_invariants import MODEL
from kernel.trapped_adapter import TrappedNorthAdapter
from kernel.kv_park import snapshot_cache, rollback_or_restore

PROMPT = "Count from 1 upward, writing one integer per line, and keep going without stopping."
MAXN = 4260
PARK_AT = 4200            # > 4096 -> every sliding layer has wrapped -> can_trim False -> restore path
K = 8


def _attach(model, tok, eos):
    a = TrappedNorthAdapter.__new__(TrappedNorthAdapter)
    a.model, a.tok, a.eos = model, tok, eos
    a.plane = None        # proven-identity native forward (run_stack_plane plane=None)
    return a


def _decode(ta, ids, maxn, park_at=None, k=0):
    """Inline trapped decode (EOS IGNORED). At park_at: snapshot -> speculate k (advance+discard) -> rollback via
    the real rollback_or_restore. Returns (out_tokens, rollback_path, trimmable_at_park)."""
    cache = make_prompt_cache(ta.model)
    nxt = ta._step_trapped(list(ids), cache); out = []; did = False; path = "none"; trim_at_park = None
    for _ in range(maxn):
        if park_at is not None and not did and len(out) == park_at:
            did = True
            snap = snapshot_cache(cache)
            s = nxt
            for _ in range(k):                                   # speculate k (advance the live ring, discard)
                s = ta._step_trapped([s], cache)
            trim_at_park = bool(can_trim_prompt_cache(cache))    # expect False post-wrap -> restore branch
            path = "trim" if trim_at_park else "restore"
            cache = rollback_or_restore(cache, snap, k)
        out.append(nxt); nxt = ta._step_trapped([nxt], cache)
    return out, path, trim_at_park


def main():
    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    eos = tok.encode("<|END_TEXT|>", add_special_tokens=False)[0]
    ta = _attach(model, tok, eos)
    ids = ta.encode_fast([{"role": "user", "content": PROMPT}])

    print(f"[oracle] straight-through decode of {MAXN} tokens (EOS ignored)...", flush=True)
    oracle, _, _ = _decode(ta, ids, MAXN)
    print(f"[test] park@{PARK_AT} -> speculate {K} -> rollback -> resume to {MAXN}...", flush=True)
    test, path, trim_at_park = _decode(ta, ids, MAXN, park_at=PARK_AT, k=K)

    eq = (oracle == test)
    mism = sum(1 for a, b in zip(oracle, test) if a != b) + abs(len(oracle) - len(test))
    restore_taken = (path == "restore")
    R = {"maxn": MAXN, "park_at": PARK_AT, "k": K, "n_oracle": len(oracle), "n_test": len(test),
         "trimmable_at_park": trim_at_park, "rollback_path": path, "restore_path_taken": restore_taken,
         "token_mismatches": mism, "tokens_eq": eq,
         "RESULT": "PASS" if (eq and restore_taken) else "FAIL"}
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"); os.makedirs(logdir, exist_ok=True)
    json.dump(R, open(os.path.join(logdir, "inv_rollback_ringcross.json"), "w"), indent=2, default=str)
    print(f"  trimmable@park={trim_at_park} -> rollback_path={path} | tokens_eq={eq} mismatches={mism} "
          f"(n {len(oracle)} vs {len(test)})", flush=True)
    print(json.dumps(R, default=str), flush=True)
    sys.exit(0 if R["RESULT"] == "PASS" else 1)


if __name__ == "__main__":
    main()
