#!/usr/bin/env python3
"""ContinuationFrame — the suspended neural process (Phase-4 / design §2b "Continuation-frame schema").

A frame is the kernel's handle on a paused forward pass: where it stopped (site_idx/token_pos), the suspended
state (hidden_ref for a parked residual; kv_cache_ref for the decode KV cache), the decode cursor, and the
capability/budget the scheduler grants it. The model never owns this — the scheduler does.

Pure dataclass (holds opaque handles; no MLX import), so it stays importable and testable without loading a
model. Fields are marked REAL-NOW (populated by the no-op trap-plane foundation against the real model) vs
STUBBED (await M1/M2/M4 wiring). For the no-op milestone a frame is CONSTRUCTED but never mutated
(capabilities empty, traps_left=0, armed=False) — proving the plane is observationally null even with frames live.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ContinuationFrame:
    # --- identity / position (REAL NOW) ---
    request_id: int                       # caller-stamped, replayable at T=0
    site_idx: int                         # trap site in {0,8,16,24,32,40,48}; resume at site_idx+1
    token_pos: int = 0                    # = cache[0].offset at park (KVCache.offset)

    # --- suspended forward pass ---
    hidden_ref: Optional[Any] = None      # REAL NOW (prefill): the residual x at the post-site_idx boundary
                                          #   (park/resume proven exact via safetensors round-trip, gateway-invariants §2)
    kv_cache_ref: Optional[Any] = None    # M2-REAL (kernel/kv_park.py): a KVSnapshot — per-layer
                                          #   (classname, deep-copied (keys,values), meta_state); park =
                                          #   snapshot_cache(cache), resume = restore_cache(snap) via from_state,
                                          #   rollback = rollback_or_restore (trim pre-wrap / restore post-wrap).
                                          #   Holds the SNAPSHOT, not the live cache (deep-copied to beat aliasing).

    # --- decode state ---
    last_token: Optional[int] = None      # M2: the int that seeds the resumed step (out[-1] at park)
    n_emitted: int = 0                    # M2: tokens emitted before the park
    eos_id: Optional[int] = None          # REAL NOW: adapter.eos
    temp: float = 0.0                     # REAL NOW: T=0 greedy (argmax)

    # --- capabilities / budget (REAL-as-INERT for the no-op plane) ---
    capabilities: frozenset = frozenset() # REAL NOW: empty == no opcode may fire
    traps_left: int = 0                   # REAL NOW: 0 == plane installed, activated nowhere
    depth_left: int = 0                   # STUBBED (M4)
    ops_used: set = field(default_factory=set)    # STUBBED (M1 ledger)
    deadline_ticks: Optional[int] = None  # STUBBED (M2 scheduler)
    armed: bool = False                   # REAL NOW: master no-op switch; False == observationally null


__all__ = ["ContinuationFrame"]
