#!/usr/bin/env python3
"""Phase-4 M3: the ENTRY TRAP — the learned syscall head classifies a request on ingress and WRITES the
authorized capability + trap budget into the ContinuationFrame. "Learned proposes (the head), the kernel
disposes (the Phase-0a capability gate authorizes the dispatch)."

A pure READ + frame-stamp: it reads a FROZEN residual at the entry site (run_to_layer returns a slice; nothing
is written back), so the model's output is bit-identical — the site's read-tap is parity-proven inert (==0) in
inv_trap_plane. No residual mutation, no MoE-gate perturbation.

Reuses north_syscall_head.SyscallHead + run_to_layer (both proven). The head is loaded from a safetensors that
records its layer (train via `north_syscall_head.py --layer L --out ...`). Design-ideal site is L0 (post dense
block); whether the entry signal is strong enough there vs the proven L24 is the empirical question the
head-accuracy comparison answers — this module works at whichever layer the loaded head was trained for.
"""
from __future__ import annotations
import numpy as np
import mlx.core as mx
from mlx.utils import tree_unflatten
from north_syscall_head import SyscallHead, CLASSES, run_to_layer
from kernel.frame import ContinuationFrame


def capability_for(cls_idx: int):
    """The capability the entry trap authorizes for a class. CLASSES[0]=='none' -> no trap (answer in-model)."""
    return None if cls_idx == 0 else f"trap:{CLASSES[cls_idx]}"


def stamp_from_class(frame: ContinuationFrame, cls_idx: int, conf: float, conf_threshold: float = 0.5) -> dict:
    """Pure frame-stamp logic (no model): write the authorized capability + 1 trap of budget into the frame
    when the head is confident; honest-boundary below threshold -> stamp NOTHING (capabilities stay empty)."""
    cap = capability_for(cls_idx) if conf >= conf_threshold else None
    if cap is not None:
        frame.capabilities = frozenset({cap})
        frame.traps_left = 1
    return {"class": CLASSES[cls_idx], "conf": round(float(conf), 3), "capability": cap,
            "capabilities": set(frame.capabilities), "traps_left": frame.traps_left}


def load_head(path: str, d: int):
    """Load a SyscallHead (+ z-score stats + its layer) from a north_syscall_head safetensors."""
    w = mx.load(path)
    layer = int(np.array(w["layer"])[0]) if "layer" in w else 24
    head = SyscallHead(d, len(CLASSES))
    params = [(k[len("head."):], v) for k, v in w.items() if k.startswith("head.")]
    head.update(tree_unflatten(params))
    mx.eval(head.parameters())
    return head, w["mu"], w["sd"], layer


class EntryTrap:
    """Reads the entry-site residual, classifies via the syscall head, stamps capabilities into a frame.

    Owns the encoding so it MATCHES the head's training distribution exactly (north_syscall_head.featurize uses
    `apply_chat_template(add_generation_prompt=True)` with the template default reasoning). Feeding a differently-
    encoded prompt (e.g. forced reasoning=False) yields an out-of-distribution residual and the head misclassifies.
    """

    def __init__(self, model, tok, head_path: str, conf_threshold: float = 0.5):
        self.model = model
        self.tok = tok
        self.head, self.mu, self.sd, self.layer = load_head(head_path, int(model.args.hidden_size))
        self.conf_threshold = conf_threshold

    def _encode(self, prompt: str):
        # EXACTLY north_syscall_head.featurize's encoding (the head's training distribution)
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)

    def classify(self, prompt: str):
        """Read the FROZEN residual at self.layer (no mutation) -> (class_idx, confidence)."""
        ids = self._encode(prompt)
        x = run_to_layer(self.model, mx.array(np.array([ids], dtype=np.int32)), self.layer).astype(mx.float32)
        xn = (x - self.mu) / self.sd
        probs = mx.softmax(self.head(xn)[0])
        idx = int(mx.argmax(probs).item())
        return idx, float(probs[idx].item())

    def stamp(self, frame: ContinuationFrame, prompt: str) -> dict:
        """Classify on entry and WRITE the authorized capability/budget into the frame (the M3 capability write)."""
        idx, conf = self.classify(prompt)
        frame.site_idx = self.layer
        return stamp_from_class(frame, idx, conf, self.conf_threshold)


__all__ = ["EntryTrap", "capability_for", "stamp_from_class", "load_head"]
