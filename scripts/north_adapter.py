#!/usr/bin/env python3
"""North-Mini-Code ModelAdapter — same interface as gptoss_adapter / mellum_adapter.

North uses reasoning on/off (chat_template `reasoning=`) and terminates answers with <|END_TEXT|>."""
from __future__ import annotations
import os
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache

NORTH = os.environ.get("NORTH_MODEL", "/Users/pierrelamy/.lmstudio/models/bsisduck/North-Mini-Code-1.0-MLX-MXFP8")


class NorthAdapter:
    name = "north-mini-code"

    def __init__(self, model_path=NORTH):
        self.model, self.tok = load(model_path); self.model.eval()
        self.eos = self.tok.encode("<|END_TEXT|>", add_special_tokens=False)[0]

    def encode_fast(self, messages):
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, reasoning=False)

    def _step(self, ids, cache):
        return int(mx.argmax(self.model(mx.array([ids], dtype=mx.int32), cache=cache)[0, -1, :]).item())

    def _gen(self, prompt, reasoning, maxn):
        ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, reasoning=reasoning)
        cache = make_prompt_cache(self.model)
        nxt = self._step(list(ids), cache); out = []
        for _ in range(maxn):
            if nxt == self.eos:
                break
            out.append(nxt); nxt = self._step([nxt], cache)
        return self.tok.decode(out), len(out)

    def gen_fast(self, prompt, maxn=1800):
        return self._gen(prompt, False, maxn)

    def gen_reasoning(self, prompt, maxn=4000):
        return self._gen(prompt, True, maxn)

    def gen_sample(self, prompt, temp=0.8, seed=0, maxn=1800):
        """Temperature-sampled decode for DIVERSE candidates — at T=0 retries are byte-identical, so an escalation
        ladder must raise temperature to get genuinely different solutions. Seeded for replay/reproducibility."""
        ids = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, reasoning=False)
        cache = make_prompt_cache(self.model)
        key = mx.random.key(int(seed))
        cur = list(ids); out = []
        inv_t = 1.0 / max(float(temp), 1e-6)
        for _ in range(maxn):
            logits = self.model(mx.array([cur], dtype=mx.int32), cache=cache)[0, -1, :]
            key, sub = mx.random.split(key)
            nxt = int(mx.random.categorical(logits * inv_t, key=sub).item())
            if nxt == self.eos:
                break
            out.append(nxt); cur = [nxt]
        return self.tok.decode(out), len(out)
