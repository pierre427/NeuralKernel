#!/usr/bin/env python3
"""gpt-oss-20b ModelAdapter — maps the harmony format to the shape_appliance interface.

gpt-oss has no North-style `END_THINK`/`START_TEXT` tokens. Instead it emits the HARMONY format:
    <|channel|>analysis<|message|>{reasoning}<|end|><|start|>assistant<|channel|>final<|message|>{answer}<|return|>
Reasoning depth is the `reasoning_effort` chat-template kwarg (low/medium/high), not a reasoning on/off
flag. So the appliance's fast path = reasoning_effort="low" (minimal analysis), and the escalation =
reasoning_effort="high". The ANSWER is the `final` channel; we extract it and hand the code block to
the same recovery/detector/validator stack used for North. EOS = <|return|> (200002)."""
from __future__ import annotations
import os
import glob, re, os, json
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache

_DEFAULT = sorted(glob.glob(os.environ.get("GPTOSS_MODEL", "/Users/pierrelamy/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/*/")))
_FINAL_RE = re.compile(r"final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.S)


class GptOssAdapter:
    name = "gpt-oss-20b"

    def __init__(self, model_path=None, fast_effort="low", reason_effort="high"):
        mp = model_path or _DEFAULT[0]
        self.model, self.tok = load(mp)
        # harmony has MULTIPLE stop tokens (<|return|>, <|endoftext|>, <|call|>); stop on ALL of them
        gc = {}
        for p in glob.glob(os.path.join(mp, "generation_config.json")):
            gc = json.load(open(p))
        eid = gc.get("eos_token_id", self.tok.eos_token_id)
        self._stops = set(eid if isinstance(eid, list) else [eid]) | {self.tok.eos_token_id}
        self.eos = self.tok.eos_token_id          # the single final-turn eos (used by the tool-dispatch loop)
        self.fast_effort = fast_effort
        self.reason_effort = reason_effort

    def _step(self, ids, cache):
        return int(mx.argmax(self.model(mx.array([ids], dtype=mx.int32), cache=cache)[0, -1, :]).item())

    def _gen(self, prompt, effort, maxn):
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, reasoning_effort=effort)
        cache = make_prompt_cache(self.model)
        nxt = self._step(list(ids), cache); out = []
        for _ in range(maxn):
            if nxt in self._stops:
                break
            out.append(nxt); nxt = self._step([nxt], cache)
        raw = self.tok._tokenizer.decode(out, skip_special_tokens=False)
        m = _FINAL_RE.search(raw)
        return (m.group(1) if m else raw), len(out)

    def encode_fast(self, messages):
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, reasoning_effort=self.fast_effort)

    def gen_fast(self, prompt, maxn=1800):
        return self._gen(prompt, self.fast_effort, maxn)

    def gen_reasoning(self, prompt, maxn=4000):
        return self._gen(prompt, self.reason_effort, maxn)


if __name__ == "__main__":
    a = GptOssAdapter()
    code, n = a.gen_fast("Write a Python function `gcd(a, b)` returning the greatest common divisor. "
                         "Return ONLY the function in a ```python code block.")
    print(f"gen_fast: {n} tok\n{code[:300]}")
