#!/usr/bin/env python3
"""Mode-B B1: a TOY CPU proof that a residual edit can STEER a downstream readout to an injected result — the
mechanism B2 needs, de-risked before training on North. No North; a tiny frozen net + the real ModeBInjector.

Setup: a FROZEN random base  h = silu(x @ W1);  logits = h @ Wr  (the base output does NOT depend on the result
r). We insert the ModeBInjector at the hidden boundary: h' = h + injector(h, r), and train ONLY the injector to
make readout(h') predict the class encoded by r. Then on HELD-OUT x we inject one-hot(c) for each class c and ask
whether the output follows c — i.e. the output TRACKS the injected result (the causal claim).

Controls (the project's discipline — "fail->pass is not proof"):
  * REAL:     r is informative (target == r's class).         expect held-out steering accuracy >> chance.
  * PLACEBO:  shuffled labels (target decoupled from r).       expect ~chance (no real steering).
  * DISABLED: up==0 -> readout(h') == readout(h) exactly.      the ==0 substrate (no steering when off).
A pass requires REAL >> PLACEBO≈chance AND disabled==identity, so the steering is real AND result-conditioned.

Run: /Users/pierrelamy/nm-mlx-venv/bin/python finetune/gpt-oss-unsloth/scripts/kernel/mode_b_toy.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from kernel.mode_b import ModeBInjector

CHECKS = []
def ok(c, l): CHECKS.append((bool(c), l)); print(f"  {'OK ' if c else 'XX '} {l}")

D, C = 32, 8                 # hidden dim, number of result classes (result = one-hot over C)
NTR, NTE, STEPS, LR = 1024, 512, 600, 1e-2


def _data(n, seed, shuffle_labels=False):
    mx.random.seed(seed)
    x = mx.random.normal((n, D))
    cls = (mx.arange(n) % C)                                  # balanced classes
    r = mx.eye(C)[cls]                                        # one-hot result vector
    y = mx.array((mx.random.permutation(n) % C)) if shuffle_labels else cls   # placebo: target decoupled from r
    return x, r, y


def _train(W1, Wr, shuffle_labels):
    inj = ModeBInjector(D, bottleneck=32, result_dim=C)
    mx.eval(inj.parameters())
    opt = optim.Adam(learning_rate=LR)
    xtr, rtr, ytr = _data(NTR, seed=1, shuffle_labels=shuffle_labels)

    def loss_fn(model, x, r, y):
        h = nn.silu(x @ W1)
        hp = h + model.delta(h, r)                            # inject at the hidden boundary
        return nn.losses.cross_entropy(hp @ Wr, y, reduction="mean")

    lvg = nn.value_and_grad(inj, loss_fn)
    for _ in range(STEPS):
        loss, grads = lvg(inj, xtr, rtr, ytr)
        opt.update(inj, grads)
        mx.eval(inj.parameters(), opt.state)
    return inj


def _accuracy(inj, W1, Wr):
    """held-out: inject one-hot(c) and check the output follows c (output tracks the injected result)."""
    xte, rte, yte = _data(NTE, seed=2, shuffle_labels=False)
    h = nn.silu(xte @ W1)
    pred = mx.argmax((h + inj.delta(h, rte)) @ Wr, axis=-1)
    return float(mx.mean(pred == yte).item())


def main():
    mx.random.seed(0)
    W1 = mx.random.normal((D, D)) * (1.0 / D ** 0.5)
    Wr = mx.random.normal((D, C)) * (1.0 / D ** 0.5)
    mx.eval(W1, Wr)
    chance = 1.0 / C

    # base (no injection) accuracy at matching the injected class — should be ~chance (base ignores r)
    xte, rte, yte = _data(NTE, seed=2)
    base_acc = float(mx.mean(mx.argmax(nn.silu(xte @ W1) @ Wr, axis=-1) == yte).item())

    real = _train(W1, Wr, shuffle_labels=False)
    placebo = _train(W1, Wr, shuffle_labels=True)
    real_acc = _accuracy(real, W1, Wr)
    placebo_acc = _accuracy(placebo, W1, Wr)

    # disabled-identity: up==0 -> readout(h') == readout(h) exactly
    off = ModeBInjector(D, bottleneck=32, result_dim=C); mx.eval(off.parameters())
    h = nn.silu(xte @ W1)
    disabled_eq = float(mx.max(mx.abs((h + off.delta(h, rte)) @ Wr - h @ Wr)).item())

    print(f"\n  chance={chance:.3f}  base(no-inject)={base_acc:.3f}  REAL={real_acc:.3f}  PLACEBO(shuffled)={placebo_acc:.3f}", flush=True)
    ok(disabled_eq == 0.0, f"disabled (up==0) -> readout unchanged, ==0 ({disabled_eq:.1e})")
    ok(real_acc > 0.8, f"REAL injector STEERS the output to the injected result (acc {real_acc:.3f} > 0.8)")
    ok(placebo_acc < chance + 0.10, f"PLACEBO (shuffled labels) ~chance (acc {placebo_acc:.3f} < {chance+0.10:.3f})")
    ok(real_acc - placebo_acc > 0.5, f"REAL >> PLACEBO (gap {real_acc-placebo_acc:.3f} > 0.5) -> steering is real, result-conditioned")

    n = sum(1 for c, _ in CHECKS if c)
    print(f"\n{'GREEN' if n == len(CHECKS) else 'RED'}: {n}/{len(CHECKS)} mode_b toy-steering checks passed", flush=True)
    print("VERDICT: a residual edit CAN steer a downstream readout to an injected result (placebo-controlled) "
          "-> the Mode-B mechanism is viable in principle; B2 tests it on real North." if n == len(CHECKS)
          else "VERDICT: toy steering did NOT clear the controls — revisit before North.", flush=True)
    sys.exit(0 if n == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
