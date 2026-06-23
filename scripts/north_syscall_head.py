#!/usr/bin/env python3
"""Bridge-expert SYSCALL head on North — the learned call-emission policy from inside the weights.

The neural-scheduler synthesis established the split: HALT is RL-grade (SFT can't teach token-level
self-termination), but the SYSCALL/trap signal is a *classifier head on the residual* — proven in the
toy (Scenario D: 100% detect, 512/512). The identity-init gateway (north_gateway_invariants.py, proven
0.0 at L24) is the substrate the noop/bridge expert rides on. This builds its control output: a head
that reads North's L24 residual and decides, IN-WEIGHTS, whether the current context needs a trap to a
hold-out-proven det-block (and which one) — replacing the brittle text-based [[fn]] emission with a
learned decision. "Learned proposes (this head), kernel disposes (the dispatch loop in tool_dispatch)."

Run: north_syscall_head.py [--layer 24] [--n-per-class 64]
"""
from __future__ import annotations
import os
import argparse, random
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load

MODEL = os.environ.get("NORTH_MODEL", "/Users/pierrelamy/.lmstudio/models/bsisduck/North-Mini-Code-1.0-MLX-MXFP8")

# the det-block tool classes the syscall head routes to (0 = no trap, answer in-model)
CLASSES = ["none", "count_byte", "c_gcd", "is_prime", "shortest_dist"]


# -------------------------------------------------------------------- dataset (varied natural prose)
# Training distribution mixes, per class, KEYWORD-explicit AND keyword-free OBLIQUE phrasings; and the
# negatives mix plain chat with HARD NEGATIVES (prose carrying the trap keywords count/gcd/prime/path but
# needing no computation). This teaches the head the *concept* (needs-exact-computation) rather than the
# surface lexicon. NOTE: these training examples are disjoint from the held-out adversarial test corpus.
def build_dataset(n_per_class, seed=0):
    rng = random.Random(seed)
    words = ["strawberry", "mississippi", "banana", "assessment", "bookkeeper", "committee",
             "raspberry", "tennessee", "balloon", "parallel", "millennium", "successfully",
             "mammal", "monsoon", "lattice", "pineapple", "vacuum", "rhythm"]
    letters = list("abcdefglmnoprstuvy")
    chat = [
        "Write a short poem about the {x}.", "What do you think about {x} as a hobby?",
        "Explain how {x} works in simple terms.", "Give me three tips for staying {x}.",
        "Summarize the plot of a story about a {x}.", "What's a good name for a {x}?",
        "Describe the feeling of a {x} morning.", "Tell me a fun fact about {x}.",
        "How should I plan a {x} for the weekend?", "Recommend a book about {x}.",
        "I have a few {x} and want to share them with friends — any ideas?",
        "Compare {x} and the alternative in a friendly way.",
    ]
    chat_fill = ["sea", "gardening", "rainbows", "calm", "dragon", "puppy", "autumn", "honey",
                 "trip", "space", "apples", "tea", "music", "kindness", "the city", "old films"]
    # HARD NEGATIVES: carry a trap keyword but need NO computation -> none (disjoint from test set)
    hard_neg = [
        "I can always count on my sister when things get hard.",
        "The account balance looked healthy this month.", "Give an honest account of what happened.",
        "The greatest film of all time, in my opinion, is a quiet one.",
        "We discovered we had a lot of common interests.", "The common cold always ruins my week.",
        "GCD Holdings reported strong revenue this quarter.",
        "We ordered prime rib for the family dinner.", "The prime suspect was released yesterday.",
        "She is in the prime of her career right now.", "Primetime shows are all reruns lately.",
        "A narrow garden path winds between the roses.", "Whoever draws the shortest straw cleans up.",
        "You have to find your own path in life.", "The shortest player still scored the most.",
        "Let's divide the room into two friendly teams.", "A great divide separated the two towns.",
        "The distance between old friends can fade fast.", "Count me in for the road trip!",
        "That meeting was the shortest one we've had all year, thankfully.",
    ]
    data = []
    for _ in range(n_per_class):
        w = rng.choice(words); c = rng.choice(letters)
        data.append((rng.choice([
            f"How many times does the letter '{c}' appear in the word '{w}'?",
            f"Count the '{c}'s in {w} for me, exactly please.",
            f"In a quick note, tell me exactly how many '{c}' characters are in \"{w}\".",
            f"I'm teaching spelling — how many '{c}' letters does {w} have?",
            # oblique (keyword-free)
            f"Tally up the '{c}' glyphs living in '{w}'.",
            f"How frequently does '{c}' surface within '{w}'?",
            f"Give the exact head-count of '{c}' inside '{w}'.",
        ]), 1))
        a, b = rng.randint(6, 999), rng.randint(6, 999)
        data.append((rng.choice([
            f"What is the greatest common divisor of {a} and {b}?",
            f"Reduce the fraction {a}/{b} — I need the gcd.",
            f"Find gcd({a}, {b}) and use it in your explanation.",
            f"What's the largest number that divides both {a} and {b}?",
            # oblique
            f"Boil the fraction {a}/{b} down to simplest form.",
            f"Biggest whole number dividing both {a} and {b} evenly?",
            f"Simplify the ratio {a}:{b} as far as it goes.",
        ]), 2))
        p = rng.randint(2, 9999)
        data.append((rng.choice([
            f"Is {p} a prime number?", f"Tell me whether {p} is prime, exactly.",
            f"I need to know if {p} has any divisors other than 1 and itself.",
            f"List the prime numbers up to {p % 100 + 20}.",
            # oblique
            f"Does {p} break apart into smaller whole-number factors?",
            f"Is {p} only divisible by one and itself?",
            f"Can {p} be split into equal whole groups bigger than one?",
        ]), 3))
        n1, n2 = rng.randint(0, 7), rng.randint(0, 7)
        data.append((rng.choice([
            f"In my weighted graph, what's the shortest path distance from node {n1} to node {n2}?",
            f"Compute the minimum total distance from {n1} to {n2} over the road network.",
            f"What is the cheapest route cost between vertices {n1} and {n2}?",
            f"Find the shortest-path length from {n1} to {n2} in the graph I described.",
            # oblique
            f"Cheapest hop sequence from node {n1} to node {n2}?",
            f"Minimum summed edge weight from vertex {n1} to vertex {n2}?",
            f"Least costly route between stops {n1} and {n2}?",
        ]), 4))
        # negatives: half plain chat, half HARD negatives
        if rng.random() < 0.5:
            data.append((rng.choice(chat).format(x=rng.choice(chat_fill)), 0))
        else:
            data.append((rng.choice(hard_neg), 0))
    rng.shuffle(data)
    return data


# ----------------------------------------------------------------- North L24 residual feature extract
def run_to_layer(model, inputs, L):
    inner = model.model
    x = inner.embed_tokens(inputs)
    for li, layer in enumerate(inner.layers):
        win = inner.window_size if layer.self_attn.use_sliding_window else None
        mask = create_attention_mask(x, None, window_size=win)
        x = layer(x, mask, None)
        if li == L:
            break
    return x[:, -1, :]  # last-token residual at layer L = the in-weights decision point


def featurize(model, tok, data, L):
    feats, labels = [], []
    for i, (prompt, lab) in enumerate(data):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
        h = run_to_layer(model, mx.array(np.array([ids], dtype=np.int32)), L).astype(mx.float32)
        mx.eval(h)
        feats.append(np.array(h[0], dtype=np.float32)); labels.append(lab)
        if (i + 1) % 40 == 0:
            print(f"  featurized {i+1}/{len(data)}", flush=True)
    return np.stack(feats), np.array(labels, dtype=np.int32)


# ----------------------------------------------------------------------------- the syscall head (MLP)
class SyscallHead(nn.Module):
    def __init__(self, d, k, hidden=128):
        super().__init__()
        self.norm = nn.RMSNorm(d, eps=1e-6)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, k)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(self.norm(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=24)   # gateway layer (nL//2), invariant-proven
    ap.add_argument("--n-per-class", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", default="/tmp/north_syscall_head.safetensors")   # distinct path to keep the L24 head
    args = ap.parse_args()

    print(f"[load] {MODEL}", flush=True)
    model, tok = load(MODEL); model.eval()
    d = int(model.args.hidden_size)
    data = build_dataset(args.n_per_class)
    ntest = max(1, len(data) // 5)
    train, test = data[ntest:], data[:ntest]
    print(f"[data] {len(data)} prompts ({len(CLASSES)} classes), train {len(train)} / test {len(test)}", flush=True)

    print(f"[extract] North L{args.layer} residuals (d={d})...", flush=True)
    Xtr, ytr = featurize(model, tok, train, args.layer)
    Xte, yte = featurize(model, tok, test, args.layer)

    # normalize features (per-dim z-score from train stats)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd

    head = SyscallHead(d, len(CLASSES))
    opt = optim.Adam(learning_rate=1e-3)
    Xtr_m, ytr_m = mx.array(Xtr), mx.array(ytr)

    def loss_fn(h, X, y):
        return nn.losses.cross_entropy(h(X), y, reduction="mean")

    lvg = nn.value_and_grad(head, loss_fn)
    print("[train] syscall head...", flush=True)
    for ep in range(args.epochs):
        loss, grads = lvg(head, Xtr_m, ytr_m)
        opt.update(head, grads); mx.eval(head.parameters(), opt.state)
        if (ep + 1) % 60 == 0:
            print(f"  epoch {ep+1}: loss {float(loss):.4f}", flush=True)

    # ---- eval: detection (trap vs no-trap) + tool-routing accuracy on held-out ----
    pred = np.array(mx.argmax(head(mx.array(Xte)), axis=1))
    routing_acc = float((pred == yte).mean())
    trap_true, trap_pred = (yte != 0), (pred != 0)
    detect_acc = float((trap_true == trap_pred).mean())
    # confusion among trap classes
    tp = int(((trap_true) & (trap_pred)).sum()); fp = int(((~trap_true) & (trap_pred)).sum())
    fn = int(((trap_true) & (~trap_pred)).sum())
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0

    print("\n===== SYSCALL HEAD (bridge-expert call-emission policy) =====", flush=True)
    print(f"  layer L{args.layer} residual -> head -> {len(CLASSES)}-way {{none, {', '.join(CLASSES[1:])}}}")
    print(f"  trap-detection accuracy (trap vs no-trap): {detect_acc*100:.1f}%  (prec {prec*100:.1f} / rec {rec*100:.1f})")
    print(f"  tool-routing accuracy (exact class):       {routing_acc*100:.1f}%")
    print(f"  test n={len(yte)}; majority-class baseline={max(np.bincount(yte))/len(yte)*100:.1f}%")

    # save head + norm stats so the runtime can load it
    flat = {f"head.{k}": v for k, v in nn.utils.tree_flatten(head.parameters())}
    flat["mu"] = mx.array(mu); flat["sd"] = mx.array(sd)
    flat["layer"] = mx.array(np.array([args.layer], dtype=np.int32))   # record which layer this head reads
    mx.save_safetensors(args.out, flat)
    print(f"  saved -> {args.out}  (layer L{args.layer}, routing {routing_acc*100:.1f}%, detect {detect_acc*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
