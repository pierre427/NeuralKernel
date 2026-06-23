# Build the Neural Microkernel from scratch — step by step

This guide is written to be followed by **either a human or an autonomous LLM agent**. Every step
states its tier, the exact command, and the expected output so success/failure is unambiguous.
Where a command needs the live model it is marked **[Tier B]**; CPU-only steps are **[Tier A]**.

- **Tier A** — CPU, no model, no network. Reproducible from this zip alone. ~30 s.
- **Tier B** — Apple Silicon + MLX + the North weights. Reproduces the bit-exact forward-pass results.
- **Tier C** — (optional) retrain the small artifacts the kernel uses (the syscall head).

Conventions: all commands are run from the bundle root unless stated. `$ROOT` = the unzipped
`neural-microkernel-bundle/` directory. The import root is always `scripts/` — set
`PYTHONPATH="$ROOT/scripts"` for any direct module run.

---

## 0. Prerequisites

| | Tier A (CPU) | Tier B (live model) |
|---|---|---|
| OS | any (Linux/macOS/WSL) | macOS 14+ on **Apple Silicon** (M-series) |
| Python | ≥ 3.9 (verified 3.9.6) | ≥ 3.9 |
| Packages | `numpy` (`psutil`, `regex` optional) | `+ mlx>=0.31`, `mlx-lm>=0.31` |
| Disk | ~5 MB (this bundle) | + ~30 GB (North weights) |
| Network | none | once, to download the weights |

MLX is Apple-Silicon-only. Tier A runs anywhere; Tier B requires an M-series Mac because the model
runs through Apple MLX. There is no CUDA path in this bundle.

---

## 1. [Tier A] Unpack and verify the deterministic core

```bash
unzip neural-microkernel-bundle.zip
cd neural-microkernel-bundle
python3 -m pip install numpy           # only hard CPU dependency
bash verify.sh
```

**Expected output** — a 24-row PASS table ending in:

```
RESULT: 28/28 suites passed.
Tier A reproduction OK.
```

If you prefer to run the suites by hand (identical to what `verify.sh` does):

```bash
cd scripts
export PYTHONPATH="$PWD"
for t in kernel/test_*.py; do python3 "$t"; done
```

Each suite prints its own `GREEN: N/N … passed`. These suites are the *specification* of the
kernel's contracts — if you are reimplementing the kernel in another language, port the suites
first; they pin down the admission gates, the hold-out gate's hostile-candidate hardening, the
sandbox's verdict-in-parent rule, the escalation ladder order, and the memory cap/eviction.

**What you just proved (no model involved):**
- Fail-closed admission — a task missing *operation*, *capability*, or *validator* is rejected
  (`test_task_graph`, `test_tools`).
- The hold-out gate rejects a degenerate generator, a hostile `__eq__`, and a `BaseException`-raising
  candidate (`test_holdout_gate`).
- The escalation ladder visits tiers in order and abstains honestly at the end (`test_escalate_solve`).
- The memory store never exceeds its byte cap and never evicts the pinned governance object
  (`test_memory_store`).

---

## 2. [Tier A] Read the architecture (so the rest makes sense)

Two documents, in this order:

1. [`docs/neural-microkernel-paper.md`](docs/neural-microkernel-paper.md) — the *what and why*:
   three planes, the three-gate contract, the forward-pass mechanisms, the self-extending plane, the
   escalation solver, and the evidence. Read §2 (architecture) and §5 (validation) at minimum, and
   §6 (limitations) before quoting any result.
2. [`docs/neural-microkernel-design.md`](docs/neural-microkernel-design.md) — the *how to build*:
   the staged **M0–M5** plan, the kernel lifecycle, the trust boundary, and the glossary. The build
   order below (§4) follows its milestones.

[`MANIFEST.md`](MANIFEST.md) maps every file in the bundle to a plane and a one-line purpose. Use it
as the index when you go to read or modify code.

---

## 3. [Tier B] Bring up the live model

### 3.1 Install MLX

```bash
python3 -m pip install -r requirements.txt    # installs mlx + mlx-lm (and numpy)
python3 -c "import mlx.core as mx; print('mlx ok', mx.default_device())"
```

### 3.2 Get the North weights (~30 GB)

The model is **North-Mini-Code-1.0-MLX-MXFP8**, a Cohere2-MoE checkpoint quantized to MXFP8 for MLX.
Download it from Hugging Face (any of these works):

```bash
# option A — huggingface-cli
python3 -m pip install huggingface_hub
huggingface-cli download bsisduck/North-Mini-Code-1.0-MLX-MXFP8 \
    --local-dir ~/models/North-Mini-Code-1.0-MLX-MXFP8

# option B — LM Studio: search "North-Mini-Code", download the MLX MXFP8 build.
#   It lands under ~/.lmstudio/models/bsisduck/North-Mini-Code-1.0-MLX-MXFP8
```

### 3.3 Point the adapter at your weights

Every Tier-B script reads the model location from the **`NORTH_MODEL`** environment variable, so you
configure it in **one place, no source edit required**:

```bash
export NORTH_MODEL="/path/to/North-Mini-Code-1.0-MLX-MXFP8"
```

`scripts/north_adapter.py` defines `NORTH = os.environ.get("NORTH_MODEL", "<the author's default>")`,
and `NorthAdapter`, the invariant batteries, the syscall head, and every `*_north.py` runner all
resolve the path through it — set `NORTH_MODEL` once and the whole Tier B uses your weights. (If you
prefer, you can still edit the default string in `north_adapter.py`.) The optional gpt-oss adapter
reads `GPTOSS_MODEL` the same way. Then smoke-test the adapter:

```bash
cd scripts && export PYTHONPATH="$PWD"
python3 -c "from north_adapter import NorthAdapter; a=NorthAdapter(); print(a.gen_fast('return the integer 7 only', maxn=8))"
```

You should see the model load and emit a short completion. The adapter exposes the kernel's neural
ABI: `gen_fast` (greedy/T=0), `gen_reasoning` (reasoning on), and `gen_sample(temp, seed)` (the
temperature-diverse decode the escalation ladder needs — at T=0 retries are byte-identical, so
diversity *requires* sampling).

> Other adapters ship too (`gptoss_adapter.py`, and the design references `mellum_adapter`): the
> kernel is model-agnostic behind this ABI. The bit-exact invariants below assume the MoE structure
> of North.

---

## 4. [Tier B] Reproduce the forward-pass invariants (residual ≡ 0)

These are the headline results: each mechanism is checked to **bit-exactness** against the live
model — a single differing element fails the battery. Run them from `scripts/` with
`PYTHONPATH=$PWD`. Each loads the model once and prints a `== 0` / `PASS` verdict; reference proof
JSON from the original run is in `scripts/kernel/logs/`.

```bash
cd scripts && export PYTHONPATH="$PWD"

python3 kernel/inv_trap_plane.py          # no-op trap plane @ sites {0,8,16,24,32,40,48}: all == 0
python3 kernel/inv_kv_park.py             # park@{1,8,35,69} resume == gen_fast, token-for-token
python3 kernel/inv_switch_isolation.py    # multi-lane interleaved == solo, each lane == gen_fast
python3 kernel/inv_speculate_rollback.py  # park -> speculate k -> rollback == straight-through
python3 kernel/inv_det_interrupt.py       # gen_constrained(None) == gen_fast; armed mask changes output
python3 kernel/inv_rollback_ringcross.py  # rollback across the >4096 KV ring-buffer boundary == 0
```

Each should report **0 mismatches**. What each one demonstrates, in plain terms:

| Battery | Mechanism | The claim it proves |
|---------|-----------|---------------------|
| `inv_trap_plane` | trap sidecars at 7 layer sites | a disarmed/identity trap perturbs nothing |
| `inv_kv_park` | KV-cache park/resume | a paused generation resumes bit-identically |
| `inv_switch_isolation` | switch-MLP expert isolation + multi-lane scheduling | context-switching N lanes ≡ running each alone |
| `inv_speculate_rollback` | speculate / verify / rollback | a discarded draft leaves no residue in cache/RNG |
| `inv_det_interrupt` | deterministic in-decode interrupt | masking forbidden tokens is exact and inert when off |
| `inv_rollback_ringcross` | ring-cross rollback | restore works even past the 4096-token sliding-window wrap (the hard case) |

The mixed cache stack matters: North uses one full-attention `KVCache` plus 36 sliding
`RotatingKVCache(4096)` layers. Post-wrap layers are **not trimmable**, so park/resume snapshots
`(.state, .meta_state)` by deep copy rather than trimming — see `kernel/kv_park.py`.

---

## 5. [Tier B] The commit trap and the entry trap

### 5.1 Commit trap — the only output path

```bash
python3 kernel/run_commit_kernel.py --n 10 --maxn 400
```

`CommitKernel.gen_committed` decodes a candidate, then routes it through `CommitTrap.commit` against
the task's contract along `syntax < schema < property < ORACLE` (the ORACLE tier *runs the code on
the task's real cases*). A correct candidate is emitted with a sealed `ProofLedger` entry; a wrong
one is **caught, not emitted**, and the failure is fed back for a re-decode (a real input change, not
a T=0 re-roll). Expected: every emitted answer is ORACLE-proven; nothing unverified escapes.

### 5.2 Entry trap — the learned classifier writes capabilities **[Tier B + a trained head]**

The entry trap reads the *frozen* L0 residual (no mutation → `gen_fast` stays bit-identical),
classifies the request with a small learned **syscall head**, and stamps an authorized
`trap:<class>` capability + one trap budget into the continuation frame. The learned component
*proposes* the capability; the Phase-0a capability gate *disposes*.

It needs a trained head file. Train one at the L0 site (Tier C, ~minutes on the Mac):

```bash
python3 north_syscall_head.py --layer 0 --out /tmp/nsh_L0.safetensors
```

Then run the trap:

```bash
python3 kernel/run_entry_trap.py --head /tmp/nsh_L0.safetensors
```

Expected: classifications at high confidence with the correct `trap:<class>` capability stamped, and
a read-inert check confirming the L0 read did not perturb the generation. (The L24 "gateway" head —
`--layer 24`, the invariant-proven midpoint — is the original; L0 is the "even safer" entry site at
~98% routing.)

---

## 6. [Tier B] The self-extending plane and the escalation solver

This is the part that grows new *proven* capabilities at runtime without ever trusting model output.

### 6.1 The hold-out gate (the kernel's notion of proof)

`kernel/holdout_gate.py` runs a candidate function on `n` freshly generated random instances and
accepts **only if every output matches an independent oracle**, with three safeguards: an asymmetric
trust boundary (generator + oracle come from the gap spec, the model supplies *only* the candidate,
the verdict is computed in the parent), a **novelty floor** (counts distinct instances; a degenerate
generator is rejected), and a **placebo control** (re-runs under a second seed; fail→pass-by-luck is
never accepted). It is hardened against hostile candidates (type-gated leaf comparison; catches
`BaseException`). The CPU suite `test_holdout_gate.py` proves all of this without a model.

### 6.2 Sandboxed candidate execution

`kernel/venv_tool.py` runs each candidate in a subprocess under a macOS `sandbox-exec` profile:
no network, no writes outside a scratch dir, its own process group (fork-bomb → `killpg`), CPU/memory
rlimits, bytecode writing off. The child computes **raw outputs only**; the parent generates
instances, runs the oracle, and compares — so even a fully malicious candidate can at most return
wrong numbers, which the gate rejects. It cannot forge a verdict because it never holds one.

### 6.3 Run the escalation solver on the two previously-failed tasks **[Tier B]**

```bash
cd scripts && export PYTHONPATH="$PWD"
python3 kernel/escalate_solve.py --task both --n 2000
```

This re-attempts `to_snake_case` (w47) and a Redis-RESP parser (c156), each judged by the hold-out
gate at 2000 novel instances per seed across two seeds with a placebo control. The ladder is:

| Tier | Strategy |
|-----:|----------|
| 1 | greedy, T=0 — one deterministic attempt |
| 2 | three diverse attempts at T=0.8 |
| 3 | repair each, fed its own held-out failure |
| 4 | research: `web_search` + `fetch_page`, then build from the references |
| 5 | abstain: *"I'm sorry, I don't know how to do that."* |

Expected (matches the paper §5.4): **w47 solves at tier 1**, **c156 at tier 2**; both pass
2000/2000 on two seeds. Every gate-proven solution is appended to
`scripts/kernel/learned_primitives.json` and registered for reuse. The oracles/generators for these
tasks live in `kernel/solve_failed_challenges.py` (the trusted source of truth — the model never sees
them).

### 6.4 Grow your own primitive

To extend the kernel with a new primitive, define a `Gap` (entry name, NL description, sample cases,
and *generator + oracle source*) and drive `SelfExtender.gate_candidate` (`kernel/self_extend.py`).
On a clean sweep — sample-test → hold-out gate → unsandboxed cross-check → second-seed re-gate →
placebo — the primitive is registered into the scheduler and becomes both callable and discoverable.
The CPU suite `test_self_extend.py` shows the full loop with a stub model.

---

## 7. [Tier B] The coding benchmark through the kernel

To confirm the kernel additions preserve end-to-end coding ability, route the benchmark *through*
the kernel on the live model:

```bash
python3 kernel/run_coding_kernel.py --model north
```

The paper reports **418/420** across five suites (warmups 49/50, ladder 100/100, challenge 199/200,
expert 50/50, property 20/20) with **zero trace self-check violations**. Telemetry is written under
`reports/telemetry/`; audit it with:

```bash
python3 kernel/analyze_telemetry.py
```

> The benchmark *fixtures* (the 420 task definitions) are part of the larger source repo, not this
> bundle. If you only have the bundle, the kernel mechanisms above (§4–§6) are the reproducible core;
> the benchmark needs the fixture suites from the repo. The `scripts/kernel/novel500/domains/*.json`
> set is included as a smaller, self-contained novel-instance evaluation corpus.

---

## 8. [Tier A] The persistent memory

`kernel/memory_store.py` is a SQLite-WAL, single-file, hard-bounded (default 100 MB) store reached
through the scheduler as the capability-gated `op:memory`. Every write is byte-accounted; over-cap
writes evict LRU **non-pinned** entries, and a pinned, never-evicted **governance object** holds the
policy in plain language and is always returned first by `consult()`. Exercise it on CPU:

```bash
cd scripts && export PYTHONPATH="$PWD" && python3 kernel/test_memory_store.py
```

Expected: `GREEN: 20/20 memory-store checks passed`, including "governance object intact after
concurrent hammering" and the hard-cap/eviction invariants.

---

## 9. Build order if you are reimplementing (the M0–M5 ladder)

From `docs/neural-microkernel-design.md` §3, the staged plan — each milestone is independently
testable, and lower milestones are the CPU suites you already ran:

- **M0** — scheduler owns I/O + tools (capability system). → `task_graph.py`, `tools*.py`.
- **M1** — post-last-layer **commit trap** as the only output path. → `commit_trap.py`, `commit_kernel.py`.
- **M2** — clean-context agents: hold/switch/resume over a continuation frame. → `frame.py`, `scheduler.py`, `kv_park.py`.
- **M3** — post-L0 **entry trap** (identity, bit-identical) + learned syscall head. → `entry_trap.py`, `north_syscall_head.py`.
- **M4** — mid-layer deterministic interrupts. → `trapped_adapter.py`, `north_det_block.py`, `inv_det_interrupt.py`.
- **M5** — scheduler-owned dynamic-k (the in-kernel scheduler *decision*). **Policy core BUILT + verified on
  North** → `k_policy.py` (`KPolicy`), `k_cost.py` (sysinfo→cost + budget accounting), `moe_entropy.py` (the
  inert gate-entropy tap), wired into `escalate_solve.py`; CPU suites `test_k_policy` / `test_k_cost` /
  `test_k_policy_ladder`; **tuning & usage in [README §5](README.md).** *(The literal raw-Metal GPU mega-kernel
  fusion is still the open perf milestone.)*
- **Fact-loop** — the **M5-governed verified-fact re-prefill** (capability multiplier) + its **live dispatch** are
  **BUILT + verified on North** → `kernel/fact_loop.py` (`governed_fact_loop`: compute a verified fact with a
  hold-out-proven det primitive, deliver it AS TOKENS, re-prefill — thin by default, spend only when the base
  fails validation and cost/budget allow; ~9.5× lift on a count task), `kernel/fact_dispatch.py` (`FactRouter` +
  `default_router` + `dispatch` + `register_fact_loop_executor` wiring `op:fact_loop` capability-gated into the
  Scheduler); CPU suites `test_fact_loop` / `test_fact_dispatch`; **tuning & usage in [README §6](README.md).**
  This is the realized Mode-A channel (facts as tokens); Mode-B residual injection stays open.

The honest status (paper §2.4, §6): M1–M4 mechanisms are **proven in isolation to residual ≡ 0** on
the live model; the integrated runtime still schedules at whole-generation granularity. M5's
*policy-fusion core* (scheduler-owned dynamic-k) is now built + verified on North (README §5); the
*literal raw-Metal GPU mega-kernel fusion* and **Mode-B mid-stack residual injection** remain the two
genuinely open items.

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `ModuleNotFoundError: north_adapter` | run from `scripts/` with `PYTHONPATH="$PWD"`; the import root is `scripts/`. |
| `ModuleNotFoundError: mlx` on Linux/Intel | MLX is Apple-Silicon-only; Tier B needs an M-series Mac. Tier A still runs everywhere. |
| Model fails to load / path error | edit `scripts/north_adapter.py` line 10 (`NORTH = …`) to your weights dir (§3.3). |
| `inv_*` battery reports a nonzero mismatch | check MLX/mlx-lm versions ≥ 0.31 and that the weights are the **MXFP8** build; bit-exactness is version-sensitive. |
| `run_entry_trap.py` can't find the head | train it first: `python3 north_syscall_head.py --layer 0 --out /tmp/nsh_L0.safetensors` (§5.2). |
| `psutil`-related warning from `sysinfo` | optional; `pip install psutil` to enable the fast path, or ignore (it degrades gracefully). |
| Coding benchmark missing fixtures | the 420-task fixtures live in the source repo, not this bundle (§7). |

---

## 11. For an autonomous LLM agent — minimal runbook

```bash
# Tier A — always do this first; it must end in 28/28.
cd neural-microkernel-bundle && python3 -m pip install numpy && bash verify.sh
#   SUCCESS iff stdout contains "RESULT: 28/28 suites passed."
#   (19 core + 3 M5: k_policy, k_cost, k_policy_ladder + 2 fact-loop: fact_loop, fact_dispatch)

# Tier B — only on Apple Silicon, only after editing north_adapter.py line 10.
cd scripts && export PYTHONPATH="$PWD"
python3 -m pip install -r ../requirements.txt
for b in inv_trap_plane inv_kv_park inv_switch_isolation inv_speculate_rollback inv_det_interrupt inv_rollback_ringcross; do
  python3 "kernel/$b.py" || echo "BATTERY FAILED: $b"
done
#   SUCCESS iff every battery reports 0 mismatches.
python3 kernel/escalate_solve.py --task both --n 2000
#   SUCCESS iff w47 solves @ tier 1 and c156 @ tier 2, both 2000/2000 x2 seeds.

# M5 — scheduler-owned dynamic-k (tuning in README §5).
python3 kernel/inv_moe_entropy.py        #   SUCCESS iff RESULT: PASS (entropy tap ==0-inert vs gen_fast)
python3 kernel/m5_dynamic_k_north.py     #   SUCCESS iff the disposed k adapts idle/busy/budget; thin on T1-pass

# Fact-loop — the M5-governed verified-fact capability multiplier + live dispatch (tuning in README §6).
python3 kernel/run_fact_loop_north.py    #   SUCCESS iff the A/B PASSES: governed_idle lifts acc near always_ground
                                         #   at avg_gens in (1,2) (thin), and cost=1.0 vetoes the spend.
python3 kernel/run_fact_dispatch_north.py #  SUCCESS iff routing+inertness hold, the hard fact lifts, cost vetoes,
                                         #   and op:fact_loop runs through the Scheduler capability-gated (fail-closed).
```

Decision rule for the agent: **never report a correctness result that was not computed by the
deterministic plane.** If a battery does not print 0 mismatches, or a gate does not pass, report the
failure verbatim — do not infer success from the model's own text. That rule *is* the kernel.
