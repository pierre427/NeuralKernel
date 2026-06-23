# Neural Microkernel — build & run bundle

A self-contained export of the **neural microkernel**: an operating-system-style kernel wrapped
around a real Mixture-of-Experts language model that treats the model as an *untrusted proposer*
behind a *trusted, deterministic disposer*. One sentence:

> **The learned component proposes; a thin, general, trusted kernel disposes.**
> And the load-bearing corollary: **validation is a syscall** — whether a generated primitive is
> correct, or a claimed fact is grounded, cannot be adjudicated by the same weights that produced
> it. It requires an independent, deterministic authority *outside* the model.

This bundle contains everything needed to **build the kernel from scratch and run it**, with two
reproduction tiers: a CPU tier that runs from this zip alone (no model, no GPU, no network), and a
live-model tier that reproduces the bit-exact forward-pass results on Apple Silicon.

---

## 1. What this is (the 60-second version)

The kernel is organized into **three planes** mirroring a privilege boundary:

| Plane | Role | Trust | Lives in |
|-------|------|-------|----------|
| **Neural** | the model's forward pass — next-token logits, expert routing, generated text | *untrusted proposer* | `scripts/north_adapter.py` (+ forward-pass hooks) |
| **Scheduler** | a fail-closed task graph: admit → sequence → park → resume → validate → commit | *dispatcher* | `scripts/kernel/task_graph.py`, `orchestrator.py` |
| **Deterministic** | pure verifiable executors: parsers, validators, oracles, the hold-out gate | *only plane allowed to call a result correct* | `scripts/kernel/holdout_gate.py`, `commit_trap.py`, … |

Around that boundary the kernel adds:

1. **A fail-closed three-gate admission contract** — an operation runs only if it is *registered*,
   its *capability is granted*, and a *validator is installed*. Omit any one → rejected, not run.
2. **Forward-pass mechanisms**, each proven to **bit-exactness** (residual ≡ 0) against a live MoE:
   commit trap, KV park/resume, switch-MLP expert isolation, speculate/verify/rollback,
   deterministic interrupt, and ring-cross (>4096-token) rollback.
3. **A self-extending deterministic plane** — when a primitive is missing, the kernel has the model
   synthesize one and admits it **only after a hold-out gate** proves it on thousands of novel
   instances against an *independent oracle*, with a placebo control (fail→pass-by-luck is rejected).
   The verdict is computed in the trusted parent; the untrusted child never grades itself.
4. **An escalation solver** — greedy → temperature-diverse → repair → web-research → honest
   abstention, every rung judged by the same gate, every proven helper persisted.
5. **A persistent, hard-bounded, always-consulted memory** governed by a self-describing policy object.

The full argument, evidence, and limitations are in [`docs/neural-microkernel-paper.md`](docs/neural-microkernel-paper.md).
The build-level architecture and the staged M0–M5 plan are in [`docs/neural-microkernel-design.md`](docs/neural-microkernel-design.md).

---

## 2. Quickstart — Tier A (CPU, no model, ~30 seconds)

Requires only Python ≥ 3.9 and `numpy`. Reproduces the 24 deterministic test suites (19 core + 3 M5
scheduler-owned-dynamic-k: `k_policy`, `k_cost`, `k_policy_ladder` — see §5 — + 2 fact-loop
capability-multiplier: `fact_loop`, `fact_dispatch` — see §6).

```bash
unzip neural-microkernel-bundle.zip && cd neural-microkernel-bundle
python3 -m pip install numpy          # the only hard CPU dependency
bash verify.sh                        # runs all 28 suites, prints a PASS/FAIL table
```

Expected last line: `RESULT: 28/28 suites passed.  Tier A reproduction OK.`

This exercises the scheduler ABI and admission gates, the hold-out gate's hostile-candidate
hardening, the sandbox's verdict-in-parent design, the escalation ladder's tier order and
abstention, the memory store's hard cap and eviction, and the fail-closed registration of every
syscall-tier tool. Live counts at time of export:

| Suite | Checks | Suite | Checks | Suite | Checks |
|-------|:-----:|-------|:-----:|-------|:-----:|
| task_graph | 26 | lane_oracles | 43 | tools | 19 |
| orchestrator | 17 | trace_tool | 15 | self_extend | 14 |
| venv_tool | 14 | investigation | 13 | holdout_gate | 12 |
| memory_store | 20 | kv_park | 10 | agent_delegate | 10 |
| escalate_solve | 10 | entry_trap | 9 | web_tools | 8 |
| commit_kernel | 7 | scheduler | 4 | det_interrupt | 3 |
| commit_trap | (all) | **k_policy** (M5) | **34** | **k_cost** (M5) | **15** |
| **k_policy_ladder** (M5) | **11** | **fact_loop** (§6) | **43** | **fact_dispatch** (§6) | **43** |

For the live-model tier (forward-pass invariants, the coding benchmark, the escalation solver with
real candidates), follow [`BUILD-FROM-SCRATCH.md`](BUILD-FROM-SCRATCH.md).

---

## 3. What is in this bundle

```
neural-microkernel-bundle/
├── README.md                 ← you are here
├── BUILD-FROM-SCRATCH.md     ← detailed, numbered build+run guide (for an LLM or a human)
├── MANIFEST.md               ← file-by-file index, labelled by plane
├── requirements.txt          ← Tier A / Tier B dependency split
├── verify.sh                 ← one-shot CPU reproduction (the 28 suites)
├── docs/
│   ├── neural-microkernel-paper.md   ← the arXiv-style writeup (architecture + evidence + limits)
│   ├── neural-microkernel-paper.pdf
│   └── neural-microkernel-design.md  ← the build-level design doc + staged M0–M5 plan
└── scripts/                  ← set PYTHONPATH here; layout mirrors the source repo
    ├── north_adapter.py      ← the MLX model adapter (the neural plane) + 12 supporting modules
    ├── …
    └── kernel/               ← the kernel package (66 modules + tests)
        ├── __init__.py
        ├── task_graph.py  commit_trap.py  holdout_gate.py  self_extend.py  escalate_solve.py …
        ├── trap_plane.py  trapped_adapter.py  entry_trap.py  kv_park.py  scheduler.py  frame.py …
        ├── learned_primitives.json   ← primitives the kernel grew + hold-out-proved (artifact)
        ├── logs/                     ← the residual ≡ 0 invariant-battery proofs (artifact)
        └── novel500/                 ← the novel-instance evaluation set
```

The `scripts/` directory is a faithful copy of the source tree's import root: every `from
north_adapter import …` and `from kernel.commit_trap import …` resolves exactly as it does in the
original repo, with `PYTHONPATH=scripts`.

---

## 4. Honest scope (read this before quoting results)

The split below is stated the same way in the paper (§2.4, §5.1, §6) and is the single most
important thing to get right when describing the system:

- **Proven to bit-exactness, in isolation:** the six forward-pass invariant batteries report
  residual ≡ 0 on the live North model — trap plane, KV park, switch-MLP isolation,
  speculate/rollback, deterministic interrupt, and ring-cross rollback.
- **Integrated, but at whole-generation granularity:** the *running* loop schedules complete
  generations (it wraps `adapter.gen_fast`); it does not yet trap *every* forward pass in the hot
  path, and the in-model commit trap is exercised by its battery but not yet live-constructed inside
  the main loop. The M5 **scheduler-owned dynamic-k policy core is BUILT + verified** on North
  (`kernel/k_policy.py`, `k_cost.py`, `moe_entropy.py` — see §5); the *literal raw-Metal GPU
  mega-kernel fusion* is a documented **nice-to-have, deliberately not built** — its payoff is
  *structural only* (North is already GPU-bound, so no decode-speed multiplier) and it adds **no new
  capability or correctness** beyond the already-proven online==offline, at the cost of a raw-Metal
  systems build below MLX's surface. It is reserved for a future high-throughput *serving* regime or a
  *systems-paper* contribution; the in-API fusion (`mx.compile` 1.20×, `mx.fast.metal_kernel`) is the
  low-risk middle path if perf is ever wanted. See `docs/neural-microkernel-design.md` §M5.

Bridging "mechanism proven" → "mechanism in the hot path" is the central future-work item. Nothing
in this bundle requires you to take the model's word for any correctness claim: every PROVEN verdict
is computed by the deterministic plane, and every live-model claim above is reproducible by the
invariant batteries in Tier B.

---

## 5. M5 — scheduler-owned dynamic-k: tuning & usage

M5's policy-fusion core makes the **branching factor `k`** (how many continuations the scheduler runs at a
decision point — diverse drafts / tools / verifiers / agents) a *scheduler-owned, dynamic* decision instead of a
constant: **the model proposes route scores; the kernel disposes `k`.** Built + verified on real North (the
Tier-2 width adapts idle→`k=5` / busy→`k=1` / budget→`k=2`; thin by default when the first attempt passes).
Files: `kernel/k_policy.py`, `kernel/k_cost.py`, `kernel/moe_entropy.py`; CPU suites `test_k_policy` (34),
`test_k_cost` (15), `test_k_policy_ladder` (11) — all in `verify.sh`.

### 5.1 The non-negotiable tuning requirement — the uncertainty LEVER

**Do NOT drive `k` from the raw MoE gate-softmax entropy.** The calibration gate (`m5_calibrate_entropy.py` +
`m5_reanalyze_calib.py`) measured it on North and found it **INVERTED** (high routing entropy → the model is
*more* likely correct, AUC ≈ 0.89) and **single-suite** — so used naively ("high entropy → widen") it is
*backwards*. `KPolicy` therefore keeps `uncertainty_calibrated=False` for entropy; entropy is *observed/logged,
not trusted*. The **proven lever is validate-the-branch** — an *independent*, ground-truth signal such as a
branch that fails its hold-out gate — fed via `KSignals.disagreement` (the trusted-independent channel), or via
`uncertainty` **only** with `uncertainty_calibrated=True` from a genuinely calibrated source. ("fail→pass is not
proof," applied to routing.)

### 5.2 `KPolicy` knobs (`kernel/k_policy.py`)

| field | default | how to tune |
|---|---|---|
| `k_min` | 1 | floor; `1` = always run the chosen continuation, `0` = may stay in-model (the degenerate syscall-head policy) |
| `k_max` | 4 | hard ceiling — **anti-runaway-via-k**; the model cannot enlarge its own branching budget |
| `widen_threshold` | 0.5 | effective-uncertainty above which discretionary widening begins (generalizes the syscall head's conf ≥ 0.5) |
| `require_calibrated` | True | **honor the calibration trap** — a raw `uncertainty` widens `k` only if flagged calibrated. Leave `True`. |
| `cost_brake` | 0.6 | cost above which discretionary widening is braked linearly to 0 at cost = 1 |
| `risk_floor_at` | 0.5 | task risk above this mandates verify branches that **cost cannot brake away** (the confidently-wrong guard) |
| `unit_cost` | 1.0 | per-continuation cost charged to the trap budget |

Doctrine: **thin by default** (`k≈k_min`); **spend** (widen) only where *calibrated* uncertainty earns it **and**
cost allows; **risk floors mandatory verification** regardless of cost/confidence. `decide(KSignals)` returns a
`KDecision` with `k`, `explore_k`, `verify_k`, `used_signal` (audit), `reason`, and `est_cost`.

### 5.3 Cost model + sysinfo sensor (`kernel/k_cost.py`)

The cost input is the kernel's *own* load. `CostModel.cost_from_sysinfo(sysinfo)` → `cost ∈ [0,1]` via the
**binding-constraint** rule (`cost = max` pressure across host load-per-core, memory %used, and scheduler queue
depth — the scarcest resource governs). Knobs: `queue_soft_cap` (8), `load_saturate` (1.0 = load == cores),
`mem_floor` (0.5 = < 50% used → no pressure). **Requirement:** install `psutil` for memory pressure to register —
without it, memory contributes 0 and only load + queue drive cost. Budget: `charge_budget(decision, remaining,
unit_cost)` clamps `k` to what the budget affords (anti-runaway at the budget level, complementing `k_max`).

### 5.4 Wiring it in — the escalation ladder

`EscalatingSolver` (`kernel/escalate_solve.py`) is the reference integration: the Tier-2 fixed `k=3` becomes a
`KPolicy` decision, **additive and backward-compatible** (`k_policy=None` → the original fixed 3, so existing
behavior is byte-unchanged):

```python
from kernel.k_policy import KPolicy
from kernel.escalate_solve import EscalatingSolver

solver = EscalatingSolver(
    model,
    k_policy    = KPolicy(k_min=1, k_max=5, widen_threshold=0.5),
    importance  = 1.0,      # task importance scales discretionary spend
    trap_budget = 8.0,      # None = untracked; a float caps total k via charge_budget
    cost_source = None,     # None = live sysinfo; or a callable()->cost in [0,1] for tests/demos
)
```

There the uncertainty lever is the **Tier-1 hold-out-gate severity** (independent, ground-truth), cost is sysinfo,
and risk/importance are task-level — exactly the §5.1 discipline.

### 5.5 Verifying M5

- **CPU (Tier A):** `bash verify.sh` runs the 3 M5 suites alongside the core 19 (28/28 total, incl. the 2 fact-loop suites — §6).
- **Live model (Tier B, North):**
  `python kernel/inv_moe_entropy.py` (entropy tap is `==0`-inert vs `gen_fast`) ·
  `python kernel/m5_calibrate_entropy.py` then `python kernel/m5_reanalyze_calib.py` (re-confirms the
  inverted/uncalibrated entropy finding — *why* §5.1 holds) ·
  `python kernel/m5_dynamic_k_north.py` (the disposed `k` adapts idle / busy / budget live; verdict →
  `logs/m5_dynamic_k_north.json`).

---

## 6. Fact-loop — the verified-fact capability multiplier: tuning & usage

The fact-loop is the kernel's **capability multiplier**: compute a verified fact with a hold-out-proven
**deterministic** primitive → inject it **AS TOKENS** (re-prefill the prompt with a "Verified facts" block) → the
model re-reads it from layer 0 and turns it into an *operand*. On North a grounded count-then-add task the base
model does unreliably lifts **0.05 → 0.475 (~9.5×)** when the verified count is delivered this way. This is the
realized counterpart of the bundle's **Mode A vs Mode B** finding (§4, design doc): a fact delivered **as tokens**
(Mode A) becomes usable because early layers process it; the *same* fact injected as a mid-stack **residual** (Mode
B) does **not** — it arrives in the wrong basis, value-output only. So the multiplier is **not** a residual injector;
it is a re-prefill, and it is **governed by M5's `KPolicy`**: thin by default, spend a second generation only when
the base attempt *fails validation* **and** cost/budget allow. Files: `kernel/fact_loop.py` (the governed loop),
`kernel/fact_dispatch.py` (the live router + scheduler executor); CPU suites `test_fact_loop` (43), `test_fact_dispatch`
(43) — both in `verify.sh`.

### 6.1 What it multiplies — and the honest cost caveat

The lift is real and large *on tasks the base does unaided but unreliably*; the **governance** is what makes it a
multiplier rather than a tax. The thin path is the whole point: a verified fact-fetch + a second full generation
costs a model call and latency, and over-grounding an *already-capable* model is pure overhead. So the doctrine is
**spend only where it is earned** (§6.3).

The caveat, stated as plainly as §5.1's calibration trap: **the thin cost-saving is workload-dependent.** The
governed loop is cheaper than "always ground" *only when a non-trivial fraction of base attempts already pass* — each
pass is a re-prefill not paid for. On a workload where the base **almost always fails** (every prompt is hard), the
governed loop spends a second gen on nearly every task, so it costs ≈ "always ground" with *added* base-attempt
overhead — there, **always-ground is the cheaper policy**. The thin win needs a base-success fraction; quote the
multiplier (the accuracy lift) and the thin saving (the gen count) *together with the mix they were measured on*
(`run_fact_loop_north.py` reports `accuracy` and `avg_gens` per arm over a half-easy/half-hard mix — see §6.5).

### 6.2 `governed_fact_loop` API + knobs (`kernel/fact_loop.py`)

`governed_fact_loop(adapter, prompt, *, fact_provider, validate, policy, sysinfo=None, cost_source=None,
budget=None, importance=0.7, maxn=512) -> FactLoopResult`. The only model surface is `adapter.gen_fast(prompt,
maxn) -> (text, n)` (no MLX, no model load — CPU-testable with a fake adapter).

| arg | default | how to tune |
|---|---|---|
| `policy` | — (required) | a `KPolicy` (the §5 object). `decide()` disposes `k`; the loop realizes **at most one** re-prefill, so the spend is binary (k≥1 → ground, k=0 → stay thin). Reuse the §5 knobs (`k_min`/`cost_brake`/`risk_floor_at`). |
| `validate` | — (required) | `validate(answer)->bool` — **the validate-the-branch lever**, judged against the **verified fact / ground truth**, NOT model self-report. This is the trusted-independent signal (§5.1): a base answer that fails it is an *independent* uncertainty of 1.0, the only thing that earns a spend. A validator that raises counts as *not validated* (fail-safe, never crashes the loop). |
| `fact_provider` | — (required) | `fact_provider(prompt)->list[Fact]` — verified **deterministic** facts (cheap, makes **no** model call). Called **only** once a spend is authorized (never on the thin path). Returning `[]` (args not extractable) makes the loop **stay thin honestly** — no false capability-spend claim. |
| `cost_source` | `None` | `callable()->cost∈[0,1]` (injectable, e.g. tests/demos). `None` → live `CostModel().cost_from_sysinfo(sysinfo)` (§5.3, kernel proprioception). A broken sensor fails **safe** to `cost=1.0` (brake toward thin). |
| `budget` | `None` | remaining trap budget (float) for `charge_budget`; `None` = untracked (admit the policy's `k`). An exhausted/clamped budget vetoes the spend (anti-runaway at the budget level, as in §5.3). |
| `importance` | `0.7` | task importance → scales the discretionary spend ceiling in `KPolicy` (same meaning as §5.4). |

`FactLoopResult` carries the **audit trail** so the proof ledger sees *why* it spent or not: `used_facts`,
`n_facts`, `total_gens` (1 thin / 2 spend), `k`, and `audit` (`used_signal`, `reason`, `cost`, `budget_note`,
`base_validated`, `final_validated`, `disposed_k`).

### 6.3 Doctrine: thin by default + the load-bearing inertness invariant

**Thin by default.** The algorithm is: (1) run the base attempt; (2) if `validate` passes → **return it, no
fact-fetch, no re-prefill** (`k=0`, one gen — the multiplier is reserved for prompts the base cannot do); (3) only
if the base **fails** do we ask `policy.decide(KSignals(disagreement=1.0, cost=…, importance=…))` whether to spend,
then `charge_budget` clamps to what the budget affords. **Cost/budget VETO the spend under load:** at `cost→1`
(host saturated) or an exhausted budget, `k` collapses to 0 — stay thin, return the (failed) base answer, and
**audit why** (`used_signal`, `budget_note`). "fail→pass is not proof," applied to spending: don't pay for a second
gen you can't afford or aren't allowed.

**The inertness invariant (load-bearing).** With no facts the loop is **byte-identical** to a plain
`adapter.gen_fast`: `augment_prompt(p, [])` returns `p` *unchanged* (no whitespace, no header), so
`fact_reprefill(adapter, p, [])` `==` `adapter.gen_fast(p, maxn)` — same text **and** `n`. The fact machinery can
never silently perturb the base path; it only ever **adds** a verified block. `test_fact_loop` pins this down (and
the empty-provider case: a spend authorized but `provider→[]` stays thin rather than burn a verbatim re-run).

### 6.4 The live dispatch (`kernel/fact_dispatch.py`)

`fact_dispatch.py` wires the loop into a single live path — the **propose-which-fact / dispose-whether-to-spend**
seam (mirrors learned-proposes / kernel-disposes):

- **`FactRouter` + `default_router()`** — an ordered registry of `FactSpec`s (lexical, deterministic, **model-free**).
  `route(prompt)` returns the first spec whose `matches` fires, else `None`. `default_router()` is seeded with three
  hold-out-proven det primitives — **`letter_count`** (`count_byte`), **`gcd`** (`c_gcd`), **`is_prime`** — each
  parsing its own operands from the prompt, computing the verified value with the primitive (no model), and
  validating a model answer against that ground truth. Register more-specific matchers first.
- **`dispatch(adapter, prompt, *, router, policy, cost_source=None, budget=None, capabilities=None, …) -> dict`** —
  routes the prompt to its verified-fact primitive and runs the governed loop, **or passes through**. Returns
  `{answer, routed, result, total_gens}`. **Capability-gated, fail-closed:** no router match → `routed=None`
  pass-through; a spec matches but its `cap` is not in `capabilities` → `routed='cap-denied:<name>'` pass-through;
  `capabilities=None` = ungated (trust the caller). **Inertness inherited:** whenever `routed` is `None`/cap-denied,
  `answer == adapter.gen_fast(prompt, maxn)[0]` exactly — a prompt the fact path does not own comes out as the bare
  model produced it.
- **`register_fact_loop_executor(scheduler, adapter, router, policy, *, op="fact_loop", cap=None, …)`** — wires the
  live dispatch as a kernel op (the `task_graph` executor ABI, matched exactly), registering an executor **and** its
  paired validator (`<op>_validated`). The op is **capability-gated kernel-side via the scheduler's `admit()`**: a
  `fact_loop` task declares `capabilities=[cap]` (default `fact:fact_loop`) and an un-granted cap is **rejected at
  admit() before the executor runs** (fail-closed — the same gate `test_capability_gate_fail_closed` exercises). A
  pass-through (chat) prompt still commits (the validator returns `True` for an unrouted JUDGMENT pass). It returns
  `{cap, validator, level}` so the caller can grant the cap and stamp the task's validator.

### 6.5 Verifying the fact-loop

- **CPU (Tier A):** `bash verify.sh` runs `test_fact_loop` + `test_fact_dispatch` alongside the rest (28/28 total).
  They prove the inertness invariant, thin-by-default, the lift path, cost/budget veto, the empty-provider honesty,
  routing, the capability gate, and the real `task_graph.Scheduler` admitting + running `op:fact_loop` (with a
  fail-closed reject of an un-granted cap) — all with a fake adapter, no model.
- **Live model (Tier B, North):**
  `python kernel/run_fact_loop_north.py` (the four-arm A/B: `base` / `always_ground` / `governed_idle` /
  `governed_busy` over the count-then-add mix — proves the multiplier lifts accuracy, the governed loop is thin
  `avg_gens∈(1,2)`, and `cost=1.0` vetoes the spend; verdict → `logs/fact_loop_north.json`; `--fake` for a CPU
  structural smoke) ·
  `python kernel/run_fact_dispatch_north.py` (the live routing + governance + inertness on real North **and** the
  scheduler-executor path: `op:fact_loop` runs through a real `Scheduler` with a hash-verified `ProofLedger`, and an
  un-granted cap is rejected at `admit()`; verdict → `logs/fact_dispatch_north.json`).

---

## License

© 2026 Pierre Lamy. All rights reserved.

This repository is published for **review and evaluation**. No open-source license is granted: you may read
and run the code to evaluate it, but copying, modification, redistribution, or derivative works require the
author's written permission. (This is GitHub's default in the absence of a LICENSE file.)
