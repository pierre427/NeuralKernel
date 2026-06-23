# Neural Microkernel — the interruptible-verified-inference end-state (design)

**Thesis.** Today the model owns control flow: it decides when to think, when to stop, when to "call a tool" by emitting text, and it is the thing that emits the final answer — so every failure mode (runaway reasoning, confidently-wrong output, prompt injection, tool misuse) is a control-flow failure the model cannot be trusted to prevent because it is also the thing failing. The end-state inverts this: **the model should not own control flow; the scheduler should.** The LLM becomes a *resumable probabilistic coroutine* — a forward pass that can be suspended at a layer boundary into an explicit continuation frame, switched away from, and resumed bit-identically — with **deterministic interrupt handlers installed at predefined neural hooks** (trap points between blocks). An in-weights router *proposes* a trap; a thin general kernel *disposes* by authorizing and executing a hold-out-proven primitive over an exact continuation, then injecting a verified result and resuming. The model proposes intent and synthesis; the kernel owns pause/dispatch/validate/resume/commit and is the only thing that touches I/O or the user. This document is the kernelized end-state that the proven pre/post shape system and the toy scheduler already point at.

---

## 0. Where this fits

This design is not a fresh start; it is the *kernelized end-state* of two artifacts that already exist and are proven in this repo:

- **`docs/north-shape-system-final-report.md`** — the pre/post (non-kernelized) version that reached **350/350** coding tasks across three suites (ladder-100, expert-50 = 50/50, challenge-200), every recovery hold-out-gated, model-agnostic behind a `ModelAdapter`. That is the deterministic plane + the validate/bail/escalate router proven *as a wrapper around* full generations.
- **`experiments/neural-scheduler/SYNTHESIS.md`** — the *toy kernel*: a from-scratch transformer where every scheduling dynamic is proven (exact park/resume under any interleaving with 0 mismatches, the learned-syscall I/O boundary at 512/512, speculative verify-and-resume, voluntary-yield priority). That is the kernel *mechanisms* proven in miniature.

This document fuses them: take the proven deterministic plane + validate/commit router (the report) and the proven kernel mechanisms (the toy), and reach *inside the real model's forward pass* — turning "wrapper around a generation" into "trap layer between the blocks." Everything below is grounded in measured results on real, quantized production models (North, Mellum, Gemma, GPT-OSS); where a piece is unbuilt it is marked **UNBUILT** or **PARTIAL**.

---

## 0b. Empirical validation & course-correction (2026-06-22, cyber-50, real MLX models)

The *thin core* of the thesis is validated end-to-end; one *layered* assumption is refuted. Three-arm causal A/B (gpt-oss-20b, cyber-50, identical grader; wall-clock in parens):

- **RAW** single-shot, no kernel: **23/50** (5m06s)
- **THIN** kernel = single-shot + deterministic grounding (`det_primitives` verified facts) + the trust-wrapper (contract-bound `ProofLedger`, capability gate, honest calibrated commit): **29/50** (5m01s)
- **HEAVY** kernel = the multi-stage ACH investigation protocol (FRAME→HYPOTHESIZE→…→REPORT): **10/50** (14m31s, and that *understates* it — load excluded)

Capability check (Mellum-12B, weak executor): raw 18/50 vs thin 17/50 (−1, within noise). Causal control: `kernel/raw_baseline.py`. Conclusions:

1. **The core thesis HOLDS.** Scheduler-owned control + deterministic-plane grounding + validate/commit (honest, audited) **is** the win — the thin kernel beats raw on a capable model (+6) *and* adds honesty/calibration/audit/safety for free at raw-tier speed.
2. **A heavy multi-stage reasoning protocol layered on top is NET-NEGATIVE** — it *halved* a capable model's score at ~3× wall-clock (it fragments holistic reasoning and a generic `enforce_contract` clobbers task-specific content). **Structure is a tool, not a hammer:** any such protocol must be FRAME-gated and validation-justified, invoked *only* where it demonstrably earns its cost. On this suite, for these models, it never did → **retire / hard-gate ACH.**
3. **The kernel is a capability *multiplier*, not a substitute.** Deterministic grounding lifts a model that can *exploit* verified facts (gpt-oss +6) but ~nothing for one that can't (Mellum ≈0). Pass-rate levers = **grounding quality + executor capability + (deferred) proposer-SFT**; the trust layer is constant-value across capability levels.

This **refines §3's build priorities**: invest in the *thin* plane (M0 grounding + M1 commit trap + the proof-carrying trust ledger), not a heavy reasoning protocol above it. Full per-task data + diffs: session memory `project_kernel_intent_reconciliation`.

---

## 1. What is already built (vision -> proven artifact)

**Thesis (supported by the evidence below): every M0–M3 *mechanism* is already proven on real, quantized production models — usually on all four (North, Mellum, Gemma, GPT-OSS), with exact invariants and adversarial/causal controls. What remains is integration/productization, not invention: M4 hidden-state injection (Mode B), M5 in-kernel/GPU scheduler, scheduler-owned I/O as a capability system, the two *mandatory* trap points (post-L0 entry, post-last commit), and clean-context agents.**

All file paths below are under `/Users/pierrelamy/Desktop/Test Harness/`; scripts are in `finetune/gpt-oss-unsloth/scripts/` unless noted. Numbers are quoted from the cited logs/reports, not paraphrased.

### 1.1 Mapping table — vision component -> existing artifact

| Vision component | Artifact (file:symbol) | Status | Measured result / invariant (evidence) |
|---|---|---|---|
| **Identity trap as sidecar** — trap-disabled must be bit-identical to base (`max_diff == 0`) | `north_gateway_invariants.py:ResidualGateway` (zero-init `up.weight`), inserted via `run_stack(..., gate_at=L)` | **PROVEN** (real North, 49-layer `cohere2_moe`, MXFP8) | `reports/north-m0/gateway_invariants2.log`: `[invariant 1] zero-init gateway after layer 24: max|gated - base| = 0.000e+00 (PASS)`. **The bug-then-fix is on record**: `gateway_invariants.log` first shows `2.943e+00 (FAIL)` from a bf16 upcast; the fix (`delta.astype(x.dtype)`, line 40) restores `0.0`. Manual `cohere2_moe` forward parity vs `model()` = `0.000e+00`. d=2048, layers=49. |
| **Exact park/resume = the continuation frame** | `north_gateway_invariants.py:run_stack(..., park_at=L)` — stash residual after layer L to safetensors, reload, resume L..N | **PROVEN** (real North) | `gateway_invariants2.log`: `[invariant 2] park@24 -> serialize residual -> resume 24..49: max|resumed - base| = 0.000e+00 (PASS)`. Note the **dtype-preserving serialization is load-bearing** (the same 2.943 divergence before the fix). This is the literal "suspend a forward pass to disk and resume bit-identically." |
| **Learned trap router (the syscall/trap-emit head)** — classifier on the residual decides *in-weights* whether to trap and to which primitive | `north_syscall_head.py:SyscallHead` (RMSNorm→Linear→GELU→Linear over L24 last-token residual); 5-way `{none, count_byte, c_gcd, is_prime, shortest_dist}` | **PROVEN** (clean), **PROVEN under adversarial gate** | `reports/north-m0/syscall_head.log`: trap-detect **100.0%**, routing **100.0%** (n=64, chance 26.6%). `syscall_v2.log` adds the **held-out adversarial gate** (52 OOD/oblique + hard-negative prompts disjoint from train): North L24 routing **98.1%**, trap-detect **100.0%**, **false-trap on chat = 0.0%**, vs bag-of-words 78.8% / 82.7% / 30.8% — VERDICT PASS, beats BoW. |
| **…same trap router across the 4 models** | `syscall_cross_model.py:featurize` (architecture-agnostic `_Tap` that runs each model's *native* forward, captures a mid-stack residual, aborts early) | **PROVEN** (4/4 models) | Same adversarial gate, all PASS, all trap-detect **100%**, all false-trap **0.0%**: North 98.1% (L24/49), **Mellum 98.1%** (L14/28), **Gemma 96.2%** (L15/30), **GPT-OSS 96.2%** (L12/24). Each beats its BoW control (≈77–81%). Logs: `reports/north-m0/syscall_{mellum,gemma,gptoss}.log`. |
| **Causal controls on the trap signal** (is it real, or head capacity?) | `north_syscall_controls.py:fit_eval` — REAL vs SHUFFLED-labels vs RANDOM-FEAT placebos; BoW control lives in `north_syscall_adversarial.py` | **PARTIAL** | The script is built and the verdict logic requires real≥90% while shuffled/random collapse to ≈chance. The **BoW keyword-triviality control is run and logged** (above). The shuffled/random-feat placebo run-log is **not persisted** in `reports/north-m0/` — re-run to capture it before citing those two numbers. (Methodology is in place; the specific shuffled/random output is unrecorded.) |
| **Trap fires → scheduler executes a trusted primitive → result injected → resume** (end-to-end, no text [[fn]] protocol) | `north_syscall_e2e.py:main` — read head on L24 → if fires, `extract_args` → call hold-out-proven `det_primitives` → PARK after 1st token, INJECT verified value, RESUME via `make_prompt_cache` | **PROVEN** (mechanism live on North) | The full "learned proposes (head), kernel disposes (dispatch)" pipeline runs end-to-end on held-out prompts (strawberry-r, gcd(1071,462), prime 1000003, etc.), contrasting model-UNAIDED vs head-gated-correct. No saved transcript log in `reports/north-m0/`, but the pipeline is wired and the upstream pieces (head 100% detect; primitives hold-out-100%) are independently proven. |
| **HOLD / DISPATCH / RESUME at token boundaries** (Mode A trampoline) — mid-stream det dispatch | `tool_dispatch.py:chat_with_tools` (uniform `[[fn(args)]]` marker → park KV → exec primitive → inject into *same* cache → resume); `tool_dispatch_native.py` = per-model native tool-call dialects | **PROVEN** (model-agnostic mechanism) | Identical loop runs on North/Mellum/GPT-OSS; the only per-model surface is `adapter.encode_fast/.eos`. This is exactly the Mode-A token-boundary trampoline the vision wants built first. (`tool_dispatch_native.py` is the productionization note — swap the uniform marker for native formats; loop unchanged.) |
| **Efficient mid-stream park/resume on the real KV cache** (continuation = KV, no re-run) | `north_mid_halt.py` — one thinking pass; at a checkpoint PARK (mark cache offset), branch speculative answer, validate, commit or `c.trim(n)` to RESUME | **PROVEN** (real North) | Docstring records North's cache is **fully trimmable (verified)** so park/resume is exact and copy-free; this is the toy's `autoregressive.py` (0 mismatches) realized on the 49-layer model, fused with the harness validator. Adaptive halt spread logged at runtime. |
| **Deterministic plane + "primitive admission control"** (hold-out gate before a primitive is trusted) | `det_primitives*.py` (`.py`, `_advanced`, `_cyber`, `_ds`, `_plt`) = the primitive library; `hold_out_generality.py` / `_more` / `hold_out_cyber.py` = the gate; `shape_recoveries.py`, `challenge_recoveries.py` = spec-matched recovery wiring | **PROVEN** | `docs/north-shape-system-final-report.md`: primitives must score **100%** on thousands of NOVEL instances vs *independent* ground truth — `dpll` **3000/3000** (vs brute-force 2ⁿ), `symbolic_differentiate` **2000/2000** (vs numeric finite-difference), `unicode_grapheme_count` **2000/2000** (vs regex `\X`); security shapes = **9 general primitives, every one 100% hold-out**. A memorized lookup scores ~0% held-out — the gate *is* the admission-control mechanism. |
| **The scheduler/router core** — validation-gated bailout + recover-direct + escalate | `shape_appliance.py:run_appliance` (recover-direct if a hold-out-proven recovery matches the contract → 0 model tokens; else fast; else escalate); validators in `output_validators.py`, detector in `shape_detector.py` | **PROVEN** | `docs/north-shape-system-final-report.md`: **350/350 coding tasks across three suites** (ladder-100, expert-50, challenge-200), every recovery hold-out-gated. expert-50 = **50/50** vs 38 (enhanced router) / 30 (reasoning-off base). The stack is **model-agnostic** (only surface is a `ModelAdapter` with `gen_fast`/`gen_reasoning`). This is the scheduler's "recover-direct / validated bail / escalate" dispatch, proven at scale. |
| **The toy kernel** — budgeted pipeline, park/resume, policies, I/O boundary, priority | `experiments/neural-scheduler/scheduler.py` (Scenarios A–E), `experiment.py` (gateway invariants + heads), `autoregressive.py`, `speculative.py`, `pertoken.py`, `rl_halt.py`, `confidence_probe.py`; capstone `SYNTHESIS.md` | **PROVEN** (toy) — dynamics real, speedup *accounted* not fused | `experiments/neural-scheduler/README.md` + `SYNTHESIS.md`: **A** online==offline exact park/resume across arbitrary interleaving (**0 mismatches**, memory isolation holds); **B/C** budgeted drain & streaming; **D** the external-I/O boundary — the **model emits the syscall (learned head), scheduler blocks/polls/delivers-by-thread-id/resumes**: **100% detect, 512/512 correct, util ~0.75** under 32% blocking; **E** voluntary-yield priority cuts HIGH-prio latency 42% (26→15). `speculative.py`: 98.7% @ depth 2.6, verifier resumes 99% of wrong drafts, live runtime 0-mismatch. |
| **The HOLD/SWITCH/RESUME = context switch over continuations** | `scheduler.py:Proc` (`h` = residual continuation, `depth`, `cum`) + step_group/switch logic | **PROVEN** (toy) | README: "a context switch = swapping which parked residual the next layer-call consumes … switching is provably exact because processes never attend across each other (memory isolation holds)." This is the toy proof behind clean-context agents (M2) — the *exactness* property is established; the in-process LLM realization is not yet built. |
| **The megakernel / in-kernel scheduler end-state (M5) + the privilege boundary** | `docs/freebsd-scheduler-lessons.md` (§3, §3a, §4) | **DESIGN, grounded** (not built) | The FreeBSD-grounded analysis establishes the architecture: "**the model is the CPU/process; the runtime is the kernel; validation is a syscall**" (§3), with a sort-by-privilege table (propose halt/route = in-model; execute validator / I/O / tool calls = kernel). §3a/§4 refine it: array-native exact ops (constrained decoding, MoE expert-steering) are an **on-stack co-processor in the GPU graph** (the kernel's FPU), and only *arbitrary untrusted execution / true I/O* traps to an external sandbox. This is the conceptual spec for M5 + scheduler-owned-I/O; it is reasoned, not yet implemented. |

### 1.2 What this inventory establishes vs. what is genuinely unbuilt

**Proven on real models (M0–M3 mechanisms):**
- The **identity-trap sidecar** is bit-identical (`max|gated−base| = 0.0` at L24 of real North) — a 129th-expert routing-slot hazard is avoided exactly as the vision requires, because the trap is an additive zero-init residual patch, not a routed expert.
- **Park/resume of the continuation** is exact both as a serialized residual (`0.0`) and as a trimmed live KV cache (`north_mid_halt.py`, verified trimmable; toy `autoregressive.py` 0 mismatches).
- The **learned trap router** generalizes adversarially (100% detect, 0% false-trap) on **all four models**, and beats both BoW (keyword-triviality) controls — the in-weights "request a typed opcode" decision is real.
- **Token-boundary dispatch (Mode A)** runs model-agnostically (`tool_dispatch.py`), end-to-end head→dispatch→inject→resume (`north_syscall_e2e.py`).
- The **deterministic plane** has working primitive-admission-control (hold-out gate at 100% on thousands of novel instances), and the **scheduler/router core** hits **350/350** with validation-gated recover-direct/bail/escalate.
- The **toy kernel** proves every scheduling dynamic — exactness under interleaving, the learned-syscall I/O boundary (512/512), speculative verify-and-resume, voluntary-yield priority.

**Genuinely unbuilt / partial (the integration & productization residue):**
- **M4 — Mode B hidden-state injection**: ~~only the token-boundary trampoline (Mode A) is built; vector injection into the residual is not.~~ **UPDATE 2026-06-23: Mode B is now PROVEN as an MVP on real North** — a trained result-conditioned residual write at site 24 makes North emit the injected value through 24 downstream MoE layers (deployment 1.0, causal/faithfulness 1.0, placebo→chance, ==0 disabled; `kernel/mode_b.py` + `mode_b_north.py`). Scope: one task / one site / single-digit values — a placebo-controlled proof-of-mechanism, not yet a general multi-token/multi-primitive injector. See §4.3.
- **M5 — in-kernel / GPU-megakernel scheduler**: the toy's speedup is *accounted, not fused* (README is explicit: "not a fused GPU pipeline"); design only in `freebsd-scheduler-lessons.md`.
- **The two mandatory trap points** (post-L0 entry trap, post-last-layer commit trap as the sole output path): the *substrate* (identity trap + commit-via-validation) is proven, but neither mandatory hook is wired as an always-on plane.
- **Clean-context in-process agents (M2)**: the exactness property (isolated continuations, exact context switch) is proven in the toy; the agent-address-space + typed-mailbox realization on a real LLM is not built.
- **Scheduler-owned I/O as a capability system**: today the model emits an in-band `[[fn]]`/native marker and the runtime executes it; a true capability/authorize/log boundary where the model *cannot* touch I/O is design-stage (`freebsd-scheduler-lessons.md` privilege table).
- **Syscall-head causal placebo run-log** (`north_syscall_controls.py`): script and BoW control exist and pass; the shuffled-label / random-feature run output is not persisted — re-run to record it.

**Naming note for the writeup:** the existing docs already use the kernel framing literally ("the model is the CPU/process, the runtime is the kernel, validation is a syscall" — `freebsd-scheduler-lessons.md`; "learned proposes, kernel disposes" — `SYNTHESIS.md`), which directly supports *Verified Inference Kernel* / *Neural Microkernel* / *Proof-Carrying Inference Runtime* as the end-state name.

---

## 2. The architecture — three planes + the kernel lifecycle

The runtime has **three planes**, separated by privilege (the FreeBSD-grounded boundary of `docs/freebsd-scheduler-lessons.md` §3):

1. **Neural plane** (the model = the CPU/process). Owns *intent, synthesis, and repair-planning*. It runs the forward pass, proposes traps via an in-weights router, proposes halts, and drafts output. It is untrusted: it can be wrong, run away, or be prompt-injected, so it holds no privileged capability.
2. **Deterministic plane** (the verified handlers = the FPU/co-processor). Hold-out-gated exact primitives (`det_primitives*.py`, admitted only by `hold_out_*.py` at 100% vs an independent oracle). Correct by independent construction; an injected result carries a real guarantee the stochastic model cannot self-supply.
3. **Scheduler plane** (the kernel). Owns *pause / dispatch / execute / validate / resume / bail / timeout / proof-ledger* and is the **only** thing that touches user I/O, network, files, and tools. It authorizes typed-opcode requests against a capability bitset, executes the handler under a hard timeout, logs to a proof ledger, decrements budgets, and is the sole commit-to-user path.

### Kernel lifecycle (ASCII)

```
                                  user request
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │   SCHEDULER INGRESS       │  (kernel) parse, assign
                         │   request_id, budgets,    │  capabilities + budgets,
                         │   capability bitset       │  pick a lane
                         └──────────────────────────┘
                              │                   │
              shape matches a │                   │ needs the model
           hold-out-proven    │                   │
           recovery (0 toks)  ▼                   ▼
                ┌──────────────────┐   ┌───────────────────────────────────────────┐
                │  DIRECT DET-BLOCK │   │              MODEL LANE                    │
                │  (deterministic   │   │                                           │
                │   plane only)     │   │   embed → Layer 0 (dense)                  │
                └──────────────────┘   │        │                                   │
                         │             │        ▼                                   │
                         │             │  ╔═══ POST-L0 ENTRY TRAP ═══╗  (mandatory)  │
                         │             │  ║ task-shape routing,      ║               │
                         │             │  ║ capability assignment,   ║               │
                         │             │  ║ arm/disarm later traps   ║               │
                         │             │  ╚══════════════════════════╝               │
                         │             │        │                                   │
                         │             │        ▼                                   │
                         │             │  Layers 1..47 (MoE top-8)                   │
                         │             │   · · mid-layer traps @ 8/16/24/32/40 · ·   │
                         │             │        │  trap fires?                       │
                         │             │        │     │ yes → PARK ─► scheduler:     │
                         │             │        │     │   authorize→exec handler→    │
                         │             │        │     │   inject result→RESUME k+1   │
                         │             │        ▼                                   │
                         │             │  Layer 48 (last)                            │
                         │             │        │                                   │
                         │             │        ▼                                   │
                         │             │  ╔═ POST-LAST COMMIT TRAP ═╗  (mandatory)   │
                         │             │  ║ validate candidate,     ║               │
                         │             │  ║ repair / early-bail,    ║               │
                         │             │  ║ escalate, or commit     ║               │
                         │             │  ╚═════════════════════════╝               │
                         │             └───────────────────────────────────────────┘
                         │                          │
                         └──────────────┬───────────┘
                                        ▼
                        ┌────────────────────────────────┐
                        │  VALIDATE / REPAIR / COMMIT      │  (kernel; proof-ledger
                        │  syntax→schema→property→oracle→  │   entry per commit)
                        │  proven; rollback on failure     │
                        └────────────────────────────────┘
                                        │
                                        ▼
                                      user
```

### Key definitions

- **Neural interrupt (trap).** A sidecar layer inserted *after* a decoder block on the residual stream — **not** a routed MoE expert (a 129th expert would steal a top-8 routing slot and perturb the gate; see §3). At init it adds an exactly-zero delta, so *trap-disabled is bit-identical to base* (`north_gateway_invariants.py`, `max_diff = 0.0`). When it fires, it parks the continuation, the scheduler executes a trusted primitive, and the verified result is injected before the pass resumes at layer k+1.
- **Continuation frame** (= a resumable request-lane). The suspended process: `{request_id, token_pos, layer_idx, hidden_ref, kv_cache_ref, decode_state, capabilities, budgets}`. `hidden_ref` (the parked residual) + `kv_cache_ref` (the mlx_lm cache) together *are* the suspended forward pass — proven serializable and resumable bit-identically (`gateway_invariants2.log`, invariant 2).
- **HOLD / SWITCH / RESUME.** First-class kernel ops over the continuation frame. HOLD parks a frame; SWITCH swaps which parked continuation the next layer-call consumes; RESUME restores and continues. Proven exact in the toy (a context switch = swapping `Proc.h`; 0 mismatches under any interleaving) because lanes never attend across each other (memory isolation).
- **Clean-context agents.** In-process sub-agents, each with its own **address space** (a fresh continuation frame + clean KV cache) and a **typed mailbox**. A parent spawns a child frame; the child returns only a *typed, validated result* into the parent's mailbox — **never its transcript** (§5.5). This keeps the scarce resource (context/compute) uncontaminated and closes the transcript-injection vector.

### 2b. The Proof-Carrying Task Graph — the kernel process table  *(BUILT: `kernel/task_graph.py`, 17/17 tests green)*

Above hold/switch/resume, clean-context agents, the primitive ABI, and the M1 commit trap sits one thing that makes the whole architecture coherent: a **kernel-owned task graph** that tracks intent, subgoals, continuations, evidence, validators, budgets, and completion state. Without it the scheduler is clever but stateless; with it the scheduler is an operating system for work. The design rule is the separation that makes everything scale: **the model is not the task tracker — the scheduler is.** The model proposes/summarizes/repairs/explains; the scheduler owns task state, dependencies, budgets, capabilities, artifacts, validation, commit, and history. The OS mapping is exact:

| OS concept | this runtime |
|---|---|
| process table | the active task graph (DAG of `Task` nodes) |
| thread state | task `status` (a 13-state machine: NEW/READY/RUNNING/BLOCKED/WAITING_*/VALIDATING/PASSED/FAILED/REPAIRING/ABORTED/COMMITTED) |
| scheduler queue | the READY set (deps satisfied) |
| syscall table | the registered executors (primitive/model_lane/agent/tool) |
| file descriptors | content-addressed **artifact refs** — tasks pass refs, never blobs (no context pollution) |
| permissions | per-task `capabilities` checked at admit |
| exit code | the validator result + `Verification` level (verified/validated/checked/judgment/unverified — never collapsed to "done") |
| kernel log | the hash-verified `ProofLedger`: one entry per transition |
| debugger | the replayable `TaskEvent` trace |

The loop is clean and bounded: **model proposes a plan → scheduler `admit`s (type-checks: unknown ops and unknown validators are *rejected*, so the model can't invent imaginary or unsafe tools) → pick a READY node → charge the trap-budget (anti-runaway) → run its allowed executor → write the output artifact → validate → append a proof entry → unblock dependents → on FAIL spawn a *clean-context repair worker* that gets only the failing slice → bail the instant the root's output contract is satisfied.** Two planners feed it: a **DeterministicPlanner** (known shape → fixed DAG from `PLAN_TEMPLATES`: structured-output, code-gen, cyber-alert, …) and a **ModelAssistedPlanner** (model proposes a DAG; the scheduler normalizes + rejects unsupported ops). Demonstrated end-to-end (`kernel/test_task_graph.py`): a structured-output plan with a schema bug runs `draft → syntax → schema:FAILED → repair(clean-context) → schema:PASSED → commit:COMMITTED`, the commit lane receives only the compact validated object (not the repair transcript), the proof ledger is hash-verified, and a `max_traps=2` budget aborts a runaway plan. This is the **early-bailout engine** the paper's validation-gated halt always implied — a request is done because the *plan says it is complete*, not because the model emitted a confident paragraph.

### 2c. The Job Kernel — intake, classification, agent-assigned plans, master rollup, plan memory  *(BUILT: `kernel/orchestrator.py`, 17/17 tests green)*

One layer up from the task graph sits the **Job Kernel** (`Orchestrator`): the front door that turns an incoming job into a running, audited work plan, **generalized across every job type the project has** (coding, cyber, structured-output, …). Pipeline: **Job ▸ `JobClassifier` (assess type) ▸ `PlanMemory.recall` (proven template + prior learnings + stored details) ▸ `DeterministicPlanner` builds a DAG with every node ASSIGNED to an agent/primitive (`model_lane` / `primitive:*` / `agent:*` / `tool:*` / `commit_trap`) ▸ `Scheduler` runs it ▸ master Rollup ▸ `PlanMemory.learn`.** The rollup is the user-facing tracker view — a per-job checklist (`✓ det_checks [primitive:cyber] (verified)`, `✗ verify_claims [agent:verifier] (unverified)`) plus a `dashboard()` master rollup across all jobs (counts, committed, grouped by type, proven templates, learnings).

The planner is itself a **memory system** (`PlanMemory`, JSON-persisted) — the moat the design calls for: it `remember`s arbitrary details keyed by job-type/job-id (e.g. "this user wants STIX output"), keeps a **proven-template library** (`best_template` = highest pass-rate per job type, so the classifier reuses what worked), accumulates **learnings** ("`det_checks` catches missing SSRF evidence"; "job X failed at: [run_tests]"), and grows from both success and failure. Demonstrated end-to-end: one orchestrator classifies + runs coding, cyber, and structured-output jobs, assigns each plan's nodes to agents, commits all three through their plans with hash-verified ledgers, rolls them up into a dashboard, and persists/reloads its memory. The principle holds all the way up: **the model is not the orchestrator — the Job Kernel owns intake, classification, planning, assignment, budgets, rollup, and memory.**

---

## 3. Staged build plan (M0–M5)

This plan turns the "interruptible neural runtime / verified inference kernel" vision into a concrete M0–M5 build order grounded in what already exists in this repo. Throughout, the controlling doctrine is the one already written down in `docs/freebsd-scheduler-lessons.md` §3–§3a: **the model is the CPU, the scheduler is the kernel, validation is a syscall, "learned proposes, kernel disposes."** Each stage names its success invariant, what is already proven here, what remains, the exact files/functions to add or extend, and the risks.

### Target substrate: North, and why a sidecar, not a 129th expert

Confirmed from the model's `config.json` at `/Users/pierrelamy/.lmstudio/models/bsisduck/North-Mini-Code-1.0-MLX-MXFP8/config.json`:

- `model_type: cohere2_moe`, `num_hidden_layers: 49`, `hidden_size: 2048`, `sliding_window: 4096`
- `num_experts: 128`, `num_experts_per_tok: 8` (top-8 routing)
- `first_k_dense_replace: 1` → **layer 0 is dense; layers 1–48 are MoE top-8.**

This is the architectural reason traps must be **sidecars, not experts**. A 129th expert competes for one of the 8 top-k routing slots per token: installing it perturbs the softmax over 128 experts, so even a zero-weight 129th expert can change *which* of the original 128 get selected (it occupies a logit), breaking bit-identity. The router's gate is a learned policy we must not disturb. A **sidecar trap** is the opposite: it runs *after* a whole decoder block on the residual stream `x`, adds a delta that is provably 0 at init, and never touches the gate. This is exactly the `ResidualGateway` pattern already proven (below). Test sites are **after blocks 8 / 16 / 24 / 32 / 40 / 48** — evenly spaced through the MoE stack, with 48 being the post-last-layer commit point and a post-block-0 site reserved for the entry trap (M3).

### Continuation-frame schema (the suspended process)

A suspended process is a partially-evaluated forward pass — the residual parked at a layer boundary, which `north_gateway_invariants.py:run_stack(park_at=...)` already serializes and resumes bit-identically. Formalize it as:

```
ContinuationFrame {
  request_id      : u64            # process id (Proc.pid in scheduler.py)
  token_pos       : int            # decode position
  layer_idx       : int            # k: resume at layer k+1
  hidden_ref      : tensor handle  # the parked residual x  (safetensors, dtype-preserving)
  kv_cache_ref    : cache handle   # mlx_lm make_prompt_cache state (the AR continuation)
  decode_state    : {rng, temp, eos, n_emitted}
  capabilities    : bitset         # which typed opcodes this frame may request
  budgets         : {traps_left, depth_left, ops_used:set, deadline_ticks}
}
```

`hidden_ref` + `kv_cache_ref` together are the "suspended process." The toy `experiments/neural-scheduler/scheduler.py` already carries this as the `Proc` dataclass (`pid, tokens, h, depth, cum, state, capabilities-via-needs_io/io_op, …`) and proved (Scenario A) that draining it under any budget/interleaving reproduces the offline model with **0 mismatches**.

### Typed-opcode ABI (the capability system)

The model never touches I/O; it *requests* a typed opcode and the scheduler authorizes/executes/logs. The opcode set is grounded in the existing det-block library:

```
OPCODE        ARGS (typed)                RESULT        VALIDATION LEVEL   HANDLER (existing)
count_byte    (s:str, ch:char)            int           proven (oracle)    det_primitives.count_byte
c_gcd         (a:int, b:int)              int           proven             det_primitives.c_gcd
is_prime      (n:int)                     bool          proven             north_syscall_e2e.is_prime
shortest_dist (graph, a, b)               num           proven             tool_dispatch._shortest_dist
sieve_primes  (n:int)                     int[]         proven             det_primitives_advanced.sieve_primes
dpll          (clauses, nv)               assign|None   proven (3000/3000) det_primitives_plt.dpll
sym_diff      (expr, var)                 str           proven (2000/2000) det_primitives_plt.symbolic_differentiate
grapheme_cnt  (s)                         int           proven (2000/2000) det_primitives_ds.unicode_grapheme_count
regex_match   (pat, s)                    bool/spans    proven             regex_cassette
validate_out  (text, otype, entry)        ok:bool       syscall (sandbox)  output_validators.is_complete_valid
exec_check    (code, entry, cases)        ok:bool       syscall (sandbox)  shape_appliance.check_cases
```

Every handler in the top group is **hold-out-proven** (see "Validation levels" below). Two ABI invariants, both already enforced in code: (1) the scheduler dispatches **positionally by value order** so schema param names need not match function params (`tool_dispatch_native.py:113`); (2) each distinct call fires **exactly once** by index, robust to the model echoing the call marker (`tool_dispatch.py:70`).

### Trap budgets (anti-runaway)

Carried in `ContinuationFrame.budgets`, enforced by the scheduler, never by the model. Grounded in `tool_dispatch.py` (`max_calls=6`, `max_tokens` cap) and the project's standing "no-retries-at-T=0 / cap runaways" memory:
- **max traps/token** (default 1; the `len(matches) > dispatched and dispatched < max_calls` guard already implements this).
- **no recursion**: a handler may not itself emit an opcode.
- **same-op-once**: `budgets.ops_used` rejects a repeat of an identical (op,args) within a frame.
- **failed-op-disables**: a handler that errors (`<err:...>` path in `tool_dispatch.py:75`) disarms that opcode for the frame.
- **depth/deadline**: `depth_left` and `deadline_ticks` bound total work — the toy's per-tick budget `W` (`scheduler.py`) is the multi-process generalization.

### Result-reinjection sequencing: Mode A before Mode B

- **Mode A — token-boundary trampoline (build first, safe).** Park the KV cache, dispatch the opcode (0 model tokens), encode the result as tokens, inject into the *same* cache, resume. This is fully proven on real models: `tool_dispatch.py:chat_with_tools` and `tool_dispatch_native.py:chat_with_native_tools` (per-model native dialect: gpt-oss harmony, Mellum `<tool_call>`, North `[[fn]]`). The end-to-end head→dispatch→inject→resume path runs in `north_syscall_e2e.py:generate(inject_after, inject_text)`.
- **Mode B — hidden-state vector injection (research, later).** Inject the result as a residual-space vector via a trained adapter at layer k+1 instead of as tokens. The substrate exists (`ResidualGateway` adds an exactly-zero delta at init), but Mode B is unproven and changes hidden state directly, so it is deferred to M4 and gated behind a bit-identity-when-disabled invariant identical to M3's.

---

### M0 — Scheduler owns I/O (capability system)

**Goal.** All I/O, tool use, and validator execution move behind the scheduler; the model only *requests* typed opcodes. Establish the privilege boundary as code.

**Success invariant.** Every external effect (file/network/tool/code-exec) flows through one dispatch chokepoint; the model touches none directly. The learned trap signal beats a keyword baseline on held-out adversarial prompts (proves the request decision is real, not lexical).

**Already done here.**
- The dispatch chokepoint and capability execution exist end-to-end. `shape_appliance.py:run_appliance` is the recover-direct / validated-bail / escalate router; `check_cases` (with `_timed` SIGALRM sandboxing, `_struct_eq`, `_coerce_int_keys`) is the execution-validator syscall handler.
- The **learned in-weights request decision** is proven on North and three other models. `north_syscall_head.py` reads North's L24 residual → 5-way {none, count_byte, c_gcd, is_prime, shortest_dist}. Measured (`reports/north-m0/syscall_head.log`, `syscall_v2.log`): **trap-detection 100%, routing 100%** on the in-distribution test; on the 52-prompt adversarial gate (keyword-free paraphrases + hard negatives) **routing 98.1%, trap-detect 100%, false-trap 0.0%, vs bag-of-words 78.8%** — VERDICT PASS.
- Causal controls done: `north_syscall_controls.py` (REAL ≫ SHUFFLED-labels ≈ RANDOM-features ≈ chance) proves the signal is in North's representation, not head capacity. Cross-model replication: `syscall_cross_model.py` + logs show gpt-oss (L12/24, 96.2%), Mellum (L14/28, 98.1%), Gemma (L15/30, 96.2%) all PASS with 0% false-traps.

**Remains to build.**
- A single `Scheduler.dispatch(opcode, args, frame)` entrypoint that **authorizes against `frame.capabilities`, executes the handler, logs to a proof-ledger, decrements budgets** — promoting today's inline `_exec`/`check_cases` calls into one mediated path. New file: `kernel/scheduler.py` (port the toy's `run_scheduler` policy/loop; replace toy `World` with the real opcode table).
- A typed opcode registry (the ABI table above) as `kernel/opcodes.py`, wrapping the existing primitives with arg-types + validation-level metadata.

**Risks.** Per-model tool dialects are the historical top false-fail (per project memory) — mitigated by `tool_dispatch_native.py`'s per-model dialect table rather than forcing one syntax. Arg-extraction from prose is a real boundary (`north_syscall_e2e.py:extract_args` already returns `None` and logs an "honest boundary" rather than guessing).

---

### M1 — Post-last-layer commit trap (the only output path)

**Goal.** A mandatory trap after layer 48 that validates the candidate output, repairs/early-bails, and is the **single commit-to-user channel**.

**Success invariant.** No token reaches the user without passing the commit trap. Validated outputs commit; invalid ones repair or escalate; nothing regresses (passing tasks are never modified — guaranteed structurally by route-on-contract).

**Already done here.** This is the most mature stage. The commit trap is essentially the existing appliance's validate/bail/escalate logic:
- `shape_appliance.py:run_appliance` already gates every output: `form_ok = is_complete_valid(draft, otype, entry)` → bail if valid, else escalate to reasoning, then `passed = check_cases(final, entry, cases)`. That is exactly validate → repair → commit.
- The toy proves the deeper version: `speculative.py` (per SYNTHESIS.md) = draft → verify → resume-99%-of-wrong-drafts, and the production analogue is the router reaching **100/100 on ladder-100** (`north_router_server.py`, beating reasoning-off 92 and uniform-bounded 89). The commit trap is "speculate/verify/rollback" at the output boundary.
- The skill-cassette commit path (regex matcher, 12/12, ~1 ms) shows repair-by-deterministic-specialist when the model has a true capability gap.

**Remains to build.**
- Wire the appliance's validate/escalate as a literal **trap handler keyed to layer 48** inside the runtime (today it wraps a full generation, not a forward-pass hook). New: `kernel/commit_trap.py` calling `output_validators.is_complete_valid` + `shape_appliance.check_cases` as the validator-sandbox syscall, with `decode_state` rollback on failure.
- A **proof-ledger** entry per commit (op, validation-level reached, pass/fail) so commits are auditable ("proof-carrying inference").

**Risks.** Validation ≠ correctness for the confidently-wrong floor (SYNTHESIS.md, `confidence_probe.py`: ~83% intrinsic floor; l80 is a real capability gap). Mitigation: the commit trap must reach **oracle/proven** validation level for opcode results, and accept the honest boundary (escalate, don't fabricate) when only `property`-level checks are available.

---

### M2 — Clean-context in-process agents (hold/switch/resume)

**Goal.** First-class HOLD / SWITCH / RESUME over the continuation frame at **token boundaries**, enabling clean-context sub-agents that merge only typed results, never transcripts.

**Success invariant.** Online runtime under *any* interleaving == offline single-process result, bit-for-bit (memory isolation holds because processes never attend across each other). Sub-agents communicate only via typed mailbox values.

**Already done here.**
- The exact invariant is **proven in the toy**: `scheduler.py` Scenario A asserts `mismatches vs offline (answer & exit-depth): 0` at budget W=1 (maximal parking/switching) *and* W=N — "switching is provably exact." The mechanism is the `Proc.h` swap.
- Park/resume of the continuation through **disk serialization** is proven exact on real North: `north_gateway_invariants.py` invariant 2 (`reports/north-m0/gateway_invariants2.log`): `park@24 → safetensors → resume 24..49: max|resumed − base| = 0.000e+00 PASS`.
- Mode-A token-boundary park/dispatch/resume on the live KV cache is proven on real models (`tool_dispatch.py`, `tool_dispatch_native.py`).

**Remains to build.**
- A `hold(frame)/switch(frame)/resume(frame)` API over `mlx_lm` KV caches (the toy used cloned tensors / disk; production needs cache-handle park/resume — the serving-infra prefix-cache machinery noted in freebsd-lessons §3).
- **Agent address spaces + typed mailboxes**: a parent spawns a child frame with its own clean KV cache; the child returns a *typed value* merged into the parent's mailbox — never its transcript. New: `kernel/agents.py` (frame table + mailbox), reusing the `World`/`by_tid` delivery pattern from `scheduler.py:run_scheduler` (the I/O boundary delivers responses by thread-id, which is the mailbox primitive).

**Risks.** The toy parks an *encoder* residual cloned in memory; the real system parks an *autoregressive* KV cache whose dtype must be preserved (the `gateway_invariants.log` history is the cautionary tale — see M3). Cache-handle aliasing across frames must be copy-on-park or isolation breaks.

---

### M3 — Post-layer-0 entry trap (scheduler ingress, bit-identical when disabled)

**Goal.** A second mandatory trap, after the dense block 0, that does task-shape routing, capability assignment, and arms/disarms later traps. It must be an **identity sidecar**: disabled ⇒ bit-identical to base.

**Success invariant — and it is already met.** `north_gateway_invariants.py` proves a zero-init `ResidualGateway` grafted into the 49-layer `cohere2_moe` stack is bit-identical to base: invariant 1 = `max|gated − base| = 0.000e+00 PASS` (`gateway_invariants2.log`). **Critically, the first run FAILED at 2.943** (`gateway_invariants.log`) because the zero delta was silently upcast (bf16/quantized base); the fix is the one load-bearing line:

```python
return x + delta.astype(x.dtype)   # north_gateway_invariants.py:40 — without .astype, max_diff = 2.943
```

This is the proof that the sidecar pattern works *and* the precise pitfall to avoid. The entry trap reuses this exact module at the post-block-0 site (layer_idx=0), which is even safer than the proven L24 site because block 0 is dense (no MoE gate to perturb at all).

**Already done here.** The identity-sidecar substrate (proven 0.0), the routing brain that rides on it (the syscall head, M0), and the manual `run_stack` forward that inserts a module after an arbitrary layer (`gate_at`).

**Remains to build.**
- Move the head's read point from L24 to **post-block-0** and retrain/eval it there (the cross-model script already parameterizes the layer via `--frac`, so this is a one-flag re-run + adversarial-gate re-pass).
- Make the entry trap *write* into `frame.capabilities` and `frame.budgets` (arm/disarm later traps, assign the opcode bitset) — the head currently only classifies; extend `SyscallHead` output to a capability vector. New: `kernel/entry_trap.py`.

**Risks.** Post-L0 residual may carry less task signal than L24 (only one block of mixing) — must re-pass the placebo + adversarial gates at the new layer before trusting it; if it fails, fall back to a two-point design (L0 arms, L24 classifies).

---

### M4 — Mid-layer deterministic interrupts (and Mode B)

**Goal.** Traps at the interior test sites (blocks 8/16/24/32/40) that fire a trusted deterministic primitive mid-forward and reinject the result; begin Mode-B hidden-state injection research.

**Success invariant.** Each interior trap, disabled, is bit-identical to base (same `ResidualGateway` 0.0 proof, re-verified per site). Fired, the injected result is correct by the hold-out-proven handler, and a fired-vs-unaided contrast shows the trap supplies a capability the model lacks.

**Already done here.**
- The head→dispatch→inject→resume pipeline at a real layer is demonstrated: `north_syscall_e2e.py` reads L24, dispatches the proven det-block (0 model tokens), parks after the first token, injects the verified value, resumes — with the decisive "model UNAIDED vs head-gated" contrast on prompts North gets wrong alone.
- The deterministic plane is **hold-out-gated** (the project's standing rule): `hold_out_generality.py` measures dpll 3000/3000, symbolic_differentiate 2000/2000, grapheme 2000/2000 vs independent oracles (brute-force SAT, finite-difference, `regex \X`). `docs/shape-catalog.md` / `docs/north-shape-system-final-report.md`: the trusted library is **hold-out-proven** (18/19 residual contracts deterministic; 9 cyber primitives 100% hold-out). A primitive is not an opcode until it passes this gate.

**Remains to build.**
- Install + re-verify the identity invariant at sites 8/16/32/40 (currently proven at 24; the script runs at any `gate_at`).
- **Mode B**: train a `ResidualGateway`-shaped injector to write the opcode result into residual space at k+1, gated behind the same disabled-bit-identity invariant. This is the research-grade piece.

**Risks.** Mode B can diverge silently (the 2.943 upcast bug class) — it must keep the `.astype(x.dtype)` discipline and a per-site identity gate. Mid-stream injection without a token boundary is unproven; Mode A remains the default and Mode B ships only if it beats Mode A under the commit trap.

---

### M5 — On-device / in-kernel scheduler (GPU megakernel)

> **Status (2026-06-23): the M5 *policy-fusion core* is BUILT + verified on real North.** M5 was scoped (with P. Lamy) to its high-value, tractable core — **scheduler-owned dynamic-k** (fuse the *policy* entropy/cost→k, not a constant k; the in-API portion of §3d) — rather than the raw-Metal persistent megakernel described below. Five rungs landed: **R1** `kernel/k_policy.py` KPolicy (CPU 34/34) → **R2** `kernel/moe_entropy.py` inert MoE gate-softmax entropy tap (`==0` token-for-token on North, 6/6) → **R3** calibration gate (`m5_calibrate_entropy.py`: gate-entropy is a strong but **inverted**, single-suite signal that survives a length control → kept *unwired*; KPolicy's lever stays validate-the-branch / disagreement) → **R4** `kernel/k_cost.py` cost accounting + sysinfo sensor (15/15) → **R5** dynamic-k wired into the `EscalatingSolver` Tier-2 width, demonstrated live on North (idle→k=5 / busy→k=1 / budget→k=2; thin by default when Tier-1 passes). **The literal raw-Metal GPU megakernel below is a documented NICE-TO-HAVE, deliberately NOT built (DECISION 2026-06-23).** Rationale: its payoff is *structural only* (no CPU on the hot path / on-GPU scheduler / in-kernel dispatch) — North's decode is already GPU-bound (~70 tok/s on a 30 GB MoE), so there is **no decode-speed multiplier** — and it adds **no new capability or correctness**: its success invariant is M2's *already-proven* online==offline (0 mismatches), merely *fused*. Against that thin, structural-only payoff it costs a raw-Metal / MTLCommandBuffer / C++ persistent-kernel systems build that lives *below* MLX's Python surface (a foreign stack, hard to hold to the `==0` bar, fights MLX's lazy-array model). **Build it only** for a genuine high-throughput **serving** regime (many concurrent lanes where per-token CPU↔GPU orchestration dominates) or a **systems-paper** contribution. **If perf is ever wanted, the low-risk middle path is the in-API fusion** (`mx.compile` step fusion, proven 1.20× in `mlx_fused_step.py`; `mx.fast.metal_kernel` function blocks, proven) — most of the practical speedup, none of the raw-Metal risk, stays in Python+MLX. See memory `project_scheduler_owned_k` + `project_neural_microkernel_endstate`.
>
> **M5 tuning & usage requirements (preserve these to use M5).** Mirrored in the bundle user guide (`docs/neural-microkernel-bundle/README.md` §5) and the setup guide (`BUILD-FROM-SCRATCH.md`).
> 1. **The uncertainty LEVER — non-negotiable.** Do **not** drive `k` from the raw MoE gate-softmax entropy: R3 calibrated it on North and found it **inverted** (high entropy → *more* likely correct, AUC ≈ 0.89) and single-suite, so naive use is backwards. `KPolicy` keeps `uncertainty_calibrated=False` for entropy (observe/log only). The proven lever is **validate-the-branch** — an independent, ground-truth signal (e.g. a branch that fails its hold-out gate) — fed via `KSignals.disagreement`, or via `uncertainty` only with `uncertainty_calibrated=True` from a genuinely calibrated source. ("fail→pass is not proof," applied to routing.)
> 2. **`KPolicy` knobs (`kernel/k_policy.py`):** `k_min` (floor; 0 = may stay in-model, 1 = always run one), `k_max` (hard ceiling — anti-runaway-via-k), `widen_threshold` (default 0.5; generalizes the syscall head's conf ≥ 0.5), `require_calibrated` (default **True** — honor the calibration trap), `cost_brake` (default 0.6 — discretionary widening braked to 0 at cost = 1), `risk_floor_at` (default 0.5 — risk above this mandates verify branches **cost cannot brake away**), `unit_cost`. Doctrine: **thin by default; spend only where calibrated uncertainty earns it and cost allows; risk floors mandatory verification.** `decide(KSignals)` → `KDecision{k, explore_k, verify_k, used_signal, reason, est_cost}`.
> 3. **Cost + budget (`kernel/k_cost.py`):** `CostModel.cost_from_sysinfo` → `cost∈[0,1]` via the binding-constraint rule (`max` over host load-per-core, memory %used, scheduler queue depth). Knobs `queue_soft_cap` (8), `load_saturate` (1.0), `mem_floor` (0.5). **Requirement:** install `psutil` for memory pressure to register (else memory → 0; load+queue still drive cost). `charge_budget(decision, remaining, unit_cost)` caps `k` to the budget (anti-runaway at the budget level).
> 4. **Wiring (reference: `kernel/escalate_solve.py`):** `EscalatingSolver(model, k_policy=KPolicy(...), importance=…, trap_budget=…, cost_source=…)` — additive/backward-compatible (`k_policy=None` → the original fixed `k=3`). Lever there = Tier-1 hold-out-gate severity (independent) + sysinfo cost + task importance. Verify: CPU `test_k_policy`/`test_k_cost`/`test_k_policy_ladder`; North `inv_moe_entropy.py`, `m5_calibrate_entropy.py`, `m5_dynamic_k_north.py`.

**Goal.** Fuse the scheduler into the device: decode + blocks + dispatch in one GPU megakernel, CPU off the hot path. The end-state in `docs/freebsd-scheduler-lessons.md` §3d.

**Success invariant.** Same correctness as M2 (online == offline, 0 mismatches) but with *fused*, not *accounted*, compute — the honest gap SYNTHESIS.md flags ("the toy counts layer-applications; it does not fuse a GPU kernel… closing that is a systems project").

**Already done here.**
- The **dynamics** are real and runnable, just CPU-accounted: `scheduler.py` Scenarios B–E measure early-exit ~30% compute saved, SRF-learned cutting mean latency 34% and recovering **86% of the oracle gain** from the in-weights halt signal, learned-syscall **100% detect / 512-512 correct** at util 0.75, and model-urgency priority cutting HIGH-prio latency 42%.
- The doctrine that the deterministic ops belong **on-stack, not as a remote syscall**, is already articulated and partially proven: freebsd-lessons §3a + `north_det_block.py` (constraint masking inside North's MLX generation loop on-device), and the recognition that North's MoE gate *is* a dispatcher (a deterministic expert is the in-graph co-processor).

**Remains to build (a systems project).**
- A Metal/MLX megakernel that runs blocks + the gateway/scheduler decision + opcode dispatch without leaving the device; CPU only for true I/O syscalls (the small external residue: untrusted code exec, network/files).
- This is explicitly **unbuilt** and the largest engineering lift; M0–M4 are the de-risked path to it.

**Risks.** The speedup is currently *accounted, not fused* (SYNTHESIS.md, honest boundary). The RL-grade learned halt that would make adaptive depth maximally effective is also unbuilt on North (SFT provably stuck at `logp ≈ −23`; the toy `rl_halt.py` de-risks the actor-critic recipe but North's halt-head is the "one real build" still pending). M5 should not block on the learned halt — the validation-gated deterministic halt (task #24, completed) is the correct stand-in.

---

### Cross-cutting: validation levels (the proof-ledger ladder)

Every trap commits at the highest level it can reach, recorded in the proof-ledger:
- **syntax/schema** — `output_validators.detect_output_type` / `is_complete_valid` (form_verify).
- **property** — structural checks (`shape_appliance._struct_eq`, contract/signature match `_contract_matches`).
- **oracle** — `check_cases` runs the candidate against provided cases in the SIGALRM sandbox.
- **proven** — the handler is a hold-out-gated primitive (`hold_out_generality.py`, 100% vs independent ground truth) — no per-call check needed; correctness is structural.

This ladder is what makes the commit trap "proof-carrying": a committed output names the validation level it cleared, and proven-level opcodes need no trust beyond the standing hold-out gate.

### Summary of build order vs. proof status

| Stage | Invariant | Status in this repo |
|---|---|---|
| M0 scheduler owns I/O | learned request beats lexical baseline; one dispatch chokepoint | **mostly built** — syscall head 98–100% across 4 models + placebo controls; needs the unified `dispatch()` chokepoint |
| M1 commit trap | only output path; validate/repair/commit | **mostly built** — appliance validate/bail/escalate → 100/100; needs layer-48 hook + proof-ledger |
| M2 clean-context agents | online==offline, 0 mismatch; typed-result merge | **mechanism proven** (toy 0-mismatch, North park@24 = 0.0); needs KV-handle hold/switch + mailboxes |
| M3 entry trap (post-L0) | disabled ⇒ bit-identical (=0.0) | **invariant proven at L24** (and the 2.943 bug fixed); needs head re-point to L0 + capability write |
| M4 mid-layer det interrupts + Mode B | per-site identity 0.0; fired result correct (hold-out-proven) | **e2e demoed at L24**; det plane hold-out-gated; **Mode B PROVEN (MVP)** — residual write sets North's output, causal=1.0, placebo-controlled (`mode_b_north.py`); scope = 1 task/site/single-digit |
| M5 GPU megakernel | fused (not accounted) compute, same correctness | **policy-fusion core BUILT** — scheduler-owned dynamic-k, 5 rungs verified on North (`k_policy`/`moe_entropy`/`k_cost` + live ladder); raw-Metal megakernel + RL halt still **frontier** |

Key file references (all absolute): `/Users/pierrelamy/Desktop/Test Harness/finetune/gpt-oss-unsloth/scripts/north_gateway_invariants.py` (identity sidecar + park/resume, proven 0.0), `.../north_syscall_head.py` + `.../north_syscall_adversarial.py` + `.../north_syscall_controls.py` + `.../syscall_cross_model.py` (learned trap router + gates), `.../north_syscall_e2e.py` (head→dispatch→inject→resume), `.../tool_dispatch.py` + `.../tool_dispatch_native.py` (Mode-A reinjection, per-model dialects), `.../shape_appliance.py` (commit-trap router + sandbox), `.../hold_out_generality.py` + `.../det_primitives*.py` (proven deterministic plane), `/Users/pierrelamy/Desktop/Test Harness/experiments/neural-scheduler/scheduler.py` + `SYNTHESIS.md` (the toy kernel, Scenarios A–E), `/Users/pierrelamy/Desktop/Test Harness/docs/freebsd-scheduler-lessons.md` (privilege-boundary doctrine), and result logs under `/Users/pierrelamy/Desktop/Test Harness/reports/north-m0/`.

---

## 4. Research positioning & novelty

This architecture sits at the intersection of several active research families. None of them, individually, is the thing we are building; the contribution is their *synthesis* — verified deterministic interrupt handlers inside the LLM forward path — and the part of that synthesis we can already back with measured, hold-out-gated results on real models.

### 4.1 Mapping: each component to its closest research family

| Our component | Closest research family | What exists there | What this project adds beyond it |
|---|---|---|---|
| **Scheduler as kernel; LLM as CPU** ("learned proposes, kernel disposes"; three planes: neural / deterministic / scheduler) | **AIOS** (LLM agent OS kernel); FreeBSD ULE as the design template (`docs/freebsd-scheduler-lessons.md`) | An OS-style kernel *around* an LLM agent — context manager, memory manager, tool scheduler as external services orchestrating black-box model calls | The kernel reaches *inside the forward pass*. The privilege boundary is derived, not asserted: validation = a syscall the token-generator structurally cannot perform on itself (`freebsd-scheduler-lessons.md` §3); the refinement (§3a–3c) shows most of the "kernel" is an *on-stack co-processor* (deterministic expert / constraint mask / in-runtime call on unified memory), not an external service — strictly tighter than AIOS's process-boundary model. |
| **Continuation frame; HOLD / SWITCH / RESUME; clean-context in-process agents** | **MemGPT / Letta** (virtual context management, OS-style paging of context, interrupts/yields) | Context as paged memory; the model "interrupts" to swap context in/out; explicit memory hierarchy | A *first-class continuation frame* — the residual stream / KV cache captured at a layer boundary IS the suspended process, proven **bit-exact** (`north_gateway_invariants.py`: park@L24 → serialize to safetensors → resume = `max_diff 0.0` on the real 49-layer cohere2_moe stack; `reports/north-m0/gateway_invariants2.log`). MemGPT pages *text*; we park/resume the *computation itself*. Agents merge **typed results, never transcripts**. |
| **KV-cache as process memory; park/resume; serving scheduler** | **vLLM / PagedAttention**, **SGLang / RadixAttention**, **TensorRT-LLM / FlashInfer** | KV cache treated as virtual memory (paging, prefix sharing, copy-on-write); production schedulers for throughput; prefix-cache machinery | We reuse this machinery for a *different purpose*: park/resume as a **context switch for a scheduler executing trusted handlers**, not just for throughput/sharing. The serving-infra prefix-cache is exactly the "cheap context switch" FreeBSD ULE requires (`freebsd-scheduler-lessons.md` §1). Trim-restore park/resume verified exact on North (task #26). |
| **Post-last-layer COMMIT trap; output valid by construction** | **XGrammar / constrained decoding** (grammar-masked logits, structured output) | A deterministic mask makes output conform to a grammar *during* decoding — output is valid by construction, no post-hoc check | Constrained decoding is the commit trap's *cousin* and its deepest form — proven on North (`north_det_block.py`: a deterministic constraint block runs in MLX's decode loop on-device, forcing properties like "no imports" / fenced block by masking before argmax; `freebsd-scheduler-lessons.md` §3a-1). We generalize from grammar-shaped masks to a **validate→repair→early-bail→commit** trap with an escalating validation ladder (syntax / schema / property / oracle / proven) and park/resume rollback on misprediction — the *only* output path to the user. |
| **Scheduler-owned tool use; deterministic offload; capability system** | **Toolformer / PAL / Gorilla / LLMCompiler** | Model learns to emit tool calls / offload computation to an interpreter / plan a DAG of tool calls; tools are deterministic and external | The model never touches I/O; it *requests typed opcodes* and the scheduler authorizes/executes/logs (a capability system). The **learned trap router is in-weights**, not a text protocol: a classifier head on the L24 residual decides *whether and which* primitive to dispatch — proven **100% routing / 100% trap-detection** held-out on North (`north_syscall_head.py`; `syscall_head.log`), generalizing to keyword-free paraphrases + hard negatives at **98.1%** while bag-of-words collapses to 78.8% (`north_syscall_adversarial.py`), and replicating cross-model on gpt-oss / Mellum / Gemma (96–98%, fp 0.0%; `syscall_{gptoss,mellum,gemma}.log`). The native-dialect dispatch (`tool_dispatch_native.py`) parks/dispatches/resumes mid-stream with **0 model tokens** for the computed result. |
| **Mid-layer deterministic interrupts; entry-trap arm/disarm; adaptive compute** | **CALM, Mixture-of-Depths, LayerSkip, PonderNet** | Adaptive per-token/per-layer compute — early-exit, depth routing, learned halting; saves compute on easy tokens | Same *mechanism* (a per-layer always-routed gateway, MoD generalized into a dispatcher), repurposed: not just "skip layers to save compute" but "**trap to a trusted deterministic handler and inject a verified result.**" The zero-init identity gateway is proven bit-identical (`gateway_invariants2.log`: identity-init after L24 = `0.0`) — and the failure log (`gateway_invariants.log`: `2.94` FAIL) is itself a load-bearing result: bit-identity requires dtype-preserving residual cast, the exact precondition for a "trap-disabled == base" guarantee. The 129th-expert hazard (a trap stealing a top-8 MoE slot) is explicitly avoided by the sidecar design. |
| **Capability system; scheduler authorizes I/O; budgets** | **MCP + tool-use security** | A protocol for exposing tools/resources to models; permissioning lives in the host/transport around the model | We push the capability check *into the runtime's dispatch path*: typed opcodes, scheduler-authorized execution, budgets that prevent scheduler-mediated runaways (max traps/token, no recursion, same-op-once, failed-op-disables), and a proof ledger. This is the existing MoE gate reframed — a *deterministic expert* in an expert slot is the in-graph capability (`freebsd-scheduler-lessons.md` §3a). |

### 4.2 The novel contribution: AI-BPF — verified deterministic interrupt handlers in the forward path

The synthesis that no neighbor provides is **verified deterministic interrupt handlers installed at predefined neural hooks inside the model's own forward computation** — the inference analogue of eBPF (verified programs at kernel hooks). AIOS schedules *around* the model; MemGPT pages its *context*; vLLM/SGLang page its *KV*; XGrammar masks its *logits*; Toolformer/PAL offload to *external* interpreters; MoD/PonderNet adapt its *depth*. Each owns one face of the cube. We put a **trap layer between the blocks**, let an **in-weights router decide to fire it**, have the **scheduler execute a trusted primitive**, and **inject a verified result back into the live computation** — then resume the same forward pass.

Two specific, measured facts are what make the "microkernel" claim **literal rather than metaphorical**:

1. **The identity-trap invariant is proven exact.** A trap that is disabled is *bit-identical* to the base model (`max_diff == 0.0` on the real 49-layer MXFP8 cohere2_moe stack, `gateway_invariants2.log`), and parking/resuming a computation across a layer boundary never changes a single bit (`0.0`, same log). A continuation frame is therefore a *real* saved process state, not an approximation — the bedrock under "process = suspended forward pass." The earlier `2.94` FAIL → `0.0` PASS transition is the honest provenance of that guarantee.

2. **The handlers are hold-out-gated, not memorized.** A primitive is only trusted as a *shape* if it matches an independent oracle on thousands of novel instances: `dpll` 3000/3000 vs brute-force, `symbolic_differentiate` 2000/2000 vs finite-difference, `unicode_grapheme_count` 2000/2000 vs `regex \X`, `regex_to_nfa_match` 3910/3910, `WaveletTree` 5000/5000, `LinkCutTree` 1222/1222 (`docs/shape-catalog.md`; `hold_out_generality.py`). This is what makes the "verified" in "verified inference kernel" earned: the deterministic plane is correct *by independent construction*, so a handler's injected result carries a real correctness guarantee the stochastic model cannot self-supply. This is the standing methodology — no primitive trusted until it passes the hold-out gate at 100%.

Together: an **in-weights syscall router** (100% held-out, causally validated against shuffled-label and random-feature placebos in `north_syscall_controls.py`, and beating bag-of-words across four model families) *proposes*; a **thin general kernel** *disposes* by dispatching a **hold-out-proven deterministic handler** across an **exact, reversible continuation frame**. That is a microkernel by its mechanisms (cheap context switch + privileged-op trap + mechanism/policy separation), not by analogy.

### 4.3 The genuinely-open research gap: result reinjection

> **Status update (2026-06-23): Mode B is now DE-RISKED as an MVP on real North.** A trained result-conditioned residual injector (zero-init `ResidualGateway` → `kernel/mode_b.py`) at site 24, with North frozen, makes the model **emit a kernel-injected single-digit value through 24 downstream MoE layers** — held-out vs the independent oracle (`mode_b_north.py`): deployment (inject true value) **1.0**, **causal/faithfulness 1.0** (the output tracks the injected value *including wrong values* → it genuinely *writes* the result, not recomputes it), **placebo → chance** (a decoupled-target injector collapses), and **`==0` when disabled**. CPU toy first (`mode_b_toy.py`, placebo-controlled) then the substrate ==0 invariant on North (`inv_mode_b.py`). Honest scope: one task / one site / single-digit values — a placebo-controlled proof-of-mechanism, not a general multi-token/multi-primitive system. The text below describes the *pre-result* open status; Mode B has now cleared its first rigorous bar.

The one component that is **not** yet de-risked is *how* the verified result re-enters the computation:

- **Mode A — token-boundary trampoline (safe; build first).** Park at a token boundary, inject the result as text/tokens, resume. This is *proven working*: mid-stream `[[fn]]` and native-dialect dispatch splice a verified value back into the stream with 0 model tokens (`tool_dispatch_native.py`, `north_fncall.py`: `count_byte('strawberry','r')→3` mid-generation), and the e2e head-gated pipeline (`north_syscall_e2e.py`) shows the model-unaided-wrong → handler-right contrast. The cost is that it operates at the text interface, not the representation.
- **Mode B — hidden-state vector injection (research-grade; later).** Inject the result *as a hidden-state edit* at layer k and resume from k+1 — operating on the representation directly. The substrate exists (the zero-init `ResidualGateway` is the injection point, and §3a's "deterministic expert in an expert slot" is the in-graph realization), but **whether a deterministic vector edit reliably steers downstream layers without destabilizing them is unproven.** This is the real frontier: it is the difference between a trampoline that the model reads as text and a true *neural* interrupt handler that writes to the residual stream. It is also where the confidence-vs-correctness floor (`confidence_probe.py`; North's l80 regex gap) bites — anything reading a wrong representation is wrong, so Mode B must be validated against the same hold-out oracle the handlers already pass, not against the model's own confidence.

**Naming.** Of the candidates, *Verified Inference Kernel* and *Interruptible Neural Runtime* are the most defensible given what is proven (the verified handlers + the exact, interruptible continuation); *Neural Microkernel* is justified by the three-mechanism kernel; *Proof-Carrying Inference Runtime* is aspirational until the proof ledger and Mode B are built.

---

## 5. Safety, invariants & the trust boundary

This section states the non-negotiables for the Interruptible Neural Runtime and grounds each one in code, a measured result, or a lesson the repo already paid for. The governing principle is the one the FreeBSD note already extracted (`docs/freebsd-scheduler-lessons.md` §3): **the model is the CPU/process, the scheduler is the kernel, validation is a syscall.** Safety is what keeps that privilege boundary real rather than decorative. Where a guarantee is unbuilt, it is marked so.

---

### 5.1 The no-op parity invariant battery (trap installed everywhere, activated nowhere → bit-identical)

The first and most load-bearing safety property: **installing the trap infrastructure must be observationally null until something deliberately fires.** If grafting a trap point changes a single output bit, every downstream "the trap helped" claim is uninterpretable, because you can no longer attribute a delta to the handler vs. the plumbing. This is exactly the discipline the project already enforced for surgical edits via placebo controls; here it becomes a forward-pass invariant.

**Proven, on real North, exact.** `finetune/gpt-oss-unsloth/scripts/north_gateway_invariants.py` reproduces both mechanism invariants on the 49-layer `cohere2_moe` MXFP8 stack (not a toy):

- **Invariant 1 — identity-init graft.** A zero-init `ResidualGateway` (`up.weight = 0`, so `delta == 0` at init) inserted after layer 24 yields `max|gated − base| = 0.0` (lines 82-85). This is the sidecar-identity-trap requirement: *trap-disabled MUST be bit-identical to base.* It is the substrate the learned syscall head rides on — `docs/north-shape-system-final-report.md:229` calls it "the noop/bridge expert, proven bit-identical `0.0` at L24."
- **Invariant 2 — exact park/resume.** Run layers 0..24, serialize the residual to disk (`mx.save_safetensors`), reload, resume 24..49 → `max|resumed − base| = 0.0` (lines 87-89). This is the continuation frame realized: a suspended forward pass is a delimited continuation, and capturing/restoring it never changes a result.

The `RESULT: "PASS"` gate is `(p_ident == 0 and p_park == 0)` — strict zero, not a tolerance (line 94). This is the right bar and the architecture should keep it: **the no-op parity gate is `== 0`, never `< epsilon`.**

**The dtype-upcast hazard — found, documented, fixed, and it is the canonical trap-safety bug.** This is the single most important cautionary tale for anyone inserting trap layers, and the repo paid for it in real divergence. Documented in three places:

- The fix lives inline in `north_gateway_invariants.py:38-40`: the gateway casts its delta back with `delta.astype(x.dtype)`, with the comment *"cast to x.dtype so the residual is not silently upcast (bf16/quantized base) — preserves bit-identity."*
- `docs/north-ratchet-final-report.md:361-364` records the failure mode and the measured magnitude: *"park/resume on a bf16/quantized stack requires dtype-preserving serialization — a naive `float32` round-trip (or a `x + 0_float32` gateway) silently upcasts the residual and breaks bit-identity (**it diverged by 2.94 before the fix**)."*
- `docs/paper/expert-row-surgery-and-intelligent-routing.md:367-374` repeats it as a formal lesson.

The SYNTHESIS table (`experiments/neural-scheduler/SYNTHESIS.md:31`) flags this as the *genuine new lesson the real stack taught that the float32 toy could not*: **"dtype-preserving serialization required (bf16 upcast diverged 2.94 before fix)."** The mandatory rule for the architecture: any trap, continuation-frame serializer, or hidden-state injection (Mode B especially) must round-trip in the base model's native dtype; an accidental `float32` promotion anywhere on the residual path silently breaks the parity invariant and the bit-identity guarantee is gone. Mode B (hidden-state vector injection) is precisely where this lands again, which is one more reason it is correctly staged after Mode A (token-boundary trampoline).

**Batch-lane / continuation isolation.** Proven on the toy (`experiments/neural-scheduler/autoregressive.py`), still toy-only on North: a budgeted scheduler interleaves generation of N sequences, parking/resuming each one's KV cache between ticks, and **interleaved output == solo output for every process, mismatches = 0** (lines 12-13, 222-232, oversubscribed N//W×). This is the property the HOLD/SWITCH/RESUME ops and the clean-context agents depend on: parking lane A and running lane B must not perturb lane A's continuation. SYNTHESIS.md rows ("AR KV-cache continuation: exact disk park/resume + interleaved generation, 0 mismatches") and the FreeBSD note ("KV-cache park/resume — verified exact on North … the toy: 0 mismatches") both ground it. **Honest boundary:** lane isolation under true batched interleaving is proven on the toy transformer; on North only single-lane park/resume bit-identity is proven (`gateway_invariants2.log`). M2 (clean-context agents) must re-prove interleaved isolation on North before it is trusted — do not assume the toy result transfers to a 128-expert top-8 MoE without re-measurement.

**Save/resume bit-identity** is the disk round-trip in Invariant 2 above — the *strongest* form of park/resume (it survives serialization to `safetensors` and reload), and it is exactly the form a real continuation frame (`hidden_ref`, `kv_cache_ref`) needs if frames are ever persisted or moved off-device.

---

### 5.2 Trap-budget / anti-runaway rules

The scheduler can *cause* the failure it exists to prevent: if a trap can fire freely, or re-fire, or recurse, the scheduler-mediated loop becomes a new runaway surface. The budgets are grounded in two things the repo already measured.

**The documented reasoning-runaway lesson (the empirical case for hard budgets).** Per the runaway analysis (`memory/project_north_mini_runaways.md`, corroborated by `docs/north-ratchet-final-report.md`): on a 592-task battery North-Mini scored 388/592, and **157 of the 204 failures (77%) were reasoning-phase runaways** — greedy decoding enters `<|START_THINKING|>`, never emits `<|END_THINKING|>`, and runs to the token cap. SYNTHESIS.md quantifies the spread as a **135× token gap** (61 tok / 1s easy vs 8278 tok / 126s hard). Two findings carry directly into trap-budget design:

- **Sampling knobs do not robustly bound a runaway** (`repetition_penalty` was knife-edge: 1.3 cured l78, 1.2 too weak, 1.5 ran away; best config 3/5; c016/x01 immune to every setting). The lesson: **you cannot make runaway structurally impossible with soft sampling pressure — you need a hard structural cap.** For the scheduler this means trap budgets must be hard counters enforced by the kernel, not penalties nudging the model.
- The cure that worked was *structural* (reasoning-off makes the runaway state unreachable), and the recommended production design is **adaptive**: stay normal, fall back only on `finish_reason:"length"`. That is the template for trap budgets — enforce a cap, escalate/disarm on hit, don't pay the cost on the easy path.

**The SIGALRM / exec-timeout precedent (the kernel-enforced wall-clock cap, already in the runners).** The architecture's "validation is a privileged syscall executed by the kernel under a timeout" is not aspirational — the harness already does exactly this for the validator sandbox:

- `harness/sandbox/local.mjs:83-136`: `runCommand` runs the candidate under `sandbox-exec` with a hard `timeout` (default `20_000` ms, `local.mjs:42`) passed to `execFileSync`; on expiry the child is killed and the result carries `killed`/`signal` (lines 133-134). This is the SIGALRM-equivalent: the kernel, not the model, owns the wall clock.
- `harness/evaluators/auto-diff.mjs:31`: `TEST_TIMEOUT_DEFAULT_MS = 60_000` for the test-execution validator, with per-step caps (10-15 s) for setup/teardown.
- `harness/sandbox/docker.mjs:84-85`: on timeout the kernel issues `docker kill` — the containerized form of the same preemption.
- `shape_appliance.py:101-112`: the in-process det-block executor wraps every primitive call in `_timed(...)` using `signal.SIGALRM` + `setitimer(ITIMER_REAL, t)` (default 5 s), raising `_Timeout`. **This is the literal SIGALRM the architecture wants** — a deterministic-plane primitive that hangs is preempted by the scheduler, not allowed to wedge the runtime.

The mapping the FreeBSD note already drew (`freebsd-scheduler-lessons.md:22`): the fixed budget is a *callout/softclock timeout* — "the worst scheduler, a blind callout … the fallback, not the policy." So the budget rules for the runtime are:

| Budget rule (from the vision) | Grounding in the repo |
|---|---|
| **max traps/token, no recursion** | the dispatch loop already caps calls: `tool_dispatch.py:57,71` enforces `max_calls=6` and dispatches each distinct `[[fn]]` *exactly once by index* ("robust to the model echoing `]]`") — the no-recursion / same-op-once primitive in concrete form |
| **same-op-once** | `tool_dispatch.py:69` ("dispatch each DISTINCT `[[...]]` exactly once by index") + the JIT compose path memoizes the validated block (`freebsd-scheduler-lessons.md:252`, "exact, reusable (memoized)") |
| **failed-op-disables** | the dispatch loop catches handler exceptions and substitutes `<err:...>` rather than retrying (`tool_dispatch.py:75-76`); the per-primitive `_Timeout` (`shape_appliance.py:108`) returns failure rather than re-entering |
| **hard wall-clock cap** | `local.mjs` 20 s / `auto-diff.mjs` 60 s / `shape_appliance` 5 s SIGALRM — kernel-owned, model cannot extend it |
| **adaptive, not uniform** | the routing result: uniform bounded-reasoning regressed to **89/100** (`north-ratchet-final-report.md:401-407`); routed = **100/100**. Budgets must be applied on escalation, not charged to the fast path |

The one rule still **only partially built**: a formal global "max traps per request" / "trap-budget exhaustion → commit-or-bail" counter on the *neural* trap path. `max_calls=6` exists for the text `[[fn]]` protocol; the learned-head trap path (`north_syscall_e2e.py`) fires once per prompt at the L24 decision point and has no multi-fire loop yet, so the global budget is currently structural (one decision point) rather than an explicit accounted counter. M1's commit trap is the natural home for "budget exhausted → early-bail" and should add the explicit counter.

---

### 5.3 The capability system + scheduler-owned I/O as the prompt-injection defense

**The non-negotiable: the model never touches I/O, network, files, or tools directly — it only emits typed opcode *requests*; the scheduler authorizes, executes, logs, and returns a typed result.** This is the privilege boundary, and it is also the prompt-injection defense: a prompt-injected instruction inside model output is, at most, a *request* for an opcode the model is not capability-granted to invoke, which the scheduler refuses. The model cannot escalate its own privileges because it structurally cannot execute the privileged op — the same reason a user-mode process can't do its own disk I/O (`freebsd-scheduler-lessons.md:44-69`).

Grounding:

- The trap protocol is *request-only* by construction. In `tool_dispatch.py`, the model emits `[[fn(args)]]` and the **runtime** parses (`_CALL_RE`, `ast.parse`), looks the name up in a fixed `TOOLS` allowlist (lines 33-38), executes, and splices the result back. The model never holds a handle to the function — it names an opcode and the kernel disposes. An unknown name simply doesn't match a key; arbitrary code in the call site can't run because `_exec` only resolves `tools[node.func.id]` from the allowlist and only `ast.literal_eval`s the arguments (`tool_dispatch.py:51-54`) — no `eval` of attacker-controlled text.
- The learned trap path is even tighter: in `north_syscall_e2e.py` the head emits only a **class label** (one of 5: `none`/`count_byte`/`c_gcd`/`is_prime`/`shortest_dist`) and the kernel does the arg extraction and dispatch (`TOOLS` dict, lines 33, 102-114). The model's entire I/O vocabulary is a 5-way categorical — the smallest possible attack surface.
- The execution sandbox enforces the capability at the OS level: `local.mjs:27-37` ships two seatbelt profiles, and the default `SANDBOX_PROFILE_NO_NET` is `(deny network-outbound)(deny network-inbound)`. Network is a *capability the scheduler grants*, off by default (`allowNetwork: false`, `local.mjs:43`). This is the capability system in its crudest but real form: the validator-execution syscall runs net-denied unless explicitly authorized.
- The FreeBSD note names the three kernel services this defense rests on (`freebsd-scheduler-lessons.md:73-75`): the validator sandbox is the **syscall handler**, and "tool calls / external I/O" are listed as **kernel-only** privileged ops (`:61`).

This is the M0 milestone ("scheduler owns I/O") and it is the most defensible piece because it's the OS model: capabilities, not trust, gate privileged action. **The architecture must keep the model's I/O surface a typed opcode enum, never free-form tool-call text that the scheduler trusts** — the `[[fn]]` text protocol was explicitly called a "portable baseline" to be replaced by each model's native, parsed tool-call format (`tool_dispatch.py:11-12`); the typed-enum head is the safer end-state.

---

### 5.4 Validation levels (syntax / schema / property / oracle / proven) and the proof-ledger / replayability requirement

The commit trap is "the only output path," so its validator must be honest about *how strongly* a candidate was checked. The repo already operates a graded ladder; here it is named explicitly, weakest→strongest:

| Level | What it proves | Grounding in the repo |
|---|---|---|
| **syntax** | parses / is a fenced code block / extractable | `output_validators.detect_output_type` + `is_complete_valid` (`shape_appliance.py:20,162-165`) — the form-verify "bail" gate |
| **schema** | matches the declared output type/shape | `form_verify` / the type+probe validator (`freebsd-scheduler-lessons.md:73,229`) |
| **property** | satisfies invariants on inputs (no exec of the candidate's own code on real cases needed) | `auto-code-property.mjs`; the **constrained-decoding** form is *valid by construction* — `north_det_block.py` masks forbidden tokens (e.g. "no imports") on-device so the property holds without a post-hoc check (`north_det_block.py:9-13,47-52`) |
| **oracle** | differential-tested against an independent reference | the regex skill-cassette's **40-probe differential oracle** (`north-ratchet-final-report.md:254-255`): the model's matcher is diffed against the cassette oracle over a battery "spanning every feature; any disagreement or crash = wrong" — this is what gates l80 |
| **proven** | hold-out gate: thousands of novel instances vs. independent ground truth, 100% | the standing rule (`memory/project_shape_generalization_methodology.md`); measured in `hold_out_*.py` and `north-shape-system-final-report.md:33-35`: `dpll` **3000/3000** vs brute-force 2ⁿ, `symbolic_differentiate` **2000/2000** vs numeric finite-difference, `unicode_grapheme_count` **2000/2000** vs regex `\X`; "9 general primitives, every one 100% hold-out" (`:91`) |

The two rules the architecture must carry from this:

1. **A deterministic primitive is not trusted until it passes the hold-out (proven) gate** — the standing methodology rule (no primitive trusted until 1000s of novel instances vs independent ground truth, 100%). "Teach shapes, not memorized answers"; a memorized lookup scores ~0% held-out (`north-shape-system-final-report.md:28-29`). The deterministic plane behind the trap is *only* as trustworthy as its weakest member's validation level, so each handler must declare its level.
2. **Confidence ≠ correctness — the commit trap must validate, not trust the model's halt.** `confidence_probe.py` (toy) and l80 (North) are the same finding: a verifier catches errors a confident model misses, and there is an intrinsic confidently-wrong floor (`SYNTHESIS.md:40`, `:54-59`). So the commit trap's gate is the validator's verdict, never the model's self-reported confidence. The learned syscall head *proposes*; the kernel's validation *disposes*.

**Proof-ledger / replayability — the strongest gate, and the cautionary result.** The promotion ledger already exists: `reports/north-m0/ledger/north-w47-r1.bundle.json` is an artifact bundle with `base_model.config_sha256`, `tokenizer_sha256`, `mlx_lm_version`, adapter `safetensors_sha256`, the exact `runtime_policy` (`temperature: 0`, `retries: 0`), full causal attribution with a **placebo control** (`best_real_absdelta: 2.75` vs `best_placebo_absdelta: 0.0`), and a `rollback` path (`north-ratchet-final-report.md:80`). Determinism is what makes the ledger replayable: temp=0 → attempts are deterministic (hence `retries: 0`, `feedback_no_retries_at_temp0.md`), so a bundle re-run reproduces bit-for-bit.

The decisive lesson is the bundle's own `specificity_caveat`: an identical contrastive LoRA trained on **12 placebo (inactive-at-fork) rows also flipped the fork (+6.5) and also scored warmups-50 50/50 with zero regression.** The verdict recorded in the artifact: *"fix-specificity NOT proven … Fork-margin flip is necessary-not-sufficient; only a harder shadow/regression gate can discriminate."* This is the safety rule, paid for in a real measurement: **fail→pass alone is NOT proof.** A trap/handler that makes a failing case pass must clear placebo-row causal controls + isomorphic/shadow evals + an artifact-bundle promotion before it is trusted (`memory/project_north_ab_ratchet_design.md`). The proof-ledger is how the commit trap stays honest: every commit carries hashes, the validation level it cleared, the policy, and a rollback — so a "verified inference kernel" is auditable and replayable, not just asserted.

---

### 5.5 Why agent transcripts must NOT merge back — only typed results

The clean-context in-process agents (M2) get separate address spaces and **typed mailboxes**, and the merge-back rule is: **merge only the typed result, never the transcript.** The justification is grounded in the project's own findings:

- **Transcripts are an injection and corruption vector; typed results are not.** §5.3's defense only holds if what re-enters the parent context is a *validated, typed value*, not free-form sub-agent prose. If transcripts merged back, a prompt-injected or runaway sub-agent (and runaways are the dominant failure mode — 77%, §5.2) would pour its unvalidated text straight into the parent's context, defeating both the capability boundary and the context hygiene. A typed result (e.g. `count_byte(...) → 3`, hold-out-proven) carries no instructions to be injected.
- **The whole point of clean context is to keep the scarce resource — context/compute — uncontaminated** (`SYNTHESIS.md:20`, "the scarce resource being scheduled is context/compute"). Merging a full transcript reintroduces exactly the runaway-prone, token-bloated state the scheduler exists to bound. The typed-mailbox discipline is the agent-level analog of the syscall boundary: cross only validated, typed values across the address-space boundary.
- It mirrors the existing dispatch contract: `tool_dispatch.py` injects only ` = {res} ` (the typed value), not the det-block's internal computation; the syscall head returns a *class label and a value*, not a narrative. The agent boundary should be the same shape — a typed mailbox, validated at the level §5.4 demands before it crosses.

**Status:** M2 clean-context agents are **not yet built on North**; the substrate (exact park/resume, toy-proven lane isolation) exists, but the agent address-space + typed-mailbox layer and the on-North interleaved-isolation re-proof are open work. The merge-back rule should be a hard invariant of that build from day one, not retrofitted.

---

### Summary of the non-negotiables

1. **Parity gate is `== 0`, not `< ε`** — trap-disabled is bit-identical to base (`north_gateway_invariants.py`, both invariants `0.0` on real North). Watch the dtype-upcast hazard (`astype(x.dtype)`; the 2.94 divergence is the documented scar) — it is the canonical Mode-B risk.
2. **Budgets are hard, kernel-owned counters, applied adaptively** — sampling pressure can't bound a runaway (knife-edge rep-penalty); SIGALRM/exec-timeout already enforces the wall clock (`local.mjs`, `auto-diff.mjs`, `shape_appliance._timed`); `max_calls`/same-op-once already exist in `tool_dispatch.py`; uniform application regresses (89 vs routed 100). **Gap:** an explicit global trap-budget counter on the learned-head path (belongs in the M1 commit trap).
3. **The model's I/O surface is a typed opcode enum the scheduler authorizes** — never free-form tool text it trusts; net-denied by default (`local.mjs` seatbelt). This is the prompt-injection defense.
4. **Validation is graded (syntax→schema→property→oracle→proven) and the commit trap trusts the verdict, not confidence** — proven = hold-out 100% vs independent ground truth (3000/3000 dpll, etc.); confidence≠correctness is measured.
5. **Promotion is ledger-gated and replayable, and fail→pass is not proof** — the `north-w47-r1.bundle.json` placebo result is the evidence: a placebo edit passed the weak gate, so trust requires placebo controls + shadow/isomorphic evals + the artifact bundle.
6. **Agents merge typed results, never transcripts** — the typed-mailbox boundary is the agent-level syscall boundary (M2, unbuilt; make the rule an invariant from the start).

Key files: `finetune/gpt-oss-unsloth/scripts/north_gateway_invariants.py`, `.../north_syscall_e2e.py`, `.../north_syscall_controls.py`, `.../north_syscall_adversarial.py`, `.../tool_dispatch.py`, `.../shape_appliance.py`, `.../north_det_block.py`, `.../hold_out_generality.py`; `harness/sandbox/local.mjs`, `harness/sandbox/docker.mjs`, `harness/evaluators/auto-diff.mjs`; `docs/freebsd-scheduler-lessons.md`, `docs/north-ratchet-final-report.md`, `docs/north-shape-system-final-report.md`, `experiments/neural-scheduler/SYNTHESIS.md`, `experiments/neural-scheduler/autoregressive.py`, `reports/north-m0/ledger/north-w47-r1.bundle.json`.

---

## 6. Applications (brief)

Each rides the same two levers the architecture buys — **localized failure injection** (a trap fires only where a capability is missing, leaving everything else bit-identical) and **early bail** (the commit trap stops bad output before it reaches the user):

- **Verified structured-output gateway** — the commit trap as a constrained-decoding-plus-validation front door: JSON/SQL/grammar outputs are valid by construction (mask) and schema/property-checked at commit, bailing on the first invalid token instead of emitting malformed structure.
- **Self-testing code agents** — a mid-layer/commit trap fires `exec_check` (the SIGALRM-sandboxed `check_cases`) on the candidate; failing code triggers localized repair-escalation, passing code commits with a proof-ledger entry — exactly the 350/350 appliance, kernelized.
- **SOC appliance** — the hold-out-proven cyber primitives (9 at 100%) become opcodes the entry trap arms by task-shape; detections commit only at oracle/proven level, with the typed-result/no-transcript boundary keeping injected log content out of the model's control flow.
- **Proof-carrying RAG** — retrieval is a scheduler-authorized opcode (capability + net-deny by default), retrieved facts merge as *typed results into the mailbox, never as transcript*, and the commit trap records which validation level each cited claim cleared.
- **Verified IaC** — generated infra plans hit a commit trap that property-checks invariants (no public bucket, no `0.0.0.0/0`) by construction and early-bails on violation, with a replayable ledger entry (hashes + policy) per committed plan.

---

## 7. Glossary

- **Post-L0 Entry Trap** — a mandatory identity sidecar after the dense block 0; does scheduler ingress: task-shape routing, capability assignment, and arm/disarm of later traps. Disabled ⇒ bit-identical to base (proven substrate at L24; re-point to L0 is M3 work).
- **Post-Last Commit Trap** — the mandatory trap after the final layer (48); the **only** output path. Validates the candidate (syntax→schema→property→oracle→proven), repairs or early-bails, and commits with a proof-ledger entry.
- **Continuation Frame** — the explicit suspended-process record `{request_id, token_pos, layer_idx, hidden_ref, kv_cache_ref, decode_state, capabilities, budgets}`; `hidden_ref` + `kv_cache_ref` are the partially-evaluated forward pass, proven serializable/resumable bit-identically.
- **Scheduler Mailbox** — the typed channel by which a frame (or sub-agent) receives results; the kernel delivers responses by thread-id. Only typed, validated values cross it — never transcripts.
- **Agent Address Space** — a sub-agent's isolated continuation frame + clean KV cache; processes never attend across each other, so a context switch between them is provably exact (toy-proven; M2 on-North).
- **Primitive ABI** — the typed-opcode contract between model and kernel: `(opcode, typed args) → typed result @ validation-level`, dispatched positionally, fired exactly once per call, every proven-level handler hold-out-gated.
- **Result Injection Policy** — how a verified result re-enters the pass: **Mode A** (token-boundary trampoline, inject as tokens — proven) or **Mode B** (hidden-state vector edit at layer k+1 — research-grade, unbuilt).
- **Proof Ledger** — the per-commit audit record: base/tokenizer/adapter hashes, runtime policy (temp=0, retries=0), the validation level cleared, placebo/causal attribution, and a rollback path — what makes inference "proof-carrying" and replayable.
- **Trap Budget** — the kernel-owned hard counters in `ContinuationFrame.budgets` that bound scheduler-mediated work: max-traps/token, no-recursion, same-op-once, failed-op-disables, depth/deadline. Hard counters, not sampling pressure (which is knife-edge); applied adaptively on escalation, not charged to the fast path.
