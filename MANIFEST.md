# MANIFEST — every file in the bundle

Grouped by plane and concern. Paths are relative to the bundle root. The import root is `scripts/`.
"Tier" indicates what is needed to exercise the file: **A** = CPU only; **B** = live North model.

## Top level

| File | Purpose |
|------|---------|
| `README.md` | Orientation, the 60-second story, Tier-A quickstart, honest scope. |
| `BUILD-FROM-SCRATCH.md` | The detailed, numbered build+run guide (human or LLM agent). |
| `MANIFEST.md` | This index. |
| `requirements.txt` | Tier-A / Tier-B dependency split. |
| `verify.sh` | One-shot CPU reproduction — runs the 28 suites, prints PASS/FAIL. |
| `docs/neural-microkernel-paper.md` / `.pdf` | The arXiv-style writeup: architecture, evidence, limitations. |
| `docs/neural-microkernel-design.md` | Build-level design + the staged M0–M5 plan + glossary. |

---

## Scheduler plane — the kernel process table & admission (Tier A)

| Module (`scripts/kernel/`) | Tier | Purpose |
|------|:--:|---------|
| `task_graph.py` | A | **Proof-Carrying Task Graph** — the kernel-owned process table; the three-gate fail-closed `admit`; Scheduler/Task state machine; ArtifactStore; planners. |
| `orchestrator.py` | A | **Job Kernel** — intake → classify → plan → run on the task graph → rollup, with plan memory. |
| `commit_trap.py` | A | **M1 commit trap** — the only output path: validate a candidate against its Contract along syntax<schema<property<oracle<proven; hash-replayable `ProofLedger`; `TrapBudget` (anti-runaway). |
| `lane_oracles.py` | A | Per-lane verification oracles compiled into runtime validator predicates. |
| `frame.py` | A | `ContinuationFrame` — the suspended neural process (capabilities, budget, KV handle). |
| `scheduler.py` | A/B | Budgeted FIFO scheduler interleaving N decode lanes, each owning its own KV cache. |
| `k_policy.py` | A | **M5 scheduler-owned dynamic-k** — `KPolicy` disposes the continuation branch-count `k` from route scores (calibration trap, risk safety-floor, cost brake, anti-runaway clamp). Tuning: README §5. |
| `k_cost.py` | A | **M5 cost + budget** — sysinfo→`cost∈[0,1]` (binding-constraint) feeding `KPolicy`; `charge_budget` caps `k`. |
| `moe_entropy.py` | B | **M5 OBSERVE half** — inert (`==0`) tap of the MoE gate-softmax entropy per decode step (calibrated out *inverted* → kept unwired; see README §5.1). |
| `fact_loop.py` | A | **Fact-loop capability multiplier** — `governed_fact_loop`: compute a verified fact with a hold-out-proven det primitive, deliver it AS TOKENS (re-prefill), GOVERNED by `KPolicy` (thin by default; spend only when the base fails validation and cost/budget allow). The realized Mode-A channel. Inertness: empty facts → prompt byte-unchanged. Tuning: README §6. |
| `fact_dispatch.py` | A | **Fact-loop live dispatch** — `FactRouter` + `default_router` (letter_count/gcd/is_prime det primitives) + `dispatch` (route → governed loop or pass-through, capability-gated fail-closed) + `register_fact_loop_executor` (wires `op:fact_loop` as a Scheduler executor, gated by `admit()`). Tuning: README §6. |
| `telemetry.py` | A | Structured, persisted per-run trace of every kernel decision. |
| `analyze_telemetry.py` | A | Reconstruct + self-check a run's flow from telemetry. |

## Deterministic plane — proof, sandbox, self-extension (Tier A)

| Module | Tier | Purpose |
|------|:--:|---------|
| `holdout_gate.py` | A | The reusable **hold-out admission gate**: n novel instances vs an independent oracle, novelty floor, placebo control, hostile-candidate hardening. |
| `venv_tool.py` | A | `venv_exec` syscall tool — runs model-written candidate code under a macOS `sandbox-exec` profile (no net, no writes, killpg, rlimits); child returns raw outputs only. |
| `self_extend.py` | A | The **self-extending pipeline**: synthesize → `gate_candidate` → register a proven primitive. |
| `escalate_solve.py` | A/B | The **escalation ladder**: greedy → diverse → repair → research → abstain; every rung judged by the gate. |
| `solve_failed_challenges.py` | A/B | Trusted oracles/generators for the previously-failed tasks (w47, c156) — the model never sees these. |
| `learned_primitives.json` | — | Artifact: primitives the kernel grew and hold-out-proved (`to_snake_case`, `resp_parser`). |

## Syscall-tier tools (Tier A)

| Module | Tier | Purpose |
|------|:--:|---------|
| `tools.py` | A | The syscall-tier **tool registry** (capability + validator + max-level per tool). |
| `tools_builtin.py` | A | Native handlers: `datetime`, `sysinfo` (graceful psutil fallback, secret redaction). |
| `trace_tool.py` | A | `trace` tool — read live/persisted scheduler telemetry + invariant self-check. |
| `web_tools.py` | A/B | Bridge to the harness web tools (`web_search` / `fetch_page`) for the research tier. |
| `agent_delegate.py` | A | Trustless agent delegation — fire an agent, reconcile its result against the trace (verdict computed kernel-side). |
| `memory_store.py` | A | The persistent, **hard-bounded** (default 100 MB) SQLite-WAL memory; LRU eviction of non-pinned entries; pinned governance object always consulted first. |

## Neural plane — the forward-pass kernel (Tier B)

| Module | Tier | Purpose |
|------|:--:|---------|
| `trap_plane.py` | B | No-op sidecar **trap plane** wired into the real North forward pass at 7 layer sites. |
| `trapped_adapter.py` | B | NorthAdapter routed through the trap plane, threading the real KV cache. |
| `commit_kernel.py` | B | M1 commit-trap as the only output path of the trapped decode. |
| `entry_trap.py` | B | **M3 entry trap** — the learned syscall head classifies on ingress and writes capabilities into the frame (reads the frozen L0 residual; gen-identical). |
| `kv_park.py` | B | Decode-time **KV park/resume** — snapshot `(.state, .meta_state)` for the mixed KVCache + RotatingKVCache stack. |
| `code_fixups.py` | B | Two model-tail improvements for the kernel's code path. |

## Invariant batteries — bit-exact proofs on the live model (Tier B)

| Module | Proves |
|------|--------|
| `inv_trap_plane.py` | no-op trap plane == native North at sites {0,8,16,24,32,40,48} (==0). |
| `inv_kv_park.py` | single-lane park/resume is bit-identical to straight-through. |
| `inv_switch_isolation.py` | interleaved multi-lane decode == solo; each lane == gen_fast. |
| `inv_speculate_rollback.py` | speculate → rollback → re-decode == straight-through. |
| `inv_det_interrupt.py` | deterministic in-decode interrupt: identity when off, exact mask when armed. |
| `inv_rollback_ringcross.py` | rollback across the RotatingKVCache >4096 ring wrap (the hard restore case). |
| `inv_moe_entropy.py` | **M5** — the MoE gate-entropy tap is `==0`-inert vs `gen_fast` (token-for-token) + the hook is live. |
| `logs/inv_*.json` | Artifact: the residual ≡ 0 proof outputs from the original run. |

## Runners, investigation & evaluation

| Module | Tier | Purpose |
|------|:--:|---------|
| `run_commit_kernel.py` | B | M1 verification — commit-trap-as-output-path over ladder coding tasks. |
| `run_entry_trap.py` | B | M3 verification — entry trap classifies real prompts + writes capabilities. |
| `run_coding_kernel.py` | B | Run the coding suite **through** the Job Kernel (`--model north`). |
| `run_novel500.py` | B | Run the novel-500 stress set through the kernel; `novel500/domains/*.json` included. |
| `run_cyber_ooda.py` / `run_cyber_mellum.py` | B | OODA / single-shot investigation flows on the Job Kernel. |
| `m5_calibrate_entropy.py` / `m5_reanalyze_calib.py` | B | **M5** — the calibration gate: does gate-entropy predict correctness? (found *inverted* + single-suite → entropy kept unwired; README §5.1). |
| `m5_dynamic_k_north.py` | B | **M5** — live demo: the disposed Tier-2 `k` adapts idle→5 / busy→1 / budget→2; thin on first-attempt pass. |
| `run_fact_loop_north.py` | B | **Fact-loop** — the four-arm A/B (base / always_ground / governed_idle / governed_busy) on the count-then-add mix: the verified-fact re-prefill lifts accuracy (~9.5×), governed stays thin (`avg_gens∈(1,2)`), cost=1.0 vetoes the spend. `--fake` = CPU smoke. README §6. |
| `run_fact_dispatch_north.py` | B | **Fact-loop** — live `dispatch()` routing + governance + inertness on North **and** the scheduler-executor path (`op:fact_loop` through a real `Scheduler`, hash-verified `ProofLedger`, fail-closed cap reject). README §6. |
| `investigation.py` | A/B | Domain-agnostic structured-investigation protocol (the shared reasoning loop). |
| `raw_baseline.py` | B | Causal control — raw single-shot baseline vs the kernel. |
| `forensics_cyber.py`, `inv_smoke.py`, `inv_code_smoke.py`, `probe_gptoss_kernel.py`, `fixture_tools.py`, `demo_real_executors.py`, `executors.py` | A/B | Investigation forensics, smokes, grounding fixtures, and the executor wiring. |

## Neural-plane support modules (`scripts/*.py`, Tier B)

| Module | Purpose |
|------|---------|
| `north_adapter.py` | **The MLX model adapter** — the neural ABI (`gen_fast`/`gen_reasoning`/`gen_sample`). Edit line 10 to point at your weights. |
| `north_gateway_invariants.py` | Milestone-1 invariants realized on real North (cohere2_moe) — the bit-identity battery substrate. |
| `north_syscall_head.py` | Trains the in-weights **syscall head** (`--layer 0` for the entry trap; `--layer 24` gateway). |
| `north_det_block.py` | Deterministic constraint op on the LLM stack (the in-decode interrupt mask). |
| `gptoss_adapter.py` | gpt-oss-20b adapter — the kernel is model-agnostic behind the same ABI. |
| `det_primitives.py`, `det_primitives_cyber.py` | The deterministic instruction set (LLM-as-JIT primitives) + cyber/DFIR primitives. |
| `output_validators.py` | Output-contract validators across output types (the bailout safety gate). |
| `shape_appliance.py`, `shape_detector.py`, `shape_recoveries.py`, `challenge_recoveries.py`, `challenge_union_recoveries.py` | Shape-routing appliance: route a task to a hold-out-proven primitive by structure (the deterministic-plane recovery library the kernel draws on). |

## Portability adjustments from the source repo (the only edits)

This bundle is a faithful copy of the source tree with exactly two adjustments so it is
self-contained outside the original repo:

1. **`scripts/north_adapter.py` line 10** — the `NORTH = "…"` weights path is the original
   author's; edit it to your download location (BUILD-FROM-SCRATCH §3.3). *(left as-is, documented)*
2. **`scripts/kernel/test_commit_kernel.py` & `run_commit_kernel.py`** — the ladder task fixture is
   now resolved **bundle-first** (`scripts/kernel/fixtures/ladder_all_cases.json`, shipped here) with
   a fallback to the repo's original `reports/north-m0/` path, so both work standalone and in-tree.
   The fixture itself (`fixtures/ladder_all_cases.json`, 100 ladder coding tasks) is included.

Everything else is byte-identical to the source modules.

## Test suites (Tier A — the executable specification)

`scripts/kernel/test_*.py` — 28 suites, run by `verify.sh`. Each pins down a contract:
`test_task_graph` (admission gates), `test_commit_trap` / `test_commit_kernel` (commit path),
`test_holdout_gate` (hostile-candidate hardening), `test_self_extend` (extension loop),
`test_escalate_solve` (ladder order + abstention), `test_venv_tool` (sandbox verdict-in-parent),
`test_memory_store` (hard cap + eviction + governance), `test_tools` / `test_trace_tool` /
`test_web_tools` / `test_agent_delegate` (syscall tools), `test_lane_oracles`, `test_orchestrator`,
`test_investigation`, `test_scheduler`, `test_kv_park`, `test_entry_trap`, `test_det_interrupt`,
the **M5 scheduler-owned-dynamic-k** suites `test_k_policy` (the policy: calibration trap, risk floor,
cost brake, anti-runaway clamp), `test_k_cost` (sysinfo→cost + budget accounting), `test_k_policy_ladder`
(the additive escalation-ladder integration) — tuning & usage README §5 — and the **fact-loop
capability-multiplier** suites `test_fact_loop` (the inertness invariant, thin-by-default, the lift path,
cost/budget veto, empty-provider honesty) and `test_fact_dispatch` (routing, capability gate, and the real
`Scheduler` admitting + running `op:fact_loop` fail-closed). Tuning & usage: README §6.
