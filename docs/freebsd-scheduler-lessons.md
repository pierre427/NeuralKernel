# Lessons from the FreeBSD Kernel for an LLM Scheduler — and the Privilege Boundary

*Pierre Lamy · research note, 2026-06-21*

The concurrent scheduler work framed the architecture as **"LLM = CPU, kernel = scheduler"**
(`experiments/neural-scheduler/SYNTHESIS.md`). FreeBSD is the best-documented general-purpose
kernel with a *measured*, non-trivial scheduler (ULE), so it is the right place to mine
lessons — and, critically, to answer the architecture question: **can the scheduling be wired
*entirely inside the model*, or do we depend on an external harness?** The OS process/kernel
split gives a clean, principled answer.

## 1. FreeBSD ULE → our system, concept by concept

| FreeBSD (ULE / kernel) | Our system | The lesson |
|---|---|---|
| **Interactivity scoring** — a thread is "interactive" iff its *measured* voluntary-sleep/run ratio is below a threshold; ULE recomputes it at wakeup points and classifies interactive vs batch | Difficulty routing. But our fixed *budget* **guesses** difficulty up front; ULE never does — it **measures behavior**. The **validation-gated halt** is the analog: don't guess a budget, *measure* whether a valid answer is producible | **Schedule on measured behavior, not a static guess.** This is exactly why the fixed-budget RL collapsed (15/15 tasks flat — the budget doesn't predict outcome) and why "passes validation → goto end" works. |
| **Three run queues per CPU** — real-time/interactive (pri 0–171), batch *calendar* queue (172–223), idle; interactive threads kept in the current queue for responsiveness | The router's **tiers** — off-path (interactive/fast) → bounded-surgery → bounded-base → bounded-feedback (batch/deep). The escalation *is* a multi-class priority ladder | **Multi-class, not uniform.** Running everything through the deep path lost (uniform-bounded 89 < routed 100). ULE keeps interactive work off the batch queue for the same reason. |
| **Preemption** — a higher-priority thread preempts the running one; the kernel checks priority on enqueue (`kern.sched.preempt_thresh`) | **Mid-stream halt** — the validator *preempts* the thinking phase the moment a valid answer is ready ("inject/obliterate the budget mid-stream") | Preemption is only viable if the **context switch is cheap**. ULE invests in cheap switch; we made park/resume cheap (trim). |
| **Context switch** — save/restore thread state (registers, stack) | **KV-cache park/resume** — the residual stream / KV cache *is* the saved context. Verified exact on North (trim restores the offset; the toy: 0 mismatches) | A suspended computation = saved context. This is the load-bearing primitive under *every* policy (adaptive depth, I/O interleave, speculative, specialist routing). |
| **Syscall / trap** — a user process **cannot** do I/O, scheduling, or memory ops itself; it traps to the kernel | **Validation / tool-call** — the model **cannot execute** the candidate code or check JSON; it *traps to the runtime* to run the validator | **This is the architecture answer (§3).** Validation is a syscall. |
| **Turnstiles + priority inheritance** — blocked threads sorted by priority; the holder inherits the waiter's priority to avoid inversion | A request **blocked on a tool/validator** shouldn't stall the queue; boost the request whose result others wait on; don't head-of-line-block the fast path behind a deep one | Avoid priority inversion in the batch/IO path; keep the interactive path responsive. |
| **Callout / softclock timeout** — a registered timer fires after N ticks | The **fixed thinking budget** is exactly this — a dumb timer. ULE prefers *measured* classification over fixed timers wherever it can | The fixed budget is the *worst* scheduler — a blind callout. Replace it with measurement (validation). It is the fallback, not the policy. |
| **Mechanism vs policy** — the kernel provides *mechanism* (run queues, switch, timers); the *policy* (ULE's equations, thresholds) is tunable, data-driven, and replaceable | The runtime provides *mechanism* (park/resume, validator-execution, dispatch); the *policy* (when to halt, which tier) can be **learned** (an in-weights gateway) or a small deterministic supervisor | **Keep the kernel thin and general; let the policy get smart.** ULE's history *is* "policy got data-driven while the mechanism stayed general." That is our roadmap. |
| **Speculative execution** (CPU microarch, and our speculative decoding) | The validation halt **is speculative**: speculate the answer is ready (force it), verify (validate), **roll back** (trim) on misprediction | Speculate + verify + cheap-rollback is the CPU's oldest trick. Park/resume is the rollback. The draft/verify of speculative decoding is the same shape. |

## 2. The four deepest lessons

1. **Measure, don't guess.** ULE's whole advance over the old BSD scheduler was replacing
   fixed priorities/time-slices with a *measured* interactivity score. Our whole advance over
   the fixed-budget halt (and the failed budget-RL) is replacing a guessed budget with a
   *measured* validation. Same lesson, 20 years apart.
2. **Mechanism vs policy is the seam to design on.** FreeBSD survives because the kernel is a
   thin, general mechanism provider and the scheduler is replaceable policy (4BSD → ULE without
   touching the syscall ABI). Our runtime should be the same: park/resume + validator-sandbox +
   dispatch are mechanism; halt/route decisions are policy that can move into the weights.
3. **Cheap context switch unlocks everything.** Preemption, interleaving, speculation, and
   priority all depend on saving/restoring a computation cheaply. We have that (trim-based
   park/resume, exact). It is the single most important primitive.
4. **Speculate + verify + rollback.** The CPU speculates and rolls back on misprediction; the
   kernel lets a process *try* and traps on fault. Our model speculates an answer (halt), the
   validator verifies, and park/resume rolls back. Design *for* misprediction — make rollback
   free — rather than trying to predict perfectly.

## 3. Can we wire it entirely inside the model? — No, and the OS tells us exactly why

The question is really about **privilege**. A user-mode process does general computation but
**structurally cannot** perform privileged operations — I/O, scheduling, memory management — on
itself; it must `trap` to the kernel. The mapping is exact:

> **The model is the CPU/process. The runtime is the kernel. Validation is a syscall.**

Sort every function by privilege:

| Function | Where it must live | Why |
|---|---|---|
| Generate tokens (general computation) | **In the model** | it *is* the CPU |
| **Propose** halt / route / "I need a tool" | **Can be in the model** — an in-weights gateway emitting control signals ("learned proposes") | this is the model *making a syscall*; the toy proved a differentiable in-weights halt signal |
| **Execute the validator** (run candidate code, parse JSON, diff vs an oracle) | **Kernel (runtime)** — a sandbox | a token-generator cannot run a Python interpreter on itself; this is privileged execution = a syscall. *"passes validation"* requires the kernel to actually run the validation |
| **KV-cache park/resume** | **Kernel** | memory management; the runtime owns the cache lifecycle |
| **Dispatch / retry / tier escalation** | **Kernel** | scheduling across generations |
| **Tool calls / external I/O** | **Kernel** | I/O is privileged by definition |

So the honest answer has two halves:

- **Entirely inside the model: no.** The halt *signal* can be in-weights, but the halt
  *criterion that makes it correct* — "the output validates" — requires **executing the
  output**, which the model cannot do to itself. Validation is a syscall; you cannot remove the
  kernel any more than a CPU can do disk I/O without one. (The model can *predict* it will
  validate — a learned confidence — but prediction ≠ the ground-truth check.)
- **A *specialized* harness: also no.** You do **not** need our bespoke harness. You need a
  **thin, general kernel** — three mechanisms: (a) **park/resume** (context switch; standard
  serving infra already has prefix-cache machinery), (b) a **validator sandbox** (the syscall
  handler; small and general — our `harness_validators` + `form_verify` + the cassette oracle
  are exactly this, ~300 lines), (c) a **dispatcher** (the scheduler; tiny). Any model docks
  into it, exactly as any program docks into an OS kernel.

**The design that falls out** is "learned proposes, kernel disposes," sharpened by FreeBSD's
mechanism/policy seam and the CPU's speculate/verify/rollback:

1. The model runs in "user mode" and **speculates** — an in-weights gateway proposes *halt now*
   (it predicts the answer will validate) and *route this way*. This policy can be fully learned.
2. The thin kernel **verifies and disposes** — runs the validator (syscall), and on success
   commits ("goto end"); on failure, **park/resume rolls back** (trim) and the model keeps
   thinking. Mispredictions are cheap by construction.
3. The kernel stays **general and replaceable** — swap models, swap validators, swap the policy,
   without touching the mechanism. ULE replaced 4BSD without breaking the ABI; the same seam
   lets a learned halt-head later replace the deterministic validator-gated halt **without
   rebuilding the runtime**.

The practical consequence: the more the model's speculation matches the kernel's verification,
the *less work the kernel does* (fewer re-validations, earlier commits) — but the kernel never
vanishes, because it owns the ground truth (execution), the memory (cache), and the scheduling.
That is not a limitation to engineer away; it is the **privilege boundary**, and respecting it
is exactly why operating systems are robust. We should build the thin kernel deliberately — and
we already have its three pieces (`north_mid_halt.py` park/resume, `harness_validators.py`
sandbox, `north_router_server.py` dispatch).

## 3a. Refinement (P. Lamy): most of the "kernel" is an ON-STACK co-processor, not a syscall

The §3 framing — "validation is a syscall to an external kernel" — is too conservative, and the
correction matters. **The model's forward pass is *already* exact, deterministic GPU
computation; only the final token *sample* is fuzzy.** So a deterministic logic block is not a
remote trap to a CPU — it is a **non-learned op the router dispatches to on the *same stack***,
a functional unit, exactly as a CPU dispatches to its FPU rather than "syscalling" to it. The
deterministic part doesn't have to leave the device *or* be fuzzy.

This is most naturally expressed as a **dispatch spectrum**, not a binary:

1. **Constrained decoding — the deepest form.** A deterministic block masks the logits *during*
   generation (a pure on-device array op), so the output is valid **by construction** — no
   post-hoc check at all. We proved this on North (`north_det_block.py`): a deterministic
   constraint block runs in North's MLX generation loop on-device and forces an exact property
   (e.g. "no imports", or a fenced code block) by masking forbidden tokens before argmax. This is
   *literally* "route to a deterministic process that executes on the GPU through a logic block,
   in a bypass step." A JSON/grammar/regex constraint is the same op with a richer mask.
2. **On-stack deterministic specialists/validators.** Bounded deterministic *algorithms* —
   regex match, JSON/XML parse, a grammar check, a type check by parsing — are co-processors the
   router fires to in-pipeline (the regex cassette, `form_verify`, the type validator). They run
   on the stack, exact, with no fuzzy sampling and no CPU round-trip required.
3. **True syscalls — the genuinely external residue.** Only *arbitrary untrusted execution*
   (running the candidate's own generated code on real inputs) and *real I/O* (files, network,
   external tools) need an actual sandbox/kernel. That residue is **much smaller** than §3
   implied — it is the execution-validators, not the form/constraint ones.

And North's architecture **already is** this dispatcher: an MoE is a learned **gate** routing
tokens to **experts**. A *deterministic expert* — a logic block in an expert slot, with the gate
biased to route the relevant tokens to it (the design note's heterogeneous side-MoE) — is the
in-graph realization of (2), no serving-loop glue at all. The gate is the router; a deterministic
expert is the co-processor; both live in the same GPU graph.

**So the sharpened answer to "entirely in the model vs. a harness":** the *intelligence* (route,
halt) lives in the weights/gate; the *deterministic mechanism* lives **on the same stack** as
non-learned ops (constraint masks, deterministic experts, in-pipeline validators); and only
*true I/O / untrusted execution* traps to an external sandbox. The privilege boundary is real but
**thin** — it is fuzzy-vs-deterministic *dispatch on one device*, with a small external residue,
not "model here, kernel over there." That is strictly better: lower latency (no round-trip),
unified memory (the KV cache and the constraint op share the device), and one scheduler over a
**heterogeneous graph of fuzzy experts + deterministic logic blocks**.

## 3b. Correction (P. Lamy): control-flow blocks ARE wireable, and MLX is the accelerator

§3a's split — "control-flow → harness-side, array-native → GPU" — was too coarse and conflated the
**eval harness** (downstream) with the **serving runtime** (the inference process). There are
**three** integration sites, all in the runtime, all proven on North:

1. **In-runtime CPU function call** — a straight-line/control-flow primitive (count_byte, c_gcd,
   gf2_xor_basis, a regex matcher, an LCT) runs *in-process during generation*: the model emits
   `[[fn(args)]]`, the runtime executes it on the CPU and **splices `→ result` back into the
   stream** (park/resume). **Proven** (`north_fncall.py`): North called `count_byte('strawberry','r')→3`
   and `count_byte('bookkeeper','e')→3` mid-generation — fixing the fuzzy miscount. This is *not*
   harness-side and *not* a GPU kernel; it is a co-processor in the inference loop. Control-flow
   does not need to parallelize to be wired — it only needs to be *callable mid-stream*.
2. **MLX/Metal GPU op or custom kernel** — array-native primitives (FFT, linalg, sort, conv,
   matmul, bit-ops) run on the GPU via `mx.fft`/`mx.linalg`/`mx.sort` and `mx.fast.metal_kernel`.
   **Proven** (`mlx_primitives.py`): FFT poly-multiply **13× over naive** on Metal, plus a custom
   Metal **popcount kernel**. No CUDA — MLX *is* the accelerator on Apple Silicon.
3. **Constrained decoding** (§3a) — a deterministic logit op shaping generation token-by-token.

**The Apple-Silicon advantage: unified memory.** The model's weights/KV cache, the CPU function
blocks, and the MLX/Metal primitives all live in **one memory**. A deterministic block — CPU or GPU —
operates on the *exact tensors the model holds, with zero host↔device copy*. CUDA pays a PCIe
round-trip for the same; we don't. So the integration we want (deterministic side-quests invoked
mid-reasoning, expert-steering) is *tighter and cheaper here than on CUDA*, and it does not require
turning control-flow into a GPU kernel — only into an in-runtime call. The right rule: **GPU-kernel
status is for *parallel speedup* (array-native); in-runtime CPU calls are for *exactness +
mid-stream availability* (control-flow); both are "wired to the LLM," both on unified memory.**

## 3c. Stay in the GPU kernel — control-flow as a Metal kernel, and the honest workload caveat

Falling out to the CPU is a GPU↔CPU **context switch**; the goal is to keep the function block
**GPU-resident** (Metal kernel, unified memory, executed by the GPU alongside the LLM, no switch).
Control-flow does *not* force a CPU fallback — a GPU thread runs loops and branches. **Proven**
(`mlx_glob_kernel.py`): a glob matcher (`*`, `?`, backtracking) runs entirely as a Metal kernel,
thousands of pairs in parallel, correct.

**Honest caveat — the GPU is not automatically faster.** 20 000 *tiny* glob matches: Metal kernel
**8 ms vs CPU 4 ms** (GPU slower), because (i) the 8 ms includes CPU-side packing of text→int arrays
(a prep cost that *vanishes when the data is already GPU-resident* — which is the whole point of
staying on the GPU), and (ii) trivial per-thread work means dispatch overhead dominates. The GPU
pays off for **high arithmetic intensity / large per-item work** (FFT poly-mult: **13×**), not for
thousands of trivial branches. So:

- **GPU-resident Metal kernels are the north star** (control-flow *and* array-native), no context
  switch — proven possible.
- **The win is workload-dependent**: array-native / high-intensity → big GPU speedup; tiny branchy
  ops → overhead-bound (still avoids the switch in a fully GPU-resident decode, but not auto-faster).
- **CPU / async-scheduler fallback** (park the task → queue a syscall → callback to the held task)
  for what the GPU truly cannot do — *not yet built*.
- **The LLM assembles GPU function blocks into solution-provider det-blocks**, spliced dynamically
  (l80-style composition, linking GPU primitives).

## 3d. The end-state: a GPU megakernel (decode + blocks + scheduler), and the boundary

The logical end of the thesis (P. Lamy) is that **nothing on the hot path touches the CPU** — the
decode loop, the deterministic function blocks, *and* the scheduler all run as **persistent GPU
kernels** on unified memory:

- **Fused decode** — the whole autoregressive loop (transformer forward + sampling + a check for
  pending function calls) runs in **one resident kernel** that loops on the GPU, instead of today's
  per-token Python/CPU orchestration. This is the "megakernel" direction in LLM inference.
- **On-GPU async scheduler** — a persistent kernel polling an **on-GPU atomic work queue**. The
  decode kernel, mid-generation, *enqueues* a function call (GPU atomic), parks that lane (KV cache =
  the saved continuation), and the scheduler kernel pulls the request, dispatches the function-block
  kernel, writes the result into unified memory, and signals the parked lane to resume — all without
  leaving the GPU. The park→queue→callback model, GPU-resident.
- **Function blocks** — Metal kernels (control-flow *and* array-native), already proven
  (glob/popcount/FFT). The LLM *assembles* them into solution-providers and steers the gate to them.

**The tractability boundary is honest and sharp:**

| layer | in MLX's Python API? | status |
|---|---|---|
| function-block Metal kernels (dispatch) | yes (`mx.fast.metal_kernel`) | ✅ proven |
| step fusion | yes (`mx.compile`) | ✅ 1.20× (`mlx_fused_step.py`) |
| **persistent kernels, on-GPU atomic work queue, single-kernel fused decode + scheduler** | **no — raw Metal / MTLCommandBuffer / C++** | **systems build (frontier)** |

So: every *building block* is proven inside MLX, but the **megakernel that assembles decode + blocks
+ scheduler into one persistent GPU kernel lives below MLX's Python surface** — it is a focused
raw-Metal systems project (cf. persistent-kernel / megakernel LLM-inference work), not a loop-tick
script. And one honest calibration: North's decode is already **GPU-bound** (~70 tok/s on a 30 GB
MoE), so the megakernel's payoff is *structural* — no context switch, an on-GPU scheduler, in-kernel
function dispatch — not a raw decode-speed multiplier. The toy scheduler
(`experiments/neural-scheduler/`) is the differentiable sandbox that already validated the
*policies* (halt, park/resume, speculative, priority); the megakernel is their GPU-resident
*systems* target.

**VERDICT (2026-06-23): a NICE-TO-HAVE, deliberately not built.** Because the payoff is *structural
only* (North is already GPU-bound, so no decode-speed multiplier) and it adds **no new capability or
correctness** — its success invariant is the *already-proven* online==offline, merely fused — the
raw-Metal/MTLCommandBuffer/C++ persistent-kernel build is **not worth its cost** at the project's
current capability-driven stage. It is correctly deferred to a future high-throughput **serving**
regime or a **systems-paper** contribution. The in-API building blocks above (`mx.compile` step
fusion 1.20×, `mx.fast.metal_kernel` function blocks) are the low-risk middle path if perf is ever
the goal; the raw-Metal megakernel is reserved, not scheduled.

## 4. What this changes about our roadmap

- The **validator sandbox is a first-class kernel service**, not an afterthought — invest in a
  fast, general one (type+probe, form, oracle), because it is the syscall every adaptive policy
  traps into.
- The **in-weights gateway** (the toy's PonderNet halt, invariants already 0.0 on North) is the
  right home for *policy* — but it **proposes**, it does not replace the validator. Train it to
  *predict validation* (cheap speculation), and let the kernel *confirm* (correctness).
- **Park/resume is the load-bearing mechanism** — keep it exact and cheap; everything (preempt,
  interleave, speculate, route-to-specialist) is a use of it.
- Stop trying to make the halt *fully* in-weights (the RL-grade trap). The OS answer is a
  *split*: smart policy in the model, thin privileged mechanism in a general kernel.

## 5. The whole architecture: the LLM as a JIT compiler with a standard library

The pieces compose into one system. At runtime the model **proposes a solution to itself,
dynamically builds and validates a deterministic block, and routes computation through it** —
and crucially it composes from a **trusted primitive library** rather than reinventing hard
primitives. This is **JIT compilation for reasoning**:

| JIT / OS | here | status (`north_compose_detblock.py`) |
|---|---|---|
| stdlib / instruction set | trusted primitives: `count_letters`, math, `match_char_class`, `form_verify`, parsers | ✓ built |
| compile (fuzzy → code) | the model *proposes a composition* of primitives (glue + control flow — its strength) | ✓ |
| verify / type guard | the harness-grounded validators gate the composition; failure → recompile | ✓ caught real bugs (`a*` on `''`, `[0-9]` on `5`) |
| execute the compiled fast path | route real inputs *through* the validated block — exact, reusable (memoized) | ✓ 9/12, exact execution |
| deopt → recompile | the repair loop on validation failure | ✓ |

**Two demonstrations on North.** (i) *The strawberry problem* — fuzzy character counting is
*unreliable* (North miscounted `bookkeeper`'s e's: said 4, there are 3), while a det-block
`count_letters` is exact; the system recognizes "this is counting" and routes to it. (ii)
*Composition resolves a capability gap* — l80 (`match_simple_regex`), which North **cannot write
from scratch (0)**, reaches **9/12** by composing the trusted `match_char_class` primitive, with
the validators correctly rejecting buggy intermediate compositions and triggering repair.

**The honest lessons.** (a) **Composition resolves capability gaps** — the model is good at glue,
bad at primitives, so hand it the primitives. (b) **The validators are the load-bearing trust
mechanism** — they are what make a *self-built* block trustworthy, which is why grounding them in
the test profile (§ report 5b-iv/§6) mattered. (c) **Primitive granularity is the design knob** —
one primitive took l80 from 0→9/12; the right decomposition (add `match_quantified`) turns the
rest into glue. (d) It **closes the skill-pack loop**: the regex cassette was a *pre-built*
primitive; here the model builds its *own* from primitives and the validators certify it —
**self-service skill packs**, synthesized and trusted at runtime.

This is the endpoint the whole project was walking toward: a **thin heterogeneous runtime** —
fuzzy experts (the model), deterministic logic blocks (on-stack primitives and composed det-blocks),
a validator sandbox (the trust gate), park/resume (cheap context switch), and a router/scheduler —
where the *intelligence* (propose, route, compose) lives in the model and the *mechanism* (verify,
execute, switch) is a thin, general kernel. Learned proposes; the kernel disposes; and the model
can extend its own deterministic instruction set, at runtime, under the validator's guard.

---

**Sources:** [Thread Scheduling in FreeBSD 5.2 (ACM Queue)](https://queue.acm.org/detail.cfm?id=1035622) ·
[An Overview of Scheduling in the FreeBSD Kernel — McKusick, BSDcan 2020](https://papers.freebsd.org/2020/BSDcan/mckusick-Scheduling_in_the_FreeBSD_Kernel.files/mckusick-Scheduling_in_the_FreeBSD_Kernel.pdf) ·
[Process Management in the FreeBSD OS — §4.3 Context Switching / §4.4 Thread Scheduling (InformIT)](https://www.informit.com/articles/article.aspx?p=2249436&seqNum=4) ·
[SMPng Design Document (FreeBSD Arch Handbook)](https://docs.freebsd.org/en/books/arch-handbook/smp/) ·
[ULE: a modern scheduler for FreeBSD (Roberson)](https://www.researchgate.net/publication/41035925_ULE_a_modern_scheduler_for_FreeBSD)
