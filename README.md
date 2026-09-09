# GCL & OpenContinualEnv — Grounded Continual Learning for Code LLMs

**A code LLM that learns new task families forever, from execution feedback alone — zero gold labels in training or in the safety gate — without forgetting what it knew.**

One consumer GPU. Every number traces to a logged, hash-verified artifact. Every experiment was pre-registered before it ran — including the failures.

*In a hurry? 30 seconds: section 1. Two minutes: sections 2–3. The story of how it was cracked: bottom of the page.*

---

## 1. The result (read this first)

The hardest number in continual learning: **hold the first family you learned at the never-trained model's level, while still updating.** Across 3 seeds, identical stream (hash `554ce43f182b`), every pre-registered check passed:

| Seed | Winning config | arith holdout | ACC | frontier | forgetting | updates accepted |
|---|---|---|---|---|---|---|
| 42 | θ=0.5, 2 witnesses | **0.600** | 0.675 | +0.650 | 0.025 | 8 |
| 43 | θ=⅔, 2 witnesses | **0.600** | **0.738** (study best) | +0.713 | 0.025 | 10 |
| 44 | θ=⅔, 2 witnesses | **0.600** | 0.675 | +0.650 | 0.025 | 7 |

What those columns mean:

- **arith holdout 0.600 = the frozen model's level.** The frozen model (never updated) scores 0.600 on the arithmetic family's *unseen* holdout tasks. Every earlier learning row in this study collapsed it to 0.200–0.400. The v9 rows match frozen **at every seed** — while accepting 7–14 LoRA updates.
- **The gate is what does it.** Same learner, gate removed (`sccl` row): 0.400 / 0.400 / 0.200. Every treatment row ≥ the no-gate baseline at every seed, never below.
- **3-seed means:** ACC **0.679±0.056**, frontier **+0.638±0.082**, forgetting **0.042±0.029** — vs no-gate 0.596±0.083 / +0.504±0.106 / 0.092±0.029.
- Verified gold-free by construction and by audit (section 9). A prior-art sweep (OpenAlex/arXiv, Aug 2026, recorded in `RESEARCH_NOTES_v5.md`) found **no existing work combining gold-free training targets with a gold-free forgetting gate in continual learning for code LLMs**.

Setup: Qwen2.5-Coder-3B-Instruct · 4 families (arithmetic, math word problems, string ops, spec-drift variants) × 8 train / 4 holdout · RTX 4060 Ti 16GB · seeds 42/43/44.

---

## 2. The problem, in three sentences

Deployed models meet new task families one after another, and fine-tuning on family N quietly destroys family 1 — catastrophic forgetting. The classic fixes (replay, regularization, gates) all assume **labeled data**, which is exactly what a deployed agent doesn't have. But a code agent does have something better than labels: **an interpreter** — it can run code and observe what happens.

So the whole framework asks one question: **is execution feedback alone enough to both learn and protect learning?**

---

## 3. The system, in one picture

```
        task spec arrives  (prompt + entry point + domain — no gold anywhere)
                         │
                         ▼
        ┌───────────────────────────────┐
        │ 1. SELF-CERTIFY               │  write own tests for the spec →
        │    gcl/selfcert.py            │  solve them → discriminative
        │                               │  cross-bag consensus vote
        └──────────────┬────────────────┘
                       │  certified (spec, solution, self-tests)
                       ▼
        ┌───────────────────────────────┐        keep
        │ 2. LoRA UPDATE (+ anchor      │ ─────────────▶  keep new adapter
        │    pulled toward frozen init) │
        └──────────────┬────────────────┘
                       │  VETO: any stored skill now fails its own
                       │        stored self-tests → roll weights back
                       ▼
        ┌───────────────────────────────┐
        │ 3. GENERALIZATION WITNESSES   │  extra gate probes manufactured
        │    (:g lane, v8–v9)           │  from UNTRAINED future stream
        │                               │  tasks — retired before their
        │                               │  source ever trains
        └───────────────────────────────┘
```

**Three mechanisms, one guarantee:**

1. **Self-certification** — the model writes its own test bag from the spec, generates candidate solutions, and only solutions passing a *cross-bag discriminative consensus* become training targets. The API is structurally gold-free: it accepts spec/entry/domain only. Unverified self-training collapses (ACC 0.237); certification restores it (0.637).
2. **Self-replay veto (RRV)** — the safety gate. Before an update is kept, previously certified skills are regenerated under the new adapter and must still pass **their own stored self-tests**. Any failure → rollback. It's regression testing, but for weights, and the test suite is self-made.
3. **Generalization witnesses + anchor** — the v9 combination. Probes drawn from *untrained future tasks* (spec-only, so they can't leak), soft pass-rate dose θ, plus a small quadratic anchor pulling trainable params toward the frozen init. This is what finally held family 1 at 0.600 *while learning*.

---

## 4. The ladder: nine iterations, most of them failures

Every row pre-registered, run, verdict recorded — kept in `RESEARCH_NOTES_v5.md` with full numbers. This table is the honest history.

| Ver | Idea | Verdict |
|---|---|---|
| v1 | self-certify + self-replay veto, gold-free loop | Learning restored without gold (ACC 0.637, 72% certified) — but family 1 holdout erodes |
| v2 | + certified rehearsal, neighborhood probes, math-RRV | Lowest gold-free forgetting (0.075) at a plasticity price |
| v3 | admission-time neighborhood certification | **Negative, all 3 seeds** — damage lives *inside* accepted updates, not in which updates get in |
| v4 | in-update anchor toward LoRA init | **RETRACTED** — anchor silently never engaged (state-dict key mismatch ⇒ penalty ≡ 0; caught because two rows were *bit-identical*). Fixed with engagement tests that fail on the old code |
| v5 | 2×2: anchor × family-stratified gate coverage | Anchor real and best-in-class for stability; arith still 0.400 — gate checks *instances*, capability erodes |
| v5b | capability probes (held-out variants per skill) | Best aggregate learner yet (ACC 0.694, math 1.000); arith stuck at 0.375 under both gate regimes — single witness too thin |
| v6 | probe ensembles K=3 + anchor | **Witness count ≠ witness sensitivity** — 100% probe blindness; composition identical to anchor alone |
| v7 | pass-rate margin vetoes (θ over n=3 draws) | **Thresholds over blind witnesses are inert** — every erosion passed at rate 1.0; strictness only destroyed plasticity |
| v8 | generalization witnesses from *untrained* future tasks | Dominates at every seed; breakthrough 2-of-3 — θ=1.0 stalls learning (4 updates) |
| **v9** | **dose scaling: 2 witnesses/family, θ∈{0.5, ⅔}, anchor** | **CONFIRMED, all 3 seeds — arith 0.600 (frozen level) with 7–14 updates** |

---

## 5. What gold labels would have bought (the honest baseline)

The same harness runs gold-using reference rows — this is the price tag the whole study is measured against:

| Row | Setup | ACC | forgetting |
|---|---|---|---|
| `vsr_nogold` | self-generated targets, **gold tests** verify | **0.758** | 0.004 |
| `sccl` (v9 winner) | self-generated targets, **self-made tests** verify | 0.679±0.056 | 0.042±0.029 |
| `vsr` (full gold) | gold reference targets + gold gate | **0.475** | 0.175 |

Two findings worth quoting: gold tests are worth roughly +0.08 ACC over the gold-free winner (real, bounded) — **but full gold supervision lands *below* the never-trained model (0.475 < 0.613)**. Gold helps when it *verifies the model's own outputs*; it hurts when it *replaces* them. Also: a new domain learned end-to-end with zero gold — the base model scores 0.0 on synthetic math, and self-certified arithmetic bootstraps it to a perfect holdout (1.0 in 4 of 5 runs).

---

## 6. What this is — and is not

| Component | Toy / naive | This repo |
|---|---|---|
| LoRA updates | mutated matrix, fake shift | real `peft` + `torch` Adam steps, loss drop logged |
| CL metrics (BWT/FWT) | fabricated 0.000 | measured from execution runs on MBPP/HumanEval splits |
| Safety gate | token penalty in reward | snapshot → update → veto eval → keep/rollback |
| Dataset integrity | train on full benchmark | content-hash canary splits; runner *raises* on overlap |
| Reproducibility | hand-written metrics | one command regenerates every number; seed-42 bit-identity enforced across versions |
| Gold-freeness | "trust me" | structural API audit + adversarial poisoned-gold tests + 3-layer run-data audit |

---

## 7. Run it yourself

```bash
git clone https://github.com/isHeSatoshi/RL-Environment-for-Continual-Learning-at-Scale.git
cd RL-Environment-for-Continual-Learning-at-Scale
uv venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e .

python gcl_smoke.py                     # 30-second sanity check
python -m pytest tests/test_sccl.py -v  # 111-test gold-free proof suite
uv run pytest tests/e2e -v              # 106 E2E tests, 4 tiers
```

The ladders (each config = one full multi-learner experiment):

```bash
python -m gcl.runner --config configs/sccl_main.json   # v1: the gold-free loop
bash scripts/v9_launcher.sh                             # v9: the confirmed result
# seeds 42/43/44 + per-seed fail-closed checks + verdict file, one command

# v2–v8 in between, same pattern:
#   configs/sccl_v2.json … sccl_v8.json + scripts/v{N}_check.py + scripts/v{N}_telemetry.py
python scripts/run_seeds.py --config configs/sccl_v9.json --seeds 42,43,44
python app.py                                           # Streamlit dashboard
```

Fail-closed launchers (`scripts/v*_launcher.sh`) refuse to start a ladder unless its wiring smoke, engagement audits, and determinism references all pass — a wrong-but-silent mechanism never gets measured.

---

## 8. Repository layout

```
open_continual_env/      lifelong MDP (Gymnasium/OpenEnv), rewards, AST-safe
                         subprocess + Docker sandboxes, trajectory store,
                         baselines (replay / LoRA / MoA / JIT-RL)
gcl/                     the learning engine
├── selfcert.py          SCCL: self-spec tests, consensus, self-replay veto
├── engine.py            PEFT LoRA training, EWC, replay, rollback
├── vault.py             skill vault (certified skills + their self-tests)
├── curriculum.py        MBPP/HumanEval loaders + anti-contamination canaries
├── experiment.py        v8/v9 witness lanes, retirement + refresh logic
└── runner.py            CLI: python -m gcl.runner --config …
configs/                 every ladder, versioned (sccl_main … sccl_v9)
scripts/                 launchers, fail-closed checkers, telemetry, seed aggregation
paper/                   LaTeX + results tables     tests/  111 proof + 106 E2E
```

---

## 9. Trust anchors

1. **Real gradients** — training loss 0.178 → 0.001 in smoke; logit shift measured, not asserted.
2. **Anti-contamination** — train/holdout split by task ID **and** SHA-256 fingerprint; the runner raises on any overlap.
3. **Execution-grounded** — every code action runs in an isolated sandbox (AST safety check, timeouts, memory limits); rewards come from real stdout/stderr.
4. **Gold-free by proof, not promise** — (1) every SCCL vault method is Task-object-free by AST audit, (2) Task-gold readers are confined to legacy VSR baselines, (3) run data contains zero gold-gate decisions or gold targets. Adversarial poisoned-gold environments test the enforcement.
5. **Determinism** — C1 checkers enforce bit-identity of prefix rows across versions; a mid-run PC shutdown + rerun reproduced a seed bit-for-bit.

Docs: [ARCHITECTURE.md](docs/ARCHITECTURE.md) · [VERIFICATION.md](docs/VERIFICATION.md) · [TEST_READY.md](TEST_READY.md) · [KAGGLE_DEPLOY.md](KAGGLE_DEPLOY.md) · full experiment history: `RESEARCH_NOTES_v5.md`

---

## In plain words — VSR, and what the hard problem actually was

### VSR, in plain words

Before SCCL there was **VSR — Verification, Self-Reflection, Rehearsal**: the model drafts its own candidate solutions, a search loop refines and rehearses them, and **gold unit tests** decide what counts as correct. It's the halfway house between this project and everyone else's — self-*generated* targets, gold-*verified* truth. It is the strongest row in the entire study (ACC 0.758, forgetting 0.004), which makes it the honest price tag on labels: gold tests buy real accuracy over the best gold-free row. And its full-gold sibling (`vsr`, gold targets *and* gold gate) lands *below* the never-trained model — gold helps only while it verifies the model's own work. SCCL's bet was that for *forgetting protection* specifically, you could replace the gold verifier with the model's own executable tests — and still hold the line.

### The challenge, in plain words

Every method that learns, forgets — and the failures are **invisible to the learner**. The gate was passing everything: every certified skill still passed its stored self-tests, while the model quietly lost the ability to solve *new* tasks in that first family. Reproduction was intact; generalization was gone. Telemetry made it undeniable — 11 of 14 accepted updates damaged the arithmetic capability while every gate check came back green.

Nine pre-registered iterations, and most of them failed — the failures were the map. The admission filter made forgetting *worse* on all three seeds (v3). The anchor's first flight never even ran — a key-name mismatch made the penalty exactly zero; caught only because two rows came out bit-identical, retracted and fixed with tests that fail on the old code (v4). Wider witnesses, stricter thresholds, ensembles — each engaged exactly as designed and each provably blind to the actual damage direction (v5–v7). You cannot threshold your way out of a coverage gap.

The crack was a change of question: stop asking *"can it still redo old work?"* and start asking *"can it still do new work?"* — v8 built gate witnesses from **untrained future stream tasks**, spec-only, retired the moment their source enters training. The strict dose stalled learning; v9 turned the dose down (θ=⅔, two witnesses per family). That's the result at the top of this file: **0.600 — the never-trained model's level — at every seed, with learning still on.**

---

## Citation

```bibtex
@software{gcl_opencontinualenv_2026,
  title = {GCL \& OpenContinualEnv: Gold-Free Continual Learning for Code LLMs},
  author = {isHeSatoshi},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/isHeSatoshi/RL-Environment-for-Continual-Learning-at-Scale}}
}
```

## License

MIT — see [LICENSE](LICENSE).
