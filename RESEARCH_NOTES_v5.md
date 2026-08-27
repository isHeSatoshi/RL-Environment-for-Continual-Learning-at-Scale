# SCCL v5 — Closing the gate's coverage gap: family-stratified self-replay veto

Status: HYPOTHESIS PRE-REGISTERED; implementation staged (learner-gated, off by
default). Runs AFTER the v4 ladder verdict. Gold-freeness constraint unchanged:
nothing below may consume gold labels in the learning loop or accept/reject
decision; gold remains measurement-only.

## The question v3/v4 left open

Across every SCCL variant and every seed, the arith holdout collapses from the
frozen baseline 0.60 to ~0.375–0.40 while certified stream accuracy rises:

| v3 seed-42 learner | arith-hold | math | string | drift | ACC   | upd/rbk | vetoes |
|--------------------|-----------|------|--------|-------|-------|---------|--------|
| frozen             | 0.600     | 0.250| 0.800  | 0.800 | 0.613 | 0/0     | –      |
| sccl               | 0.400     | 0.500| 0.800  | 0.600 | 0.575 | 17/3    | 3      |
| sccl_n             | 0.375     | 0.750| 0.600  | 0.800 | 0.631 | 9/0     | 0      |
| sccl_promote       | 0.400     | 0.750| 0.600  | 0.600 | 0.588 | 12/6    | 6      |
| vsr_nogold         | 0.400     | 0.750| 0.600  | 0.800 | 0.637 | 12/4    | –      |

Yet the vault DOES certify arith: seed-42 sccl's final vault holds 6 arith /
4 math / 5 string / 5 drift skills, and all 17 accepted updates passed the RRV
gate with zero vetoes on sccl. How can 6 certified arith skills coexist with a
collapsed arith holdout?

## Post-mortem (seed-42 artifacts) — mechanism pinned

1. **The RRV veto has a recency window.** `selfreplay_veto` checks
   `skills[-check_skills:]` (vault.py:509) — the LAST 3 certified skills, not a
   sample across the vault. The stream order is arith → math → string → drift,
   so by the time string/drift updates run, every arith skill has left the
   window. The gate cannot see arith regressions during the phases where arith
   is damaged.
2. **Every recorded veto confirms the blind spot.** All `broke`/`broke_probes`
   entries across sccl (3 vetoes) and sccl_promote (6 vetoes) name only
   active/just-prior-family skills: drift broke drift_4/drift_5; string/drift
   broke mbpp_259/mbpp_308 and probe mbpp_587:p. ZERO arith task_ids ever
   appear in any broke list — not because arith never regressed, but because
   arith was never checked.
3. **More vetoes did not help arith.** sccl_promote fires 6 vetoes (2× sccl)
   via neighborhood-probe checking and still lands at arith 0.400. The probes
   catch same-family generalization loss; they do nothing for abandoned older
   families.
4. **Gold telemetry quantifies the blind spot (measurement-only, post-hoc).**
   The gate's gold_telemetry probe (holdout[:4], never in-loop) shows that of
   sccl's 17 accepted updates, a gold holdout gate would have REJECTED 15
   (probe score 0.600 -> 0.200–0.400 on each) and accepted only 2. So the
   damage WAS per-update detectable by a gate with global coverage — but the
   same gold gate is over-eager: rejecting 88% of updates would have halted
   nearly all learning (stream ACC would collapse). Hence neither extreme
   works: the gold-free RRV gate is locally right but globally blind; the gold
   gate is globally aware but plasticity-crushing. The fix must be STRUCTURAL:
   keep global capability intact without per-update global verification. That
   is exactly v4 (anchor: reduce damage per update) and v5 (stratified veto:
   keep every certified family under cheap self-test coverage).

So there are two candidate sub-mechanisms for the collapse:

- **(C) Coverage gap:** arith skills WOULD fail (or at least register damage)
  if checked during string/drift phases, but the recency window never asks.
- **(G) Generalization gap:** certified arith instances keep passing (memorized
  solutions survive) while the underlying general arith circuit erodes — the
  gate checks instances, not capability.

v4's in-update base anchor attacks the damage itself (regularization); v5
attacks the gate's field of view (verification). They compose.

## v5 hypothesis (pre-registered)

**H1 (coverage):** Family-stratified veto sampling — checking the most recent
skill(s) of EVERY certified family instead of the global last-3 — keeps old
families under gate protection and lifts arith-holdout above sccl without a
prohibitive plasticity cost.

Mechanism design (all gold-free):
- `selfreplay_veto(..., stratified=False)`: when stratified, build the check
  pool by grouping vault skills by `family` and taking the newest
  `ceil(check_skills / n_families)` per family (deterministic; no RNG so runs
  stay reproducible). Probe pool unchanged.
- Learner-gated via `sccl_stratified_learners` config list (default empty →
  bit-identical behavior for all existing rows; required for v4 determinism
  cross-checks).
- New learner `sccl_strat` = SCCLLearner subclass, name "sccl_strat", added to
  LEARNERS registry + consistency test (lesson of the v4 registry bug).
- check_skills stays 3 (identical to the sccl control): with 4 certified
  families, ceil(3/4) = 1 newest skill per family = 4 checks, so the ONLY
  variable vs the control is the pool policy (stratified vs recency).
- The veto stays conservative: a skill is retained iff ANY of n_samples=2
  regenerations passes all its self-tests, so stratification widens coverage
  without tightening per-skill tolerance.

## Decision rule (pre-registered, seed-42 ladder)

> SUPERSEDED by the 2x2 factorial design at the bottom of this file (added after
> the v4 verdict). Kept here as the original pre-registration for the record.

Ladder: frozen / sccl (control, must bit-reproduce v3 seed-42: ACC 0.575,
arith 0.400) / sccl_strat / vsr_nogold. Same seeded protocol (sccl index 1 →
seed 43; sccl_strat index 2 → seed 44 — note sccl_strat draws a DIFFERENT seed
than sccl; only sccl-vs-v3 bit-reproduction is the determinism control).

- **SUCCESS** if `sccl_strat.arith_hold > sccl.arith_hold` AND
  `sccl_strat.frontier >= sccl.frontier - 0.02`.
- If success: multi-seed (stream seeds 43, 44; learner indices fixed by
  position) before any headline claim — same protocol as v3.
- **Branch A — coverage confirmed but partial:** arith lifts but stays < 0.55:
  both (C) and (G) contribute → v5b adds CAPABILITY PROBES: for each certified
  arith skill, generate 2 numeric variants (operand perturbation, self-tests
  derived from the variant spec, certified by majority vote at admission —
  reusing v3 CheckNbhd machinery), and include them in the stratified pool.
  Still gold-free.
- **Branch B — plasticity price:** stratified vetoes rise AND frontier drops
  below sccl − 0.02: the gate catching cross-family damage is over-eager →
  allow bounded damage: veto only if >=2 skills break, or require 0/2 AND a
  margin re-check (regenerate 2 more samples before veto). Pre-register the
  tie-break: pick the variant with higher frontier.
- **Branch C — null:** arith stays ≈ 0.40 despite every family being checked:
  (G) dominates — certified instances pass while capability erodes. Then the
  fix must be capability-level: heritage probes (v5b) and/or compose with the
  winning v4 anchor arm. Record as an informative negative result; the
  instance-vs-capability gap becomes the paper's central diagnosis.

## Composition with v4

v4 (in-update anchor) and v5 (stratified veto) are orthogonal: anchor reduces
damage per update; stratified veto catches residual damage the anchor leaks.
If v4 produces a winning arm (per its own pre-registered rule), the follow-up
ladder is `sccl + anchor_best + stratified` — predicted additive: anchor
protects uncertified general capability in-gradient, stratified veto protects
certified-but-abandoned families out-of-gradient. If v4 is null, v5 stands
alone as the gate-side fix.

## Open diagnostics (run after v4 ladder frees the GPU)

- Direct test of (C): re-execute the final seed-42 sccl adapter against the 6
  vault arith skills (2 samples each, gold-free self-tests). If >=1 fails,
  (C) is confirmed pre-v5-run; if all pass, (G) is implicated. ~2 GPU-min.
- Per-phase arith erosion curve: instrument intermediate holdout evals
  (measurement-only, gold used ONLY as metric, never in-loop) to locate WHEN
  arith drops during the stream.

## v4 VERDICT (seed-42 ladder, runs/sccl_v4) — recorded 2026-08-27

Pre-registered rule: success = anchor row arith_holdout > sccl AND frontier >=
sccl.frontier - 0.02. Result: BOTH anchor rows FAIL on arith.

| learner        | ACC   | BWT    | forget | frontier | upd/rbk | arith | math  | string | drift |
|----------------|-------|--------|--------|----------|---------|-------|-------|--------|-------|
| frozen         | 0.613 | +0.132 | 0.025  | +0.588   | 0/0     | 0.600 | 0.250 | 0.800  | 0.800 |
| sccl           | 0.575 | +0.095 | 0.075  | +0.500   | 17/3    | 0.400 | 0.500 | 0.800  | 0.600 |
| sccl_anchor_lo | 0.688 | +0.207 | 0.075  | +0.613   | 16/0    | 0.400 | 0.750 | 0.800  | 0.800 |
| sccl_anchor_hi | 0.525 | +0.045 | 0.125  | +0.400   | 14/4    | 0.200 | 0.500 | 0.800  | 0.600 |
| vsr_nogold     | 0.637 | +0.157 | 0.125  | +0.512   | 12/4    | 0.400 | 0.750 | 0.600  | 0.800 |

Determinism: sccl and vsr_nogold bit-reproduce v3 seed-42 (|d|=0.000). Audit
trail: sccl all λ=0.0, anchor_lo all λ=0.1, anchor_hi all λ=0.5 — engagement
and isolation confirmed on the real ladder.

Two clean mechanistic findings:
- **F1 (anchor helps stream):** λ=0.1 is the best plasticity arm we have ever
  recorded — ACC 0.688 and frontier +0.613 BOTH exceed frozen, zero rollbacks,
  math/drift/string all at or above sccl. A trust-region pull keeps each update
  small enough to avoid clobbering just-certified skills, boosting acquisition
  AND retention of the stream. λ=0.5 over-shoots: plasticity collapses
  (ACC 0.525) and arith gets WORSE (0.200), so the benefit is non-monotone.
- **F2 (anchor does NOT help uncertified arith):** arith-holdout is 0.400 under
  λ=0.1 — IDENTICAL to sccl. So the arith erosion is not driven by the gross
  cumulative LoRA drift that a global L2 pull penalizes. It lives in specific
  arith-critical directions that a weak isotropic anchor tolerates. (λ=0.5
  damages arith further, consistent with a strong isotropic pull perturbing
  those same directions via the larger step.)

**Conclusion:** v4's anchor and v5's coverage are complementary, not redundant.
Anchor fixes the stream-retention side; only gate COVERAGE (or targeted
subspace protection) can fix uncertified arith. This upgrades the staged v5 run
from a single-mechanism test to a clean 2x2 factorial (anchor x stratified).

## v5 upgraded design: 2x2 factorial (anchor x coverage)

Ladder (torch_seed=42; seed = 42+learner index; compare WITHIN run):
- 0 frozen              — control (bit-identical to v3/v4)
- 1 sccl                — neither (bit-reproduces v3/v4 seed-42: ACC 0.575/arith 0.400)
- 2 sccl_strat          — coverage only (v5 primary hypothesis H1)
- 3 sccl_anchor_lo      — anchor only (re-verify v4 best arm at this seed)
- 4 sccl_anchor_strat   — BOTH (composition; the candidate breakthrough row)
- 5 vsr_nogold          — gold-test reference

Hypotheses:
- H1 (coverage): sccl_strat.arith > sccl.arith, frontier >= sccl - 0.02.
- H2 (composition): sccl_anchor_strat.arith > sccl.arith AND high ACC.
- BREAKTHROUGH target: sccl_anchor_strat approaches frozen arith (~0.60) while
  keeping anchor-like ACC (~0.68) — solving the plasticity-stability-
  generalization trilemma in one gold-free configuration.
- Interaction: is anchor+stratified super-additive on frontier?

Decision rule:
- If sccl_anchor_strat hits arith >= 0.55 AND frontier >= sccl.frontier - 0.02:
  candidate mechanism FOUND -> multi-seed (43,44) before headline claim.
- If only sccl_strat lifts arith (no composition needed): coverage alone is the
  fix; report anchor as stream-only helper.
- If neither lifts arith: (G) generalization gap dominates -> targeted subspace
  protection (Fisher-weighted anchor on arith directions / gradient
  projection), still gold-free. Record as informative negative result.
