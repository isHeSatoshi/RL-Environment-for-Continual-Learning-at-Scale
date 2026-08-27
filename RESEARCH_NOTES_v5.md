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
