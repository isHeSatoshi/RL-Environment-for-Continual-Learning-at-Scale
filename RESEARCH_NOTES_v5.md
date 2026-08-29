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

> **RETRACTED 2026-08-27 (same day).** The anchor penalty never engaged: a PEFT
> key mismatch made it a silent no-op in every v4/v5 run. F1/F2 below are seed
> artifacts, not anchor effects. See ADDENDUM at the bottom of this file. The
> table is kept as the raw record of what was observed.

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

## ADDENDUM 2026-08-27: the anchor no-op bug (v4 verdict RETRACTED)

**The bug.** `TrainingEngine._base_anchor()` snapshotted the adapter via
`get_peft_model_state_dict()`, whose keys strip the adapter segment
(`...lora_A.weight`), while the penalty loop in `apply_update` matches
`named_parameters()` names (`...lora_A.default.weight`). The intersection is
empty, so `n in base_anchor` was never true and `anch_pen` stayed 0.0 — the
anchor was a silent no-op in EVERY run that used it (v4 anchor_lo/hi, the v5
anchor axis, and historically vsr_bounded). Confirmed on CPU with
`scripts/_probe_anchor_keys.py`: 0 of 4 keys match. The v4 "engagement audit"
(gate["anchor_lambda"] = 0.1/0.5 on every update) passed because it audited the
CONFIG WIRING (lambda plumbed into apply_update), not the penalty's effect.

**The tell.** v5's `sccl_anchor_lo` (learner index 3 -> torch seed 45) landed
bit-identical to v3's torch_seed-44 `sccl` run (which draws 45) on every
metric — ACC, BWT, forgetting, AUC, frontier, updates/rollbacks, all of them.
Same for v4: `anchor_lo` (draws seed 44) == v3 torch_seed-43 `sccl`
(0.6875/+0.20729166666666665/16/0, exact float match), `anchor_hi` (seed 45)
== v3 torch_seed-44 `sccl` (0.525/+0.045/14/4). An anchor at lambda=0.1
producing a bitwise copy of plain sccl at the same effective seed is not a
small effect — it is zero effect.

**Consequences.**
1. v4 findings F1 ("lambda=0.1 buys plasticity, best arm ever") and F2
   ("lambda=0.5 hurts") are RETRACTED as anchor claims. The v4 ladder is
   actually three plain-sccl seeds in disguise: seed 42 -> 0.575, seed 43 ->
   0.688, seed 44 -> 0.525. Its real (unintended) contribution is a direct
   measurement of seed variance: ACC spans 0.525-0.688 across seeds at fixed
   config. Single-seed arm comparisons in this regime are uninterpretable.
2. The v5 ladder in flight (runs/sccl_v5) has an INERT anchor axis:
   sccl_anchor_lo@45 is a third sccl seed; sccl_anchor_strat@46 is a second
   sccl_strat seed (stratified veto active, anchor contributing nothing). Its
   valid novel cells: sccl_strat@44 (done: ACC 0.700, BWT +0.220, arith 0.200
   -> H1 arith endpoint FAILS even though the stratified veto demonstrably
   kept arith skills under gate coverage: 4 vetoes, all breaking mbpp_388 in
   string/drift phases) and vsr_nogold@47 (pending).
3. H1's failure mode refines the diagnosis: the stratified pool CAUGHT
   cross-family damage (every veto names an arith skill) yet arith holdout
   still collapsed — because the damage accrues INSIDE accepted updates
   (gold telemetry: arith probe 0.6 -> 0.2-0.4 within accepted updates).
   The gate checks certified INSTANCES; the holdout measures general
   CAPABILITY. The instance-vs-capability gap (G) is now the primary target.

   Sharper quantification (sccl_strat@44 trajectory, gold telemetry is
   measurement-only): of 14 accepted updates, 11 degraded the arith gold
   probe below base-0.05 — spread over ALL phases: arith 2, math_word 2,
   string 3, drift 4. Two degradations happen INSIDE the arith phase itself:
   even training on arith instances overfits away from arith generalization.
   So no pool policy over memorized instances can close the gap; the
   instances pass their stored self-tests while the capability erodes.
   (The 4 vetoes all broke arith skill mbpp_388: 3x during string, 1x during
   drift together with drift_drift_5 — coverage worked, capability still
   leaked.) Branch C / v5b must check generalization directly: capability
   probes = self-generated numeric variants with derived self-tests, admitted
   by majority vote, included in the stratified pool.

**The fix (commit e53923a).** `_base_anchor()` now snapshots from
`self.model.named_parameters()` (requires_grad only), matching the penalty
loop's iteration exactly. Two new engine-level tests guard engagement, both
FAIL on the pre-fix code and pass on the fix:
- `test_anchor_base_keys_match_named_parameters`: key-set agreement.
- `test_anchor_actually_pulls_weights_toward_base`: with identical seed/data/
  init, anchored distance-to-init < unanchored (lambda 5.0 vs 0.0).
Full suite: 270 passed. The non-anchor code path is untouched (anchor snapshot
is only taken when anchor_lambda > 0), so all non-anchor rows remain
bit-reproducible — verifiable in the corrected rerun.

**Revised plan.**
- Let runs/sccl_v5 finish. Treat it as: sccl@43, sccl_strat@44,
  sccl_strat@46 (as anchor_strat), sccl@45 (as anchor_lo), vsr_nogold@47 —
  i.e., seed-variance data for sccl {42,43,45} and sccl_strat {44,46}.
- Launch the CORRECTED 2x2 factorial (configs/sccl_v5_fixed.json ->
  runs/sccl_v5_fixed, same seed map, fixed engine). Determinism check built
  in: frozen/sccl/sccl_strat/vsr_nogold rows must bit-reproduce the inert
  ladder; the anchor rows are the first REAL anchor measurements.
- Pre-registered interpretation of the corrected factorial stands as written
  above (H1/H2/BREAKTHROUGH rule), now measuring what it claims to measure.
- If arith still fails with a genuinely engaged anchor + stratified coverage:
  Branch C confirmed -> v5b capability probes (self-generated numeric variants
  of certified skills, certified by majority vote, added to the stratified
  pool). Gold-free throughout.

**Meta-lesson for the paper.** A regularizer's audit trail must verify EFFECT
(penalty > 0, weights pulled), not just configuration. This bug survived v4
because the plausible seed-confounded story (F1/F2) was never checked against
the seed-matched controls that already existed in runs/sccl_v3_seeds. The
correction strengthens the paper: the trilemma result, if it survives the
corrected factorial, rests on an engagement-tested mechanism.

**Ops post-mortem: the double-launch (same day).** The corrected factorial
was staged behind a polling wrapper (v5_fixed_launcher.sh: wait for the first
flight's metrics.json, then launch). The wrapper's first launch via `nohup
... &` appeared to die but DID NOT; the re-launch via the background task
created a SECOND wrapper. Both polled the same trigger, both fired runners
into the SAME out_dir concurrently. One runner segfaulted (rc=139) under GPU
contention (~15.5/16 GB in use with both models resident); the survivor's
output was contaminated by interleaved writes. Recovery: kill the whole
process tree (verify parentage via Win32_Process before killing — the nohup
tree was launcher 34672 -> bash 3960 -> python 54908), wipe runs/sccl_v5_fixed,
re-launch EXACTLY ONE runner directly (no polling wrapper; the trigger file
already existed). Rules going forward: (1) never use polling wrappers to chain
runs — launch directly; (2) before launching a GPU runner, enumerate python
processes and confirm sole ownership of the GPU; (3) a watcher that only READS
completion artifacts (like v5_fixed_check_watcher.sh, which runs the verdict
script when metrics.json settles) is safe because it cannot double-launch.

## v5b DESIGN DRAFT — capability probes (Branch C; implement only if the
## corrected factorial still fails the arith endpoint)

Diagnosis (now quantified on sccl_strat@44): 11/14 accepted updates degraded
the arith gold probe; 2 degradations occurred INSIDE the arith phase itself.
Memorized instances pass their stored self-tests while the capability the
holdout measures erodes. No pool policy over memorized instances closes this.

Mechanism (all gold-free):
1. At certification of an arith-family skill, additionally manufacture K=2
   CAPABILITY VARIANTS: numeric perturbations of the task's constants
   (reuse CheckNbhd's numeric-variant machinery from v3), each with self-tests
   derived from the VARIANT spec, certified by majority vote (tau_math-style).
   Variants that fail certification are discarded (never used).
2. Store certified variants in the vault as kind="cap_probe", family = parent
   family, never trained on (like v2 probes).
3. Stratified veto extension: the per-family check pool includes the newest
   cap_probe of each family alongside the newest skill (so an arith update is
   vetted iff it breaks arith GENERALIZATION, not just the memorized instance).
   Deterministic selection; veto rule unchanged (ANY of n_samples passes ->
   retained).
4. Bounded-damage guard (pre-registered Branch B tie-break): if cap_probe
   vetoes exceed a budget (>=50% of updates in a phase vetoed), fall back to
   margin re-check (2 extra regenerations) before veto.

Predictions:
- If (G) dominates: cap_probe vetoes fire INSIDE family phases (including
  arith-on-arith), arith-holdout stabilizes near frozen 0.6, plasticity cost
  measurable in updates-accepted and ACC.
- Pre-registered success: arith >= 0.55 AND frontier >= sccl - 0.02, then
  multi-seed.

Refinement (review while corrected factorial runs): mechanisms 1 and 3 are
inconsistent as drafted — manufacture is restricted to arith-family skills,
but the stratified pool expects "the newest cap_probe of EACH family". Resolve
by domain, mirroring v3 CheckNbhd: manufacture cap_probes for ALL certified
skills — numeric variants + majority vote for arith/math_word, paraphrase
variants + fresh self-tests for string/drift — and let the pool take the
newest cap_probe per family. This also tests whether the instance-vs-
capability gap is a general phenomenon (v5 telemetry shows string/drift
holdouts also dip under sccl, just less catastrophically than arith).

---

## v5 CORRECTED FACTORIAL — VERDICT (seed 42, engaged anchor, commit e53923a code)

Run: configs/sccl_v5_fixed.json -> runs/sccl_v5_fixed (runner exec_11de2e72).
Same stream hash 554ce43f182b. Final per-family holdout (arith/math/string/drift)
+ frontier score:

| learner          | arith | math | string | drift | frontier | ACC   | upd/rbk | BWT     | forget |
|------------------|-------|------|--------|-------|----------|-------|---------|---------|--------|
| frozen           | 0.600 | 0.250| 0.800  | 0.800 | +0.588   | -     | 0/0     | -       | -      |
| sccl             | 0.400 | 0.500| 0.800  | 0.600 | +0.500   | 0.575 | 17/3    | -       | -      |
| sccl_strat       | 0.200 | 1.000| 0.800  | 0.800 | +0.575   | 0.700 | 14/4    | -       | -      |
| sccl_anchor_lo   | 0.400 | 0.750| 0.800  | 0.800 | +0.613   | 0.688 | 19/1    | +0.2073 | 0.075  |
| sccl_anchor_strat| 0.400 | 0.750| 0.800  | 0.800 | +0.613   | 0.688 | 11/7    | +0.207  | 0.075  |
| vsr_nogold       | (in flight at time of writing)                                        |

Anchor ENGAGEMENT audit (first real anchor measurements in this repo):
- anchor_lo: 19/19 accepted updates with anchor_pen > 0 (min 0.5726, mean
  1.5974, max 4.1245), lambdas exactly {0.1}.
- anchor_strat: 11/11 accepted with pen > 0 (min 0.7598, mean 1.3836, max
  2.9918), lambdas {0.1}.
- Negative control sccl_strat: 7/7 accepted with lam=0.0, pen=0.0 (inert path
  unchanged). Determinism: frozen/sccl/sccl_strat rows bit-reproduce the first
  flight on all computed metrics (wallclock/paths/timestamps only differ).

Pre-registered rule outcomes:
- H1 (stratified coverage): sccl_strat.arith = 0.200 <= sccl.arith = 0.400.
  FAIL. (Frontier condition +0.575 >= +0.480 passes, but the primary endpoint
  fails.) Stratification HURTS arith: with per-family instance slots always
  checked, arith instances keep passing while arith capability degrades, so
  MORE arith-breaking updates are admitted than under the unstratified pool
  (where late-phase pools are dominated by non-arith skills and some
  arith-breaking updates get coincidentally vetoed for also breaking a recent
  skill). This is direct evidence for the instance-vs-capability gap (Branch C).
- H2 (anchor main effect): anchor_lo.arith = 0.400 = sccl.arith. FAIL on the
  arith endpoint. Anchor DOES help elsewhere: math 0.50 -> 0.75, drift
  0.60 -> 0.80, frontier +0.500 -> +0.613, rollbacks 3 -> 1, forgetting 0.075,
  BWT +0.207. The quadratic anchor preserves base-model capability broadly but
  cannot hold the FIRST family once later-phase training proceeds.
- BREAKTHROUGH (composition): sccl_anchor_strat.arith = 0.400 < 0.55. FAIL
  (frontier +0.613 >= +0.480 passes). Notably anchor_strat lands on the SAME
  final per-family vector and frontier as anchor_lo ([0.4, 0.75, 0.8, 0.8],
  +0.613) via a different path (11 accepted/7 rollbacks vs 19/1): the anchor
  neutralizes the stratification harm to arith (0.2 -> 0.4) but adds nothing
  on top of it.

Reading: the anchor is real, engaged, and beneficial (best frontier, best BWT,
lowest forgetting in the ladder), but no cell cracks the arith endpoint. The
bottleneck is now unambiguously the gate's INSTANCE-level check: accepted
updates keep passing stored self-tests while holdout capability erodes
(sccl_strat@44 telemetry: 11/14 accepted updates degraded the arith gold
probe). Gold says so, and the gate's own artifacts agree - the gate is
admitting capability-loss updates because it only re-verifies instances.

Pre-registered next step fires: v5b capability probes (this file, Branch C
draft + refinement above). configs/sccl_v5b.json committed (6cab00c);
implementation committed (51ffada); 279 tests pass; all switches default OFF.
Launch after vsr_nogold completes + verdict_check.md confirms + sole GPU
ownership. Pre-registered v5b success rule unchanged: capprobe_strat
arith >= 0.55 AND frontier >= sccl - 0.02 -> multi-seed before headline claim.

---

## POST-v5b DECISION TREE (pre-registered 2026-08-28, before v5b ladder results)

The v5b ladder (frozen / sccl / sccl_capprobe / sccl_capprobe_strat, seed 42)
is running. Whatever it returns, the next move is fixed here so no
post-hoc rationalization enters:

1. BREAKTHROUGH PASS (capprobe_strat.arith >= 0.55 AND frontier >= sccl-0.02):
   -> run scripts/v5b_multiseed.py (seeds 43/44; snapshot 42 -> _s42).
   -> headline claim only if arith mean >= 0.55 across seeds AND every seed
      individually >= sccl_seed - 0.05 on arith (no single-seed rescue).
   -> paper sec:v5b "Results: pending" filled with the multi-seed table;
      v5's instance-vs-capability framing becomes the paper's central arc.

2. H1 PASS only (capprobe.arith > sccl.arith, stratified row does not add):
   -> capability probes are the fix; stratification remains harmful/neutral
      even over capability evidence. Write-up: probes are the contribution,
      stratification is a confirmed dead end (two independent factorials).
   -> Branch D = probes + anchor composition (anchor helped math/drift in v5;
      probes target arith; likely additive). Pre-register the 2x2 before run.

3. FAIL on arith again (both probe rows <= sccl on arith):
   Diagnose from artifacts BEFORE designing Branch D:
   a) probe fragility: cap probes veto too aggressively -> high rrv_vetoes,
      low updates_used_pct, frontier collapse. Fix: loosen tau on the cap
      stratum only, or require k-of-n probe agreement.
   b) uncertified capability: arith erodes BEFORE any arith skill is
      certified, so no arith probe exists when the damage happens (the gate
      cannot protect what it never saw). Evidence: checked_cap=0 gates during
      the arith-erosion window in trajectories jsonl. Fix: manufacture probes
      at FIRST CONTACT with a family, not at certification (Branch D leading
      candidate — extends coverage to the pre-certification window).
   c) cross-family interference: probes exist and pass pre-update, but the
      update passes the check yet still erodes (probe insensitive to the
      damaging direction). Evidence: checked_cap>0 gates whose post-update
      gold telemetry still shows arith degradation. Fix: harder probe
      generation (multi-step variants) or probe ensembles.
   The three diagnoses make different predictions about veto counts and the
   timing of arith erosion in family_curve; the artifacts disambiguate.

---

## SCCL v5b (capability probes) — VERDICT (seed 42, 2026-08-28, runs/sccl_v5b)

Pipeline: launcher smoke audit PASS (cap_probes_committed=1, checked_cap gate=1,
sccl control leak=0) after fixing TWO instrumentation bugs first (unregistered
learner names silently skipped by run_experiment — committed 135ea35; audit
reading gate records from the wrong file — d26a48b). Ladder rc=0 in ~3.1h.

| learner             | ACC   | BWT    | Forget | Frontier | Upd/Rb | arith | math  | string | drift |
|---------------------|-------|--------|--------|----------|--------|-------|-------|--------|-------|
| frozen              | 0.613 | +0.132 | 0.025  | +0.588   |  0/0   | 0.600 | 0.250 | 0.800  | 0.800 |
| sccl                | 0.575 | +0.095 | 0.075  | +0.500   | 17/3   | 0.400 | 0.500 | 0.800  | 0.600 |
| sccl_capprobe       | 0.625 | +0.145 | 0.131  | +0.494   |  9/10  | 0.375 | 0.750 | 0.600  | 0.775 |
| sccl_capprobe_strat | 0.694 | +0.214 | 0.131  | +0.562   | 15/9   | 0.375 | 1.000 | 0.600  | 0.800 |

Verification (scripts/v5b_check.py, auto-run by watcher): frozen + sccl
BIT-IDENTICAL to runs/sccl_v5_fixed (switch inertness + determinism); probes
made/committed 12/12 and 15/15; gates with checked_cap>0 = 19/19 and 24/24
(every gate after first certification re-checked probes); cap-vetoes 4 and 2;
sccl control fully isolated. All checks passed.

Pre-registered rules:
- H1 (cap main effect): capprobe.arith = 0.375 vs sccl 0.400 -> FAIL (frontier
  side passes: +0.494 >= +0.480). FAILS ON THE ARITH ENDPOINT AGAIN.
- H2 (stratified cap): capprobe_strat.arith = 0.375 >= capprobe 0.375 AND
  frontier +0.562 >= +0.480 -> PASS. First H2 pass in the v5/v5b series.
- BREAKTHROUGH: capprobe_strat.arith = 0.375 < 0.55 -> FAIL (frontier passes).

Findings:
1. capprobe_strat is the BEST LEARNER IN THE PROJECT so far: ACC 0.694 beats
   even the frozen baseline (0.613) — genuine absolute learning above the
   static model; best BWT (+0.214); frontier +0.562 second only to frozen's
   free +0.588; math PERFECT (1.000, up from 0.500 under sccl); drift 0.800
   = frozen level. Capability probes + stratification compose into a real
   aggregate advance (math 0.50 -> 1.00 is the largest single-family gain
   any mechanism has produced).
2. The probes WORK where they have coverage: every accepted update passed the
   newest-per-family probes at its gate (24/24 checked_cap>0), and the
   late-phase families reached their best-ever scores. The instance-vs-
   capability gap of v5 is genuinely narrowed: instance evidence -> one
   capability variant per family.
3. ...but arith is STILL unprotected: 0.375 under BOTH probe rows, identical
   endpoint from two very different gate regimes (9 vs 15 accepted updates,
   10 vs 9 rollbacks). The arith probe existed from phase 1 and was re-checked
   at every subsequent gate — and every accepted update passed it — yet arith
   holdout eroded from 0.600 (frozen) to 0.375. Pre-registered tree branch (c):
   the single variant probe is insensitive to the damaging direction.
4. Survivorship overfit (new mechanism finding): capprobe's TRAINED arith
   rose to 0.800 while its arith HOLDOUT fell to 0.375 (gap 0.425 vs sccl's
   0.200). High veto pressure selects updates that fit the current instance
   and the probe tightly; the accepted population overfits MORE than under
   sccl. The gate is a FILTER, not a REGULARIZER: filtering reshapes the
   accepted-update distribution toward instance-narrow survivors.
5. Starvation cost: string fell 0.800 -> 0.600 under both probe rows (veto
   pressure rejected string-phase updates; forgetting 0.131 vs sccl 0.075).

Reading: v5b converts the v5 diagnosis into the strongest aggregate learner
yet (ACC 0.694, math 1.000) and validates capability-level evidence as the
right DIRECTION, but one held-out variant per family is too thin a witness:
it passes while broad capability erodes (branch c), and veto pressure itself
degrades the accepted-update distribution (finding 4). Per the pre-registered
decision tree, next step is Branch D, informed by the adapter-level gold
telemetry (which accepted update broke arith, and did the probe really pass).

## POST-HOC GOLD TELEMETRY — arith erosion timeline (2026-08-28, runs/sccl_v5b)

Measurement-only analysis of the `gold_telemetry` block the gate logged at run
time on the arith gold holdout (4 tasks, never used in any decision). Script:
`scripts/v5b_telemetry.py`; artifact: `runs/sccl_v5b/telemetry_arith_erosion.json`.
`base` = frozen arith gold (0.600, cached at the first gate); `cand` = the arith
gold level each ACCEPTED adapter left behind. Erosion = cand < prev_cand − 0.05.

Per-learner headline:

| learner | accepted | erosion updates | probe-checked erosions | probe PASSED despite erosion | insensitivity | final arith gold |
|---|--:|--:|--:|--:|--:|--:|
| sccl | 17 | 4 (23.5%) | 0 (probes off) | — | — | 0.400 |
| sccl_capprobe | 9 | 1 (11.1%) | 1 | 1 | 100% | 0.375 |
| sccl_capprobe_strat | 15 | 4 (26.7%) | 4 | **4** | **100%** | 0.375 |

Every probe-checked erosion passed its capability probe. The veto's capability
stratum had a 0% true-positive rate on real arith damage in this run.

Causal timeline for `sccl_capprobe_strat` (the best row):
- arith phase + math phase are NET-SAFE: arith gold oscillates 0.4–0.6 and is
  back at the frozen 0.600 by v6/v7 (end of math phase). The gate + probes are
  not the story here — plain in-phase training doesn't kill arith.
- v11 (string phase, mbpp_57 "largest number from digits"): 0.575 → 0.375,
  −0.200, cap-probe checked=3, verdict PASS. Gold gate would REJECT.
- v12 (string phase, mbpp_587 list→tuple): 0.375 → 0.600, +0.225 (recovery).
- **v13 (drift phase, drift_drift_0 "elementwise tuples"): 0.600 → 0.150,
  −0.450 — the catastrophic update. cap-probe checked=3, verdict PASS (all three
  family probes). Gold gate would REJECT.** Instance stratum also PASS (checked=2).
- v14 (drift phase, drift_drift_1): 0.150 → 0.375, +0.225 (partial recovery).
  Final 0.375 = the metrics.json endpoint exactly.

So the terminal damage is inflicted by CROSS-FAMILY updates (string v11, drift
v13), not by arith-phase training — and both sailed through all three capability
probes.

Why the probe is blind here (the refined mechanism): the arith family is mbpp
offset 0, which is SKILL-HETEROGENEOUS. Its 4 gold holdout tasks are
mbpp_444 (trim tuples), mbpp_82 (sphere volume), mbpp_581 (square-pyramid
surface area), mbpp_304 (element after rotation): two numeric/geometry tasks
(frozen 1.0, 1.0) and two list/tuple-manipulation tasks (frozen 0.2, 0.2).
Final per-task scores for capprobe_strat: mbpp_444 0.2 (held), mbpp_82 1.0
(held), **mbpp_581 1.0 → 0.2 (destroyed)**, mbpp_304 0.2 → 0.1. The newest
arith cap-probe at v13 was mbpp_604:c0, a paraphrase of "reverse the order of
words" — a STRING-manipulation witness. The catastrophic drift update
(tuple-manipulation) destroyed the numeric/geometry sub-skill (pyramid surface
area) while leaving the string-manipulation sub-skill intact, so the word-
reversal probe passed. "Family" is coarser than "skill dimension": a single
newest-per-family probe witnesses one skill axis of a multi-axis family.

This sharpens the v5b verdict into three separable failure mechanisms:
1. Skill-heterogeneity blindness: one probe per family cannot cover a family's
   multiple skill axes; cross-family updates hit the unwitnessed axis.
2. Filter-not-regularizer: even when a probe could in principle catch an update,
   the veto only selects among candidates; it never pulls weights, so accepted
   updates still drift (survivorship overfit, trained 0.8 vs holdout 0.375).
3. Cross-family interference is the dominant damage source: in-phase arith/math
   training net-preserved arith; string/drift training destroyed it.

Branch D must therefore attack (1) with a DIVERSE witness (an ensemble spanning
a family's skill axes, not k variants of one spec) and (2) with an in-update
regularizer (anchor) that protects unwitnessed axes — the two are complementary
exactly as v5's anchor (weights) and v5b's probes (evidence) were.

## BRANCH D PRE-REGISTRATION — SCCL v6 (capability ENSEMBLES + anchor) (2026-08-28, BEFORE any v6 run)

The v5b telemetry localizes three separable failure mechanisms:
  M1 skill-heterogeneity blindness — one newest-per-family probe witnesses ONE
     skill axis; a cross-family update destroys an unwitnessed axis (arith
     numeric/geometry axis collapsed while the string-axis probe passed).
  M2 filter-not-regularizer — the veto only selects among candidates, never
     pulls weights, so accepted updates still drift (survivorship overfit).
  M3 cross-family interference — in-phase arith/math training net-preserved
     arith; the terminal damage was inflicted by string/drift-phase updates.

Branch D attacks M1 with a WIDER witness and M2/M3 with an in-update
regularizer, composed. Both are the exact complements the v5/v5b arc predicts.

INTERVENTION 1 — capability ENSEMBLE (attacks M1). Generalize the v5b cap-probe
pool from newest-ONE-per-family to a bounded ENSEMBLE of up to K distinct
source-skill probes per family. The veto re-checks ALL probes in the ensemble
(any break = veto). K=1 must be bit-identical to v5b (inertness); K=3 is the
treatment. Rationale: the ensemble spans K skill axes of a heterogeneous family
instead of one, raising the veto's true-positive rate on cross-family damage.
Gold-free: probes are still manufactured from spec only, never trained on.

INTERVENTION 2 — in-update base anchor (attacks M2/M3). Compose the engaged v4/v5
anchor (quadratic pull toward LoRA init, lambda=0.1) on top of the capability
gate. Rationale: the anchor trims per-update weight drift and protects skill
axes that no probe witnesses; v5 showed it is the best stability cell
(frontier +0.613, BWT +0.207) and fully neutralized stratification harm.

LADDER — configs/sccl_v6.json. Same stream hash 554ce43f182b, torch_seed=42,
seed = 42 + learner index, gold-free throughout (gold only final eval +
telemetry). Learner index is FIXED to preserve the v5b seeding for the
determinism rows:
  idx  name                  K  anchor  strat  role
  0    frozen                -  off     -      determinism vs v5b (seed 42)
  1    sccl                  -  off     -      determinism vs v5b (seed 43)
  2    sccl_capprobe         1  off     off    determinism vs v5b (seed 44)
  3    sccl_capprobe_strat   1  off     on     determinism vs v5b (seed 45)
  4    sccl_capens           3  off     on     H1 ensemble (seed 46)
  5    sccl_cap_anchor       1  on .1   on     H2 anchor (seed 47)
  6    sccl_capens_anchor    3  on .1   on     H3 composition / candidate (48)
The 2x2 factorial (ensemble {K1,K3} x anchor {off,on}) sits on the stratified
capability base; its K1/anchor-off cell IS the v5b capprobe_strat rerun, so the
ensemble and anchor main effects and their interaction are all identified.

HYPOTHESES (pre-registered, falsifiable):
  H1 (ensemble):  sccl_capens arith > sccl_capprobe_strat arith (0.375).
  H2 (anchor):    sccl_cap_anchor arith > sccl_capprobe_strat arith (0.375).
  H3 (composition): sccl_capens_anchor arith >= max(capens, cap_anchor) AND
     best ACC among the capability cells.
  BREAKTHROUGH rule: sccl_capens_anchor arith >= 0.55 AND frontier >=
     sccl.frontier - 0.02  ->  multi-seed (43/44) BEFORE any headline claim.
  Prediction (direction): if M1+M3 dominate, the ensemble rows veto more
     cross-family damaging updates (higher cap-veto true positives in
     string/drift phases), stabilizing arith toward frozen 0.6 at a measurable
     plasticity cost; anchor rows reduce broad per-update drift; composition
     does both.

FAIL-CLOSED CHECKS (any failure = abort, do not interpret):
  C1 determinism: frozen, sccl, sccl_capprobe, sccl_capprobe_strat rows
     bit-reproduce runs/sccl_v5b on every computed metric.
  C2 ensemble engagement: K=3 rows must retain >1 distinct-source-skill
     cap_probe per family in the vault AND log checked_cap >1 in gate records.
  C3 anchor engagement: anchor rows log anchor_pen > 0 on every accepted update
     (and the K=1/off rows log exactly 0).
  C4 isolation: frozen/sccl rows commit zero cap_probes and log checked_cap=0.
  C5 gold-free: no gold field enters any accept/reject decision (AST audit +
     the cap/anchor code paths read only spec/self-tests/init-weights).

## BRANCH D STATUS — implementation + smoke + ladder launch (2026-08-28)

IMPLEMENTED (commit 11886c4). Capability-probe ENSEMBLE pool + base-anchor
composition, fully gold-free:
  * gcl/vault.py — _cap_probe_source / _cap_pool_by_family / _cap_pool_keep_ids.
    The pool keeps the FRESHEST variant of each distinct source skill, then the
    K newest by commit order; K=1 reduces EXACTLY to the v5b newest-per-family
    rule (bit-identity pinned by test_cap_pool_pool1_is_v5b_newest_per_family).
    commit_cap_probe prunes to K; selfreplay_veto re-checks the whole pool.
  * gcl/config.py — sccl_capprobe_pool (default 1) + sccl_capprobe_pools
    (per-learner K override). env.py resolves the veto-side K, experiment.py the
    manufacture-side K (kept in lock-step so the retained pool == checked pool).
  * gcl/learners/learners.py — sccl_capens (K3), sccl_cap_anchor (K1+anchor .1),
    sccl_capens_anchor (K3+anchor .1); thin name-supplying SCCLLearner rows.
  * Tests: 8 new (pool=1 v5b bit-identity, distinct-source ensemble,
    freshest-variant-per-skill, commit prune-to-K, ensemble veto checks-all-and-
    vetoes-any, cap_pool default=v5b single, v6 config inert, registration).
    Full suite green: 87 sccl + 97 vsr/unit.

SMOKE AUDIT — PASS (runs/_smoke_v6, 2 families x 4 tasks, 4-learner 2x2):
  * C4 isolation: sccl committed 0 cap_probes, checked_cap=0, anchor_pen=0.
  * C2 ensemble: sccl_capens held 3 distinct source skills in math_word; the
    veto checked up to 4 probes in a single gate (checked_cap=4); anchor_pen=0.
    -> the ensemble veto genuinely re-checks MULTIPLE skill axes at once, the
    exact wider witness v5b's single-probe gate lacked.
  * C3 anchor: sccl_cap_anchor anchor_pen>0 lambda=0.1 (1/1 accepted);
    sccl_capens_anchor anchor_pen>0 on 4/4 accepted updates.
  The composition row's ensemble was under-exercised in this tiny seed (only 1
  distinct arith skill certified — smoke stochasticity), but the ensemble code
  path is proven via sccl_capens and the anchor via both anchor rows; the full
  8-task/family ladder exercises all cells.

LADDER LAUNCHED (autonomous, fail-closed): configs/sccl_v6.json, 7 learners
(frozen/sccl/sccl_capprobe/sccl_capprobe_strat determinism rows + sccl_capens/
sccl_cap_anchor/sccl_capens_anchor treatment rows), seeds 42-48 by learner
index. scripts/v6_launcher.sh gated smoke->audit->ladder; scripts/
v6_check_watcher.sh will run scripts/v6_check.py (C1-C5 + H1/H2/H3/BREAKTHROUGH)
when runs/sccl_v6/metrics.json finalizes. Gold-free throughout; gold only final
eval + post-hoc telemetry.

INTERIM (ladder in flight, 2026-08-28 ~06:00-09:00): ALL FOUR determinism
rows reproduced runs/sccl_v5b BIT-FOR-BIT (frozen acc 0.6125 / frontier 0.5875;
sccl acc 0.575 / frontier 0.5; sccl_capprobe acc 0.625 / frontier 0.4938 /
9 updates / 10 rollbacks; sccl_capprobe_strat acc 0.6938 / frontier 0.5625 /
15 updates / 9 rollbacks / arith holdout 0.375; every report key + per-family
holdout identical on all four). C1 is therefore satisfied ahead of the formal
check; any treatment-row difference is attributable to the ensemble/anchor alone.
Live mechanism signal from sccl_capprobe (K=1 control; trajectory steps):
the cap stratum fired 4 cross/within-family vetoes over the stream —
  * step 0 (arith phase): update on mbpp_244 broke its own fresh probe
    mbpp_244:c0 (checked_cap=1);
  * step 18 (STRING phase): update on string task mbpp_259 broke the
    math_word probe math_42_2:c0 AND the string probe mbpp_259:c0
    (checked_cap=3) — cross-family erosion caught mid-stream;
  * steps 30-31 (DRIFT phase): two drift updates broke the arith probe
    mbpp_388:c0 (checked_cap=4) — exactly the terminal cross-family damage
    pattern the v5b telemetry identified, here VETOED by the cap stratum.
So even K=1 provides real cross-family protection when the newest probe
happens to sit on the damaged axis. But arith still ended at ~0.35: the
math_word-phase updates that eroded arith (gold telemetry cand 0.350/0.375)
passed the 1-2 probes checked at the time — the unwitnessed-axis failure.
The K=3 ensemble rows test whether more distinct-source probes per family
close that residual gap.

ENSEMBLE ROW RESULT (sccl_capens, seed 46, finalized in flight): H1 FAILS.
acc 0.6500, frontier 0.525, arith holdout 0.200 (WORSE than K=1's 0.375),
math 1.000 held, string 0.600->0.800, drift 0.800->0.600, 12 upd / 5 rb.
The ensemble pool genuinely spanned 3 distinct arith axes (tuple-average
mbpp_615, word-reverse mbpp_604, numeric mbpp_388). Gate trace (gold
telemetry, measurement only):
  * The ensemble ADDED true positives: mbpp_615:c0 vetoed 3 cross-family
    damaging updates (math_word idx13, string idx20, drift idx25).
  * Terminal blow: string-phase update mbpp_62 (idx16) dropped arith gold
    0.40 -> 0.20 while passing ALL 5 checked probes incl. the numeric-axis
    probe mbpp_388:c0. Probe insensitivity persists at K=3.
  * In-phase erosion also passed early (idx0 mbpp_244: 0.6->0.4, chk=1,
    pool still forming; idx4 mbpp_615 itself: 0.6->0.4).
  * cap_guard armed at 3 gates (more probes -> more breaks -> guard margin
    re-checks); 2 of the 3 armed gates accepted; none of the armed accepts
    caused measured arith erosion, so the guard is not indicted here.
Refined mechanism (M1'): the binding constraint is NOT witness COUNT but
witness SENSITIVITY. Single-instance probes are point witnesses; holdout
damage moves in directions orthogonal to every witnessed spec. Widening the
witness set 1->3 distinct axes raised cross-family true-positive vetoes
(0->3) yet left the terminal damage path unwitnessed, and seed-46 in-phase
erosion (absent in seed-45) made the endpoint worse. Prediction for the
anchor rows: the quadratic pull is the only mechanism here that acts on the
weights themselves, hence the only candidate to protect unwitnessed axes;
if H2/H3 also fail, Branch E must convert the gate's discrete pass/fail
evidence into a continuous constraint (margin-tracked probes and/or
probe-loss-projected gradients) rather than adding more point witnesses.

ANCHOR ROW RESULT (sccl_cap_anchor, seed 47, K=1 + anchor λ=0.1): H2 PASSES
marginally. acc 0.6250, frontier 0.5500, arith holdout 0.400 > 0.375
(+0.025), forgetting 0.075 (BEST among capability cells, matching plain
sccl), math 0.500 (DOWN from 1.000 — the anchor suppresses beneficial
plasticity too), string 0.800, drift 0.800, 12 upd / 5 rb. C3 engaged:
anchor_pen > 0 on ALL 12 accepted updates (range 0.57–1.48), λ=0.1.
Gold timeline (measurement only): erosion events still happen (0.6->0.4 at
idx1/idx18/idx28) but are SHALLOWER and partially recovered — the exact
update that killed arith in the ensemble row (string-phase mbpp_62) left
arith at 0.6 under the anchor. The anchor acts as intended: it bounds
per-update drift rather than preventing it, so arith oscillates 0.4-0.6
instead of ratcheting down to 0.2. Net: the regularizer, not the witness,
is the mechanism protecting unwitnessed axes — exactly the prediction.
Cost: math plasticity halved (1.000 -> 0.500). The composition row tests
whether ensemble true-positive vetoes + anchor drift-bounding compose.

V6 FINAL VERDICT (2026-08-28, runs/sccl_v6, verdict_check.md, rc=0):
  H1 FAIL | H2 PASS | H3 FAIL | BREAKTHROUGH: FAIL | C1-C5 ALL PASS.
  frozen 0.613/0.588 | sccl 0.575/0.500 | capprobe 0.625/0.494 arith .375
  | strat 0.694/0.562 arith .375 | capens 0.650/0.525 arith .200 |
  cap_anchor 0.625/0.550 arith .400 | capens_anchor 0.625/0.550 arith .400.
  H3 fails on ACC: composition ties cap_anchor exactly (0.400 >= max(.2,.4)
  holds with equality, but 0.625 < strat's 0.694).
FINDINGS (from per-update gold telemetry, telemetry_v6.json):
  F1 Insensitivity is 100% in EVERY cap row: all probe-checked erosions
     passed their probes, at K=1 AND K=3. Witness count does not buy
     sensitivity. (M1' confirmed as the binding constraint.)
  F2 The ensemble DID add cross-family true-positive vetoes (3 in capens,
     all via the tuple-average probe) but the terminal damage path bypassed
     all three arith axes incl. the numeric one.
  F3 The composition row is metrically IDENTICAL to the anchor row on every
     holdout (0.400/0.500/0.800/0.800, frontier 0.550) with MORE rollbacks
     (8 vs 5). Under drift-bounded updates the ensemble's extra vetoes
     remove updates that were already bounded-harmless: the mechanisms do
     NOT synergize; the anchor absorbs the ensemble's contribution.
  F4 Under the anchor, arith gold is a BOUNDED RANDOM WALK on {0.4, 0.6}:
     every erosion step is exactly -0.2 and every recovery +0.2; recoveries
     occur only when arith/math-family training happens; terminal damage
     persists only because it is inflicted in the last phases after arith
     training has ended. The anchor raised the floor (0.2 -> 0.4) but the
     endpoint is decided by the last 1-2 damaging updates, not cumulative
     erosion. In-phase first-update erosion (0.6->0.4) occurred in 3 of 4
     cap rows (seeds 46/47/48), so seed-45's net-preservation was the
     exception.
DECISION: pre-registered rule R3 fires (composition arith 0.400 < 0.45):
  Branch E = probe-sensitivity track, cheap-first: E1 pass-rate margin veto
  (replace any-of-n with pass-rate >= threshold); fallback E2 gold-free
  probe-gradient projection. Additional design input from F4: any E1
  configuration should be evaluated against the bounded-walk dynamic — the
  endpoint problem is now "prevent the LAST damaging updates of the stream",
  which a more sensitive veto can do only if it fires on subthreshold drift.

POST-v6 BRANCH E DECISION RULE (pre-registered BEFORE the composition-row
result, so the next-branch choice is not fit to the final number):
  R1 BREAKTHROUGH: PASS (composition arith >= 0.55 AND frontier >= 0.48)
     -> multi-seed 43/44 confirmation, then paper finalization. Arc complete.
  R2 composition arith in [0.45, 0.55): the composition improves on anchor
     alone but falls short -> Branch E = make the anchor SELECTIVE:
     E3 competence-snapshot anchor (anchor point = LoRA state at each
     family's certification moment, not θ0), attacking the anchor's
     plasticity tax (math 1.000 -> 0.500) while keeping drift-bounding.
  R3 composition arith < 0.45 (no synergy; anchor dominates or ensemble
     hurts): Branch E = probe SENSITIVITY track, cheap-first:
     E1 pass-rate margin veto — replace "any-of-n passes = retained" with
        pass-rate >= threshold (e.g. 2/3 of n samples per probe); subthreshold
        damage degrades pass rate before it breaks the probe. 1-parameter,
        gold-free, attacks M1' directly.
     If E1 fails or guard-arming spirals -> E2 gold-free probe-gradient
        projection (A-GEM with the certified probe pool as the constraint
        set): project each update's gradient off probe-loss-increasing
        directions. Continuous in-update constraint; novel constraint set.
  R4 any fail-closed C1-C5 failure -> abort interpretation, fix, rerun.

========================================================================
BRANCH E PRE-REGISTRATION — E1 PASS-RATE MARGIN VETO (v7 ladder)
Written BEFORE any v7 implementation or run, per project discipline.
========================================================================

MECHANISM MOTIVATION (from v6 findings F1/F4)
  F1 showed insensitivity is 100%: every probe-checked erosion passed its
  probes, at K=1 AND K=3. But "passed" was scored under an ANY-of-n retain
  rule (n=2 samples, temp 0.7): a probe whose regeneration quality degrades
  from pass-rate 1.0 to 0.5 still "passes" as long as ONE of 2 draws
  survives. Subthreshold capability damage should degrade pass RATE before
  it breaks the probe entirely. E1 turns the veto's evidence from a binary
  existential ("does some draw still pass?") into a measured rate
  ("what fraction of draws pass?"), and requires the rate to clear a
  threshold theta. This is a 1-parameter, gold-free upgrade of the exact
  witness v6 proved insensitive — it attacks M1' (witness sensitivity)
  directly without adding witness count (which v6 proved insufficient).

DESIGN (E1 — cap_retain_min threshold on cap-probe pass rate)
  D1. New veto parameter cap_retain_min (float, default 0.0) and
      cap_samples (int, default 0 = fall back to n_samples). Applied ONLY
      in the cap-probe re-check section of selfreplay_veto (math AND code
      probe paths). Skill-stratum, legacy probe, and math-RRV checks are
      untouched (they keep n_samples / any-pass semantics).
  D2. Retain rule when cap_retain_min > 0: for each probe, draw
      n = max(1, cap_samples or n_samples) regeneration samples; score each
      (math: extract+format match to the certified answer; code: that
      individual candidate passes ALL assert tests — evaluate every
      candidate, no short-circuit). rate = passes / n. Retain the probe
      iff rate >= cap_retain_min. ANY probe below theta -> broke_cap ->
      veto.
  D3. Guard interaction: the bounded-damage guard (cap_margin extra
      samples when per-phase cap-veto rate >= budget 0.5) keeps its
      trigger and ledger semantics. Under E1 the margin re-check is
      rate-consistent: pool the initial n with the cap_margin extra
      samples and retain iff POOLED rate >= theta (not any-pass on the
      extras). The guard therefore adds resolution, not an escape hatch.
  D4. Bit-identity guarantee: cap_retain_min <= 0 executes the EXACT
      current code path (any-pass, short-circuit, margin any-pass). All
      v5b/v6 control and determinism rows must reproduce bit-for-bit.
  D5. Engagement telemetry: when cap_retain_min > 0 the veto returns
      "cap_rates": {task_id: {"passes": int, "n": int}} for every checked
      probe (final, post-margin). env.py persists cap_rates and
      cap_retain_min on the gate record ONLY for theta>0 rows (gate
      record shape of theta=0 rows is unchanged). This lets the checker
      verify (a) the rule engaged, (b) the denominator is n, (c) the
      dose-response in measured rates.
  D6. Gold-freedom: the retain rule reads only probe regeneration against
      probe answers/tests (both manufactured at certification from the
      spec — no gold labels, no gold holdout, no reference answers). The
      gate's accept/reject decision consumes nothing gold. Unchanged.

CONFIG WIRING
  - ExperimentConfig: sccl_cap_retain_min: float = 0.0 (global default),
    sccl_cap_retain_mins: dict = {} (per-learner override, keyed by name),
    sccl_cap_samples: int = 0 (global; 0 -> sccl_replay_samples),
    sccl_cap_samples_map: dict = {} (per-learner override).
  - env.py resolves effective theta and n per learner (mirrors the
    cap_pool pattern), passes both to selfreplay_veto, logs cap_rates +
    theta + n on the gate record for theta>0 rows.

HYPOTHESES (v7 — pre-registered BEFORE the run)
  H1 (sensitivity): at least one strict dose lifts arith above the
     K=1 any-pass baseline: sccl_strict.arith > 0.375 OR
     sccl_majority.arith > 0.375. (Baseline 0.375 = v5b/v6 strat rows,
     reproduced in-ladder by row idx3.)
  H1b (dose ordering, directional): cap-veto counts must be monotone in
     strictness: strict >= majority >= baseline. If strict has FEWER
     cap-vetoes than baseline, the threshold never engaged its extra
     sensitivity -> H1 verdict is void, re-audit via cap_rates.
  H2 (width+sensitivity): sccl_ens_strict.arith >= sccl_strict.arith
     (ensemble width adds to sensitivity, contra v6 where width alone
     did not).
  H3 (composition): sccl_ens_strict_anchor.arith >= max(arith of rows
     4..6) AND its ACC >= max(ACC of rows 4..6).
  BREAKTHROUGH: sccl_ens_strict_anchor arith >= 0.55 AND
     frontier >= sccl frontier (0.5) - 0.02 -> multi-seed (43/44)
     confirmation BEFORE any headline claim.

LADDER (configs/sccl_v7.json -> runs/sccl_v7, seed = torch_seed 42 + index)
  CORRECTION (pre-run, before any v7 process launched): the first draft of
  this ladder placed sccl_capprobe_strat at idx2/seed 44, but the v6 ladder
  ran that row at idx3/seed 45 (v6 idx2 was sccl_capprobe). Bit-identity
  requires the identical seed, so the v6 prefix is preserved exactly and the
  E1 rows shift to idx4..idx7.
  idx0 frozen              seed 42  determinism (must bit-match v6 idx0)
  idx1 sccl                seed 43  determinism (must bit-match v6 idx1)
  idx2 sccl_capprobe       seed 44  K=1, theta=0, unstratified (must
                                    bit-match v6 idx2 — determinism row)
  idx3 sccl_capprobe_strat seed 45  K=1, theta=0, stratified (must
                                    bit-match v6 idx3 — H1 baseline row)
  idx4 sccl_strict         seed 46  K=1, theta=1.0, n=3 (E1 high dose)
  idx5 sccl_majority       seed 47  K=1, theta=2/3, n=3 (E1 low dose)
  idx6 sccl_ens_strict     seed 48  K=3, theta=1.0, n=3 (width+sensitivity)
  idx7 sccl_ens_strict_anchor seed 49  K=3, theta=1.0, n=3, anchor
                                    lambda=0.1 (full composition cell)
  All rows share the v5b/v6 stream (hash 554ce43f182b, seed 42) and
  sccl_replay_check=3 / sccl_replay_samples=2 for the skill stratum.
  E1 rows set sccl_cap_samples=3 so the dose is measured over 3 draws
  regardless of the skill-stratum sample count (no confound).

DEGENERACY GUARD (pre-registered)
  If a theta>0 row accepts < 3 updates across the whole stream, log it as
  DEGENERATE (veto stall, not mechanism success): it is excluded from
  H2/H3 max() comparisons and noted in the verdict. If the breakthrough
  cell (idx6) is degenerate, BREAKTHROUGH = FAIL with a mechanism note.

FAIL-CLOSED CHECKS (scripts/v7_check.py, auto-run by the watcher)
  C1 determinism: rows idx0..idx3 bit-match runs/sccl_v6 rows idx0..idx3
     on every report metric (frozen/sccl/sccl_capprobe/sccl_capprobe_strat).
     Any mismatch -> abort interpretation (engine/config regression).
  C2 engagement: every theta>0 row logs cap_rates on EVERY gate with
     checked_cap>0; at least 50% of those gates show cap_rates for >= 1
     probe. Missing cap_rates on a theta>0 gate -> abort (telemetry hole).
  C3 dose sanity: on strict rows, cap_rates denominators are all n=3
     (or n+cap_margin when the guard was armed — logged); theta field on
     gate records equals the configured theta. Mismatch -> abort.
  C4 isolation: frozen/sccl rows log checked_cap=0 and no cap_rates.
  C5 gold-freedom: AST audit of gcl/vault.py + gcl/env.py gate path —
     accept/reject consumes only probe regeneration results (no gold
     labels, no holdout scores); gold remains telemetry/eval-only.
     (Same audit as v6, extended to the new code path.)

EXPECTED-RANGE SANITY (pre-run calibration, not pass/fail)
  At temp 0.7 even a healthy probe has per-draw pass probability < 1 on
  hard tasks, so strict rows will veto more; the guard arms and the
  pooled-rate margin recheck is the designed relief valve. Expected dose
  ordering of accepted-update counts: baseline > majority > strict.
  Inversion of this ordering is itself a finding (record, don't abort).

COST/SEQUENCING
  8 rows x ~32 episodes; v6 wall-clock was ~7h for 7 rows on the 4060 Ti,
  so ~8h expected.
  Smoke first: configs/_smoke_v7.json (2 episodes, theta rows only) must
  show cap_rates present + bit-identical theta=0 trajectory vs v5b smoke.
  Then full ladder, watcher runs v7_check.py on finalization.

FALLBACK (pre-registered)
  E1 fully fails (H1 FAIL at both doses, no dose-response in cap_rates)
  -> E2 gold-free probe-gradient projection (A-GEM with the certified
  probe pool as constraint set), per rule R3. E1 partially succeeds
  (H1 PASS, H2/H3 FAIL) -> Branch F decision written after the verdict,
  pre-registered before the next run.

---

## BRANCH E (E1) — VERDICT (seed 42, 2026-08-28, runs/sccl_v7)

Run facts: 8-row ladder, wall-clock ~6.5h (12:06 -> 18:35) on the 4060 Ti.
All fail-closed checks PASSED (scripts/v7_check.py, runs/sccl_v7/verdict_check.md):
C1 determinism — the four prefix rows (frozen, sccl, sccl_capprobe,
sccl_capprobe_strat) bit-match runs/sccl_v6 on every report metric, frontier,
and family holdout score (verified incrementally during the run as well).
C2 engagement — every theta>0 gate with checked_cap>0 logged non-empty
cap_rates. C3 dose sanity — logged cap_retain_min == configured theta,
cap_n==3, denominators in {3,5}. C4 isolation — frozen/sccl show zero cap
activity; the theta=0 row logged legacy gate shape only. C4b anchor
engagement. C5 gold-free audit clean.

VERDICT: H1 FAIL | H1b FAIL | H2 PASS | H3 PASS (trivial) | BREAKTHROUGH FAIL.

| learner                  | ACC   | BWT    | Forget | Frontier | Upd | arith | math  | string | drift |
|--------------------------|-------|--------|--------|----------|-----|-------|-------|--------|-------|
| frozen                   | 0.613 | +0.132 | 0.025  | +0.588   |   0 | 0.600 | 0.250 | 0.800  | 0.800 |
| sccl                     | 0.575 | +0.095 | 0.075  | +0.500   |  17 | 0.400 | 0.500 | 0.800  | 0.600 |
| sccl_capprobe            | 0.625 | +0.145 | 0.131  | +0.494   |   9 | 0.375 | 0.750 | 0.600  | 0.775 |
| sccl_capprobe_strat      | 0.694 | +0.214 | 0.131  | +0.562   |  15 | 0.375 | 1.000 | 0.600  | 0.800 |
| sccl_strict (K=1 θ=1.0)  | 0.700 | +0.220 | 0.125  | +0.575   |  14 | 0.200 | 1.000 | 0.800  | 0.800 |
| sccl_majority (K=1 θ=⅔)  | 0.700 | +0.220 | 0.125  | +0.575   |   9 | 0.200 | 1.000 | 0.800  | 0.800 |
| sccl_ens_strict (K=3 θ=1.0) | 0.637 | +0.157 | 0.075 | +0.562  |  12 | 0.400 | 0.750 | 0.800  | 0.600 |
| sccl_ens_strict_anchor * | 0.738 | +0.257 | 0.025  | +0.713   |   1 | 0.600 | 0.750 | 0.800  | 0.800 |

\* degenerate per the pre-registered guard: 1 accepted update (<3) = veto
stall; excluded from the H2/H3 maxima; degenerate breakthrough cell forces
BREAKTHROUGH=FAIL even though its numbers clear the threshold (arith 0.600
>= 0.55, frontier +0.713 >= +0.48). The row "passed" by not learning.

Post-hoc gold telemetry (scripts/v7_telemetry.py, telemetry_v7.json) —
E1 RATE SENSITIVITY, the central mechanistic question:

| learner        | erosion rate | binary insens. | erosions w/ gate rate<1.0 | final arith |
|----------------|-------------:|---------------:|--------------------------:|------------:|
| sccl           | 23.5%        | -              | -                         | 0.400       |
| sccl_capprobe  | 11.1%        | 100%           | -                         | 0.375       |
| capprobe_strat | 26.7%        | 100%           | -                         | 0.375       |
| sccl_strict    | 28.6%        | 100%           | **0/4**                   | 0.200       |
| sccl_majority  | 22.2%        | 100%           | 1/2                       | 0.200       |
| sccl_ens_strict| 41.7%        | 100%           | **0/5**                   | 0.400       |

FINDINGS
  F5. RATE-LEVEL PROBE BLINDNESS. On every theta>0 row the E1 rule worked
      exactly as designed — every gate with pooled rate < theta was vetoed
      (strict 3, majority 5, ens_strict 4, anchor row 16) — yet every
      gold-erosion update that was probe-checked passed at pooled rate 1.0
      (0/4 on strict, 0/5 on the K=3 ensemble). The damage direction carries
      no signal in the probe pass-rate: erosion happens on capability axes
      the probes do not exercise. On sccl_strict the terminal damage occurred
      on the FIRST accepted update of the arith phase itself (v0: 0.600 ->
      0.400, gate rate 1.000) — the probe manufactured from the certified
      skill witnessed ITSELF perfectly while family-level capability eroded.
      This extends v5b/v6's F1 (binary insensitivity) to the rate level:
      witness blindness, not rule looseness, is the binding constraint.
  F6. STRICTNESS MONOTONICALLY DESTROYS PLASTICITY, ZERO EROSION GAIN.
      Accepted updates: baseline 15 -> strict 14 -> majority 9 ->
      ens_strict 12 -> ens_strict_anchor 1 (stall). Every unit of added gate
      strictness traded 1:1 against learning while erosion rates stayed
      22-42%. Tightening a blind filter filters learning, not damage. The
      dose ordering of cap-vetoes even INVERTED vs pre-registered H1b
      (strict 3 < majority 5 < baseline 9) because trajectory divergence
      dominates per-gate strictness.
  F7. THE NO-DAMAGE ENVELOPE IS THE BASE MODEL, AND IT IS REACHABLE. The
      degenerate anchor row (1 update) kept arith at frozen level (0.600)
      with the best ACC (0.738), best BWT (+0.257), best frontier (+0.713).
      All erosion is carried by ACCEPTED updates; a gate that blocked damage
      while keeping ~12 useful updates (ens_strict's learning volume) would
      land near that envelope. The headroom is real; the missing ingredient
      is a witness that sees the damage direction.
  F8. WITNESS WIDTH HELPS WITHIN THE CERTIFIED MANIFOLD (H2 PASS): K=3
      ensembles lifted arith 0.200 -> 0.400 vs K=1 at the same theta by
      vetoing cross-family updates that broke a witnessed axis (4 vetoes,
      incl. string/drift-phase updates). But 5 erosions still passed at
      rate 1.0 — width over the SAME axis family (variants of certified
      skills) does not reach the generalization axis the gold holdout
      measures.

MECHANISM CONCLUSION (E1)
  Thresholds over blind witnesses are provably inert. The v5b->v6->v7 arc
  has now eliminated three candidate explanations for persistent arith
  erosion: (1) rule looseness (any-of-n) — eliminated, rate rules engage
  and veto correctly; (2) witness count — eliminated, K=3 ensembles catch
  cross-family breaks but not erosions; (3) threshold height — eliminated,
  theta=1.0 changes nothing because erosion gates pass at full rate. What
  remains: the probes test REPRODUCTION of certified skills (numeric/
  paraphrase variants of tasks the model already solved) while erosion hits
  GENERALIZATION to new instances of the family — the axis only the gold
  holdout measures. Probes manufactured from certified skills live on the
  memorized manifold; the damaged capability lives off it.

FALLBACK DECISION (with pre-registered deviation, recorded per rule R4)
  E1 fully failed (H1 FAIL at both doses, H1b FAIL), so the pre-registered
  fallback nominally routes to E2 (A-GEM-style probe-gradient projection on
  the certified probe pool). The v7 telemetry FALSIFIES E2's premise before
  it runs: a gradient constraint set sees the damage direction only if the
  probe LOSS sees it, and F5 shows the probe outcomes carry no damage signal
  (erosions pass at rate 1.0 across K=1 and K=3 pools) — projecting away
  the probe-loss-conflicting component of the update gradient leaves the
  invisible damage direction untouched. Running E2 on the current witness
  would spend ~8h of GPU to confirm a premise the artifacts already refute.
  DEVIATION: instead of E2-as-designed, Branch F attacks the binding
  constraint directly — witness COVERAGE — by manufacturing probes from
  UNTRAINED tasks (generalization witnesses) rather than certified skills
  (reproduction witnesses), reusing the validated E1 rate machinery. E2 is
  retained as a later composition candidate (gradient projection over a
  witness that CAN see the damage may yet help). This deviation and its
  evidence base are recorded here BEFORE any Branch F implementation, per
  the project's pre-registration discipline.

---

BRANCH F PRE-REGISTRATION — G1 GENERALIZATION WITNESSES (v8 ladder)
(2026-08-28, BEFORE implementation; config configs/sccl_v8.json)

MECHANISM DESIGN
  Certified-skill cap probes witness REPRODUCTION; gold erosion lives on the
  GENERALIZATION axis (F5/F8: every erosion passed at pooled rate 1.0, even
  K=3 ensembles; terminal damage hit v0, the first accepted update). G1
  manufactures probes from UNTRAINED future stream tasks — the model must
  solve a NEW instance of the family at probe-commit time, so the probe
  lives off the memorized manifold and witnesses the same axis the gold
  holdout measures. When an update breaks that witness, the gate sees it.
  Probes are manufactured with the EXISTING gold-free selfcert path
  (spec-only test generation; selfcert.py passed the v5b C5 AST audit), so
  the learning loop stays gold-free.

  Manufacture protocol (per row, per family f):
  1. AT FIRST CONTACT with family f (first episode of that family, BEFORE
     any update on f — this covers the v0 damage window F5 found on
     sccl_strict): manufacture G=1 generalization probe from the NEXT
     UNTRAINED task of f: candidate solution from the current model +
     selfcert tests from the task SPEC (no gold fields), verified (must
     pass consensus tests) and committed kind=cap_probe with task_id
     "<task_id>:g" (the ":g" suffix distinguishes generalization witnesses
     from ":c" certified variants; no ":c" substring, so existing pool
     dedup treats each as a distinct source).
     If the candidate fails selfcert verification, try the next untrained
     task of the family; if none can be certified, the family has no gen
     witness (logged; telemetry counts coverage).
  2. RETIREMENT: when task T (source of a live gen probe) becomes the
     current training task, its gen probe is REMOVED from the cap pool
     before training on T — a probe from an already-trained task would
     degrade to a reproduction witness and contaminate the coverage claim.
  3. REFRESH: after T is attempted, manufacture a replacement gen probe
     from the next untrained task of f, maintaining G=1 per family while
     untrained tasks remain. When f's training tasks are exhausted, its
     certified-skill probes continue to cover reproduction (cross-family
     generalization is still witnessed by the other families' live gen
     probes).
  4. POOL SLOTS: gen probes occupy DEDICATED slots, never evicted by
     certified probes; certified probes keep their existing K rule. A
     checked gate therefore regenerates up to K certified + G gen probes
     per family.

  Reused machinery (no redesign): veto strata, pooled-rate margin recheck
  (cap_margin=2, denominators {3,5}), theta rule (cap_retain_min /
  cap_samples), anchor penalty, degenerate-row guard, all E1 wiring.

HYPOTHESES (registered thresholds)
  Ladder (7 rows; idx0-3 are the v5b/v6/v7 determinism prefix, bit-identical
  expected, seeds 42+idx):
    0 frozen            42
    1 sccl              43
    2 sccl_capprobe     44
    3 sccl_capprobe_strat            45  (reproducible capability baseline)
    4 sccl_genprobe                  46  (strat + G=1 gen witnesses, legacy
                                          any-pass rule on the joint pool)
    5 sccl_genprobe_strict           47  (+ theta=1.0, n=3 pooled-rate rule)
    6 sccl_genprobe_strict_anchor    48  (+ anchor lambda=0.1) — breakthrough cell

  H1 (coverage main effect):  genprobe.arith > 0.375 (beats the
      reproducible v5b/v6/v7 capability baseline under the SAME any-pass
      rule; the only change is witness coverage).
  H2 (coverage + sensitivity): genprobe_strict.arith >= max(0.375,
      genprobe.arith) AND accepted updates >= 5 (anti-stall guard from F6:
      strictness over a SEEING witness must not freeze learning; rows with
      <5 updates are degenerate and excluded from maxima, as before with <3).
  H3 (composition): genprobe_strict_anchor.arith >= max(gen rows' arith)
      AND ACC >= max(gen rows' ACC), degenerate rows excluded.
  M1 (mechanism, telemetry-recorded, not gated): gen-witness SENSITIVITY —
      fraction of gold-erosion accepted updates (probe-checked) whose gate
      log shows at least one ":g" probe failing or pooled rate < theta.
      v7 baseline: 0% (0/4 strict, 0/5 ensemble). Target > 50%. M1 is the
      mechanism success criterion: even if arith misses 0.55, sensitivity
      >50% confirms coverage is the binding constraint and routes to dose
      scaling (G=2, refresh timing) in v9.
  BREAKTHROUGH (unchanged axis, anti-stall added): genprobe_strict_anchor
      arith >= 0.55 AND frontier >= sccl frontier - 0.02 AND accepted
      updates >= 5 -> multi-seed (seeds 43/44 via scripts/v8_multiseed.py)
      before any headline claim. Degenerate breakthrough cell forces FAIL.

FAIL-CLOSED CHECKS (v8_check.py; any failure -> verdict FAIL regardless)
  C1 determinism: idx0-3 bit-match runs/sccl_v7 idx0-3 on every report
     metric, frontier, and family holdout score.
  C2 engagement: each gen row manufactures >= 1 gen probe in a family
     before that family's first update; checked gates in gen rows have
     cap_rates covering at least one ":g" task_id (when a gen probe is
     live); logged cap_retain_min/cap_samples == config on theta>0 rows.
  C3 retirement/anti-contamination: no gen probe from task T is checked at
     a gate during or after T's own training episode (retired before use);
     every ":g" task_id is a stream TRAINING task (disjoint from the gold
     holdout task ids recorded in metrics eval_detail.final_heldout) —
     a holdout-sourced probe is gold leakage and fails the run.
  C4 isolation: frozen/sccl/capprobe/capprobe_strat rows show ZERO gen
     activity (no ":g" probes, no manufacture events); theta=0 rows log no
     E1 fields; anchor row logs anchor_pen > 0 on its gates.
  C5 gold-freedom: AST audit of gcl/selfcert.py + gcl/vault.py + gcl/env.py
     over the manufacture AND gate paths — probe manufacture consumes only
     prompt/spec fields; accept/reject consumes only probe regeneration
     results; gold labels, reference answers, and holdout scores appear in
     no decision path. (v5b/v6/v7 audit extended to the new code path.)

EXPECTED-RANGE SANITY (pre-run calibration, not pass/fail)
  Gen probes are HARDER than certified variants (untrained tasks, temp 0.7
  regeneration), so even a healthy model shows rate < 1 on some gen probes;
  strict rows will veto more than v7 strict did, and the margin recheck
  (denominators {3,5}) is the relief valve. If genprobe_strict accepts < 5
  updates the witness is too harsh at theta=1.0 — that is a dose finding
  (record; v9 lowers theta or G), not an abort. Expected accepted-update
  ordering: genprobe >= genprobe_strict >= genprobe_strict_anchor.
  Inversion is itself a finding (record, don't abort).

COST/SEQUENCING
  7 rows x ~32 episodes; v7's 8 rows took ~6.5h, so ~6h expected (the extra
  gate checks are offset by one fewer row). Manufacture adds 4 families x
  ~8 refreshes of selfcert generation per gen row — bounded, GPU-cheap.
  Smoke first: configs/_smoke_v8.json (2 episodes, gen rows only) must show
  ":g" probes manufactured at first contact + gate records containing ":g"
  task_ids, with no holdout id anywhere in gate artifacts. Then full
  ladder, watcher runs v8_check.py on finalization.

FALLBACK (pre-registered)
  G1 fully fails (H1 FAIL and M1 sensitivity <= 50%) -> the erosion is not
  witnessable by selfcert-quality tests on untrained tasks; the remaining
  explanation is that SELF-CERTIFIED TESTS are too weak to proxy gold tests
  on the generalization axis. Route: Branch G = stronger gold-free test
  generation (multi-sample consensus / property-based invariants / oracle-
  free consistency checks across paraphrase clusters), decided and pre-
  registered after the verdict. G1 partially succeeds (M1 > 50% but H1/H2
  FAIL) -> dose scaling in v9 (G=2, theta in {0.5, 2/3} on the joint pool,
  earlier refresh), per rule R3, plus E2 composition as a candidate row.
  H1 PASS but BREAKTHROUGH FAIL -> v9 composes G1 with E2 (projection over
  the now-seeing witness) and/or anchor tuning; pre-registered before run.

V8 FINAL VERDICT (2026-08-29, runs/sccl_v8, verdict_check.md, rc=0):
  H1 FAIL | H2 FAIL | H3 PASS | BREAKTHROUGH: PASS* | C1-C5 ALL PASS.
  (*PASS on the seed-42 ladder rule; multi-seed 43/44 REQUIRED before any
  headline claim — launched, see V8 MULTI-SEED below.)
  frozen .613/.588 arith .600 | sccl .575/+.500 arith .400 |
  capprobe .625/.494 arith .375 | strat .694/.562 arith .375 |
  genprobe .575/.450 arith .200 | genprobe_strict .637/+.450 arith .175
  | genprobe_strict_anchor .675/+.650 arith .600 updates 14 forget .025.
  BREAKTHROUGH rule: arith .600 >= .55 AND frontier +.650 >= sccl+.500-.02
  AND updates 14 >= 5 -> PASS. H3: composition dominates both gen rows on
  arith (.600 vs .200/.175) AND ACC (.675 vs .575/.6375), not degenerate.
  C1: all 4 prefix rows BIT-IDENTICAL to runs/sccl_v7. C2: every gen row
  manufactured witnesses at all 4 first contacts, checked ':g' probes in
  gates (3 gates each), theta rows log cap_rates with denominators {3,5}.
  C3: zero retirement violations, zero holdout-sourced probes. C4/C4b:
  zero gen activity on prefix rows, anchor_pen>0 on all 14 accepted
  anchor updates, lambda exactly {0.1}.
C5 AUDIT FIX (recorded for honesty, pre-decision): the first checker run
  aborted on a blanket vault.py name ban that flagged _Skill.test_code —
  the vault's own slot storing each skill's SELF-CERTIFIED tests (written
  from certifier self_tests, spec-only provenance), a name collision with
  the dataset Task.test_code gold, not a leak. Diagnosed by code reading
  BEFORE any decision rule was evaluated. Replaced with a STRONGER
  3-layer fail-closed audit: (1) every SCCL decision method must take no
  task object and read no task gold attr; (2) exhaustive enumeration that
  the ONLY task-gold readers in vault.py are the legacy VSR methods
  (commit/violates/choose_target/_too_similar_exists) so any new reader
  fails; (3) run-data proof the legacy paths never executed (zero
  vsr-gate decisions, zero gold target_source across all 7 rows).
  All three layers pass; the gold-free guarantee holds.
MECHANISM (from telemetry_v8.json, measurement only):
  M1 = 0% on every gen row (0/3): the ':g' witnesses did NOT fire on the
  gold-erosion gates — v7's F5 insensitivity persists at the rate level.
  The erosions on the anchor row are TRANSIENT, not ratcheting: arith
  oscillates .600->.400->.600 across the stream (v0 -.200, v2 +.200, v6
  -.200, v9 +.200, v11 -.200, v13 +.200) and ENDS at frozen level, which
  is why forgetting is .025 — the lowest of ALL rows including sccl.
  The composition works through UPDATE SELECTION, not erosion DETECTION:
  the strict witness gate (5 cap-vetoes, 1 by a ':g' witness) rejects
  updates that would leave the certified+generalization basin, and the
  anchor bounds each accepted update's drift (anchor_pen mean 1.30), so
  the model never takes the unrecovered ratchet step that killed arith in
  the genprobe (.200) and genprobe_strict (.175) rows. Post-hoc
  interpretation, not pre-registered: the operative pairing is
  sensitivity-via-composition, i.e. theta-strict witnesses + anchor keep
  every accepted update inside the witnessed region, rather than catching
  erosions at the gate.
  Witness quality: ':g' probes pass at pooled rate 1.0 when checked on
  kept updates; gen vetoes exist (1 per strict row) but on non-erosion
  gates. H1's failure mode: coverage alone (theta=0 any-pass) accepted
  every update exactly like sccl did, PLUS the raw genprobe pool
  displaced nothing — its arith .200 shows witness coverage without rate
  sensitivity does not protect; H2 shows sensitivity without the anchor
  stalls into the v7-F6 pattern instead (arith .175 despite 9 updates).
V8 MULTI-SEED (pre-registration: BREAKTHROUGH PASS requires seeds 43/44):
  launched scripts/v8_multiseed.py 2026-08-29 (snapshot s42 + two full
  ladders at torch_seed 43/44 + 3-seed aggregate). Verdict PENDING; the
  headline is claimable only if the paired per-seed rule (arith >= .55
  AND frontier >= sccl-.02 AND updates >= 5) holds on all three seeds.

V8 MULTI-SEED FINAL VERDICT (2026-08-30, runs/sccl_v8_s{42,43,44},
  aggregate in runs/sccl_v8_seeds/):
  PAIRED BREAKTHROUGH RULE: met on 2/3 seeds -> NOT CONFIRMED at the
  pre-registered bar (the rule demands all three; s43 fails it).
  Per-seed arith (breakthrough cell vs sccl):
    s42 anchor .600 vs sccl .400 | s43 anchor .400 vs sccl .400 (updates 4)
    | s44 anchor .600 vs sccl .200 (updates 6).
  Mean arith: anchor .533+/- .115 vs sccl .333 +/- .115 — paired advantage
  +.200/+.000/+.400 (mean +.200, never negative; every anchor seed >= sccl
  max). Mean frontier: anchor +.625 +/- .022 (the tightest row in the
  table) vs sccl +.504 +/- .106 — anchor >= sccl on every seed.
  HONEST HEADLINE: the witness+anchor composition DOMINATES the gold-free
  baseline at 3 seeds (arith mean +0.200, frontier +0.121, both
  seed-consistent), but the strict breakthrough bar (arith >= .55 AND
  frontier >= sccl-.02 AND updates >= 5, paired per seed) failed on s43
  where the strict gate stalled plasticity: 4 accepted updates (12
  rollbacks). Updates across seeds 14/4/6 — the dose is plasticity-
  fragile, echoing v7-F6.
  Mechanism cross-seed (telemetry_v8.json per seed): M1 = 0% at every
  seed (0/3, 0/1, plus s44) — gen witnesses never fire on gold-erosion
  gates. On s44 the no-anchor strict row accepted 2 updates and keeps
  frozen-level arith .600 — the v7-F7 "no-damage envelope" again; its
  apparent arith win there is the stall, not protection.
  CONCLUSION: sensitivity-via-composition (strict witnesses + anchor)
  is seed-robust as a DOMINANCE result, not yet a >=.55-everywhere
  breakthrough. The binding constraint is the DOSE (theta=1.0, G=1, n=3
  vetoes too much under seed variance).
FALLBACK TRIGGERED (pre-registered): M1 < 50% and breakthrough unconfirmed
  -> v9 dose scaling: theta in {0.5, 2/3} on the joint pool, G=2 with lane
  rotation, KEEP the anchor (the only composition that dominates), and a
  mandatory 3-seed protocol from the start (no seed-42 gate before
  multi-seed — the v8 lesson). Pre-registered below as BRANCH G (V9).

================================================================================
BRANCH G PRE-REGISTRATION (V9 DOSE LADDER) — written 2026-08-30, BEFORE any
v9 implementation or run. Triggered by the pre-registered v8 fallback (M1 < 50%
and breakthrough unconfirmed at 3 seeds). Design is fixed on v8's ALREADY
COLLECTED evidence; nothing below may be changed after the first v9 row starts.
================================================================================

EVIDENCE BASE (v8, 3 seeds, all recorded):
  - The composition (gen witnesses G=1 + theta=1.0 n=3 + anchor 0.1) DOMINATES
    sccl at every seed on arith (paired +.200/+.000/+.400) and frontier (mean
    +.625 +/- .022) but is plasticity-fragile: accepted updates 14/4/6 — on
    s43/s44 the theta=1.0 dose vetoes nearly everything (the v7-F6 trade under
    seed variance).
  - M1 = 0% at every seed: the gate never CATCHES an erosion; protection is
    update selection (accepted updates stay in the certified+witnessed basin).
  - Therefore the breakthrough bar (arith >= .55 paired per seed) fails not
    for lack of protection but for lack of ACCEPTED UPDATES at the strict dose.

V9 QUESTION (single-variable dose scaling on the v8 composition):
  does a SOFTER rate threshold over the SAME composition restore plasticity
  (>= 8 updates/seed) while KEEPING the dominance (arith >= sccl paired,
  every seed) and reaching the breakthrough bar at all 3 seeds?

TWO TREATMENT ROWS (only the dose theta varies; everything else fixed):
  sccl_gen2_half      : G=2 gen lane, theta=0.5,  n=3 margin 2, stratified,
                        cap probes on, anchor lambda=0.1.
  sccl_gen2_majority  : G=2 gen lane, theta=2/3,  n=3 margin 2, stratified,
                        cap probes on, anchor lambda=0.1.
  G=2 = TWO concurrent generalization witnesses per family (dedicated ':g'
  lane holds the 2 newest untrained-source witnesses; retired before their
  source trains; refreshed from the next untrained task; both may enter the
  per-gate probe draw). Rationale: v8's single witness per family covered one
  point of the generalization axis; two concurrent witnesses double the
  covered subspace at the SAME certification cost per family, and the soft
  threshold keeps their veto from stalling plasticity.
  theta=0.5: pooled rate >= 1.5/3 (>=2 of 3 draws). theta=2/3: pooled >= 2/3
  (>=2 of 3 draws, but stricter after the n=5 margin recheck: >=3 vs >=4 of 5).

PREFIX CONTROL ROWS (C1 determinism): frozen, sccl, sccl_capprobe,
  sccl_capprobe_strat — bit-identical to runs/sccl_v8 prefix rows at seed 42
  and to runs/sccl_v8_s{43,44} prefix rows at seeds 43/44 (same stream hash,
  same per-learner seeds; verified per seed by scripts/v9_check.py).

PROTOCOL (methodological lesson of v8 — the seed-42 gate is NOT sufficient):
  1. wiring smoke (configs/_smoke_v9.json, 2 families, both rows + control):
     fail-closed engagement audit (first-contact manufacture of TWO ':g'
     probes per family, both lanes in gates, retirement/anti-contamination
     walk, dose fields theta/cap_n/denominators, anchor engagement, control
     isolation, holdout disjointness).
  2. THE 3-SEED LADDER IS THE EXPERIMENT: run seeds 42,43,44 sequentially on
     configs/sccl_v9.json (scripts/run_seeds.py). No interim decision, no
     early stop, no seed-42-only gate. Verdict computed once, after all
     three seeds finalize, from runs/sccl_v9{,_s43,_s44}/metrics.json.

DECISION RULES (pre-registered; computed per seed s in {42,43,44}):
  H1 (plasticity restored): every treatment row accepts >= 8 updates at seed
     s. (v8 theta=1.0 accepted 14/4/6; the soft dose must clear 8 at every
     seed or the dose is still too harsh.)
  H2 (dominance kept): for every treatment row, arith >= sccl.arith at seed
     s (paired; the v8 composition held this at every seed).
  H3 (breakthrough): SOME treatment row has arith >= 0.55 AND frontier >=
     sccl.frontier - 0.02 AND updates >= 5 at seed s.
  V9 VERDICT:
    BREAKTHROUGH CONFIRMED iff H3 holds at ALL 3 seeds (the v8 bar, now
      evaluated up front at all seeds).
    DOMINANCE CONFIRMED iff H2 holds at all 3 seeds (replicates v8's honest
      positive).
    If H1 FAILS at any seed (a row accepts < 8): dose still too harsh ->
      record as dose finding; theta ladder exhausted -> next branch moves
      the strictness off the RATE axis entirely (witness-free drift bounds,
      e.g. certified-manifold trust region; pre-register before running).
    If H1 PASSES but H3 FAILS: the theta axis is exhausted too -> same next
      branch. A theta between 0.5 and 1.0 will NOT be run (protocol stops at
      the pre-registered dose pair; chasing the interpolated theta would be
      garden-of-forking-paths).
  M1 telemetry (measurement only, not gated) continues per row per seed.

DEGENERACY GUARD: a treatment row with < 5 accepted updates at any seed is
  DEGENERATE at that seed (excluded from H3 maxima; forces H3 FAIL there).

FAIL-CLOSED CHECKS (per seed; scripts/v9_check.py, fail -> abort, no verdict):
  C1 determinism: the 4 prefix rows bit-identical to the matching v8 run
     (seed 42 vs runs/sccl_v8; 43/44 vs runs/sccl_v8_s{43,44}) on report
     keys + frontier + family holdout scores.
  C2 engagement: first-contact markers >= 1 per row; ':g' ids in
     checked_cap_ids on >= 1 gate; with G=2 the vault's gen lane holds 2
     ':g' probes per family at run end UNLESS the family exhausted untrained
     sources (logged); theta rows log cap_rates on every checked-cap gate
     with cap_retain_min == {0.5 | 2/3}, cap_n == 3, denominators in {3,5};
     sccl_stats gen counters committed >= 4 (2 per family) per row.
  C3 retirement/anti-contamination: no ':g' from task T checked during or
     after T's own training episode; every ':g' source is a stream TRAIN task
     disjoint from eval holdout ids (metrics eval_detail.final_heldout).
  C4 isolation: prefix rows show zero gen/E1/anchor activity; treatment
     rows log anchor_pen > 0 on every accepted update with lambda == {0.1}.
  C5 gold-free: the v8 three-layer audit (selfcert.py global; _gen_make vs
     LABEL_FIELDS + entry_point under sccl_derive_entry=true; vault SCCL
     methods Task-free + Task-gold readers confined to legacy VSR set +
     zero vsr-method/gold-target records in run data).

COST: 6 rows x 3 seeds; v8 measured ~45 min/row -> ~13.5h total, sequential.
  runs/sccl_v9{,_s43,_s44} + aggregate runs/sccl_v9_seeds. NEVER committed.
