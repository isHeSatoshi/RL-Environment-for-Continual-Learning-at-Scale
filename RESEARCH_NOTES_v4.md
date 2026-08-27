# SCCL v4 — In-Update Base Anchoring (research notes)

Date: 2026-08-27 (authored while multi-seed v3 seeds 43/44 run in background).

## Motivation: gates are insufficient for capability preservation

From the seeded v3 ladder (RESEARCH_NOTES_v3.md, seed-42): EVERY learning row — gold-free
SCCL variants AND the gold-verified reference vsr_nogold — collapses the arithmetic HOLDOUT
(a never-trained generalization probe) to 0.38–0.40 while frozen holds 0.60. The RRV gate
checks CERTIFIED vault skills; the base model's UNCERTIFIED general capability is never in
the vault, so no accept/reject verdict defends it. The damage is inflicted INSIDE an accepted
update, which no gate decision can prevent. => attack the update mechanism, not the gate.

## Mechanism (gold-free)

During each LoRA update, add a quadratic pull of the trainable (LoRA) parameters toward their
value at LoRA initialization — exactly the frozen base model (disable_adapter state):
    loss += 0.5 * lambda * sum((p - base[n])^2)   over requires-grad LoRA params
- Gold-free: the anchor target is the model's own initialization; no labels anywhere.
- Decision-independent: it changes HOW an accepted update moves weights, never WHICH updates
  are accepted. The RRV safety semantics are untouched (unit-tested: anchor strength does not
  flip accept/reject).
- Per-learner (`sccl_anchor_learners`) so it can be cleanly A/B'd in the ladder; per-learner
  strength override (`sccl_anchor_lambdas`) enables a lambda ablation in one run.
- Audit trail: `anchor_lambda` is logged in the accepted-update return (update_info), so a
  real run's trajectory can be verified gold-free post-hoc.

## Ladder (configs/sccl_v4.json, seeded torch_seed=42, same stream hash 554ce43f182b)

frozen (control) | sccl (v1 ref) | sccl_anchor_lo (λ=0.1) | sccl_anchor_hi (λ=0.5) | vsr_nogold (gold ref)
Anchor learners = v1 + ONLY the anchor (no replay/probe/nbhd) to isolate the mechanism.
vsr_nogold sits at learner index 4 with torch_seed 42, identical to the v3 run -> its row
should reproduce v3's (determinism cross-check).

## Pre-registered decision rule (avoid post-hoc rationalization)

- Hypothesis: a moderate anchor preserves the base model's UNCERTIFIED arith-holdout
  generalization while NOT paying a prohibitive plasticity price.
- Success (within-run, seed-42): sccl_anchor_*.arith_holdout > sccl.arith_holdout
  (anchor preserves generalization) AND sccl_anchor_*.frontier >= sccl.frontier - 0.02
  (no large plasticity cost). Report the lambda that best trades off the two.
- If BOTH anchor rows preserve arith-holdout ~ frozen (0.60) but collapse frontier,
  the anchor is too strong / uniform anchoring is the wrong mechanism -> next step is
  targeted anchoring (EWC-style Fisher weighting on self-certified pairs, or gradient
  projection into the null-space of important base directions), still gold-free.
- If anchor rows do NOT lift arith-holdout above sccl, the erosion is not fixable by a
  uniform L2 pull and the diagnosis must be revisited.
- Any single-seed ordering is provisional; the within-run paired comparison (same stream,
  same seed) is the primary signal. Multi-seed mean±std required before a headline claim.

## Status

- Implemented + unit tested (4 anchor tests + config-vs-registry consistency test; full
  suite 61 passed). Committed daa9fde (mechanism) + 31cfe38 (audit trail + smoke config).
- BUG FOUND & FIXED (commit 7a8a779): sccl_anchor_lo/hi were NOT in the LEARNERS registry,
  so experiment.py silently skipped them (first smoke ran 1/2 learners). Root cause: unit
  tests exercise the env, never the registry. Fix: SCCLAnchorLo/HiLearner registered;
  regression test asserts every learner named in any config/*.json exists in LEARNERS.
  Also: anchor_lambda now persisted into the gate dict (registry.json audit trail).
- Real-engine smoke (configs/_smoke_v4.json) VERIFIED: sccl gate entries anchor_lambda=0.0,
  sccl_anchor_lo entries anchor_lambda=0.1 — engagement + isolation confirmed on GPU.
- Seeded v4 ladder LAUNCHED 2026-08-27 ~14:55 (runs/sccl_v4, torch_seed=42). Determinism
  cross-checks built in: sccl at idx1 (seed 43) should reproduce v3 seed-42's sccl row;
  vsr_nogold at idx4 (seed 46) should reproduce v3's. Frozen already bit-identical
  (0.613/0.025/+0.5875).
- Watcher: scripts/wait_and_analyze_v4.py — waits for completion, then evaluates the
  pre-registered rule and writes runs/sccl_v4_verdict.md.
