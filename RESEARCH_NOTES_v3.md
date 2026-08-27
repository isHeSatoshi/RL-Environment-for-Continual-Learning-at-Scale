# SCCL v3 — research notes (living document)

Seeded v3 ladder: `runs/sccl_v3` (configs/sccl_v3.json, torch_seed=42, stream_hash 554ce43f182b clean).
Learners: frozen, sccl (v1 ref), sccl_n (v1 + admission filter), sccl_promote (v2 + probe curriculum), vsr_nogold.
Frozen control bit-identical across all three runs: ACC 0.6125 / forget 0.025 / frontier +0.588.

## CASE STUDY: v3 filter catches a conf-1.0 gold-wrong target (task mbpp_615, arith)

Same task, same seeded stream, two learners:

| learner | cert conf | nbhd verdict | gold pass_rate | trained? |
|---------|-----------|--------------|----------------|----------|
| sccl (v1 ref) | 1.0 | — (no filter) | **0.0** | **YES (committed wrong target)** |
| sccl_n (v3)   | 1.0 | REJECT `winner_fails_variant` | 0.0 | no |

Point-certification was maximally confident (1.0) in a target that was gold-wrong
(0.0 pass rate). v1 trained on it and committed it to the vault. v3's admission-time
neighborhood check (paraphrase + fresh self-tests) caught it BEFORE any gradient was
spent. This is the mechanism's value in a single concrete instance: it filters the
confidently-wrong, instance-narrow solutions that point-cert alone lets through.

Gold here is measurement-only (telemetry); it played no role in the admission decision.

## Pre-registered decision rule for v3 claims (avoid post-hoc rationalization)

- Hypothesis: sccl_n has LOWER forgetting than sccl and does NOT pay v2's plasticity
  price (sccl_v2 dropped updates 12 vs 19 and ACC to 0.688).
- Success on seed-42: sccl_n.forgetting <= sccl.forgetting AND sccl_n.frontier >= sccl.frontier.
- Any single-seed ordering is provisional; the multi-seed finalist run (scripts/run_seeds.py
  --seeds 42,43,44) establishes it with mean±std. Do NOT assert mechanism superiority from
  seed-42 alone.

## Audit tooling

- `python scripts/analyze_nbhd.py --run runs/sccl_v3 --learner sccl_n --baseline sccl`
  Reports over-filter rate (rejected & gold-correct), correct-skepticism (rejected &
  gold-wrong), admitted-gold-wrong. Handles code (pass_rate) and math (match) gold.
- `python -m gcl.report --aggregate <dir>_seeds --out paper/results_v3.tex --prefix vthree`
  Emits mean±std macros from run_seeds aggregate.json.

## SEED-42 FULL-RUN RESULTS (honest negative/mixed for the v3 hypothesis)

| learner | ACC | forget | frontier | upd/rbk |
|---------|-----|--------|----------|---------|
| frozen | 0.613 | 0.025 | +0.588 | 0/0 |
| sccl (v1) | 0.575 | 0.075 | +0.500 | 17/3 |
| sccl_n (v3) | 0.631 | 0.131 | +0.500 | 9/0 |

Pre-registered success criterion NOT met: sccl_n.forgetting 0.131 > sccl 0.075, and
frontier ties at +0.500 (both below frozen +0.588). Per-family (final_heldout):
arith sccl_n 0.38 ≈ sccl 0.40 (frozen 0.60) — the filter did NOT protect the target
generalization metric; math 0.75 (vs sccl 0.50) better; string 0.60 (vs sccl 0.80) worse;
drift 0.80 (vs sccl 0.60) better. Higher ACC came from math+drift luck, not arith.

Admission audit (analyze_nbhd.py), FINAL: 16/32 checked, 7 rejected.
  admitted n=9  mean gold 0.667 (gold-perfect 6/9); admitted-gold-wrong 3/9
  rejected n=7  mean gold 0.571 (gold-perfect 4/7)
  over-filter (rejected & gold==1.0): 4/7; correct-skepticism (rejected & gold<1.0): 3/7
  baseline sccl certified pool mean gold 0.617 -> filter raises admitted quality to 0.667.
Cert rate collapsed: sccl_n 0.28 (9/32) vs sccl 0.62 (20/32) — the filter halves plasticity.

## COMPLETE SEED-42 LEADERBOARD (all 5 learners, canary clean, hash 554ce43f182b)

| learner | ACC | forget | frontier | upd/rbk | arith-hold |
|---------|-----|--------|----------|---------|-----------|
| frozen | 0.613 | 0.025 | **+0.588** | 0/0 | 0.60 |
| vsr_nogold (gold) | **0.637** | 0.125 | +0.512 | 12/4 | 0.40 |
| sccl_n (v3) | 0.631 | 0.131 | +0.500 | 9/0 | 0.38 |
| sccl_promote | 0.588 | 0.125 | +0.463 | 12/6 | 0.40 |
| sccl (v1) | 0.575 | 0.075 | +0.500 | 17/3 | 0.40 |

## CORE UNSOLVED PROBLEM (most important seed-42 signal)

EVERY learning row — gold-free AND gold-verified vsr_nogold — collapses arith holdout to
0.38-0.40 while frozen holds 0.60. Any weight update on this stream erodes arithmetic
generalization regardless of gate (gold or gold-free). vsr_nogold (0.40) is NO better than
gold-free sccl (0.40) on the target metric this seed. The bottleneck is not the gate; it is
that continual LoRA fine-tuning on narrow/drifted tasks overwrites the base model's arith
generalization. NOTE this is seed-42 only: in the v2 rerun vsr_nogold held arith at 0.68,
so the gold-advantage on arith is itself seed-dependent. Multi-seed decides.

## VARIANCE FINDING (reinforces seeded protocol)

vsr_nogold forgetting: 0.004 (v2 rerun, unseeded) -> 0.125 (v3 seed-42, seeded). Even the
gold-verified reference varies enormously run-to-run. Single-run "best row" claims are
unsafe; this is precisely why the headline uses multi-seed mean±std.

## METHOD REFINEMENT: the "over-filter" proxy is ambiguous

`over-filter rate = rejected & gold==1.0` conflates two cases gold CANNOT separate,
because gold pass_rate only measures the ORIGINAL spec:
  (a) the winner is genuinely instance-narrow — passes original gold but fails any
      rephrasing. Rejecting it is CORRECT (this is the filter's entire purpose).
  (b) the paraphrase drifted in meaning, or a fresh self-test is buggy/over-specific,
      so a genuinely-general winner fails the variant. Rejecting it is an ERROR.
So "4/7 gold-correct rejected" is an UPPER BOUND on true over-filtering, not a measured
error rate. Some may be legitimate instance-narrow catches. This must be stated honestly
in the paper rather than reported as a filter defect.

Seed-42 robust reading: the admission filter halves plasticity (9 vs 17 updates) and on
this seed does NOT convert that conservatism into arith-holdout protection. Whether that
is seed noise or a real cost is exactly what the multi-seed run decides. Do not redesign
the filter on one seed.

## Open questions

- Multi-seed (43,44): is sccl_n's seed-42 pattern (higher forgetting, no arith gain,
  halved updates) robust, or inside the run-to-run variance seen for unseeded sccl?
  → ANSWERED below: it is ROBUST.
- If robust: the admission filter as specified is too aggressive — candidate fixes are a
  margin-based rejection (reject only on majority variant-test failure) or a self-
  consistency check on the variant tests themselves; but do NOT pick a fix until
  multi-seed confirms the cost is real.
  → SUPERSEDED: multi-seed shows the cost is real, but "fixing the filter" is the wrong
  frame — forgetting here is not dominated by target quality at all (see verdict). The
  right response is acting on the update itself (v4, RESEARCH_NOTES_v4.md).

## MULTI-SEED VERDICT (seeds 42/43/44) — v3 hypothesis NOT supported, per pre-registered rule

Aggregate: runs/sccl_v3_seeds/aggregate.json (rails passed: single stream hash
554ce43f182b across seeds, all canaries clean). Paper macros: paper/results_v3.tex
(prefix vthree, mean±std, n=3). Watcher scripts/wait_and_aggregate_v3.py produced both
automatically after seed 44 completed.

| learner | ACC | forget | frontier |
|---------|-----|--------|----------|
| frozen | 0.613 ± 0.000 | 0.025 ± 0.000 | +0.588 ± 0.000 |
| sccl (v1) | 0.596 ± 0.083 | 0.092 ± 0.029 | +0.504 ± 0.106 |
| sccl_n (v3) | 0.650 ± 0.038 | 0.129 ± 0.004 | +0.521 ± 0.042 |
| sccl_promote | 0.635 ± 0.050 | 0.110 ± 0.031 | +0.525 ± 0.078 |
| vsr_nogold (gold) | 0.629 ± 0.007 | 0.113 ± 0.033 | +0.517 ± 0.031 |

Pre-registered rule: success = sccl_n.forgetting <= sccl.forgetting AND
sccl_n.frontier >= sccl.frontier. VERDICT: FAIL on forgetting, robustly. Per-seed
forgetting sccl_n 0.131 / 0.125 / 0.131 vs sccl 0.075 / 0.075 / 0.125 — higher on ALL
three seeds; std 0.004 means sccl_n is the most STABLE row in the table and it is stably
worse on the stability metric. Frontier +0.521 vs +0.504: a tie inside noise.

What the filter DOES buy: highest mean ACC (0.650), lowest ACC variance (0.038), best
BWT (+0.170), and (seed-42 audit) higher admitted-target gold quality 0.617 → 0.667 at
half the plasticity (mean updates 14.0 vs 15.7). Accuracy up, accuracy-stability up,
forgetting up too: admission filtering removes bad TARGETS, but forgetting on this
stream is not dominated by target quality — it is dominated by weight drift INSIDE
accepted updates.

Per-family arith holdout (final), per seed — the v4 motivation, now multi-seed:

| seed | frozen | sccl | sccl_n | sccl_promote | vsr_nogold |
|------|--------|------|--------|--------------|------------|
| 42 | 0.60 | 0.40 | 0.375 | 0.40 | 0.40 |
| 43 | 0.60 | 0.40 | 0.400 | 0.375 | 0.40 |
| 44 | 0.60 | 0.20 | 0.175 | 0.60 | 0.15 |

Seeds 42+43: EVERY learning row ≤ 0.40 while frozen holds 0.60. Seed 44: three of four
learning rows collapse to 0.15–0.20; sccl_promote is the single exception (0.60). No
gate variant — gold-free or gold-verified — protects the unvisited arith holdout
consistently; sccl_n protects it on NO seed. Note seed-44 sccl_promote is exactly the
single-seed-luck pattern this protocol exists to catch — do not chase it.

CONCLUSION: gates (v1–v3) protect certified skills; they cannot protect uncertified
general capability damaged inside an accepted update. Act on the update itself →
v4 in-update base anchoring (RESEARCH_NOTES_v4.md, configs/sccl_v4.json).
