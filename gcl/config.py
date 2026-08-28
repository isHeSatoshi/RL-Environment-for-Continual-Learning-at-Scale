"""Experiment configuration — the single source of truth for a run (I4)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


@dataclass
class ExperimentConfig:
    # model
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"   # smoke; 1.5B/4B for signal/scale
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    # lora
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    learning_rate: float = 2e-4
    train_steps_per_update: int = 3
    max_seq_len: int = 512
    # stream
    families: List[str] = field(default_factory=list)
    tasks_per_family: int = 8
    episodes_per_task: int = 1
    max_attempts_per_task: int = 2
    # safety gate
    gate_epsilon: float = 0.02          # max allowed holdout regression to promote
    holdout_size: int = 8
    # clip/budget
    max_updates: int = 200
    # ---- Verified Skill Regeneration (VSR) ----
    use_vsr_gate: bool = False          # primary safety = vault-test veto (vs noisy holdout-eps)
    use_reference_injection: bool = False  # train toward external truth, not the model's own guess
    vsr_learners: List[str] = field(default_factory=lambda: ["vsr"])  # which learners get the vault
    refinject_learners: List[str] = field(default_factory=lambda: ["vsr"])  # which get gold/reference injection
    vault_commit_min: float = 0.9       # min reward to ADMIT a skill to the vault
    vault_retrieve_k: int = 3           # skills injected as in-context grounding per step
    vault_gate_check: int = 3           # verified skills re-checked on every gated update

    # ---- bounded VSR (anti-collapse) -----------------------------------------
    anchor_lambda: float = 0.0          # EWC-lite base pull on LoRA toward init (vsr_bounded > 0)
    anchor_learners: List[str] = field(default_factory=list)  # learners that get bounded updates
    replay_frac: float = 0.0            # fraction of replay buffer to mix per update (>0 => anti-overfit)
    replay_frac_learners: List[str] = field(default_factory=list)  # learners that get replay mixing
    lr_decay: bool = False              # if True, use bounded_lr() (decays with update count)
    vault_dedup_sim: float = 0.995      # above this similarity -> skip re-commit (dedup same skill)
    vault_dedup_learners: List[str] = field(default_factory=list)

    # ---- Self-taught (no-gold) rollout ----
    self_taught_learners: List[str] = field(default_factory=lambda: ["vsr_nogold", "vsr_self"])
    self_taught_k: int = 4              # initial diverse samples per task
    self_taught_temp: float = 0.7       # sampling temperature for diversity
    self_taught_repair_rounds: int = 1  # in-context repair rounds on failure
    self_taught_repair_k: int = 2       # candidates per repair round (greedy-ish)

    # ---- Self-Certified Continual Learning (SCCL): fully gold-free ----
    sccl_learners: List[str] = field(default_factory=lambda: ["sccl"])
    sccl_k: int = 6                     # diverse candidates per certification
    sccl_temp: float = 0.8              # candidate sampling temperature
    sccl_test_bags: int = 2             # independent self-test bags (consensus)
    sccl_tests_per_bag: int = 4         # asserts requested per bag
    sccl_tau: float = 0.65              # certification threshold (code)
    sccl_tau_math: float = 0.6          # certification threshold (math majority)
    sccl_derive_entry: bool = True      # derive entry name from spec (no gold parse)
    sccl_nogate_learners: List[str] = field(default_factory=list)   # certify, no RRV veto
    sccl_nocons_learners: List[str] = field(default_factory=list)   # certify, no consensus
    sccl_replay_check: int = 3          # certified skills re-checked per gated update
    sccl_replay_samples: int = 2        # regeneration attempts per skill in RRV veto
    sccl_gate_probe: int = 4            # gold-agreement telemetry probe size (eval only)
    # ---- SCCL v2: self-manufactured stability (still fully gold-free) ----
    sccl_replay_learners: List[str] = field(default_factory=list)  # certified-rehearsal learners
    sccl_replay_k: int = 0              # certified vault pairs stride-mixed into each update
    sccl_probe_learners: List[str] = field(default_factory=list)   # neighborhood-probe learners
    sccl_probes: int = 1                # probes generated per certified task
    sccl_probe_check: int = 0           # probes re-checked per RRV veto
    sccl_rrv_math: int = 0              # math vault entries re-checked per RRV veto
    # ---- probe curriculum (v3 candidate; off unless wired into a learner) ----
    sccl_probe_promote_learners: List[str] = field(default_factory=list)
    sccl_probe_promote_age: int = 0     # RRV probe-checks survived before a probe
                                        # graduates to a certified rehearsal pair
    # ---- SCCL v3: neighborhood certification (admission-time, gold-free) ----
    sccl_nbhd_learners: List[str] = field(default_factory=list)  # a cert may train
                                        # only if it is also consistent on a
                                        # self-generated spec variant (paraphrase
                                        # for code; numeric variant for math)
    sccl_nbhd_tests: int = 3            # self-tests written for the code variant
    # ---- SCCL v4: in-update base anchoring (gold-free capability preservation) ----
    # The RRV gate protects CERTIFIED vault skills, but the base model's UNCERTIFIED
    # general capability (e.g. arithmetic holdout) is damaged INSIDE an accepted
    # update, which no accept/reject gate can prevent. v4 pulls the LoRA update
    # toward the frozen base (= LoRA init; no labels) to preserve it. Per-learner
    # so it can be isolated in the ladder; lambda is the quadratic pull strength.
    sccl_anchor_learners: List[str] = field(default_factory=list)
    sccl_anchor_lambda: float = 0.0     # >0 => in-update base anchor for listed learners
    # per-learner override of the anchor strength (for a lambda ablation in one
    # ladder); falls back to sccl_anchor_lambda when a learner is absent here.
    sccl_anchor_lambdas: dict = field(default_factory=dict)
    # ---- SCCL v5: family-stratified RRV veto pool (gold-free gate coverage) ----
    # The default veto re-checks only skills[-sccl_replay_check:], a RECENCY
    # window; once the stream moves on, older certified families leave the gate's
    # field of view (seed-42 post-mortem: every veto broke only active-family
    # skills while arith eroded unchecked). Listed learners draw the check pool
    # newest-per-family instead. Off by default => pre-v5 rows bit-identical.
    sccl_stratified_learners: List[str] = field(default_factory=list)
    # ---- SCCL v5b: capability probes (gold-free capability-level gate) ----
    # v5 telemetry (Branch C, sccl_strat@44): 11/14 accepted updates degraded
    # the arith capability while memorized instances still passed their stored
    # self-tests — the instance-vs-capability gap. Neither a recency nor a
    # stratified pool over memorized instances can see it. Listed learners
    # manufacture certified CAPABILITY VARIANTS (numeric perturbations for
    # math domains; paraphrases with fresh self-tests for code domains) for
    # every certified skill, store the NEWEST one per family as kind=
    # "cap_probe" (never trained on), and the RRV veto re-checks them: an
    # update is vetted iff it breaks a family's GENERALIZATION, not just its
    # memorized points. Bounded-damage guard: once cap-probe vetoes reach the
    # budget fraction of a phase's update attempts, the veto grants margin
    # re-checks (extra regenerations) before breaking, so a noisy probe cannot
    # collapse plasticity. Off by default => pre-v5b rows bit-identical.
    sccl_capprobe_learners: List[str] = field(default_factory=list)
    sccl_capprobes: int = 0             # cap probes manufactured per certified skill
    sccl_capprobe_check: int = 0        # >0 => veto re-checks the per-family pool
    sccl_capprobe_budget: float = 0.5   # phase veto fraction that arms the guard
    sccl_capprobe_margin: int = 2       # extra regenerations under the guard
    # ---- SCCL v6: capability-probe ENSEMBLE pool (Branch D) ----
    # v5b telemetry (runs/sccl_v5b/telemetry_arith_erosion.json): every
    # probe-checked arith erosion PASSED its probe (100% insensitivity) because
    # the newest-per-family pool witnesses ONE skill axis of a heterogeneous
    # family; the catastrophic update destroyed an unwitnessed axis. Listed
    # pool sizes >1 keep up to K distinct-source-skill probes per family
    # (freshest variant per skill, K freshest skills) and the veto re-checks
    # ALL of them. sccl_capprobe_pool is the default (1 = v5b behaviour,
    # bit-identical); sccl_capprobe_pools overrides per learner name.
    sccl_capprobe_pool: int = 1
    sccl_capprobe_pools: dict = field(default_factory=dict)
    # ---- SCCL v7 (Branch E, E1): pass-rate margin veto ----
    # v6 telemetry (F1): 100% probe insensitivity under the any-of-n retain
    # rule — a probe whose regeneration quality degrades to 50% still
    # "passes" as long as one of n draws survives. E1 measures the pass RATE
    # over cap_samples regenerations per cap probe and retains iff
    # rate >= sccl_cap_retain_min (theta). theta <= 0 keeps the EXACT legacy
    # any-pass path (bit-identical for all pre-v7 rows); the *_mins / *_map
    # dicts override per learner name. sccl_cap_samples=0 falls back to
    # sccl_replay_samples. Pre-registered in RESEARCH_NOTES_v5.md (Branch E).
    sccl_cap_retain_min: float = 0.0
    sccl_cap_retain_mins: dict = field(default_factory=dict)
    sccl_cap_samples: int = 0
    sccl_cap_samples_map: dict = field(default_factory=dict)
    # output
    out_dir: str = "runs/run"
    seed: int = 42
    torch_seed: int = 0               # 0 = unseeded (legacy runs); >0 => seed
                                      # torch/cuda/numpy/random per learner
                                      # (torch_seed + learner index) for
                                      # reproducible certification sampling
    # rl / flags (honest GRPO: only if trl present + explicitly enabled)
    enable_grpo: bool = False

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "ExperimentConfig":
        with open(path) as f:
            return ExperimentConfig(**json.load(f))
