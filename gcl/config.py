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
