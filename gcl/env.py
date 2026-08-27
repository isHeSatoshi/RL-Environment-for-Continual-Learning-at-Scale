"""GroundedContinualEnv: lifelong MDP with the deployment SafetyGate in-loop.

step(action) executes:  verified reward -> learning op (store/update/consolidate)
-> for UPDATE it performs a gated update: snapshot -> gradient -> holdout gate
(base vs candidate via adapter disable) -> keep (register) or rollback. This is
the safe-continual-weight-update story realized concretely (I1 + I3).
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# engine imports torch at module scope; guard so env/mechanics are testable on a
# bare CPU box (the real TrainingEngine path re-exports the same helpers).
try:
    from .engine import extract_code, _build_prompt
except Exception:  # torch absent (e.g. headless test env)
    import re  # needed by fallback regexes below
    def extract_code(text: str) -> str:
        text = (text or "")
        def _strip_fence(block: str) -> str:
            b = block.strip()
            b = re.sub(r"^```(?:python)?\s*\n?", "", b, count=1)
            b = re.sub(r"\s*```$", "", b, count=1)
            return b.strip()
        blocks = []
        for m in re.finditer(r"```(?:[Pp]ython)?\s*\n?(.*?)```", text, flags=re.DOTALL):
            cand = _strip_fence(m.group(1))
            if cand.strip():
                blocks.append(cand)
        def _score(b: str) -> int:
            has_def = "def " in b
            body = re.sub(r"#.*", "", b).strip()
            is_stub = bool(re.match(r"^def\s+\w+\s*\(.*\):\s*(pass|\.\.\.)\s*$", body))
            return (2 if (has_def and not is_stub) else (1 if has_def else 0))
        if blocks:
            best = max(blocks, key=_score)
            if _score(best) > 0:
                return best
        matches = list(re.finditer(r"(?m)^(\s*def\s+\w+\s*\(.*)$", text))
        if matches:
            tail = text[matches[-1].start():]
            tail = re.split(r"\n\s*```", tail, maxsplit=1)[0]
            return tail.strip()
        # no compilable code anywhere (think-only / prose) — return empty so the
        # verifier treats it as a non-answer instead of a SyntaxError
        return ""  # engine real path


# ====[ /fallback cluster ]====

    def _build_prompt(task) -> str:
        if getattr(task, "domain", "code") == "math":
            return (f"Solve and give ONLY the final numeric answer.\n{task.prompt}\nAnswer: ")
        ep = getattr(task, "entry_point", "") or ""
        name_hint = f" The function MUST be named `{ep}`." if ep else ""
        return ("You are an expert Python programmer. Write ONLY the complete function "
                f"inside a ```python block. No analysis, no <think>, no empty 'pass' stubs."
                f"{name_hint}\n{task.prompt}\n```python\n")

try:  # VSR is optional so the frozen controls keep running on old configs
    from .vault import SkillVault
except Exception:  # pragma: no cover
    SkillVault = None


def _ground_prompt(base_prompt: str, retrieved) -> str:
    """Prepend up to `k` verified sibling solutions as in-context examples."""
    if not retrieved:
        return base_prompt
    shots = []
    for t in retrieved:
        code = (getattr(t, "code", None) or getattr(t, "generated_code", None) or "").strip()
        if code:
            shots.append(f"# Example solution (verified):\n```python\n{code}\n```")
    if not shots:
        return base_prompt
    return ("\n".join(shots) + "\n\n# Now solve the following.\n" + base_prompt)


class LearnOp(str, enum.Enum):
    IGNORE = "ignore"
    STORE = "store"
    UPDATE_LORA = "update_lora"
    CONSOLIDATE = "consolidate"
    REQUEST_REVIEW = "request_review"


@dataclass
class Action:
    answer: str = ""
    learn_op: LearnOp = LearnOp.IGNORE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    task_id: str
    family: str
    prompt: str
    domain: str
    step: int
    mem_stats: Dict[str, Any] = field(default_factory=dict)
    perf: Dict[str, float] = field(default_factory=dict)


@dataclass
class Trajectory:
    traj_id: str
    task_id: str
    family: str
    prompt: str
    answer: str
    extracted: str
    reward: float
    pass_rate: float
    success: bool
    learn_op: str
    update_info: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    def to_dict(self):
        return asdict(self)


# Safe terminal observation used when the stream is finished.
from .curriculum import Task as _Task
_TERMINAL_TASK = _Task(task_id="TERMINAL", family="DONE", prompt="", domain="code",
                       test_code="", reference_answer="", entry_point="")


class GroundedContinualEnv:
    def __init__(self, config, engine, verifier, stream, eval_hook=None,
                 gate_epsilon: Optional[float] = None, holdout: Optional[List] = None,
                 vault = None, vsr_gate: Optional[bool] = None, sccl: bool = False):
        self.cfg = config
        self.engine = engine
        self.verifier = verifier
        self.stream = stream
        self.eval_hook = eval_hook
        self.epsilon = config.gate_epsilon if gate_epsilon is None else gate_epsilon
        self.holdout = holdout or []
        self.enabled_ops_budget = config.max_updates
        self.vault = vault            # SkillVault or None (VSR)
        self.sccl = bool(sccl)        # Self-Certified CL: gold-free gate (RRV)
        self._gold_base_score: Optional[float] = None  # telemetry cache (never gates)
        # VSR mechanics are per-learner: reference injection is on iff a vault is
        # attached for THIS learner; the vault-test gate is separately toggleable.
        # SCCL disables both: its target + gate come from self-certification only.
        self._use_reference_injection = bool(vault is not None) and not self.sccl
        self._use_vsr_gate = ((bool(getattr(config, "use_vsr_gate", False)) if vsr_gate is None
                              else bool(vsr_gate)) and vault is not None) and not self.sccl
        self._lr_decay = bool(getattr(config, "use_lr_decay", False))
        self._anchor_lambda = float(getattr(config, "anchor_lambda", 0.0))
        self._replay_frac = float(getattr(config, "replay_frac", 0.0))
        self._dedup_enabled = bool(getattr(config, "use_vault_dedup", False))
        self._dedup_sim = float(getattr(config, "vault_dedup_sim", 0.95))
        self.last_retrieved: List = []  # set per-step for recall measurement
        self.reset()

    def _obs(self) -> Observation:
        task = self._task()
        return Observation(task_id=task.task_id, family=task.family, domain=task.domain,
                           prompt=task.prompt, step=self.t,
                           mem_stats={"replay": len(getattr(self.engine, "_replay", [])),
                                      "vault": len(self.vault) if self.vault else 0,
                                      "adapter_version": self.engine.registry.active_version,
                                      "rollbacks": self.rollback_count},
                           perf={"recent_mean": (sum(self.rewards[-8:]) / min(8, len(self.rewards))) if self.rewards else 0.0,
                                 "updates": self.update_count})

    def build_prompt(self) -> str:
        """VSR-conditioned prompt: prior verified skills as in-context grounding when
        available (forward transfer), else the raw task prompt. Controls see the
        unchanged prompt (`act_prompt` handles non-VSR)."""
        task = self._task()
        base = _build_prompt(task)
        if self._use_reference_injection and self.vault is not None:
            retrieved = self.vault.retrieve(task.prompt, k=self.cfg.vault_retrieve_k)
            self.last_retrieved = retrieved
            return _ground_prompt(base, retrieved)
        return base

    def reset(self):
        self.t = 0
        self.family_idx = 0
        self.task_idx = 0
        self.done = False
        self.trajectories: List[Trajectory] = []
        self.rewards: List[float] = []
        self.update_count = 0
        self.rollback_count = 0
        return self._obs()

    def _family(self):
        return self.stream[self.family_idx]

    def _task(self):
        # Guard terminal boundary: after finishing the last family, done=True is set
        # before _obs() builds the terminal observation; return a sentinel instead of
        # indexing past the end of the task list.
        if self.done or self.family_idx >= len(self.stream):
            return _TERMINAL_TASK
        fam = self._family()
        if self.task_idx >= len(fam.tasks):
            return _TERMINAL_TASK
        return fam.tasks[self.task_idx]

    def _obs(self) -> Observation:
        t = self._task()
        return Observation(task_id=t.task_id, family=t.family, domain=t.domain,
                           prompt=t.prompt, step=self.t,
            mem_stats={"replay": len(getattr(self.engine, "_replay", [])),
                       "vault": len(self.vault) if self.vault is not None else 0,
                       "adapter_version": self.engine.registry.active_version,
                       "rollbacks": self.rollback_count},
                           perf={"recent_mean": (sum(self.rewards[-8:]) / min(8, len(self.rewards))) if self.rewards else 0.0,
                                 "updates": self.update_count})

    def _task_passes(self, code: str, tests: str, ref: str = "") -> bool:
        try:
            _, info, _ = self.verifier.reward(domain="code", code=code,
                                              test_code=tests, reference_answer=ref)
            return float(info.get("pass_rate", 0.0)) >= 1.0 and bool(info.get("success", False))
        except Exception:
            return False

    def _gated_update(self, pairs, op, task=None, candidate_code: str = "",
                      retrieved: Optional[List] = None,
                      sccl_cert: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Snapshot -> grad update -> safety gate -> keep/rollback (I3).

        Three gate regimes:
          * VSR:  provable vault-test veto (gold tests; the VSR contribution).
          * SCCL: Self-Replay Veto — regenerate previously SELF-certified skills
                  under the updated adapter and re-run THEIR OWN self-tests.
                  Fully gold-free: no gold test, reference, or holdout touches
                  the accept/reject decision.
          * holdout_eps: legacy noisy margin (kept for back-compat configs).
        """
        eng = self.engine
        snap = eng._snapshot()
        use_vsr = bool(getattr(self.cfg, "use_vsr_gate", False)) and self.vault is not None and task is not None and self._use_vsr_gate
        name = getattr(self.cfg, "_learner_name", "")
        use_sccl = (self.sccl and self.vault is not None and task is not None
                    and name not in set(getattr(self.cfg, "sccl_nogate_learners", [])))
        sccl_nogate = (self.sccl and not use_sccl
                       and name in set(getattr(self.cfg, "sccl_nogate_learners", [])))

        method = "sccl_rrv" if use_sccl else ("none" if sccl_nogate else ("vsr" if use_vsr else "holdout_eps"))
        gate: Dict[str, Any] = {"epsilon": self.epsilon, "method": method}

        if not use_vsr and not use_sccl and not sccl_nogate:
            base_score = eng.holdout_score(self.holdout, self.verifier, adapter_on=False) if self.holdout else 1.0
            cand_before = eng.holdout_score(self.holdout, self.verifier, adapter_on=True) if self.holdout else base_score
            gate.update({"base_score": base_score, "cand_before": cand_before})

        lr = eng.bounded_lr() if self._lr_decay else None
        anch = self._anchor_lambda if (self._anchor_lambda > 0) else 0.0
        # ---- SCCL v4: per-learner in-update BASE ANCHOR (gold-free). Pulls the
        # LoRA update toward the frozen base so accepted updates preserve the base
        # model's UNCERTIFIED general capability, which the RRV gate cannot protect.
        # Anchor target is LoRA init (= frozen base); no labels anywhere. ----
        if name in set(getattr(self.cfg, "sccl_anchor_learners", [])):
            _lamap = getattr(self.cfg, "sccl_anchor_lambdas", {}) or {}
            _sa = float(_lamap.get(name, getattr(self.cfg, "sccl_anchor_lambda", 0.0)))
            if _sa > 0:
                anch = _sa
        rf = self._replay_frac if (self._replay_frac > 0) else 0.0
        rp = self.vault.to_pairs()[:int(max(1, len(self.vault._skills) * rf))] if (rf > 0 and self.vault is not None) else None
        # ---- SCCL v2: CERTIFIED REHEARSAL (gold-free) ------------------------
        # Mix stride-sampled pairs from the SELF-certified vault into every
        # update so prior (cross-family) certified skills keep receiving
        # gradient. The replay buffer itself is certified-correct by
        # construction — nothing here touches gold.
        rk = (int(getattr(self.cfg, "sccl_replay_k", 0))
              if name in set(getattr(self.cfg, "sccl_replay_learners", [])) else 0)
        if rk > 0 and self.vault is not None:
            vp = self.vault.to_pairs()
            if vp:
                if len(vp) > rk:
                    idxs = sorted({int(i * len(vp) / rk) for i in range(rk)})
                    rp = [vp[i] for i in idxs]
                else:
                    rp = list(vp)
                rf = 1.0  # pairs already selected above; pass them through as-is
        m = eng.apply_update(pairs, lr=lr, anchor_lambda=anch, replay_frac=rf,
                             replay_pairs=rp)
        gate["anchor_lambda"] = anch  # v4 audit trail: persisted with the gate verdict

        if use_sccl:
            probe_learner = name in set(getattr(self.cfg, "sccl_probe_learners", []))
            veto = self.vault.selfreplay_veto(
                eng, self.verifier,
                check_skills=getattr(self.cfg, "sccl_replay_check", 3),
                n_samples=getattr(self.cfg, "sccl_replay_samples", 2),
                check_probes=int(getattr(self.cfg, "sccl_probe_check", 0)) if probe_learner else 0,
                check_math=int(getattr(self.cfg, "sccl_rrv_math", 0)) if probe_learner else 0)
            gate.update({"veto": veto["veto"], "veto_reason": veto["reason"],
                         "checked": veto["checked"], "broke": veto["broke"],
                         "checked_probes": veto.get("checked_probes", 0),
                         "broke_probes": veto.get("broke_probes", []),
                         "checked_math": veto.get("checked_math", 0),
                         "broke_math": veto.get("broke_math", []),
                         "skipped": veto.get("skipped", [])})
            accepted = not veto["veto"]
            # ---- Gold telemetry ONLY (post-hoc gate-agreement analysis). ----
            # These scores never enter `accepted`; they let the paper quantify
            # how often the gold-free RRV gate agrees with a gold holdout gate.
            probe = int(getattr(self.cfg, "sccl_gate_probe", 0))
            if probe > 0 and self.holdout:
                hs = self.holdout[:probe]
                if self._gold_base_score is None:
                    self._gold_base_score = eng.holdout_score(hs, self.verifier, adapter_on=False)
                cand_h = eng.holdout_score(hs, self.verifier, adapter_on=True)
                gate["gold_telemetry"] = {"base": self._gold_base_score, "cand": cand_h,
                                          "gold_gate_ok": bool(cand_h >= self._gold_base_score - self.epsilon)}
        elif use_vsr:
            veto = self.vault.violates(task, candidate_code, self.verifier,
                                       retrieved=retrieved,
                                       check_skills=getattr(self.cfg, "vault_gate_check", 3),
                                       domain=getattr(task, "domain", "code"))
            if self.holdout:  # cheap secondary floor
                base_h = eng.holdout_score(self.holdout, self.verifier, adapter_on=False)
                cand_h = eng.holdout_score(self.holdout, self.verifier, adapter_on=True)
                gate.update({"base_h": base_h, "cand_h": cand_h,
                             "holdout_floor_ok": cand_h >= base_h - self.epsilon})
            gate.update({"veto": veto["veto"], "veto_reason": veto["reason"],
                         "checked": veto["checked"], "broke": veto["broke"]})
            accepted = not veto["veto"]
        elif sccl_nogate:
            # Ablation: certified target, NO safety gate. Gold telemetry only,
            # so the paper can show what a gold gate would have decided.
            accepted = True
            probe = int(getattr(self.cfg, "sccl_gate_probe", 0))
            if probe > 0 and self.holdout:
                hs = self.holdout[:probe]
                if self._gold_base_score is None:
                    self._gold_base_score = eng.holdout_score(hs, self.verifier, adapter_on=False)
                cand_h = eng.holdout_score(hs, self.verifier, adapter_on=True)
                gate["gold_telemetry"] = {"base": self._gold_base_score, "cand": cand_h,
                                          "gold_gate_ok": bool(cand_h >= self._gold_base_score - self.epsilon)}
        else:
            cand_after = eng.holdout_score(self.holdout, self.verifier, adapter_on=True) if self.holdout else 1.0
            accepted = (cand_after >= gate["base_score"] - self.epsilon)
            gate.update({"cand_after": cand_after})
        gate["accepted"] = accepted

        if accepted:
            meta = eng.register_adapter(op, {**m, "gate": gate})
            self.update_count += 1
            # ---- Probe curriculum (v3 candidate, gold-free): probes that have
            # survived enough RRV checks graduate to certified rehearsal pairs.
            pp_age = (int(getattr(self.cfg, "sccl_probe_promote_age", 0))
                      if name in set(getattr(self.cfg, "sccl_probe_promote_learners", []))
                      else 0)
            if pp_age > 0 and self.vault is not None:
                gate["probes_promoted"] = self.vault.promote_probes(pp_age)
            return {"executed": True, "accepted": True, "loss": m["loss_end"],
                    "grad_norm": m["grad_norm"], "adapter_version": meta.version,
                    "hash": meta.content_hash, "anchor_lambda": anch, "gate": gate}
        eng._restore(snap)
        self.rollback_count += 1
        return {"executed": True, "accepted": False,
                "reason": ("rrv_veto" if use_sccl else ("vault_veto" if use_vsr else "holdout_regression")),
                "gate": gate}

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        task = self._task()
        extracted = extract_code(action.answer) if task.domain == "code" else action.answer
        # NOTE: this reward is computed against GOLD tests for telemetry/measurement
        # only. In SCCL mode it never influences any learning decision (target,
        # gate, and commit all come from self-certification — see below).
        reward, info, res = self.verifier.reward(domain=task.domain, code=extracted,
                                                 test_code=task.test_code,
                                                 reference_answer=task.reference_answer)
        op = action.learn_op
        update_info: Dict[str, Any] = {"op": op.value, "executed": False}
        can_update = self.update_count < self.enabled_ops_budget

        use_vsr = self._use_reference_injection and self.vault is not None
        use_vsr_gate = self._use_vsr_gate and self.vault is not None
        mode = "sccl" if (self.sccl and self.vault is not None) else ("vsr" if use_vsr else "base")

        # Gold reference may also be supplied at act-time via metadata for the
        # reference-injection / ablation paths; NEVER enters the model prompt.
        # For self-taught (no-gold) learners, task.reference_answer must be ignored.
        self_taught = getattr(self.cfg, "_learner_name", "") in set(
            getattr(self.cfg, "self_taught_learners", ["vsr_nogold", "vsr_self"]))
        if self_taught:
            gold_available = ""
        else:
            gold_from_meta = (action.metadata or {}).get("reference_answer", "")
            gold_available = task.reference_answer or gold_from_meta

        # ---- VSR: retrieve verified skills for grounding (measured for FWT) ----
        retrieved: List = []
        if self.vault is not None:
            try:
                retrieved = self.vault.retrieve(task.prompt, k=self.cfg.vault_retrieve_k)
            except Exception:
                retrieved = []
        self.last_retrieved = retrieved
        recall_hit = False

        # ---- Training target: sccl cert > gold > verified-skill > (correct) self --
        passed = float(info.get("pass_rate", 0.0)) >= 1.0 and bool(info.get("success", False))
        sccl_meta: Dict[str, Any] = {}
        pair_prompt = _build_prompt(task)
        if mode == "sccl":
            # Fully gold-free target: the self-certified code + self-tests from
            # metadata. trainability is decided by certification confidence alone.
            sccl_meta = (action.metadata or {}).get("sccl", {}) or {}
            cert_code = str(sccl_meta.get("code", "") or "")
            if task.domain == "math":
                target_code = cert_code.strip()
                pair_target = (" " + target_code) if target_code else ""
            else:
                # The certifier already extracted executable fence-free code
                # (imports intact); re-running extract_code here would hit its
                # last-def fallback and strip leading import lines.
                target_code = cert_code.strip()
                pair_target = target_code
                if sccl_meta.get("prompt"):
                    pair_prompt = sccl_meta["prompt"]
            target_source = "self_certified"
            target_verified = bool(sccl_meta.get("found"))
            # Structural self-consistency: a certified code target must pass its
            # own certifying self-tests verbatim before it may train the adapter.
            # Catches any post-certification mangling (e.g. lost imports).
            if target_verified and target_code and task.domain == "code":
                st = sccl_meta.get("self_tests") or []
                if st:
                    _, st_info, _ = self.verifier.reward(
                        domain="code", code=target_code,
                        test_code="\n".join(st), reference_answer="")
                    if float(st_info.get("pass_rate", 0.0)) < 1.0:
                        target_verified = False
            trainable = target_verified and bool(target_code)
        elif use_vsr:
            cand = self.vault.choose_target(
                task, extracted, reward, self.verifier,
                gold=gold_available, retrieved=retrieved, domain=task.domain)
            target_code = cand.code
            target_source = cand.source
            target_verified = cand.verified
            recall_hit = cand.source == "verified_skill"
            pair_target = target_code
            # self = model already correct; gold/verified_skill = provable correct.
            # For nogold (vault empty), target_code=model output when verified==True.
            # If no verified target exists, we still train ALLOWEDLY when correction
            # is possible (vault.retrieve found a passing skill for this family).
            trainable = target_verified or (cand.source == "verified_skill")
        else:
            target_code = extracted
            target_source = "self"
            target_verified = passed
            pair_target = target_code
            meta_trainable = (action.metadata or {}).get("trainable", None)
            trainable = bool(meta_trainable) if meta_trainable is not None \
                else len((extracted or "").strip()) > 0

        good_pair = {"prompt": pair_prompt, "target": pair_target}
        update_info["target_source"] = target_source
        update_info["target_verified"] = target_verified

        if op == LearnOp.STORE:
            self.engine._replay.append(good_pair)
            update_info["executed"] = True
        elif op == LearnOp.UPDATE_LORA and can_update and trainable:
            update_info = {"op": op.value,
                           **self._gated_update([good_pair], "update_lora",
                                                task=task, candidate_code=target_code,
                                                retrieved=retrieved,
                                                sccl_cert=sccl_meta if mode == "sccl" else None),
                           "target_source": target_source,
                           "target_verified": target_verified}
        elif op == LearnOp.UPDATE_LORA and can_update and not trainable:
            update_info["executed"] = False
            update_info["reason"] = "no_verified_target"
        elif op == LearnOp.CONSOLIDATE and can_update:
            m = self.engine.consolidate_ewc([good_pair])
            update_info = {"op": op.value, "executed": True, "ewc_params": m.get("ewc_params", 0)}
            self.update_count += 1
        elif op == LearnOp.REQUEST_REVIEW:
            update_info = {"op": op.value, "executed": False, "review": "queued"}

        # ---- Corrective update: model wrong but we HAVE verified truth ----
        if use_vsr and can_update and (not passed) and target_verified and \
                target_source in ("gold", "verified_skill") and reward < self.cfg.vault_commit_min:
            # Probe: would this exact verified answer survive the current adapter?
            would_hold = self._task_passes(target_code, task.test_code, task.reference_answer)
            if not would_hold:
                corr = self._gated_update([good_pair], "corrective",
                                          task=task, candidate_code=target_code,
                                          retrieved=retrieved)
                update_info["corrective"] = corr
                update_info["corrective"]["accepted"] = corr.get("accepted", False)

        # ---- Commit a genuinely-new verified skill to the vault ---------------
        if use_vsr and passed:
            self.vault.commit(task, extracted, reward, domain=task.domain,
                              min_reward=self.cfg.vault_commit_min,
                              pass_rate=float(info.get("pass_rate", 0.0)),
                              dedup_sim=self._dedup_sim if self._dedup_enabled else 0.0)
        elif mode == "sccl" and target_verified and bool(target_code):
            self.vault.commit_certified(task_id=task.task_id, family=task.family,
                                        spec=task.prompt, prompt=pair_prompt,
                                        code=target_code,
                                        self_tests=sccl_meta.get("self_tests", []) or [],
                                        conf=float(sccl_meta.get("confidence", 0.0)),
                                        domain=task.domain,
                                        entry=sccl_meta.get("entry", ""),
                                        dedup_sim=self._dedup_sim if self._dedup_enabled else 0.0)

        traj = Trajectory(traj_id=f"t{self.t}_{task.task_id}", task_id=task.task_id,
                          family=task.family, prompt=task.prompt, answer=action.answer,
                          extracted=extracted, reward=reward, pass_rate=info.get("pass_rate", 0.0),
                          success=bool(info.get("success", False)), learn_op=op.value,
                          update_info=update_info)
        self.trajectories.append(traj)
        self.rewards.append(reward)
        self.t += 1
        # roll updates committed via a corrective pass (they run through _gated_update,
        # which already increments update_count, so they are not double-counted here)

        self.task_idx += 1
        if self.task_idx >= len(self._family().tasks):
            finished = self.family_idx
            self.family_idx += 1
            self.task_idx = 0
            self.done = self.family_idx >= len(self.stream)
            if self.eval_hook is not None and not self.done:
                try:
                    self.eval_hook(self.engine, finished)
                except Exception:
                    pass
        self.done = self.family_idx >= len(self.stream)

        o = self._obs() if not self.done else Observation(task_id="DONE", family="", prompt="",
                                                          domain="code", step=self.t)
        step_info = {"reward": reward, "verifier": info, "learn_op": op.value,
                     "update_info": update_info, "task_id": traj.task_id, "family": traj.family,
                     "vsr": {"recall_hit": recall_hit,
                             "n_retrieved": len(retrieved),
                             "target_source": update_info.get("target_source", "self"),
                             "vault_size": len(self.vault) if self.vault is not None else 0},
                     "sccl": ({"found": bool(sccl_meta.get("found")),
                               "confidence": float(sccl_meta.get("confidence", 0.0)),
                               "n_tests": int(sccl_meta.get("n_tests", 0)),
                               "n_discriminative": int(sccl_meta.get("n_discriminative", 0)),
                               "gate_method": (update_info.get("gate", {}) or {}).get("method", ""),
                               "gate_accepted": update_info.get("accepted", None)}
                              if mode == "sccl" else {})}
        # Propagate per-step audit fields the experiment attached to the action
        # (v3 neighborhood verdict, v2 probe manufacture). Telemetry only —
        # neither field influenced the step's decision.
        for _k in ("sccl_nbhd", "sccl_probe"):
            if action.metadata and _k in action.metadata:
                step_info[_k] = action.metadata[_k]
        return o, reward, self.done, step_info
