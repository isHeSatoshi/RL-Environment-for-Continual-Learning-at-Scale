"""End-to-end experiment runner — one command, real numbers (I4).

Protocol per learner (fresh engine => no leakage across learners):
  1. zero-shot eval on every family (frozen base, before any update)    -> baseline b_j
  2. online stream: for each episode the learner picks a learning op; the env
     verifies reward and (for UPDATE) performs a *gated* real gradient update.
  3. final eval: frozen policy over ALL families (adapter as it ended)     -> R_final[j]

Because there is a single stream walk, forgetting is revealed as a *drop* in the
final eval relative to that family's own zero-shot score (i.e., learned more on
later families at the cost of earlier ones).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import torch

from .config import ExperimentConfig
from .curriculum import Family, Task, canary_report
from .engine import TrainingEngine, _build_prompt, extract_code
from .env import GroundedContinualEnv, Action, LearnOp
from .learners.learners import LEARNERS, ControllerLearner
from .measure import (report as build_report, bwt_from_matrix, forgetting_from_matrix,
                      fwt_from_matrix, acc_from_matrix)
from .verify import Verifier
from .sandbox import PythonSandbox


class _FamilyProxy:
    """Duck-typed Family wrapper so held-out tasks can be passed to _eval_family."""
    def __init__(self, name, tasks):
        self.name = name
        self.tasks = tasks


def _task_reference_for_obs(env, obs) -> str:
    """Gold reference for the task the env is ABOUT to step (pre-step mirror).
    This is metadata only (never in the prompt) so VSR can correct toward it."""
    try:
        t = env._task()
        return getattr(t, "reference_answer", "") or ""
    except Exception:
        return ""


def _eval_family(engine, verifier, family, adapter_on=True) -> float:
    if not family.tasks:
        return 0.0
    scores = []
    for t in family.tasks:
        txt = engine.generate(_build_prompt(t), adapter_on=adapter_on)
        code = extract_code(txt) if t.domain == "code" else txt
        r, info, _ = verifier.reward(domain=t.domain, code=code, test_code=t.test_code,
                                     reference_answer=t.reference_answer)
        scores.append(r)
    return float(sum(scores) / len(scores))


def _eval_family_detail(engine, verifier, family, adapter_on=True):
    """Same scoring as _eval_family plus per-task records, so failure modes can
    be analyzed post-hoc without re-running GPU evaluation."""
    recs = []
    for t in (family.tasks or []):
        txt = engine.generate(_build_prompt(t), adapter_on=adapter_on)
        code = extract_code(txt) if t.domain == "code" else txt
        r, info, _ = verifier.reward(domain=t.domain, code=code, test_code=t.test_code,
                                     reference_answer=t.reference_answer)
        recs.append({"task_id": getattr(t, "task_id", ""), "score": float(r),
                     "pass_rate": float(info.get("pass_rate", 0.0)),
                     "success": bool(info.get("success", False))})
    mean = float(sum(x["score"] for x in recs) / len(recs)) if recs else 0.0
    return mean, recs


# incremental checkpointing: save after EVERY learner so a Kaggle/Lightning
# timeout never wipes an entire run (Fix 1 from the diagnostic report).
def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _save_checkpoint(reports: Dict[str, Any], out_dir: str) -> None:
    _atomic_write_json(os.path.join(out_dir, "metrics_partial.json"), reports)
    _write_summary(reports, os.path.join(out_dir, "summary_partial.md"))


def run_experiment(cfg: ExperimentConfig, learner_names: List[str],
                   families: List[Family], out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    cfg.out_dir = out_dir
    cfg.to_json(os.path.join(out_dir, "config.json"))
    canary = canary_report(families)
    if not canary["clean"]:
        raise RuntimeError(f"Anti-contamination check FAILED: {canary['overlap']} "
                           f"holdout ids overlap training stream (hash {canary['stream_hash']})")
    reports: Dict[str, Any] = {"config": cfg.__dict__, "canary": canary, "learners": {}}
    holdout = [t for f in families for t in f.holdout][: cfg.holdout_size]

    for idx, name in enumerate(learner_names):
        if name not in LEARNERS:
            # Fail LOUD: a silently-skipped learner name is an instrumentation
            # failure (the v5b smoke caught sccl_capprobe_strat being dropped
            # here — an entire experimental cell would have gone missing).
            raise ValueError(f"unknown learner name {name!r}; registered: "
                             f"{sorted(LEARNERS)}")
        print(f"\n[GCL] === Starting Learner {idx + 1}/{len(learner_names)}: {name} ===", flush=True)
        cfg._learner_name = name  # also enable bounded-update knobs scoped to this learner
        _ts = int(getattr(cfg, "torch_seed", 0))
        if _ts:  # 0 keeps legacy unseeded behaviour (bit-compat with old runs)
            import random as _pyrandom
            import numpy as _np
            _pyrandom.seed(_ts + idx)
            _np.random.seed(_ts + idx)
            torch.manual_seed(_ts + idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(_ts + idx)
            print(f"[GCL] [{name}] RNG seeded (torch_seed={_ts} + idx {idx})", flush=True)
        engine = TrainingEngine(cfg, adapter_root=os.path.join(out_dir, f"adapters_{name}"))
        verifier = Verifier(sandbox=PythonSandbox())
        learner = LEARNERS[name](cfg)
        nF = len(families)
        # VSR is a per-learner capability (clean contrast + ablations). Learners in
        # cfg.vsr_learners get the Skill Vault (reference injection + vault-test
        # gate); cfg.refinject_learners get reference injection WITHOUT the vault
        # gate (isolates the contribution of the safety mechanism); the rest stay
        # on the original holdout-eps gate.
        vault = None
        vsr_gate = False
        vsr_set = set(getattr(cfg, "vsr_learners", ["vsr"]))
        ref_set = set(getattr(cfg, "refinject_learners", ["vsr"]))
        sccl_set = (set(getattr(cfg, "sccl_learners", [])) |
                    set(getattr(cfg, "sccl_nogate_learners", [])) |
                    set(getattr(cfg, "sccl_nocons_learners", [])) |
                    set(getattr(cfg, "sccl_replay_learners", [])) |
                    set(getattr(cfg, "sccl_probe_learners", [])) |
                    set(getattr(cfg, "sccl_nbhd_learners", [])) |
                    set(getattr(cfg, "sccl_capprobe_learners", [])))
        is_sccl = name in sccl_set
        certifier = None
        if is_sccl:
            from .vault import SelfCertVault
            from .selfcert import SelfCertifier
            vault = SelfCertVault(directory=os.path.join(out_dir, f"vault_{name}"))
            certifier = SelfCertifier(
                k=getattr(cfg, "sccl_k", 6), temp=getattr(cfg, "sccl_temp", 0.8),
                test_bags=getattr(cfg, "sccl_test_bags", 2),
                tests_per_bag=getattr(cfg, "sccl_tests_per_bag", 4),
                tau=getattr(cfg, "sccl_tau", 0.65),
                tau_math=getattr(cfg, "sccl_tau_math", 0.6),
                consensus=name not in set(getattr(cfg, "sccl_nocons_learners", [])),
                nbhd_tests=getattr(cfg, "sccl_nbhd_tests", 3))
            print(f"[GCL] [{name}] SelfCert Vault enabled (gold-free RRV gate, "
                  f"tau={certifier.tau}, consensus={certifier.consensus}) -> {vault.directory}", flush=True)
        elif name in vsr_set or name in ref_set:
            from .vault import SkillVault
            vault = SkillVault(directory=os.path.join(out_dir, f"vault_{name}"))
            vsr_gate = name in vsr_set
            print(f"[GCL] [{name}] Skill Vault enabled (gate={'on' if vsr_gate else 'off'}) -> {vault.directory}", flush=True)

        # 1) true zero-shot baseline (adapter disabled == frozen base)
        print(f"[GCL] [{name}] Evaluating zero-shot baseline across {nF} families...", flush=True)
        zero_shot_detail: Dict[str, Any] = {}
        zero_shot = []
        for fam in families:
            s, recs = _eval_family_detail(engine, verifier, fam, adapter_on=False)
            zero_shot.append(s)
            zero_shot_detail[fam.name] = recs
        print(f"[GCL] [{name}] Zero-shot baseline scores: {[round(s, 3) for s in zero_shot]}", flush=True)
        first_contact = list(zero_shot)   # acc when family i is *first* evaluated during stream

        def first_contact_hook(eng, fi):
            if fi < nF - 1:
                first_contact[fi + 1] = _eval_family(eng, verifier, families[fi + 1], adapter_on=True)

        # Gold-free ladder semantics: the unsafe baselines (selfdistill/execfilter)
        # must NOT get a gold holdout gate (that would make them safe by gold).
        # SCCL variants receive the holdout ONLY as gold telemetry for the
        # gate-agreement analysis; their accept/reject decision never uses it.
        gate_holdout = [] if name in ("selfdistill", "execfilter") else holdout
        env = GroundedContinualEnv(cfg, engine, verifier, families,
                                   eval_hook=first_contact_hook,
                                   gate_epsilon=cfg.gate_epsilon, holdout=gate_holdout,
                                   vault=vault, vsr_gate=vsr_gate, sccl=is_sccl)
        # 2) continual stream
        trajs_path = os.path.join(out_dir, f"trajectories_{name}.jsonl")
        learning_curve, rewards, fam_curve = [], [], []
        updates = rollbacks = 0
        recall_hits = recall_probe_total = 0
        sccl_stats = {"steps": 0, "certified": 0, "rrv_updates": 0, "rrv_vetoes": 0,
                      "gold_probes": 0, "gold_agree": 0, "cert_conf_sum": 0.0,
                      "probes_made": 0, "probes_committed": 0, "probes_promoted": 0,
                      "nbhd_checked": 0, "nbhd_rejected": 0,
                      "cap_probes_made": 0, "cap_probes_committed": 0,
                      "cap_guard_armed": 0}
        # ---- SCCL v8 (Branch F, G1): GENERALIZATION witnesses --------------
        # For genprobe learners the veto pool carries a dedicated lane of
        # probes manufactured from UNTRAINED future stream tasks (spec-only
        # certify — the same gold-free path as v5b cap probes, never a gold
        # field): at FIRST CONTACT with a family, manufacture from the next
        # untrained task; RETIRE a gen probe before its source task trains;
        # REFRESH from the following untrained task. gen_state per family:
        # tasks (stream order), next_i (next candidate index), live (source
        # task_id of the currently live witness or None). Off unless listed
        # in cfg.sccl_genprobe_learners with sccl_genprobes > 0, so all
        # pre-v8 rows stay bit-identical.
        gen_on = (is_sccl and vault is not None
                  and name in set(getattr(cfg, "sccl_genprobe_learners", []))
                  and int(getattr(cfg, "sccl_genprobes", 0)) > 0)
        if gen_on:
            sccl_stats.update({"gen_probes_made": 0, "gen_probes_committed": 0,
                               "gen_probes_retired": 0})
        gen_state: Dict[str, Dict[str, Any]] = {}
        gen_last_family = None

        def _gen_make(family_name: str) -> bool:
            """G1: manufacture one generalization witness for `family_name`
            from the next UNTRAINED stream task. Spec-only: certify sees the
            candidate task's prompt/domain (+ derived entry) and nothing
            else — no gold test_code, no reference answer, no holdout task.
            Walks candidates until one certifies (found + usable self-tests)
            or the family's untrained tasks are exhausted. Returns True iff
            a witness was committed."""
            st = gen_state.get(family_name)
            if st is None:
                return False
            G = int(getattr(cfg, "sccl_genprobes", 0))
            _cpools = getattr(cfg, "sccl_capprobe_pools", {}) or {}
            _cpool = int(_cpools.get(
                name, getattr(cfg, "sccl_capprobe_pool", 1)))
            while st["live"] is None and st["next_i"] < len(st["tasks"]):
                t = st["tasks"][st["next_i"]]
                st["next_i"] += 1
                cert_entry = (None if getattr(cfg, "sccl_derive_entry", True)
                              else getattr(t, "entry_point", ""))
                try:
                    cr = certifier.certify(engine, verifier, t.prompt,
                                           t.domain, entry=cert_entry)
                except Exception:
                    cr = None
                sccl_stats["gen_probes_made"] += 1
                if cr is None or not cr.found:
                    continue
                if t.domain != "math" and not (cr.self_tests or []):
                    continue
                committed = vault.commit_cap_probe(
                    task_id=t.task_id + ":g", family=t.family,
                    spec=t.prompt, prompt=cr.prompt, code=cr.code,
                    self_tests=list(cr.self_tests or []),
                    conf=float(cr.confidence),
                    domain=getattr(cr, "domain", "") or t.domain,
                    entry=getattr(cr, "entry", "") or "",
                    pool=_cpool, gen_pool=G)
                sccl_stats["gen_probes_committed"] += int(bool(committed))
                if committed:
                    st["live"] = t.task_id
                    return True
            return False
        t0 = time.time()
        obs = env.reset()
        last_family_seen = 0
        fam_rewards: List[float] = []
        # Flush trajectory rows every N steps so partial runs survive a hard kill
        FLUSH_EVERY = 4
        with open(trajs_path, "w") as tf:
            while not env.done:
                task = env._task()
                st = None
                meta_extra: Dict[str, Any] = {}
                if is_sccl:
                    # ---- SCCL: gold-free self-certification (spec only) ----
                    # The certifier sees ONLY task.prompt (+ domain); entry names
                    # are self-derived unless configured otherwise. Gold test_code /
                    # reference_answer are never passed into this path.
                    cert_entry = None if getattr(cfg, "sccl_derive_entry", True) else task.entry_point
                    cr = certifier.certify(engine, verifier, task.prompt, task.domain,
                                           entry=cert_entry)
                    # ---- SCCL v3: NEIGHBORHOOD certification at admission ----
                    # Spec-only check (check_neighborhood sees spec/domain/cert,
                    # never a Task or gold field). A point-cert that is shown to
                    # be instance-narrow is demoted BEFORE it can train, commit,
                    # or arm a probe — targeting generalization damage at the
                    # source instead of vetoing it at the gate after the fact.
                    nbhd = None
                    if cr.found and name in set(getattr(cfg, "sccl_nbhd_learners", [])):
                        try:
                            nbhd = certifier.check_neighborhood(engine, verifier,
                                                                task.prompt,
                                                                task.domain, cr)
                        except Exception:
                            nbhd = {"robust": True, "reason": "nbhd_error_open"}
                        if nbhd is not None and not nbhd.get("robust", True):
                            cr.found = False  # demote: not trainable/committable
                            cr.diagnostics["nbhd"] = nbhd.get("reason", "fragile")
                    raw = cr.code or ""
                    gold_ref = ""
                    meta_extra["sccl"] = cr.to_dict()
                    if nbhd is not None:
                        meta_extra["sccl_nbhd"] = nbhd
                        sccl_stats["nbhd_checked"] += 1
                        sccl_stats["nbhd_rejected"] += int(not nbhd.get("robust", True))
                    sccl_stats["steps"] += 1
                    sccl_stats["certified"] += int(cr.found)
                    sccl_stats["cert_conf_sum"] += float(cr.confidence)
                    # ---- SCCL v2: manufacture a certified neighborhood probe ----
                    # Spec-only call: make_probe(spec, domain, CertResult) — the
                    # probe path never sees a Task object or any gold field.
                    if cr.found and vault is not None and \
                            name in set(getattr(cfg, "sccl_probe_learners", [])) and \
                            int(getattr(cfg, "sccl_probes", 0)) > 0:
                        try:
                            probe = certifier.make_probe(engine, verifier, task.prompt,
                                                         task.domain, cr)
                        except Exception:
                            probe = None
                        if probe is not None:
                            sccl_stats["probes_made"] += 1
                            committed = vault.commit_probe(
                                task_id=task.task_id + ":p", family=task.family,
                                spec=probe["spec"], prompt=probe["prompt"],
                                code=probe["code"],
                                self_tests=probe.get("self_tests") or [],
                                conf=float(probe.get("confidence", 0.0)),
                                domain=probe.get("domain", task.domain),
                                entry=probe.get("entry", ""))
                            sccl_stats["probes_committed"] += int(bool(committed))
                            meta_extra["sccl_probe"] = {"made": True,
                                                        "committed": bool(committed)}
                        else:
                            meta_extra["sccl_probe"] = {"made": False}
                    # ---- SCCL v5b: manufacture certified CAPABILITY probes ----
                    # Same spec-only machinery as v2 probes (make_probe sees
                    # spec/domain/CertResult, never a Task or gold field), but
                    # committed as kind="cap_probe" with newest-per-family
                    # replacement: the veto pool polices each family's
                    # GENERALIZATION with its freshest certified variant.
                    n_cap = int(getattr(cfg, "sccl_capprobes", 0))
                    if cr.found and vault is not None and n_cap > 0 and \
                            name in set(getattr(cfg, "sccl_capprobe_learners", [])):
                        # SCCL v6: per-learner ensemble pool size (must match the
                        # veto-side resolution in env.py so the pool the gate
                        # checks is exactly the pool that is retained).
                        _cpools = getattr(cfg, "sccl_capprobe_pools", {}) or {}
                        _cpool = int(_cpools.get(
                            name, getattr(cfg, "sccl_capprobe_pool", 1)))
                        cap_made = []
                        for i in range(n_cap):
                            try:
                                cp = certifier.make_probe(engine, verifier,
                                                          task.prompt,
                                                          task.domain, cr)
                            except Exception:
                                cp = None
                            if cp is None:
                                cap_made.append(False)
                                continue
                            sccl_stats["cap_probes_made"] += 1
                            committed = vault.commit_cap_probe(
                                task_id=task.task_id + f":c{i}",
                                family=task.family,
                                spec=cp["spec"], prompt=cp["prompt"],
                                code=cp["code"],
                                self_tests=cp.get("self_tests") or [],
                                conf=float(cp.get("confidence", 0.0)),
                                domain=cp.get("domain", task.domain),
                                entry=cp.get("entry", ""),
                                pool=_cpool)
                            sccl_stats["cap_probes_committed"] += int(bool(committed))
                            cap_made.append(bool(committed))
                        meta_extra["sccl_cap_probe"] = {"made": cap_made}
                else:
                    # VSR: retrieval-grounded generation (forward transfer). Controls:
                    # unchanged learner prompt. Gold reference is metadata only.
                    prompt = env.build_prompt() if vault is not None else learner.act_prompt(obs)
                    # Self-taught (no-gold) learners run pass@K + self-repair instead of a
                    # single generation; they learn only from execution reward.
                    is_self_taught = name in set(getattr(cfg, "self_taught_learners", ["vsr_nogold", "vsr_self"]))
                    if is_self_taught:
                        from .selftaught import self_taught_solve
                        st = self_taught_solve(engine, verifier, task,
                                               k_samples=getattr(cfg, "self_taught_k", 4),
                                               temp=getattr(cfg, "self_taught_temp", 0.7),
                                               repair_rounds=getattr(cfg, "self_taught_repair_rounds", 1),
                                               repair_k=getattr(cfg, "self_taught_repair_k", 2),
                                               commit_min=getattr(cfg, "vault_commit_min", 0.9))
                        # Feed the verified best self-solution as the env action; env.step
                        # will verify it and decide whether to gated-update.
                        raw = st["code"] if st["found"] else (st["code"] or "")
                        # when nothing was found, still step with empty answer to signal failure
                        if not raw.strip():
                            raw = ""
                        gold_ref = ""
                    else:
                        raw = engine.generate(prompt, adapter_on=True)
                        gold_ref = _task_reference_for_obs(env, obs)
                    if name == "execfilter":
                        # gold-free trainability: does the self-output EXECUTE cleanly?
                        if task.domain == "code":
                            code_chk = extract_code(raw)
                            res_chk = verifier.sandbox.execute(code_chk, test_code="")
                            meta_extra["trainable"] = bool(
                                res_chk.exit_code == 0 and not res_chk.error_type)
                        else:
                            meta_extra["trainable"] = bool((raw or "").strip())
                # ---- SCCL v8 (Branch F, G1): generalization witnesses ----
                # Hooks run BEFORE env.step so (a) the first update of a
                # family is already witnessed by a probe manufactured from an
                # UNTRAINED future task of that family (the v0 damage window
                # v7 telemetry found on sccl_strict), and (b) a witness is
                # RETIRED before its own source task can train — after
                # training it would degrade to a reproduction witness.
                _gen_retired = False
                if gen_on:
                    if task.family != gen_last_family:
                        gen_last_family = task.family
                        gen_state[task.family] = {
                            "tasks": list(env._family().tasks),
                            "next_i": env.task_idx + 1,
                            "live": None}
                        if _gen_make(task.family):
                            meta_extra["sccl_gen_probe"] = {
                                "first_contact": task.family,
                                "src": gen_state[task.family]["live"]}
                    _gs = gen_state.get(task.family)
                    if _gs is not None and _gs["live"] == task.task_id:
                        if vault.retire_cap_probe(task.task_id + ":g"):
                            sccl_stats["gen_probes_retired"] += 1
                        _gs["live"] = None
                        _gen_retired = True
                op = learner.decide(obs, None, False)
                obs2, reward, done, info = env.step(Action(
                    answer=raw, learn_op=op,
                    metadata={"reference_answer": gold_ref, "vault_enabled": bool(vault is not None),
                              "self_taught": {"found": (st["found"] if st else None)},
                              **meta_extra}))
                if isinstance(learner, ControllerLearner):
                    learner.learn(reward)
                rewards.append(reward)
                ui = info["update_info"]
                if gen_on and _gen_retired:
                    # G1 REFRESH: the retired witness's source task is now
                    # trained; manufacture the replacement from the next
                    # UNTRAINED task so the family stays witnessed.
                    _gen_make(task.family)
                vsr = info.get("vsr", {})
                if vsr:
                    recall_probe_total += int(vsr.get("n_retrieved", 0) > 0)
                    recall_hits += int(bool(vsr.get("recall_hit", False)))
                if is_sccl:
                    gate = ui.get("gate") or {}
                    if gate.get("method") == "sccl_rrv" and ui.get("executed"):
                        sccl_stats["rrv_updates"] += 1
                        sccl_stats["rrv_vetoes"] += int(not ui.get("accepted", True))
                        sccl_stats["probes_promoted"] += int(gate.get("probes_promoted", 0))
                        sccl_stats["cap_guard_armed"] += int(bool(gate.get("cap_guard_armed", False)))
                    gt = gate.get("gold_telemetry")
                    if isinstance(gt, dict):
                        sccl_stats["gold_probes"] += 1
                        # agreement: gold gate verdict vs the gold-free decision taken
                        sccl_stats["gold_agree"] += int(bool(gt.get("gold_gate_ok")) == bool(ui.get("accepted", True)))
                if ui.get("op") == "update_lora" and ui.get("accepted"):
                    updates += 1
                if ui.get("op") == "update_lora" and ui.get("executed") and not ui.get("accepted", True):
                    rollbacks += 1
                corr = ui.get("corrective")
                if isinstance(corr, dict) and corr.get("accepted"):
                    updates += 1
                cur_family = info["family"]
                if cur_family != families[last_family_seen].name:
                    fam_curve.append(sum(fam_rewards) / max(1, len(fam_rewards)))
                    fam_rewards = []
                    last_family_seen += 1
                fam_rewards.append(reward)
                tf.write(json.dumps({**info, "obs": obs.__dict__}) + "\n")
                # incremental trajectory flush — partial runs survive hard kill (Fix 2)
                if len(rewards) % FLUSH_EVERY == 0:
                    tf.flush(); os.fsync(tf.fileno())
                obs = obs2

                ep_count = len(rewards)
                if ep_count % 5 == 0 or env.done:
                    recent_avg = sum(rewards[-5:]) / max(1, len(rewards[-5:]))
                    print(f"[GCL] [{name}] Ep {ep_count:3d} | Family: {cur_family} | Reward (last 5): {recent_avg:.3f} | Updates: {updates} | Rollbacks: {rollbacks}", flush=True)

        if fam_rewards:
            fam_curve.append(sum(fam_rewards) / max(1, len(fam_rewards)))

        # 3) final eval — HELD-OUT (never shown during the stream) so accuracy reflects
        # retained general competence, not memorization of the training tasks.
        print(f"[GCL] [{name}] Stream complete! Evaluating final accuracy across all families (HELD-OUT + trained)...", flush=True)
        final_trained, final_trained_detail = [], {}
        for fam in families:
            s, recs = _eval_family_detail(engine, verifier, fam, adapter_on=True)
            final_trained.append(s)
            final_trained_detail[fam.name] = recs
        final_heldout, final_heldout_detail = [], {}
        for f in families:
            if getattr(f, "holdout", None):
                s, recs = _eval_family_detail(engine, verifier, _FamilyProxy(f.name, f.holdout),
                                              adapter_on=True)
                final_heldout.append(s)
                final_heldout_detail[f.name] = recs
            else:
                final_heldout.append(0.0)
        R = [[0.0] * nF for _ in range(2)]
        R[0] = list(zero_shot)          # before any learning (trained tasks)
        R[1] = list(final_trained)      # after the whole stream (trained)

        acc = acc_from_matrix([final_heldout])  # <<< held-out generalization, the headline metric
        per_family_drop = [max(0.0, zero_shot[j] - final_heldout[j]) for j in range(nF)]
        forgetting = sum(per_family_drop) / len(per_family_drop)
        bwt = sum(final_heldout[j] - zero_shot[j] for j in range(nF)) / nF
        rep = build_report(name, R, zero_shot, fam_curve, rewards, updates, cfg.max_updates)
        rep.acc = acc
        rep.bwt = bwt
        rep.forgetting = forgetting
        rep.updates = updates
        rep.fwt = fwt_from_matrix(R, first_contact)   # true forward transfer

        frontier = rep.acc - rep.forgetting
        recall_rate = (recall_hits / max(1, recall_probe_total)) if recall_probe_total else 0.0
        print(f"[GCL] [{name}] Finished in {round(time.time() - t0, 1)}s -> ACC: {acc:.3f} | BWT: {bwt:+.3f} | Forgetting: {forgetting:.3f} | Frontier: {frontier:+.4f} | Recall: {recall_rate:.3f}\n", flush=True)
        reports["learners"][name] = {
            "report": rep.to_dict(), "R_pairs": {"zero_shot": zero_shot, "first_contact": first_contact, "final": final_heldout,
                                                    "final_trained": final_trained},
            "eval_detail": {"zero_shot": zero_shot_detail,
                            "final_trained": final_trained_detail,
                            "final_heldout": final_heldout_detail},
            "family_curve": fam_curve, "updates": updates, "rollbacks": rollbacks,
            "wallclock_s": round(time.time() - t0, 1), "trajectories": trajs_path,
            "frontier_score": round(frontier, 4),
            "vsr": {"enabled": vault is not None, "recall_rate": round(recall_rate, 4),
                    "recall_hits": recall_hits, "recall_probes": recall_probe_total,
                    "vault_size": (len(vault) if vault is not None else 0)},
            "sccl": {**sccl_stats,
                     "cert_rate": round(sccl_stats["certified"] / max(1, sccl_stats["steps"]), 4),
                     "mean_conf": round(sccl_stats["cert_conf_sum"] / max(1, sccl_stats["steps"]), 4),
                     "gold_agreement": round(sccl_stats["gold_agree"] / max(1, sccl_stats["gold_probes"]), 4)},
            "adapter_history": engine.registry.history()[:5] + (["..."] if len(engine.registry.history()) > 5 else []),
        }
        # ---- INCREMENTAL CHECKPOINT (survives Kaggle/Lightning 12h kill) ----
        _save_checkpoint(reports, out_dir)
        print(f"[GCL] [{name}] checkpoint saved -> {out_dir}/metrics_partial.json", flush=True)
        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(reports, f, indent=2)
    _write_summary(reports, os.path.join(out_dir, "summary.md"))
    return reports


def _write_summary(reports: Dict[str, Any], path: str) -> None:
    canary = reports.get("canary", {})
    cfg = reports.get("config", {})
    lines = [
        "# Grounded Continual Learning — Experiment Summary", "",
        f"- Anti-contamination: train∩holdout = **{canary.get('overlap','?')}** (hash `{canary.get('stream_hash','')}`)",
        f"- Model `{cfg.get('model_name')}`, LoRA r={cfg.get('lora_r')}, updates cap={cfg.get('max_updates')}",
        "", "| Learner | ACC ↑ | BWT ↑ | Forgetting ↓ | AUC ↑ | Stability | Updates | Rollbacks | Frontier ↑ | VSR |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for name, d in reports.get("learners", {}).items():
        r = d["report"]
        v = d.get("vsr", {})
        vsr_cell = f"vault={v.get('vault_size',0)}/recall={v.get('recall_rate',0):.2f}" if v.get("enabled") else "-"
        lines.append(f"| {name} | {r['acc']:.3f} | {r['bwt']:+.3f} | {r['forgetting']:.3f} | {r['auc']:.3f} | {r['stability']:.3f} | {d['updates']} | {d['rollbacks']} | {d['frontier_score']:+.3f} | {vsr_cell} |")
    lines.append("")
    lines.append("All values derive from `metrics.json` + per-learner trajectories produced by an actually-executed run.")
    with open(path, "w") as f:
        f.write("\n".join(lines))
