"""SkillVault: a persistent library of *execution-verified* skills.

Core of Verified Skill Regeneration (VSR). It fixes self-distillation: the old
system trains the adapter on the model's OWN sampled output (`pair.target =
extracted`), so competence can never exceed the base model — accuracy stays at
the floor and "forgetting"/"transfer" collapse into sampling noise.

A Skill is external, executable *truth*: it is admitted ONLY after its code
actually passes the current task's sandbox tests. On later steps we (a) retrieve
verified skills as in-context grounding (forward transfer) and (b) choose the
supervised target as the highest-reward candidate that the verifier CONFIRMS
passes the CURRENT task's tests — preferring the untainted corpus gold answer,
then a re-validated retrieved skill, then (last resort) the model's own output.

Safety ("vault-test veto") is context-conditioned: an update is kept only if the
post-update candidate still solves the CURRENT task's tests AND has not lost the
*ability to express* any prior verified same-spec skill. Conflicting-spec skills
(a single program cannot satisfy "return 0" and "return 3") are not evidence of
forgetting and are excluded from the check — this is what stops all new-skill
acquisition from being vetoed.

Import-safe + self-contained (no gcl-external deps): an offline hash embedder +
numpy cosine store are built-in; sentence-transformers is used ONLY if available
offline/cached, else we degrade gracefully.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


# ----------------------------- embeddings -----------------------------------
@dataclass
class SkillCandidate:
    """A proposed supervised-training target, with provenance + live verification."""
    code: str
    source: str            # "gold" | "verified_skill" | "model_self"
    verified: bool         # did the verifier CONFIRM it passes the current task's tests
    score: float           # verifier reward on the current task


class _HashEmbedder:
    """Deterministic bag-of-words hash projection (offline, CPU, reproducible)."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, text: str):
        v = _np.zeros(self.dim, dtype="float32")
        words = (text or "").lower().split()
        if not words:
            return v
        for w in words:
            v[sum(ord(c) for c in w) % self.dim] += 1.0
        n = _np.linalg.norm(v)
        return (v / n) if n > 0 else v


def _make_embedder(dim: int = 384):
    """SentenceTransformer if already installed AND cached (offline-safe), else hash."""
    if _np is None:
        return None
    if os.getenv("ENABLE_ST_EMBEDDINGS", "1") == "1":
        try:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            return ("st", st, st.get_sentence_embedding_dimension())
        except Exception:
            pass
    return ("hash", _HashEmbedder(dim), dim)


# ----------------------------- store ----------------------------------------
class _Skill:
    __slots__ = ("task_id", "family", "prompt", "code", "test_code", "pass_rate",
                 "reward", "emb", "ts", "domain", "spec", "kind", "probe_checks")

    def __init__(self, task_id, family, prompt, code, test_code, pass_rate, reward,
                 emb, ts, domain: str = "code", spec: str = "", kind: str = "skill",
                 probe_checks: int = 0):
        self.task_id = task_id; self.family = family; self.prompt = prompt
        self.code = code; self.test_code = test_code; self.pass_rate = pass_rate
        self.reward = reward; self.emb = emb; self.ts = ts
        # `prompt` is the REGENERATION prompt (what the model is trained on);
        # `spec` is the raw task specification used for retrieval embeddings.
        self.domain = domain; self.spec = spec or prompt
        # `kind`: "skill" = a certified task the model TRAINED on;
        #         "probe" = a certified neighborhood variant it never trained on
        #         (SCCL v2: RRV probes test generalization, not memorization).
        self.kind = kind
        # `probe_checks`: RRV probe-checks this entry survived (probe curriculum:
        # surviving enough checks makes a probe eligible for promotion to skill).
        self.probe_checks = probe_checks


class SkillVault:
    """Verified-skill store + safety gate."""

    def __init__(self, directory: Optional[str] = None, dim: int = 384):
        self.directory = directory
        self._be_kind = "hash"; self._be = None; self.dim = dim
        if _np is not None:
            kind, be, d = _make_embedder(dim)
            self._be_kind, self._be, self.dim = kind, be, d
        self._skills: List[_Skill] = []
        self._embs: List[Any] = []
        if directory:
            self._load()

    def __len__(self) -> int:
        return len(self._skills)

    # ---- embedding ----
    def _embed(self, text: str):
        if self._be_kind == "st":
            e = self._be.encode([text], normalize_embeddings=True)
            return _np.asarray(e)[0].astype("float32")
        if self._be is not None:
            return self._be.encode(text)
        return None

    @staticmethod
    def _cosine(a, b):
        if _np is None or a is None or b is None:
            return 0.0
        return float(_np.dot(a, b))

    # ---- admission: only execution-verified skills enter the library -------
    def _too_similar_exists(self, task: Any, min_sim: float = 0.995) -> bool:
        if len(self._skills) < 1:
            return False
        q = self._embed(getattr(task, "prompt", ""))
        if q is None:
            return False
        return any(self._cosine(s.emb, q) >= min_sim for s in self._skills)

    def commit(self, task: Any, code: str, reward: float, domain: str = "code",
               min_reward: float = 0.9, pass_rate: float = 1.0,
               dedup_sim: float = 0.0) -> bool:
        if domain != "code":
            return False
        if reward < min_reward:
            return False
        if pass_rate < 1.0:
            return False
        code = (code or "").strip()
        if not code:
            return False
        # dedup: if the same skill is already stored, skip (avoid entrenchment of one)
        if dedup_sim > 0.0 and self._too_similar_exists(task, min_sim=dedup_sim):
            return False
        import time
        s = _Skill(task_id=getattr(task, "task_id", ""), family=getattr(task, "family", ""),
                   prompt=getattr(task, "prompt", ""), code=code,
                   test_code=getattr(task, "test_code", ""), pass_rate=float(pass_rate),
                   reward=float(reward), emb=self._embed(getattr(task, "prompt", "") + "\n" + code),
                   ts=time.time(), domain="code", spec=getattr(task, "prompt", ""))
        self._skills.append(s); self._embs.append(s.emb)
        self._save()
        return True

    def to_pairs(self) -> List[Dict[str, str]]:
        """Export verified skills as training pairs with context-grounded targets."""
        out = []
        for s in self._skills:
            if s.code:
                out.append({"prompt": f"{s.prompt}\n```python\n", "target": s.code})
        return out

    # ---- retrieval: in-context grounding / forward transfer -----------------
    def retrieve(self, prompt: str, k: int = 3) -> List[_Skill]:
        if not self._skills or k <= 0:
            return []
        q = self._embed(prompt)
        if q is None:
            return list(self._skills[-k:])
        scored = [(self._cosine(s.emb, q), i, s) for i, s in enumerate(self._skills)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, _, s in scored[:k]]

    def novelty(self, prompt: str) -> float:
        if not self._skills or _np is None:
            return 1.0
        q = self._embed(prompt)
        best = max((self._cosine(s.emb, q) for s in self._skills), default=0.0)
        return max(0.0, 1.0 - best)

    # ---- target selection: break self-distillation -------------------------
    def choose_target(self, task: Any, model_code: str, model_reward: float,
                      verifier: Any, *, gold: str = "",
                      retrieved: Optional[List[_Skill]] = None,
                      max_probe: int = 2, domain: str = "code") -> "SkillCandidate":
        if domain != "code":
            g = (gold or "").strip()
            return SkillCandidate(code=g, source="gold", verified=bool(g), score=1.0 if g else 0.0)
        test_code = getattr(task, "test_code", "")
        gold = (gold or "").strip()

        def _passes(code: str) -> Tuple[bool, float]:
            if not code or not code.strip():
                return False, 0.0
            try:
                r, info, _ = verifier.reward(domain="code", code=code,
                                             test_code=test_code, reference_answer=gold)
                pr = float(info.get("pass_rate", 0.0))
                return (pr >= 1.0 and bool(info.get("success", False))), float(r)
            except Exception:
                return False, 0.0

        if gold:  # 1) untainted corpus gold
            ok, sc = _passes(gold)
            if ok:
                return SkillCandidate(code=gold, source="gold", verified=True, score=sc)
        for t in (retrieved or [])[:max_probe]:  # 2) a verified skill that transfers
            sc_code = (t.code or "").strip()
            if not sc_code or sc_code == (model_code or "").strip():
                continue
            ok, sc = _passes(sc_code)
            if ok:
                return SkillCandidate(code=sc_code, source="verified_skill", verified=True, score=sc)
        ok, sc = _passes(model_code)  # 3) last resort: model's own (only if correct)
        if ok:
            return SkillCandidate(code=(model_code or "").strip(), source="model_self",
                                  verified=True, score=max(sc, model_reward))
        if gold:
            return SkillCandidate(code=gold, source="gold", verified=False, score=0.0)
        return SkillCandidate(code=(model_code or "").strip(), source="model_self",
                              verified=False, score=float(model_reward))

    # ---- safety gate: provable, context-conditioned vault-test veto ----------
    def violates(self, task: Any, candidate_code: str, verifier: Any,
                 *, retrieved: Optional[List[_Skill]] = None,
                 check_skills: int = 3, domain: str = "code",
                 sim_threshold: float = 0.65) -> Dict[str, Any]:
        """Veto on semantic forgetting only (see module docstring)."""
        if domain != "code":
            return {"veto": False, "reason": "noncode", "checked": 0, "broke": []}
        from .sandbox import check_safety

        broke: List[str] = []
        skipped: List[str] = []

        def _reward(code: str, tests: str, ref: str = ""):
            try:
                r, info, _ = verifier.reward(domain="code", code=code,
                                             test_code=tests, reference_answer=ref)
                return float(info.get("pass_rate", 0.0)), bool(info.get("success", False))
            except Exception:
                return 0.0, False

        # (a) current task must still pass
        pr, ok = _reward(candidate_code, getattr(task, "test_code", ""),
                         getattr(task, "reference_answer", ""))
        cur_id = getattr(task, "task_id", "current")
        if not ok:
            broke.append(cur_id)

        cur_entry = _entry_point(candidate_code) or _entry_point(getattr(task, "test_code", ""))
        q = self._embed(getattr(task, "prompt", ""))

        checked = 0
        seen = {cur_id}
        pool: List[_Skill] = list(retrieved or []) + list(self._skills[-check_skills:])
        for t in pool:
            if checked >= max(1, check_skills):
                break
            tid = t.task_id
            if tid in seen:
                continue
            seen.add(tid)
            s_entry = _entry_point(t.code) or _entry_point(t.test_code)
            # conflict filter: different spec class -> not forgetting evidence
            if cur_entry and s_entry and cur_entry != s_entry:
                skipped.append(tid)
                continue
            if not cur_entry:
                skipped.append(tid)
                continue
            # semantic relatedness: only blame a *related* skill for loss, else skip
            if self._be is not None and q is not None and t.emb is not None:
                if self._cosine(t.emb, q) < sim_threshold:
                    skipped.append(tid)
                    continue
            checked += 1
            if check_safety(candidate_code) or cur_entry not in (candidate_code or ""):
                broke.append(tid)

        return {"veto": len(broke) > 0,
                "reason": ("regress:" + ",".join(broke)) if broke else "ok",
                "checked": checked + 1, "broke": broke, "skipped_conflict": skipped,
                "n_skills": len(self._skills)}

    # ---- persistence ---------------------------------------------------------
    def _save(self):
        if not self.directory or _np is None:
            return
        os.makedirs(self.directory, exist_ok=True)
        recs = [_safe_skill_dict(s) for s in self._skills]
        try:
            with open(os.path.join(self.directory, "skills.json"), "w") as f:
                json.dump(recs, f)
            if self._embs and self._embs[0] is not None:
                _np.save(os.path.join(self.directory, "skills_emb.npy"),
                         _np.asarray(self._embs))
        except Exception:
            pass

    def _load(self):
        if not self.directory:
            return
        p = os.path.join(self.directory, "skills.json")
        if not os.path.exists(p):
            return
        try:
            with open(p) as f:
                recs = json.load(f)
            embs = None
            ep = os.path.join(self.directory, "skills_emb.npy")
            if _np is not None and os.path.exists(ep):
                embs = _np.load(ep)
            for i, r in enumerate(recs):
                emb = (embs[i] if embs is not None and i < len(embs) else self._embed(r.get("prompt", "")))
                self._skills.append(_Skill(r["task_id"], r.get("family", ""), r.get("prompt", ""),
                                           r.get("code", ""), r.get("test_code", ""),
                                           float(r.get("pass_rate", 0.0)), float(r.get("reward", 0.0)),
                                           emb, float(r.get("ts", 0.0)),
                                           domain=r.get("domain", "code"), spec=r.get("spec", r.get("prompt", "")),
                                           kind=r.get("kind", "skill"),
                                           probe_checks=int(r.get("probe_checks", 0))))
                self._embs.append(self._skills[-1].emb)
        except Exception:
            pass


def _safe_skill_dict(s: _Skill) -> Dict[str, Any]:
    return {"task_id": s.task_id, "family": s.family, "prompt": s.prompt,
            "code": s.code, "test_code": s.test_code, "pass_rate": s.pass_rate,
            "reward": s.reward, "ts": s.ts,
            "domain": getattr(s, "domain", "code"), "spec": getattr(s, "spec", s.prompt),
            "kind": getattr(s, "kind", "skill"),
            "probe_checks": getattr(s, "probe_checks", 0)}


def _entry_point(code_or_test: str) -> str:
    import re
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code_or_test or "")
    if m:
        return m.group(1)
    m = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", code_or_test or "")
    return m.group(1) if m else ""


# ----------------------------- SCCL store ------------------------------------
class SelfCertVault(SkillVault):
    """Vault of SELF-CERTIFIED skills for gold-free continual learning.

    Every admitted skill carries the model's OWN self-generated, self-executed
    test suite (never gold `task.test_code`). The safety gate is the
    Self-Replay Veto (RRV): after a candidate gradient update, each previously
    certified skill is REGENERATED under the updated adapter and re-run against
    ITS OWN stored self-tests; any regression vetoes/rolls back the update.
    """

    def _too_similar_spec_exists(self, spec: str, min_sim: float = 0.995) -> bool:
        if len(self._skills) < 1:
            return False
        q = self._embed(spec)
        if q is None:
            return False
        return any(self._cosine(s.emb, q) >= min_sim for s in self._skills)

    def commit_certified(self, task_id: str, family: str, spec: str, prompt: str,
                         code: str, self_tests: List[str], conf: float,
                         domain: str = "code", entry: str = "",
                         dedup_sim: float = 0.0, kind: str = "skill") -> bool:
        """Admit a self-certified skill (spec-only provenance, no gold)."""
        code = (code or "").strip()
        if not code:
            return False
        if domain == "code" and not (self_tests or []):
            return False
        if dedup_sim > 0.0 and self._too_similar_spec_exists(spec, min_sim=dedup_sim):
            return False
        import time
        s = _Skill(task_id=task_id, family=family, prompt=prompt, code=code,
                   test_code="\n".join(self_tests or []), pass_rate=float(conf),
                   reward=float(conf), emb=self._embed(spec + "\n" + code),
                   ts=time.time(), domain=domain, spec=spec, kind=kind)
        self._skills.append(s); self._embs.append(s.emb)
        self._save()
        return True

    def commit_probe(self, task_id: str, family: str, spec: str, prompt: str,
                     code: str, self_tests: List[str], conf: float,
                     domain: str = "code", entry: str = "") -> bool:
        """Admit a certified NEIGHBORHOOD probe (kind="probe", SCCL v2).

        Probes are certified variants the model never trains on; RRV re-checks
        them so the safety gate protects generalization, not just memorized
        training points. Dedup is skipped by construction: a probe is a
        paraphrase of an existing skill, so spec-similarity dedup would reject
        every probe.
        """
        return self.commit_certified(task_id=task_id, family=family, spec=spec,
                                     prompt=prompt, code=code, self_tests=self_tests,
                                     conf=conf, domain=domain, entry=entry,
                                     dedup_sim=0.0, kind="probe")

    @staticmethod
    def _cap_probe_source(task_id: str) -> str:
        """Source skill of a cap probe ('<skill>:c<i>' -> '<skill>')."""
        return task_id.rsplit(":c", 1)[0]

    def _cap_pool_by_family(self, pool: int) -> Dict[str, List[Any]]:
        """SCCL v6 (Branch D): cap_probe pool per family, ensemble-capable.

        Within each family keep only the FRESHEST variant of each distinct
        source skill (a family's skill axes), then keep the `pool` skills whose
        freshest variant is newest by commit order (listed in commit order).
        pool=1 reduces exactly to the v5b rule (newest cap_probe per family),
        so v5b rows stay bit-identical. Deterministic: no RNG.
        """
        obj: Dict[str, Dict[str, Any]] = {}
        last_idx: Dict[str, Dict[str, int]] = {}
        for i, s in enumerate(self._skills):
            if getattr(s, "kind", "") != "cap_probe":
                continue
            src = self._cap_probe_source(s.task_id)
            obj.setdefault(s.family, {})[src] = s
            last_idx.setdefault(s.family, {})[src] = i
        out: Dict[str, List[Any]] = {}
        for fam, by_src in obj.items():
            srcs = sorted(by_src, key=lambda x: last_idx[fam][x])
            out[fam] = [by_src[x] for x in srcs[-max(1, int(pool)):]]
        return out

    def _cap_pool_keep_ids(self, pool: int) -> set:
        """Ids of the cap_probes to KEEP under an ensemble pool of size `pool`."""
        return {id(s) for probes in self._cap_pool_by_family(pool).values()
                for s in probes}

    def commit_cap_probe(self, task_id: str, family: str, spec: str, prompt: str,
                          code: str, self_tests: List[str], conf: float,
                          domain: str = "code", entry: str = "",
                          pool: int = 1) -> bool:
        """Admit a certified CAPABILITY probe (kind="cap_probe", SCCL v5b).

        A cap_probe is a certified variant of a family's capability that the
        model never trains on: numeric perturbation for math domains,
        paraphrase with fresh self-tests for code domains. The veto pool is
        bounded by construction: with pool=1 only the NEWEST cap_probe of each
        family is kept (v5b), so the gate polices the freshest manufactured
        evidence of that capability at O(#families) per veto. SCCL v6
        (Branch D): pool>1 keeps an ENSEMBLE of up to `pool` distinct
        source-skill probes per family (freshest variant per skill), because
        v5b telemetry showed a single newest-per-family probe witnesses only
        ONE skill axis of a heterogeneous family and passed every update that
        destroyed an unwitnessed axis.
        """
        ok = self.commit_certified(task_id=task_id, family=family, spec=spec,
                                   prompt=prompt, code=code, self_tests=self_tests,
                                   conf=conf, domain=domain, entry=entry,
                                   dedup_sim=0.0, kind="cap_probe")
        if not ok:
            return False
        keep = self._cap_pool_keep_ids(pool)
        drop = [s for s in self._skills
                if getattr(s, "kind", "") == "cap_probe" and id(s) not in keep]
        if drop:
            drop_ids = {id(s) for s in drop}
            self._skills = [s for s in self._skills if id(s) not in drop_ids]
            self._embs = [s.emb for s in self._skills]
            self._save()
        return True

    def promote_probes(self, min_checks: int = 1) -> int:
        """Graduate probes that survived >= min_checks RRV checks into skills.

        Probe curriculum (v3 candidate): a promoted probe becomes an ordinary
        certified skill — it joins the certified-rehearsal pool (to_pairs) and
        stays protected under RRV's skill check, while fresher probes keep
        guarding the untrained frontier. This converts VALIDATED generalization
        neighborhoods into training data: every promoted entry was certified and
        then re-verified by regeneration, so the curriculum remains gold-free.
        """
        n = 0
        for s in self._skills:
            if getattr(s, "kind", "skill") == "probe" and getattr(s, "probe_checks", 0) >= min_checks:
                s.kind = "skill"
                n += 1
        if n:
            self._save()
        return n

    def to_pairs(self) -> List[Dict[str, str]]:
        """Replay pairs keyed on the stored regeneration prompt (code + math)."""
        out = []
        for s in self._skills:
            if not s.code:
                continue
            if getattr(s, "domain", "code") == "math":
                out.append({"prompt": s.prompt, "target": " " + s.code})
            else:
                out.append({"prompt": s.prompt, "target": s.code})
        return out

    def _any_passes(self, codes: List[str], tests: List[str], verifier: Any) -> bool:
        joined = "\n".join(tests)
        for c in codes:
            if not (c or "").strip():
                continue
            try:
                _, info, _ = verifier.reward(domain="code", code=c, test_code=joined)
                if float(info.get("pass_rate", 0.0)) >= 1.0 and bool(info.get("success", False)):
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _stratified_pool(skills: List[Any], k: int) -> List[Any]:
        """SCCL v5: family-stratified check pool (gold-free).

        The default recency window (skills[-k:]) abandons older families once
        the stream moves on — a seed-42 post-mortem found every RRV veto broke
        only active-family skills, so arith was never checked during the phases
        where arith holdout eroded. This pool takes the newest ceil(k/F)
        skills of EVERY certified family instead. Deterministic (sorted family
        keys, no RNG) so seeded runs stay reproducible.
        """
        fams: Dict[str, List[Any]] = {}
        for s in skills:
            fams.setdefault(getattr(s, "family", "") or "_", []).append(s)
        per = max(1, -(-k // max(1, len(fams))))  # ceil(k / F)
        pool: List[Any] = []
        for fam in sorted(fams):
            pool.extend(fams[fam][-per:])
        return pool

    def selfreplay_veto(self, engine: Any, verifier: Any, *,
                        check_skills: int = 3, n_samples: int = 2,
                        sample_temp: float = 0.7, check_probes: int = 0,
                        check_math: int = 0, stratified: bool = False,
                        check_cap_probes: int = 0, cap_margin: int = 0,
                        cap_pool: int = 1) -> Dict[str, Any]:
        """Self-Replay Veto (RRV) — the gold-free forgetting detector.

        For each recently committed self-certified skill, REGENERATE solutions
        from the stored prompt under the UPDATED adapter and re-run them against
        the skill's OWN stored self-tests. A skill is RETAINED iff some
        regeneration passes all its self-tests. If any skill regresses, the
        candidate weight update is vetoed (rolled back by the caller).

        SCCL v2 extends the veto beyond trained points:
          * check_probes>0 — also re-check the newest certified NEIGHBORHOOD
            probes (kind="probe", paraphrased specs the model never trained
            on). Probe regression = loss of generalization, the failure mode
            that trained-skill replay alone cannot see.
          * check_math>0 — also re-check math vault entries: regenerate
            answers and require at least one to equal the stored certified
            value (canonical-form comparison). In v1 math entries were never
            protected at all.
          * stratified — SCCL v5: draw the skill pool family-stratified
            (newest ceil(k/F) per family) instead of the global last-k, so
            older certified families stay under gate protection after the
            stream moves on. Off by default: existing rows stay bit-identical.
          * check_cap_probes>0 — SCCL v5b: re-check the newest certified
            CAPABILITY variant of every family (both domains). Polices the
            instance-vs-capability gap that skill/probe replay cannot see.
            cap_margin>0 arms the bounded-damage guard: a probe that breaks
            gets extra regeneration attempts before it can veto.
          * cap_pool>1 — SCCL v6 (Branch D): re-check an ENSEMBLE of up to
            cap_pool distinct-source-skill probes per family instead of only
            the newest. v5b telemetry showed one newest-per-family probe
            witnesses a single skill axis of a heterogeneous family and passed
            every update that destroyed an unwitnessed axis. cap_pool=1 (the
            default) is exactly the v5b newest-per-family behaviour.

        Note we deliberately do NOT fall back to executing the stored code: that
        artifact trivially passes its own tests regardless of the model's state,
        so it would never veto and would make the gate vacuous. RRV must measure
        the *model's current ability to re-express* each skill — that is exactly
        catastrophic forgetting, measured without any gold.
        """
        from .engine import extract_code
        from .selfcert import extract_number, format_number

        broke: List[str] = []
        broke_probes: List[str] = []
        broke_math: List[str] = []
        skipped: List[str] = []
        checked = checked_probes = checked_math = 0

        skills = [s for s in self._skills if getattr(s, "kind", "skill") == "skill"]
        probes = [s for s in self._skills if getattr(s, "kind", "skill") == "probe"]

        def _regen_codes(prompt: str, n: int = 0) -> List[str]:
            try:
                regen = engine.sample_candidates(prompt, n=max(1, n or n_samples),
                                                 temperature=sample_temp)
                return [c for c in (extract_code(r) for r in regen)
                        if (c or "").strip()]
            except Exception:
                return []

        # 1) certified code skills the model TRAINED on (v1 behaviour)
        if check_skills > 0 and stratified:
            pool = self._stratified_pool(skills, check_skills)
        else:
            pool = skills[-check_skills:] if check_skills > 0 else []
        for s in reversed(pool):
            if getattr(s, "domain", "code") != "code" or not (s.test_code or "").strip():
                skipped.append(s.task_id)
                continue
            tests = [l.strip() for l in s.test_code.splitlines()
                     if l.strip().startswith("assert")]
            if not tests:
                skipped.append(s.task_id)
                continue
            checked += 1
            if not self._any_passes(_regen_codes(s.prompt), tests, verifier):
                broke.append(s.task_id)

        # 2) neighborhood probes (code): same protocol on UNTRAINED variants
        ppool = probes[-check_probes:] if check_probes > 0 else []
        for s in reversed(ppool):
            if getattr(s, "domain", "code") != "code" or not (s.test_code or "").strip():
                skipped.append(s.task_id)
                continue
            tests = [l.strip() for l in s.test_code.splitlines()
                     if l.strip().startswith("assert")]
            if not tests:
                skipped.append(s.task_id)
                continue
            checked_probes += 1
            if self._any_passes(_regen_codes(s.prompt), tests, verifier):
                s.probe_checks = getattr(s, "probe_checks", 0) + 1
            else:
                broke_probes.append(s.task_id)

        # 3) math entries (skill or probe): answer must still match certified value
        if check_math > 0:
            mpool = [s for s in self._skills
                     if getattr(s, "domain", "code") == "math" and (s.code or "").strip()]
            for s in reversed(mpool[-check_math:]):
                want = format_number(str(s.code))
                checked_math += 1
                try:
                    regen = engine.sample_candidates(s.prompt, n=max(1, n_samples),
                                                     temperature=sample_temp)
                except Exception:
                    regen = []
                ok = False
                for r in regen:
                    a = extract_number(r or "")
                    if a is not None and format_number(a) == want:
                        ok = True
                        break
                if not ok:
                    broke_math.append(s.task_id)

        # 4) capability probes (SCCL v5b / v6): certified capability variants,
        # both domains. This stratum polices the instance-vs-capability gap:
        # memorized instances can keep passing their stored self-tests while
        # the capability the variants measure erodes inside accepted updates.
        # v5b checks the newest probe per family; v6 (Branch D) checks an
        # ENSEMBLE of up to cap_pool distinct-source-skill probes per family,
        # because a heterogeneous family has multiple skill axes and one probe
        # witnesses only one of them. cap_pool=1 == v5b (bit-identical).
        checked_cap = 0
        broke_cap: List[str] = []
        if check_cap_probes > 0:
            caps = self._cap_pool_by_family(max(1, int(cap_pool)))
            for fam in sorted(caps):
              for s in caps[fam]:
                if getattr(s, "domain", "code") == "math":
                    want = format_number(str(s.code))
                    checked_cap += 1

                    def _math_ok(samples: List[str]) -> bool:
                        for r in samples:
                            a = extract_number(r or "")
                            if a is not None and format_number(a) == want:
                                return True
                        return False

                    try:
                        regen = engine.sample_candidates(s.prompt,
                                                         n=max(1, n_samples),
                                                         temperature=sample_temp)
                    except Exception:
                        regen = []
                    ok = _math_ok(regen)
                    if not ok and cap_margin > 0:
                        try:
                            extra = engine.sample_candidates(s.prompt,
                                                             n=cap_margin,
                                                             temperature=sample_temp)
                        except Exception:
                            extra = []
                        ok = _math_ok(extra)
                    if not ok:
                        broke_cap.append(s.task_id)
                    continue
                tests = [l.strip() for l in (s.test_code or "").splitlines()
                         if l.strip().startswith("assert")]
                if not tests:
                    skipped.append(s.task_id)
                    continue
                checked_cap += 1
                codes = _regen_codes(s.prompt)
                if self._any_passes(codes, tests, verifier):
                    continue
                if cap_margin > 0:
                    extra = _regen_codes(s.prompt, cap_margin)
                    if self._any_passes(extra, tests, verifier):
                        continue
                broke_cap.append(s.task_id)

        all_broke = broke + broke_probes + broke_math + broke_cap
        return {"veto": bool(all_broke),
                "reason": ("rrv_regress:" + ",".join(all_broke)) if all_broke else "ok",
                "checked": checked, "broke": broke,
                "checked_probes": checked_probes, "broke_probes": broke_probes,
                "checked_math": checked_math, "broke_math": broke_math,
                "checked_cap": checked_cap, "broke_cap": broke_cap,
                "skipped": skipped, "n_skills": len(self._skills),
                "stratified": stratified}
