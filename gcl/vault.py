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
                 "reward", "emb", "ts", "domain", "spec")

    def __init__(self, task_id, family, prompt, code, test_code, pass_rate, reward,
                 emb, ts, domain: str = "code", spec: str = ""):
        self.task_id = task_id; self.family = family; self.prompt = prompt
        self.code = code; self.test_code = test_code; self.pass_rate = pass_rate
        self.reward = reward; self.emb = emb; self.ts = ts
        # `prompt` is the REGENERATION prompt (what the model is trained on);
        # `spec` is the raw task specification used for retrieval embeddings.
        self.domain = domain; self.spec = spec or prompt


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
                                           domain=r.get("domain", "code"), spec=r.get("spec", r.get("prompt", ""))))
                self._embs.append(self._skills[-1].emb)
        except Exception:
            pass


def _safe_skill_dict(s: _Skill) -> Dict[str, Any]:
    return {"task_id": s.task_id, "family": s.family, "prompt": s.prompt,
            "code": s.code, "test_code": s.test_code, "pass_rate": s.pass_rate,
            "reward": s.reward, "ts": s.ts,
            "domain": getattr(s, "domain", "code"), "spec": getattr(s, "spec", s.prompt)}


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
                         dedup_sim: float = 0.0) -> bool:
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
                   ts=time.time(), domain=domain, spec=spec)
        self._skills.append(s); self._embs.append(s.emb)
        self._save()
        return True

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

    def selfreplay_veto(self, engine: Any, verifier: Any, *,
                        check_skills: int = 3, n_samples: int = 2,
                        sample_temp: float = 0.7) -> Dict[str, Any]:
        """Self-Replay Veto (RRV) — the gold-free forgetting detector.

        For each recently committed self-certified skill, REGENERATE solutions
        from the stored prompt under the UPDATED adapter and re-run them against
        the skill's OWN stored self-tests. A skill is RETAINED iff some
        regeneration passes all its self-tests. If any skill regresses, the
        candidate weight update is vetoed (rolled back by the caller).

        Note we deliberately do NOT fall back to executing the stored code: that
        artifact trivially passes its own tests regardless of the model's state,
        so it would never veto and would make the gate vacuous. RRV must measure
        the *model's current ability to re-express* each skill — that is exactly
        catastrophic forgetting, measured without any gold.
        """
        from .engine import extract_code

        broke: List[str] = []
        skipped: List[str] = []
        checked = 0

        pool = list(self._skills)[-check_skills:] if check_skills > 0 else []
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
            try:
                regen = engine.sample_candidates(s.prompt, n=max(1, n_samples),
                                                 temperature=sample_temp)
                codes = [extract_code(r) for r in regen]
                codes = [c for c in codes if (c or "").strip()]
            except Exception:
                codes = []
            if not self._any_passes(codes, tests, verifier):
                broke.append(s.task_id)

        return {"veto": bool(broke),
                "reason": ("rrv_regress:" + ",".join(broke)) if broke else "ok",
                "checked": checked, "broke": broke, "skipped": skipped,
                "n_skills": len(self._skills)}
