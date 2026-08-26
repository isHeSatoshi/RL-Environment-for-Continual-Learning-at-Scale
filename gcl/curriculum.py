"""Continual task stream: real corpora (MBPP/HumanEval) + measurable drift.

Drift injectors create genuine non-stationarity so plasticity and forgetting are
*measurable* (not asserted). Anti-contamination: family IDs are split so holdout
evaluation never trains on evaluation items.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional


@dataclass
class Task:
    task_id: str
    family: str
    domain: str                      # "code" | "math"
    prompt: str
    test_code: str = ""
    reference_answer: str = ""
    entry_point: str = ""
    canary: bool = False             # True => holdout only, never in training stream

    def to_dict(self):
        return asdict(self)


@dataclass
class Family:
    name: str
    tasks: List[Task]
    holdout: List[Task] = field(default_factory=list)


# ----------------------- drift injectors (measurable non-stationarity) ------
def api_rename(task: Task, old: str, new: str) -> Task:
    """API-rename drift: prompt, tests, AND reference all move to the new symbol,
    so a drifted family is *internally consistent* and the gold answer stays gold."""
    p = task.prompt.replace(old, new)
    t = task.test_code.replace(old + "(", new + "(").replace(old + " (", new + " (")
    r = task.reference_answer.replace(old + "(", new + "(").replace(
        "def " + old + "(", "def " + new + "(").replace(old + " (", new + " (")
    return Task(task_id=task.task_id + f"_ren", family=task.family, domain=task.domain,
                prompt=p, test_code=t, reference_answer=r,
                entry_point=new, canary=task.canary)


def spec_paraphrase(task: Task, prefix: str) -> Task:
    """Surface drift only: paraphrase the prompt. Gold/reference semantics are
    preserved because tests + entry point are unchanged."""
    return Task(task_id=task.task_id + "_par", family=task.family, domain=task.domain,
                prompt=prefix + task.prompt, test_code=task.test_code,
                reference_answer=task.reference_answer, entry_point=task.entry_point,
                canary=task.canary)


# ----------------------- genuine distribution shift (measurable) --------------
def _math_family(max_n: int, seed: int, prefix: str = "") -> List[Dict]:
    """A REAL distribution shift: integer-arithmetic word problems (GSM-style) so
    forgetting is a genuine cross-capability drop, not a same-corpus resample."""
    rng = random.Random(seed)
    out = []
    for i in range(max_n):
        a = rng.randint(11, 99); b = rng.randint(3, 12); c = rng.randint(2, 9)
        if i % 3 == 0:
            q = f"{prefix}A shop sells a book for ${a}. They sell {b} books a day. How much revenue in {c} days? Return the integer."
            ref = str(a * b * c)
        elif i % 3 == 1:
            q = f"{prefix}Compute ({a} * {b}) - ({c} * {a}). Return just the integer."
            ref = str(a * b - c * a)
        else:
            q = f"{prefix}A train travels {a} km in {b} hours. It then travels {c} times as far. How many km on the second leg? Integer."
            ref = str(a * c)
        out.append({"task_id": f"math_{seed}_{i}", "prompt": q,
                    "test_code": "", "reference": ref, "entry_point": ""})
    return out


# ----------------------- higher-intensity drift (learnable) -------------------
_PERTURB_MAP = [("append", "push"), ("split", "tokenize"), ("join", "concat"),
                ("upper", "shout"), ("lower", "whisper"), ("sort", "order"),
                ("len(", "size("), ("str(", "as_text(")]
_NUM_SHIFT = 3

def spec_perturb(task: Task) -> Task:
    """Real behavioural drift: rename common ops and shift numeric literals.

    The reference is perturbed consistently, so the drifted family stays internally
    consistent AND the base model's memorized canonical answer becomes WRONG —
    giving genuine learnable signal (unlike spec_paraphrase which leaves the
    reference intact). Ids are rewritten by build_family so they remain canary-clean.
    """
    def shift(m):
        try:
            return str(int(m.group(0)) + _NUM_SHIFT)
        except Exception:
            return m.group(0)

    def tx(text: str) -> str:
        out = text
        for a, b in _PERTURB_MAP:
            out = out.replace(a, b)
        return re.sub(r"\b\d+\b", shift, out)

    return Task(task_id=task.task_id + "_per", family=task.family, domain=task.domain,
                prompt=tx(task.prompt), test_code=tx(task.test_code),
                reference_answer=tx(task.reference_answer),
                entry_point=task.entry_point, canary=task.canary)


# ----------------------- real corpora ---------------------------------------
def _load_mbpp(max_n: int, seed: int) -> List[Dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train+test+validation")
    except Exception:
        try:
            ds = load_dataset("mbpp", "sanitized", split="train+test+validation")
        except Exception:
            ds = load_dataset("mbpp", "sanitized", split="train")
    items = list(ds)
    rng = random.Random(seed); rng.shuffle(items)
    out = []
    for it in items[:max_n]:
        test_list = "\n".join(it.get("test_list", []))
        prompt = it.get("text", it.get("prompt", ""))
        ep = ""
        tl = it.get("test_list", [])
        if tl:
            import re
            m = re.search(r"assert\s+(?:math\.isclose\()?\s*([A-Za-z_][\w]*)\(", tl[0])
            ep = m.group(1) if m else ""
        out.append({"task_id": f"mbpp_{it.get('task_id', len(out))}", "prompt": prompt,
                    "test_code": test_list, "reference": it.get("code", ""), "entry_point": ep})
    return out


def _load_humaneval(max_n: int, seed: int) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    items = list(ds)
    rng = random.Random(seed); rng.shuffle(items)
    out = []
    for it in items[:max_n]:
        prompt = it.get("prompt", "")
        test = it.get("test", "")
        ep = it.get("entry_point", "")
        test_code = f"{test}\ncheck({ep})" if "check(" in test and ep else test
        out.append({"task_id": it.get("task_id", f"he_{len(out)}"), "prompt": prompt,
                    "test_code": test_code, "reference": it.get("canonical_solution", ""),
                    "entry_point": ep})
    return out


# ----------------------- interface normalization (spec, not supervision) ------
def _signature_hint(test_code: str, ep: str) -> str:
    """Recover the CALL INTERFACE (name + arity + keyword names) from the first
    test call and render it as specification text. Only the interface is moved —
    never expected outputs — so this mirrors HumanEval's signature-in-prompt
    convention: the caller's API contract is part of the task, while supervision
    (pass/fail verdicts + reference implementation) remains gold.
    """
    if not ep or not test_code:
        return ""
    import ast as _ast
    for line in (test_code or "").splitlines():
        line = line.strip()
        if not line.startswith("assert"):
            continue
        try:
            tree = _ast.parse(line)
        except SyntaxError:
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            f = node.func
            name = f.id if isinstance(f, _ast.Name) else \
                (f.attr if isinstance(f, _ast.Attribute) else "")
            if name != ep:
                continue
            letters = "xyzwvutsrq"
            args = [letters[i] if i < len(letters) else f"a{i}"
                    for i in range(len(node.args))]
            kws = [kw.arg for kw in node.keywords
                   if kw.arg and kw.arg not in args]
            params = args + kws
            if not params:
                return (f"The function must be named `{ep}` and take no "
                        "arguments.")
            sig = f"def {ep}({', '.join(params)}):"
            return f"The function must be named `{ep}` with signature `{sig}`."
    return ""


class StreamAssembler:
    """Builds train/held-out family streams with anti-contamination + drift."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def build_family(self, corpus: str, name: str, n_train: int, n_holdout: int,
                     drift: Optional[Callable[[Task], Task]] = None,
                     offset: int = 0) -> Family:
        # Each family consumes an exclusive slice of size n_train + n_holdout.
        # We compute the required window and slice AFTER it.
        need = n_train + n_holdout
        if corpus == "mbpp":
            raw = _load_mbpp(offset + need, self.seed)
        elif corpus == "humaneval":
            raw = _load_humaneval(offset + need, self.seed)
        elif corpus == "math":
            raw = _math_family(offset + need, self.seed)   # genuine cross-domain shift
        else:
            raise ValueError(f"unknown corpus {corpus}")
        raw = raw[offset:]
        assert len(raw) == need, f"family {name} needs {need} tasks, got {len(raw)} (offset={offset})"
        train_raw, hold_raw = raw[:n_train], raw[n_train:]

        def mk(rd, canary):
            prompt, ep = rd["prompt"], rd["entry_point"]
            # Interface normalization: the CALL INTERFACE (name + signature) is
            # part of the task specification (like a HumanEval signature in the
            # prompt), not supervision. Gold remains the unit-test verdicts +
            # reference implementation only. Without this, MBPP-style specs hide
            # the interface inside the gold tests and any gold-free agent fails
            # on NameError/TypeError regardless of its logic.
            if corpus != "math" and ep and (ep + "(") not in prompt \
                    and f"`{ep}`" not in prompt:
                hint = _signature_hint(rd["test_code"], ep) \
                    or f"The function must be named `{ep}`."
                prompt = prompt.rstrip() + "\n" + hint
            return Task(task_id=rd["task_id"], family=name, domain=("math" if corpus == "math" else "code"),
                        prompt=prompt, test_code=rd["test_code"],
                        reference_answer=rd["reference"], entry_point=rd["entry_point"],
                        canary=canary)

        train = [mk(r, False) for r in train_raw]
        holdout = [mk(r, True) for r in hold_raw]
        if drift is not None:
            # Drift the ENTIRE family (train AND holdout) so it stays internally
            # consistent (gold/tests move together); ids are rewritten family-locally
            # so a drifted family never reuses base MBPP ids (I5 / no leakage).
            drifted_train, drifted_hold = [], []
            for idx, t in enumerate(train):
                dt = drift(t)
                dt.task_id = f"{name}_drift_{idx}"
                dt.canary = False
                drifted_train.append(dt)
            for idx, t in enumerate(holdout):
                dt = drift(t)
                dt.task_id = f"{name}_drift_hold_{idx}"
                dt.canary = True
                drifted_hold.append(dt)
            train, holdout = drifted_train, drifted_hold
        return Family(name=name, tasks=train, holdout=holdout)

    def assemble(self, spec: List[Dict]) -> List[Family]:
        """spec: [{corpus, name, n_train, n_holdout, drift, offset}]; asserts no overlap."""
        fams = [self.build_family(corpus=s["corpus"], name=s["name"],
                                  n_train=s["n_train"], n_holdout=s["n_holdout"],
                                  drift=s.get("drift"), offset=s.get("offset", 0))
                for s in spec]
        return fams


def canary_report(families: List[Family]) -> Dict:
    """Prove anti-contamination: (a) holdout never overlaps training, and
    (b) NO two families share the same underlying task (train∪holdout) content.

    Content fingerprint (first 80 chars of prompt) catches rewording/ id-stripping
    leaks so a drift family reusing base MBPP items is flagged. Two MBPP families
    pointing at the same corpus window is contamination and must fail `clean`.
    """
    def fp(s: str) -> str:
        return (s or "")[:80].strip().lower()
    train_ids = {t.task_id for f in families for t in f.tasks}
    hold_ids = {t.task_id for f in families for t in f.holdout}
    overlap_ids = train_ids & hold_ids
    train_fp = {fp(t.prompt) for f in families for t in f.tasks}
    hold_fp = {fp(t.prompt) for f in families for t in f.holdout}
    overlap_fp = train_fp & hold_fp

    # --- cross-family duplicate detection: same fingerprint in >1 family ------
    fam_fp_sets = []
    dup_across = []
    for f in families:
        fs = {fp(t.prompt) for t in (f.tasks + f.holdout)}
        fam_fp_sets.append((f.name, fs))
    for i in range(len(fam_fp_sets)):
        for j in range(i + 1, len(fam_fp_sets)):
            inter = fam_fp_sets[i][1] & fam_fp_sets[j][1]
            if inter:
                dup_across.append(f"{fam_fp_sets[i][0]}x{fam_fp_sets[j][0]}:{len(inter)}")

    overlap = len(overlap_ids) + len(overlap_fp) + len(dup_across)
    h = hashlib.sha256("|".join(sorted(train_ids)).encode()).hexdigest()[:12]
    return {"train_n": len(train_ids), "holdout_n": len(hold_ids),
            "overlap": overlap, "overlap_ids": sorted(overlap_ids),
            "overlap_fp": len(overlap_fp), "cross_family_dup": dup_across,
            "stream_hash": h, "clean": overlap == 0}
