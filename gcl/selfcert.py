"""Self-Certified Continual Learning (SCCL) — gold-free verification core.

This module closes the gold gap left by the earlier "no-gold" path: the
self-taught learner still verified candidates against `task.test_code` (gold
unit tests) and used a gold-parsed entry point. Here the model manufactures its
OWN executable ground truth from the SPECIFICATION ALONE:

  1. SELF-SPEC TESTS     — sample independent bags of literal `assert` tests
                           generated purely from the natural-language spec.
  2. DISCRIMINATIVE FILTER — keep only tests that split the candidate pool
                           (tests everyone passes/no one passes carry no signal).
                           Exception: a UNANIMOUS pool (every candidate passes
                           every test) is itself evidence — it certifies already-
                           mastered skills so the vault can protect them.
  3. CONSENSUS CERTIFICATION — confidence = discriminative pass rate weighted
                           by cross-bag agreement; certify if above tau.
  4. SELF-REPLAY VETO    — after each gradient update, regenerate previously
                           certified skills and re-run THEIR OWN self-tests;
                           regression vetoes/rolls back the update (gold-free
                           replacement of the holdout-epsilon safety gate).

Gold-free guarantee is STRUCTURAL: every public function takes only
(spec, entry, domain) — never a Task object — so `task.test_code` and
`task.reference_answer` are unreachable from this module by construction.
For math, certification is majority-vote self-consistency (no gold answer).
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Prompt builders (spec-only inputs — no gold fields)
# ---------------------------------------------------------------------------

_ENTRY_PROMPT = (
    "A Python function must satisfy this specification:\n{spec}\n\n"
    "Propose a short snake_case name for it. Reply with ONLY the function "
    "definition header, e.g.\n"
    "def my_function(x):\n"
)

_TESTGEN_PROMPT = (
    "You are writing unit tests for a Python function.\n"
    "The function specification is:\n{spec}\n\n"
    "The function is named `{entry}`.\n"
    "Write exactly {n} independent `assert` statements that a CORRECT "
    "implementation must pass.{variant}\n"
    "Rules:\n"
    "- Use ONLY concrete literal inputs and expected outputs "
    "(numbers, strings, lists, tuples, booleans).\n"
    "- Each assert on its own line, starting with the word `assert`.\n"
    "- Derive expected outputs strictly from the specification.\n"
    "- Do NOT define the function. No prose, no code fences. "
    "Output only the assert lines.\n"
)

_SOLUTION_PROMPT_CODE = (
    "You are an expert Python programmer. Write ONLY the complete function "
    "inside a ```python block. No analysis, no empty 'pass' stubs."
    " The function MUST be named `{entry}`.\n{spec}\n```python\n"
)

_SOLUTION_PROMPT_MATH = (
    "Solve and give ONLY the final numeric answer.\n{spec}\nAnswer: "
)

# ---- SCCL v2: neighborhood-probe generation (spec only — no gold fields) ----
_PARAPHRASE_PROMPT = (
    "Rewrite the following programming task specification in different words. "
    "Preserve EVERY requirement: the same inputs, the same outputs, the same "
    "constraints and edge cases. Change only the wording, not the meaning. "
    "Do not solve the task. Output ONLY the rewritten specification.\n{spec}\n"
)

_MATH_VARIANT_PROMPT = (
    "Here is a math word problem:\n{spec}\n\n"
    "Write ONE similar word problem of the same type using DIFFERENT numbers. "
    "Keep it solvable with the same method. Output ONLY the new problem text.\n"
)


def sccl_prompt(spec: str, entry: str, domain: str = "code") -> str:
    """Solution prompt built ONLY from spec + self-derived entry name."""
    if domain == "math":
        return _SOLUTION_PROMPT_MATH.format(spec=spec)
    return _SOLUTION_PROMPT_CODE.format(spec=spec, entry=entry or "solution")

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.FunctionDef,
                    ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
                    ast.AnnAssign, ast.Global, ast.Nonlocal)
_FORBIDDEN_CALLS = {"eval", "exec", "open", "input", "__import__", "compile",
                    "globals", "locals", "getattr", "setattr", "vars"}


def parse_entry_name(text: str, fallback: str = "solution") -> str:
    """Extract a proposed function name from `def name(...)` text."""
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", text or "")
    return m.group(1) if m else fallback


def parse_asserts(text: str, entry: str, max_tests: int = 8) -> List[str]:
    """Extract safe, executable, entry-calling assert lines from model output.

    A line is kept only if it: starts with `assert`, references `{entry}(`,
    parses as a single ast.Assert statement, and contains no definitions,
    imports, assignments, or dangerous calls. Duplicates are dropped.
    """
    out: List[str] = []
    seen = set()
    if not entry:
        return out
    call_pat = re.compile(r"\b" + re.escape(entry) + r"\s*\(")
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("`-* ").strip()
        if not line.startswith("assert"):
            continue
        if not call_pat.search(line):
            continue
        try:
            tree = ast.parse(line)
        except SyntaxError:
            continue
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
            continue
        bad = False
        for node in ast.walk(tree):
            if isinstance(node, _FORBIDDEN_NODES):
                bad = True
                break
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name) and f.id in _FORBIDDEN_CALLS:
                    bad = True
                    break
        if bad:
            continue
        key = " ".join(line.split())
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= max_tests:
            break
    return out


def extract_number(text: str) -> Optional[str]:
    """GSM-style numeric extraction (matches verifier._eval_math convention)."""
    if not text:
        return None
    ex = ""
    if "####" in text:
        ex = text.split("####")[-1].strip().splitlines()[0].strip().lower()
    elif "\boxed{" in text:
        try:
            ex = text.split("\boxed{")[1].split("}")[0].strip().lower()
        except IndexError:
            ex = ""
    if not ex:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        ex = nums[-1] if nums else ""
    ex = ex.replace(",", "").strip().rstrip(".")
    if not ex:
        return None
    try:
        float(ex)
    except ValueError:
        return None
    return ex


def format_number(v: str) -> str:
    """Canonical integer formatting when the value is integral."""
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (ValueError, OverflowError):
        return str(v)


# ---------------------------------------------------------------------------
# Certification result
# ---------------------------------------------------------------------------

@dataclass
class CertResult:
    found: bool = False
    code: str = ""
    confidence: float = 0.0
    domain: str = "code"
    entry: str = ""
    n_candidates: int = 0
    n_tests: int = 0
    n_discriminative: int = 0
    pass_rate: float = 0.0
    bag_agreement: float = 0.0
    margin: float = 0.0
    self_tests: List[str] = field(default_factory=list)   # certifying suite
    prompt: str = ""                                       # regeneration prompt
    verdicts: List[List[int]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verdicts"] = [list(map(int, v)) for v in self.verdicts]
        return d


# ---------------------------------------------------------------------------
# The certifier
# ---------------------------------------------------------------------------

class SelfCertifier:
    """Gold-free certification via self-spec tests + consensus.

    The certifier never receives a Task object — only (spec, entry, domain) —
    so no gold field is reachable from here by construction.
    """

    def __init__(self, k: int = 6, temp: float = 0.8, test_bags: int = 2,
                 tests_per_bag: int = 4, tau: float = 0.65, tau_math: float = 0.6,
                 max_tests: int = 8, weak_cap: float = 0.5,
                 margin_bonus: float = 0.05, margin_min: float = 0.15,
                 consensus: bool = True):
        self.k = max(2, int(k))
        self.temp = float(temp)
        self.test_bags = max(1, int(test_bags))
        self.tests_per_bag = max(2, int(tests_per_bag))
        self.tau = float(tau)
        self.tau_math = float(tau_math)
        self.max_tests = int(max_tests)
        self.weak_cap = float(weak_cap)
        self.margin_bonus = float(margin_bonus)
        self.margin_min = float(margin_min)
        self.consensus = bool(consensus)  # ablation: sccl_nocons

    # ---- entry-name derivation (spec only) ---------------------------------
    def propose_entry(self, engine, spec: str) -> str:
        # Fast path: when the specification itself names the interface (e.g.
        # "The function must be named `foo`." or a def header), read it
        # directly — still spec-only, no gold fields involved.
        m = re.search(r"named\s+`([A-Za-z_]\w*)`", spec or "")
        if m:
            return m.group(1)
        m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", spec or "")
        if m:
            return m.group(1)
        txt = engine.generate(_ENTRY_PROMPT.format(spec=spec), adapter_on=True,
                              greedy=True)
        return parse_entry_name(txt)

    # ---- self-test bag generation (spec only) ------------------------------
    def generate_test_bags(self, engine, spec: str, entry: str) -> List[List[str]]:
        bags: List[List[str]] = []
        seen = set()
        for b in range(self.test_bags):
            variant = "" if b == 0 else f" (variation {b}: prefer different inputs/cases)"
            prompt = _TESTGEN_PROMPT.format(spec=spec, entry=entry,
                                            n=self.tests_per_bag, variant=variant)
            txt = engine.generate(prompt, adapter_on=True, greedy=(b == 0))
            tests = parse_asserts(txt, entry, max_tests=self.max_tests)
            kept = [t for t in tests if " ".join(t.split()) not in seen]
            for t in kept:
                seen.add(" ".join(t.split()))
            if kept:
                bags.append(kept)
        return bags

    # ---- verdict vectors: run one candidate against a test set -------------
    def _verdicts(self, verifier, code: str, tests: List[str]) -> Tuple[List[int], float]:
        if not tests or not (code or "").strip():
            return [0] * len(tests), 0.0
        try:
            _, info, res = verifier.reward(domain="code", code=code,
                                           test_code="\n".join(tests))
        except Exception:
            return [0] * len(tests), 0.0
        mask = list(getattr(res, "test_results", []) or []) if res is not None else []
        if len(mask) != len(tests):
            # rebuild from the aggregate when per-assert vectors are unavailable
            passed = int(info.get("tests_passed", 0))
            mask = ([1] * passed) + ([0] * (len(tests) - passed))
        v = [1 if x else 0 for x in mask[:len(tests)]]
        pr = float(sum(v)) / max(1, len(v))
        return v, pr

    # ---- code certification -------------------------------------------------
    def certify_code(self, engine, verifier, spec: str,
                     entry: Optional[str] = None) -> CertResult:
        from .engine import extract_code  # lazy: keeps module importable w/o torch
        if not entry:
            entry = self.propose_entry(engine, spec)
        sol_prompt = sccl_prompt(spec, entry, "code")

        # 1) diverse candidate pool: temperature samples + one greedy
        raw = list(engine.sample_candidates(sol_prompt, n=self.k,
                                            temperature=self.temp))
        raw.append(engine.generate(sol_prompt, adapter_on=True, greedy=True))
        pool: List[str] = []
        seen = set()
        for r in raw:
            code = extract_code(r)
            if not code.strip():
                continue
            key = " ".join(code.split())
            if key in seen:
                continue
            seen.add(key)
            pool.append(code)
        if not pool:
            return CertResult(domain="code", entry=entry, prompt=sol_prompt,
                              diagnostics={"fail": "no_candidates"})

        # 2) self-test bags
        bags = self.generate_test_bags(engine, spec, entry)
        all_tests: List[str] = [t for bag in bags for t in bag]
        if not all_tests:
            return CertResult(domain="code", entry=entry, prompt=sol_prompt,
                              n_candidates=len(pool),
                              diagnostics={"fail": "no_tests_generated"})

        # 3) verdict matrix: candidates x tests
        verdicts: List[List[int]] = []
        pass_rates: List[float] = []
        for code in pool:
            v, pr = self._verdicts(verifier, code, all_tests)
            verdicts.append(v)
            pass_rates.append(pr)

        # 4) discriminative filter: a test must split the pool
        n_sols = len(pool)
        col_sums = [sum(verdicts[i][j] for i in range(n_sols))
                    for j in range(len(all_tests))]
        disc_idx = [j for j, s in enumerate(col_sums) if 0 < s < n_sols]
        # UNANIMOUS POOL: every candidate passes every self-test. No test can
        # discriminate, but the consensus itself is the evidence — certify so
        # already-mastered skills enter the vault and arm the self-replay veto.
        # Distinct from unanimous FAILURE (nothing passes anywhere), which
        # stays weak and capped.
        unanimous = (not disc_idx) and len(all_tests) >= 2 \
            and all(s == n_sols for s in col_sums)
        weak = (not disc_idx) and not unanimous
        if not disc_idx:
            disc_idx = list(range(len(all_tests)))  # fallback; weak capped below

        # 5) consensus: best per-bag discriminative pass rates must agree
        bag_bounds: List[Tuple[int, int]] = []
        lo = 0
        for bag in bags:
            hi = lo + len(bag)
            bag_bounds.append((lo, hi))
            lo = hi
        bag_rates: List[float] = []
        for (a, b) in bag_bounds:
            idx = [j for j in disc_idx if a <= j < b]
            if not idx:
                continue
            best_bag = max(sum(verdicts[i][j] for j in idx) / len(idx)
                           for i in range(n_sols))
            bag_rates.append(best_bag)
        agreement = (1.0 - (max(bag_rates) - min(bag_rates))
                     if len(bag_rates) >= 2 else 1.0)

        # 6) score each candidate on discriminative tests
        scores: List[float] = []
        for i in range(n_sols):
            pr_disc = sum(verdicts[i][j] for j in disc_idx) / max(1, len(disc_idx))
            score = pr_disc * (0.7 + 0.3 * agreement) if self.consensus else pr_disc
            scores.append(score)
        order = sorted(range(n_sols), key=lambda i: scores[i], reverse=True)
        best_i = order[0]
        margin = scores[order[0]] - scores[order[1]] if n_sols > 1 else 0.0
        conf = scores[best_i] + (self.margin_bonus if margin >= self.margin_min else 0.0)
        if weak:
            conf = min(conf, self.weak_cap)
        conf = float(min(1.0, max(0.0, conf)))

        # 7) certifying suite = discriminative tests the winner passes
        winner_tests = [all_tests[j] for j in disc_idx if verdicts[best_i][j] == 1]
        found = conf >= self.tau and len(winner_tests) > 0

        return CertResult(found=found, code=pool[best_i], confidence=conf,
                          domain="code", entry=entry, n_candidates=n_sols,
                          n_tests=len(all_tests),
                          n_discriminative=0 if (weak or unanimous) else len(disc_idx),
                          pass_rate=pass_rates[best_i], bag_agreement=agreement,
                          margin=margin, self_tests=winner_tests, prompt=sol_prompt,
                          verdicts=verdicts,
                          diagnostics={"weak_pool": weak, "unanimous": unanimous,
                                       "scores": [round(s, 3) for s in scores],
                                       "col_sums": col_sums})

    # ---- math certification: majority-vote self-consistency -----------------
    def certify_math(self, engine, spec: str) -> CertResult:
        sol_prompt = sccl_prompt(spec, "", "math")
        raw = list(engine.sample_candidates(sol_prompt, n=self.k,
                                            temperature=self.temp))
        raw.append(engine.generate(sol_prompt, adapter_on=True, greedy=True))
        votes: Dict[str, int] = {}
        for r in raw:
            a = extract_number(r)
            if a is not None:
                key = format_number(a)
                votes[key] = votes.get(key, 0) + 1
        if not votes:
            return CertResult(domain="math", prompt=sol_prompt,
                              n_candidates=len(raw),
                              diagnostics={"fail": "no_numeric_answers"})
        best_val, count = max(votes.items(), key=lambda kv: kv[1])
        counts = sorted(votes.values(), reverse=True)
        margin = (counts[0] - counts[1]) / max(1, len(raw)) if len(counts) > 1 else 1.0
        conf = float(count) / max(1, len(raw))
        found = conf >= self.tau_math
        return CertResult(found=found, code=best_val, confidence=conf, domain="math",
                          entry="", n_candidates=len(raw), n_tests=sum(votes.values()),
                          pass_rate=conf, margin=margin, self_tests=[],
                          prompt=sol_prompt, diagnostics={"votes": votes})

    # ---- dispatcher ----------------------------------------------------------
    def certify(self, engine, verifier, spec: str, domain: str,
                entry: Optional[str] = None) -> CertResult:
        if domain == "math":
            return self.certify_math(engine, spec)
        return self.certify_code(engine, verifier, spec, entry=entry)

    # ---- neighborhood probes (SCCL v2, spec only) -----------------------------
    @staticmethod
    def _passes_tests(verifier, code: str, tests: List[str]) -> bool:
        if not (code or "").strip() or not tests:
            return False
        try:
            _, info, _ = verifier.reward(domain="code", code=code,
                                         test_code="\n".join(tests))
            return (float(info.get("pass_rate", 0.0)) >= 1.0
                    and bool(info.get("success", False)))
        except Exception:
            return False

    def make_probe(self, engine, verifier, spec: str, domain: str,
                   base: CertResult) -> Optional[Dict[str, Any]]:
        """Manufacture a certified NEIGHBORHOOD probe for a certified skill.

        Gold-free by construction: inputs are (spec, domain, base CertResult) —
        no task object, no gold field. The probe gives the Self-Replay Veto a
        notion of GENERALIZATION: instead of only asking "can the model still
        re-express the exact skill it trained on?", RRV also asks "can it still
        solve a nearby task it never trained on?".

        code : paraphrase the spec, re-certify the paraphrase, and keep the
               probe only under BIDIRECTIONAL cross-validation — the base
               winner passes the probe's self-tests AND the probe winner
               passes the base's self-tests. Agreement in both directions is
               the strongest spec-only evidence that the paraphrase preserved
               semantics.
        math : sample a numeric variant of the problem (different numbers,
               same method) and certify its answer by majority vote.
        Returns a probe dict (spec/prompt/code/self_tests/...) or None.
        """
        if domain == "math":
            variant = engine.generate(_MATH_VARIANT_PROMPT.format(spec=spec),
                                      adapter_on=True, greedy=False)
            variant = (variant or "").strip()
            if len(variant) < 20:
                return None
            cr = self.certify_math(engine, variant)
            if not cr.found:
                return None
            return {"kind": "probe", "domain": "math", "spec": variant,
                    "prompt": cr.prompt, "code": cr.code, "self_tests": [],
                    "entry": "", "confidence": cr.confidence}

        para = engine.generate(_PARAPHRASE_PROMPT.format(spec=spec),
                               adapter_on=True, greedy=False)
        para = (para or "").strip()
        if len(para) < 20 or " ".join(para.split()) == " ".join((spec or "").split()):
            return None
        entry = base.entry or ""
        if entry and f"named `{entry}`" not in para:
            para = para + f" The function must be named `{entry}`."
        cr = self.certify_code(engine, verifier, para, entry=entry)
        if not cr.found or not cr.self_tests or not base.self_tests:
            return None
        # bidirectional cross-validation (spec-only evidence of same semantics)
        if not self._passes_tests(verifier, base.code, cr.self_tests):
            return None
        if not self._passes_tests(verifier, cr.code, base.self_tests):
            return None
        return {"kind": "probe", "domain": "code", "spec": para,
                "prompt": cr.prompt, "code": cr.code,
                "self_tests": list(cr.self_tests), "entry": entry,
                "confidence": cr.confidence}
