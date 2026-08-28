"""SCCL (Self-Certified Continual Learning) mechanics tests — torch-free.

Validates the gold-free contract on the CPU interpreter using stub engines and
the REAL sandbox verifier. Two layers of proof:

  1. STATIC: gcl/selfcert.py must never touch gold fields (no attribute access
     to test_code / reference_answer / entry_point / test_list on any object).
  2. DYNAMIC: with POISONED gold (gold tests assert wrong behaviour), the SCCL
     env path must follow self-certification metadata — training when certified,
     refusing when not, and vetoing via Self-Replay regardless of gold reward.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcl.selfcert import (SelfCertifier, CertResult, parse_asserts, parse_entry_name,
                          extract_number, format_number, sccl_prompt)
from gcl.vault import SelfCertVault
from gcl.verify import Verifier
from gcl.sandbox import PythonSandbox
from gcl.config import ExperimentConfig
from gcl.curriculum import Task
from gcl.env import GroundedContinualEnv, Action, LearnOp

GOLD_FIELDS = {"test_code", "reference_answer", "entry_point", "test_list", "reference"}


# ---------------------------------------------------------------------------
# 1) STATIC gold-free guarantee
# ---------------------------------------------------------------------------

def test_selfcert_module_never_touches_gold_fields():
    """selfcert.py must not read gold fields off any object (structural proof)."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "gcl", "selfcert.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in GOLD_FIELDS:
            violations.append(f"line {node.lineno}: .{node.attr}")
    assert not violations, f"selfcert.py touches gold fields: {violations}"


def test_certifier_api_is_spec_only():
    """Every public certifier entry point takes (spec, domain, entry) — no Task."""
    import inspect
    for fn in (SelfCertifier.certify, SelfCertifier.certify_code,
               SelfCertifier.certify_math, SelfCertifier.propose_entry,
               SelfCertifier.generate_test_bags, SelfCertifier.make_probe,
               SelfCertifier.check_neighborhood):
        params = list(inspect.signature(fn).parameters)
        assert "task" not in params, f"{fn.__name__} accepts a task -> gold leak risk"


# ---------------------------------------------------------------------------
# 2) Parsing helpers
# ---------------------------------------------------------------------------

def test_parse_asserts_filters_and_dedupes():
    txt = """Here are tests:
```python
assert add2(1, 2) == 3
assert add2(0, 0) == 0
assert add2(1,2) == 3
assert other(1) == 1
import os; assert add2(2, 2) == 4
x = 1
assert eval("add2(1,1)") == 2
def add2(a, b): return a + b
assert add2(-1, 1) == 0
```
"""
    out = parse_asserts(txt, "add2")
    assert "assert add2(1, 2) == 3" in out
    assert "assert add2(0, 0) == 0" in out
    assert "assert add2(-1, 1) == 0" in out
    # no duplicate, no foreign entry, no import/eval/def lines
    assert len(out) == len(set(out))
    assert not any("other(" in t for t in out)
    assert not any("import" in t or "eval" in t or t.startswith("def") for t in out)
    assert parse_asserts(txt, "add2", max_tests=2) == out[:2]
    assert parse_asserts(txt, "") == []


def test_parse_entry_name():
    assert parse_entry_name("def next_square(n):\n    return n") == "next_square"
    assert parse_entry_name("I propose\n```python\ndef add2(a,b):\n```") == "add2"
    assert parse_entry_name("no def here", fallback="solution") == "solution"


def test_extract_and_format_number():
    assert extract_number("The total is 42.") == "42"
    assert extract_number("#### 1,234") == "1234"
    assert extract_number("\\boxed{17}") == "17"
    assert extract_number("no numbers") is None
    assert format_number("42.0") == "42"
    assert format_number("3.5") == "3.5"


def test_sccl_prompt_uses_only_spec_and_entry():
    p = sccl_prompt("Return the sum.", "add2", "code")
    assert "add2" in p and "Return the sum." in p
    m = sccl_prompt("What is 2+2?", "", "math")
    assert "ONLY the final numeric answer" in m


def test_interface_name_is_in_spec_and_propose_fast_path():
    """MBPP specs must carry the interface name (HumanEval-style), and
    propose_entry must read it from the spec without calling the model."""
    from gcl.curriculum import StreamAssembler
    fam = StreamAssembler(seed=42).build_family("mbpp", "arith", 2, 1, offset=0)
    for t in fam.tasks + fam.holdout:
        if t.entry_point:
            assert f"`{t.entry_point}`" in t.prompt, t.task_id
    sig_tasks = [t for t in fam.tasks + fam.holdout
                 if t.entry_point and "signature" in t.prompt]
    assert sig_tasks, "expected at least one MBPP task with a signature hint"

    class _NoGen:
        def generate(self, *a, **k):
            raise AssertionError("fast path must not call the model")

    t0 = fam.tasks[0]
    ep = SelfCertifier().propose_entry(_NoGen(), t0.prompt)
    assert ep == t0.entry_point
    # math prompts carry no function name
    mfam = StreamAssembler(seed=42).build_family("math", "m", 2, 1, offset=0)
    assert "`" not in mfam.tasks[0].prompt


def test_signature_hint_recovers_interface_only():
    """The hint moves only the CALL INTERFACE into the spec — name, arity and
    keyword names; never expected outputs."""
    from gcl.curriculum import _signature_hint
    assert _signature_hint("assert f(1, 2) == 3\nassert f(0, 0) == 0", "f") == \
        "The function must be named `f` with signature `def f(x, y):`."
    assert "no arguments" in _signature_hint("assert g() == 1", "g")
    assert "def h(x, y):" in _signature_hint("assert h(x=1, y=2) == 3", "h")
    # nested wrapper calls still resolve the entry-point call
    assert "def p(x):" in _signature_hint("assert math.isclose(p(0.5), 0.5)", "p")
    assert _signature_hint("", "f") == ""
    assert _signature_hint("assert f(1) == 1", "") == ""


# ---------------------------------------------------------------------------
# Stub engine: canned generations routed by prompt content
# ---------------------------------------------------------------------------

class StubEngine:
    """Returns scripted outputs. `mode` selects candidate quality for tests."""

    def __init__(self):
        self.test_bags = [
            ["assert add2(1, 2) == 3", "assert add2(0, 0) == 0"],
            ["assert add2(-1, 1) == 0", "assert add2(2, 3) == 5"],
        ]
        self.candidates = [
            "```python\ndef add2(a, b):\n    return a + b\n```",          # correct
            "```python\ndef add2(a, b):\n    return a - b\n```",          # wrong
            "```python\ndef add2(a, b):\n    return 7\n```",              # wrong
        ]
        self.greedy_solution = self.candidates[0]
        self.math_answers = ["42", "The answer is 42", "#### 42", "41", "42", "42", "43"]
        self._bag_calls = 0
        self.regen_code = None  # for RRV veto tests

    def generate(self, prompt, adapter_on=True, greedy=False):
        if "Propose a short snake_case name" in prompt:
            return "def add2(a, b):"
        if "writing unit tests" in prompt:
            bag = self.test_bags[min(self._bag_calls, len(self.test_bags) - 1)]
            self._bag_calls += 1
            return "\n".join(bag)
        if "ONLY the final numeric answer" in prompt:
            return self.math_answers[0]
        return self.greedy_solution

    def sample_candidates(self, prompt, n, temperature=0.7, top_p=0.95,
                          adapter_on=True, max_new_tokens=None):
        if "ONLY the final numeric answer" in prompt:
            return self.math_answers[:n]
        if self.regen_code is not None:
            return [self.regen_code] * n
        return self.candidates[:max(1, n)]


# ---------------------------------------------------------------------------
# 3) Certification logic (real sandbox execution)
# ---------------------------------------------------------------------------

def test_certify_code_finds_correct_candidate():
    sc = SelfCertifier(k=3, temp=0.8, test_bags=2, tests_per_bag=2, tau=0.65)
    eng, ver = StubEngine(), Verifier(sandbox=PythonSandbox())
    cr = sc.certify_code(eng, ver, "Return the sum of two integers a and b.",
                         entry="add2")
    assert cr.found, f"expected certification, got conf={cr.confidence} diag={cr.diagnostics}"
    assert "a + b" in cr.code
    assert cr.confidence >= 0.65
    assert cr.n_candidates == 3
    assert cr.n_tests == 4
    assert cr.n_discriminative >= 1
    assert cr.self_tests, "certifying suite must be non-empty"
    # the certifying suite must actually pass the winner (executed, not assumed)
    _, info, _ = ver.reward(domain="code", code=cr.code,
                            test_code="\n".join(cr.self_tests))
    assert info["pass_rate"] >= 1.0


def test_certify_code_refuses_weak_pool():
    """Candidates that fail EVERY self-test form a weak pool: confidence is
    capped below tau, so nothing is certified (no training on garbage)."""
    sc = SelfCertifier(k=3, temp=0.8, test_bags=2, tests_per_bag=2, tau=0.65)
    eng, ver = StubEngine(), Verifier(sandbox=PythonSandbox())
    eng.candidates = ["```python\ndef add2(a, b):\n    return 99\n```",
                      "```python\ndef add2(a, b):\n    return 7\n```"]
    eng.greedy_solution = eng.candidates[0]
    cr = sc.certify_code(eng, ver, "Return the sum of two integers a and b.",
                         entry="add2")
    assert not cr.found
    assert cr.confidence < 0.65
    assert cr.diagnostics.get("weak_pool") is True


def test_certify_code_unanimous_pool_certifies():
    """When EVERY candidate passes EVERY self-test there is nothing left to
    discriminate, but the unanimous consensus is itself evidence. Mastered
    skills must certify so the vault can protect them via self-replay veto."""
    sc = SelfCertifier(k=3, temp=0.8, test_bags=2, tests_per_bag=2, tau=0.65)
    eng, ver = StubEngine(), Verifier(sandbox=PythonSandbox())
    eng.candidates = ["```python\ndef add2(a, b):\n    return a + b\n```",
                      "```python\ndef add2(a, b):\n    return b + a\n```"]
    eng.greedy_solution = eng.candidates[0]
    cr = sc.certify_code(eng, ver, "Return the sum of two integers a and b.",
                         entry="add2")
    assert cr.found, f"unanimous pool must certify: conf={cr.confidence} diag={cr.diagnostics}"
    assert cr.diagnostics.get("unanimous") is True
    assert cr.diagnostics.get("weak_pool") is False
    assert cr.confidence >= 0.65
    assert cr.n_discriminative == 0
    assert len(cr.self_tests) == cr.n_tests  # the whole suite certifies
    _, info, _ = ver.reward(domain="code", code=cr.code,
                            test_code="\n".join(cr.self_tests))
    assert info["pass_rate"] >= 1.0


def test_certify_code_nocons_ablation_scores_higher_or_equal():
    """Without consensus weighting the raw discriminative rate is used."""
    ver = Verifier(sandbox=PythonSandbox())
    cr_cons = SelfCertifier(k=3, test_bags=2, consensus=True).certify_code(
        StubEngine(), ver, "Return the sum of two integers a and b.", entry="add2")
    cr_nocons = SelfCertifier(k=3, test_bags=2, consensus=False).certify_code(
        StubEngine(), ver, "Return the sum of two integers a and b.", entry="add2")
    assert cr_nocons.bag_agreement == cr_cons.bag_agreement
    # winner identical; consensus can only shrink or equal the score (factor <= 1)
    assert "a + b" in cr_nocons.code
    assert cr_nocons.confidence >= cr_cons.confidence - 1e-9


def test_certify_math_majority_vote():
    sc = SelfCertifier(k=6, tau_math=0.6)
    eng = StubEngine()
    cr = sc.certify_math(eng, "What is 6 times 7?")
    assert cr.found
    assert cr.code == "42"
    # 6 samples + 1 greedy = 7 answers; 6 of the scripted answers say 42
    assert abs(cr.confidence - 6 / 7) < 1e-6
    assert cr.diagnostics["votes"]["42"] == 6
    assert cr.diagnostics["votes"]["41"] == 1


def test_certify_dispatcher():
    ver = Verifier(sandbox=PythonSandbox())
    cr_m = SelfCertifier(k=6).certify(StubEngine(), ver, "What is 6 times 7?", "math")
    assert cr_m.domain == "math"
    cr_c = SelfCertifier(k=3).certify(StubEngine(), ver, "Return the sum.", "code",
                                      entry="add2")
    assert cr_c.domain == "code"


# ---------------------------------------------------------------------------
# 4) SelfCertVault + Self-Replay Veto
# ---------------------------------------------------------------------------

_CERT_CODE = "def add2(a, b):\n    return a + b"
_CERT_TESTS = ["assert add2(1, 2) == 3", "assert add2(0, 0) == 0"]


def test_commit_certified_and_to_pairs():
    v = SelfCertVault()
    ok = v.commit_certified(task_id="t1", family="f", spec="Return the sum.",
                            prompt="You are an expert Python programmer.\nReturn the sum.\n```python\n",
                            code=_CERT_CODE, self_tests=_CERT_TESTS, conf=0.9)
    assert ok and len(v) == 1
    # empty suite or empty code is refused
    assert not v.commit_certified("t2", "f", "s", "p", _CERT_CODE, [], 0.9)
    assert not v.commit_certified("t3", "f", "s", "p", "", _CERT_TESTS, 0.9)
    pairs = v.to_pairs()
    assert pairs[0]["target"] == _CERT_CODE and "```python" in pairs[0]["prompt"]


def test_commit_certified_dedup():
    # Stored embedding is embed(spec + code), so use a threshold below the
    # spec-vs-(spec+code) similarity of near-identical specs.
    v = SelfCertVault()
    assert v.commit_certified("t1", "f", "Return the sum of two ints.", "p",
                              _CERT_CODE, _CERT_TESTS, 0.9, dedup_sim=0.5)
    # near-identical spec is deduped
    assert not v.commit_certified("t2", "f", "Return the sum of two ints.", "p",
                                  _CERT_CODE, _CERT_TESTS, 0.9, dedup_sim=0.5)
    assert len(v) == 1


def test_rrv_veto_passes_when_skill_regenerates():
    v, ver, eng = SelfCertVault(), Verifier(sandbox=PythonSandbox()), StubEngine()
    v.commit_certified("t1", "f", "Return the sum.", "PROMPT fid:t1\n```python\n",
                       _CERT_CODE, _CERT_TESTS, 0.9)
    eng.regen_code = "```python\ndef add2(a, b):\n    return a + b\n```"
    res = v.selfreplay_veto(eng, ver, check_skills=3, n_samples=2)
    assert not res["veto"], res
    assert res["checked"] == 1


def test_rrv_veto_fires_when_skill_lost():
    """The gate must NOT fall back to the stored artifact: regeneration only."""
    v, ver, eng = SelfCertVault(), Verifier(sandbox=PythonSandbox()), StubEngine()
    v.commit_certified("t1", "f", "Return the sum.", "PROMPT fid:t1\n```python\n",
                       _CERT_CODE, _CERT_TESTS, 0.9)
    eng.regen_code = "```python\ndef add2(a, b):\n    return a - b\n```"  # broken
    res = v.selfreplay_veto(eng, ver, check_skills=3, n_samples=2)
    assert res["veto"] and "t1" in res["broke"], res


def test_rrv_empty_vault_is_noop():
    v, ver, eng = SelfCertVault(), Verifier(sandbox=PythonSandbox()), StubEngine()
    res = v.selfreplay_veto(eng, ver, check_skills=3, n_samples=1)
    assert not res["veto"] and res["checked"] == 0


# ---------------------------------------------------------------------------
# 5) DYNAMIC poisoned-gold env test (decisions must ignore gold)
# ---------------------------------------------------------------------------

class _Meta:
    def __init__(self, v):
        self.version = v
        self.content_hash = f"h{v}"


class _Registry:
    def __init__(self):
        self.active_version = -1
        self.n = 0

    def register(self, op, meta):
        self.n += 1
        self.active_version = self.n - 1
        return _Meta(self.active_version)

    def history(self):
        return []


class FakeEngine:
    """Minimal engine for env-level SCCL tests (no torch)."""

    def __init__(self, cfg, regen_passes=True):
        self.cfg = cfg
        self.registry = _Registry()
        self._replay = []
        self.updates_done = 0
        self.regen_passes = regen_passes
        self.update_calls = 0
        self.last_update_kw = None

    def _snapshot(self):
        return {"n": self.updates_done, "replay": list(self._replay)}

    def _restore(self, snap):
        self.updates_done = snap["n"]
        self._replay = snap["replay"]

    def apply_update(self, pairs, **kw):
        self.update_calls += 1
        self.last_update_kw = dict(kw)
        self._replay.extend(pairs)
        self.updates_done += 1
        return {"loss_start": 1.0, "loss_end": 0.2, "grad_norm": 0.5,
                "n_pairs": len(pairs)}

    def sample_candidates(self, prompt, n, temperature=0.7, top_p=0.95,
                          adapter_on=True, max_new_tokens=None):
        code = (_CERT_CODE if self.regen_passes
                else "def add2(a, b):\n    return a - b")
        return ["```python\n" + code + "\n```"] * max(1, n)

    def holdout_score(self, holdout, verifier, adapter_on):
        return 0.5

    def register_adapter(self, op, meta):
        return self.registry.register(op, meta)


def _poisoned_task():
    """Gold is WRONG on purpose: correct code gets gold reward ~0."""
    return Task(task_id="t_gold_wrong", family="fam", domain="code",
                prompt="Return the sum of two integers a and b.",
                test_code="assert add2(1, 2) == 99",          # poisoned gold
                reference_answer="def add2(a, b):\n    return 99",
                entry_point="add2")


def _sccl_env(regen_passes=True, learner="sccl", learners=None, nogate=None):
    cfg = ExperimentConfig(out_dir="runs/_test_sccl")
    cfg._learner_name = learner
    cfg.sccl_learners = learners or ["sccl"]
    cfg.sccl_nogate_learners = nogate or []
    cfg.sccl_gate_probe = 0
    eng = FakeEngine(cfg, regen_passes=regen_passes)
    ver = Verifier(sandbox=PythonSandbox())
    vault = SelfCertVault()
    env = GroundedContinualEnv(cfg, eng, ver, [_make_family(_poisoned_task())],
                               holdout=[], vault=vault, sccl=True)
    return env, eng, ver, vault


def _make_family(task):
    from gcl.curriculum import Family
    return Family(name="fam", tasks=[task], holdout=[])


def _cert_meta(found=True, code=_CERT_CODE, tests=None):
    return {"found": found, "code": code, "confidence": 0.9 if found else 0.2,
            "self_tests": tests if tests is not None else list(_CERT_TESTS),
            "prompt": "You are an expert Python programmer.\nReturn the sum of two integers a and b.\n```python\n",
            "entry": "add2", "n_tests": 2, "n_discriminative": 2}


def test_sccl_trains_on_certification_despite_poisoned_gold():
    """Gold reward is ~0 (poisoned), but a certified target must still train."""
    env, eng, ver, vault = _sccl_env()
    # sanity: gold telemetry reward for the correct code must be ~0 under poisoned gold
    r, info, _ = ver.reward(domain="code", code=_CERT_CODE,
                            test_code="assert add2(1, 2) == 99", reference_answer="")
    assert info["pass_rate"] == 0.0
    o, reward, done, step_info = env.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    ui = step_info["update_info"]
    assert reward < 0.5, "poisoned gold should score the correct code ~0 (telemetry)"
    assert ui["target_source"] == "self_certified"
    assert ui.get("executed") and ui.get("accepted"), ui
    assert ui["gate"]["method"] == "sccl_rrv"
    assert step_info["sccl"]["found"] is True
    assert len(vault) == 1, "certified skill must be committed"


def test_sccl_refuses_uncertified_despite_passing_gold():
    """Answer passes POISONED gold (high reward) but is NOT certified -> no update."""
    env, eng, ver, vault = _sccl_env()
    bad_for_spec = "def add2(a, b):\n    return 99"   # passes poisoned gold
    r, info, _ = ver.reward(domain="code", code=bad_for_spec,
                            test_code="assert add2(1, 2) == 99", reference_answer="")
    assert info["pass_rate"] >= 1.0, "sanity: must pass poisoned gold"
    o, reward, done, step_info = env.step(Action(
        answer=bad_for_spec, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=False, code=bad_for_spec)}))
    ui = step_info["update_info"]
    assert reward > 0.5, "telemetry shows gold would love this answer"
    assert not ui.get("executed"), ui
    assert ui.get("reason") == "no_verified_target"
    assert eng.update_calls == 0
    assert len(vault) == 0


def test_sccl_rrv_rolls_back_when_prior_skill_regresses():
    """Even a certified, gold-irrelevant update must roll back if the RRV gate
    detects a previously certified skill can no longer be regenerated."""
    env, eng, ver, vault = _sccl_env(regen_passes=False)
    # seed a prior certified skill
    vault.commit_certified("prior", "fam0", "Return the sum.",
                           "PROMPT\n```python\n", _CERT_CODE, _CERT_TESTS, 0.9)
    o, reward, done, step_info = env.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    ui = step_info["update_info"]
    assert ui.get("executed") and not ui.get("accepted"), ui
    assert ui["gate"]["method"] == "sccl_rrv" and ui["gate"]["veto"]
    assert env.rollback_count == 1 and env.update_count == 0


def test_sccl_nogate_ablation_has_no_gate():
    env, eng, ver, vault = _sccl_env(regen_passes=False, learner="sccl_nogate",
                                     learners=["sccl"], nogate=["sccl_nogate"])
    vault.commit_certified("prior", "fam0", "Return the sum.",
                           "PROMPT\n```python\n", _CERT_CODE, _CERT_TESTS, 0.9)
    o, reward, done, step_info = env.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    ui = step_info["update_info"]
    assert ui["gate"]["method"] == "none"
    assert ui.get("accepted"), "nogate ablation trains unconditionally"
    assert env.rollback_count == 0


def test_sccl_math_target_format():
    from gcl.curriculum import Family
    cfg = ExperimentConfig(out_dir="runs/_test_sccl")
    cfg._learner_name = "sccl"
    cfg.sccl_learners = ["sccl"]
    cfg.sccl_gate_probe = 0
    task = Task(task_id="m1", family="fam", domain="math",
                prompt="What is 6 times 7?", test_code="",
                reference_answer="999",          # poisoned gold
                entry_point="")
    eng = FakeEngine(cfg)
    ver = Verifier(sandbox=PythonSandbox())
    env = GroundedContinualEnv(cfg, eng, ver, [Family("fam", [task], [])],
                               holdout=[], vault=SelfCertVault(), sccl=True)
    meta = {"found": True, "code": "42", "confidence": 0.86, "self_tests": [],
            "prompt": "Solve and give ONLY the final numeric answer.\nWhat is 6 times 7?\nAnswer: ",
            "entry": "", "n_tests": 7, "n_discriminative": 0}
    o, reward, done, step_info = env.step(Action(
        answer="42", learn_op=LearnOp.UPDATE_LORA, metadata={"sccl": meta}))
    ui = step_info["update_info"]
    assert reward == 0.0, "poisoned gold reference -> telemetry reward 0"
    assert ui.get("executed") and ui.get("accepted"), ui
    assert ui["target_source"] == "self_certified"


# ---------------------------------------------------------------------------
# 6) SCCL v2 — self-manufactured stability (rehearsal + neighborhood probes)
# ---------------------------------------------------------------------------

class _ProbeEngine(StubEngine):
    """Scripted engine for make_probe tests: emits a paraphrase / math variant,
    then swaps candidate quality for the PROBE certification pass."""

    def __init__(self, paraphrase="Compute the total of the two integer arguments a and b and return it.",
                 probe_pool=None, greedy_for_probe=None,
                 math_variant="What is 7 times 8? Give only the final number."):
        super().__init__()
        self.paraphrase = paraphrase
        self.probe_pool = probe_pool
        self.greedy_for_probe = greedy_for_probe
        self.math_variant = math_variant
        self._in_probe = False

    def generate(self, prompt, adapter_on=True, greedy=False):
        if "Rewrite the following programming task specification" in prompt:
            self._in_probe = True
            if self.probe_pool is not None:
                self.candidates = list(self.probe_pool)
            if self.greedy_for_probe is not None:
                self.greedy_solution = self.greedy_for_probe
            return self.paraphrase
        if "Write ONE similar word problem" in prompt:
            return self.math_variant
        return super().generate(prompt, adapter_on=adapter_on, greedy=greedy)


def _base_cert(self_tests=None):
    return CertResult(found=True, code=_CERT_CODE, confidence=0.9, domain="code",
                      entry="add2", self_tests=list(self_tests or _CERT_TESTS),
                      prompt="You are an expert Python programmer.\nReturn the sum.\n```python\n")


def test_make_probe_code_bidirectional_accept():
    """Paraphrase re-certifies and BOTH directions of cross-validation pass:
    the probe is admitted with the probe winner's code + tests."""
    sc = SelfCertifier(k=3, temp=0.8, test_bags=2, tests_per_bag=2, tau=0.65)
    eng, ver = _ProbeEngine(), Verifier(sandbox=PythonSandbox())
    base = _base_cert(self_tests=["assert add2(1, 2) == 3", "assert add2(0, 0) == 0"])
    probe = sc.make_probe(eng, ver, "Return the sum of two integers a and b.",
                          "code", base)
    assert probe is not None, "bidirectional cross-validation should admit the probe"
    assert probe["kind"] == "probe" and probe["domain"] == "code"
    assert probe["entry"] == "add2"
    assert "named `add2`" in probe["spec"], "interface hint must ride along the paraphrase"
    assert probe["spec"] != "Return the sum of two integers a and b."
    assert probe["self_tests"], "probe must carry its own certifying suite"
    assert probe["confidence"] >= 0.65
    # probe winner must actually pass the probe suite (executed, not assumed)
    _, info, _ = ver.reward(domain="code", code=probe["code"],
                            test_code="\n".join(probe["self_tests"]))
    assert info["pass_rate"] >= 1.0


def test_make_probe_code_rejected_when_cross_validation_fails():
    """Probe winner passes its OWN suite but fails the BASE suite: the
    paraphrase is not semantically equivalent -> probe rejected (None)."""
    # probe pool: "return 0" passes some add2 self-tests, "return 99" none.
    # The winner ("return 0") fails the base suite -> second direction fails.
    sc = SelfCertifier(k=3, temp=0.8, test_bags=2, tests_per_bag=2, tau=0.65)
    eng = _ProbeEngine(probe_pool=["```python\ndef add2(a, b):\n    return 0\n```"],
                       greedy_for_probe="```python\ndef add2(a, b):\n    return 99\n```")
    ver = Verifier(sandbox=PythonSandbox())
    base = _base_cert(self_tests=["assert add2(9, 9) == 18"])
    probe = sc.make_probe(eng, ver, "Return the sum of two integers a and b.",
                          "code", base)
    assert probe is None, "cross-validation failure must reject the probe"


def test_make_probe_code_rejected_when_paraphrase_uncertifiable():
    """If the paraphrase itself cannot be certified, no probe exists."""
    sc = SelfCertifier(k=3, temp=0.8, test_bags=2, tests_per_bag=2, tau=0.65)
    eng = _ProbeEngine(probe_pool=["```python\ndef add2(a, b):\n    return 99\n```"],
                       greedy_for_probe="```python\ndef add2(a, b):\n    return 7\n```")
    ver = Verifier(sandbox=PythonSandbox())
    probe = sc.make_probe(eng, ver, "Return the sum of two integers a and b.",
                          "code", _base_cert())
    assert probe is None


def test_make_probe_math_variant_certified_by_majority():
    sc = SelfCertifier(k=6, tau_math=0.6)
    eng = _ProbeEngine()
    probe = sc.make_probe(eng, Verifier(sandbox=PythonSandbox()),
                          "What is 6 times 7?", "math",
                          CertResult(found=True, code="42", domain="math"))
    assert probe is not None
    assert probe["kind"] == "probe" and probe["domain"] == "math"
    assert probe["code"] == "42"          # majority-vote certified answer
    assert "7 times 8" in probe["spec"]
    assert probe["self_tests"] == []


def test_make_probe_math_rejects_uncertified_variant():
    sc = SelfCertifier(k=6, tau_math=0.99)   # impossible threshold
    eng = _ProbeEngine()
    probe = sc.make_probe(eng, Verifier(sandbox=PythonSandbox()),
                          "What is 6 times 7?", "math",
                          CertResult(found=True, code="42", domain="math"))
    assert probe is None


def test_commit_probe_stores_kind_and_persists(tmp_path):
    v = SelfCertVault(directory=str(tmp_path / "vault"))
    ok = v.commit_probe(task_id="t1:p", family="f", spec="Paraphrase of sum.",
                        prompt="PROMPT\n```python\n", code=_CERT_CODE,
                        self_tests=_CERT_TESTS, conf=0.9)
    assert ok and len(v) == 1
    assert v._skills[-1].kind == "probe"
    # probes ride along in replay pairs too (certified pairs are certified pairs)
    pairs = v.to_pairs()
    assert pairs and pairs[0]["target"] == _CERT_CODE
    # kind survives a save/load round-trip
    v2 = SelfCertVault(directory=str(tmp_path / "vault"))
    assert len(v2) == 1 and v2._skills[-1].kind == "probe"


def test_commit_certified_defaults_to_skill_kind():
    v = SelfCertVault()
    v.commit_certified("t1", "f", "s", "p", _CERT_CODE, _CERT_TESTS, 0.9)
    assert v._skills[-1].kind == "skill"


def test_rrv_check_skills_ignores_probes():
    """v1 compatibility: check_skills must only touch kind='skill' entries, so
    a probe-only vault never vetoes under the legacy call shape."""
    v, ver, eng = SelfCertVault(), Verifier(sandbox=PythonSandbox()), StubEngine()
    v.commit_probe("t1:p", "f", "Paraphrase.", "PROMPT\n```python\n",
                   _CERT_CODE, _CERT_TESTS, 0.9)
    eng.regen_code = "```python\ndef add2(a, b):\n    return a - b\n```"  # broken
    res = v.selfreplay_veto(eng, ver, check_skills=3, n_samples=2)
    assert not res["veto"] and res["checked"] == 0


def test_rrv_probe_check_fires_on_untrained_regression():
    """The v2 gate protects a GENERALIZATION neighborhood: a probe the model
    never trained on must veto the update when it can no longer be regenerated."""
    v, ver, eng = SelfCertVault(), Verifier(sandbox=PythonSandbox()), StubEngine()
    v.commit_certified("t1", "f", "Return the sum.", "PROMPT s\n```python\n",
                       _CERT_CODE, _CERT_TESTS, 0.9)
    v.commit_probe("t1:p", "f", "Paraphrase.", "PROMPT p\n```python\n",
                   _CERT_CODE, _CERT_TESTS, 0.9)
    eng.regen_code = "```python\ndef add2(a, b):\n    return a + b\n```"
    res = v.selfreplay_veto(eng, ver, check_skills=3, n_samples=2, check_probes=2)
    assert not res["veto"] and res["checked_probes"] == 1
    # now the adapter loses the skill: both the skill AND the probe regress
    eng.regen_code = "```python\ndef add2(a, b):\n    return 0\n```"
    res = v.selfreplay_veto(eng, ver, check_skills=3, n_samples=2, check_probes=2)
    assert res["veto"]
    assert "t1" in res["broke"] and "t1:p" in res["broke_probes"]
    assert res["reason"].startswith("rrv_regress:")


def test_rrv_probe_check_off_by_default():
    v, ver, eng = SelfCertVault(), Verifier(sandbox=PythonSandbox()), StubEngine()
    v.commit_probe("t1:p", "f", "Paraphrase.", "PROMPT p\n```python\n",
                   _CERT_CODE, _CERT_TESTS, 0.9)
    eng.regen_code = "```python\ndef add2(a, b):\n    return 0\n```"  # broken
    res = v.selfreplay_veto(eng, ver, check_skills=3, n_samples=2)
    assert not res["veto"] and res["checked_probes"] == 0


def test_rrv_math_check():
    """math entries were UNPROTECTED in v1; v2 re-generates answers and vetoes
    when the certified value can no longer be reproduced."""
    v, ver = SelfCertVault(), Verifier(sandbox=PythonSandbox())
    eng = StubEngine()
    math_prompt = "Solve and give ONLY the final numeric answer.\nWhat is 6 times 7?\nAnswer: "
    v.commit_certified("m1", "fam", "What is 6 times 7?", math_prompt, "42",
                       [], 0.86, domain="math")
    res = v.selfreplay_veto(eng, ver, check_skills=3, n_samples=3, check_math=1)
    assert not res["veto"] and res["checked_math"] == 1
    eng.math_answers = ["41", "41", "41"]   # the answer drifted
    res = v.selfreplay_veto(eng, ver, check_skills=3, n_samples=3, check_math=1)
    assert res["veto"] and "m1" in res["broke_math"]
    # canonical-form match: 42.0 still counts as 42
    eng.math_answers = ["The answer is 42.0", "41", "41"]
    res = v.selfreplay_veto(eng, ver, check_skills=3, n_samples=3, check_math=1)
    assert not res["veto"]


def test_sccl_certified_rehearsal_mixes_vault_pairs():
    """Certified rehearsal: a replay learner's update must train on stride-sampled
    certified vault pairs in addition to the new certified target."""
    env, eng, ver, vault = _sccl_env(learner="sccl_replay", learners=["sccl_replay"])
    env.cfg.sccl_replay_learners = ["sccl_replay"]
    env.cfg.sccl_replay_k = 2
    for i in range(3):
        vault.commit_certified(f"t{i}", "fam", f"Spec {i}.",
                               f"PROMPT {i}\n```python\n", _CERT_CODE,
                               _CERT_TESTS, 0.9)
    o, reward, done, step_info = env.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    ui = step_info["update_info"]
    assert ui.get("executed") and ui.get("accepted"), ui
    kw = eng.last_update_kw
    assert kw is not None and kw.get("replay_pairs") is not None
    assert len(kw["replay_pairs"]) == 2, "stride sample of 2 from 3 vault pairs"
    assert kw["replay_frac"] == 1.0
    assert all("target" in p and "prompt" in p for p in kw["replay_pairs"])


def test_sccl_v1_learner_gets_no_rehearsal():
    env, eng, ver, vault = _sccl_env()
    env.cfg.sccl_replay_learners = ["sccl_replay"]   # 'sccl' is NOT listed
    env.cfg.sccl_replay_k = 2
    vault.commit_certified("t0", "fam", "Spec.", "PROMPT\n```python\n",
                           _CERT_CODE, _CERT_TESTS, 0.9)
    env.step(Action(answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
                    metadata={"sccl": _cert_meta(found=True)}))
    kw = eng.last_update_kw
    assert kw.get("replay_pairs") is None and kw.get("replay_frac") == 0.0


def test_sccl_probe_learner_rrv_vetoes_on_probe_regression():
    """A probe learner's gate re-checks probes; a non-probe learner's gate
    ignores the exact same probe-only vault."""
    env, eng, ver, vault = _sccl_env(regen_passes=False, learner="sccl_probe",
                                     learners=["sccl_probe"])
    env.cfg.sccl_probe_learners = ["sccl_probe"]
    env.cfg.sccl_probe_check = 2
    env.cfg.sccl_rrv_math = 0
    # probe-ONLY vault: v1 skill check has nothing to look at
    vault.commit_probe("t0:p", "fam", "Paraphrase.", "PROMPT p\n```python\n",
                       _CERT_CODE, _CERT_TESTS, 0.9)
    o, reward, done, step_info = env.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    ui = step_info["update_info"]
    assert ui.get("executed") and not ui.get("accepted"), ui
    assert ui["gate"]["broke_probes"] == ["t0:p"]
    assert env.rollback_count == 1

    # same vault, v1 learner: probes are invisible to its gate
    env2, eng2, ver2, vault2 = _sccl_env(regen_passes=False)
    env2.cfg.sccl_probe_learners = ["sccl_probe"]
    env2.cfg.sccl_probe_check = 2
    vault2.commit_probe("t0:p", "fam", "Paraphrase.", "PROMPT p\n```python\n",
                        _CERT_CODE, _CERT_TESTS, 0.9)
    o, reward, done, step_info = env2.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    ui = step_info["update_info"]
    assert ui.get("accepted"), "v1 learner's gate must not consult probes"
    assert ui["gate"]["checked_probes"] == 0


def test_learner_registry_has_v2_names():
    from gcl.learners.learners import LEARNERS, SCCLLearner
    for n in ("sccl_replay", "sccl_probe", "sccl_v2", "sccl_promote", "sccl_n"):
        assert n in LEARNERS, f"{n} missing from LEARNERS registry"
        assert issubclass(LEARNERS[n], SCCLLearner)


# ---------------------------------------------------------------------------
# 7) Probe curriculum (v3 candidate): probes that survive RRV checks graduate
#    to certified rehearsal pairs. Still fully gold-free.
# ---------------------------------------------------------------------------

def test_rrv_probe_check_increments_probe_checks():
    """A passing probe check bumps probe_checks; a regressing probe's counter
    stays put (it goes to broke_probes instead)."""
    v, ver, eng = SelfCertVault(), Verifier(sandbox=PythonSandbox()), StubEngine()
    v.commit_probe("t1:p", "f", "Paraphrase.", "PROMPT p\n```python\n",
                   _CERT_CODE, _CERT_TESTS, 0.9)
    eng.regen_code = "```python\ndef add2(a, b):\n    return a + b\n```"
    v.selfreplay_veto(eng, ver, check_skills=3, n_samples=2, check_probes=1)
    assert v._skills[-1].probe_checks == 1
    v.selfreplay_veto(eng, ver, check_skills=3, n_samples=2, check_probes=1)
    assert v._skills[-1].probe_checks == 2
    eng.regen_code = "```python\ndef add2(a, b):\n    return 0\n```"  # regress
    res = v.selfreplay_veto(eng, ver, check_skills=3, n_samples=2, check_probes=1)
    assert res["veto"] and "t1:p" in res["broke_probes"]
    assert v._skills[-1].probe_checks == 2, "regressing check must not count"


def test_promote_probes_graduates_to_skill():
    """promote_probes flips eligible probes to skills, reports the count, and
    the graduate then falls under RRV's SKILL check (not the probe check)."""
    v, ver, eng = SelfCertVault(), Verifier(sandbox=PythonSandbox()), StubEngine()
    v.commit_certified("t1", "f", "Return the sum.", "PROMPT s\n```python\n",
                       _CERT_CODE, _CERT_TESTS, 0.9)
    v.commit_probe("t1:p", "f", "Paraphrase.", "PROMPT p\n```python\n",
                   _CERT_CODE, _CERT_TESTS, 0.9)
    v._skills[-1].probe_checks = 2
    n = v.promote_probes(min_checks=2)
    assert n == 1
    kinds = {s.task_id: s.kind for s in v._skills}
    assert kinds == {"t1": "skill", "t1:p": "skill"}
    # second call is a no-op: nothing left to promote
    assert v.promote_probes(min_checks=2) == 0
    # the graduate is now reachable by the v1 skill protocol
    eng.regen_code = "```python\ndef add2(a, b):\n    return a + b\n```"
    res = v.selfreplay_veto(eng, ver, check_skills=5, n_samples=2, check_probes=2)
    assert res["checked"] == 2 and res["checked_probes"] == 0


def test_promote_probes_respects_age_threshold():
    v = SelfCertVault()
    v.commit_probe("p1", "f", "s", "p\n```python\n", _CERT_CODE, _CERT_TESTS, 0.9)
    v.commit_probe("p2", "f", "s", "p\n```python\n", _CERT_CODE, _CERT_TESTS, 0.9)
    v._skills[0].probe_checks = 1
    v._skills[1].probe_checks = 3
    assert v.promote_probes(min_checks=3) == 1
    kinds = {s.task_id: s.kind for s in v._skills}
    assert kinds == {"p1": "probe", "p2": "skill"}


def test_probe_checks_persist_roundtrip(tmp_path):
    v = SelfCertVault(directory=str(tmp_path / "vault"))
    v.commit_probe("p1", "f", "s", "p\n```python\n", _CERT_CODE, _CERT_TESTS, 0.9)
    v._skills[-1].probe_checks = 3
    v._save()
    v2 = SelfCertVault(directory=str(tmp_path / "vault"))
    assert v2._skills[-1].probe_checks == 3 and v2._skills[-1].kind == "probe"


def test_env_promotes_probes_after_accepted_update():
    """Promote learner: a probe surviving its RRV check graduates once the
    update is ACCEPTED; the gate reports the promotion count."""
    env, eng, ver, vault = _sccl_env(learner="sccl_promote",
                                     learners=["sccl_promote"])
    env.cfg.sccl_probe_learners = ["sccl_promote"]
    env.cfg.sccl_probe_check = 1
    env.cfg.sccl_rrv_math = 0
    env.cfg.sccl_probe_promote_learners = ["sccl_promote"]
    env.cfg.sccl_probe_promote_age = 1
    vault.commit_probe("t0:p", "fam", "Paraphrase.", "PROMPT p\n```python\n",
                       _CERT_CODE, _CERT_TESTS, 0.9)
    o, reward, done, step_info = env.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    ui = step_info["update_info"]
    assert ui.get("executed") and ui.get("accepted"), ui
    assert ui["gate"]["checked_probes"] == 1
    assert ui["gate"].get("probes_promoted") == 1
    probe = next(s for s in vault._skills if s.task_id == "t0:p")
    assert probe.kind == "skill", "surviving probe must graduate"


def test_env_no_promotion_for_other_learners():
    """Same vault and flags, but the learner is not in the promote set: the
    probe keeps guarding the frontier as an UNTRAINED check."""
    env, eng, ver, vault = _sccl_env(learner="sccl_v2", learners=["sccl_v2"])
    env.cfg.sccl_probe_learners = ["sccl_v2"]
    env.cfg.sccl_probe_check = 1
    env.cfg.sccl_rrv_math = 0
    env.cfg.sccl_probe_promote_learners = ["sccl_promote"]  # sccl_v2 NOT listed
    env.cfg.sccl_probe_promote_age = 1
    vault.commit_probe("t0:p", "fam", "Paraphrase.", "PROMPT p\n```python\n",
                       _CERT_CODE, _CERT_TESTS, 0.9)
    o, reward, done, step_info = env.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    ui = step_info["update_info"]
    assert ui.get("accepted"), ui
    assert ui["gate"].get("probes_promoted", 0) == 0
    probe = next(s for s in vault._skills if s.task_id == "t0:p")
    assert probe.kind == "probe"


def test_env_no_promotion_on_vetoed_update():
    """A vetoed (rolled-back) update must not graduate probes either."""
    env, eng, ver, vault = _sccl_env(regen_passes=False, learner="sccl_promote",
                                     learners=["sccl_promote"])
    env.cfg.sccl_probe_learners = ["sccl_promote"]
    env.cfg.sccl_probe_check = 1
    env.cfg.sccl_rrv_math = 0
    env.cfg.sccl_probe_promote_learners = ["sccl_promote"]
    env.cfg.sccl_probe_promote_age = 1
    vault.commit_probe("t0:p", "fam", "Paraphrase.", "PROMPT p\n```python\n",
                       _CERT_CODE, _CERT_TESTS, 0.9)
    o, reward, done, step_info = env.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    ui = step_info["update_info"]
    assert ui.get("executed") and not ui.get("accepted"), ui
    assert "probes_promoted" not in ui["gate"]
    probe = next(s for s in vault._skills if s.task_id == "t0:p")
    assert probe.kind == "probe"


# ---------------------------------------------------------------------------
# 8) SCCL v3 — NEIGHBORHOOD certification at admission (gold-free)
# ---------------------------------------------------------------------------

def test_nbhd_code_robust_when_winner_passes_variant_tests():
    """A winner that also passes fresh self-tests written for a paraphrased
    spec variant is neighborhood-consistent -> admitted."""
    sc = SelfCertifier(k=3, temp=0.8, test_bags=2, tests_per_bag=2, tau=0.65)
    eng, ver = _ProbeEngine(), Verifier(sandbox=PythonSandbox())
    res = sc.check_neighborhood(eng, ver,
                                "Return the sum of two integers a and b.",
                                "code", _base_cert())
    assert res["robust"] is True, res
    assert res["reason"] == ""
    assert "named `add2`" in res["variant"], "interface hint must ride along"
    assert res["variant"] != "Return the sum of two integers a and b."


def test_nbhd_code_rejects_instance_narrow_winner():
    """A winner that only solves the exact instance (fails the variant suite)
    is instance-narrow -> REJECTED at admission (evidence-based, closed)."""
    sc = SelfCertifier(k=3, temp=0.8, test_bags=2, tests_per_bag=2, tau=0.65)
    eng, ver = _ProbeEngine(), Verifier(sandbox=PythonSandbox())
    narrow = CertResult(found=True, code="def add2(a, b):\n    return 0",
                        confidence=0.9, domain="code", entry="add2",
                        self_tests=["assert add2(0, 0) == 0"], prompt="p")
    res = sc.check_neighborhood(eng, ver,
                                "Return the sum of two integers a and b.",
                                "code", narrow)
    assert res["robust"] is False, res
    assert res["reason"] == "winner_fails_variant"


def test_nbhd_code_open_when_paraphrase_unavailable():
    """If the variant cannot be manufactured, the neighborhood question is
    unanswerable -> ADMIT (open), do not punish the cert."""
    sc = SelfCertifier(k=3, tau=0.65)
    eng = _ProbeEngine(paraphrase="too short")   # < 20 chars
    ver = Verifier(sandbox=PythonSandbox())
    res = sc.check_neighborhood(eng, ver,
                                "Return the sum of two integers a and b.",
                                "code", _base_cert())
    assert res["robust"] is True
    assert res["reason"] == "variant_generation_failed_open"


def test_nbhd_code_open_when_no_variant_tests():
    sc = SelfCertifier(k=3, tau=0.65)
    eng = _ProbeEngine()
    eng.test_bags = [["no asserts here at all"]]  # parse_asserts -> []
    ver = Verifier(sandbox=PythonSandbox())
    res = sc.check_neighborhood(eng, ver,
                                "Return the sum of two integers a and b.",
                                "code", _base_cert())
    assert res["robust"] is True
    assert res["reason"] == "variant_no_tests_open"


def test_nbhd_math_robust_when_variant_self_consistent():
    """math: the model must remain self-consistent (majority vote) on a numeric
    variant — evidence the METHOD, not the memorized answer, is held."""
    sc = SelfCertifier(k=6, tau_math=0.6)
    eng = _ProbeEngine()
    res = sc.check_neighborhood(eng, Verifier(sandbox=PythonSandbox()),
                                "What is 6 times 7?", "math",
                                CertResult(found=True, code="42", domain="math"))
    assert res["robust"] is True, res
    assert res["reason"] == ""
    assert "7 times 8" in res["variant"]


def test_nbhd_math_rejects_inconsistent_variant():
    sc = SelfCertifier(k=6, tau_math=0.99)   # impossible threshold
    eng = _ProbeEngine()
    res = sc.check_neighborhood(eng, Verifier(sandbox=PythonSandbox()),
                                "What is 6 times 7?", "math",
                                CertResult(found=True, code="42", domain="math"))
    assert res["robust"] is False
    assert res["reason"] == "variant_not_certified"


def test_nbhd_math_open_when_variant_generation_fails():
    sc = SelfCertifier(k=6, tau_math=0.6)
    eng = _ProbeEngine(math_variant="short")
    res = sc.check_neighborhood(eng, Verifier(sandbox=PythonSandbox()),
                                "What is 6 times 7?", "math",
                                CertResult(found=True, code="42", domain="math"))
    assert res["robust"] is True
    assert res["reason"] == "variant_generation_failed_open"


def test_nbhd_config_fields_parse():
    cfg = ExperimentConfig()
    assert cfg.sccl_nbhd_learners == []
    assert cfg.sccl_nbhd_tests == 3
    cfg.sccl_nbhd_learners = ["sccl_n"]
    assert cfg.sccl_nbhd_learners == ["sccl_n"]


def test_env_propagates_nbhd_and_probe_audit_fields_to_step_info():
    """Regression: the experiment attaches sccl_nbhd / sccl_probe to the action
    metadata; env.step must carry them into step_info so trajectory rows are
    auditable per-step (analyze_nbhd.py). Telemetry only — the fields must NOT
    influence the accept/reject decision, which stays certification+RRV driven."""
    env, eng, ver, vault = _sccl_env(learner="sccl_n", learners=["sccl_n"])
    nbhd = {"robust": True, "reason": "winner_passes_variant"}
    probe = {"made": True, "committed": True}
    o, reward, done, step_info = env.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True),
                  "sccl_nbhd": nbhd, "sccl_probe": probe}))
    assert step_info.get("sccl_nbhd") == nbhd
    assert step_info.get("sccl_probe") == probe
    # the step still behaves identically: certified target trains and is accepted
    ui = step_info["update_info"]
    assert ui.get("executed") and ui.get("accepted"), ui


def test_env_no_audit_fields_when_absent_from_metadata():
    """Rows without a neighborhood/probe attachment must not gain the keys
    (keeps the trajectory schema backward-compatible with v1/v2 runs)."""
    env, eng, ver, vault = _sccl_env()
    o, reward, done, step_info = env.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    assert "sccl_nbhd" not in step_info
    assert "sccl_probe" not in step_info



# ---------------------------------------------------------------------------
# 7) SCCL v4: in-update BASE ANCHOR (gold-free capability preservation)
# ---------------------------------------------------------------------------

def test_sccl_anchor_engaged_for_listed_learner():
    """A learner in sccl_anchor_learners with sccl_anchor_lambda>0 must pass the
    anchor strength through to apply_update (in-update pull toward frozen base)."""
    env, eng, ver, vault = _sccl_env(learner="sccl_anchor", learners=["sccl_anchor"])
    env.cfg.sccl_anchor_learners = ["sccl_anchor"]
    env.cfg.sccl_anchor_lambda = 0.4
    o, reward, done, step_info = env.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    ui = step_info["update_info"]
    assert ui.get("executed") and ui.get("accepted"), ui
    assert eng.last_update_kw is not None
    assert abs(eng.last_update_kw["anchor_lambda"] - 0.4) < 1e-9, \
        f"anchor must be engaged for listed learner, got {eng.last_update_kw['anchor_lambda']}"
    # audit trail: the anchor strength must be logged in update_info so a real
    # run's trajectory can be verified gold-free post hoc
    assert abs(ui["anchor_lambda"] - 0.4) < 1e-9


def test_sccl_anchor_not_engaged_for_unlisted_learner():
    """Isolation: a learner NOT in sccl_anchor_learners must get anchor_lambda=0,
    so the anchor can be cleanly A/B'd in the ladder without cross-contamination."""
    env, eng, ver, vault = _sccl_env(learner="sccl", learners=["sccl"])
    env.cfg.sccl_anchor_learners = ["sccl_anchor"]   # sccl is NOT listed
    env.cfg.sccl_anchor_lambda = 0.4
    o, reward, done, step_info = env.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    assert eng.last_update_kw["anchor_lambda"] == 0.0, \
        "unlisted learner must not receive the anchor"


def test_sccl_anchor_default_off_is_goldfree_and_decision_independent():
    """With no anchor learners configured the anchor stays 0 (legacy behaviour),
    and the accept/reject decision is unchanged. The anchor is a parameter-space
    penalty and never reads gold; here we confirm zeroing/setting it does not flip
    the gate outcome on a poisoned-gold task (decision stays certification+RRV)."""
    env, eng, ver, vault = _sccl_env(learner="sccl_anchor", learners=["sccl_anchor"])
    # default config -> no anchor learners -> anchor_lambda must be 0
    o, reward, done, step_info = env.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    assert eng.last_update_kw["anchor_lambda"] == 0.0
    ui = step_info["update_info"]
    assert ui.get("executed") and ui.get("accepted"), ui
    # now engage the anchor heavily; the gate outcome must be IDENTICAL (gold-free
    # anchor cannot change which updates are accepted, only how they move weights)
    env2, eng2, ver2, vault2 = _sccl_env(learner="sccl_anchor", learners=["sccl_anchor"])
    env2.cfg.sccl_anchor_learners = ["sccl_anchor"]
    env2.cfg.sccl_anchor_lambda = 5.0
    o2, reward2, done2, step_info2 = env2.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    ui2 = step_info2["update_info"]
    assert ui2.get("executed") and ui2.get("accepted") == ui.get("accepted"), \
        "anchor strength must not change the accept/reject decision"
    assert eng2.last_update_kw["anchor_lambda"] == 5.0


def test_sccl_anchor_per_learner_lambda_map():
    """sccl_anchor_lambdas overrides the scalar per learner so a lambda ablation
    can run in a single ladder; an unlisted anchor learner falls back to the scalar."""
    env, eng, ver, vault = _sccl_env(learner="sccl_anchor_hi", learners=["sccl_anchor_hi"])
    env.cfg.sccl_anchor_learners = ["sccl_anchor_lo", "sccl_anchor_hi"]
    env.cfg.sccl_anchor_lambda = 0.1                     # fallback scalar
    env.cfg.sccl_anchor_lambdas = {"sccl_anchor_hi": 0.5}  # override for this learner
    o, reward, done, step_info = env.step(Action(
        answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
        metadata={"sccl": _cert_meta(found=True)}))
    assert abs(eng.last_update_kw["anchor_lambda"] - 0.5) < 1e-9
    # fallback: a listed learner with no map entry uses the scalar
    env2, eng2, ver2, vault2 = _sccl_env(learner="sccl_anchor_lo", learners=["sccl_anchor_lo"])
    env2.cfg.sccl_anchor_learners = ["sccl_anchor_lo", "sccl_anchor_hi"]
    env2.cfg.sccl_anchor_lambda = 0.1
    env2.cfg.sccl_anchor_lambdas = {"sccl_anchor_hi": 0.5}
    env2.step(Action(answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
                     metadata={"sccl": _cert_meta(found=True)}))
    assert abs(eng2.last_update_kw["anchor_lambda"] - 0.1) < 1e-9


def test_v4_ladder_learners_are_registered_and_isolated():
    """Every learner named in the v4 configs must exist in the LEARNERS registry
    (experiment.py silently skips unregistered names — the anchor rows would
    otherwise never run), and the anchor rows must be SCCL learners so they ride
    the gold-free self-cert path."""
    import json, glob
    from gcl.learners.learners import (LEARNERS, SCCLLearner,
                                       SCCLAnchorLoLearner, SCCLAnchorHiLearner)
    for path in glob.glob(os.path.join(os.path.dirname(__file__), "..", "configs", "*.json")):
        try:
            cfg = json.load(open(path))
        except Exception:
            continue  # legacy/commented files the runner never loads
        for name in cfg.get("learners", []):
            assert name in LEARNERS, f"{os.path.basename(path)}: learner '{name}' not in LEARNERS (would be silently skipped)"
    # anchor rows are SCCL subclasses (gold-free self-cert + RRV gate)
    assert issubclass(SCCLAnchorLoLearner, SCCLLearner)
    assert issubclass(SCCLAnchorHiLearner, SCCLLearner)
    assert LEARNERS["sccl_anchor_lo"] is SCCLAnchorLoLearner
    assert LEARNERS["sccl_anchor_hi"] is SCCLAnchorHiLearner
    # anchor rows must be listed in sccl_learners (so the vault/certifier is wired)
    v4 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "sccl_v4.json")))
    for name in ("sccl_anchor_lo", "sccl_anchor_hi"):
        assert name in v4["experiment"]["sccl_learners"], f"{name} missing from sccl_learners"
        assert name in v4["experiment"]["sccl_anchor_learners"], f"{name} missing from sccl_anchor_learners"


def test_v5_stratified_pool_selection():
    """SCCL v5: the family-stratified check pool must take the newest
    ceil(k/F) skill of EVERY certified family (deterministic, no RNG), so an
    older family never leaves the gate's field of view. The recency window it
    replaces (skills[-k:]) is the seed-42 blind spot this fixes."""
    from gcl.vault import SelfCertVault

    class _S:  # minimal stand-in exposing only what the pool reads
        def __init__(self, task_id, family):
            self.task_id = task_id
            self.family = family

    # arith certified early, then math, string, drift (stream order)
    skills = ([_S(f"a{i}", "arith") for i in range(4)]
              + [_S(f"m{i}", "math_word") for i in range(2)]
              + [_S(f"s{i}", "string") for i in range(3)]
              + [_S(f"d{i}", "drift") for i in range(3)])
    pool = SelfCertVault._stratified_pool(skills, 3)
    ids = [s.task_id for s in pool]
    # ceil(3/4)=1 newest per family, sorted family keys -> deterministic order
    assert ids == ["a3", "d2", "m1", "s2"], ids
    # every family represented exactly once (newest of each)
    assert sorted(s.family for s in pool) == ["arith", "drift", "math_word", "string"]
    # k=8 -> ceil(8/4)=2 per family; newest-two of each, order preserved
    pool8 = SelfCertVault._stratified_pool(skills, 8)
    assert [s.task_id for s in pool8] == ["a2", "a3", "d1", "d2", "m0", "m1", "s1", "s2"]
    # determinism: identical input -> identical pool (no hidden RNG)
    assert [s.task_id for s in SelfCertVault._stratified_pool(skills, 3)] == ids
    # empty-family fallback and empty-skill edge cases
    assert SelfCertVault._stratified_pool([], 3) == []
    mixed = [_S("x0", ""), _S("y0", "arith")]
    assert [s.task_id for s in SelfCertVault._stratified_pool(mixed, 2)] == ["x0", "y0"]


def test_v5_strat_learner_registered_and_wired():
    """The v5 factorial rows must be registered SCCL subclasses (else
    experiment.py would silently skip them — the v4 registry bug), and the v5
    config must wire the 2x2 (anchor x stratified) design exactly: sccl is the
    neither control (recency veto, no anchor), sccl_strat coverage-only,
    sccl_anchor_lo anchor-only, sccl_anchor_strat BOTH."""
    import json
    from gcl.learners.learners import (LEARNERS, SCCLLearner, SCCLStratLearner,
                                       SCCLAnchorLoLearner, SCCLAnchorStratLearner)
    for cls in (SCCLStratLearner, SCCLAnchorLoLearner, SCCLAnchorStratLearner):
        assert issubclass(cls, SCCLLearner)
    assert LEARNERS["sccl_strat"] is SCCLStratLearner
    assert LEARNERS["sccl_anchor_strat"] is SCCLAnchorStratLearner
    v5 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "sccl_v5.json")))
    exp = v5["experiment"]
    for n in ("sccl", "sccl_strat", "sccl_anchor_lo", "sccl_anchor_strat"):
        assert n in exp["sccl_learners"], f"{n} missing from sccl_learners"
    # coverage factor: exactly the two stratified rows; sccl stays recency control
    assert set(exp["sccl_stratified_learners"]) == {"sccl_strat", "sccl_anchor_strat"}
    assert "sccl" not in exp["sccl_stratified_learners"]
    assert "sccl_anchor_lo" not in exp["sccl_stratified_learners"]
    # anchor factor: exactly the two anchor rows at lambda=0.1; sccl/sccl_strat off
    assert set(exp["sccl_anchor_learners"]) == {"sccl_anchor_lo", "sccl_anchor_strat"}
    assert exp["sccl_anchor_lambdas"] == {"sccl_anchor_lo": 0.1, "sccl_anchor_strat": 0.1}
    assert "sccl" not in exp["sccl_anchor_learners"]
    assert "sccl_strat" not in exp["sccl_anchor_learners"]
    # the 2x2 must be clean: every learner appears in exactly the intended cell
    assert v5["learners"] == ["frozen", "sccl", "sccl_strat", "sccl_anchor_lo",
                              "sccl_anchor_strat", "vsr_nogold"]


def test_v5_fixed_config_is_same_design():
    """configs/sccl_v5_fixed.json is the corrected rerun of the v5 factorial
    after the anchor no-op fix (e53923a). It must be the IDENTICAL design —
    same stream, seeds, learners, gate wiring — differing only in _comment and
    out_dir, so its rows are directly comparable to the first flight and the
    non-anchor rows double as the determinism check across the engine fix."""
    import json
    cfgs = os.path.join(os.path.dirname(__file__), "..", "configs")
    a = json.load(open(os.path.join(cfgs, "sccl_v5.json")))
    b = json.load(open(os.path.join(cfgs, "sccl_v5_fixed.json")))
    assert [k for k in a if a[k] != b[k]] == ["_comment", "experiment", "out_dir"]
    ea, eb = a["experiment"], b["experiment"]
    assert [k for k in ea if ea[k] != eb[k]] == ["out_dir"]
    assert b["out_dir"] == "runs/sccl_v5_fixed"
    assert b["experiment"]["out_dir"] == "runs/sccl_v5_fixed"


def test_every_config_learner_is_registered():
    """Regression: run_experiment used to SILENTLY SKIP learner names missing
    from the LEARNERS registry (`if name not in LEARNERS: continue`). The v5b
    smoke caught sccl_capprobe_strat being dropped that way — an entire
    experimental cell would have gone missing from the run with no error.
    Every learner named in any shipped config must be registered, and the
    runner must now fail loud on unknown names."""
    import glob
    import json
    from gcl.learners.learners import LEARNERS
    cfg_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    bad = []
    for path in sorted(glob.glob(os.path.join(cfg_dir, "*.json"))):
        cfg = json.load(open(path))
        for name in cfg.get("learners", []):
            if name not in LEARNERS:
                bad.append((os.path.basename(path), name))
    assert bad == [], f"unregistered learner names in configs: {bad}"


def test_run_experiment_rejects_unknown_learner():
    """The dispatch loop must raise (not skip) on an unknown learner name."""
    import inspect
    from gcl import experiment
    src = inspect.getsource(experiment.run_experiment)
    # structural assertion: the guard raises ValueError instead of `continue`
    assert "raise ValueError" in src and "unknown learner" in src


def test_v5b_learners_registered():
    from gcl.learners.learners import (LEARNERS, SCCLCapProbeLearner,
                                       SCCLCapProbeStratLearner, SCCLLearner)
    assert LEARNERS["sccl_capprobe"] is SCCLCapProbeLearner
    assert LEARNERS["sccl_capprobe_strat"] is SCCLCapProbeStratLearner
    assert issubclass(SCCLCapProbeLearner, SCCLLearner)
    assert issubclass(SCCLCapProbeStratLearner, SCCLLearner)


# ---------------------------------------------------------------------------
# SCCL v4/v5 base anchor — ENGINE-LEVEL engagement tests.
# The original v4 tests only checked config wiring and the audit trail, and a
# PEFT key mismatch (state-dict keys strip the adapter segment: 'lora_A.weight'
# vs named_parameters' 'lora_A.default.weight') made the anchor a silent no-op
# across the whole v4 ladder. These tests fail on that bug class.
# ---------------------------------------------------------------------------

def _tiny_anchor_engine(seed: int):
    """CPU scratch engine: tiny from-scratch GPT2 + PEFT LoRA + cached tokenizer.
    Never touches the real 3B checkpoint or the GPU."""
    import torch
    from types import SimpleNamespace
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
    from peft import LoraConfig, get_peft_model
    from gcl.engine import TrainingEngine

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mcfg = GPT2Config(vocab_size=len(tok), n_positions=64, n_embd=32,
                      n_layer=2, n_head=2)
    base = GPT2LMHeadModel(mcfg)
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                                            bias="none", target_modules=["c_attn"],
                                            task_type="CAUSAL_LM"))
    eng = TrainingEngine.__new__(TrainingEngine)  # skip HF model loading
    eng.cfg = SimpleNamespace(train_steps_per_update=3, learning_rate=3e-3,
                              max_seq_len=64)
    eng.device = "cpu"
    eng.tokenizer = tok
    eng.model = model
    eng._lora_base = None
    eng._fisher = None
    eng._anchor = None
    eng._replay = []
    eng.updates_done = 0
    eng._update_idx = 0
    return eng


def test_anchor_base_keys_match_named_parameters():
    """Regression: _base_anchor() keys must intersect named_parameters() names,
    else the penalty loop matches nothing and the anchor silently does nothing
    (the v4 ladder bug)."""
    eng = _tiny_anchor_engine(0)
    anchor = eng._base_anchor()
    np_names = {n for n, p in eng.model.named_parameters() if p.requires_grad}
    assert anchor, "base anchor snapshot is empty"
    assert set(anchor.keys()) == np_names, (
        "anchor keys must be exactly the trainable named_parameters names; "
        "PEFT state-dict keys strip the adapter segment and match nothing")


def test_anchor_actually_pulls_weights_toward_base():
    """Functional engagement: identical seed, data, and init; the anchored
    update must end strictly CLOSER to LoRA init than the unanchored one.
    A no-op anchor (zero penalty) makes the two distances equal."""
    import torch
    pairs = [{"prompt": "def add(a, b):\n", "target": "    return a + b\n"}]

    def run(anchor_lambda: float):
        eng = _tiny_anchor_engine(123)  # identical init each call
        init = {n: p.detach().clone() for n, p in eng.model.named_parameters()
                if p.requires_grad}
        torch.manual_seed(7)  # identical data/ordering path
        out = eng.apply_update([dict(p) for p in pairs], anchor_lambda=anchor_lambda)
        dist = sum(float((p - init[n]).pow(2).sum())
                   for n, p in eng.model.named_parameters() if p.requires_grad)
        return dist, out["anchor_pen"]

    d_free, ap_free = run(0.0)
    d_anch, ap_anch = run(5.0)
    assert d_free > 0.0, "unanchored update did not move the weights at all"
    assert d_anch < d_free, (
        f"anchor had no effect (dist anchored={d_anch:.6g} >= free={d_free:.6g}); "
        "the penalty term is not reaching the loss")
    # the per-update audit field must make (non-)engagement visible in the run
    # artifacts: 0.0 on every update is the v4 no-op signature.
    assert ap_free == 0.0
    assert ap_anch > 0.0, "anchor_pen must log > 0 when the anchor engages"


def test_v5_veto_default_path_unchanged():
    """Back-compat: the veto's DEFAULT pool must remain the pure recency window
    skills[-k:] so every pre-v5 row stays bit-identical. We verify by asserting
    the stratified branch is only taken when stratified=True is passed."""
    import inspect
    from gcl.vault import SelfCertVault
    sig = inspect.signature(SelfCertVault.selfreplay_veto)
    assert sig.parameters["stratified"].default is False, \
        "stratified must default to False (pre-v5 determinism)"
    src = inspect.getsource(SelfCertVault.selfreplay_veto)
    assert "skills[-check_skills:]" in src, "recency window must remain the default pool"


# ---------------------------------------------------------------------------
# SCCL v5b: capability probes (instance-vs-capability gap, gold-free)
# ---------------------------------------------------------------------------

_CAP_CODE = "def add2(a, b):\n    return a + b"
_CAP_TESTS_VARIANT = ["assert add2(5, 5) == 10", "assert add2(-3, 3) == 0"]


def test_commit_cap_probe_keeps_newest_per_family():
    v = SelfCertVault()
    assert v.commit_cap_probe("t1:c0", "famA", "spec A v1", "p1", _CAP_CODE,
                              _CAP_TESTS_VARIANT, 0.9)
    assert v.commit_cap_probe("t2:c0", "famA", "spec A v2", "p2", _CAP_CODE,
                              _CAP_TESTS_VARIANT, 0.9)
    assert v.commit_cap_probe("t3:c0", "famB", "spec B v1", "p3", _CAP_CODE,
                              _CAP_TESTS_VARIANT, 0.9)
    caps = [s for s in v._skills if s.kind == "cap_probe"]
    assert len(caps) == 2, "pool must keep only the newest cap_probe per family"
    by_fam = {s.family: s.task_id for s in caps}
    assert by_fam == {"famA": "t2:c0", "famB": "t3:c0"}, by_fam
    # ordinary skills must be untouched by the replacement pass
    v.commit_certified("t4", "famA", "plain skill", "p4", _CERT_CODE, _CERT_TESTS, 0.9)
    caps = [s for s in v._skills if s.kind == "cap_probe"]
    assert len(caps) == 2 and any(s.task_id == "t4" and s.kind == "skill"
                                  for s in v._skills)


def test_rrv_cap_probe_off_by_default():
    v, ver, eng = SelfCertVault(), Verifier(sandbox=PythonSandbox()), StubEngine()
    v.commit_cap_probe("t1:c0", "famA", "spec", "PROMPT\n```python\n", _CAP_CODE,
                       _CAP_TESTS_VARIANT, 0.9)
    eng.regen_code = "```python\ndef add2(a, b):\n    return a - b\n```"  # broken
    res = v.selfreplay_veto(eng, ver, check_skills=3, n_samples=2)
    assert not res["veto"] and res["checked_cap"] == 0, \
        "cap stratum must be inert unless check_cap_probes > 0"


def test_rrv_cap_probe_fires_on_capability_loss():
    """The instance-vs-capability gap: the memorized skill still regenerates
    fine, but the family's certified CAPABILITY variant breaks -> veto."""
    v, ver, eng = SelfCertVault(), Verifier(sandbox=PythonSandbox()), StubEngine()
    v.commit_certified("t1", "famA", "Return the sum.", "PROMPT fid:t1\n```python\n",
                       _CERT_CODE, _CERT_TESTS, 0.9)
    v.commit_cap_probe("t1:c0", "famA", "variant spec", "PROMPT v\n```python\n",
                       _CAP_CODE, _CAP_TESTS_VARIANT, 0.9)
    # regeneration passes the skill's OWN tests but fails the variant's tests
    eng.regen_code = ("```python\ndef add2(a, b):\n"
                      "    return 3 if (a, b) == (1, 2) else a - b\n```")
    res = v.selfreplay_veto(eng, ver, check_skills=3, n_samples=2,
                            check_cap_probes=1)
    assert res["checked"] == 1 and not res["broke"], "skill itself is retained"
    assert res["veto"] and "t1:c0" in res["broke_cap"], res
    assert res["checked_cap"] == 1


def test_rrv_cap_probe_checks_every_family():
    v, ver, eng = SelfCertVault(), Verifier(sandbox=PythonSandbox()), StubEngine()
    v.commit_cap_probe("tA:c0", "famA", "spec A", "PA\n```python\n", _CAP_CODE,
                       _CAP_TESTS_VARIANT, 0.9)
    v.commit_cap_probe("tB:c0", "famB", "spec B", "PB\n```python\n", _CAP_CODE,
                       _CAP_TESTS_VARIANT, 0.9)
    eng.regen_code = "```python\ndef add2(a, b):\n    return a + b\n```"
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                            check_cap_probes=1)
    assert not res["veto"] and res["checked_cap"] == 2, \
        "pool must police the newest cap_probe of EVERY family"


def test_rrv_cap_probe_math():
    v, ver = SelfCertVault(), Verifier(sandbox=PythonSandbox())
    eng = StubEngine()
    math_prompt = ("Solve and give ONLY the final numeric answer.\n"
                   "What is 6 times 7?\nAnswer: ")
    v.commit_cap_probe("m1:c0", "arith", "What is 6 times 7?", math_prompt,
                       "42", [], 0.86, domain="math")
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=3,
                            check_cap_probes=1)
    assert not res["veto"] and res["checked_cap"] == 1
    eng.math_answers = ["41", "41", "41"]
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=3,
                            check_cap_probes=1)
    assert res["veto"] and "m1:c0" in res["broke_cap"]
    eng.math_answers = ["The answer is 42.0", "41", "41"]  # canonical form
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=3,
                            check_cap_probes=1)
    assert not res["veto"]


class _MarginEngine:
    """First regeneration batch is broken; the margin re-check batch recovers."""

    def __init__(self):
        self.calls = 0

    def sample_candidates(self, prompt, n, temperature=0.7, top_p=0.95,
                          adapter_on=True, max_new_tokens=None):
        self.calls += 1
        if self.calls == 1:
            return ["```python\ndef add2(a, b):\n    return a - b\n```"] * max(1, n)
        return ["```python\ndef add2(a, b):\n    return a + b\n```"] * max(1, n)


def test_rrv_cap_probe_margin_recheck_saves_noisy_probe():
    """Bounded-damage guard: with cap_margin armed, a probe whose first batch
    fails gets extra regenerations before it may veto."""
    v, ver = SelfCertVault(), Verifier(sandbox=PythonSandbox())
    v.commit_cap_probe("t1:c0", "famA", "variant spec", "PROMPT v\n```python\n",
                       _CAP_CODE, _CAP_TESTS_VARIANT, 0.9)
    eng = _MarginEngine()
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                            check_cap_probes=1, cap_margin=0)
    assert res["veto"], "without the guard the noisy probe vetoes"
    eng = _MarginEngine()
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                            check_cap_probes=1, cap_margin=2)
    assert not res["veto"] and eng.calls == 2, \
        "margin re-check must recover a probe the first batch lost"


def test_v5b_cap_guard_arms_after_budget():
    """Env-level ledger: once cap-probe vetoes reach the budget fraction of a
    phase's update attempts, the gate arms the margin guard (audit field)."""
    from gcl.curriculum import Family
    cfg = ExperimentConfig(out_dir="runs/_test_sccl")
    cfg._learner_name = "sccl_cap"
    cfg.sccl_learners = ["sccl_cap"]
    cfg.sccl_nogate_learners = []
    cfg.sccl_gate_probe = 0
    cfg.sccl_capprobe_learners = ["sccl_cap"]
    cfg.sccl_capprobe_check = 1
    cfg.sccl_capprobe_budget = 0.5
    eng = FakeEngine(cfg, regen_passes=False)
    ver = Verifier(sandbox=PythonSandbox())
    vault = SelfCertVault()
    t1 = _poisoned_task()
    t2 = Task(task_id="t_gold_wrong_2", family="fam", domain="code",
              prompt="Return the sum of two integers a and b.",
              test_code="assert add2(2, 2) == 99",          # poisoned gold
              reference_answer="def add2(a, b):\n    return 99",
              entry_point="add2")
    env = GroundedContinualEnv(cfg, eng, ver,
                               [Family(name="fam", tasks=[t1, t2], holdout=[])],
                               holdout=[], vault=vault, sccl=True)
    vault.commit_cap_probe("seed:c0", "fam", "variant spec",
                           "PROMPT v\n```python\n", _CAP_CODE,
                           _CAP_TESTS_VARIANT, 0.9)
    # broken regen (regen_passes=False) fails the cap probe on every attempt
    _, _, _, s1 = env.step(Action(answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
                                  metadata={"sccl": _cert_meta(found=True)}))
    g1 = s1["update_info"]["gate"]
    assert g1["veto"] and g1["broke_cap"], g1
    assert g1["cap_guard_armed"] is False, "guard cannot arm on the first attempt"
    _, _, _, s2 = env.step(Action(answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
                                  metadata={"sccl": _cert_meta(found=True)}))
    g2 = s2["update_info"]["gate"]
    assert g2["cap_guard_armed"] is True, \
        "100% cap-veto rate >= 0.5 budget must arm the guard"


def test_v5b_default_config_is_inert():
    """All v5b switches default OFF so pre-v5b rows stay bit-identical."""
    cfg = ExperimentConfig(out_dir="runs/_test_sccl")
    assert cfg.sccl_capprobe_learners == []
    assert cfg.sccl_capprobes == 0
    assert cfg.sccl_capprobe_check == 0
    import inspect
    from gcl.vault import SelfCertVault
    sig = inspect.signature(SelfCertVault.selfreplay_veto)
    assert sig.parameters["check_cap_probes"].default == 0
    assert sig.parameters["cap_margin"].default == 0


# ---------------------------------------------------------------------------
# SCCL v6 (Branch D): capability-probe ENSEMBLE pool.
# The v5b telemetry (runs/sccl_v5b/telemetry_arith_erosion.json) measured 100%
# probe insensitivity: every probe-checked arith erosion PASSED its single
# newest-per-family cap probe. The cause is skill-heterogeneity blindness — a
# heterogeneous family has MULTIPLE skill axes and one newest-per-family probe
# witnesses only ONE axis, so an update that destroys an unwitnessed axis sails
# through the gate. Branch D generalizes the pool to an ENSEMBLE of up to K
# distinct-source-skill probes per family. These tests pin two guarantees:
#   (a) K=1 reduces EXACTLY to the v5b newest-per-family rule (bit-identity, so
#       the v5b determinism rows are untouched), and
#   (b) K>1 keeps the freshest variant of K DISTINCT source skills and the veto
#       re-checks ALL of them (any break -> veto).
# ---------------------------------------------------------------------------

def test_cap_probe_source_parse():
    from gcl.vault import SelfCertVault
    assert SelfCertVault._cap_probe_source("mbpp_604:c0") == "mbpp_604"
    assert SelfCertVault._cap_probe_source("mbpp_604:c7") == "mbpp_604"
    assert SelfCertVault._cap_probe_source("math_word_x:c1") == "math_word_x"
    # a plain skill id (no ":c" suffix) is its own source
    assert SelfCertVault._cap_probe_source("mbpp_604") == "mbpp_604"


def test_cap_pool_pool1_is_v5b_newest_per_family():
    """K=1 must reduce exactly to v5b: the single NEWEST cap_probe per family,
    regardless of how many distinct source skills / variants are stored. This is
    the bit-identity guarantee that keeps every v5b row reproducible."""
    v = SelfCertVault()
    # commit order: A:c0, B:c0, A:c1, C:c0, B:c1  (A,B,C distinct source skills)
    for tid in ["A:c0", "B:c0", "A:c1", "C:c0", "B:c1"]:
        assert v.commit_cap_probe(tid, "famA", f"spec {tid}", f"p-{tid}",
                                  _CAP_CODE, _CAP_TESTS_VARIANT, 0.9, pool=99)
    pool = v._cap_pool_by_family(1)
    assert list(pool) == ["famA"]
    assert [s.task_id for s in pool["famA"]] == ["B:c1"], \
        "K=1 must select exactly the globally-newest cap_probe (v5b rule)"


def test_cap_pool_ensemble_keeps_distinct_source_skills():
    """K=3 keeps up to three DISTINCT source skills per family; a family with
    only one source skill yields a one-probe pool even under K=3."""
    v = SelfCertVault()
    for tid in ["A:c0", "B:c0", "C:c0", "D:c0"]:
        assert v.commit_cap_probe(tid, "famA", f"spec {tid}", f"p-{tid}",
                                  _CAP_CODE, _CAP_TESTS_VARIANT, 0.9, pool=99)
    pool = v._cap_pool_by_family(3)
    ids = [s.task_id for s in pool["famA"]]
    assert len(ids) == 3
    # freshest-variant-per-skill then newest-by-commit => B,C,D (A is oldest)
    assert ids == ["B:c0", "C:c0", "D:c0"], ids
    assert len({SelfCertVault._cap_probe_source(t) for t in ids}) == 3, \
        "ensemble must span DISTINCT source skills"
    # single-skill family: K=3 still yields one probe (nothing to ensemble over)
    v2 = SelfCertVault()
    for tid in ["X:c0", "X:c1", "X:c2"]:
        assert v2.commit_cap_probe(tid, "famB", f"spec {tid}", f"p-{tid}",
                                   _CAP_CODE, _CAP_TESTS_VARIANT, 0.9, pool=99)
    pool2 = v2._cap_pool_by_family(3)
    assert [s.task_id for s in pool2["famB"]] == ["X:c2"], \
        "multiple variants of ONE skill collapse to the freshest variant"


def test_cap_pool_ensemble_freshest_variant_per_skill():
    """Within a source skill only the FRESHEST variant is eligible, so the
    ensemble witnesses each skill axis with its latest certified evidence."""
    v = SelfCertVault()
    for tid in ["A:c0", "B:c0", "A:c1", "A:c2"]:
        assert v.commit_cap_probe(tid, "famA", f"spec {tid}", f"p-{tid}",
                                  _CAP_CODE, _CAP_TESTS_VARIANT, 0.9, pool=99)
    pool = v._cap_pool_by_family(3)
    ids = [s.task_id for s in pool["famA"]]
    assert "A:c2" in ids and "A:c0" not in ids and "A:c1" not in ids, \
        f"only the freshest variant of A may appear, got {ids}"
    assert ids == ["B:c0", "A:c2"], ids


def test_commit_cap_probe_ensemble_prunes_to_k():
    """commit_cap_probe(pool=K) must prune the family pool down to K distinct
    source skills after every admission (bounded by construction)."""
    v = SelfCertVault()
    for tid in ["A:c0", "B:c0", "C:c0", "D:c0"]:
        assert v.commit_cap_probe(tid, "famA", f"spec {tid}", f"p-{tid}",
                                  _CAP_CODE, _CAP_TESTS_VARIANT, 0.9, pool=3)
    caps = [s for s in v._skills if s.kind == "cap_probe"]
    assert len(caps) == 3, "pool must be pruned to K=3 distinct skills"
    assert {s.task_id for s in caps} == {"B:c0", "C:c0", "D:c0"}
    # a fresh variant of an IN-POOL skill replaces it in place, no growth
    assert v.commit_cap_probe("B:c1", "famA", "spec B:c1", "p-B:c1",
                              _CAP_CODE, _CAP_TESTS_VARIANT, 0.9, pool=3)
    caps = [s for s in v._skills if s.kind == "cap_probe"]
    assert len(caps) == 3
    assert {s.task_id for s in caps} == {"B:c1", "C:c0", "D:c0"}


class _PerPromptEngine:
    """Regeneration engine whose code depends on a marker in the prompt, so a
    test can break exactly one cap probe while the others pass."""

    def __init__(self, break_marker=None):
        self.break_marker = break_marker

    def sample_candidates(self, prompt, n, temperature=0.7, top_p=0.95,
                          adapter_on=True, max_new_tokens=None):
        if self.break_marker and self.break_marker in prompt:
            return ["```python\ndef add2(a, b):\n    return a - b\n```"] * max(1, n)
        return ["```python\ndef add2(a, b):\n    return a + b\n```"] * max(1, n)


def test_rrv_cap_ensemble_checks_all_and_vetoes_any_break():
    """The ensemble stratum must re-check EVERY pooled probe (checked_cap == K)
    and veto if ANY one breaks — this is the wider witness v5b lacked."""
    v, ver = SelfCertVault(), Verifier(sandbox=PythonSandbox())
    for tid, marker in [("A:c0", "MA"), ("B:c0", "MB"), ("C:c0", "MC")]:
        assert v.commit_cap_probe(tid, "famA", f"spec {tid}",
                                  f"PROMPT {marker}\n```python\n",
                                  _CAP_CODE, _CAP_TESTS_VARIANT, 0.9, pool=3)
    # all probes regenerate correct code -> no veto, all three checked
    eng = _PerPromptEngine(break_marker=None)
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                            check_cap_probes=1, cap_pool=3)
    assert not res["veto"] and res["checked_cap"] == 3, res
    # break ONLY probe B -> veto names it, the other two still pass
    eng = _PerPromptEngine(break_marker="MB")
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                            check_cap_probes=1, cap_pool=3)
    assert res["veto"] and res["broke_cap"] == ["B:c0"], res
    assert res["checked_cap"] == 3, "all pooled probes must be re-checked"


def test_rrv_cap_pool_default_is_v5b_single_probe():
    """With cap_pool=1 (default) the veto checks only the newest probe per
    family — the exact v5b stratum. checked_cap must equal #families, not K."""
    v, ver = SelfCertVault(), Verifier(sandbox=PythonSandbox())
    # store 3 distinct skills but read the pool with the K=1 default
    for tid, marker in [("A:c0", "MA"), ("B:c0", "MB"), ("C:c0", "MC")]:
        assert v.commit_cap_probe(tid, "famA", f"spec {tid}",
                                  f"PROMPT {marker}\n```python\n",
                                  _CAP_CODE, _CAP_TESTS_VARIANT, 0.9, pool=99)
    eng = _PerPromptEngine(break_marker=None)
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                            check_cap_probes=1)  # cap_pool defaults to 1
    assert res["checked_cap"] == 1, \
        "cap_pool=1 must check exactly one (newest) probe per family, v5b-style"
    assert not res["veto"]


def test_v6_default_config_is_inert():
    """All v6 ensemble switches default to the v5b configuration (pool=1, no
    per-learner overrides) so pre-v6 rows stay bit-identical."""
    import inspect
    from gcl.vault import SelfCertVault
    cfg = ExperimentConfig(out_dir="runs/_test_sccl")
    assert cfg.sccl_capprobe_pool == 1
    assert cfg.sccl_capprobe_pools == {}
    sig = inspect.signature(SelfCertVault.selfreplay_veto)
    assert sig.parameters["cap_pool"].default == 1
    csig = inspect.signature(SelfCertVault.commit_cap_probe)
    assert csig.parameters["pool"].default == 1


def test_v6_learners_registered():
    from gcl.learners.learners import (LEARNERS, SCCLLearner, SCCLCapEnsLearner,
                                       SCCLCapAnchorLearner, SCCLCapEnsAnchorLearner)
    assert LEARNERS["sccl_capens"] is SCCLCapEnsLearner
    assert LEARNERS["sccl_cap_anchor"] is SCCLCapAnchorLearner
    assert LEARNERS["sccl_capens_anchor"] is SCCLCapEnsAnchorLearner
    for cls in (SCCLCapEnsLearner, SCCLCapAnchorLearner, SCCLCapEnsAnchorLearner):
        assert issubclass(cls, SCCLLearner)


# ---------------------------------------------------------------------------
# SCCL v7 (Branch E, E1): pass-rate margin veto.
# v6 telemetry (F1) measured 100% probe insensitivity under the any-of-n
# retain rule: every probe-checked erosion passed its probes, because a probe
# whose regeneration quality degrades to 50% still "passes" as long as ONE of
# n draws survives. E1 measures the pass RATE over n regeneration draws per
# cap probe and retains iff rate >= cap_retain_min (theta). These tests pin:
#   (a) theta<=0 executes the EXACT legacy any-pass path (bit-identity for all
#       pre-v7 rows: same draws, same verdicts, no cap_rates in the result),
#   (b) theta>0 vetoes on sub-threshold rate and reports per-probe rates,
#   (c) under the bounded-damage guard the margin draws POOL into the rate
#       (rate-consistent relief, not an any-pass escape hatch), and
#   (d) the env wiring persists cap_rates + dose on the gate record.
# ---------------------------------------------------------------------------

_GOOD_CODE = "```python\ndef add2(a, b):\n    return a + b\n```"
_BAD_CODE = "```python\ndef add2(a, b):\n    return a - b\n```"


class _RateEngine:
    """Regeneration engine that returns a scripted draw pattern per prompt.

    `pattern` cycles over draws: pattern[i % len(pattern)] is draw i. Records
    every requested batch size so tests can pin the number of draws.
    """

    def __init__(self, pattern):
        self.pattern = list(pattern)
        self.calls = []
        self._draw = 0

    def sample_candidates(self, prompt, n, temperature=0.7, top_p=0.95,
                          adapter_on=True, max_new_tokens=None):
        n = max(1, n)
        self.calls.append(n)
        out = [self.pattern[(self._draw + j) % len(self.pattern)]
               for j in range(n)]
        self._draw += n
        return out


def _cap_vault():
    v = SelfCertVault()
    assert v.commit_cap_probe("t1:c0", "famA", "variant spec",
                              "PROMPT v\n```python\n", _CAP_CODE,
                              _CAP_TESTS_VARIANT, 0.9)
    return v


def test_v7_default_config_is_inert():
    """All E1 switches default OFF (theta=0) so pre-v7 rows stay bit-identical."""
    import inspect
    from gcl.vault import SelfCertVault
    cfg = ExperimentConfig(out_dir="runs/_test_sccl")
    assert cfg.sccl_cap_retain_min == 0.0
    assert cfg.sccl_cap_retain_mins == {}
    assert cfg.sccl_cap_samples == 0
    assert cfg.sccl_cap_samples_map == {}
    sig = inspect.signature(SelfCertVault.selfreplay_veto)
    assert sig.parameters["cap_retain_min"].default == 0.0
    assert sig.parameters["cap_samples"].default == 0


def test_rrv_e1_theta_zero_is_exact_legacy_path():
    """theta=0 must behave EXACTLY like v5b/v6 even when cap_samples is set:
    same draw count (n_samples, not cap_samples), any-of-n retain rule, and
    NO cap_rates key in the result (legacy gate-record shape)."""
    ver = Verifier(sandbox=PythonSandbox())
    # 1-of-3 draws broken: any-pass retains, a strict rule would veto
    eng = _RateEngine([_GOOD_CODE, _GOOD_CODE, _BAD_CODE])
    res = _cap_vault().selfreplay_veto(eng, ver, check_skills=0, n_samples=3,
                                       check_cap_probes=1, cap_samples=3)
    assert not res["veto"], "theta=0 keeps the any-of-n retain rule"
    assert "cap_rates" not in res, "theta=0 must not add E1 telemetry"
    assert eng.calls == [3], "theta=0 draws n_samples, ignoring cap_samples"
    # cap_samples set to a different value: still ignored at theta=0
    eng = _RateEngine([_GOOD_CODE, _GOOD_CODE, _BAD_CODE])
    res = _cap_vault().selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                                       check_cap_probes=1, cap_samples=3)
    assert not res["veto"] and "cap_rates" not in res
    assert eng.calls == [2], "legacy path must request exactly n_samples draws"


def test_rrv_e1_strict_vetoes_partial_pass():
    """theta=1.0 (strict): a probe at 1/3 health passed under v6's any-pass
    rule must now veto, with its measured rate reported for telemetry."""
    ver = Verifier(sandbox=PythonSandbox())
    eng = _RateEngine([_GOOD_CODE, _BAD_CODE, _BAD_CODE])
    res = _cap_vault().selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                                       check_cap_probes=1,
                                       cap_retain_min=1.0, cap_samples=3)
    assert res["veto"] and res["broke_cap"] == ["t1:c0"], res
    assert res["checked_cap"] == 1
    assert res["cap_rates"] == {"t1:c0": {"passes": 1, "n": 3}}, res["cap_rates"]
    assert eng.calls == [3], "E1 draws cap_samples, not n_samples"
    # all draws pass -> retained, rate 3/3
    eng = _RateEngine([_GOOD_CODE])
    res = _cap_vault().selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                                       check_cap_probes=1,
                                       cap_retain_min=1.0, cap_samples=3)
    assert not res["veto"]
    assert res["cap_rates"] == {"t1:c0": {"passes": 3, "n": 3}}


def test_rrv_e1_majority_threshold():
    """theta=2/3 (majority): 2/3 draws passing is retained; 1/3 vetoes. This
    is the dose-response pair the v7 ladder A/Bs against the strict dose."""
    ver = Verifier(sandbox=PythonSandbox())
    eng = _RateEngine([_GOOD_CODE, _GOOD_CODE, _BAD_CODE])
    res = _cap_vault().selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                                       check_cap_probes=1,
                                       cap_retain_min=2 / 3, cap_samples=3)
    assert not res["veto"], "2/3 rate clears the majority threshold"
    assert res["cap_rates"] == {"t1:c0": {"passes": 2, "n": 3}}
    eng = _RateEngine([_GOOD_CODE, _BAD_CODE, _BAD_CODE])
    res = _cap_vault().selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                                       check_cap_probes=1,
                                       cap_retain_min=2 / 3, cap_samples=3)
    assert res["veto"] and res["broke_cap"] == ["t1:c0"], \
        "1/3 rate must veto under the majority threshold"


def test_rrv_e1_margin_pools_into_rate():
    """Guard semantics under E1: the margin re-check draws POOL into the rate
    (rate-consistent relief), they are NOT an any-pass escape hatch."""
    ver = Verifier(sandbox=PythonSandbox())
    # 2/3 < theta=0.8 -> guard margin of 2 good draws -> pooled 4/5 = 0.8 OK
    eng = _RateEngine([_GOOD_CODE, _GOOD_CODE, _BAD_CODE, _GOOD_CODE, _GOOD_CODE])
    res = _cap_vault().selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                                       check_cap_probes=1, cap_margin=2,
                                       cap_retain_min=0.8, cap_samples=3)
    assert not res["veto"], "pooled margin rate 4/5 clears theta=0.8"
    assert eng.calls == [3, 2], "margin re-check draws exactly cap_margin"
    assert res["cap_rates"] == {"t1:c0": {"passes": 4, "n": 5}}
    # 2/3 < theta=0.8, margin draws still leave pooled 3/5 = 0.6 -> veto
    eng = _RateEngine([_GOOD_CODE, _GOOD_CODE, _BAD_CODE, _GOOD_CODE, _BAD_CODE])
    res = _cap_vault().selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                                       check_cap_probes=1, cap_margin=2,
                                       cap_retain_min=0.8, cap_samples=3)
    assert res["veto"] and res["broke_cap"] == ["t1:c0"], \
        "pooled margin may not rescue a rate still below theta"
    assert res["cap_rates"] == {"t1:c0": {"passes": 3, "n": 5}}


def test_rrv_e1_extraction_failure_counts_as_nonpass():
    """A draw that yields no extractable code counts against the rate — the
    denominator is draws REQUESTED, so the rate measures behaviour, not the
    extraction yield of surviving candidates."""
    ver = Verifier(sandbox=PythonSandbox())
    eng = _RateEngine([_GOOD_CODE, "no code here at all", _GOOD_CODE])
    res = _cap_vault().selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                                       check_cap_probes=1,
                                       cap_retain_min=1.0, cap_samples=3)
    assert res["veto"], "an unextractable draw must break the strict rate"
    assert res["cap_rates"] == {"t1:c0": {"passes": 2, "n": 3}}


def test_rrv_e1_math_rate():
    """E1 applies to math probes too: rate = matching draws / requested."""
    ver = Verifier(sandbox=PythonSandbox())
    math_prompt = ("Solve and give ONLY the final numeric answer.\n"
                   "What is 6 times 7?\nAnswer: ")
    v = SelfCertVault()
    assert v.commit_cap_probe("m1:c0", "arith", "What is 6 times 7?",
                              math_prompt, "42", [], 0.86, domain="math")
    eng = StubEngine()
    eng.math_answers = ["42", "41", "42"]
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                            check_cap_probes=1,
                            cap_retain_min=1.0, cap_samples=3)
    assert res["veto"] and res["broke_cap"] == ["m1:c0"], \
        "one wrong math draw must veto under the strict rate"
    assert res["cap_rates"] == {"m1:c0": {"passes": 2, "n": 3}}
    eng = StubEngine()
    eng.math_answers = ["42", "41", "42"]
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                            check_cap_probes=1,
                            cap_retain_min=2 / 3, cap_samples=3)
    assert not res["veto"], "2/3 math rate clears the majority threshold"


def test_e1_env_wiring_logs_dose_and_rates():
    """Env-level wiring: a theta>0 learner's gate record persists the dose
    (cap_retain_min, cap_n) and per-probe cap_rates; a theta=0 learner's gate
    record carries NONE of the E1 fields (legacy shape)."""
    from gcl.curriculum import Family
    cfg = ExperimentConfig(out_dir="runs/_test_sccl")
    cfg._learner_name = "sccl_e1"
    cfg.sccl_learners = ["sccl_e1"]
    cfg.sccl_nogate_learners = []
    cfg.sccl_gate_probe = 0
    cfg.sccl_capprobe_learners = ["sccl_e1"]
    cfg.sccl_capprobe_check = 1
    cfg.sccl_cap_retain_mins = {"sccl_e1": 1.0}
    cfg.sccl_cap_samples_map = {"sccl_e1": 3}
    eng = FakeEngine(cfg, regen_passes=True)
    ver = Verifier(sandbox=PythonSandbox())
    vault = SelfCertVault()
    t1 = _poisoned_task()
    t2 = Task(task_id="t_e1_ok", family="fam", domain="code",
              prompt="Return the sum of two integers a and b.",
              test_code="assert add2(2, 2) == 4",
              reference_answer="def add2(a, b):\n    return 4",
              entry_point="add2")
    env = GroundedContinualEnv(cfg, eng, ver,
                               [Family(name="fam", tasks=[t1, t2], holdout=[])],
                               holdout=[], vault=vault, sccl=True)
    vault.commit_cap_probe("seed:c0", "fam", "variant spec",
                           "PROMPT v\n```python\n", _CAP_CODE,
                           _CAP_TESTS_VARIANT, 0.9)
    _, _, _, s1 = env.step(Action(answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
                                  metadata={"sccl": _cert_meta(found=True)}))
    g1 = s1["update_info"]["gate"]
    assert g1.get("cap_retain_min") == 1.0, g1
    assert g1.get("cap_n") == 3, g1
    assert g1.get("cap_rates", {}).get("seed:c0") == {"passes": 3, "n": 3}, g1
    assert not g1["veto"], "all-pass regen must clear the strict rate"
    # theta=0 learner: no E1 fields on the gate record
    cfg2 = ExperimentConfig(out_dir="runs/_test_sccl")
    cfg2._learner_name = "sccl_legacy"
    cfg2.sccl_learners = ["sccl_legacy"]
    cfg2.sccl_nogate_learners = []
    cfg2.sccl_gate_probe = 0
    cfg2.sccl_capprobe_learners = ["sccl_legacy"]
    cfg2.sccl_capprobe_check = 1
    eng2 = FakeEngine(cfg2, regen_passes=True)
    env2 = GroundedContinualEnv(cfg2, eng2, ver,
                                [Family(name="fam", tasks=[t1, t2], holdout=[])],
                                holdout=[], vault=SelfCertVault(), sccl=True)
    env2.vault.commit_cap_probe("seed:c0", "fam", "variant spec",
                                "PROMPT v\n```python\n", _CAP_CODE,
                                _CAP_TESTS_VARIANT, 0.9)
    _, _, _, s2 = env2.step(Action(answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
                                   metadata={"sccl": _cert_meta(found=True)}))
    g2 = s2["update_info"]["gate"]
    assert "cap_rates" not in g2 and "cap_retain_min" not in g2 \
        and "cap_n" not in g2, "theta=0 gate record must keep the legacy shape"


def test_v7_learners_registered():
    from gcl.learners.learners import (LEARNERS, SCCLLearner, SCCLStrictLearner,
                                       SCCLMajorityLearner, SCCLEnsStrictLearner,
                                       SCCLEnsStrictAnchorLearner)
    assert LEARNERS["sccl_strict"] is SCCLStrictLearner
    assert LEARNERS["sccl_majority"] is SCCLMajorityLearner
    assert LEARNERS["sccl_ens_strict"] is SCCLEnsStrictLearner
    assert LEARNERS["sccl_ens_strict_anchor"] is SCCLEnsStrictAnchorLearner
    for cls in (SCCLStrictLearner, SCCLMajorityLearner,
                SCCLEnsStrictLearner, SCCLEnsStrictAnchorLearner):
        assert issubclass(cls, SCCLLearner)


# ---------------------------------------------------------------------------
# SCCL v8 (Branch F, G1): generalization witnesses.
# v7 telemetry (F5/F8): certified-skill probes live on the memorized manifold
# — every erosion passed at the pooled rate because the witnesses only test
# REPRODUCTION of trained skills. G1 manufactures witnesses from UNTRAINED
# future stream tasks via spec-only certify (task_id suffix ':g'), checks them
# with the same E1 rate machinery, keeps them in a DEDICATED pool lane, and
# RETIREs each witness before its source task enters training (it would
# degrade to a reproduction witness and contaminate the coverage claim).
# These tests pin:
#   (a) the two-lane pool: ':g' slots are never evicted by certified commits,
#   (b) gen_pool=0 executes the exact legacy pool (bit-identity),
#   (c) retire_cap_probe removes the exact task_id,
#   (d) the veto re-checks ':g' witnesses, vetoes on break, reports rates and
#       emits checked_cap_ids ONLY when cap_gen_pool>0,
#   (e) default-config inertness + learner registration,
#   (f) env wiring persists cap_gen_pool + checked_cap_ids on gen rows only.
# ---------------------------------------------------------------------------


def test_cap_pool_gen_lane_dedicated():
    """The ':g' lane is dedicated: certified commits never evict a gen
    witness, and the gen lane keeps the newest gen_pool per family."""
    v = SelfCertVault()
    assert v.commit_cap_probe("A:c0", "famA", "spec A", "PROMPT MA\n```python\n",
                              _CAP_CODE, _CAP_TESTS_VARIANT, 0.9,
                              pool=1, gen_pool=1)
    assert v.commit_cap_probe("g1:g", "famA", "spec g1", "PROMPT G1\n```python\n",
                              _CAP_CODE, _CAP_TESTS_VARIANT, 0.9,
                              pool=1, gen_pool=1)
    assert v.commit_cap_probe("g2:g", "famA", "spec g2", "PROMPT G2\n```python\n",
                              _CAP_CODE, _CAP_TESTS_VARIANT, 0.9,
                              pool=1, gen_pool=1)
    pool = v._cap_pool_by_family(1, gen_pool=1)["famA"]
    assert [s.task_id for s in pool] == ["A:c0", "g2:g"], \
        "certified lane first (newest skill), then the newest gen witness"
    # a newer certified skill replaces the older one; the gen lane is untouched
    assert v.commit_cap_probe("B:c0", "famA", "spec B", "PROMPT MB\n```python\n",
                              _CAP_CODE, _CAP_TESTS_VARIANT, 0.9,
                              pool=1, gen_pool=1)
    pool = v._cap_pool_by_family(1, gen_pool=1)["famA"]
    assert [s.task_id for s in pool] == ["B:c0", "g2:g"]
    ids = {s.task_id for s in v._skills if getattr(s, "kind", "") == "cap_probe"}
    assert "g2:g" in ids, "a certified commit must never evict a ':g' witness"
    assert "g1:g" not in ids, "the gen lane evicts oldest-first on its own"


def test_cap_pool_gen_pool_zero_is_exact_legacy():
    """gen_pool=0 executes the exact pre-v8 rule (bit-identity for all
    pre-v8 rows): a ':g' id, if present, is grouped like any other distinct
    source skill instead of getting a dedicated lane."""
    v = SelfCertVault()
    for tid in ("A:c0", "B:c0"):
        assert v.commit_cap_probe(tid, "famA", f"spec {tid}",
                                  f"PROMPT {tid}\n```python\n",
                                  _CAP_CODE, _CAP_TESTS_VARIANT, 0.9,
                                  pool=99, gen_pool=99)
    assert v.commit_cap_probe("g1:g", "famA", "spec g1", "PROMPT g1\n```python\n",
                              _CAP_CODE, _CAP_TESTS_VARIANT, 0.9,
                              pool=99, gen_pool=99)
    legacy = v._cap_pool_by_family(2)            # gen_pool defaults to 0
    explicit = v._cap_pool_by_family(2, 0)
    assert [s.task_id for s in legacy["famA"]] == \
        [s.task_id for s in explicit["famA"]]
    assert [s.task_id for s in legacy["famA"]] == ["B:c0", "g1:g"], \
        "legacy rule: newest 2 distinct sources by commit order"
    # the two-lane read keeps BOTH certified skills PLUS the gen witness
    two_lane = v._cap_pool_by_family(2, gen_pool=1)
    assert [s.task_id for s in two_lane["famA"]] == ["A:c0", "B:c0", "g1:g"]


def test_retire_cap_probe_removes_exact_id():
    """Retirement removes exactly the cap probe whose source task entered
    training — the anti-contamination step of the G1 protocol."""
    v = SelfCertVault()
    assert v.commit_cap_probe("A:c0", "famA", "spec A", "PROMPT A\n```python\n",
                              _CAP_CODE, _CAP_TESTS_VARIANT, 0.9,
                              pool=3, gen_pool=2)
    assert v.commit_cap_probe("g1:g", "famA", "spec g1", "PROMPT g1\n```python\n",
                              _CAP_CODE, _CAP_TESTS_VARIANT, 0.9,
                              pool=3, gen_pool=2)
    assert v.commit_cap_probe("g2:g", "famA", "spec g2", "PROMPT g2\n```python\n",
                              _CAP_CODE, _CAP_TESTS_VARIANT, 0.9,
                              pool=3, gen_pool=2)
    assert v.retire_cap_probe("g1:g") is True
    ids = [s.task_id for s in v._skills if getattr(s, "kind", "") == "cap_probe"]
    assert ids == ["A:c0", "g2:g"], "exact-id removal, other probes untouched"
    pool = v._cap_pool_by_family(3, gen_pool=2)["famA"]
    assert [s.task_id for s in pool] == ["A:c0", "g2:g"]
    assert v.retire_cap_probe("g1:g") is False    # already retired
    assert v.retire_cap_probe("nope:g") is False  # unknown id


def test_rrv_gen_probe_checked_and_vetoes():
    """The veto re-checks the ':g' lane: a broken gen witness vetoes and is
    named; checked_cap_ids is emitted ONLY when cap_gen_pool>0."""
    v, ver = SelfCertVault(), Verifier(sandbox=PythonSandbox())
    assert v.commit_cap_probe("A:c0", "famA", "spec A", "PROMPT MA\n```python\n",
                              _CAP_CODE, _CAP_TESTS_VARIANT, 0.9,
                              pool=1, gen_pool=1)
    assert v.commit_cap_probe("gsrc:g", "famA", "spec g", "PROMPT MG\n```python\n",
                              _CAP_CODE, _CAP_TESTS_VARIANT, 0.9,
                              pool=1, gen_pool=1)
    eng = _PerPromptEngine(break_marker=None)
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                            check_cap_probes=1, cap_pool=1, cap_gen_pool=1)
    assert not res["veto"] and res["checked_cap"] == 2, res
    assert res["checked_cap_ids"] == ["A:c0", "gsrc:g"], \
        "certified lane checked first, then the gen lane, in check order"
    # break ONLY the gen witness -> veto names it, the certified probe passes
    eng = _PerPromptEngine(break_marker="MG")
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                            check_cap_probes=1, cap_pool=1, cap_gen_pool=1)
    assert res["veto"] and res["broke_cap"] == ["gsrc:g"], res
    assert res["checked_cap_ids"] == ["A:c0", "gsrc:g"]
    # cap_gen_pool=0 -> pre-v8 result shape: no checked_cap_ids key
    eng = _PerPromptEngine(break_marker=None)
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                            check_cap_probes=1, cap_pool=1)
    assert "checked_cap_ids" not in res, \
        "pre-v8 gate records must keep their exact shape"


def test_rrv_gen_probe_rate_under_theta():
    """Gen witnesses are rate-checked exactly like certified probes at
    theta>0: their ':g' id carries a cap_rates entry (the v8 H1 mechanism)."""
    v, ver = SelfCertVault(), Verifier(sandbox=PythonSandbox())
    assert v.commit_cap_probe("gsrc:g", "famA", "spec g", "PROMPT MG\n```python\n",
                              _CAP_CODE, _CAP_TESTS_VARIANT, 0.9,
                              pool=1, gen_pool=1)
    eng = _RateEngine([_GOOD_CODE, _BAD_CODE, _BAD_CODE])
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                            check_cap_probes=1, cap_pool=1, cap_gen_pool=1,
                            cap_retain_min=1.0, cap_samples=3)
    assert res["veto"], "a 1/3 pass rate must fail the strict threshold"
    assert res["cap_rates"]["gsrc:g"]["passes"] == 1
    assert res["cap_rates"]["gsrc:g"]["n"] == 3


def test_rrv_gen_math_probe_checked():
    """A math ':g' witness is checked via the canonical-number path and can
    veto alone (the math_word family's gen lane)."""
    ver = Verifier(sandbox=PythonSandbox())
    v = SelfCertVault()
    math_prompt = ("Solve and give ONLY the final numeric answer.\n"
                   "What is 6 times 7?\nAnswer: ")
    assert v.commit_cap_probe("msrc:g", "arith", "untrained math spec",
                              math_prompt, "42", [], 0.86,
                              domain="math", pool=1, gen_pool=1)
    eng = StubEngine()
    eng.math_answers = ["42", "42"]
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                            check_cap_probes=1, cap_pool=1, cap_gen_pool=1)
    assert not res["veto"] and res["checked_cap_ids"] == ["msrc:g"], res
    eng = StubEngine()
    eng.math_answers = ["41", "41"]
    res = v.selfreplay_veto(eng, ver, check_skills=0, n_samples=2,
                            check_cap_probes=1, cap_pool=1, cap_gen_pool=1)
    assert res["veto"] and res["broke_cap"] == ["msrc:g"], res


def test_v8_default_config_is_inert():
    """All G1 switches default OFF so pre-v8 rows stay bit-identical."""
    import inspect
    from gcl.vault import SelfCertVault
    cfg = ExperimentConfig(out_dir="runs/_test_sccl")
    assert cfg.sccl_genprobe_learners == []
    assert cfg.sccl_genprobes == 0
    sig = inspect.signature(SelfCertVault.selfreplay_veto)
    assert sig.parameters["cap_gen_pool"].default == 0
    psig = inspect.signature(SelfCertVault._cap_pool_by_family)
    assert psig.parameters["gen_pool"].default == 0
    csig = inspect.signature(SelfCertVault.commit_cap_probe)
    assert csig.parameters["gen_pool"].default == 0


def test_v8_learners_registered():
    from gcl.learners.learners import (LEARNERS, SCCLLearner,
                                       SCCLGenProbeLearner,
                                       SCCLGenProbeStrictLearner,
                                       SCCLGenProbeStrictAnchorLearner)
    assert LEARNERS["sccl_genprobe"] is SCCLGenProbeLearner
    assert LEARNERS["sccl_genprobe_strict"] is SCCLGenProbeStrictLearner
    assert LEARNERS["sccl_genprobe_strict_anchor"] is SCCLGenProbeStrictAnchorLearner
    for cls in (SCCLGenProbeLearner, SCCLGenProbeStrictLearner,
                SCCLGenProbeStrictAnchorLearner):
        assert issubclass(cls, SCCLLearner)


def test_v8_env_wiring_logs_gen_lane_and_checked_ids():
    """Env-level wiring: a gen row's gate record persists cap_gen_pool and
    the exact checked_cap_ids (the C2/C3 audit hooks); a non-gen row's gate
    record carries NEITHER field (pre-v8 shape)."""
    from gcl.curriculum import Family
    cfg = ExperimentConfig(out_dir="runs/_test_sccl")
    cfg._learner_name = "sccl_genprobe"
    cfg.sccl_learners = ["sccl_genprobe"]
    cfg.sccl_nogate_learners = []
    cfg.sccl_gate_probe = 0
    cfg.sccl_capprobe_learners = ["sccl_genprobe"]
    cfg.sccl_capprobe_check = 1
    cfg.sccl_genprobe_learners = ["sccl_genprobe"]
    cfg.sccl_genprobes = 1
    eng = FakeEngine(cfg, regen_passes=True)
    ver = Verifier(sandbox=PythonSandbox())
    vault = SelfCertVault()
    t1 = _poisoned_task()
    t2 = Task(task_id="t_v8_ok", family="fam", domain="code",
              prompt="Return the sum of two integers a and b.",
              test_code="assert add2(2, 2) == 4",
              reference_answer="def add2(a, b):\n    return 4",
              entry_point="add2")
    env = GroundedContinualEnv(cfg, eng, ver,
                               [Family(name="fam", tasks=[t1, t2], holdout=[])],
                               holdout=[], vault=vault, sccl=True)
    vault.commit_cap_probe("seed:c0", "fam", "variant spec",
                           "PROMPT v\n```python\n", _CAP_CODE,
                           _CAP_TESTS_VARIANT, 0.9)
    vault.commit_cap_probe("src:g", "fam", "untrained spec",
                           "PROMPT g\n```python\n", _CAP_CODE,
                           _CAP_TESTS_VARIANT, 0.9, pool=1, gen_pool=1)
    _, _, _, s1 = env.step(Action(answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
                                  metadata={"sccl": _cert_meta(found=True)}))
    g1 = s1["update_info"]["gate"]
    assert g1.get("cap_gen_pool") == 1, g1
    assert g1.get("checked_cap_ids") == ["seed:c0", "src:g"], g1
    # non-gen row: same vault contents, no G1 fields on the gate record
    cfg2 = ExperimentConfig(out_dir="runs/_test_sccl")
    cfg2._learner_name = "sccl_strict"
    cfg2.sccl_learners = ["sccl_strict"]
    cfg2.sccl_nogate_learners = []
    cfg2.sccl_gate_probe = 0
    cfg2.sccl_capprobe_learners = ["sccl_strict"]
    cfg2.sccl_capprobe_check = 1
    eng2 = FakeEngine(cfg2, regen_passes=True)
    env2 = GroundedContinualEnv(cfg2, eng2, ver,
                                [Family(name="fam", tasks=[t1, t2], holdout=[])],
                                holdout=[], vault=SelfCertVault(), sccl=True)
    env2.vault.commit_cap_probe("seed:c0", "fam", "variant spec",
                                "PROMPT v\n```python\n", _CAP_CODE,
                                _CAP_TESTS_VARIANT, 0.9)
    env2.vault.commit_cap_probe("src:g", "fam", "untrained spec",
                                "PROMPT g\n```python\n", _CAP_CODE,
                                _CAP_TESTS_VARIANT, 0.9, pool=1, gen_pool=1)
    _, _, _, s2 = env2.step(Action(answer=_CERT_CODE, learn_op=LearnOp.UPDATE_LORA,
                                   metadata={"sccl": _cert_meta(found=True)}))
    g2 = s2["update_info"]["gate"]
    assert "cap_gen_pool" not in g2 and "checked_cap_ids" not in g2, \
        "non-gen gate records must keep the pre-v8 shape"
