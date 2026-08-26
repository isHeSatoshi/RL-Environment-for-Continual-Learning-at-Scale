"""Isolated Python execution sandbox producing verified, ground-truth rewards.

This is the foundation of every number in the paper: a code answer is executed
in a throwaway subprocess and scored by *actually running* its unit tests.
No learned reward model, no substring heuristics masquerading as learning (I2).
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional


UNSAFE_IMPORTS = {"subprocess", "shutil", "socket", "pty", "paramiko"}
UNSAFE_CALLS = {"eval", "exec", "__import__", "compile", "open"}
UNSAFE_ATTRS = {"system", "popen", "spawn", "remove", "unlink", "rmdir", "environ"}


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    success: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    pass_rate: float = 0.0
    execution_time: float = 0.0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    safety_violation: bool = False
    # per-assert outcomes aligned with assert-line order (self-certification needs
    # verdict vectors, not just the aggregate pass rate)
    test_results: List[bool] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


_RUNNER = r"""
import json, time, traceback, io, contextlib, sys

def main():
    with open("input.json") as f:
        d = json.load(f)
    code, test_code = d.get("code",""), d.get("test","")
    res = {"exit_code":1,"success":False,"tests_passed":0,"tests_total":0,
           "pass_rate":0.0,"error_type":None,"error_message":None,"stderr_extra":""}
    ns = {}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<code>","exec"), ns)
        ok = True
    except Exception as e:
        ok = False
        res["error_type"] = type(e).__name__; res["error_message"] = str(e)
        res["stderr_extra"] = traceback.format_exc()
    res["stdout"] = buf.getvalue()
    if ok and test_code.strip():
        lines = [l.strip() for l in test_code.splitlines()
                 if l.strip() and not l.strip().startswith("#") and l.strip().startswith("assert")]
        if not lines:
            try:
                ns2 = dict(ns)
                with contextlib.redirect_stdout(buf):
                    exec(compile(test_code, "<test>", "exec"), ns2)
                res.update(tests_passed=1, tests_total=1, pass_rate=1.0, success=True)
            except Exception as e:
                res["error_type"] = type(e).__name__; res["error_message"] = str(e)
        else:
            res["tests_total"] = len(lines)
            res["results"] = []
            for st in lines:
                try:
                    exec(st, ns); res["tests_passed"] += 1; res["results"].append(True)
                except Exception as e:
                    res["results"].append(False)
                    if not res["error_type"]:
                        res["error_type"] = type(e).__name__; res["error_message"] = f"{st} -> {e}"
            res["pass_rate"] = res["tests_passed"]/max(1,res["tests_total"])
            res["success"] = res["tests_passed"] == res["tests_total"] and res["tests_total"]>0
            res["exit_code"] = 0 if res["success"] else 1
    elif ok:
        res.update(tests_passed=1, tests_total=1, pass_rate=1.0, success=True, exit_code=0)
    with open("result.json","w") as f:
        json.dump(res, f)

if __name__ == "__main__":
    main()
"""


def check_safety(code: str) -> Optional[str]:
    """Static AST + lexical safety screen. Returns a reason string or None."""
    if not code:
        return None
    for pat in (r"__import__", r"os\.(system|popen|remove|unlink|rmdir)",
                r"\beval\s*\(", r"\bexec\s*\(", r"rm\s+-rf"):
        if re.search(pat, code):
            return f"lexical:{pat}"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # syntax handled downstream
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in UNSAFE_IMPORTS:
                    return f"import:{a.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in UNSAFE_IMPORTS:
                return f"from-import:{node.module}"
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in UNSAFE_CALLS:
                return f"call:{f.id}"
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                if f.value.id == "os" and f.attr in UNSAFE_ATTRS:
                    return f"attr:os.{f.attr}"
    return None


class PythonSandbox:
    def __init__(self, python: Optional[str] = None, timeout: float = 6.0):
        # Use the *parent* interpreter by default so the venv python runs the sandbox
        self.python = python or sys.executable
        self.timeout = timeout

    def execute(self, code: str, test_code: str = "", timeout: Optional[float] = None) -> ExecutionResult:
        t0 = time.perf_counter()
        safety = check_safety(code)
        if safety:
            return ExecutionResult(stderr=safety, exit_code=1, success=False, tests_total=1,
                                   pass_rate=0.0, execution_time=0.0, error_type="SecurityViolation",
                                   error_message=safety, safety_violation=True)
        to = float(timeout if timeout is not None else self.timeout)
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "input.json"), "w") as f:
                json.dump({"code": code or "", "test": test_code or ""}, f)
            with open(os.path.join(td, "runner.py"), "w") as f:
                f.write(_RUNNER)
            try:
                proc = subprocess.run([self.python, "runner.py"], cwd=td, capture_output=True,
                                      text=True, timeout=to)
                elapsed = time.perf_counter() - t0
            except subprocess.TimeoutExpired:
                return ExecutionResult(stderr=f"Timeout>{to}s", exit_code=-1, success=False,
                                       tests_total=1, pass_rate=0.0, execution_time=to,
                                       error_type="TimeoutError")
            rf = os.path.join(td, "result.json")
            if os.path.exists(rf):
                try:
                    with open(rf) as f:
                        r = json.load(f)
                    return ExecutionResult(
                        stdout=r.get("stdout", proc.stdout or ""), stderr=proc.stderr or r.get("stderr_extra",""),
                        exit_code=r.get("exit_code", proc.returncode), success=r.get("success", False),
                        tests_passed=r.get("tests_passed",0), tests_total=r.get("tests_total",0),
                        pass_rate=r.get("pass_rate",0.0), execution_time=elapsed,
                        error_type=r.get("error_type"), error_message=r.get("error_message"),
                        test_results=[bool(x) for x in r.get("results", [])])
                except Exception as e:
                    return ExecutionResult(stdout=proc.stdout or "", stderr=f"parse:{e}", exit_code=1,
                                           success=False, tests_total=1, pass_rate=0.0, execution_time=elapsed,
                                           error_type="ParseError")
            return ExecutionResult(stdout=proc.stdout or "", stderr=proc.stderr or "no-result",
                                   exit_code=proc.returncode or 1, success=False, tests_total=1,
                                   pass_rate=0.0, execution_time=elapsed, error_type="ExecutionError")
