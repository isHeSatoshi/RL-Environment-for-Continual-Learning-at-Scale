"""Retroactive per-task evaluation for completed runs.

Runs launched before per-task eval_detail logging (e.g. runs/sccl_v2) only have
family-level scores in metrics.json. This script loads each learner's FINAL
adapter from <run>/adapters_<name>/ and re-scores every family task-by-task
(train split and holdout split), writing <run>/eval_detail_retro.json.

Gold labels are used here ONLY for measurement (final evaluation), never for
any learning decision — consistent with the project's gold-free learning
guarantee. Generation is greedy, so per-family means should reproduce the run's
logged R_pairs almost exactly; the script verifies this and reports max drift.

Usage:
  python scripts/eval_detail_retro.py --config configs/sccl_v2.json --run runs/sccl_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gcl.config import ExperimentConfig
from gcl.curriculum import Family, canary_report
from gcl.engine import TrainingEngine, _build_prompt, extract_code
from gcl.experiment import _FamilyProxy
from gcl.runner import load_families
from gcl.verify import Verifier
from gcl.sandbox import PythonSandbox


def _eval_detail(engine, verifier, family) -> Dict[str, Any]:
    recs = []
    for t in (family.tasks or []):
        txt = engine.generate(_build_prompt(t), adapter_on=True)
        code = extract_code(txt) if t.domain == "code" else txt
        r, info, _ = verifier.reward(domain=t.domain, code=code, test_code=t.test_code,
                                     reference_answer=t.reference_answer)
        recs.append({"task_id": getattr(t, "task_id", ""), "score": float(r),
                     "pass_rate": float(info.get("pass_rate", 0.0)),
                     "success": bool(info.get("success", False))})
    mean = float(sum(x["score"] for x in recs) / len(recs)) if recs else 0.0
    return {"mean": mean, "records": recs}


def _load_final_adapter_state(engine, run_dir: str, name: str):
    """Load learner's final adapter weights into the engine; return meta or None."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    reg_path = os.path.join(run_dir, f"adapters_{name}", "registry.json")
    if not os.path.exists(reg_path):
        return None
    with open(reg_path) as f:
        reg = json.load(f)
    av = reg.get("active_version", -1)
    metas = reg.get("metas", [])
    if av < 0 or av >= len(metas):
        return None
    vdir = metas[av]["path"]
    sd_file = os.path.join(vdir, "adapter_model.safetensors")
    if not os.path.exists(sd_file):
        return None
    sd = load_file(sd_file)
    set_peft_model_state_dict(engine.model, sd)
    loaded_hash = engine.registry._hash_adapter_state(get_peft_model_state_dict(engine.model))
    return {"version": av, "expected_hash": metas[av].get("content_hash", ""),
            "loaded_hash": loaded_hash, "path": vdir}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="original experiment config (has stream spec)")
    ap.add_argument("--run", required=True, help="completed run directory")
    ap.add_argument("--learners", default=None, help="comma-separated subset (default: all)")
    args = ap.parse_args()

    with open(args.config) as f:
        raw = json.load(f)
    cfg = ExperimentConfig(**raw["experiment"])
    families: List[Family] = load_families(raw["stream"])

    # Integrity: rebuilt stream must match the run's logged canary hash.
    met_path = os.path.join(args.run, "metrics.json")
    if not os.path.exists(met_path):
        met_path = os.path.join(args.run, "metrics_partial.json")
    with open(met_path) as f:
        metrics = json.load(f)
    run_hash = metrics.get("canary", {}).get("stream_hash", "")
    rebuilt = canary_report(families)
    if rebuilt["stream_hash"] != run_hash:
        raise RuntimeError(f"stream hash mismatch: rebuilt {rebuilt['stream_hash']} "
                           f"vs run {run_hash} — config does not match this run")
    if not rebuilt["clean"]:
        raise RuntimeError("rebuilt stream fails anti-contamination canary")
    print(f"[retro] stream verified (hash {run_hash}, canary clean)")

    names = [n.strip() for n in args.learners.split(",") if n.strip()] \
        if args.learners else list(metrics.get("learners", {}).keys())

    cfg.out_dir = args.run
    engine = TrainingEngine(cfg, adapter_root=tempfile.mkdtemp(prefix="retro_scratch_"))
    verifier = Verifier(sandbox=PythonSandbox())

    # Zero-shot (base model, adapter disabled) — computed once, shared.
    print("[retro] zero-shot baseline (adapter disabled)...", flush=True)
    zero_shot: Dict[str, Any] = {}
    for fam in families:
        recs = []
        for t in (fam.tasks or []):
            txt = engine.generate(_build_prompt(t), adapter_on=False)
            code = extract_code(txt) if t.domain == "code" else txt
            r, info, _ = verifier.reward(domain=t.domain, code=code, test_code=t.test_code,
                                         reference_answer=t.reference_answer)
            recs.append({"task_id": getattr(t, "task_id", ""), "score": float(r),
                         "pass_rate": float(info.get("pass_rate", 0.0)),
                         "success": bool(info.get("success", False))})
        zero_shot[fam.name] = {"mean": float(sum(x["score"] for x in recs) / len(recs)) if recs else 0.0,
                               "records": recs}

    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    init_sd = {k: v.detach().clone() for k, v in
               get_peft_model_state_dict(engine.model).items()}  # LoRA init == base

    out: Dict[str, Any] = {"zero_shot": zero_shot, "learners": {}, "verify": {}}
    for name in names:
        if name == "frozen":
            set_peft_model_state_dict(engine.model, {k: v.detach().clone() for k, v in init_sd.items()})
            ameta = {"version": -1, "note": "frozen base (LoRA init)"}
        else:
            ameta = _load_final_adapter_state(engine, args.run, name)
            if ameta is None:
                print(f"[retro] {name}: no adapter registry — skipping", flush=True)
                continue
            if ameta.get("expected_hash") and ameta["expected_hash"] != ameta["loaded_hash"]:
                print(f"[retro] {name}: WARNING hash mismatch "
                      f"({ameta['expected_hash']} vs {ameta['loaded_hash']})", flush=True)
        print(f"[retro] {name}: adapter={ameta.get('version')} — evaluating "
              f"{len(families)} families task-by-task...", flush=True)
        detail = {"final_trained": {}, "final_heldout": {}}
        drift = 0.0
        lm = metrics["learners"].get(name, {})
        rp = lm.get("R_pairs", {})
        for fi, fam in enumerate(families):
            d_tr = _eval_detail(engine, verifier, fam)
            d_ho = _eval_detail(engine, verifier, _FamilyProxy(fam.name, fam.holdout))
            detail["final_trained"][fam.name] = d_tr["records"]
            detail["final_heldout"][fam.name] = d_ho["records"]
            # reproducibility check vs logged family-level scores
            logged_h = (rp.get("final") or [None] * len(families))[fi]
            if logged_h is not None:
                drift = max(drift, abs(d_ho["mean"] - logged_h))
        out["learners"][name] = detail
        out["verify"][name] = {"adapter": ameta, "max_abs_drift_vs_logged": drift}
        print(f"[retro] {name}: done (max drift vs logged final_heldout = {drift:.4f})", flush=True)

    out_path = os.path.join(args.run, "eval_detail_retro.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[retro] wrote {out_path}")
    # concise failure summary
    for name, det in out["learners"].items():
        parts = []
        for fn, recs in det["final_heldout"].items():
            failed = [r["task_id"] for r in recs if r["score"] < 0.999]
            if failed:
                parts.append(f"{fn}: {','.join(failed)}")
        print(f"  {name:<14} " + ("; ".join(parts) if parts else "all holdouts pass"))


if __name__ == "__main__":
    main()
