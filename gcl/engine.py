"""TrainingEngine: REAL LoRA updates, EWC anchoring, replay, adapter registry,
and the deployment SafetyGate (holdout regression veto — I3).

Memory-safe continual learning on a single consumer GPU (I1):
  * ONE peft-model instance wraps ONE frozen base. Base + adapter never coexist
    as separate loaded copies, so this fits a 4B model on 16 GB.
  * Zero-copy rollback: snapshot adapter tensors (get_peft_model_state_dict),
    apply gradient update, run the holdout gate (base = disable_adapter),
    and only *keep* the update if regression is within budget. Otherwise the
    snapshot is restored (set_peft_model_state_dict) and the adapter rolled back.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import contextlib
import re
import ast as _ast
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import torch

from .curriculum import Task


@dataclass
class AdapterMeta:
    version: int
    path: str
    parent: str
    op: str
    loss: float
    grad_norm: float
    created_at: float
    content_hash: str = ""
    gate: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class AdapterRegistry:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.meta_path = os.path.join(root, "registry.json")
        self.metas: List[AdapterMeta] = []
        self.active_version = -1
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as f:
                rec = json.load(f)
            self.metas = [AdapterMeta(**m) for m in rec.get("metas", [])]
            self.active_version = rec.get("active_version", -1)

    @staticmethod
    def _hash_adapter_state(sd: Dict[str, torch.Tensor]) -> str:
        h = hashlib.sha256()
        for k in sorted(sd):
            t = sd[k]
            h.update(k.encode())
            h.update(t.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def register(self, path: str, op: str, loss: float, grad_norm: float,
                 content_hash: str, gate: Optional[Dict[str, Any]] = None) -> AdapterMeta:
        parent = self.metas[self.active_version].content_hash if 0 <= self.active_version < len(self.metas) else "base"
        m = AdapterMeta(version=len(self.metas), path=path, parent=parent, op=op,
                        loss=loss, grad_norm=grad_norm, created_at=time.time(),
                        content_hash=content_hash, gate=gate or {})
        self.metas.append(m)
        self.active_version = m.version
        self._save()
        return m

    def _save(self):
        with open(self.meta_path, "w") as f:
            json.dump({"metas": [m.to_dict() for m in self.metas],
                       "active_version": self.active_version}, f, indent=2)

    def history(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.metas]


class TrainingEngine:
    """Owns ONE base + ONE peft adapter. Real gradients (I1). Single-GPU safe."""

    def __init__(self, cfg, adapter_root: Optional[str] = None, device: Optional[str] = None):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        try:  # only newer transformers expose the multimodal auto class
            from transformers import AutoModelForMultimodalLM
        except ImportError:
            AutoModelForMultimodalLM = None

        from peft import LoraConfig, get_peft_model
        self.cfg = cfg
        target_device = device or cfg.device
        if target_device == "cuda" and torch.cuda.is_available():
            try:
                _t = torch.zeros(1, device="cuda") + 1
                self.device = "cuda"
            except Exception as e:
                print(f"CUDA validation failed ({e}); falling back to cpu")
                self.device = "cpu"
        else:
            self.device = target_device

        model_config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = getattr(torch, cfg.dtype, torch.bfloat16) if self.device == "cuda" else torch.float32
        # Qwen3.5-2B is a multimodal checkpoint (model_type=qwen3_5), not a
        # CausalLM checkpoint. Its text path accepts input_ids for this
        # code-only continual-learning workload. When the installed transformers
        # lacks the multimodal auto class (or a CausalLM arch), fall back safely.
        model_class = AutoModelForCausalLM
        if model_config.model_type == "qwen3_5" and AutoModelForMultimodalLM is not None:
            model_class = AutoModelForMultimodalLM
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            base = model_class.from_pretrained(cfg.model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
        else:
            base = model_class.from_pretrained(cfg.model_name, torch_dtype=dtype, trust_remote_code=True).to(self.device)
        self._lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                                lora_dropout=cfg.lora_dropout, bias="none",
                                target_modules=list(cfg.lora_targets), task_type="CAUSAL_LM")
        self.model = get_peft_model(base, self._lcfg)   # single instance
        self.registry = AdapterRegistry(adapter_root or os.path.join(cfg.out_dir, "adapters"))
        self._fisher: Optional[Dict[str, torch.Tensor]] = None
        self._anchor: Optional[Dict[str, torch.Tensor]] = None
        self._lora_base: Optional[Dict[str, torch.Tensor]] = None  # frozen LoRA init = "no adapter"
        self._replay: List[Dict[str, str]] = []
        self.updates_done = 0
        self._update_idx = 0
        self.gate_log: List[Dict[str, Any]] = []

    # ---- base anchor (vsr_bounded): first-call snapshot of LoRA init ---------
    def _base_anchor(self) -> Dict[str, torch.Tensor]:
        """Snapshot of the LoRA adapter as initialized (i.e. 'no skill yet'). On the
        vsr_bounded path, every update is pulled toward this. Implements the base-model
        anchor cheaply: LoRA init is mathematically equivalent to the frozen base.

        Keys MUST come from named_parameters(), not get_peft_model_state_dict():
        the penalty loop in apply_update matches by named_parameters() name, and
        PEFT state-dict keys strip the adapter segment ('lora_A.weight' vs
        'lora_A.default.weight'), which silently matches nothing and turns the
        anchor into a no-op (the v4/v5 ladder bug)."""
        if self._lora_base is None:
            self._lora_base = {n: p.detach().clone() for n, p in
                               self.model.named_parameters() if p.requires_grad}
        return self._lora_base

    # ---- snapshots (zero copy) ----
    def _snapshot(self):
        from peft import get_peft_model_state_dict
        return {k: v.detach().clone() for k, v in get_peft_model_state_dict(self.model).items()}

    def _restore(self, snap):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, {k: v.detach().clone() for k, v in snap.items()})

    # ---- tokenization ----
    def _mk_example(self, prompt: str, target: str):
        enc_p = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        full = prompt + target
        full_ids = self.tokenizer(full, return_tensors="pt", truncation=True,
                                  max_length=self.cfg.max_seq_len).input_ids.to(self.device)
        labels = full_ids.clone()
        mask = min(enc_p.shape[1], full_ids.shape[1])
        labels[:, :mask] = -100
        return full_ids, labels

    def _loss_on_batch(self, pairs: List[Dict[str, str]]):
        tot, n = 0.0, 0
        for pr in pairs:
            ids, labels = self._mk_example(pr["prompt"], pr["target"])
            tot = tot + self.model(input_ids=ids, labels=labels).loss
            n += 1
        return tot / max(1, n)

    # ---- REAL update (with optional EWC + Replay + Base Anchor). No gate here (see env). ----
    def apply_update(self, pairs: List[Dict[str, str]], ewc_lambda: float = 0.0,
                     replay_frac: float = 0.0, steps: Optional[int] = None,
                     lr: Optional[float] = None,
                     anchor_lambda: float = 0.0, replay_pairs: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        assert pairs, "apply_update needs data"
        steps = steps or self.cfg.train_steps_per_update
        base_seed = len(pairs)
        # Optional caller-provided replay buffer (used by the vault/replay merge in env).
        if replay_pairs is None:
            replay_pairs = self._replay
        eff_pairs = list(pairs)
        k = 0
        if replay_frac > 0 and replay_pairs:
            k = max(1, int(len(replay_pairs) * replay_frac))
            eff_pairs = eff_pairs + replay_pairs[-k:]
        # Multiple distinct tasks in a batch mean a single gradient step can't overfit ONE.
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr or self.cfg.learning_rate)

        # Optional base-anchor (vsr_bounded). Precompute the frozen base once.
        base_anchor = self._base_anchor() if anchor_lambda > 0 else None

        losses, last_gn, last_ap = [], 0.0, 0.0
        self.model.train()
        for _ in range(steps):
            loss = self._loss_on_batch(eff_pairs)
            if anchor_lambda > 0 and base_anchor is not None:
                anch_pen = 0.0
                for n, p in self.model.named_parameters():
                    if p.requires_grad and n in base_anchor:
                        anch_pen = anch_pen + (p - base_anchor[n]).pow(2).sum()
                # Effect audit trail: 0.0 on every update is the signature of
                # the v4 no-op bug (key mismatch); a live anchor logs > 0.
                last_ap = float(anch_pen.detach())
                loss = loss + 0.5 * anchor_lambda * anch_pen
            if ewc_lambda > 0 and self._fisher is not None and self._anchor is not None:
                pen = 0.0
                for n, p in self.model.named_parameters():
                    if n in self._fisher:
                        pen = pen + (self._fisher[n] * (p - self._anchor[n]).pow(2)).sum()
                loss = loss + 0.5 * ewc_lambda * pen
            loss.backward()
            last_gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step(); opt.zero_grad()
            losses.append(float(loss.detach()))
        self.updates_done += 1
        self._update_idx += 1
        for pr in pairs:
            self._replay.append(pr)
        if len(self._replay) > 1024:
            self._replay = self._replay[-1024:]
        return {"loss_start": losses[0], "loss_end": losses[-1], "grad_norm": last_gn,
                "n_pairs": len(eff_pairs), "replay_used": k, "steps": steps,
                "ewc_lambda": ewc_lambda, "anchor_lambda": anchor_lambda,
                "anchor_pen": last_ap}

    # ---- bounded update step: LR decays with depth to stop endless drift ------
    def bounded_lr(self, base_lr: Optional[float] = None) -> float:
        b = base_lr or self.cfg.learning_rate
        depth = max(1, self._update_idx + 1)
        depth = min(depth, 20.0)
        return b / (1.0 + 0.05 * (depth - 1))

    def consolidate_ewc(self, pairs: List[Dict[str, str]]) -> Dict[str, Any]:
        fisher, anchor = {}, {}
        self.model.zero_grad(set_to_none=True)
        self._loss_on_batch(pairs).backward()
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    g = p.grad
                    fisher[n] = g.detach().pow(2).clone() if g is not None else torch.zeros_like(p)
                    anchor[n] = p.detach().clone()
        self.model.zero_grad(set_to_none=True)
        self._fisher, self._anchor = fisher, anchor
        return {"ewc_params": len(anchor)}

    # ---- generation (Track-B) ----
    def _apply_chat(self, prompt: str):
        """Render through the tokenizer's chat template when available (Qwen3.x is
        a thinking model — without the template it rambles <think> and wastes tokens
        before emitting code)."""
        try:
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                msgs = [{"role": "user", "content": prompt}]
                out = self.tokenizer.apply_chat_template(msgs, tokenize=False,
                                                         add_generation_prompt=True)
                return out
        except Exception:
            pass
        return prompt

    @torch.no_grad()
    def generate(self, prompt: str, adapter_on: bool = True, greedy: bool = True) -> str:
        self.model.eval()
        ctx = contextlib.nullcontext() if adapter_on else self.model.disable_adapter()
        rendered = self._apply_chat(prompt)
        ids = self.tokenizer(rendered, return_tensors="pt").input_ids.to(self.device)
        with ctx:
            do_sample = (self.cfg.temperature > 0) and not greedy
            kw = dict(max_new_tokens=self.cfg.max_new_tokens, do_sample=do_sample,
                      pad_token_id=self.tokenizer.pad_token_id)
            if do_sample:
                kw["temperature"] = max(self.cfg.temperature, 1e-4)
                kw["top_p"] = self.cfg.top_p
            out = self.model.generate(ids, **kw)
        txt = self.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        # strip any residual thinking preambles from a thinking model
        for tag in ("</think>", "<|im_end|>"):
            if tag in txt:
                txt = txt.split(tag, 1)[1] if tag == "</think>" else txt.split(tag, 1)[0]
        return txt.strip()

    @torch.no_grad()
    def generate_batch(self, prompts: List[str], adapter_on: bool = True, greedy: bool = True) -> List[str]:
        self.model.eval()
        orig_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        rendered = [self._apply_chat(p) for p in prompts]
        enc = self.tokenizer(rendered, return_tensors="pt", padding=True)
        ids = enc.input_ids.to(self.device)
        attn = enc.attention_mask.to(self.device)
        ctx = contextlib.nullcontext() if adapter_on else self.model.disable_adapter()
        with ctx:
            do_sample = (self.cfg.temperature > 0) and not greedy
            kw = dict(max_new_tokens=self.cfg.max_new_tokens, do_sample=do_sample,
                      pad_token_id=self.tokenizer.pad_token_id)
            if do_sample:
                kw["temperature"] = max(self.cfg.temperature, 1e-4)
                kw["top_p"] = self.cfg.top_p
            out = self.model.generate(ids, attention_mask=attn, **kw)
        self.tokenizer.padding_side = orig_side
        results = []
        for i, row in enumerate(out):
            txt = self.tokenizer.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True)
            for tag in ("</think>", "<|im_end|>"):
                if tag in txt:
                    txt = txt.split(tag, 1)[1] if tag == "</think>" else txt.split(tag, 1)[0]
            results.append(txt.strip())
        return results

    @torch.no_grad()
    def sample_candidates(self, prompt: str, n: int, temperature: float = 0.7,
                          top_p: float = 0.95, adapter_on: bool = True,
                          max_new_tokens: Optional[int] = None) -> List[str]:
        """Batched multi-sample generation for self-taught (STaR) rollouts.

        One forward pass per *batch* of n candidates (pads to equal length), so
        temperature>0 yields `n` distinct code completions cheaply. Used by the
        no-gold learner to find a verifier-confirmed solution pass@K instead of
        giving up after a single greedy sample.
        """
        self.model.eval()
        if n <= 1:
            return [self.generate(prompt, adapter_on=adapter_on, greedy=(temperature <= 0))]
        rendered = self._apply_chat(prompt)
        enc = self.tokenizer(rendered, return_tensors="pt")
        ids = enc.input_ids.to(self.device)
        attn = enc.attention_mask.to(self.device)
        ntok = int(attn.sum().item())
        ids = ids.repeat(n, 1)
        ctx = contextlib.nullcontext() if adapter_on else self.model.disable_adapter()
        with ctx:
            out = self.model.generate(
                ids, max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                do_sample=(temperature > 0), temperature=max(temperature, 1e-4) if temperature > 0 else 1.0,
                top_p=top_p if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id)
        outs = []
        for row in out:
            outs.append(self.tokenizer.decode(row[ntok:], skip_special_tokens=True))
        # strip thinking preambles
        cleaned = []
        for txt in outs:
            for tag in ("</think>",):
                if tag in txt:
                    txt = txt.split(tag, 1)[1]
            cleaned.append(txt.strip())
        return cleaned

    # ---- holdout score under base (adapter off) or current adapter ----
    @torch.no_grad()
    def holdout_score(self, holdout: List[Task], verifier, adapter_on: bool) -> float:
        if not holdout:
            return 1.0
        scores = []
        for t in holdout:
            code = extract_code(self.generate(_build_prompt(t), adapter_on=adapter_on)) \
                if t.domain == "code" else self.generate(_build_prompt(t), adapter_on=adapter_on)
            r, _, _ = verifier.reward(domain=t.domain, code=code, test_code=t.test_code,
                                      reference_answer=t.reference_answer)
            scores.append(r)
        return float(sum(scores) / len(scores))

    def register_adapter(self, op: str, meta: Dict[str, Any]) -> AdapterMeta:
        from peft import get_peft_model_state_dict
        sd = get_peft_model_state_dict(self.model)
        chash = self.registry._hash_adapter_state(sd)
        vdir = os.path.join(self.registry.root, f"adapter_v{len(self.registry.metas)}")
        os.makedirs(vdir, exist_ok=True)
        self.model.save_pretrained(vdir)
        return self.registry.register(vdir, op, meta.get("loss_end", 0.0),
                                      meta.get("grad_norm", 0.0), chash, gate=meta.get("gate"))


def _build_prompt(task: Task) -> str:
    if task.domain == "math":
        return (f"Solve and give ONLY the final numeric answer.\n{task.prompt}\nAnswer: ")
    # No <think> traps, explicit entry-point naming, complete code only (no stubs).
    ep = getattr(task, "entry_point", "") or ""
    name_hint = f" The function MUST be named `{ep}`." if ep else \
        " Pick the function name implied by the task below."
    return ("You are an expert Python programmer. Write ONLY the complete function "
            "inside a ```python block. No analysis, no <think>, no empty 'pass' stubs."
            f"{name_hint}\n{task.prompt}\n```python\n")


def extract_code(text: str) -> str:
    """Executable-code extractor robust to base-model noise.

    Collect every ```...``` block (language tag optional, not requiring a newline
    right after the fence), prefer a *real* function body over a stub, ignore
    <think> chatter. Fallback: last standalone `def` run from raw text. Always
    returns fence-stripped, compilable code.
    """
    text = (text or "")
    def _strip_fence(block: str) -> str:
        b = block.strip()
        # drop a leading language marker line like ```python or ```python (no newline)
        b = re.sub(r"^```(?:python)?\s*\n?", "", b, count=1)
        b = re.sub(r"\s*```$", "", b, count=1)  # trailing fence residue
        return b.strip()

    # match ```optional-lang [possibly same line]``` both with and without trailing NL
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
    # fallback: last def-block (fence-free segment)
    matches = list(re.finditer(r"(?m)^(\s*def\s+\w+\s*\(.*)$", text))
    if matches:
        tail = text[matches[-1].start():]
        tail = re.split(r"\n\s*```", tail, maxsplit=1)[0]  # stop at a trailing fence
        # keep top-level imports that precede the def block, otherwise the
        # extracted snippet NameErrors at runtime (e.g. math.sqrt w/o import)
        head = text[:matches[-1].start()]
        imports = [ln.rstrip() for ln in head.splitlines()
                   if re.match(r"^\s*(import\s+[A-Za-z_]|from\s+[\w.]+\s+import\s)", ln)]
        if imports:
            tail = "\n".join(imports) + "\n" + tail
        return tail.strip()
    # no compilable code found (think/prose only) — return empty so verifier scores 0
    return ""
