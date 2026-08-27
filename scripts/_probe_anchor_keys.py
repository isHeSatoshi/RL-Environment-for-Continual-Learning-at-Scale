"""CPU-only probe: do get_peft_model_state_dict keys match named_parameters keys?

The v4/v5 base anchor (engine.py apply_update) computes the penalty as
    for n, p in self.model.named_parameters():
        if p.requires_grad and n in base_anchor:   # base_anchor from get_peft_model_state_dict
If the key formats differ, the anchor is a silent no-op.
"""
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

base = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
# wrap in a module with a named submodule to mimic model.model... naming depth
m = torch.nn.ModuleDict({"model": base})
cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["0", "1"], bias="none")
pm = get_peft_model(m, cfg)

np_keys = [n for n, p in pm.named_parameters() if p.requires_grad]
sd_keys = list(get_peft_model_state_dict(pm).keys())
print("named_parameters (trainable):")
for k in np_keys:
    print("  ", k)
print("get_peft_model_state_dict keys:")
for k in sd_keys:
    print("  ", k)
inter = set(np_keys) & set(sd_keys)
print("\nintersection size:", len(inter), "of", len(np_keys), "named_parameters /", len(sd_keys), "state_dict keys")
print("ANCHOR WOULD BE A NO-OP" if not inter else "anchor keys match")
