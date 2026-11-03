"""
OPLoRA (Orthogonal Projection LoRA) - manual Task 1, Step 4.

For every target weight W0 = U S V^T (dout x din) we freeze the leading-k singular bases
U_k (dout x k), V_k (din x k) and use the *effective* projected factors

    B~ = B - U_k (U_k^T B)          (dout x r)
    A~ = A - (A V_k) V_k^T          (r x din)
    dW = (alpha / r) * B~ A~

The implementation re-uses PEFT's LoRA layers (so initial A/B are identical to LoRA-16
under the same seed) and only swaps their forward pass. Bases are computed once in FP32.
Dense projection matrices (U_k U_k^T etc.) are never formed.
"""
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import LORA_TARGETS


def _is_target(name: str, targets):
    return name.split(".")[-1] in targets


@torch.no_grad()
def compute_bases(base_model: nn.Module, k: int, targets=LORA_TARGETS, device="cuda"):
    """
    SVD of every target nn.Linear weight (FP32). Returns
      bases: {module_name: (U_k, V_k)} on `device` in FP32,
      info:  {"preprocess_seconds", "basis_bytes", "n_matrices"}
    `module_name` is the name inside the *base* (non-PEFT) model, e.g. "model.layers.0.mlp.down_proj".
    """
    bases, nbytes = {}, 0
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for name, mod in base_model.named_modules():
        if isinstance(mod, nn.Linear) and _is_target(name, targets):
            W = mod.weight.detach().to(device=device, dtype=torch.float32)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            Uk = U[:, :k].contiguous()
            Vk = Vh[:k].T.contiguous()
            bases[name] = (Uk, Vk)
            nbytes += Uk.numel() * 4 + Vk.numel() * 4
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    info = {"preprocess_seconds": time.perf_counter() - t0, "basis_bytes": nbytes, "n_matrices": len(bases)}
    return bases, info


def projected_factors(A: torch.Tensor, B: torch.Tensor, Uk: torch.Tensor, Vk: torch.Tensor):
    """A~ , B~ in FP32 (effective factors, no dense projection matrices)."""
    A32, B32 = A.float(), B.float()
    A_t = A32 - (A32 @ Vk) @ Vk.T
    B_t = B32 - Uk @ (Uk.T @ B32)
    return A_t, B_t


def _oplora_forward(self, x, *args, **kwargs):
    """Replacement forward for a PEFT lora.Linear module with attached bases."""
    result = self.base_layer(x, *args, **kwargs)
    adapter = self.active_adapters[0] if hasattr(self, "active_adapters") else "default"
    A = self.lora_A[adapter].weight
    B = self.lora_B[adapter].weight
    with torch.autocast(device_type="cuda", enabled=False):
        A_t, B_t = projected_factors(A, B, self.oplora_Uk, self.oplora_Vk)
    xd = self.lora_dropout[adapter](x)
    delta = F.linear(F.linear(xd, A_t.to(xd.dtype)), B_t.to(xd.dtype))
    return result + delta * self.scaling[adapter]


def attach_oplora(peft_model, bases: dict):
    """Patch every PEFT LoRA Linear whose base-model name is in `bases`."""
    from peft.tuners.lora import Linear as LoraLinear
    n = 0
    for name, mod in peft_model.named_modules():
        if isinstance(mod, LoraLinear):
            key = name.replace("base_model.model.", "", 1)
            if key not in bases:
                raise KeyError(f"No OPLoRA basis for {key}")
            Uk, Vk = bases[key]
            mod.register_buffer("oplora_Uk", Uk, persistent=False)
            mod.register_buffer("oplora_Vk", Vk, persistent=False)
            mod.forward = types.MethodType(_oplora_forward, mod)
            n += 1
    return n


@torch.no_grad()
def projected_state_dict(peft_model, bases: dict):
    """
    Full state dict of the PEFT model where every lora_A / lora_B weight is replaced by its
    projected counterpart A~ / B~. Passing this to PeftModel.save_pretrained(state_dict=...)
    exports a standard PEFT adapter whose plain LoRA forward equals the OPLoRA forward.
    """
    from peft.tuners.lora import Linear as LoraLinear
    sd = {k: v for k, v in peft_model.state_dict().items()}
    for name, mod in peft_model.named_modules():
        if isinstance(mod, LoraLinear):
            key = name.replace("base_model.model.", "", 1)
            Uk, Vk = bases[key]
            for adapter in mod.lora_A.keys():
                A = mod.lora_A[adapter].weight
                B = mod.lora_B[adapter].weight
                A_t, B_t = projected_factors(A, B, Uk, Vk)
                sd[f"{name}.lora_A.{adapter}.weight"] = A_t.to(A.dtype).cpu()
                sd[f"{name}.lora_B.{adapter}.weight"] = B_t.to(B.dtype).cpu()
    return sd


@torch.no_grad()
def verify_projection(peft_model, bases: dict, atol=1e-4):
    """Numerical check of the OPLoRA constraints: U_k^T B~ = 0 and A~ V_k = 0."""
    from peft.tuners.lora import Linear as LoraLinear
    worst = 0.0
    for name, mod in peft_model.named_modules():
        if isinstance(mod, LoraLinear):
            key = name.replace("base_model.model.", "", 1)
            Uk, Vk = bases[key]
            for adapter in mod.lora_A.keys():
                A_t, B_t = projected_factors(mod.lora_A[adapter].weight, mod.lora_B[adapter].weight, Uk, Vk)
                worst = max(worst, (Uk.T @ B_t).abs().max().item(), (A_t @ Vk).abs().max().item())
    return {"max_abs_residual": worst, "ok": worst < atol}
