"""KERN-Lab-0.5-624M runtime for Transformers AutoClass and the standalone loader."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model
from transformers import Qwen3Config, Qwen3ForCausalLM


class QwenSwiGLUExpert(nn.Module):
    """Exact structural twin of Qwen3MLP (SwiGLU without bias)."""

    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        factory = {"dtype": dtype} if dtype is not None else {}
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, **factory)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, **factory)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, **factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FP32RouterLinear(nn.Linear):
    """Linear router kept strictly in torch.float32 across device moves and dtype casts."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False, dtype=torch.float32)

    def _apply(self, fn, recurse: bool = True):  # type: ignore[override]
        old_weight = self.weight.detach()
        super()._apply(fn, recurse=recurse)
        if self.weight.dtype != torch.float32:
            self.weight.data = old_weight.to(device=self.weight.device, dtype=torch.float32)
        return self


class KernQwen3UpcycledMoEMLP(nn.Module):
    """Inference-safe Sparse MoE block (1 shared + N routed Top-K experts) for Qwen3 Layer 14."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        num_routed_experts: int = 3,
        top_k: int = 2,
        shared_alpha: float = 0.5,
        renormalize_topk: bool = True,
        unnormalized_scale: float = 1.0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_routed_experts = int(num_routed_experts)
        self.top_k = int(top_k)
        self.shared_alpha = float(shared_alpha)
        self.renormalize_topk = bool(renormalize_topk)
        self.unnormalized_scale = float(unnormalized_scale)

        self.shared_expert = QwenSwiGLUExpert(self.hidden_size, self.intermediate_size, dtype=dtype)
        self.routed_experts = nn.ModuleList(
            [
                QwenSwiGLUExpert(self.hidden_size, self.intermediate_size, dtype=dtype)
                for _ in range(self.num_routed_experts)
            ]
        )
        self.router = FP32RouterLinear(self.hidden_size, self.num_routed_experts)
        self.register_buffer("expert_bias", torch.zeros(self.num_routed_experts, dtype=torch.float32))

        # Ephemeral routing telemetry (non-persistent; supports both prefill and KV-cache decode)
        self.register_buffer(
            "eval_cumulative_tokens_per_expert",
            torch.zeros(self.num_routed_experts, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("eval_valid_tokens_routed", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("eval_padding_tokens_ignored", torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer("eval_forward_calls", torch.zeros((), dtype=torch.long), persistent=False)
        self._active_attention_mask: torch.Tensor | None = None

    def _apply(self, fn, recurse: bool = True):  # type: ignore[override]
        old_bias = self.expert_bias
        super()._apply(fn, recurse=recurse)
        if self.expert_bias.dtype != torch.float32:
            self.expert_bias = old_bias.to(device=self.expert_bias.device, dtype=torch.float32)
        return self

    def reset_routing_stats(self) -> None:
        self.eval_cumulative_tokens_per_expert.zero_()
        self.eval_valid_tokens_routed.zero_()
        self.eval_padding_tokens_ignored.zero_()
        self.eval_forward_calls.zero_()

    def get_routing_stats(self) -> dict[str, Any]:
        counts = [int(x) for x in self.eval_cumulative_tokens_per_expert.detach().cpu().tolist()]
        total_disp = sum(counts)
        shares = [round(c / float(total_disp), 6) for c in counts] if total_disp > 0 else [0.0] * self.num_routed_experts
        if total_disp > 0:
            ent = float(-sum(s * math.log(s) for s in shares if s > 0.0) / math.log(self.num_routed_experts))
            max_load = float(max(counts) / (total_disp / float(self.num_routed_experts)))
        else:
            ent = 0.0
            max_load = 0.0
        return {
            "forward_calls": int(self.eval_forward_calls.item()),
            "valid_tokens_routed": int(self.eval_valid_tokens_routed.item()),
            "padding_tokens_ignored": int(self.eval_padding_tokens_ignored.item()),
            "cumulative_tokens_per_expert": counts,
            "expert_token_fraction": shares,
            "cumulative_dispatch_normalized_entropy": round(ent, 6),
            "cumulative_max_load_ratio": round(max_load, 6),
        }

    def _resolve_valid_mask(
        self,
        num_tokens: int,
        batch_prefix_shape: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor | None:
        raw_mask = self._active_attention_mask
        if raw_mask is None:
            return None
        if tuple(raw_mask.shape) == batch_prefix_shape:
            return raw_mask.reshape(-1).to(device=device) > 0
        if int(raw_mask.numel()) == num_tokens:
            return raw_mask.reshape(-1).to(device=device) > 0
        if (
            raw_mask.ndim == 2
            and len(batch_prefix_shape) == 2
            and raw_mask.shape[0] == batch_prefix_shape[0]
            and raw_mask.shape[1] >= batch_prefix_shape[1]
        ):
            return raw_mask[:, -batch_prefix_shape[1] :].reshape(-1).to(device=device) > 0
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]
        valid_mask = self._resolve_valid_mask(num_tokens, tuple(orig_shape[:-1]), x.device)

        shared_out = self.shared_expert(x_flat)

        with torch.amp.autocast(device_type=x.device.type if x.device.type in ("cuda", "cpu") else "cpu", enabled=False):
            router_logits = F.linear(x_flat.float(), self.router.weight.float()) + self.expert_bias.float()
            probs_full = torch.softmax(router_logits, dim=-1)
            topk_vals, topk_idx = torch.topk(probs_full, k=self.top_k, dim=-1)
            if self.renormalize_topk:
                topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            else:
                topk_weights = topk_vals * self.unnormalized_scale

        routed_out = torch.zeros_like(x_flat, dtype=torch.float32)
        for e_idx in range(self.num_routed_experts):
            mask = topk_idx == e_idx
            if not mask.any():
                continue
            token_pos, k_pos = torch.nonzero(mask, as_tuple=True)
            selected_x = x_flat.index_select(0, token_pos)
            exp_y = self.routed_experts[e_idx](selected_x).float()
            w = topk_weights[token_pos, k_pos].unsqueeze(-1)
            routed_out.index_add_(0, token_pos, exp_y * w)

        out = (self.shared_alpha * shared_out.float() + (1.0 - self.shared_alpha) * routed_out).to(dtype=x.dtype)

        with torch.no_grad():
            if valid_mask is not None:
                topk_valid = topk_idx[valid_mask]
                n_valid = int(valid_mask.sum().item())
                n_pad = int(num_tokens - n_valid)
            else:
                topk_valid = topk_idx
                n_valid = int(num_tokens)
                n_pad = 0
            if n_valid > 0:
                counts = torch.bincount(topk_valid.reshape(-1), minlength=self.num_routed_experts).to(dtype=torch.long)
            else:
                counts = torch.zeros(self.num_routed_experts, dtype=torch.long, device=x.device)
            self.eval_cumulative_tokens_per_expert.add_(counts)
            self.eval_valid_tokens_routed.add_(n_valid)
            self.eval_padding_tokens_ignored.add_(n_pad)
            self.eval_forward_calls.add_(1)

        return out.reshape(orig_shape)


def _attach_mask_hooks(model: nn.Module) -> None:
    if getattr(model, "_kern_moe_mask_hook_handles", None):
        return

    def _pre_hook(mod: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        attn = kwargs.get("attention_mask")
        if attn is None and len(args) >= 2 and isinstance(args[1], torch.Tensor):
            attn = args[1]
        active = attn if isinstance(attn, torch.Tensor) else None
        for child in mod.modules():
            if isinstance(child, KernQwen3UpcycledMoEMLP):
                child._active_attention_mask = active

    def _post_hook(mod: nn.Module, _args: tuple[Any, ...], _out: Any) -> None:
        for child in mod.modules():
            if isinstance(child, KernQwen3UpcycledMoEMLP):
                child._active_attention_mask = None

    h_pre = model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    h_post = model.register_forward_hook(_post_hook)
    model._kern_moe_mask_hook_handles = (h_pre, h_post)


def _install_moe(model: Qwen3ForCausalLM, meta: dict[str, Any], dtype: torch.dtype) -> None:
    upcycle = meta["upcycle_config"]
    layer = int(meta["target_layer_idx"])
    if meta.get("topology") != "upcycle_moe" or int(upcycle["num_shared_experts"]) != 1:
        raise ValueError("Expected the KERN sparse MoE topology with one shared expert")
    if not 0 <= layer < len(model.model.layers) or int(upcycle["target_layer_idx"]) != layer:
        raise ValueError("Invalid or inconsistent KERN layer index")
    config = model.config
    model.model.layers[layer].mlp = KernQwen3UpcycledMoEMLP(
        config.hidden_size,
        config.intermediate_size,
        num_routed_experts=int(upcycle["num_routed_experts"]),
        top_k=int(upcycle["top_k"]),
        shared_alpha=float(upcycle["shared_alpha"]),
        renormalize_topk=bool(upcycle["renormalize_topk"]),
        unnormalized_scale=float(upcycle.get("unnormalized_scale", 1.0)),
        dtype=dtype,
    )


class KernQwen3Config(Qwen3Config):
    """Qwen3 configuration with a distinct AutoConfig identity for KERN."""

    model_type = "kern_qwen3_moe"


class KernQwen3ForCausalLM(Qwen3ForCausalLM):
    """Qwen3 with the KERN FFN installed before Transformers loads SafeTensors."""

    config_class = KernQwen3Config
    _keep_in_fp32_modules_strict = ["router", "expert_bias"]

    def __init__(self, config: KernQwen3Config) -> None:
        super().__init__(config)
        _install_moe(self, config.kern_adaptation, self.model.embed_tokens.weight.dtype)
        _attach_mask_hooks(self)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        output_loading_info = kwargs.pop("output_loading_info", False)
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, output_loading_info=True, **kwargs
        )
        problems = {
            key: loading_info[key]
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading_info.get(key)
        }
        if problems:
            raise RuntimeError(f"KERN checkpoint did not load exactly: {problems}")
        moe = model.model.layers[int(model.config.kern_adaptation["target_layer_idx"])].mlp
        if moe.router.weight.dtype != torch.float32 or moe.expert_bias.dtype != torch.float32:
            raise RuntimeError("KERN requires FP32 routing tensors")
        if model.model.embed_tokens.weight is not model.lm_head.weight:
            raise RuntimeError("KERN expects tied input embeddings and LM head")
        moe.reset_routing_stats()
        return (model, loading_info) if output_loading_info else model

    def _apply(self, fn, recurse: bool = True):  # type: ignore[override]
        rotary = self.model.rotary_emb
        old_freq = rotary.inv_freq
        old_original = rotary.original_inv_freq
        super()._apply(fn, recurse=recurse)
        if rotary.inv_freq.dtype != torch.float32:
            rotary.inv_freq = old_freq.to(device=rotary.inv_freq.device, dtype=torch.float32)
        if rotary.original_inv_freq.dtype != torch.float32:
            rotary.original_inv_freq = old_original.to(device=rotary.original_inv_freq.device, dtype=torch.float32)
        return self


def load_kern_qwen06b_v1(
    model_dir: str | Path,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Qwen3ForCausalLM, dict[str, Any]]:
    """Load standalone KERN v1 adapted Qwen3-0.6B MoE weights preserving tying, FP32 router, and FP32 RoPE."""
    base = Path(model_dir)
    cfg_raw = json.loads((base / "config.json").read_text(encoding="utf-8"))
    kern_meta = dict(cfg_raw.get("kern_adaptation", {}))
    if kern_meta.get("topology") != "upcycle_moe":
        raise ValueError(f"Expected kern_adaptation.topology == 'upcycle_moe', got {kern_meta.get('topology')!r}")

    if isinstance(device, str) and device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    qwen_cfg = Qwen3Config.from_dict({**cfg_raw, "model_type": "qwen3"})
    # Build on CPU first so non-persistent RoPE inv_freq stays exact float32
    model = Qwen3ForCausalLM(qwen_cfg)
    rotary = model.model.rotary_emb
    inv_freq_f32 = rotary.inv_freq.detach().clone().to(dtype=torch.float32)
    orig_inv_f32 = (
        rotary.original_inv_freq.detach().clone().to(dtype=torch.float32)
        if hasattr(rotary, "original_inv_freq")
        else None
    )

    target_layer = int(kern_meta["target_layer_idx"])
    up_cfg = dict(kern_meta["upcycle_config"])
    _install_moe(model, kern_meta, torch.float32)
    model = model.to(dtype=dtype)

    # Restore exact float32 RoPE buffers before loading weights & moving to target device
    rotary.inv_freq = inv_freq_f32
    if orig_inv_f32 is not None:
        rotary.original_inv_freq = orig_inv_f32

    load_model(model, str(base / "model.safetensors"), strict=True)
    model = model.to(device=dev)
    rotary.inv_freq = inv_freq_f32.to(device=dev, dtype=torch.float32)
    if orig_inv_f32 is not None:
        rotary.original_inv_freq = orig_inv_f32.to(device=dev, dtype=torch.float32)

    if qwen_cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
        if model.model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr():
            raise RuntimeError("Failed to preserve tied embed_tokens/lm_head pointer after load_model")

    loaded_mlp = model.model.layers[target_layer].mlp
    if not isinstance(loaded_mlp, KernQwen3UpcycledMoEMLP):
        raise RuntimeError(f"Layer {target_layer} mlp was not reconstructed as KernQwen3UpcycledMoEMLP")
    if loaded_mlp.router.weight.dtype != torch.float32:
        raise RuntimeError(f"Router weight drifted to {loaded_mlp.router.weight.dtype}, expected torch.float32")
    if loaded_mlp.expert_bias.dtype != torch.float32:
        raise RuntimeError(f"expert_bias drifted to {loaded_mlp.expert_bias.dtype}, expected torch.float32")
    if len(loaded_mlp.routed_experts) != int(up_cfg["num_routed_experts"]):
        raise RuntimeError("Missing routed_experts in reconstructed MoE block")

    _attach_mask_hooks(model)
    model.eval()
    return model, kern_meta
