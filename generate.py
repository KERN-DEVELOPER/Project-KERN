"""Standalone text generation CLI for KERN v1 (Qwen3-0.6B-Base Layer-14 MoE adaptation)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modeling_kern_qwen06b_v1 import KernQwen3UpcycledMoEMLP, load_kern_qwen06b_v1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate text from KERN v1 adapted Qwen3-0.6B-Base Layer-14 MoE package."
    )
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    dev_str = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    model, kern_meta = load_kern_qwen06b_v1(args.model_dir, device=dev_str, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, config=model.config, local_files_only=True, trust_remote_code=False
    )
    dev = next(model.parameters()).device

    target_layer = int(kern_meta["target_layer_idx"])
    moe_layer = model.model.layers[target_layer].mlp
    if isinstance(moe_layer, KernQwen3UpcycledMoEMLP):
        moe_layer.reset_routing_stats()

    enc = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(dev)
    attention_mask = enc["attention_mask"].to(dev)

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)

    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(args.max_new_tokens),
        "use_cache": (not args.no_cache),
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else model.config.eos_token_id,
        "eos_token_id": model.config.eos_token_id,
    }
    if float(args.temperature) <= 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = float(args.temperature)
        gen_kwargs["top_p"] = float(args.top_p)

    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(**gen_kwargs)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        peak_alloc_mib = float(torch.cuda.max_memory_allocated(dev)) / (1024.0 ** 2)
        peak_resv_mib = float(torch.cuda.max_memory_reserved(dev)) / (1024.0 ** 2)
    else:
        peak_alloc_mib = 0.0
        peak_resv_mib = 0.0
    wall_sec = max(1e-6, time.perf_counter() - t0)

    gen_tokens = out_ids[0, input_ids.shape[1] :].tolist()
    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    routing_stats = moe_layer.get_routing_stats() if isinstance(moe_layer, KernQwen3UpcycledMoEMLP) else {}
    payload = {
        "prompt": args.prompt,
        "generated_text": generated_text,
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": len(gen_tokens),
        "tokens_per_sec": round(len(gen_tokens) / wall_sec, 2),
        "wall_sec": round(wall_sec, 4),
        "device": str(dev),
        "dtype": str(dtype).replace("torch.", ""),
        "use_cache": (not args.no_cache),
        "peak_alloc_mib": round(peak_alloc_mib, 2),
        "peak_reserved_mib": round(peak_resv_mib, 2),
        "weights_tied": bool(model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()),
        "router_dtype": str(moe_layer.router.weight.dtype).replace("torch.", ""),
        "routing_stats": routing_stats,
    }
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(generated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
