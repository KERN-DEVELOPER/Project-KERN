"""Optional localhost-only Gradio interface for text continuation."""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modeling_kern_qwen06b_v1 import load_kern_qwen06b_v1


def build_interface(model_dir: Path, device: str):
    import gradio as gr

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model, _ = load_kern_qwen06b_v1(model_dir, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, config=model.config, local_files_only=True, trust_remote_code=False
    )

    def continue_text(prompt: str, max_new_tokens: int) -> str:
        if not prompt.strip():
            raise gr.Error("Enter a text prefix to continue.")
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        if prompt_tokens + int(max_new_tokens) > model.config.max_position_embeddings:
            raise gr.Error("The prompt and requested continuation exceed the configured context.")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True)

    return gr.Interface(
        fn=continue_text,
        inputs=[
            gr.Textbox(label="Text to continue", lines=4, value="A group of researchers discovered"),
            gr.Slider(1, 256, value=48, step=1, label="New tokens"),
        ],
        outputs=gr.Textbox(label="Continuation", lines=6),
        title="KERN-Lab-0.5-624M",
        description="Experimental base model for text continuation. Runs locally.",
        api_name="generate",
        analytics_enabled=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    options = parser.parse_args()
    device = options.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    build_interface(options.model_dir, device).launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
