# Project KERN

**Practical language-model architecture research on consumer hardware.** The first public release, **KERN-Lab-0.5-624M**, asks a focused question: can a small pretrained Qwen model retain its text modeling quality when one dense feed-forward block is replaced by a sparse mixture of experts?

[Model and weights on Hugging Face](https://huggingface.co/VAL2409/KERN-Lab-0.5-624M) · [Architecture notes](ARCHITECTURE.md) · [Changelog](CHANGELOG.md)

## First public release

| Property | Value |
|---|---|
| Public model name | KERN-Lab-0.5-624M |
| Unique resident parameters | 624,364,544 |
| Active parameters per token (counting convention) | 614,927,360 |
| Transformer layers | 28 |
| Base / tokenizer | [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), original tokenizer |
| Sparse block | Zero-based layer 14, one shared expert and three routed experts, top-2 |
| Weight precision | BF16, with an FP32 router |
| Configured context | 32,768 tokens, inherited from Qwen |
| Adaptation | 200 optimizer steps, 203,948 useful tokens |
| License | Apache-2.0 |

The 624M model name uses a rounded family label. The table gives exact counts.

## Development results

Bits per UTF-8 byte (BPB), lower is better. The development samples contain 45,088 bytes (A) and 262,144 bytes (B). Set B was used to select the checkpoint.

| Model | Set A BPB | Set B BPB |
|---|---:|---:|
| Original Qwen | 0.728382 | 1.091826 |
| Trained dense control | 0.720180 | 1.090734 |
| KERN MoE seed 42 (release checkpoint) | **0.720886** | **1.088243** |
| KERN MoE seed 1337 | 0.720581 | 1.088777 |

The result on B is a development observation. An independent final test would be needed to establish a general quality gain. The 155 passing tests were from the project's internal engineering suite. Short generation and KV-cache checks cover their tested inputs. Long-context quality, overall reasoning gains, runtime speedups and semantic expert specialization remain unverified.

## Run the model

[Open the Google Colab notebook](https://colab.research.google.com/github/arthurvalentimfarias/Project-KERN/blob/main/KERN_Colab.ipynb) for a guided setup. For a local run, use Python 3.12 and install [PyTorch for your platform](https://pytorch.org/get-started/locally/). Choose the CUDA build to run on an NVIDIA GPU.

~~~bash
python -m pip install "huggingface_hub>=1.0"
hf download VAL2409/KERN-Lab-0.5-624M --local-dir ./model
python -m pip install -r model/requirements.txt
python model/generate.py --model-dir ./model --prompt "The capital of France is" --max-new-tokens 16 --device cpu --dtype float32
~~~

For GPU, change the last two options to `--device cuda --dtype bfloat16` on a BF16-capable NVIDIA GPU. Use `--dtype float32` for GPU devices without BF16 support. Treat the prompt as a text prefix for the base model.

### Transformers AutoClass

AutoClass reconstructs the sparse layer using the model repository's Python module. Review that module and set `trust_remote_code=True` for the reviewed snapshot. The example below uses the downloaded local copy. For a reproducible download, pass `--revision` with the Hub commit SHA to `hf download`.

~~~python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

path = "./model"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(path, config=config, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    path, config=config, dtype=dtype, trust_remote_code=True, local_files_only=True
).to(device).eval()
inputs = tokenizer("The capital of France is", return_tensors="pt").to(device)
with torch.inference_mode():
    generated = model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True)
print(tokenizer.decode(generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
~~~

### Optional local web interface

~~~bash
python -m pip install -r model/requirements-ui.txt
python model/app.py --model-dir ./model --device auto
~~~

Open the localhost URL printed by Gradio. One text box accepts a prefix and returns its continuation.

## Scope and provenance

This is an experimental base model for text continuation. It has no instruction tuning or chat alignment. Outputs can be incorrect, repetitive, or unsafe. Review outputs before relying on them.

The pinned upstream revision is `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. The approved release weights have SHA-256 `cc1ac73225ccc6013b1e73d91a6299078f7acd1b67940bf5277328dac43b0fd2`. See [NOTICE](NOTICE) for source attribution and modifications, [LICENSE](LICENSE) for license terms, and [TRADEMARK.md](TRADEMARK.md) for guidance on the Project KERN name.
