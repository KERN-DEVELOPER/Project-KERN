# KERN-Lab-0.5-624M architecture

The pinned Qwen3-0.6B-Base checkpoint contributes 28 Transformer layers, grouped-query attention, RoPE, and tied token embeddings and output weights. In zero-based layer 14, the original SwiGLU feed-forward block is replaced with one frozen shared expert and three routed experts. The router computes scores in FP32. It selects the top two routed experts per token and renormalizes their weights. The block returns an equal mixture of the shared path and the weighted routed sum. The remaining Transformer layers retain the upstream architecture and weights.

The release contains 624,364,544 unique resident parameters. The per-token active parameter convention counts all shared and dense weights plus the selected routed experts, totaling 614,927,360. The 624M name is a family label. These counts do not measure FLOPs, memory traffic, or speed.

Transformers uses the custom `auto_map` classes to install the sparse block before loading the published SafeTensors state. The standalone loader provides a second way to load the same file. Both reconstruct the tied head and the FP32 router. The model uses the upstream tokenizer. See the [README](README.md) for evaluation limits and the tested loading modes.
