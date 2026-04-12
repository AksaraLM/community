# RFC-001: Model Architecture

| Field | Value |
|-------|-------|
| **RFC Number** | 001 |
| **Title** | aksaraLLM Base Model Architecture |
| **Author** | [To be filled by architecture team] |
| **Status** | Draft |
| **Created** | 2026-04-10 |

## Summary

This RFC proposes the base architecture for the aksaraLLM model family, targeting a decoder-only Transformer with modern improvements for efficient training and inference.

## Motivation

We need to decide on a concrete architecture before beginning training experiments. The architecture should:
1. Be proven at scale (evidence from existing models)
2. Support efficient training on limited compute
3. Support efficient inference (quantization, KV cache)
4. Be compatible with existing ecosystems (HuggingFace, vLLM, llama.cpp)

## Proposed Architecture

### Base Architecture: Decoder-Only Transformer (LLaMA-style)

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Architecture | Decoder-only Transformer | Proven, well-understood |
| Attention | Grouped-Query Attention (GQA) | Faster inference, less KV cache |
| Positional Encoding | Rotary Position Embeddings (RoPE) | Proven, supports length extrapolation |
| Normalization | RMSNorm (pre-norm) | Faster than LayerNorm, stable training |
| Activation | SwiGLU | Better performance than ReLU/GELU |
| Embedding | Tied input/output embeddings | Reduces parameters |
| Bias | No bias in attention/MLP | Simpler, follows modern practice |

### Model Sizes

| Model | Layers | Hidden | Heads | KV Heads | Intermediate | Params |
|-------|--------|--------|-------|----------|-------------|--------|
| aksaraLLM-125M | 12 | 768 | 12 | 12 | 3072 | 125M |
| aksaraLLM-350M | 24 | 1024 | 16 | 4 | 4096 | 350M |
| aksaraLLM-1B | 24 | 2048 | 16 | 4 | 5504 | 1.1B |
| aksaraLLM-7B | 32 | 4096 | 32 | 8 | 11008 | 6.7B |

### Context Length
- Initial: 4096 tokens
- Extended (via continued pre-training): 8192-32768 tokens
- RoPE theta: 500,000 (supports long context)

### Tokenizer
- Type: BPE (SentencePiece or HuggingFace Tokenizers)
- Vocab size: 65,536
- Special focus on Indonesian/SEA language coverage

## Alternatives Considered

### Mixture of Experts (MoE)
- **Pro**: More parameters per FLOP
- **Con**: Complex infrastructure, harder to serve, less ecosystem support
- **Decision**: Defer to v2. Start with dense Transformer for simplicity.

### State Space Models (Mamba-style)
- **Pro**: Linear scaling with sequence length
- **Con**: Still maturing, limited ecosystem support
- **Decision**: Monitor progress, consider for future versions.

### Multi-Head Attention (MHA) vs GQA
- **Pro MHA**: Simpler
- **Pro GQA**: 2-4x faster inference, less memory
- **Decision**: GQA for 1B+ models, MHA for 125M/350M.

## Open Questions

- [ ] Exact vocab size (64K vs 128K) — depends on tokenizer fertility analysis
- [ ] Whether to use sliding window attention for efficiency
- [ ] Flash Attention version requirements
- [ ] Exact intermediate size ratios

## References

- [LLaMA 2 Paper](https://arxiv.org/abs/2307.09288)
- [Mistral 7B Paper](https://arxiv.org/abs/2310.06825)
- [OLMo Paper](https://arxiv.org/abs/2402.00838)
- [GQA Paper](https://arxiv.org/abs/2305.13245)
- [RoPE Paper](https://arxiv.org/abs/2104.09864)
- [SwiGLU Paper](https://arxiv.org/abs/2002.05202)
