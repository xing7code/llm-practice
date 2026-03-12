# llm-practice

A personal repo for practicing LLM infrastructure from scratch in PyTorch. Built for learning and interview prep — no shortcuts, no wrapper libraries.

## Contents

### `transformer.py`
A full transformer implementation with:
- Multi-head self-attention with GQA (grouped query attention) and RoPE
- KV cache for efficient autoregressive inference
- Top-k / top-p sampling
- Flash attention (online softmax, block-sparse)
- Tensor parallelism (TP) + sequence parallelism (SP)
- Context parallelism via ring attention (causal, zigzag-balanced)

## Roadmap

- [ ] Data parallelism (DDP)
- [ ] Pipeline parallelism (1F1B schedule)
- [ ] Mixture of Experts (MoE)
- [ ] Expert parallelism (EP)
- [ ] ZeRO-1 / ZeRO-2 / ZeRO-3

## Goals

Each implementation aims to be:
- **From scratch** — core logic in plain PyTorch, no Megatron/DeepSpeed dependencies
- **Readable** — written for clarity and learning, not production throughput
- **Correct** — tested against known-good references where possible

## License

MIT
