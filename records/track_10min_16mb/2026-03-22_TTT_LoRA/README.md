# 2026-03-22: TTT LoRA on SOTA Model

## Summary

Adds **Test-Time Training with LoRA adapters** to the current SOTA
(10L Int5MLP + BigramHash(10240) + SmearGate + MuonWD=0.04 + SWA50 + sliding window eval).
Zero artifact cost — LoRA weights are initialized fresh per document at eval time and never stored.

## Key Changes

### TTT LoRA Evaluation (`eval_val_ttt_lora`)

- Finds document boundaries in validation data using BOS tokens (id=1)
- For each document, maintains a `BatchedTTTLoRA` adapter (Q, V projections + lm_head per block)
- Processes document in `chunk_size=256` token chunks using a sliding context window
- Before scoring each chunk: run one Adam step to adapt the LoRA to the preceding context
- Score is computed on the current chunk **before** the Adam step updates it (true online eval)

### Model Changes (eval-only, no architecture change during training)

- `CausalSelfAttention.forward_ttt`: adds Q and V deltas from LoRA adapters
- `Block.forward_ttt`: applies LoRA through attention, routes pre-norm activations to LoRA
- `GPT.forward_ttt`: full forward with LoRA adapters, returns per-token NLL [bsz, seqlen]

### Hyperparameters

| Param | Value | Notes |
|-------|-------|-------|
| `ttt_lora_rank` | 8 | LoRA rank per layer |
| `ttt_lora_lr` | 0.01 | Adam lr for TTT updates |
| `ttt_chunk_size` | 256 | Tokens per TTT chunk |
| `ttt_batch_size` | 64 | Documents per TTT batch |
| `ttt_eval_seq_len` | 2048 | Max context window |

## Expected Improvement

- Reference TTT LoRA baseline: -0.032 BPB (1.2244 → 1.1928, ~2.6% relative)
- On SOTA (1.1428): conservative estimate -0.011 BPB → target ~1.132 BPB

## Byte Budget

Zero bytes — no model weights stored. Only code size increases (~8KB).

## Timing

TTT eval is roughly proportional to `(val_tokens / chunk_size) × (forward + backward)`.
With 8xH100 and batching, estimated 2-4 minutes additional eval time.

Tuning knobs if timing is tight:
- Reduce `TTT_LORA_RANK` to 4
- Increase `TTT_CHUNK_SIZE` to 512
- Reduce `TTT_BATCH_SIZE` to 32
