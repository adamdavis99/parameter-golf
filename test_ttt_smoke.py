"""
CPU smoke test for TTT LoRA eval — no GPU or real data needed.

Verifies:
  - eval_val_sliding runs without errors
  - eval_val_ttt_lora runs without errors (full TTT code path)
  - _find_docs correctly locates BOS-marked document boundaries
  - No NaN/inf in any output

Run:
  python test_ttt_smoke.py
"""
from __future__ import annotations

import contextlib
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Patch torch.autocast to a no-op before importing the TTT module.
# The TTT functions hardcode device_type="cuda"; on CPU this context manager
# is not needed, so we replace it for the duration of this script.
torch.autocast = lambda *a, **kw: contextlib.nullcontext()

TTT_DIR = Path(__file__).parent / "records/track_10min_16mb/2026-03-22_TTT_LoRA"
sys.path.insert(0, str(TTT_DIR))
import train_gpt as ttt  # noqa: E402


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_bin_shard(path: Path, tokens: list[int]) -> None:
    """Write tokens as a 20240520-format .bin shard."""
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        np.array(tokens, dtype=np.uint16).tofile(f)


def fake_luts(vocab_size: int, device: torch.device):
    """Return (base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)."""
    base_bytes = torch.ones(vocab_size, dtype=torch.int16, device=device)
    has_space = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    is_boundary = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    is_boundary[0] = True  # token 0 = padding
    is_boundary[1] = True  # token 1 = BOS
    return base_bytes, has_space, is_boundary


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cpu")
    torch.manual_seed(42)

    # ── tiny model config ────────────────────────────────────────────────────
    args = ttt.Hyperparameters()
    args.num_layers = 2
    args.model_dim = 64
    args.num_heads = 2
    args.num_kv_heads = 2
    args.mlp_mult = 2.0
    args.vocab_size = 128
    args.bigram_vocab_size = 64
    args.bigram_dim = 32
    args.tie_embeddings = True
    args.tied_embed_init_std = 0.02
    args.logit_softcap = 30.0
    args.rope_base = 10000.0
    args.qk_gain_init = 1.0
    # sliding eval
    args.train_seq_len = 64
    args.eval_stride = 32
    args.eval_batch_seqs = 4
    # TTT eval
    args.ttt_lora_rank = 4
    args.ttt_lora_lr = 0.01
    args.ttt_chunk_size = 32
    args.ttt_eval_seq_len = 128
    args.ttt_batch_size = 2
    args.beta1 = 0.9
    args.beta2 = 0.95

    model = ttt.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── fake token stream: 3 docs separated by BOS (id=1) ───────────────────
    # layout: [BOS, tok*199, BOS, tok*199, BOS, tok*199]  → 600 tokens total
    fake_tokens = torch.randint(2, args.vocab_size, (600,)).tolist()
    fake_tokens[0] = 1    # doc 1 BOS
    fake_tokens[200] = 1  # doc 2 BOS
    fake_tokens[400] = 1  # doc 3 BOS

    # ── LUTs (fake: 1 byte per token, no spaces, no extra boundaries) ────────
    base_bytes_lut, has_space_lut, is_boundary_lut = fake_luts(args.vocab_size, device)

    # ── Test 1: _find_docs ───────────────────────────────────────────────────
    print("\n[1] _find_docs...")
    all_toks = torch.tensor(fake_tokens)
    docs = ttt._find_docs(all_toks)
    assert len(docs) == 3, f"Expected 3 docs, got {len(docs)}: {docs}"
    for i, (start, length) in enumerate(docs):
        assert all_toks[start] == 1, f"Doc {i} does not start with BOS (start={start})"
    print(f"    OK — found {len(docs)} docs: starts={[s for s,_ in docs]}, lengths={[l for _,l in docs]}")

    # ── Test 2: eval_val_sliding ─────────────────────────────────────────────
    print("\n[2] eval_val_sliding...")
    val_tokens = torch.tensor(fake_tokens, dtype=torch.int64)
    loss_s, bpb_s = ttt.eval_val_sliding(
        args, model,
        rank=0, world_size=1,
        device=device,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_space_lut,
        is_boundary_token_lut=is_boundary_lut,
        stride=args.eval_stride,
        batch_seqs=args.eval_batch_seqs,
    )
    assert not math.isnan(bpb_s) and not math.isinf(bpb_s), f"NaN/inf in sliding BPB: {bpb_s}"
    assert bpb_s > 0, f"Non-positive sliding BPB: {bpb_s}"
    print(f"    OK — sliding_loss={loss_s:.4f}  sliding_bpb={bpb_s:.4f}")

    # ── Test 3: eval_val_ttt_lora ────────────────────────────────────────────
    print("\n[3] eval_val_ttt_lora...")
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = Path(tmpdir) / "val_00000.bin"
        make_bin_shard(shard_path, fake_tokens)
        args.val_files = str(Path(tmpdir) / "val_*.bin")

        loss_t, bpb_t = ttt.eval_val_ttt_lora(
            args, model,
            rank=0, world_size=1,
            device=device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_space_lut,
            is_boundary_token_lut=is_boundary_lut,
        )

    assert not math.isnan(bpb_t) and not math.isinf(bpb_t), f"NaN/inf in TTT BPB: {bpb_t}"
    assert bpb_t > 0, f"Non-positive TTT BPB: {bpb_t}"
    print(f"    OK — ttt_loss={loss_t:.4f}      ttt_bpb={bpb_t:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SMOKE TEST PASSED")
    print(f"  sliding_bpb = {bpb_s:.4f}")
    print(f"  ttt_bpb     = {bpb_t:.4f}")
    print("(Model is random/untrained — BPB values are not meaningful)")
    print("=" * 50)


if __name__ == "__main__":
    main()
