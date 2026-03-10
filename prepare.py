"""
One-time data preparation + runtime utilities for autoresearch (Apple Silicon).

Usage:
    python prepare.py              # download data + prepare shards
    python prepare.py --tiny       # use tiny subset for quick testing

Data is stored in ~/.cache/autoresearch-mps/.
"""

import os
import sys
import math
import pickle
import argparse
import numpy as np
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 512         # context length (shorter than Karpathy's 2048 for MPS speed)
TIME_BUDGET = 600          # training time budget in seconds (2 minutes)
EVAL_TOKENS = 10 * 131072  # tokens for validation eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-mps")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# We use the GPT-2 tokenizer (50257 vocab) — no training needed
ENCODING_NAME = "gpt2"

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(tiny=False):
    """Download TinyStories dataset and tokenize into binary shards."""
    os.makedirs(DATA_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")

    if os.path.exists(train_path) and os.path.exists(val_path):
        train_size = os.path.getsize(train_path) // 2  # uint16
        val_size = os.path.getsize(val_path) // 2
        print(f"Data already prepared: train={train_size:,} tokens, val={val_size:,} tokens")
        return

    print("Downloading TinyStories dataset...")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train+validation")

    # Split: last 5000 docs for val, rest for train
    texts = [row["text"] for row in ds]
    if tiny:
        texts = texts[:20000]
    val_texts = texts[-5000:]
    train_texts = texts[:-5000]

    enc = tiktoken.get_encoding(ENCODING_NAME)
    eot = enc.eot_token  # <|endoftext|>

    def tokenize_and_save(text_list, out_path, label):
        all_tokens = []
        for i, text in enumerate(text_list):
            tokens = enc.encode_ordinary(text)
            all_tokens.append(eot)
            all_tokens.extend(tokens)
            if (i + 1) % 50000 == 0:
                print(f"  {label}: tokenized {i+1}/{len(text_list)} documents...")
        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(out_path)
        print(f"  {label}: {len(all_tokens):,} tokens saved to {out_path}")

    tokenize_and_save(train_texts, train_path, "train")
    tokenize_and_save(val_texts, val_path, "val")
    print("Data preparation complete.")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def get_tokenizer():
    """Return the tiktoken GPT-2 encoding."""
    return tiktoken.get_encoding(ENCODING_NAME)

def get_vocab_size():
    """Return vocab size for the GPT-2 tokenizer."""
    return 50257

def make_dataloader(split, batch_size, seq_len, device="mps"):
    """
    Simple random-offset dataloader from pre-tokenized binary file.
    Yields (x, y) where x and y are (B, T) tensors on device.
    """
    assert split in ("train", "val")
    filename = "train.bin" if split == "train" else "val.bin"
    filepath = os.path.join(DATA_DIR, filename)
    assert os.path.exists(filepath), f"Data file not found: {filepath}. Run prepare.py first."

    data = np.memmap(filepath, dtype=np.uint16, mode='r')
    n = len(data)

    while True:
        offsets = torch.randint(0, n - seq_len - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in offsets])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+seq_len].astype(np.int64)) for i in offsets])
        yield x.to(device), y.to(device)

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loss(model, batch_size, seq_len, device="mps"):
    """
    Evaluate average cross-entropy loss on validation set.
    Returns val_loss (lower is better).
    """
    model.eval()
    val_loader = make_dataloader("val", batch_size, seq_len, device=device)
    steps = EVAL_TOKENS // (batch_size * seq_len)
    total_loss = 0.0
    for _ in range(steps):
        x, y = next(val_loader)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    model.train()
    return total_loss / steps

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for autoresearch")
    parser.add_argument("--tiny", action="store_true", help="Use tiny subset for testing")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()
    prepare_data(tiny=args.tiny)
    print()
    print("Done! Ready to train with: uv run train.py")
