"""
Autoresearch training script (Apple Silicon / MPS).

This is the ONLY file the agent modifies.
Contains: GPT model, optimizer, training loop.

Usage: uv run train.py
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import (
    MAX_SEQ_LEN, TIME_BUDGET, get_vocab_size,
    make_dataloader, evaluate_loss,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Model
N_LAYER = 6
N_HEAD = 6
N_EMBD = 192
DROPOUT = 0.1

# Optimizer
LEARNING_RATE = 1e-03
WEIGHT_DECAY = 0e+00
BETAS = (0.9, 0.95)

# Training
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 1
WARMUP_STEPS = 50

# Eval
EVAL_BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x):
        seq_len = x.size(-2)
        return self.cos[:seq_len], self.sin[:seq_len]

def apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        cos, sin = self.rotary(q)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        hidden = 4 * n_embd
        self.up = nn.Linear(n_embd, hidden, bias=False)
        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, dropout, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits

    def num_params(self):
        # Don't double-count tied weights
        return sum(p.numel() for p in self.parameters()) - self.tok_emb.weight.numel()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    device = "mps"
    torch.manual_seed(42)

    vocab_size = get_vocab_size()
    model = GPT(vocab_size, N_LAYER, N_HEAD, N_EMBD, DROPOUT).to(device)
    num_params = model.num_params()
    print(f"Model parameters: {num_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = make_dataloader("train", BATCH_SIZE, MAX_SEQ_LEN, device=device)

    # Warmup + cosine decay schedule
    def get_lr(step):
        if step < WARMUP_STEPS:
            return LEARNING_RATE * (step + 1) / WARMUP_STEPS
        return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * step / max_steps_estimate))

    max_steps_estimate = 2000  # rough estimate for schedule

    # --- Training loop (fixed time budget) ---
    print(f"Training for {TIME_BUDGET}s on {device}...")
    model.train()
    step = 0
    tokens_seen = 0
    best_train_loss = float("inf")
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed >= TIME_BUDGET:
            break

        # Update learning rate
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = next(train_loader)
            logits, loss = model(x, y)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tokens_seen += BATCH_SIZE * MAX_SEQ_LEN * GRAD_ACCUM_STEPS
        step += 1

        if step % 50 == 0:
            print(f"  step {step:5d} | loss {accum_loss:.4f} | lr {lr:.2e} | tokens {tokens_seen/1e6:.1f}M | {elapsed:.0f}s")

    training_seconds = time.time() - start_time

    # --- Evaluation ---
    print("Evaluating...")
    eval_start = time.time()
    val_loss = evaluate_loss(model, EVAL_BATCH_SIZE, MAX_SEQ_LEN, device=device)
    eval_seconds = time.time() - eval_start
    total_seconds = training_seconds + eval_seconds

    # --- Summary ---
    print("---")
    print(f"val_loss:         {val_loss:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"total_tokens_M:   {tokens_seen/1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params/1e6:.1f}")
    print(f"depth:            {N_LAYER}")

if __name__ == "__main__":
    main()
