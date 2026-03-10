# Condition B — Reflection Notes

*After every experiment, reflect on what happened. No predictions, no theory document.*

---

## Exp 0 — Baseline
val_loss = 4.804913, 111 steps in 120s budget.

The baseline uses a 6-layer, 384-dim model with 6 heads. With batch size 8 and no gradient accumulation, it gets through 111 steps. The model is fairly large for the time budget — it may be worth exploring whether smaller/faster models can do more steps and learn more, or whether bigger models learn enough per step to justify fewer iterations.

---

## Exp 1 — LR 3e-4 -> 1e-3
val_loss = 5.331203, 66 steps. Discard.

Much worse. The higher LR caused instability with this large model.

---

## Exp 2 — LR 3e-4 -> 2e-3
val_loss = 5.262449, 71 steps. Discard.

Even higher LR is also bad. Higher LRs don't help with the big model.

---

## Exp 3 — N_LAYER=4, N_HEAD=3, N_EMBD=192
val_loss = 5.289694, 131 steps. Discard.

Too small — insufficient capacity.

---

## Exp 4 — N_LAYER=4, N_HEAD=4, N_EMBD=256
val_loss = 4.298028, 179 steps. NEW BEST.

Big improvement! Smaller model = more steps = better performance within the time budget.

---

## Exp 5 — WARMUP_STEPS: 50 -> 10
val_loss = 4.124383, 196 steps. NEW BEST.

Reducing warmup helped. With only ~180 steps, 50-step warmup wastes too much training time.

---

## Exp 6-8 — WARMUP variations (0, 5, 20)
All worse than WARMUP=10. WARMUP=10 is optimal for this model size.

---

## Exp 9 — LR: 3e-4 -> 6e-4
val_loss = 3.907157. NEW BEST. Higher LR works with smaller model.

---

## Exp 10 — LR: 6e-4 -> 1e-3
val_loss = 3.598455. NEW BEST. LR=1e-3 works great with the smaller model! Same LR that was disastrous with 6 layers works well with 4 layers.

---

## Exp 11 — LR: 1e-3 -> 2e-3
val_loss = 3.980995. Discard. LR=2e-3 is too aggressive.

---

## Exp 12 — DROPOUT: 0.1 -> 0.0
val_loss = 3.518768. NEW BEST. With only 120s of training, there's no overfitting risk. Dropout just slows learning.

---

## Exp 13 — DROPOUT: 0.0 -> 0.05
val_loss = 3.580576. Discard. Any dropout hurts.

---

## Exp 14-16 — N_LAYER sweep (5, 3, 2)
5 layers: worse. 3 layers: better (3.483062). 2 layers: even better (3.411116). The trend of fewer layers = more steps = better continues. 2 layers is optimal.

---

## Exp 17-19 — N_EMBD sweep (384, 128, 192)
All worse than 256. 384 is too slow, 128 has insufficient capacity, 192 is close but 256 wins (at this stage).

---

## Exp 20-21 — N_HEAD sweep (2, 8)
Both worse than 4. N_HEAD=4 is optimal.

---

## Exp 22-23 — WEIGHT_DECAY sweep (0.01, 0.0)
0.01 slightly better than 0.1, and 0.0 even better. No regularization needed in this short-training regime.

---

## Exp 24-26 — BATCH_SIZE sweep (16, 4, 2)
16: much worse (too few steps). 4: better than 8. 2: worse than 4 at that point. Batch size 4 was the sweet spot initially.

---

## Exp 27-28 — BETAS sweep
(0.9,0.99) improved to 3.239009. (0.9,0.999) improved further to 3.205589. Higher beta2 helps — smoother gradient estimates benefit this noisy, short-training regime.

---

## Exp 29 — GRAD_ACCUM=2
val_loss = 3.272305. Discard. Accumulation halves optimizer steps.

---

## Exp 30-32 — LR sweep with new config
LR=2e-3: worse. LR=6e-4: better (3.174884)! LR=3e-4: worse. With BETAS=(0.9,0.999), lower LR works better.

---

## Exp 33 — N_LAYER: 2 -> 3
val_loss = 3.181223. Close but worse. 2 layers remains optimal.

---

## Exp 34-35 — WARMUP revisit
WARMUP=5: 3.133402 (better!). WARMUP=0: 3.123270 (even better!). With the optimized config, no warmup works best.

---

## Exp 36 — BATCH_SIZE: 4 -> 2
val_loss = 3.085317. Better! With all optimizations, BS=2 now works (1135 steps). The pattern of more steps being better continues.

---

## Exp 37 — LR: 6e-4 -> 1e-3 with BS=2
val_loss = 3.042566. Better! With 1138 steps, higher LR is justified again.

---

## Exp 38 — LR: 1e-3 -> 2e-3
val_loss = 3.647152. Way worse. LR=1e-3 is the ceiling.

---

## Exp 39-48 — Various architecture and optimizer explorations
All worse. The current config (N_LAYER=2, N_EMBD=256, N_HEAD=4, BS=2) is well-optimized.

---

## Exp 49-57 — Extensive re-exploration
Tried many variations from current best. None improved. The config seems well-converged.

---

## Exp 58 — N_EMBD: 256 -> 192
val_loss = 3.033880. NEW BEST! With 1256 steps (vs 1138 with 256), the smaller embedding pays off. The extra steps from the faster model compensate for reduced capacity.

---

## Exp 59-62 — Fine-tuning around best
N_EMBD=128: worse (too small). N_HEAD=3: slightly worse. N_HEAD=6: virtually identical but slightly worse. LR=6e-4: worse. The current config is the optimum.

---

## Summary
Starting from val_loss=4.804913, optimized to val_loss=3.033880 (37% reduction).

Final best config: N_LAYER=2, N_HEAD=4, N_EMBD=192, DROPOUT=0.0, LR=1e-3, WD=0.0, BS=2, GRAD_ACCUM=1, WARMUP=0, BETAS=(0.9,0.999).

Key findings:
1. Throughput dominates: smaller, faster models that can do more training steps in 120s beat larger models.
2. No regularization needed: DROPOUT=0.0 and WEIGHT_DECAY=0.0 are optimal for short training.
3. BETAS=(0.9,0.999) gives best gradient smoothing for noisy small-batch training.
4. Batch size 2 maximizes optimizer steps per second.
5. No warmup works best when training is very short.
6. LR=1e-3 is the sweet spot for the final small model.

---
