# Theory Document — Condition C (Final)

## Confirmed Principles

### Architecture
1. **Small models beat large models** at 120s training budget. 1.3M params >> 14.2M params (val_loss 3.03 vs 4.39). The throughput gain from smaller models outweighs the capacity loss.
2. **Depth is more efficient than width** for language modeling. 5L/128E > 3L/192E despite similar param counts. Compositional structure requires depth.
3. **Optimal embedding dimension is 128** for this budget. 96E=3.10, 128E=3.03, 192E=3.21, 256E=3.72. The sweet spot balances vocab projection quality with speed.
4. **Optimal depth is 5 layers at 128E.** 3L=3.07, 4L=3.22, 5L=3.03, 6L=3.06, 8L=3.30. 5L is the best depth-throughput trade-off.
5. **2 heads is optimal for 128E** (head_dim=64). 4 heads (head_dim=32) is worse — attention quality matters more than diversity at this scale.

### Optimizer
6. **High learning rate is critical.** LR=2e-3 >> 3e-4 (the default was ~10x too conservative). With few training steps, each step must extract maximum learning.
7. **beta2=0.999 is optimal** (Adam default), not beta2=0.95 (GPT convention). Higher beta2 provides more stable optimization, especially important with no regularization and high LR. This was a major surprise — earlier tests with beta2=0.99 at a different architecture failed, but beta2=0.999 works great.
8. **LR and beta2 interact:** With beta2=0.999, the effective LR is amplified (stable denominator), so nominal LR=2e-3 is better than 3e-3. With beta2=0.95, LR=3e-3 was optimal.
9. **Warmup is essential** for high LR. Warmup=5 and warmup=0 both caused catastrophic instability. Warmup=20 is optimal for LR=2e-3/beta2=0.999. For LR=3e-3/beta2=0.95, warmup=50 was needed.
10. **Batch size 4 is optimal.** BS=2 gives more steps but noisier gradients. BS=8+ gives too few steps. BS=4 balances gradient quality and step count.

### Regularization
11. **All regularization is harmful** at 120s training budget. Dropout=0, WD=0 is optimal. The model never approaches overfitting in ~800 steps. Both dropout and weight decay waste capacity and slow convergence.
12. **Dropout is more harmful at greater depth.** At 3L, dropout had negligible effect. At 6L, removing dropout improved val_loss by 0.06. Cumulative dropout across layers reduces learning signal significantly.
13. **Weight decay is less important with beta2=0.999.** WD=0.0 and WD=0.2 gave nearly identical results (3.06 vs 3.06). The stable second moments from beta2=0.999 overshadow WD's effect.

### Hardware (MPS)
14. **MPS has massive throughput variance.** Same config can get 479-825 steps (40% variance) in the same 120s budget. Thermal throttling and system load are major factors.
15. **MPS per-step overhead is significant.** Halving model size does NOT double step count. The per-step overhead is hardware-limited, not compute-limited at these small model sizes.

## Refuted Hypotheses
1. **"Larger batch = more tokens = better"** — FALSE. BS=32 was catastrophically bad (6.82 val_loss). Step count matters more than tokens/step.
2. **"Weight decay always helps with optimization stability"** — FALSE for this regime. WD=0 is optimal.
3. **"beta2=0.95 is universally better for LLM training"** — FALSE for short training. beta2=0.999 is better.
4. **"More capacity always helps if you can train enough"** — FALSE. Even with optimized training, 192E is worse than 128E. The throughput cost outweighs capacity benefit.

## Open Questions (Answered)
- **Optimal model size for 120s on MPS?** 1.3M params (5L/2H/128E). Gets ~800 steps at stable throughput.
- **Does effective batch size matter more than raw BS?** No — BS=4/GA=1 and BS=2/GA=2 give nearly identical results. What matters is optimization step count and gradient quality.

## Meta-Observations
- **Prediction accuracy improved over time.** Early predictions were off by 0.5-1.0+ (learning the system). By exp 30+, predictions were typically within 0.05 (well-calibrated). The biggest surprises were:
  1. beta2=0.999 being good (exp 35, error=+0.17)
  2. MPS throughput variance (exp 44, error=-0.29)
  3. Large batch size being catastrophic (exp 2, error=-3.32)
- **Best val_loss trajectory:** 4.39 -> 4.03 -> 3.41 -> 3.33 -> 3.26 -> 3.24 -> 3.20 -> 3.14 -> 3.12 -> 3.11 -> 3.08 -> 3.06 -> 3.05 -> 3.03
- **Final best:** val_loss=3.030 (exp 63), 31% improvement from baseline
