# Theory Document — Condition C, Phase 2

## Transfer from Phase 1 (120s budget)

### Confirmed Principles (Phase 1)
1. Small models beat large models at short budgets (1.3M >> 14.2M params)
2. Depth > width: 5L/128E > 3L/192E
3. Optimal embedding = 128 for 120s budget
4. Optimal depth = 5 layers at 128E
5. 2 heads optimal for 128E (head_dim=64)
6. High LR critical: 2e-3 >> 3e-4
7. beta2=0.999 optimal (not 0.95)
8. LR and beta2 interact: beta2=0.999 amplifies effective LR
9. Warmup=20 essential for LR=2e-3/beta2=0.999
10. BS=4 optimal at 120s
11. All regularization harmful at 120s (DO=0, WD=0)
12. MPS has massive throughput variance (~40%)

### Phase 2 Hypotheses (600s = 5x more time)

**H1: Larger models may now benefit.** With 5x more training time, the throughput-capacity tradeoff shifts. Models that were undertrained at 120s might now get enough steps.

**H2: Batch size may increase.** With more steps available, BS=8 might beat BS=4. Gradient quality improvement matters more when you have enough steps.

**H3: Regularization may now help.** With 5x more steps, the model may start to overfit. Small dropout or weight decay could become beneficial.

**H4: Learning rate may need adjustment.** With more steps, the cosine decay schedule covers more ground.

**H5: Architecture depth may change.** More training time means deeper models can converge.

## Phase 2 Confirmed
1. Phase 1 config transfers to 600s: val_loss improved from 3.03 to 2.67 with 5x training time.
2. MPS scaling is sublinear: got 2580 steps (3.2x) in 5x time, not 5x steps.
3. **Cosine LR schedule wrapping is a major issue.** max_steps_estimate=2000 causes wrapping at BS=4 (2580 steps). BS=8 gives 1766 steps, keeping the schedule clean. This alone improved val_loss by 0.21.
4. **BS=8 optimal at 600s** (Phase 1 had BS=4). More time shifts the step-count/gradient-quality tradeoff toward quality.
5. **Phase 1 belief "BS=4 optimal" is INVALIDATED at 600s.** BS=8 is clearly better.
6. **Optimal model size shifts with budget.** At 120s, 1.3M (128E) was optimal. At 600s, 3.0M (192E) is optimal. 5.2M (256E) is still too large. The capacity-throughput sweet spot scales with compute budget.
7. **beta2=0.99 beats beta2=0.999 at 600s.** More training steps benefit from faster moment adaptation. beta2=0.95 is too aggressive (catastrophically worse).
8. **LR=2e-3 remains optimal.** Neither 1e-3 nor 3e-3 improved results at 600s.
9. **5 layers remains optimal depth** at both 128E and 192E. 4L and 6L are both worse.
10. **4 heads optimal at 192E (head_dim=48).** 2H (head_dim=96) and 6H (head_dim=32) are both worse. Wider models need more attention heads than narrow models (128E used 2H).
11. **Dropout still harmful at 600s.** DO=0.05 worsens results (2.47 vs 2.43).
12. **WD=0.01 is marginally beneficial** at 600s (within noise).
13. **Warmup 20-50 are equivalent** at 128E. warmup=10 is too little.
14. **MPS thermal throttling is a severe confound.** Step counts can drop from ~1500 to ~600-800, making results unreliable. Multiple runs affected (exps 8, 16, 18, 19).

## Phase 2 Refuted
1. "BS=4 is optimal" -- only true at 120s. At 600s, BS=8 is much better (2.46 vs 2.67).
2. "128E is optimal embedding" -- only true at 120s. At 600s, 192E is better (2.38 vs 2.43).
3. "2 heads optimal" -- only true at 128E. At 192E, 4 heads is optimal.
4. "beta2=0.999 optimal" -- only true at ~800 steps. At ~1500-1900 steps, beta2=0.99 is better.

## Final Best Config (val_loss=2.379145)
- Architecture: 5L/4H/192E (3.0M params)
- Optimizer: LR=2e-3, WD=0.01, betas=(0.9, 0.99)
- Training: BS=8, GA=1, warmup=50, DO=0.0
- ~1525 steps, ~6.2M tokens in 600s

## Experiments 18-19: Final Attention & Warmup Tests
- **Exp 18 (6H/head_dim=32):** val_loss=2.629 (834 steps, MPS throttled). Even accounting for throttling, loss trajectory was worse than 4H throughout. Confirms head_dim=48 is the sweet spot at 192E.
- **Exp 19 (warmup=20 at 192E):** val_loss=2.751 (738 steps, MPS throttled). Inconclusive due to throttling. warmup=50 retained.
- Both experiments were heavily confounded by MPS thermal throttling (step counts ~50% of expected). The diminishing returns zone was reached at exp 13; experiments 14-19 all failed to improve.

## Key Scaling Insight
The optimal configuration depends heavily on compute budget. As budget increases from 120s to 600s:
- Model size increases (1.3M -> 3.0M)
- Batch size increases (4 -> 8)
- beta2 decreases (0.999 -> 0.99)
- Head count increases with width (2 -> 4)
- Depth stays constant (5L)
- LR stays constant (2e-3)
- Regularization goes from harmful to neutral/marginal

## Phase 2 Complete (20 experiments, 0-19)
Final best: val_loss=2.379145 (exp 13). The optimization trajectory shows three phases:
1. **Infrastructure fixes (exps 0-1):** BS=8 fixed cosine schedule wrapping (-0.21 improvement)
2. **Optimizer tuning (exps 4-10):** beta2=0.99, WD=0.01, warmup=50 (-0.03 cumulative)
3. **Architecture scaling (exp 13):** 192E width increase (-0.05 improvement)
4. **Diminishing returns (exps 14-19):** No further improvement found

Total improvement from Phase 1 baseline: 3.03 -> 2.379 = -0.65 (21% reduction in val_loss).
