# Lab Notebook — Condition C, Phase 2

## Experiment 0: Baseline
**Config:** 5L/2H/128E, DO=0, WD=0, LR=2e-3, BS=4, WU=20, BETAS=(0.9,0.999), GA=1
**Hypothesis:** Running the Phase 1 best config with 600s (5x more time) should substantially improve val_loss. With ~800 steps at 120s, we should get ~4000 steps at 600s (linear scaling assumption).
**Predicted val_loss:** 2.55 (Phase 1 best was 3.03; 5x more steps should yield ~0.5 improvement based on typical log-linear scaling of loss with compute)
**Reasoning:** Language model loss typically improves log-linearly with compute. 5x more steps = ~0.5 improvement on the loss scale. However, MPS overhead may limit the speedup, and diminishing returns could reduce the gain.
**Result:** val_loss=2.669242, 2580 steps, 1.3M params, 5.3M tokens, 600.2s training
**Prediction error:** +0.12 (predicted 2.55, got 2.67). Slightly worse than predicted.
**Learning:** The Phase 1 best config transfers reasonably well to 600s. Got 2580 steps (vs ~800 at 120s = 3.2x scaling, not 5x -- MPS overhead limits scaling). CRITICAL OBSERVATION: The cosine LR schedule has max_steps_estimate=2000, but the model ran 2580 steps. After step 2000, the cosine wraps and LR climbs back up from near-zero. This is suboptimal -- the final 580 steps were training at increasing LR, potentially destabilizing learned representations. This is the #1 thing to fix.

Also note: Phase 2 A baseline with same config got val_loss=3.031 (which is essentially the Phase 1 result since A doesn't transfer knowledge, just config). Phase 2 B started with a different Phase 1 best (2L/4H/192E) and got baseline 2.785. Our 2.669 starting point is already strong.

---

## Experiment 1: BS=8 (fix LR schedule wrapping)
**Config change:** BATCH_SIZE=4 -> 8
**Hypothesis:** Increasing batch size to 8 will reduce steps from ~2580 to ~1290, keeping the cosine LR schedule within its max_steps_estimate=2000 range. The cleaner LR schedule should more than compensate for fewer steps. Also, at 600s budget, BS=8 provides better gradient quality.
**Predicted val_loss:** 2.52 (improvement from LR schedule fix + gradient quality)
**Reasoning:** Phase 2 B found BS=8 was optimal at 600s (improved from 2.78 to 2.57). The LR schedule wrapping issue is likely a major performance bottleneck. BS=8 with ~1290 steps should have proper cosine decay.
**Result:** val_loss=2.455858, 1766 steps, 1.3M params, 7.2M tokens, 600.3s training
**Prediction error:** -0.06 (predicted 2.52, got 2.46). Better than predicted!
**Learning:** MASSIVE improvement (2.67 -> 2.46 = -0.21). BS=8 fixed the cosine schedule (1766 steps < 2000). Two benefits: (1) cosine schedule now decays properly to low LR at end of training, (2) less noisy gradients from larger batches. This already beats Phase 2 B's final best of 2.441! The LR schedule fix was the key insight. Note: 7.2M tokens seen (vs 5.3M at BS=4) -- larger batch sees more tokens per step.

---

## Experiment 2: Larger model — 6L/4H/256E
**Config change:** N_LAYER=5->6, N_HEAD=2->4, N_EMBD=128->256
**Hypothesis:** With 600s training, a larger model (~5.2M vs 1.3M params) should have enough steps to train. More capacity should help. Phase 1 showed 128E was optimal at 120s because larger models were too slow, but 600s changes that tradeoff.
**Predicted val_loss:** 2.40 (slight improvement from more capacity, but fewer steps due to larger model)
**Reasoning:** Model will go from ~1766 to ~900 steps (larger model = slower). Capacity gain from 4x parameters should offset step count reduction. Phase 2 A benefited from larger models.
**Result:** val_loss=2.901234, 1205 steps, 6.3M params, 4.9M tokens
**Prediction error:** +0.50 (predicted 2.40, got 2.90). Much worse than predicted!
**Learning:** The larger model is WAY worse (2.90 vs 2.46). Despite 4.8x more params, it only got 1205 steps and saw 4.9M tokens (vs 7.2M). The 6.3M model is still severely undertrained at 600s. Phase 1 principle "small models beat large models" STILL HOLDS at 600s! The capacity-throughput tradeoff hasn't shifted enough. Revert to 5L/2H/128E.

---

## Experiment 3: Moderate depth increase — 6L/2H/128E
**Config change:** N_LAYER=5->6 (keeping everything else at best)
**Hypothesis:** Adding 1 layer (1.3M -> 1.6M params) is a minimal capacity increase. At 120s, 5L and 6L were very close (3.03 vs 3.06). With more training at 600s, the extra layer might now pay off.
**Predicted val_loss:** 2.44 (slight improvement, depth helps with more training time)
**Reasoning:** The extra layer adds ~23% params but enables more compositional computation. With ~1600 steps (slightly fewer than 1766), the throughput cost is small. Phase 1 showed 6L was close to 5L -- with more training, the depth advantage should manifest.
**Result:** val_loss=2.458175, 1898 steps, 1.6M params, 7.8M tokens
**Prediction error:** +0.02 (predicted 2.44, got 2.46). Very close prediction!
**Learning:** 6L is essentially tied with 5L (2.458 vs 2.456). The extra layer doesn't help enough to overcome the small throughput cost. 5L remains optimal -- this confirms Phase 1 findings even at 600s. Architecture is not the lever; optimizer tuning is where the gains are.

---

## Experiment 4: beta2=0.99 (from 0.999)
**Config change:** BETAS=(0.9,0.999) -> (0.9,0.99)
**Hypothesis:** Phase 2 B found that betas=(0.9,0.99) with BS=8 was a major improvement (2.57 -> 2.46). With more training steps, beta2=0.99 provides faster adaptation of second moments. At 120s, beta2=0.999 was optimal because of few steps (stable denominator needed). At 600s with ~1766 steps, faster moment adaptation may help.
**Predicted val_loss:** 2.42 (modest improvement from better moment adaptation)
**Reasoning:** beta2=0.99 means the denominator of Adam adapts in ~100 steps instead of ~1000. With 1766 steps, this gives better per-param learning rate adaptation throughout training.
**Result:** val_loss=2.429258, 1959 steps, 1.3M params, 8.0M tokens
**Prediction error:** +0.01 (predicted 2.42, got 2.43). Excellent prediction!
**Learning:** NEW BEST (2.456 -> 2.429). beta2=0.99 is better than 0.999 at 600s, confirming the hypothesis. The faster moment adaptation helps with more training steps. This reverses Phase 1's finding where beta2=0.999 was optimal -- the key factor is training steps: more steps = faster adaptation benefits.

Updated theory: beta2 optimal depends on step count. ~800 steps -> 0.999, ~1900 steps -> 0.99.

---

## Experiment 5: Small weight decay WD=0.01
**Config change:** WEIGHT_DECAY=0.0 -> 0.01
**Hypothesis:** Phase 2 B found that WD=0.01 was their final improvement (2.445 -> 2.441). With 600s training and ~1960 steps, the model may be approaching mild overfitting. Tiny weight decay should regularize without hurting convergence.
**Predicted val_loss:** 2.41 (small improvement, regularization now useful)
**Reasoning:** At 120s, WD was harmful (too few steps, regularization wastes capacity). At 600s, the model trains long enough that WD can help generalization. WD=0.01 is very small -- just enough to prevent weight explosion without significant capacity loss.
**Result:** val_loss=2.427666, 1948 steps, 1.3M params, 8.0M tokens
**Prediction error:** +0.02 (predicted 2.41, got 2.43). Close.
**Learning:** Marginal improvement (2.429 -> 2.428, within MPS variance). WD=0.01 neither helps nor hurts significantly. Keep it since it's technically the best. The gains from optimizer tuning are becoming marginal -- we're in the diminishing returns zone.

---

## Experiment 6: beta2=0.95 (from 0.99)
**Config change:** BETAS=(0.9,0.99) -> (0.9,0.95)
**Hypothesis:** Phase 2 B found beta2=0.95 was good (2.536). Our current best has beta2=0.99 (2.428). beta2=0.95 provides even faster moment adaptation but more noise. With ~1900 steps, beta2=0.95 might overadapt (too much noise in the denominator).
**Predicted val_loss:** 2.45 (worse -- beta2=0.95 is too aggressive, 0.99 is the sweet spot)
**Reasoning:** Phase 2 B went from beta2=0.95 to 0.99 and improved. Our setup already benefits from beta2=0.99. Going to 0.95 will increase noise without benefit.
**Result:** val_loss=2.721557
**Prediction error:** +0.27 (predicted 2.45, got 2.72). Even worse than predicted!
**Learning:** beta2=0.95 is catastrophically worse (2.43 -> 2.72). The noisy second moments at beta2=0.95 destabilize training significantly. beta2=0.99 is clearly the sweet spot for ~1900 steps. Revert.

---

## Experiment 7: LR=1e-3 (from 2e-3)
**Config change:** LEARNING_RATE=2e-3 -> 1e-3
**Hypothesis:** With beta2=0.99, the effective LR is higher than with beta2=0.999 (less damped denominator). The current LR=2e-3 may be slightly too high. LR=1e-3 with beta2=0.99 might be a better match.
**Predicted val_loss:** 2.42 (slight improvement from better LR-beta2 match)
**Reasoning:** Phase 1 showed LR and beta2 interact. With beta2=0.99, the denominator adapts faster, so a lower nominal LR might be optimal.
**Result:** val_loss=2.559193, 1972 steps, 1.3M params
**Prediction error:** +0.14 (predicted 2.42, got 2.56). Much worse.
**Learning:** LR=1e-3 is too low (2.56 vs 2.43). Despite more steps being similar, lower LR means each step extracts less learning. LR=2e-3 remains optimal. The LR-beta2 interaction from Phase 1 doesn't work the same way here -- with 600s, you want aggressive learning from each step. Revert.

---

## Experiment 8: LR=3e-3 (from 2e-3)
**Config change:** LEARNING_RATE=2e-3 -> 3e-3
**Hypothesis:** Since LR=1e-3 was too low, let's test if LR=3e-3 works better. At 120s, LR=3e-3 was near-optimal with beta2=0.95. With beta2=0.99 and more steps, 3e-3 might be too high.
**Predicted val_loss:** 2.50 (worse -- 3e-3 with beta2=0.99 may overshoot, similar to Phase 1 finding)
**Reasoning:** Phase 1 showed LR=2e-3 beat LR=3e-3 with beta2=0.999. Same principle likely applies with beta2=0.99 -- higher beta2 amplifies effective LR, making 3e-3 too aggressive.
**Result:** val_loss=2.675338, 1000 steps (MPS throttled), 4.1M tokens
**Prediction error:** +0.18 (predicted 2.50, got 2.68). Much worse, also affected by MPS throttling.
**Learning:** LR=3e-3 is too high with beta2=0.99 (2.68 vs 2.43). Also, MPS thermal throttling reduced throughput to 1000 steps (vs normal ~1950). LR=2e-3 is clearly the sweet spot. Revert.

---

## Experiment 9: warmup=10 (from 20)
**Config change:** WARMUP_STEPS=20 -> 10
**Hypothesis:** With ~1950 steps and beta2=0.99, warmup=20 uses ~1% of training for warmup. Reducing to warmup=10 saves 10 warmup steps and starts full-LR training sooner. Beta2=0.99 adapts faster than 0.999, so less warmup might be needed.
**Predicted val_loss:** 2.43 (slightly worse or same -- warmup=20 was already well-tuned)
**Reasoning:** Phase 1 found warmup=20 optimal for LR=2e-3/beta2=0.999. With beta2=0.99, the optimizer warms up faster naturally, but warmup=10 might be too little for LR=2e-3.
**Result:** val_loss=2.486213, 1756 steps, 7.2M tokens
**Prediction error:** +0.06 (predicted 2.43, got 2.49). Slightly worse than predicted.
**Learning:** warmup=10 is worse (2.49 vs 2.43). Confirms warmup=20 is the sweet spot. Too little warmup destabilizes early training with LR=2e-3. Revert.

---

## Experiment 10: warmup=50 (from 20)
**Config change:** WARMUP_STEPS=20 -> 50
**Hypothesis:** Testing warmup=50. Phase 2 B found warmup=50 slightly worse than warmup=20 (2.459 vs 2.445). Our architecture differs (5L/128E vs 2L/192E), so the result might differ.
**Predicted val_loss:** 2.44 (slightly worse -- warmup=50 wastes 30 extra steps at low LR)
**Reasoning:** With ~1950 steps, warmup=50 means 2.6% of training at below-peak LR. These are wasted steps that could be learning at full speed. warmup=20 is likely better.
**Result:** val_loss=2.427042, 1983 steps, 8.1M tokens
**Prediction error:** -0.01 (predicted 2.44, got 2.43). Excellent prediction!
**Learning:** warmup=50 is essentially tied with warmup=20 (2.427 vs 2.428). Within MPS variance. Warmup in the 20-50 range doesn't matter much for this config. Keep warmup=50 since it's marginally better.

---

## Experiment 11: GA=2 (effective BS=16)
**Config change:** GRAD_ACCUM_STEPS=1 -> 2 (with BS=8, effective BS=16)
**Hypothesis:** Gradient accumulation doubles effective batch size without doubling memory per-step. However, it halves optimizer steps (~1950 -> ~975). At BS=16 effective, gradients should be very clean, but fewer steps may hurt.
**Predicted val_loss:** 2.50 (worse -- too few optimizer steps, similar to BS=16 result at 120s)
**Reasoning:** Phase 1 showed BS=16 was catastrophic. GA=2 gives the same effective BS=16 gradient quality but with the same per-step speed as BS=8 (2 micro-batches per step). However, steps will be halved, which was devastating in Phase 1.
**Result:** val_loss=2.429832, 991 steps, 1.3M params, 8.1M tokens
**Prediction error:** -0.07 (predicted 2.50, got 2.43). Better than predicted but still worse than best.
**Learning:** GA=2 barely hurts (2.430 vs 2.427). The gradient quality from effective BS=16 compensates well for having half the steps. But no improvement. GA=1 is still optimal. Revert.

---

## Experiment 12: Dropout=0.05 (from 0.0)
**Config change:** DROPOUT=0.0 -> 0.05
**Hypothesis:** With ~2000 steps and 8M tokens seen, the model might benefit from light regularization. WD=0.01 helped marginally; dropout=0.05 might add complementary regularization. Phase 1 found dropout harmful at 120s, but 600s is different.
**Predicted val_loss:** 2.44 (slightly worse -- dropout still likely harmful, model not overfitting much)
**Reasoning:** Phase 1 showed dropout is harmful even at 6L. At 5L with 1.3M params, the model is unlikely to overfit in 2000 steps on TinyStories. Dropout wastes capacity during forward passes.
**Result:** val_loss=2.473626, 1894 steps, 1.3M params
**Prediction error:** +0.03 (predicted 2.44, got 2.47). Worse as expected.
**Learning:** Dropout STILL harmful at 600s (2.47 vs 2.43). Phase 1 finding confirmed again. The model doesn't overfit in ~2000 steps. Revert.

---

## Experiment 13: N_EMBD=192, N_HEAD=4 (moderate width increase)
**Config change:** N_EMBD=128->192, N_HEAD=2->4 (head_dim stays 48, slightly less than 64)
**Hypothesis:** At 600s, a wider model (3.0M vs 1.3M) might now have enough training time. Phase 1 found 192E too slow at 120s, but with 5x more time the capacity-throughput tradeoff may shift.
**Predicted val_loss:** 2.48 (worse -- 192E was worse than 128E even at 120s with optimal training)
**Reasoning:** The model goes from 1.3M to ~3.0M params, halving throughput. With BS=8, steps drop from ~1950 to ~1300. The extra capacity may not compensate for fewer optimization steps and lower token count.
**Result:** val_loss=2.379145, 1525 steps, 3.0M params, 6.2M tokens
**Prediction error:** -0.10 (predicted 2.48, got 2.38). MAJOR SURPRISE -- much better than predicted!
**Learning:** HUGE BREAKTHROUGH (2.427 -> 2.379 = -0.048). The wider model (3.0M vs 1.3M) works BETTER at 600s despite fewer steps (1525 vs 1950). This REVERSES Phase 1's finding that 128E is optimal! At 600s, the capacity-throughput tradeoff shifts: 3.0M params with 1525 steps > 1.3M params with 1950 steps. The key insight: exp 2 failed (6.3M, 2.90) because it was too large, but 3.0M is the right size at 600s. Need to explore further: what about 4L/4H/256E or 6L/4H/192E?

---

## Experiment 14: 6L/4H/192E (add depth to the wider model)
**Config change:** N_LAYER=5->6 (with 192E/4H)
**Hypothesis:** Since 5L/192E was a huge improvement, adding depth might help further. 6L/4H/192E = ~3.5M params. At 120s, deeper models were better. The extra layer adds compositional power.
**Predicted val_loss:** 2.37 (very slight improvement -- depth helps but throughput cost is small)
**Reasoning:** Going from 5L to 6L adds ~17% params. Steps drop from ~1525 to ~1400. The depth-throughput tradeoff at this model size should be similar to Phase 1: minimal impact.
**Result:** val_loss=2.381481, 1401 steps, 3.5M params, 5.7M tokens
**Prediction error:** +0.01 (predicted 2.37, got 2.38). Close.
**Learning:** 6L is essentially tied with 5L at 192E (2.381 vs 2.379). Same finding as Phase 1: 5L is the sweet spot for depth. Keep 5L.

---

## Experiment 15: 5L/4H/256E (even wider)
**Config change:** N_EMBD=192->256, N_LAYER=6->5 (revert depth)
**Hypothesis:** If 192E improved from 2.43 to 2.38 vs 128E, maybe 256E is even better? 5L/4H/256E = ~5.2M params (vs 3.0M). Exp 2 tried 6L/4H/256E (6.3M) and got 2.90 -- but that was before optimizer tuning. With the optimized optimizer settings, 5.2M might work.
**Predicted val_loss:** 2.40 (slight improvement from more capacity, but model is getting large)
**Reasoning:** The throughput penalty is significant (5.2M = ~1000 steps vs 1525 at 192E). We're approaching the size where the Chinchilla scaling law says we need more data. But the optimized LR/beta2/warmup settings may help.
**Result:** val_loss=2.803223, 1282 steps, 5.2M params, 5.3M tokens
**Prediction error:** +0.40 (predicted 2.40, got 2.80). Much worse!
**Learning:** 256E is WAY too large (2.80 vs 2.38). 5.2M params at 1282 steps is still undertrained. The optimal model size at 600s is ~3.0M (192E). The capacity-throughput curve has a clear peak at 192E for this budget. Revert to 192E.

Updated theory: At 600s, optimal model size is ~3.0M (5L/4H/192E). 128E (1.3M) is too small (2.43), 256E (5.2M) is too large (2.80). The sweet spot maximizes (capacity * step_count).

---

## Experiment 16: 4L/4H/192E (less depth with 192E)
**Config change:** N_LAYER=5->4 (with 192E/4H)
**Hypothesis:** Maybe 4L is better than 5L at 192E? 4L/4H/192E = ~2.4M params. Fewer layers = more steps, but less compositional depth.
**Predicted val_loss:** 2.39 (slightly worse -- 5L was optimal at 128E, should be similar at 192E)
**Reasoning:** Phase 1 showed 4L was worse than 5L. The depth advantage should persist at 192E.
**Result:** val_loss=2.850086, 633 steps, 2.4M params, 2.6M tokens
**Prediction error:** +0.46 (predicted 2.39, got 2.85). MASSIVELY worse than predicted!
**Learning:** 4L/4H/192E is catastrophically worse than 5L/4H/192E (2.85 vs 2.38). Only 633 steps in 600s -- severe MPS throttling occurred (normally ~1700 steps at this size). The poor step count is the primary culprit. Even accounting for throttling, 4L being worse than 5L confirms that depth is critical for the 192E model. Revert to 5L.

---

## Experiment 17: N_HEAD=2 with 192E (head_dim=96)
**Config change:** N_HEAD=4->2 (keeping 5L/192E, all else same as best)
**Hypothesis:** With 192E and 2 heads, each head has dim=96 (vs 48 with 4 heads). Larger heads can attend to more complex patterns per head. Phase 1 found 2 heads optimal at 128E (head_dim=64). At 192E, 2 heads gives head_dim=96 which may be even better for capturing long-range dependencies.
**Predicted val_loss:** 2.39 (slightly worse -- 4 heads was part of the 192E breakthrough, 2 heads may lose attention diversity)
**Reasoning:** The 192E improvement came with N_HEAD=4. Reducing to 2 heads trades attention diversity for per-head capacity. At 128E, 2 heads (dim=64) was optimal, but 192E may need more attention diversity.
**Result:** val_loss=2.441453, 1296 steps, 3.0M params, 5.3M tokens
**Prediction error:** +0.05 (predicted 2.39, got 2.44). Worse than predicted.
**Learning:** N_HEAD=2 at 192E is notably worse (2.44 vs 2.38). 4 heads is better than 2 heads at 192E. This contrasts with Phase 1 where 2 heads was optimal at 128E. The key is head_dim: at 128E/2H, head_dim=64 was ideal. At 192E/4H, head_dim=48 works better than 192E/2H head_dim=96. Wider models benefit from more attention diversity. Revert to N_HEAD=4.

---

## Experiment 18: N_HEAD=6 with 192E (head_dim=32)
**Config change:** N_HEAD=4->6 (keeping 5L/192E, all else same as best)
**Hypothesis:** Since 4 heads beat 2 heads at 192E, maybe 6 heads (head_dim=32) is even better. More heads = more attention diversity. However, head_dim=32 is quite small and may limit per-head pattern complexity.
**Predicted val_loss:** 2.39 (slightly worse -- head_dim=32 may be too small, 48 is likely the sweet spot)
**Reasoning:** There's likely a sweet spot for head_dim. Phase 1: 64 optimal at 128E. At 192E: 48 (4H) > 96 (2H). Going to 32 (6H) may overshoot. Diminishing returns from more heads when each head can't capture enough pattern complexity.
**Result:** val_loss=2.629415, 834 steps, 3.0M params, 3.4M tokens
**Prediction error:** +0.24 (predicted 2.39, got 2.63). Much worse, heavily affected by MPS throttling.
**Learning:** 6 heads (2.629) is much worse than 4 heads (2.379), but the result is confounded by severe MPS throttling -- only 834 steps vs ~1525 expected. The loss trajectory was also worse throughout training (loss at step 800 was 2.82 vs typical ~2.7). Even accounting for throttling, 6H/head_dim=32 appears inferior: smaller heads limit per-head pattern complexity. Head count ranking at 192E: 4H (2.379) >> 2H (2.441) >> 6H (2.629, throttled). The 4H result (head_dim=48) is clearly the sweet spot. Revert.

---

## Experiment 19: warmup=20 with 192E (from 50)
**Config change:** WARMUP_STEPS=50->20 (reverting to best config otherwise: 5L/4H/192E)
**Hypothesis:** warmup=50 was adopted from exp 10 which tested it at 128E (1.3M params, ~1950 steps). At 192E (3.0M params, ~1525 steps), the model has fewer steps so warmup=50 wastes a larger fraction (3.3% vs 2.6%). warmup=20 saves 30 steps of sub-peak LR training, which might matter more at 192E's lower step count.
**Predicted val_loss:** 2.38 (very similar -- warmup=20 and 50 were tied at 128E, should be similar at 192E)
**Reasoning:** At 128E, warmup=50 (2.427) and warmup=20 (2.428) were within noise. At 192E with fewer steps, the extra 30 warmup steps represent a slightly larger cost, so warmup=20 might have a tiny edge. But the effect is likely within MPS variance.
**Result:** val_loss=2.750785, 738 steps, 3.0M params, 3.0M tokens
**Prediction error:** +0.37 (predicted 2.38, got 2.75). Much worse, heavily affected by MPS throttling.
**Learning:** Only 738 steps in 600s -- severe MPS throttling again. This result is unreliable due to the throttling confound. Cannot draw valid conclusions about warmup=20 vs 50 from this data. Revert to warmup=50 (best config).

---

## Final Summary (Experiments 16-19)
All 4 experiments worsened or were confounded by MPS throttling. The best config remains exp 13: 5L/4H/192E, LR=2e-3, WD=0.01, BS=8, beta2=0.99, warmup=50, DO=0, GA=1, val_loss=2.379145.

Key findings from exps 16-19:
- **Exp 16 (4L):** Fewer layers is worse, even accounting for throttling. 5L optimal at 192E.
- **Exp 17 (2H):** 2 heads worse than 4 at 192E. head_dim=48 > head_dim=96.
- **Exp 18 (6H):** 6 heads worse than 4 (throttled, but trajectory also worse). head_dim=32 < head_dim=48.
- **Exp 19 (warmup=20):** Inconclusive due to throttling. warmup=50 retained as default.

MPS thermal throttling was a major confound in exps 16, 18, 19. Step counts were 633, 834, 738 respectively vs expected ~1500. This makes direct comparisons unreliable for those experiments.

