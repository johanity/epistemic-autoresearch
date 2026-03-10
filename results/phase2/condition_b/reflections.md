# Phase 2 Condition B — Reflections

Starting from Phase 1 best config: 2L/4H/192E, dropout=0, WD=0, LR=1e-3, BS=2, warmup=0, betas=(0.9,0.999).
Phase 1 best val_loss was 3.033880 with 120s budget. Now we have 600s (5x more time).

## Exp 0 — Baseline (Phase 1 best)
val_loss=2.784552, 4292 steps. More time helps the Phase 1 config but 4292 steps with BS=2 is a LOT of noisy updates.

## Exp 1-3 — Architecture scaling
All worse. 4L/256E (2.976), 3L/192E (2.794), 2L/256E (2.891). Even with 600s, throughput still matters. The 2L/192E architecture remains optimal.

## Exp 4-6 — LR and warmup tweaks with BS=2
LR=6e-4 worse, LR=2e-3 much worse, warmup=10 worse. The optimizer settings from Phase 1 are locally optimal for BS=2.

## Exp 7-9 — Batch size sweep (CRITICAL FINDING)
BS=4 (2.647) and BS=8 (2.568) both beat BS=2 (2.785). BS=16 (2.690) is worse. With 600s of training, the model gets enough steps even at BS=8 (1199 steps), and the cleaner gradients from larger batches help convergence significantly. This is the biggest difference from Phase 1 where BS=2 was optimal for 120s.

## Exp 10-11 — Architecture with BS=8
3L/192E (2.580) and 2L/256E (2.573) both very close but slightly worse than 2L/192E. The throughput advantage of 2L/192E persists even at BS=8.

## Exp 12-14 — LR sweep with BS=8
LR=3e-3 (3.021) and LR=2e-3 (2.768) too high. LR=6e-4 (2.595) slightly worse. LR=1e-3 remains optimal.

## Exp 15-16 — Betas sweep with BS=8
(0.9,0.95) gave 2.536, (0.9,0.99) gave 2.459. Both better than (0.9,0.999). With BS=8, gradients are cleaner and you need less second-moment smoothing. In Phase 1 with BS=2, noisy gradients needed (0.9,0.999). Now (0.9,0.99) is the sweet spot.

## Exp 17-18 — Warmup sweep
warmup=20 (2.445) improved over warmup=0. warmup=50 (2.459) slightly worse. With ~1800 steps, 20-step warmup (about 1% of training) is helpful. In Phase 1 with ~1200 steps, warmup wasn't worth the cost.

## Exp 19 — Weight decay
WD=0.01 (2.441) slightly better than WD=0.0. With longer training, a tiny bit of regularization helps prevent overfitting. In Phase 1, training was too short for WD to matter.

## Summary of Phase 2 vs Phase 1 differences
The 600s budget fundamentally changes the optimizer sweet spot while the architecture stays the same:
1. Architecture: 2L/4H/192E remains optimal (throughput still matters)
2. Batch size: BS=8 >> BS=2 (the biggest shift — cleaner gradients matter more when you have enough steps)
3. Betas: (0.9,0.99) beats (0.9,0.999) (less smoothing needed with cleaner gradients)
4. Warmup: 20 steps now helps (worth the cost with 1800+ total steps)
5. Weight decay: 0.01 slightly helps (longer training = slight overfitting risk)
6. LR: 1e-3 still optimal

Final best: val_loss=2.441034 (down from 2.784552 baseline, 12.3% improvement)
Config: 2L/4H/192E, LR=1e-3, BS=8, WD=0.01, DO=0.0, betas=(0.9,0.99), warmup=20, grad_accum=1
