# Experiment Contract — Transfer Experiment

**Date:** March 8, 2026
**Branch:** `autoresearch/transfer-mar8`
**Author:** Johan David Bonilla
**Title:** "Search Is Not Learning: Prediction Error as the Source of Transfer in Autonomous Research"

This document is written BEFORE any experimental code runs. It defines the hypotheses, conditions, metrics, fairness constraints, and failure modes. It is the contract against which the experiment will be judged.

---

## 1. Research Question

**Does preserving prediction errors and explicit theory updates — rather than just outcomes — enable autonomous research loops to transfer understanding across environment shifts?**

Subsidiary question: **Is the mechanism prediction error specifically, or does any form of post-hoc reflection suffice?**

---

## 2. Hypotheses

### Primary Hypothesis
Autonomous research loops that preserve prediction errors and explicit theory updates will:
- Plateau later
- Repeat fewer failed directions
- Accumulate more reusable understanding
- **Transfer that understanding to a new environment faster**

...than loops that preserve outcomes alone.

### Secondary Hypothesis
Prediction-driven theory updates outperform post-hoc reflection alone, indicating that violated expectation — not just extra deliberation — is the key source of compounding understanding.

### Null Hypothesis
All three conditions perform equivalently after the environment shift, because the domain knowledge from Environment A does not meaningfully transfer to Environment B.

---

## 3. Three Conditions

### Condition A — Outcome-Only (Karpathy Baseline)
- **Loop:** mutate hyperparameters → train → score val_loss → keep if improved, discard if not
- **Memory:** Best config (train.py state) only
- **No LLM.** Pure random search + keep/discard. This is the autoresearch paradigm.
- **What transfers to Phase 2:** Best train.py config. Nothing else.
- **Implementation:** Based on existing `bruteforce_runner.py`

### Condition B — Reflection-Only (Thinking Time Control)
- **Loop:** LLM proposes change → train → score → LLM reflects on result → keep/discard
- **Memory:** A running reflection log (free-form notes after each experiment)
- **No prediction.** The LLM thinks about results after the fact but never commits to a prediction beforehand.
- **What transfers to Phase 2:** Best config + final reflection notes
- **Purpose:** Controls for "prediction is just extra thinking time." If C beats B, the mechanism is prediction error structure, not just additional reasoning.

### Condition C — Epistemic (Prediction-Error)
- **Loop:** LLM forms hypothesis → writes specific numerical prediction → train → measures prediction error → updates persistent theory → keep/discard
- **Memory:** `theory.md` (evolving principles), `lab_notebook.md` (hypothesis + prediction + result + learning per experiment), `results.tsv` (predicted vs actual)
- **What transfers to Phase 2:** Explicit theory state (theory.md). No raw experiment history, no code state beyond best config.
- **Purpose:** Tests whether prediction error as a first-class learning signal enables compounding understanding that transfers.

---

## 4. Two Phases — The Causal Inversion Shift

### Phase 1: Environment A — Short Training Budget
- **TIME_BUDGET = 120 seconds** (current setup)
- **Expected optimal region:** Small models (1.8M params), high throughput, LR=2e-3
- **Each condition runs N experiments** (N=126, matching Karpathy's March 8 session)
- **Key dynamic:** Throughput dominates capacity. Smaller models win because they process more tokens in the fixed time window.

### Phase 2: Environment B — Long Training Budget
- **TIME_BUDGET = 600 seconds** (5 minutes)
- **Expected optimal region:** Larger models become viable. The throughput-capacity tradeoff shifts.
- **Each condition runs N experiments** (same N as Phase 1)
- **Key dynamic:** The principles learned in Phase 1 ("smaller is better") are now WRONG or at least need revision. This is the causal inversion.

### Why This Shift Is Powerful
- Condition A learned "this config works" — it has no theory about WHY. It will keep trying small models.
- Condition B reflected on "small models won" — but without prediction, the surprise of large models working won't be salient.
- Condition C predicted "throughput > capacity BECAUSE of the 2-min budget" — when the budget changes, the theory directly implies the prediction should change. The prediction error on the first Phase 2 experiment is the learning signal.

---

## 5. Seven Metrics

| # | Metric | What It Measures | Collected For |
|---|--------|-----------------|---------------|
| 1 | **Best val_loss** | Raw performance | A, B, C |
| 2 | **Time to plateau** | Experiment number where improvement stalls (no improvement for 3+ consecutive experiments) | A, B, C |
| 3 | **Repeated bad direction rate** | Fraction of experiments that try a category already failed 2+ times | A, B, C |
| 4 | **Prediction calibration (MAE)** | Mean absolute error of predicted vs actual val_loss | C only (B and A don't predict) |
| 5 | **Transferable principle count** | Number of principles from Phase 1 that correctly predict Phase 2 behavior | C primarily (B from reflection notes) |
| 6 | **Adaptation speed** | Number of Phase 2 experiments to match or beat Phase 1 best | A, B, C |
| 7 | **Search diversity** | Number of unique hyperparameter categories changed across all experiments | A, B, C |

### Derived Metrics
- **Late-stage marginal gain:** Improvement in last 5 experiments vs first 5 (measures compounding vs plateau)
- **Phase 2 first-experiment quality:** How good is the first Phase 2 experiment? (Tests whether transferred knowledge helps immediately)

---

## 6. Fairness Constraints

All three conditions MUST share:

| Constraint | Value | Rationale |
|-----------|-------|-----------|
| Base model architecture | GPT (from train.py) | Same model class |
| Dataset | TinyStories (from prepare.py) | Same data |
| Training time per experiment | TIME_BUDGET seconds | Same compute |
| Number of experiments per phase | N (=126) | Same search budget |
| Mutation space | Same hyperparameter ranges | Same search space |
| Evaluation metric | val_loss from evaluate_loss() | Same target |
| Reasoning capability | Claude Opus 4.6 (B and C) or none (A) | See note below |

### Note on Reasoning Capability
Condition A uses NO LLM — it is pure random search (`condition_a.py`). This faithfully represents the autoresearch paradigm.

Conditions B and C are both run by **Claude Code itself** (Claude Opus 4.6), following different `program.md` instructions. Same model, same context window, same tools. The only difference is the loop structure defined in the instructions:
- B follows `prompts/program_b.md` (reflect after, no prediction)
- C follows `prompts/program_c.md` (predict before, measure error after, update theory)

This is identical to how the first paper was produced — Claude Code was the agent.

### What Each Condition Knows at Phase 2 Start
| Condition | Carries Forward |
|-----------|----------------|
| A | Best train.py config only |
| B | Best train.py config + final reflection notes (text blob) |
| C | Best train.py config + theory.md (structured principles + refuted hypotheses) |

No condition carries raw experiment logs, intermediate configs, or code diffs into Phase 2.

---

## 7. Failure Modes and Safeguards

### FM1: Claude Makes One Loop Smarter
**Risk:** The epistemic prompt inadvertently gives C more reasoning power (e.g., more tokens, better instructions).
**Safeguard:** B and C prompts are length-matched. Both get the same number of LLM calls per experiment. Both see the same information (current config, last result). Only the STRUCTURE of the prompt differs.

### FM2: Unequal Compute
**Risk:** One condition runs more experiments or gets more training time.
**Safeguard:** The harness enforces exactly N experiments per condition per phase. TIME_BUDGET is set in prepare.py (read-only). The harness times each run and logs actual training seconds.

### FM3: Information Leaks Between Phases
**Risk:** Condition A retains hidden state (e.g., the runner remembers which mutations failed).
**Safeguard:** Phase 2 starts fresh for each condition. Only the explicitly defined transfer state is carried over (see §6). The harness resets all state between phases.

### FM4: Optimizing for the Metric
**Risk:** Results are cherry-picked or a single metric is gamed.
**Safeguard:** All 7 metrics are computed automatically by the harness. No manual selection. All raw data is logged.

### FM5: Prompt Leakage
**Risk:** Condition C's prompt implicitly teaches the LLM about the domain (e.g., "throughput matters more than capacity").
**Safeguard:** Neither B nor C prompts contain domain knowledge. They only describe the PROCESS (how to reflect vs how to predict). Domain knowledge must be discovered through experimentation.

---

## 8. Karpathy Sanity Test

> "Would Andrej Karpathy reasonably say that Condition A faithfully represents the autoresearch paradigm?"

Condition A is a direct re-implementation of the autoresearch loop:
- Randomly mutate train.py hyperparameters
- Run training for fixed time budget
- Check val_loss
- Keep if improved, discard if not
- Log to results.tsv
- Repeat

The mutation space matches what an LLM agent would explore. The keep/discard logic is identical. The only simplification is using random mutation instead of LLM-generated mutations — this is CONSERVATIVE: it removes the possibility that the LLM in conditions B/C provides advantage through better mutation proposals (rather than through loop structure).

---

## 9. Pre-Registration of Expected Results

Before running, I predict:

### Phase 1 (Short Budget)
- All three conditions will find the throughput > capacity insight (small models win)
- C will find it faster (fewer experiments) because prediction forces engagement
- A will find it by luck (random search eventually hits small models)
- B will find it but may not articulate WHY clearly
- Best val_loss: similar across conditions (the landscape isn't that hard)

### Phase 2 (Long Budget — The Real Test)
- **A will thrash.** It will start with the small-model config and try random mutations. Without understanding WHY small models won, it has no reason to try large models.
- **B will adapt slowly.** The reflection notes say "small models worked" but without prediction error, the first large-model success won't trigger structured revision.
- **C will adapt fastest.** Theory.md says "throughput > capacity BECAUSE of 2-min budget." When the budget changes to 5 min, the theory directly implies: re-examine this principle. The first prediction error will trigger theory revision.

### Killer Result I'm Looking For
C matches or beats A/B in Phase 1, then **dramatically outperforms** in Phase 2 adaptation speed. The gap should be largest in:
- Adaptation speed (metric #6)
- Repeated bad direction rate (metric #3) — A keeps trying small models
- Phase 2 first-experiment quality — C's theory predicts what to try

---

## 10. Stopping Criteria

The experiment is COMPLETE when:
1. All three conditions have run N experiments in Phase 1
2. All three conditions have run N experiments in Phase 2
3. All 7 metrics are computed
4. Raw logs are saved

The experiment is INVALID if:
1. Any condition crashes > 30% of experiments
2. TIME_BUDGET differs between conditions by > 5%
3. Any condition runs a different number of experiments

---

## 11. Repository Structure

```
karp/
├── EXPERIMENT_CONTRACT.md     ← this file
├── CLAUDE_PROGRESS.md         ← live progress log
├── prepare.py                 ← FIXED — do not modify
├── train.py                   ← modified by conditions during experiments
├── harness.py                 ← orchestrates all conditions + phases
├── condition_a.py             ← brute-force runner (no LLM)
├── prompts/
│   ├── program_b.md           ← B's instructions for Claude Code (reflection-only)
│   ├── program_c.md           ← C's instructions for Claude Code (epistemic)
│   └── propose_change.md      ← reference: hyperparameter search space
├── results/
│   ├── phase1/
│   │   ├── condition_a/       ← results.tsv, train.py snapshots
│   │   ├── condition_b/       ← results.tsv, reflections.md
│   │   └── condition_c/       ← results.tsv, theory.md, lab_notebook.md
│   └── phase2/
│       ├── condition_a/
│       ├── condition_b/
│       └── condition_c/
├── analysis/
│   ├── metrics.py             ← compute all 7 metrics from raw data
│   └── plots.py               ← generate comparison visualizations
└── paper/                     ← paper drafts
```

---

## 12. Implementation Plan

### Step 1: Build the Harness
- `harness.py`: CLI that runs all conditions sequentially
- Manages Phase 1 → Phase 2 transition
- Enforces fairness constraints (same N, same TIME_BUDGET)
- Saves all state to results/ directories

### Step 2: Implement Condition A
- Adapt existing `bruteforce_runner.py`
- Same mutation space, same keep/discard logic
- Output: results.tsv per phase

### Step 3: Implement Condition B
- LLM proposes changes (sees current config + last result)
- LLM reflects after each experiment (free-form)
- NO prediction, NO theory file
- Output: results.tsv + reflections.md per phase

### Step 4: Implement Condition C
- LLM forms hypothesis, writes prediction, proposes change
- After experiment: measures prediction error, updates theory
- Maintains theory.md + lab_notebook.md
- Output: results.tsv + theory.md + lab_notebook.md per phase

### Step 5: Run Phase 1
- All three conditions, N=15 experiments each, TIME_BUDGET=120s

### Step 6: Transfer State
- Extract transfer state per condition (see §6)
- Set TIME_BUDGET=600s
- Reset experiment counters

### Step 7: Run Phase 2
- All three conditions, N=15 experiments each, TIME_BUDGET=600s

### Step 8: Compute Metrics + Analyze
- Run metrics.py on all results
- Generate plots
- Write findings

---

## 13. Final Safeguard Prompt

Before running, answer:

> "List the three ways this experiment could accidentally favor the epistemic loop unfairly. Explain how the design prevents each."

1. **More reasoning tokens:** B and C prompts are length-matched. Both get the same number of LLM calls. ✓
2. **Domain knowledge in the prompt:** Neither prompt contains domain-specific knowledge (e.g., "small models are better"). Only process instructions. ✓
3. **Unequal search guidance:** Both B and C use the LLM to PROPOSE changes (same proposal prompt). The only difference is what happens AFTER the result. ✓

---

## 14. Pre-Registration of Condition C Hypotheses (Added Before C Starts)

**Timestamp:** 2026-03-09, Phase 1 B still running (33/64), C not yet started.

These hypotheses are registered NOW, before Condition C runs a single experiment, so they cannot be fitted to C's results after the fact.

### H1 — Phase 1 Final Performance
C will match or modestly beat B on final Phase 1 best val_loss, but the margin will be small (<0.1). Phase 1 is a simple, throughput-dominated regime where reflection may already be sufficient. The prediction-error mechanism should show its advantage primarily in Phase 2.

### H2 — Search Efficiency (Repeated Bad Direction Rate)
C will have a lower repeated-bad-direction rate than B. Explicit prediction forces the agent to commit to an expectation before each run, making surprises more legible and reducing repeated mistakes in the same category.

### H3 — Prediction Calibration
C's prediction MAE (mean absolute error between predicted and actual val_loss) will decrease over time across Phase 1. If this holds, the agent is building an increasingly accurate model of the search landscape — not just searching, but learning the terrain.

### H4 — Theory Quality (Conditional Rules)
C will produce more conditional principles than B's reflections. Not "small is better" but "under 120s, throughput dominates capacity BECAUSE X." This conditionality is what enables transfer — it encodes the boundary conditions under which a rule holds.

### H5 — Phase 2 Adaptation Speed (The Money Hypothesis)
C will beat both A and B on adaptation speed (metric #6) when the time budget shifts from 120s to 600s. C's theory should encode regime-dependent rules that directly signal what to revise. B's reflections may contain the right insight but in unstructured form. A has no mechanism to adapt at all.

### H6 — Phase 2 First-Experiment Quality
C's first Phase 2 experiment will be closer to the Phase 2 optimum than A's or B's first experiment. C's theory should predict that the throughput-capacity tradeoff shifts under a longer budget, leading to a better initial config choice.

---

## 15. Known Limitations and Threats to Validity

Registered before results are complete, to preempt rather than respond to criticism.

### L1 — Semantic Priors
Conditions B and C use Claude Opus 4.6, which has pretrained knowledge about what "dropout," "warmup," "learning rate," etc. typically do. Condition A has no such knowledge. This is intentional — the experiment tests loop structure, not vocabulary. But the claim must stay narrow: the advantage is in the process of converting experience into reusable rules, not in having prior domain knowledge. The LLM's semantic priors are held constant between B and C; only the loop structure differs.

### L2 — Baseline Variance
A's baseline val_loss was ~4.15; B's was ~4.80 from the same nominal config. This reflects stochastic training variance. B started from a worse position and still surpassed A, which if anything strengthens B's result. But single-run comparisons carry noise. Confirmatory reruns of best configs are needed before making strong final claims.

### L3 — Single Task, Single Regime
TinyStories with a 120s/600s training budget is a narrow setting. Results should be presented as a controlled demonstration of the mechanism, not universal proof that prediction error beats search in all domains.

### L4 — Phase Ordering
B runs before C due to GPU contention. Any environmental drift (temperature, background processes, library updates) between runs is a potential confound. Mitigated by: same hardware, same code, same seeds where applicable, same TIME_BUDGET enforcement.

### L5 — A Is a Weak Baseline
Condition A is specifically a best-config-only random mutation loop. Stronger non-LLM baselines (Bayesian optimization, Hyperband, ASHA, evolutionary search with population memory) would be tougher competitors. A faithfully represents the Karpathy autoresearch paradigm, but "A loses" does not mean "all non-LLM search loses."

---

## 16. Exact Metric Definitions

Locked before final analysis. No retroactive changes.

| Metric | Exact Definition |
|--------|-----------------|
| **Best val_loss** | Minimum val_loss across all experiments in the condition's results.tsv |
| **Keep rate** | Count of rows where status="keep" / total experiment count (excluding header) |
| **Longest discard streak** | Maximum number of consecutive rows with status="discard" |
| **Time to plateau** | First experiment number E such that experiments E, E+1, E+2 all have status="discard" AND none of them beat the current best. If no such E exists, plateau = N (never plateaued). |
| **Repeated bad direction rate** | Count of experiments that (a) change a hyperparameter category AND (b) that category has already produced 2+ discards in prior experiments AND (c) this experiment is also a discard, divided by total experiments. |
| **Prediction calibration (MAE)** | Mean of |predicted_val_loss - actual_val_loss| across all C experiments. Computed in rolling windows of 5 to show trajectory. |
| **Adaptation speed** | In Phase 2: the experiment number at which the condition first matches or beats its own Phase 1 best val_loss. |
| **Search diversity** | Count of unique hyperparameter categories modified across all experiments (max 10). |
