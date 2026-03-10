"""
Condition C — Epistemic (Prediction-Error)

Loop: LLM forms hypothesis → writes prediction → train → measures prediction
error → updates theory → keep/discard
Memory: theory.md (evolving principles), lab_notebook.md (experiment journal),
results.tsv (predicted vs actual)

Automated via LLM API calls (litellm).
"""

import os
import re
import json
import shutil

from condition_a import (
    HYPERPARAM_RANGES, read_current_params, write_params, run_training, TRAIN_PY,
)
from litellm import completion

MODEL = "claude-sonnet-4-20250514"


def _format_params(params):
    lines = []
    for k, v in sorted(params.items()):
        lines.append(f"  {k} = {v}")
    return "\n".join(lines)


def _format_search_space():
    lines = []
    for k, v in HYPERPARAM_RANGES.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _call_llm(messages):
    resp = completion(model=MODEL, messages=messages, max_tokens=3000, temperature=0.7)
    return resp.choices[0].message.content


def _parse_json_from_response(text):
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Try to find the largest JSON object
    matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL))
    if matches:
        for m in reversed(matches):  # try largest first
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON found in response: {text[:200]}")


def _validate_changes(changes, current_params):
    valid = {}
    for key, val in changes.items():
        if key not in HYPERPARAM_RANGES:
            continue
        if key == "BETAS" and isinstance(val, list):
            val = tuple(val)
        if val in HYPERPARAM_RANGES[key] or (key == "BETAS" and tuple(val) in HYPERPARAM_RANGES[key]):
            valid[key] = val if key != "BETAS" else tuple(val)

    if not valid:
        return valid

    n_embd = valid.get("N_EMBD", current_params.get("N_EMBD", 192))
    n_head = valid.get("N_HEAD", current_params.get("N_HEAD", 3))
    if n_embd % n_head != 0:
        valid_heads = [h for h in HYPERPARAM_RANGES["N_HEAD"] if n_embd % h == 0]
        if valid_heads:
            valid["N_HEAD"] = valid_heads[0]
        else:
            valid.pop("N_EMBD", None)
            valid.pop("N_HEAD", None)

    return valid


def run_condition_c(num_experiments, output_dir, initial_train_py=None,
                    initial_theory=None):
    """
    Run Condition C: epistemic/prediction-error loop.

    Args:
        num_experiments: Number of experiments to run (including baseline)
        output_dir: Directory to save results
        initial_train_py: Optional path to initial train.py state
        initial_theory: Optional path to theory.md to carry forward
    Returns:
        dict with best_loss, best_config_path, results_path
    """
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.tsv")
    theory_path = os.path.join(output_dir, "theory.md")
    notebook_path = os.path.join(output_dir, "lab_notebook.md")

    # Save original train.py
    with open(TRAIN_PY) as f:
        original_content = f.read()

    # If initial state provided, use it
    if initial_train_py:
        shutil.copy2(initial_train_py, TRAIN_PY)

    # Initialize files
    with open(results_path, 'w') as f:
        f.write("exp\tval_loss\tpredicted\tstatus\tdescription\n")

    if initial_theory and os.path.exists(initial_theory):
        shutil.copy2(initial_theory, theory_path)
    else:
        with open(theory_path, 'w') as f:
            f.write("# Theory Document — Condition C\n\n")
            f.write("## Confirmed Principles\n(none yet)\n\n")
            f.write("## Refuted Hypotheses\n(none yet)\n\n")
            f.write("## Open Questions\n- What is the baseline performance?\n")
            f.write("- What is the optimal model size for this time budget?\n\n")

    with open(notebook_path, 'w') as f:
        f.write("# Lab Notebook — Condition C\n\n")

    print(f"\n{'='*60}")
    print(f"CONDITION C — EPISTEMIC / PREDICTION-ERROR (LLM-Automated)")
    print(f"Running {num_experiments} experiments")
    print(f"{'='*60}")

    # Run baseline
    print("\n[C-0] Running baseline...")
    baseline = run_training()
    if baseline is None:
        print("Baseline CRASHED. Aborting.")
        with open(TRAIN_PY, 'w') as f:
            f.write(original_content)
        return None

    best_loss = baseline["val_loss"]
    best_content = open(TRAIN_PY).read()
    current_params = read_current_params(TRAIN_PY)
    steps = int(baseline.get("num_steps", 0))
    params_m = baseline.get("num_params_M", 0)

    desc = f"Baseline: {', '.join(f'{k}={v}' for k, v in sorted(current_params.items()))} ({params_m:.1f}M params, {steps} steps)"
    print(f"  Baseline: val_loss={best_loss:.6f}, steps={steps}")

    with open(results_path, 'a') as f:
        f.write(f"0\t{best_loss:.6f}\tN/A\tbaseline\t{desc}\n")

    with open(notebook_path, 'a') as f:
        f.write(f"## Experiment 0: Baseline\n")
        f.write(f"**Config:** {_format_params(current_params)}\n")
        f.write(f"**Result:** val_loss = {best_loss:.6f}, {steps} steps, {params_m:.1f}M params\n")
        f.write(f"**Learning:** Baseline established.\n\n---\n\n")

    with open(os.path.join(output_dir, "train_exp0.py"), 'w') as f:
        f.write(best_content)

    all_results = [{"exp": 0, "val_loss": best_loss, "status": "baseline"}]

    # Experiment loop
    for exp_num in range(1, num_experiments):
        print(f"\n[C-{exp_num}/{num_experiments - 1}]")

        # Read current state
        with open(results_path) as f:
            results_content = f.read()
        with open(theory_path) as f:
            theory_content = f.read()
        with open(notebook_path) as f:
            notebook_content = f.read()
        current_params = read_current_params(TRAIN_PY)

        # Phase 1: THINK — hypothesis + prediction + proposal
        think_prompt = f"""You are a scientist optimizing a GPT language model. You hypothesize, predict, test, measure error, and update theory.

Current best val_loss: {best_loss:.6f}
Current hyperparameters:
{_format_params(current_params)}

Search space (choose from these values only):
{_format_search_space()}
Constraint: N_EMBD must be divisible by N_HEAD.

Your theory:
{theory_content}

Recent lab notebook (last 2000 chars):
{notebook_content[-2000:]}

Results:
{results_content}

TASK: Propose the next experiment.
1. Form a hypothesis grounded in your theory
2. Make a specific numerical prediction for val_loss
3. Choose 1-2 hyperparameter changes to test it
4. Explain your reasoning

Return ONLY a JSON object:
{{
  "hypothesis": "What you think will happen and why",
  "predicted_val_loss": 2.50,
  "reasoning": "The causal mechanism you expect",
  "param_changes": {{"PARAM_NAME": value}},
  "description": "Brief description for results log"
}}"""

        try:
            response = _call_llm([{"role": "user", "content": think_prompt}])
            proposal = _parse_json_from_response(response)
            changes = _validate_changes(proposal["param_changes"], current_params)
            predicted = float(proposal.get("predicted_val_loss", 0))
            hypothesis = proposal.get("hypothesis", "")
            reasoning = proposal.get("reasoning", "")
            description = proposal.get("description", "LLM-proposed change")
        except Exception as e:
            print(f"  LLM proposal failed: {e}. Using random mutation.")
            from condition_a import mutate_params
            changes = mutate_params(current_params)
            predicted = best_loss
            hypothesis = "Fallback to random mutation"
            reasoning = f"LLM call failed: {e}"
            desc_parts = [f"{k}: {current_params.get(k)} -> {v}" for k, v in changes.items()]
            description = "; ".join(desc_parts) if desc_parts else "random mutation"

        if not changes:
            print("  No valid changes proposed, skipping")
            continue

        desc_parts = [f"{k}: {current_params.get(k)} -> {v}" for k, v in changes.items()]
        print(f"  Trying: {'; '.join(desc_parts)}")
        print(f"  Prediction: {predicted:.4f}")

        # Write pre-experiment notebook entry
        with open(notebook_path, 'a') as f:
            f.write(f"## Experiment {exp_num}: {description}\n")
            f.write(f"**Config change:** {'; '.join(desc_parts)}\n")
            f.write(f"**Hypothesis:** {hypothesis}\n")
            f.write(f"**Predicted val_loss:** {predicted:.2f}\n")
            f.write(f"**Reasoning:** {reasoning}\n")

        write_params(TRAIN_PY, changes)

        # Phase 2: TEST
        result = run_training()

        if result is None:
            print(f"  CRASH")
            with open(results_path, 'a') as f:
                f.write(f"{exp_num}\t0.000000\t{predicted:.2f}\tcrash\t{description}\n")
            with open(notebook_path, 'a') as f:
                f.write(f"**Result:** CRASH\n\n---\n\n")
            with open(TRAIN_PY, 'w') as f:
                f.write(best_content)
            continue

        val_loss = result["val_loss"]
        steps = int(result.get("num_steps", 0))
        params_m = result.get("num_params_M", 0)
        pred_error = val_loss - predicted

        if val_loss < best_loss:
            improvement = best_loss - val_loss
            print(f"  KEEP: val_loss={val_loss:.6f} (improved by {improvement:.6f}), pred_error={pred_error:+.2f}")
            best_loss = val_loss
            best_content = open(TRAIN_PY).read()
            status = "improved"
            with open(os.path.join(output_dir, f"train_exp{exp_num}.py"), 'w') as f:
                f.write(best_content)
        else:
            diff = val_loss - best_loss
            print(f"  DISCARD: val_loss={val_loss:.6f} (worse by {diff:.6f}), pred_error={pred_error:+.2f}")
            status = "worsened"
            with open(TRAIN_PY, 'w') as f:
                f.write(best_content)

        with open(results_path, 'a') as f:
            f.write(f"{exp_num}\t{val_loss:.6f}\t{predicted:.2f}\t{status}\t{description}\n")

        all_results.append({"exp": exp_num, "val_loss": val_loss, "status": status,
                           "predicted": predicted, "pred_error": pred_error})

        # Phase 3: LEARN — reflect, update notebook and theory
        learn_prompt = f"""You are a scientist who just ran an experiment.

Experiment {exp_num}: {description}
Hypothesis: {hypothesis}
Predicted val_loss: {predicted:.4f}
Actual val_loss: {val_loss:.6f}
Prediction error: {pred_error:+.4f} (positive = worse than predicted, negative = better)
Steps: {steps}, Params: {params_m:.1f}M
Status: {status} (best so far: {best_loss:.6f})

Your current theory:
{theory_content}

TASK: Provide your learning and theory updates.

Return ONLY a JSON object:
{{
  "learning": "What this result teaches you. Be specific about what was confirmed or refuted.",
  "theory_updates": "Any additions or changes to your theory (confirmed principles, refuted hypotheses, new open questions). Return empty string if no updates.",
  "prediction_accuracy_note": "Brief note on whether your predictions are improving"
}}"""

        try:
            learn_response = _call_llm([{"role": "user", "content": learn_prompt}])
            learning = _parse_json_from_response(learn_response)
            learning_text = learning.get("learning", "")
            theory_updates = learning.get("theory_updates", "")
        except Exception as e:
            learning_text = f"Learning extraction failed: {e}"
            theory_updates = ""

        # Update notebook
        with open(notebook_path, 'a') as f:
            f.write(f"**Result:** val_loss={val_loss:.6f}, {steps} steps, {params_m:.1f}M params\n")
            f.write(f"**Prediction error:** {pred_error:+.4f} (predicted {predicted:.2f}, got {val_loss:.4f})\n")
            f.write(f"**Learning:** {learning_text}\n\n---\n\n")

        # Update theory if there are updates
        if theory_updates and theory_updates.strip():
            with open(theory_path) as f:
                current_theory = f.read()

            update_theory_prompt = f"""Here is the current theory document:

{current_theory}

Based on experiment {exp_num}, apply these updates:
{theory_updates}

Return the COMPLETE updated theory document. Keep the same structure (Confirmed Principles, Refuted Hypotheses, Open Questions). Add or modify entries based on the new evidence. Be concise."""

            try:
                new_theory = _call_llm([{"role": "user", "content": update_theory_prompt}])
                # Only update if response looks like a theory doc
                if "Confirmed" in new_theory or "Refuted" in new_theory or "Principles" in new_theory:
                    with open(theory_path, 'w') as f:
                        f.write(new_theory)
            except Exception:
                pass  # Keep existing theory on failure

    # Save final best
    with open(os.path.join(output_dir, "train_best.py"), 'w') as f:
        f.write(best_content)

    # Compute prediction stats
    pred_errors = [r["pred_error"] for r in all_results if "pred_error" in r]
    mae = sum(abs(e) for e in pred_errors) / len(pred_errors) if pred_errors else 0

    summary = {
        "condition": "C",
        "num_experiments": num_experiments,
        "best_val_loss": best_loss,
        "total_experiments_run": len(all_results),
        "keeps": sum(1 for r in all_results if r["status"] == "improved"),
        "discards": sum(1 for r in all_results if r["status"] == "worsened"),
        "crashes": sum(1 for r in all_results if r.get("status") == "crash"),
        "mean_absolute_prediction_error": mae,
        "prediction_errors": pred_errors,
    }
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"CONDITION C COMPLETE. Best val_loss: {best_loss:.6f}")
    print(f"Mean prediction error: {mae:.4f}")
    print(f"{'='*60}")

    with open(TRAIN_PY, 'w') as f:
        f.write(original_content)

    return {
        "best_loss": best_loss,
        "best_config_path": os.path.join(output_dir, "train_best.py"),
        "results_path": results_path,
        "theory_path": theory_path,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experiments", type=int, default=20)
    parser.add_argument("--output-dir", default="results/phase1/condition_c")
    parser.add_argument("--initial-train-py", default=None)
    parser.add_argument("--initial-theory", default=None)
    args = parser.parse_args()

    run_condition_c(args.num_experiments, args.output_dir,
                    args.initial_train_py, args.initial_theory)
