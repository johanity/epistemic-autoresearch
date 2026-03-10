"""
Condition B — Reflection-Only (Thinking Time Control)

Loop: LLM proposes change → train → LLM reflects → keep/discard
Memory: Running reflection log (free-form notes after each experiment)
No prediction. The LLM thinks about results after the fact but never commits
to a prediction beforehand.

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
    """Format params dict for display."""
    lines = []
    for k, v in sorted(params.items()):
        lines.append(f"  {k} = {v}")
    return "\n".join(lines)


def _format_search_space():
    """Format the search space for the prompt."""
    lines = []
    for k, v in HYPERPARAM_RANGES.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _call_llm(messages):
    """Call LLM and return text response."""
    resp = completion(model=MODEL, messages=messages, max_tokens=2000, temperature=0.7)
    return resp.choices[0].message.content


def _parse_json_from_response(text):
    """Extract JSON from LLM response."""
    # Try to find JSON block
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Try raw JSON
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON found in response: {text[:200]}")


def _validate_changes(changes, current_params):
    """Validate proposed changes against search space and constraints."""
    valid = {}
    for key, val in changes.items():
        if key not in HYPERPARAM_RANGES:
            continue
        # Handle BETAS specially — might come as list
        if key == "BETAS" and isinstance(val, list):
            val = tuple(val)
        if val in HYPERPARAM_RANGES[key] or (key == "BETAS" and tuple(val) in HYPERPARAM_RANGES[key]):
            valid[key] = val if key != "BETAS" else tuple(val)

    if not valid:
        return valid

    # Enforce N_EMBD % N_HEAD == 0
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


def run_condition_b(num_experiments, output_dir, initial_train_py=None):
    """
    Run Condition B: reflection-only LLM loop.

    Args:
        num_experiments: Number of experiments to run (including baseline)
        output_dir: Directory to save results
        initial_train_py: Optional path to initial train.py state
    Returns:
        dict with best_loss, best_config_path, results_path
    """
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.tsv")
    reflections_path = os.path.join(output_dir, "reflections.md")

    # Save original train.py
    with open(TRAIN_PY) as f:
        original_content = f.read()

    # If initial state provided, use it
    if initial_train_py:
        shutil.copy2(initial_train_py, TRAIN_PY)

    # Initialize files
    with open(results_path, 'w') as f:
        f.write("exp\tval_loss\tsteps\tstatus\tdescription\n")
    with open(reflections_path, 'w') as f:
        f.write("# Condition B — Reflection Notes\n\n")
        f.write("*After every experiment, reflect on what happened. No predictions, no theory document.*\n\n---\n\n")

    print(f"\n{'='*60}")
    print(f"CONDITION B — REFLECTION-ONLY (LLM-Automated)")
    print(f"Running {num_experiments} experiments")
    print(f"{'='*60}")

    # Run baseline
    print("\n[B-0] Running baseline...")
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

    desc = f"Baseline: {', '.join(f'{k}={v}' for k, v in sorted(current_params.items()))}"
    print(f"  Baseline: val_loss={best_loss:.6f}, steps={steps}")

    with open(results_path, 'a') as f:
        f.write(f"0\t{best_loss:.6f}\t{steps}\tkeep\t{desc}\n")

    # Baseline reflection
    with open(reflections_path, 'a') as f:
        f.write(f"## Exp 0 — Baseline\n")
        f.write(f"val_loss = {best_loss:.6f}, {steps} steps.\n\n")
        f.write(f"Starting point established.\n\n---\n\n")

    with open(os.path.join(output_dir, "train_exp0.py"), 'w') as f:
        f.write(best_content)

    all_results = [{"exp": 0, "val_loss": best_loss, "status": "keep"}]

    # Experiment loop
    for exp_num in range(1, num_experiments):
        print(f"\n[B-{exp_num}/{num_experiments - 1}]")

        # Read current state
        with open(results_path) as f:
            results_content = f.read()
        with open(reflections_path) as f:
            reflections_content = f.read()
        current_params = read_current_params(TRAIN_PY)

        # Ask LLM to propose a change
        propose_prompt = f"""You are an autonomous ML researcher optimizing a GPT language model.
You propose changes, run experiments, and reflect on results. You do NOT make predictions.

Current best val_loss: {best_loss:.6f}
Current hyperparameters:
{_format_params(current_params)}

Search space (choose from these values only):
{_format_search_space()}
Constraint: N_EMBD must be divisible by N_HEAD.

Results so far:
{results_content}

Your reflections so far:
{reflections_content[-3000:]}

TASK: Propose the next experiment. Change 1-2 hyperparameters.
Think about what you've learned from your reflections, what hasn't been tried,
and what might improve val_loss.

Return ONLY a JSON object:
{{
  "param_changes": {{"PARAM_NAME": value, ...}},
  "description": "Brief description for the results log"
}}"""

        try:
            response = _call_llm([{"role": "user", "content": propose_prompt}])
            proposal = _parse_json_from_response(response)
            changes = _validate_changes(proposal["param_changes"], current_params)
            description = proposal.get("description", "LLM-proposed change")
        except Exception as e:
            print(f"  LLM proposal failed: {e}. Using random mutation.")
            from condition_a import mutate_params
            changes = mutate_params(current_params)
            desc_parts = [f"{k}: {current_params.get(k)} -> {v}" for k, v in changes.items()]
            description = "; ".join(desc_parts) if desc_parts else "random mutation"

        if not changes:
            print("  No valid changes proposed, skipping")
            continue

        desc_parts = [f"{k}: {current_params.get(k)} -> {v}" for k, v in changes.items()]
        print(f"  Trying: {'; '.join(desc_parts)}")
        print(f"  Desc: {description}")

        write_params(TRAIN_PY, changes)

        # Run training
        result = run_training()

        if result is None:
            print(f"  CRASH")
            with open(results_path, 'a') as f:
                f.write(f"{exp_num}\t0.000000\t0\tcrash\t{description}\n")
            with open(TRAIN_PY, 'w') as f:
                f.write(best_content)
            continue

        val_loss = result["val_loss"]
        steps = int(result.get("num_steps", 0))
        params_m = result.get("num_params_M", 0)

        if val_loss < best_loss:
            improvement = best_loss - val_loss
            print(f"  KEEP: val_loss={val_loss:.6f} (improved by {improvement:.6f})")
            best_loss = val_loss
            best_content = open(TRAIN_PY).read()
            status = "keep"
            with open(os.path.join(output_dir, f"train_exp{exp_num}.py"), 'w') as f:
                f.write(best_content)
        else:
            diff = val_loss - best_loss
            print(f"  DISCARD: val_loss={val_loss:.6f} (worse by {diff:.6f})")
            status = "discard"
            with open(TRAIN_PY, 'w') as f:
                f.write(best_content)

        with open(results_path, 'a') as f:
            f.write(f"{exp_num}\t{val_loss:.6f}\t{steps}\t{status}\t{description}\n")

        all_results.append({"exp": exp_num, "val_loss": val_loss, "status": status})

        # Ask LLM to reflect
        reflect_prompt = f"""You just ran an experiment on a GPT language model.

Experiment {exp_num}: {description}
Result: val_loss = {val_loss:.6f}, {steps} steps, {params_m:.1f}M params
Status: {status} (best so far: {best_loss:.6f})
Previous params: {_format_params(current_params)}
Changed: {json.dumps({k: v for k, v in changes.items()}, default=str)}

Write a brief reflection (3-5 sentences):
- What happened?
- Why do you think it worked or didn't?
- What patterns do you notice?

Return ONLY the reflection text, no JSON."""

        try:
            reflection = _call_llm([{"role": "user", "content": reflect_prompt}])
        except Exception as e:
            reflection = f"Reflection failed: {e}"

        with open(reflections_path, 'a') as f:
            f.write(f"## Exp {exp_num} — {description}\n")
            f.write(f"val_loss = {val_loss:.6f}, {steps} steps. {'KEEP' if status == 'keep' else 'Discard'}.\n\n")
            f.write(f"{reflection.strip()}\n\n---\n\n")

    # Save final best
    with open(os.path.join(output_dir, "train_best.py"), 'w') as f:
        f.write(best_content)

    summary = {
        "condition": "B",
        "num_experiments": num_experiments,
        "best_val_loss": best_loss,
        "total_experiments_run": len(all_results),
        "keeps": sum(1 for r in all_results if r["status"] == "keep"),
        "discards": sum(1 for r in all_results if r["status"] == "discard"),
        "crashes": sum(1 for r in all_results if r.get("status") == "crash"),
    }
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"CONDITION B COMPLETE. Best val_loss: {best_loss:.6f}")
    print(f"{'='*60}")

    with open(TRAIN_PY, 'w') as f:
        f.write(original_content)

    return {
        "best_loss": best_loss,
        "best_config_path": os.path.join(output_dir, "train_best.py"),
        "results_path": results_path,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experiments", type=int, default=20)
    parser.add_argument("--output-dir", default="results/phase1/condition_b")
    parser.add_argument("--initial-train-py", default=None)
    args = parser.parse_args()

    run_condition_b(args.num_experiments, args.output_dir, args.initial_train_py)
