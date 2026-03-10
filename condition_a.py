"""
Condition A — Outcome-Only (Karpathy Baseline)

Loop: mutate → train → score → keep/discard
Memory: Best config only
No LLM. Pure random search.

This faithfully represents the autoresearch paradigm.
"""

import os
import re
import random
import subprocess
import time
import json
import shutil

TRAIN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")

# The hyperparameter search space — identical across all conditions
HYPERPARAM_RANGES = {
    "N_LAYER": [2, 3, 4, 5, 6, 8],
    "N_HEAD": [2, 3, 4, 6, 8],
    "N_EMBD": [96, 128, 192, 256, 384, 512],
    "DROPOUT": [0.0, 0.05, 0.1, 0.15, 0.2],
    "LEARNING_RATE": [1e-4, 3e-4, 6e-4, 1e-3, 2e-3, 3e-3],
    "WEIGHT_DECAY": [0.0, 0.01, 0.05, 0.1, 0.2],
    "BATCH_SIZE": [2, 4, 8, 16, 32],
    "GRAD_ACCUM_STEPS": [1, 2, 4],
    "WARMUP_STEPS": [0, 5, 10, 20, 50],
    "BETAS": [(0.9, 0.95), (0.9, 0.99), (0.9, 0.999)],
}


def read_current_params(filepath):
    """Read current hyperparameter values from train.py."""
    with open(filepath) as f:
        content = f.read()
    params = {}
    for key in HYPERPARAM_RANGES:
        pattern = rf'^{key}\s*=\s*(.+)$'
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            val = match.group(1).strip()
            try:
                params[key] = eval(val)
            except Exception:
                params[key] = val
    return params


def write_params(filepath, changes):
    """Write hyperparameter values to train.py."""
    with open(filepath) as f:
        content = f.read()
    for key, value in changes.items():
        if isinstance(value, float) and value < 0.01:
            val_str = f"{value:.0e}"
        elif isinstance(value, tuple):
            val_str = repr(value)
        else:
            val_str = repr(value)
        content = re.sub(
            rf'^({key}\s*=\s*)(.+)$',
            rf'\g<1>{val_str}',
            content,
            flags=re.MULTILINE,
        )
    with open(filepath, 'w') as f:
        f.write(content)


def mutate_params(current_params):
    """Randomly change 1-2 hyperparameters."""
    num_changes = random.choice([1, 1, 1, 2])
    keys = random.sample(list(HYPERPARAM_RANGES.keys()), num_changes)
    changes = {}
    for key in keys:
        options = [v for v in HYPERPARAM_RANGES[key] if v != current_params.get(key)]
        if options:
            changes[key] = random.choice(options)

    # Enforce N_EMBD % N_HEAD == 0
    n_embd = changes.get("N_EMBD", current_params.get("N_EMBD", 192))
    n_head = changes.get("N_HEAD", current_params.get("N_HEAD", 3))
    if n_embd % n_head != 0:
        valid_heads = [h for h in HYPERPARAM_RANGES["N_HEAD"] if n_embd % h == 0]
        if valid_heads:
            changes["N_HEAD"] = random.choice(valid_heads)
        else:
            if "N_EMBD" in changes:
                del changes["N_EMBD"]
            if "N_HEAD" in changes:
                del changes["N_HEAD"]
    return changes


def run_training(timeout=900):
    """Run train.py and return parsed results dict or None on failure."""
    try:
        result = subprocess.run(
            ["uv", "run", "train.py"],
            capture_output=True, text=True, timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        output = result.stdout + result.stderr

        parsed = {}
        for key in ["val_loss", "training_seconds", "total_seconds",
                     "total_tokens_M", "num_steps", "num_params_M", "depth"]:
            match = re.search(rf'^{key}:\s+([\d.]+)', output, re.MULTILINE)
            if match:
                parsed[key] = float(match.group(1))

        if "val_loss" in parsed:
            return parsed
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_condition_a(num_experiments, output_dir, initial_train_py=None):
    """
    Run Condition A: brute-force search.

    Args:
        num_experiments: Number of experiments to run
        output_dir: Directory to save results
        initial_train_py: Optional path to initial train.py state
    Returns:
        dict with best_loss, best_config, results_path
    """
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.tsv")

    # Save original train.py
    with open(TRAIN_PY) as f:
        original_content = f.read()

    # If initial state provided, use it
    if initial_train_py:
        shutil.copy2(initial_train_py, TRAIN_PY)

    # Initialize results
    with open(results_path, 'w') as f:
        f.write("exp\tval_loss\tsteps\tparams_M\ttraining_sec\tstatus\tdescription\n")

    # Run baseline
    print(f"\n{'='*60}")
    print(f"CONDITION A — BRUTE FORCE (Karpathy Baseline)")
    print(f"Running {num_experiments} experiments")
    print(f"{'='*60}")

    print("\n[A-0] Running baseline...")
    baseline = run_training()
    if baseline is None:
        print("Baseline CRASHED. Aborting Condition A.")
        with open(TRAIN_PY, 'w') as f:
            f.write(original_content)
        return None

    best_loss = baseline["val_loss"]
    best_content = open(TRAIN_PY).read()
    steps = int(baseline.get("num_steps", 0))
    params_m = baseline.get("num_params_M", 0)
    train_sec = baseline.get("training_seconds", 0)

    print(f"  Baseline: val_loss={best_loss:.6f}, steps={steps}, params={params_m:.1f}M")
    with open(results_path, 'a') as f:
        f.write(f"0\t{best_loss:.6f}\t{steps}\t{params_m:.1f}\t{train_sec:.1f}\tkeep\tbaseline\n")

    # Save baseline config
    with open(os.path.join(output_dir, "train_exp0.py"), 'w') as f:
        f.write(best_content)

    # Experiment loop
    all_results = [{"exp": 0, "val_loss": best_loss, "status": "keep", "description": "baseline"}]

    for exp_num in range(1, num_experiments + 1):
        print(f"\n[A-{exp_num}/{num_experiments}]")

        current_params = read_current_params(TRAIN_PY)
        changes = mutate_params(current_params)

        if not changes:
            print("  No valid mutations found, skipping")
            continue

        desc_parts = []
        for key, val in changes.items():
            old = current_params.get(key, "?")
            desc_parts.append(f"{key}: {old} -> {val}")
        description = "; ".join(desc_parts)
        print(f"  Trying: {description}")

        write_params(TRAIN_PY, changes)

        t0 = time.time()
        result = run_training()
        elapsed = time.time() - t0

        if result is None:
            print(f"  CRASH ({elapsed:.0f}s)")
            with open(results_path, 'a') as f:
                f.write(f"{exp_num}\t0.000000\t0\t0.0\t0.0\tcrash\t{description}\n")
            all_results.append({"exp": exp_num, "val_loss": 0, "status": "crash", "description": description})
            # Revert to best
            with open(TRAIN_PY, 'w') as f:
                f.write(best_content)
            continue

        val_loss = result["val_loss"]
        steps = int(result.get("num_steps", 0))
        params_m = result.get("num_params_M", 0)
        train_sec = result.get("training_seconds", 0)

        if val_loss < best_loss:
            improvement = best_loss - val_loss
            print(f"  KEEP: val_loss={val_loss:.6f} (improved by {improvement:.6f}), steps={steps}, {elapsed:.0f}s")
            best_loss = val_loss
            best_content = open(TRAIN_PY).read()
            status = "keep"
            # Save winning config
            with open(os.path.join(output_dir, f"train_exp{exp_num}.py"), 'w') as f:
                f.write(best_content)
        else:
            diff = val_loss - best_loss
            print(f"  DISCARD: val_loss={val_loss:.6f} (worse by {diff:.6f}), steps={steps}, {elapsed:.0f}s")
            status = "discard"
            # Revert to best
            with open(TRAIN_PY, 'w') as f:
                f.write(best_content)

        with open(results_path, 'a') as f:
            f.write(f"{exp_num}\t{val_loss:.6f}\t{steps}\t{params_m:.1f}\t{train_sec:.1f}\t{status}\t{description}\n")
        all_results.append({"exp": exp_num, "val_loss": val_loss, "status": status, "description": description})

    # Save final best config
    with open(os.path.join(output_dir, "train_best.py"), 'w') as f:
        f.write(best_content)

    # Save summary
    summary = {
        "condition": "A",
        "num_experiments": num_experiments,
        "best_val_loss": best_loss,
        "total_experiments_run": len(all_results),
        "keeps": sum(1 for r in all_results if r["status"] == "keep"),
        "discards": sum(1 for r in all_results if r["status"] == "discard"),
        "crashes": sum(1 for r in all_results if r["status"] == "crash"),
    }
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"CONDITION A COMPLETE. Best val_loss: {best_loss:.6f}")
    print(f"{'='*60}")

    # Restore original train.py
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
    parser.add_argument("--num-experiments", type=int, default=15)
    parser.add_argument("--output-dir", default="results/phase1/condition_a")
    args = parser.parse_args()

    run_condition_a(args.num_experiments, args.output_dir)
