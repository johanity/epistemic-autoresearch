"""
Experiment Harness — Transfer Experiment Orchestrator

Runs all three conditions (A, B, C) across both phases, with support
for multiple independent runs for statistical power.

Usage:
    # Single condition/phase:
    uv run harness.py --phase 1 --condition A --num-experiments 20
    uv run harness.py --phase 1 --condition B --num-experiments 20
    uv run harness.py --phase 1 --condition C --num-experiments 20

    # Full run (both phases, all conditions):
    uv run harness.py --full-run --num-experiments 20

    # Multiple runs for statistics:
    uv run harness.py --full-run --num-experiments 20 --num-runs 5

    # Specific run ID (for resuming):
    uv run harness.py --full-run --num-experiments 20 --run-id 3
"""

import os
import re
import json
import shutil
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(BASE_DIR, "train.py")
PREPARE_PY = os.path.join(BASE_DIR, "prepare.py")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PROGRESS_LOG = os.path.join(BASE_DIR, "CLAUDE_PROGRESS.md")

from condition_a import run_condition_a
from condition_b import run_condition_b
from condition_c import run_condition_c


def set_time_budget(seconds):
    """Modify TIME_BUDGET in prepare.py."""
    with open(PREPARE_PY) as f:
        content = f.read()
    content = re.sub(
        r'^(TIME_BUDGET\s*=\s*)\d+',
        rf'\g<1>{seconds}',
        content,
        flags=re.MULTILINE,
    )
    with open(PREPARE_PY, 'w') as f:
        f.write(content)
    print(f"  TIME_BUDGET set to {seconds}s")


def get_time_budget():
    """Read current TIME_BUDGET from prepare.py."""
    with open(PREPARE_PY) as f:
        content = f.read()
    match = re.search(r'^TIME_BUDGET\s*=\s*(\d+)', content, re.MULTILINE)
    return int(match.group(1)) if match else 120


def log_progress(message):
    """Append to CLAUDE_PROGRESS.md."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(PROGRESS_LOG, 'a') as f:
        f.write(f"\n**{timestamp}** — {message}\n")
    print(f"[LOG] {message}")


def reset_train_py_to_default():
    """Reset train.py to the clean baseline template."""
    baseline_path = os.path.join(BASE_DIR, "train_baseline.py")
    shutil.copy2(baseline_path, TRAIN_PY)


def get_output_dir(run_id, phase, condition):
    """Get the output directory for a specific run/phase/condition."""
    if run_id is None:
        return os.path.join(RESULTS_DIR, f"phase{phase}", f"condition_{condition.lower()}")
    return os.path.join(RESULTS_DIR, f"run{run_id}", f"phase{phase}", f"condition_{condition.lower()}")


def run_single(phase, condition, num_experiments, run_id=None):
    """Run a single condition in a single phase."""
    time_budget = 120 if phase == 1 else 600
    output_dir = get_output_dir(run_id, phase, condition)

    run_label = f" (run {run_id})" if run_id is not None else ""
    print("=" * 70)
    print(f"HARNESS — Phase {phase}, Condition {condition}{run_label}")
    print(f"TIME_BUDGET = {time_budget}s, N = {num_experiments}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Set time budget
    original_budget = get_time_budget()
    set_time_budget(time_budget)

    # Reset train.py to defaults
    reset_train_py_to_default()

    # Check for Phase 2 transfer state
    initial_train_py = None
    initial_theory = None
    if phase == 2:
        p1_dir = get_output_dir(run_id, 1, condition)
        p1_best = os.path.join(p1_dir, "train_best.py")
        if os.path.exists(p1_best):
            initial_train_py = p1_best
            print(f"  Transferring best config from Phase 1: {p1_best}")
        else:
            print(f"  WARNING: No Phase 1 best config found at {p1_best}")

        # For Condition C, also transfer theory
        if condition == "C":
            p1_theory = os.path.join(p1_dir, "theory.md")
            if os.path.exists(p1_theory):
                initial_theory = p1_theory
                print(f"  Transferring theory from Phase 1: {p1_theory}")

    log_progress(f"Phase {phase} Condition {condition}{run_label} starting — "
                 f"TIME_BUDGET={time_budget}s, N={num_experiments}")

    # Run the appropriate condition
    if condition == "A":
        result = run_condition_a(num_experiments, output_dir, initial_train_py)
    elif condition == "B":
        result = run_condition_b(num_experiments, output_dir, initial_train_py)
    elif condition == "C":
        result = run_condition_c(num_experiments, output_dir, initial_train_py,
                                 initial_theory)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    if result:
        log_progress(f"Phase {phase} Condition {condition}{run_label} complete — "
                     f"best_loss={result['best_loss']:.6f}")
    else:
        log_progress(f"Phase {phase} Condition {condition}{run_label} FAILED")

    # Restore TIME_BUDGET
    set_time_budget(original_budget)
    return result


def run_full(num_experiments, run_id=None):
    """Run all conditions across both phases."""
    run_label = f" (run {run_id})" if run_id is not None else ""
    print("\n" + "=" * 70)
    print(f"FULL RUN{run_label} — All conditions, both phases")
    print(f"N = {num_experiments} experiments per condition per phase")
    print("=" * 70)

    results = {}

    # Phase 1: all three conditions
    for cond in ["A", "B", "C"]:
        result = run_single(1, cond, num_experiments, run_id)
        results[f"P1_{cond}"] = result

    # Phase 2: all three conditions (with transfer from Phase 1)
    for cond in ["A", "B", "C"]:
        result = run_single(2, cond, num_experiments, run_id)
        results[f"P2_{cond}"] = result

    # Print summary
    print("\n" + "=" * 70)
    print(f"FULL RUN COMPLETE{run_label}")
    print("=" * 70)
    print(f"\n{'Phase':<8} {'Cond':<6} {'Best val_loss':<15}")
    print("-" * 30)
    for key in ["P1_A", "P1_B", "P1_C", "P2_A", "P2_B", "P2_C"]:
        r = results.get(key)
        phase = key[:2]
        cond = key[3:]
        loss = f"{r['best_loss']:.6f}" if r else "FAILED"
        print(f"{phase:<8} {cond:<6} {loss:<15}")

    # Save run summary
    run_dir = os.path.join(RESULTS_DIR, f"run{run_id}") if run_id else RESULTS_DIR
    os.makedirs(run_dir, exist_ok=True)
    summary = {
        "run_id": run_id,
        "num_experiments": num_experiments,
        "timestamp": datetime.now().isoformat(),
        "results": {k: {"best_loss": v["best_loss"]} if v else None
                    for k, v in results.items()},
    }
    with open(os.path.join(run_dir, "run_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Transfer Experiment Harness")
    parser.add_argument("--phase", type=int, choices=[1, 2])
    parser.add_argument("--condition", choices=["A", "B", "C"])
    parser.add_argument("--num-experiments", type=int, default=20,
                        help="Experiments per condition per phase (default: 20)")
    parser.add_argument("--full-run", action="store_true",
                        help="Run all conditions across both phases")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of independent runs (for statistics)")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Specific run ID (default: auto-increment)")
    parser.add_argument("--start-run", type=int, default=1,
                        help="Starting run ID for multi-run (default: 1)")
    args = parser.parse_args()

    if args.full_run:
        if args.num_runs > 1:
            all_results = {}
            for i in range(args.start_run, args.start_run + args.num_runs):
                print(f"\n{'#' * 70}")
                print(f"# RUN {i} of {args.start_run + args.num_runs - 1}")
                print(f"{'#' * 70}")
                results = run_full(args.num_experiments, run_id=i)
                all_results[i] = results

            # Print aggregate summary
            print("\n" + "=" * 70)
            print("AGGREGATE RESULTS")
            print("=" * 70)
            for key in ["P1_A", "P1_B", "P1_C", "P2_A", "P2_B", "P2_C"]:
                losses = []
                for run_results in all_results.values():
                    r = run_results.get(key)
                    if r:
                        losses.append(r["best_loss"])
                if losses:
                    mean = sum(losses) / len(losses)
                    std = (sum((x - mean) ** 2 for x in losses) / len(losses)) ** 0.5
                    print(f"  {key}: mean={mean:.4f} ± {std:.4f} (n={len(losses)})")

            # Save aggregate
            agg_path = os.path.join(RESULTS_DIR, "aggregate_results.json")
            agg = {}
            for key in ["P1_A", "P1_B", "P1_C", "P2_A", "P2_B", "P2_C"]:
                losses = []
                for run_results in all_results.values():
                    r = run_results.get(key)
                    if r:
                        losses.append(r["best_loss"])
                if losses:
                    mean = sum(losses) / len(losses)
                    std = (sum((x - mean) ** 2 for x in losses) / len(losses)) ** 0.5
                    agg[key] = {"mean": mean, "std": std, "values": losses}
            with open(agg_path, 'w') as f:
                json.dump(agg, f, indent=2)
            print(f"\nAggregate results saved to {agg_path}")
        else:
            run_id = args.run_id or args.start_run
            run_full(args.num_experiments, run_id=run_id)

    elif args.phase and args.condition:
        run_single(args.phase, args.condition, args.num_experiments, args.run_id)
    else:
        parser.error("Either --full-run or both --phase and --condition are required")


if __name__ == "__main__":
    main()
