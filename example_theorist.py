"""Example: using Theorist with the epistemic-autoresearch training loop."""

import theorist


@theorist.experiment(
    search_space={
        "lr": [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
        "n_layer": [3, 4, 5, 6, 8],
        "n_embd": [64, 128, 192, 256],
        "batch_size": [2, 4, 8, 16],
        "dropout": [0.0, 0.05, 0.1],
        "weight_decay": [0.0, 0.01, 0.1],
    },
    metric="val_loss",
    minimize=True,
)
def train(config):
    """Run a single training experiment and return the metric."""
    import subprocess, json
    result = subprocess.run(
        ["python", "train.py", "--config", json.dumps(config)],
        capture_output=True, text=True
    )
    # Parse val_loss from training output
    for line in result.stdout.strip().split("\n"):
        if line.startswith("val_loss:"):
            return {"val_loss": float(line.split(":")[1].strip())}
    raise RuntimeError(f"Training failed:\n{result.stderr}")


if __name__ == "__main__":
    results = train.optimize(n=20)
    print(f"Best config: {results.best_config}")
    print(f"Best val_loss: {results.best_metric}")
