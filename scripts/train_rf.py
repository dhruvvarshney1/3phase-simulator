"""Train and evaluate the Random Forest next-state surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from rf_surrogate import RFSurrogate, load_dataset


def train(dataset: str, output_dir: str, estimators: int = 200, max_depth: int = 20) -> dict:
    features, targets = load_dataset(dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    surrogate = RFSurrogate.fit(
        x_train, y_train, n_estimators=estimators, max_depth=max_depth
    )
    prediction = surrogate.model.predict(x_test)
    metrics = {
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "mae": mean_absolute_error(y_test, prediction, multioutput="raw_values").tolist(),
        "rmse": np.sqrt(mean_squared_error(y_test, prediction, multioutput="raw_values")).tolist(),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    surrogate.save(output)
    with (output / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/rf_dataset.npz")
    parser.add_argument("--output-dir", default="results/rf_models")
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=20)
    args = parser.parse_args()
    train(args.dataset, args.output_dir, args.estimators, args.max_depth)
