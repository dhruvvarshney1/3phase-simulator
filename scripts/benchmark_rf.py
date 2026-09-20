"""Benchmark RF prediction against the previous-state initializer on a dataset."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from rf_surrogate import RFSurrogate, load_dataset


def benchmark(dataset: str, model_dir: str) -> dict:
    features, targets = load_dataset(dataset)
    surrogate = RFSurrogate.load(model_dir)
    start = time.perf_counter()
    prediction = surrogate.model.predict(features)
    rf_seconds = time.perf_counter() - start
    baseline = features[:, :3]
    result = {
        "rows": int(len(features)),
        "rf_prediction_seconds": rf_seconds,
        "baseline_mae": float(np.mean(np.abs(baseline - targets))),
        "rf_mae": float(np.mean(np.abs(prediction - targets))),
        "rf_rmse": float(np.sqrt(np.mean((prediction - targets) ** 2))),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/rf_dataset.npz")
    parser.add_argument("--model-dir", default="results/rf_models")
    args = parser.parse_args()
    benchmark(args.dataset, args.model_dir)
