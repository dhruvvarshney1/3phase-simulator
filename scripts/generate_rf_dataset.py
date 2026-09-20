"""Generate RF next-state training data from small simulator runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import smoke_test_config, override
from residual import pack_state
from rf_surrogate import build_features, save_dataset
from simulator import run_simulation


def generate_dataset(n_cases: int, output: str, seed: int = 2024) -> None:
    rng = np.random.default_rng(seed)
    feature_batches, target_batches = [], []
    for case in range(n_cases):
        cfg = smoke_test_config()
        cfg = override(cfg, "fractal", seed=int(rng.integers(0, 2**31 - 1)))
        cfg = override(cfg, "init", p_init=float(rng.uniform(3500.0, 4500.0)))
        cfg = override(cfg, "wells", injector_rate=float(rng.uniform(300.0, 500.0)),
                       producer_bhp=float(rng.uniform(1000.0, 2500.0)))
        out = run_simulation(cfg, init_guess_mode="previous", verbose=False)
        model = out["context"]
        snapshots = out["snapshots"]
        times = sorted(snapshots)
        for t0, t1 in zip(times[:-1], times[1:]):
            current = snapshots[t0]
            target = snapshots[t1]
            x_current = pack_state(current["p"], current["Sw"], current["Sg"])
            x_target = pack_state(target["p"], target["Sw"], target["Sg"])
            feature_batches.append(build_features(x_current, model, t1 - t0, t0))
            target_batches.append(x_target.reshape(-1, 3))
        print(f"case {case + 1}/{n_cases}: {len(times) - 1} transitions")
    if not feature_batches:
        raise RuntimeError("no transitions were produced; check the report-time configuration")
    save_dataset(output, np.vstack(feature_batches), np.vstack(target_batches))
    print(f"wrote {sum(batch.shape[0] for batch in feature_batches)} rows to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--output", default="data/rf_dataset.npz")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    generate_dataset(args.cases, args.output, args.seed)
