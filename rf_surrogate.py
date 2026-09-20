"""Random-forest next-state surrogate for Newton initial guesses.

The surrogate is deliberately an initializer only: every prediction is clipped to
physical bounds and then corrected by the fully implicit Newton solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from grid import pack_state, unpack_state

FEATURE_NAMES = (
    "p", "sw", "sg", "phi", "perm", "distance_producer", "distance_injector",
    "injector_rate", "producer_bhp", "dt", "time", "neighbor_p", "neighbor_sw",
    "neighbor_sg", "delta_p", "delta_sw", "delta_sg",
)


def build_features(x: np.ndarray, model: Any, dt: float, time_days: float,
                   x_prev: np.ndarray | None = None) -> np.ndarray:
    """Build one feature row per cell for a current simulator state."""
    p, sw, sg = unpack_state(x)
    grid = model.grid
    if model.wells is None:
        producer_distance = np.zeros(grid.n_cells)
        injector_distance = np.zeros(grid.n_cells)
        injector_rate = np.zeros(grid.n_cells)
        producer_bhp = np.zeros(grid.n_cells)
    else:
        producer_distance = grid.distance_to_cell(*grid.cell_ij(model.wells.producer_cell))
        injector_distance = grid.distance_to_cell(*grid.cell_ij(model.wells.injector_cell))
        injector_rate = np.full(grid.n_cells, model.wells.injector_rate)
        producer_bhp = np.full(grid.n_cells, model.wells.producer_bhp)
    if x_prev is None:
        delta_p = np.zeros_like(p)
        delta_sw = np.zeros_like(sw)
        delta_sg = np.zeros_like(sg)
    else:
        old_p, old_sw, old_sg = unpack_state(x_prev)
        delta_p, delta_sw, delta_sg = p - old_p, sw - old_sw, sg - old_sg
    return np.column_stack((
        p, sw, sg, model.rock.phi0, model.rock.perm,
        producer_distance, injector_distance, injector_rate, producer_bhp,
        np.full(grid.n_cells, dt), np.full(grid.n_cells, time_days),
        grid.neighbor_mean(p), grid.neighbor_mean(sw), grid.neighbor_mean(sg),
        delta_p, delta_sw, delta_sg,
    ))


@dataclass
class RFSurrogate:
    """Multi-output random forest predicting the next cell state."""

    model: Any
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @classmethod
    def fit(cls, features: np.ndarray, targets: np.ndarray, **kwargs: Any) -> "RFSurrogate":
        from sklearn.ensemble import RandomForestRegressor

        estimator = RandomForestRegressor(n_jobs=-1, random_state=42, **kwargs)
        estimator.fit(np.asarray(features, dtype=float), np.asarray(targets, dtype=float))
        return cls(estimator)

    def predict_state(self, features: np.ndarray, model: Any) -> np.ndarray:
        """Predict and physically bound a packed next-state vector."""
        prediction = np.asarray(self.model.predict(features), dtype=float)
        if prediction.ndim != 2 or prediction.shape[1] != 3:
            raise ValueError("RF model must predict columns (pressure, Sw, Sg)")
        p, sw, sg = prediction.T
        p = np.maximum(p, model.cfg.newton.p_min)
        sw = np.clip(sw, 0.0, 1.0)
        sg = np.clip(sg, 0.0, 1.0)
        max_total = 1.0 - model.relperm.cfg.sor
        total = sw + sg
        mask = total > max_total
        sw[mask] *= max_total / total[mask]
        sg[mask] *= max_total / total[mask]
        return pack_state(p, sw, sg)

    def predict_initial_guess(self, x: np.ndarray, model: Any, dt: float,
                              time_days: float) -> np.ndarray:
        features = build_features(x, model, dt, time_days)
        return self.predict_state(features, model)

    def save(self, directory: str | Path) -> str:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "rf_surrogate.joblib"
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)
        return str(path)

    @classmethod
    def load(cls, directory: str | Path) -> "RFSurrogate":
        path = Path(directory) / "rf_surrogate.joblib"
        payload = joblib.load(path)
        return cls(payload["model"], tuple(payload.get("feature_names", FEATURE_NAMES)))


def save_dataset(path: str | Path, features: np.ndarray, targets: np.ndarray) -> None:
    """Save a portable training dataset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=np.asarray(features), targets=np.asarray(targets),
                        feature_names=np.asarray(FEATURE_NAMES))


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return np.asarray(data["features"], dtype=float), np.asarray(data["targets"], dtype=float)
