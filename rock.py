"""rock.py -- Porosity and permeability fields, optional rock compressibility.

Porosity
--------
Reference porosity ``phi0`` per cell comes from the fractal generator
(default) or is the constant ``rock.phi_const`` (0.18).

Optional pressure dependence (README section 4.6, OFF by default):

    phi(p)    = phi0 * (1 + c_r * (p - p_ref)),   c_r = 3e-6 1/psi, p_ref = 4000 psi
    dphi/dp   = phi0 * c_r

Permeability (README section 6)
-------------------------------
Isotropic, correlated to porosity and *not* pressure dependent:

    k = k_ref * (phi0 / phi_ref)^k_exp,   clipped to [k_min, k_max]   [md]

with the MVP defaults k_ref = 100 md, phi_ref = 0.18, k_exp = 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from config import RockConfig, SimulationConfig, default_config, override
from fractal_porosity import generate_porosity
from grid import Grid, build_grid


def permeability_from_porosity(phi: np.ndarray, cfg: RockConfig) -> np.ndarray:
    """Return k [md] = k_ref * (phi/phi_ref)^k_exp, clipped to [k_min, k_max]."""
    k = cfg.k_ref * (np.asarray(phi, dtype=float) / cfg.phi_ref) ** cfg.k_exp
    return np.clip(k, cfg.k_min, cfg.k_max)


@dataclass(frozen=True, eq=False)
class Rock:
    """Per-cell rock properties (arrays are read-only).

    Attributes
    ----------
    phi0 : (n_cells,) porosity at the reference pressure.
    perm : (n_cells,) isotropic permeability [md] (kx = ky).
    pressure_dependent : whether porosity depends on pressure.
    c_r, p_ref : rock compressibility [1/psi] and reference pressure [psi].
    """

    phi0: np.ndarray
    perm: np.ndarray
    pressure_dependent: bool
    c_r: float
    p_ref: float

    def porosity(self, p: np.ndarray | float | None = None) -> np.ndarray:
        """Return porosity at pressure ``p``; ``phi0`` if pressure dependence is off."""
        if not self.pressure_dependent or p is None:
            return self.phi0
        return self.phi0 * (1.0 + self.c_r * (np.asarray(p, dtype=float) - self.p_ref))

    def dphi_dp(self, p: np.ndarray | float | None = None) -> np.ndarray:
        """Return dphi/dp (zeros if pressure dependence is off)."""
        if not self.pressure_dependent:
            return np.zeros_like(self.phi0)
        return self.phi0 * self.c_r

    def summary(self) -> Dict[str, float]:
        """Return descriptive statistics of porosity and permeability."""
        return {
            "phi_mean": float(self.phi0.mean()), "phi_std": float(self.phi0.std()),
            "phi_min": float(self.phi0.min()), "phi_max": float(self.phi0.max()),
            "k_mean": float(self.perm.mean()), "k_min": float(self.perm.min()),
            "k_max": float(self.perm.max()),
        }


def build_rock(cfg: SimulationConfig, grid: Grid, phi_field: np.ndarray | None = None) -> Rock:
    """Build the :class:`Rock` for ``cfg`` on ``grid``.

    ``phi_field`` (flat, length n_cells) overrides the generated/constant
    porosity, which is convenient for tests with hand-made heterogeneity.
    """
    if phi_field is not None:
        phi0 = np.array(phi_field, dtype=float).ravel()
        if phi0.size != grid.n_cells:
            raise ValueError("phi_field must have one value per cell")
    elif cfg.rock.use_fractal:
        phi0 = generate_porosity(grid.nx, grid.ny, cfg.fractal).ravel()
    else:
        phi0 = np.full(grid.n_cells, cfg.rock.phi_const, dtype=float)
    if np.any(phi0 <= 0.0) or np.any(phi0 >= 1.0):
        raise ValueError("porosity must lie in (0, 1)")

    perm = permeability_from_porosity(phi0, cfg.rock)
    phi0.setflags(write=False)
    perm.setflags(write=False)
    return Rock(phi0=phi0, perm=perm, pressure_dependent=cfg.rock.pressure_dependent,
                c_r=cfg.rock.c_r, p_ref=cfg.rock.p_ref)


def main() -> None:
    """Build the default rock and verify the porosity-permeability relations."""
    cfg = default_config()
    grid = build_grid(cfg.grid)
    rock = build_rock(cfg, grid)
    stats = rock.summary()
    for key, value in stats.items():
        print(f"{key:>9}: {value:.4f}")

    assert 0.05 <= stats["phi_min"] and stats["phi_max"] <= 0.30
    assert 0.1 <= stats["k_min"] and stats["k_max"] <= 5000.0
    assert np.isclose(permeability_from_porosity(np.array([0.18]), cfg.rock)[0], 100.0)
    order = np.argsort(rock.phi0)
    assert np.all(np.diff(rock.perm[order]) >= -1e-12), "k must increase with phi"

    # constant-porosity option
    const = build_rock(override(cfg, "rock", use_fractal=False), grid)
    assert np.allclose(const.phi0, 0.18) and np.allclose(const.perm, 100.0)

    # pressure dependence (off by default, correct when enabled)
    assert np.array_equal(rock.porosity(np.full(grid.n_cells, 3000.0)), rock.phi0)
    on = Rock(rock.phi0, rock.perm, True, 3.0e-6, 4000.0)
    p = np.full(grid.n_cells, 3000.0)
    assert np.allclose(on.porosity(p), rock.phi0 * (1.0 + 3.0e-6 * (3000.0 - 4000.0)))
    h = 1.0
    fd = (on.porosity(p + h) - on.porosity(p - h)) / (2.0 * h)
    assert np.allclose(fd, on.dphi_dp(p), rtol=1e-8)
    print("Rock self-checks: PASSED")


if __name__ == "__main__":
    main()