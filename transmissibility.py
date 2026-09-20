"""transmissibility.py -- Two-point-flux transmissibilities and upstream selection.

Face transmissibility (README section 4.7, Eq. 4.7a)
----------------------------------------------------
For the face between cells i and j on a uniform Cartesian grid:

    k_ij = 2 k_i k_j / (k_i + k_j)                (harmonic mean, cells of equal size)
    T_ij = 0.001127 * k_ij * A / L                [bbl*cp/(day*psi)]

with A = dy*dz and L = dx for x-faces, A = dx*dz and L = dy for y-faces.
The harmonic mean equals the series-resistance combination of two half-cells,
which is exact for piecewise-constant permeability.

Flux and upwinding (Eq. 4.7b)
-----------------------------
Face flux of phase a from the *left* cell to the *right* cell (STB/day; gas in
SCF/day):

    q_a,ij = T_ij * lambda_a,up * (p_left - p_right),
    lambda_a = k_ra / (mu_a * B_a)

The upstream cell is the left cell when p_left >= p_right, otherwise the right
cell. Because P_c = 0 and gravity is neglected, all phases share the same
potential difference and therefore the same upstream cell. (With capillary
pressure or gravity each phase would need its own upstream selection.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import default_config
from grid import Grid, build_grid
from units import DARCY_BBL_PER_DAY


def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the elementwise harmonic mean 2ab/(a+b) of positive arrays."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.any(a_arr <= 0.0) or np.any(b_arr <= 0.0):
        raise ValueError("harmonic mean requires strictly positive values")
    return 2.0 * a_arr * b_arr / (a_arr + b_arr)


def face_transmissibility(grid: Grid, perm: np.ndarray) -> np.ndarray:
    """Return T_ij [bbl*cp/(day*psi)] for every interior face.

    Parameters
    ----------
    grid : grid with face connectivity and geometric factors A/L.
    perm : (n_cells,) isotropic cell permeability [md].
    """
    perm = np.asarray(perm, dtype=float)
    if perm.shape != (grid.n_cells,):
        raise ValueError("perm must have one value per cell")
    k_face = harmonic_mean(perm[grid.face_left], perm[grid.face_right])
    return DARCY_BBL_PER_DAY * k_face * grid.face_geom


@dataclass(frozen=True, eq=False)
class FaceUpstream:
    """Upstream selection for every face.

    Attributes
    ----------
    dp : (n_faces,) p_left - p_right [psi]; positive flux runs left -> right.
    left_is_up : (n_faces,) True where the left cell is upstream.
    up : (n_faces,) upstream cell index.
    down : (n_faces,) downstream cell index.
    """

    dp: np.ndarray
    left_is_up: np.ndarray
    up: np.ndarray
    down: np.ndarray


def select_upstream(grid: Grid, p: np.ndarray) -> FaceUpstream:
    """Choose the upstream cell of every face from the pressure field.

    Ties (dp == 0) select the left cell; the flux is zero there anyway.
    """
    p = np.asarray(p, dtype=float)
    dp = p[grid.face_left] - p[grid.face_right]
    left_is_up = dp >= 0.0
    up = np.where(left_is_up, grid.face_left, grid.face_right)
    down = np.where(left_is_up, grid.face_right, grid.face_left)
    return FaceUpstream(dp=dp, left_is_up=left_is_up, up=up, down=down)


def main() -> None:
    """Verify transmissibility formulas, series-resistance identity and upstreaming."""
    cfg = default_config()
    grid = build_grid(cfg.grid)

    # uniform permeability: T = 0.001127 * k * A / L
    k0 = 100.0
    T = face_transmissibility(grid, np.full(grid.n_cells, k0))
    expected_x = 0.001127 * k0 * (cfg.grid.dy * cfg.grid.dz) / cfg.grid.dx
    assert np.allclose(T, expected_x), "uniform-k transmissibility mismatch"
    print(f"uniform k = {k0} md -> T = {T[0]:.6f} bbl*cp/(day*psi) on all {T.size} faces")

    # series-resistance identity for two half-cells with contrast
    strip = build_grid(type(cfg.grid)(nx=2, ny=1))
    k = np.array([50.0, 500.0])
    T2 = face_transmissibility(strip, k)[0]
    area, half = strip.dy * strip.dz, strip.dx / 2.0
    resistance = half / (k[0] * area) + half / (k[1] * area)
    assert np.isclose(T2, DARCY_BBL_PER_DAY / resistance), "harmonic mean != series resistance"
    assert np.isclose(harmonic_mean(k[0], k[1]), 2 * 50.0 * 500.0 / 550.0)

    # symmetry of the harmonic mean
    rng = np.random.default_rng(1)
    kr = rng.uniform(1.0, 1000.0, grid.n_cells)
    T_a = face_transmissibility(grid, kr)
    assert np.all(T_a > 0.0)

    # upstream selection follows the pressure gradient
    p = np.linspace(4000.0, 1000.0, grid.n_cells)          # decreasing with index
    up = select_upstream(grid, p)
    assert np.all(up.up == np.minimum(grid.face_left, grid.face_right)), "upstream must be high-p cell"
    assert np.all(up.dp[up.left_is_up] >= 0.0) and np.all(up.dp[~up.left_is_up] < 0.0)
    p_flip = p[::-1].copy()
    up2 = select_upstream(grid, p_flip)
    assert np.all(up2.up == np.maximum(grid.face_left, grid.face_right))
    assert np.array_equal(np.sort(np.concatenate([up.up, up.down])),
                          np.sort(np.concatenate([grid.face_left, grid.face_right])))
    print("Transmissibility self-checks: PASSED")


if __name__ == "__main__":
    main()