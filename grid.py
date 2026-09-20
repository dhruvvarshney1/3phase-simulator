"""grid.py -- 2D Cartesian grid, face connectivity and global unknown ordering.

Cell indexing (README section 4.1)
----------------------------------
    cell_index(i, j) = j * nx + i        (i along x, j along y, zero-based)

Global unknown ordering (README section 4.2) -- CELL-BLOCKED
------------------------------------------------------------
The three primary unknowns of cell ``c`` are stored contiguously:

    x[3*c + 0] = p_o   (psi)
    x[3*c + 1] = S_w
    x[3*c + 2] = S_g

The residual uses the same blocking: row ``3*c + 0`` is the water equation,
``3*c + 1`` the oil equation, ``3*c + 2`` the gas equation of cell ``c``.
Each Jacobian row then couples only to its own cell and its <= 4 neighbours,
giving <= 15 nonzero columns per row. This ordering is used identically in
``residual.py``, ``jacobian.py`` and ``newton_solver.py``.

Faces
-----
All interior faces are stored in flat arrays, x-direction faces first, then
y-direction faces. Boundaries are no-flow, so no boundary faces exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from config import GridConfig, default_config
from units import pore_volume_bbl

NVAR: int = 3
IDX_P: int = 0
IDX_SW: int = 1
IDX_SG: int = 2

# Neighbour slots in ``Grid.neighbors`` columns
WEST, EAST, SOUTH, NORTH = 0, 1, 2, 3


@dataclass(frozen=True, eq=False)
class Grid:
    """Immutable 2D Cartesian grid with precomputed connectivity.

    Attributes
    ----------
    nx, ny, nz, dx, dy, dz : grid dimensions (ft for spacings).
    n_cells : nx * ny.
    i_idx, j_idx : (n_cells,) integer cell coordinates.
    x_center, y_center : (n_cells,) cell-centre coordinates in ft.
    bulk_volume : (n_cells,) bulk volume dx*dy*dz [ft^3].
    neighbors : (n_cells, 4) neighbour cell indices (W, E, S, N); -1 = boundary.
    face_left, face_right : (n_faces,) cell indices on either side of each face.
    face_dir : (n_faces,) 0 for x-direction face, 1 for y-direction face.
    face_area : (n_faces,) flow area [ft^2] (dy*dz for x faces, dx*dz for y faces).
    face_length : (n_faces,) centre-to-centre distance [ft] (dx or dy).
    face_geom : (n_faces,) A / L [ft], the geometric factor of the transmissibility.
    n_xfaces : number of x-direction faces (they occupy the first slots).
    """

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    n_cells: int
    i_idx: np.ndarray
    j_idx: np.ndarray
    x_center: np.ndarray
    y_center: np.ndarray
    bulk_volume: np.ndarray
    neighbors: np.ndarray
    face_left: np.ndarray
    face_right: np.ndarray
    face_dir: np.ndarray
    face_area: np.ndarray
    face_length: np.ndarray
    face_geom: np.ndarray
    n_xfaces: int

    # ------------------------------------------------------------------
    @property
    def nc(self) -> int:
        """Legacy alias for :attr:`n_cells`."""
        return self.n_cells

    @property
    def n_faces(self) -> int:
        """Total number of interior faces."""
        return int(self.face_left.size)

    @property
    def n_unknowns(self) -> int:
        """Total number of primary unknowns (3 per cell)."""
        return NVAR * self.n_cells

    @property
    def shape(self) -> Tuple[int, int]:
        """Array shape (ny, nx) for reshaping flat cell arrays into maps."""
        return (self.ny, self.nx)

    # ------------------------------------------------------------------
    def cell_index(self, i: np.ndarray | int, j: np.ndarray | int) -> np.ndarray | int:
        """Return the flat cell index ``j*nx + i`` (validates bounds)."""
        ia, ja = np.asarray(i), np.asarray(j)
        if np.any(ia < 0) or np.any(ia >= self.nx) or np.any(ja < 0) or np.any(ja >= self.ny):
            raise ValueError(f"cell coordinates ({i}, {j}) outside {self.nx}x{self.ny} grid")
        result = ja * self.nx + ia
        return int(result) if np.ndim(result) == 0 else result

    def cell_ij(self, index: np.ndarray | int) -> Tuple[np.ndarray | int, np.ndarray | int]:
        """Return (i, j) for a flat cell index."""
        idx = np.asarray(index)
        if np.any(idx < 0) or np.any(idx >= self.n_cells):
            raise ValueError("cell index outside grid")
        i, j = idx % self.nx, idx // self.nx
        if np.ndim(idx) == 0:
            return int(i), int(j)
        return i, j

    def pore_volume(self, porosity: np.ndarray | float) -> np.ndarray:
        """Return per-cell pore volume in bbl = Vb * phi / 5.615."""
        return np.asarray(pore_volume_bbl(self.bulk_volume, porosity), dtype=float)

    def distance_to_cell(self, i: int, j: int) -> np.ndarray:
        """Return the distance [ft] from every cell centre to the centre of cell (i, j)."""
        c = self.cell_index(i, j)
        return np.hypot(self.x_center - self.x_center[c], self.y_center - self.y_center[c])

    def neighbor_mean(self, field: np.ndarray) -> np.ndarray:
        """Return the mean of ``field`` over existing (non-boundary) neighbours.

        Cells without any neighbour (1x1 grid) return their own value.
        """
        values = np.asarray(field, dtype=float)
        valid = self.neighbors >= 0
        safe = np.where(valid, self.neighbors, 0)
        total = np.where(valid, values[safe], 0.0).sum(axis=1)
        count = valid.sum(axis=1)
        return np.where(count > 0, total / np.maximum(count, 1), values)

    def to_map(self, field: np.ndarray) -> np.ndarray:
        """Reshape a flat (n_cells,) array into an (ny, nx) map."""
        return np.asarray(field).reshape(self.ny, self.nx)


def CartesianGrid(
    nx: int,
    ny: int,
    nz: int = 1,
    dx: float = 50.0,
    dy: float = 50.0,
    dz: float = 20.0,
) -> Grid:
    """Build a grid using the legacy constructor-style API."""
    return build_grid(GridConfig(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz))


def build_grid(cfg: GridConfig) -> Grid:
    """Construct a :class:`Grid` from a :class:`GridConfig`."""
    nx, ny, nz = int(cfg.nx), int(cfg.ny), int(cfg.nz)
    if nz != 1:
        raise ValueError("only nz == 1 (2D) grids are supported")
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    if min(cfg.dx, cfg.dy, cfg.dz) <= 0.0:
        raise ValueError("cell dimensions must be positive")

    n = nx * ny
    idx = np.arange(n, dtype=np.int64).reshape(ny, nx)
    i_idx = (np.arange(n) % nx).astype(np.int64)
    j_idx = (np.arange(n) // nx).astype(np.int64)
    x_center = (i_idx + 0.5) * cfg.dx
    y_center = (j_idx + 0.5) * cfg.dy
    bulk = np.full(n, cfg.dx * cfg.dy * cfg.dz, dtype=float)

    neighbors = -np.ones((n, 4), dtype=np.int64)
    neighbors[idx[:, 1:].ravel(), WEST] = idx[:, :-1].ravel()
    neighbors[idx[:, :-1].ravel(), EAST] = idx[:, 1:].ravel()
    neighbors[idx[1:, :].ravel(), SOUTH] = idx[:-1, :].ravel()
    neighbors[idx[:-1, :].ravel(), NORTH] = idx[1:, :].ravel()

    xl, xr = idx[:, :-1].ravel(), idx[:, 1:].ravel()   # x-direction faces
    yl, yr = idx[:-1, :].ravel(), idx[1:, :].ravel()   # y-direction faces
    n_xf, n_yf = xl.size, yl.size

    face_left = np.concatenate([xl, yl]).astype(np.int64)
    face_right = np.concatenate([xr, yr]).astype(np.int64)
    face_dir = np.concatenate([np.zeros(n_xf, dtype=np.int64), np.ones(n_yf, dtype=np.int64)])
    face_area = np.concatenate([np.full(n_xf, cfg.dy * cfg.dz), np.full(n_yf, cfg.dx * cfg.dz)])
    face_length = np.concatenate([np.full(n_xf, cfg.dx), np.full(n_yf, cfg.dy)])

    return Grid(
        nx=nx, ny=ny, nz=nz, dx=float(cfg.dx), dy=float(cfg.dy), dz=float(cfg.dz),
        n_cells=n, i_idx=i_idx, j_idx=j_idx, x_center=x_center, y_center=y_center,
        bulk_volume=bulk, neighbors=neighbors,
        face_left=face_left, face_right=face_right, face_dir=face_dir,
        face_area=face_area, face_length=face_length,
        face_geom=face_area / face_length, n_xfaces=int(n_xf),
    )


# ---------------------------------------------------------------------------
# Global unknown ordering helpers (cell-blocked)
# ---------------------------------------------------------------------------
def dof_index(cells: np.ndarray | int, var: int) -> np.ndarray | int:
    """Return the global unknown/equation index ``3*cell + var``."""
    return NVAR * np.asarray(cells) + var if np.ndim(cells) else NVAR * int(cells) + var


def pack_state(p: np.ndarray, sw: np.ndarray, sg: np.ndarray) -> np.ndarray:
    """Pack per-cell (p, S_w, S_g) arrays into the cell-blocked global vector."""
    p, sw, sg = (np.asarray(a, dtype=float) for a in (p, sw, sg))
    if not (p.shape == sw.shape == sg.shape and p.ndim == 1):
        raise ValueError("p, sw, sg must be 1D arrays of equal length")
    x = np.empty(NVAR * p.size, dtype=float)
    x[IDX_P::NVAR] = p
    x[IDX_SW::NVAR] = sw
    x[IDX_SG::NVAR] = sg
    return x


def unpack_state(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack a cell-blocked global vector into copies of (p, S_w, S_g)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size % NVAR != 0:
        raise ValueError("state vector length must be a multiple of 3")
    return (x[IDX_P::NVAR].copy(), x[IDX_SW::NVAR].copy(), x[IDX_SG::NVAR].copy())


def main() -> None:
    """Build the default grid and run structural self-checks."""
    cfg = default_config()
    grid = build_grid(cfg.grid)
    print(f"grid {grid.nx} x {grid.ny} x {grid.nz}: {grid.n_cells} cells, "
          f"{grid.n_unknowns} unknowns, {grid.n_faces} faces ({grid.n_xfaces} in x)")

    expected_faces = (grid.nx - 1) * grid.ny + grid.nx * (grid.ny - 1)
    assert grid.n_faces == expected_faces, "face count mismatch"

    # neighbour symmetry: if a's east neighbour is b, then b's west neighbour is a
    for a, b in ((WEST, EAST), (SOUTH, NORTH)):
        for c in range(grid.n_cells):
            nb = grid.neighbors[c, b]
            if nb >= 0:
                assert grid.neighbors[nb, a] == c, "neighbour symmetry violated"

    # neighbour mean of a linear field equals the field at interior cells
    interior = (grid.i_idx > 0) & (grid.i_idx < grid.nx - 1) & \
               (grid.j_idx > 0) & (grid.j_idx < grid.ny - 1)
    lin = grid.i_idx.astype(float) + 2.0 * grid.j_idx
    assert np.allclose(grid.neighbor_mean(lin)[interior], lin[interior]), "neighbour mean failed"

    # index round trip and state packing
    c = grid.cell_index(2, 2)
    assert grid.cell_ij(c) == (2, 2) and c == 2 * grid.nx + 2
    p, sw, sg = np.arange(grid.n_cells, dtype=float), np.full(grid.n_cells, 0.25), np.zeros(grid.n_cells)
    p2, sw2, sg2 = unpack_state(pack_state(p, sw, sg))
    assert np.array_equal(p, p2) and np.array_equal(sw, sw2) and np.array_equal(sg, sg2)
    assert int(dof_index(c, IDX_SW)) == 3 * c + 1

    # geometry
    assert np.isclose(grid.bulk_volume.sum(), grid.nx * grid.ny * cfg.grid.dx * cfg.grid.dy * cfg.grid.dz)
    print(f"injector cell (2,2) -> index {c}; total pore volume at phi=0.18: "
          f"{grid.pore_volume(0.18).sum():.4e} bbl")

    strip = build_grid(GridConfig(nx=10, ny=1))
    assert strip.n_faces == 9 and strip.n_xfaces == 9
    print("Grid self-checks: PASSED")


if __name__ == "__main__":
    main()