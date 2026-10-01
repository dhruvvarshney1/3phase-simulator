"""Jacobian assembly for the current :mod:`residual` model contract."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from residual import ReservoirModel, compute_residual, phase_upstream
from wells import producer_rates

__all__ = [
    "numerical_jacobian", "analytical_jacobian", "build_jacobian",
    "jacobian_relative_error", "max_nnz_per_row",
]


def numerical_jacobian(
    model: ReservoirModel,
    x: np.ndarray,
    m_old: np.ndarray,
    dt: float,
    eps_p: float = 1.0e-2,
    eps_s: float = 1.0e-6,
) -> csr_matrix:
    """Return a central finite-difference Jacobian of the current residual."""
    n = x.size
    matrix = np.zeros((n, n), dtype=float)
    for column in range(n):
        step = eps_p if column % 3 == 0 else eps_s
        plus, minus = x.copy(), x.copy()
        plus[column] += step
        minus[column] -= step
        if column % 3:
            plus[column] = min(plus[column], 1.0)
            minus[column] = max(minus[column], 0.0)
            step = 0.5 * (plus[column] - minus[column])
        matrix[:, column] = (
            compute_residual(model, plus, m_old, dt)
            - compute_residual(model, minus, m_old, dt)
        ) / (2.0 * step)
    return csr_matrix(matrix)


def analytical_jacobian(
    model: ReservoirModel,
    x: np.ndarray,
    m_old: np.ndarray,
    dt: float,
) -> csr_matrix:
    """Assemble the sparse analytic Jacobian of the fully implicit residual (vectorized)."""
    n_cells = model.grid.n_cells
    p, sw, sg = _state_arrays(x)
    props = model.pvt.evaluate(p)
    kr = model.relperm.evaluate(sw, sg)
    phi = model.rock.porosity(p)
    dphi = model.rock.dphi_dp(p)
    pv = model.pv0 * phi

    dlam_w, dlam_o, dlam_g = _mobility_derivatives(props, kr)
    rows, cols, values = [], [], []

    def add(r: np.ndarray, c: np.ndarray, v: np.ndarray) -> None:
        rows.append(np.ravel(r))
        cols.append(np.ravel(c))
        values.append(np.ravel(v))

    # Accumulation derivatives: each cell's three equations depend only on its
    # own pressure and saturations.
    so = 1.0 - sw - sg
    dm = np.zeros((n_cells, 3, 3), dtype=float)
    dm[:, 0, 0] = model.pv0 * (dphi * sw / props.bw - phi * sw * props.dbw_dp / props.bw**2)
    dm[:, 0, 1] = pv / props.bw
    dm[:, 1, 0] = model.pv0 * (dphi * so / props.bo - phi * so * props.dbo_dp / props.bo**2)
    dm[:, 1, 1] = -pv / props.bo
    dm[:, 1, 2] = -pv / props.bo
    gas_term = sg / props.bg + props.rs * so / props.bo
    dm[:, 2, 0] = model.pv0 * (
        dphi * gas_term
        + phi * (-sg * props.dbg_dp / props.bg**2
                  + props.drs_dp * so / props.bo
                  - props.rs * so * props.dbo_dp / props.bo**2)
    )
    dm[:, 2, 1] = -pv * props.rs / props.bo
    dm[:, 2, 2] = pv * (1.0 / props.bg - props.rs / props.bo)
    cell_eq = 3 * np.arange(n_cells)[:, None, None] + np.arange(3)[None, :, None]
    cell_var = 3 * np.arange(n_cells)[:, None, None] + np.arange(3)[None, None, :]
    add(np.broadcast_to(cell_eq, dm.shape), np.broadcast_to(cell_var, dm.shape), dm / dt)

    # Face flux derivatives, each phase upwinded on its own potential
    # (matching the residual's >= tie convention).
    left, right = model.grid.face_left, model.grid.face_right
    trans = model.trans
    ups_w, ups_o, ups_g = phase_upstream(model, p)
    lam = (
        kr.krw / (props.mu_w * props.bw),
        kr.kro / (props.mu_o * props.bo),
        kr.krg / (props.mu_g * props.bg),
    )

    def phase_derivs(ups, mob, dmob):
        """d(T*mob_up*dPhi)/d(left vars), d/d(right vars): each (n_faces, 3)."""
        d_left = np.zeros((left.size, 3))
        d_right = np.zeros((left.size, 3))
        m_up = mob[ups.up]
        d_left[:, 0] = trans * m_up
        d_right[:, 0] = -trans * m_up
        d_up = (trans * ups.dp)[:, None] * dmob[ups.up]
        d_left[ups.left_is_up] += d_up[ups.left_is_up]
        d_right[~ups.left_is_up] += d_up[~ups.left_is_up]
        return d_left, d_right

    w_l, w_r = phase_derivs(ups_w, lam[0], dlam_w)
    o_l, o_r = phase_derivs(ups_o, lam[1], dlam_o)
    g_l, g_r = phase_derivs(ups_g, lam[2], dlam_g)
    # dissolved gas: R_s(oil-upstream) * q_o
    rs_up = props.rs[ups_o.up][:, None]
    gd_l, gd_r = rs_up * o_l, rs_up * o_r
    drs_term = props.drs_dp[ups_o.up] * trans * ups_o.dp * lam[1][ups_o.up]
    gd_l[ups_o.left_is_up, 0] += drs_term[ups_o.left_is_up]
    gd_r[~ups_o.left_is_up, 0] += drs_term[~ups_o.left_is_up]

    var = np.arange(3)[None, :]
    for equation, (d_l, d_r) in enumerate(((w_l, w_r), (o_l, o_r), (g_l + gd_l, g_r + gd_r))):
        row_l = np.broadcast_to((3 * left + equation)[:, None], d_l.shape)
        row_r = np.broadcast_to((3 * right + equation)[:, None], d_l.shape)
        col_l = 3 * left[:, None] + var
        col_r = 3 * right[:, None] + var
        add(row_l, col_l, d_l)
        add(row_l, col_r, d_r)
        add(row_r, col_l, -d_l)
        add(row_r, col_r, -d_r)

    if model.wells is not None:
        w = model.wells
        c = w.producer_cells
        rates = producer_rates(w, model.pvt, model.relperm, p[c], sw[c], sg[c])
        add(np.broadcast_to(3 * c[:, None, None] + np.arange(3)[None, :, None], rates.jac.shape),
            np.broadcast_to(3 * c[:, None, None] + np.arange(3)[None, None, :], rates.jac.shape),
            rates.jac)

    return coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(3 * n_cells, 3 * n_cells)).tocsr()


def _state_arrays(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size % 3:
        raise ValueError("state vector length must be a multiple of 3")
    return x[0::3], x[1::3], x[2::3]


def _mobility_derivatives(props, kr):
    lam_w = kr.krw / (props.mu_w * props.bw)
    lam_o = kr.kro / (props.mu_o * props.bo)
    lam_g = kr.krg / (props.mu_g * props.bg)
    dlam_w = np.column_stack((
        -lam_w * props.dbw_dp / props.bw,
        kr.dkrw_dsw / (props.mu_w * props.bw),
        kr.dkrw_dsg / (props.mu_w * props.bw),
    ))
    dlam_o = np.column_stack((
        -lam_o * (props.dmu_o_dp / props.mu_o + props.dbo_dp / props.bo),
        kr.dkro_dsw / (props.mu_o * props.bo),
        kr.dkro_dsg / (props.mu_o * props.bo),
    ))
    dlam_g = np.column_stack((
        -lam_g * props.dbg_dp / props.bg,
        kr.dkrg_dsw / (props.mu_g * props.bg),
        kr.dkrg_dsg / (props.mu_g * props.bg),
    ))
    return dlam_w, dlam_o, dlam_g


def build_jacobian(
    model: ReservoirModel,
    x: np.ndarray,
    m_old: np.ndarray,
    dt: float,
    method: str = "analytical",
) -> csr_matrix:
   
    if method == "analytical":
        return analytical_jacobian(
            model, x, m_old, dt
        )
    elif method == "numerical":
        return numerical_jacobian(
            model, x, m_old, dt
        )
    else:
        raise ValueError(
            f"Unknown Jacobian method '{method}'"
        )


def jacobian_relative_error(J_analytical: csr_matrix, J_numerical: csr_matrix) -> float:
    difference = (J_analytical - J_numerical).toarray()
    denominator = np.max(np.abs(J_numerical.toarray()))
    return float(np.max(np.abs(difference)) / denominator) if denominator else float(np.max(np.abs(difference)))


def max_nnz_per_row(J: csr_matrix) -> int:
    return int(np.max(np.diff(J.tocsr().indptr))) if J.shape[0] else 0
