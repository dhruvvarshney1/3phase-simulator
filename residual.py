"""residual.py -- Fully implicit residual of the three-phase black-oil model.

Governing equations (README section 4.3), surface-volume conservation
(water, oil in STB; gas in SCF), backward Euler in time (section 5.1):

    R_w,i = [M_w,i(x^{n+1}) - M_w,i(x^n)]/dt + sum_faces q_w,ij - Q_w,i
    R_o,i = [M_o,i(x^{n+1}) - M_o,i(x^n)]/dt + sum_faces q_o,ij - Q_o,i
    R_g,i = [M_g,i(x^{n+1}) - M_g,i(x^n)]/dt + sum_faces (q_g,ij + Rs_up q_o,ij) - Q_g,i

with the in-place (accumulation) masses, PV0 = Vb / 5.615 [bbl]:

    M_w = PV0 * phi * S_w / B_w
    M_o = PV0 * phi * S_o / B_o,                 S_o = 1 - S_w - S_g
    M_g = PV0 * phi * (S_g / B_g + R_s * S_o / B_o)          (free + dissolved gas)

Face flux (positive from the left to the right cell of a face):

    q_a,ij = T_ij * lambda_a,up(a) * dPhi_a,           lambda_a = k_ra / (mu_a B_a)
    dPhi_a = (p_left - p_right) - gamma_a * (D_left - D_right)
    dissolved gas flux = R_s,up(o) * q_o,ij            (R_s of the oil-upstream cell)

gamma_a = rho_a g [psi/ft] is a constant per phase (``cfg.pvt.gamma_*``, zero =
no gravity) and D is depth, positive downward (``grid.depth``). Each phase is
upwinded on its own potential. P_c = 0, so all phases share p.

Sign convention: a face flux q leaves the left cell (+q) and enters the right
cell (-q). Q is the net well source (positive = injection).

Global ordering (cell-blocked, see grid.py): row 3*c + a is equation ``a``
(0 water, 1 oil, 2 gas) of cell ``c``; the unknown vector is x[3c:3c+3] =
(p, S_w, S_g).

Convergence scaling (README section 5.2)
----------------------------------------
Raw residuals mix STB/day (water, oil) with SCF/day (gas) and cell sizes; a
psi-versus-saturation mismatch also affects the update. Each equation is
scaled by its cell's pore-volume throughput per day,

    scale_{c,a} = max( M_{c,a}(x^n), 1e-3 * mean_c M_{.,a}(x^n), tiny ) / dt,

(floored at ``newton.char_rate_floor``), so that ``R / scale`` is the
imbalance as a *fraction of the cell's own inventory per step*. Requiring
max|R/scale| < 1e-6 therefore has the same physical meaning for water, oil
and gas, for every cell size, and for every dt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from config import SimulationConfig, default_config, override, validate_config
from grid import NVAR, Grid, build_grid, pack_state, unpack_state
from pvt import PVT, PVTProperties
from relperm import RelPerm, RelPermResult
from rock import Rock, build_rock
from transmissibility import FaceUpstream, face_transmissibility, select_upstream
from units import BBL_FT3
from wells import Wells, WellTerms, build_wells, injector_bhp, well_terms


# ---------------------------------------------------------------------------
# Static model bundle
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ReservoirModel:
    """Everything that does not change during a simulation.

    Attributes
    ----------
    cfg, grid, rock, pvt, relperm : configuration and property objects.
    wells : :class:`Wells` or None (well-less model, used in closed-reservoir tests).
    trans : (n_faces,) face transmissibilities [bbl*cp/(day*psi)].
    pv0 : (n_cells,) bulk volume / 5.615 [bbl], the pore-volume prefactor.
    """

    cfg: SimulationConfig
    grid: Grid
    rock: Rock
    pvt: PVT
    relperm: RelPerm
    wells: Optional[Wells]
    trans: np.ndarray
    pv0: np.ndarray


def build_model(cfg: SimulationConfig, phi_field: np.ndarray | None = None,
                with_wells: bool = True) -> ReservoirModel:
    """Assemble a :class:`ReservoirModel` from a configuration.

    ``phi_field`` optionally overrides the porosity map (flat array);
    ``with_wells=False`` builds a closed, well-less model.
    """
    validate_config(cfg, check_wells=with_wells)
    grid = build_grid(cfg.grid)
    rock = build_rock(cfg, grid, phi_field=phi_field)
    wells = build_wells(cfg, grid, rock) if with_wells else None
    return ReservoirModel(
        cfg=cfg, grid=grid, rock=rock, pvt=PVT(cfg.pvt), relperm=RelPerm(cfg.relperm),
        wells=wells, trans=face_transmissibility(grid, rock.perm),
        pv0=grid.bulk_volume / BBL_FT3,
    )


def uniform_state(model: ReservoirModel) -> np.ndarray:
    """Return the uniform initial state vector from ``cfg.init``."""
    n, s = model.grid.n_cells, model.cfg.init
    return pack_state(np.full(n, s.p_init), np.full(n, s.sw_init), np.full(n, s.sg_init))


def _check_state(model: ReservoirModel, x: np.ndarray) -> None:
    """Raise ValueError if ``x`` has the wrong length."""
    if np.shape(x) != (model.grid.n_unknowns,):
        raise ValueError(f"state vector must have length {model.grid.n_unknowns}")


# ---------------------------------------------------------------------------
# Cell properties
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class CellFields:
    """Per-cell state, properties (with derivative arrays) and mobilities."""

    p: np.ndarray
    sw: np.ndarray
    sg: np.ndarray
    so: np.ndarray
    props: PVTProperties
    kr: RelPermResult
    phi: np.ndarray
    dphi_dp: np.ndarray
    lam_w: np.ndarray
    lam_o: np.ndarray
    lam_g: np.ndarray


def evaluate_cells(model: ReservoirModel, x: np.ndarray) -> CellFields:
    """Evaluate PVT, relative permeability, porosity and mobilities for every cell."""
    _check_state(model, x)
    p, sw, sg = unpack_state(x)
    props = model.pvt.evaluate(p)
    kr = model.relperm.evaluate(sw, sg)
    return CellFields(
        p=p, sw=sw, sg=sg, so=1.0 - sw - sg, props=props, kr=kr,
        phi=model.rock.porosity(p), dphi_dp=model.rock.dphi_dp(p),
        lam_w=kr.krw / (props.mu_w * props.bw),
        lam_o=kr.kro / (props.mu_o * props.bo),
        lam_g=kr.krg / (props.mu_g * props.bg),
    )


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------
def accumulation_from_fields(model: ReservoirModel, f: CellFields) -> np.ndarray:
    """Return in-place masses M as an (n_cells, 3) array: [water STB, oil STB, gas SCF]."""
    c = model.pv0 * f.phi
    m = np.empty((model.grid.n_cells, NVAR))
    m[:, 0] = c * f.sw / f.props.bw
    m[:, 1] = c * f.so / f.props.bo
    m[:, 2] = c * (f.sg / f.props.bg + f.props.rs * f.so / f.props.bo)
    return m


def accumulation(model: ReservoirModel, x: np.ndarray) -> np.ndarray:
    """Return the (n_cells, 3) in-place mass array for state ``x``."""
    return accumulation_from_fields(model, evaluate_cells(model, x))


def in_place_totals(model: ReservoirModel, x: np.ndarray) -> np.ndarray:
    """Return total in-place (water STB, oil STB, gas incl. dissolved SCF)."""
    return accumulation(model, x).sum(axis=0)


# ---------------------------------------------------------------------------
# Fluxes
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class FaceFluxes:
    """Interface fluxes (positive left -> right).

    q : (n_faces, 3) water STB/d, oil STB/d, total gas SCF/d (free + dissolved).
    q_g_free, q_g_diss : the two gas contributions [SCF/d].
    """

    q: np.ndarray
    q_g_free: np.ndarray
    q_g_diss: np.ndarray


def phase_upstream(model: ReservoirModel, p: np.ndarray) -> tuple[FaceUpstream, ...]:
    """Return the (water, oil, gas) upstream selections from the phase potentials."""
    g = model.grid
    d_depth = g.depth[g.face_left] - g.depth[g.face_right]
    c = model.cfg.pvt
    return tuple(select_upstream(g, p, gamma * d_depth) for gamma in (c.gamma_w, c.gamma_o, c.gamma_g))


def compute_face_fluxes(model: ReservoirModel, f: CellFields, ups: tuple[FaceUpstream, ...]) -> FaceFluxes:
    """Compute phase-upwinded two-point fluxes of all phases on every face."""
    uw, uo, ug = ups
    q_w = model.trans * uw.dp * f.lam_w[uw.up]
    q_o = model.trans * uo.dp * f.lam_o[uo.up]
    q_gf = model.trans * ug.dp * f.lam_g[ug.up]
    q_gd = f.props.rs[uo.up] * q_o
    return FaceFluxes(q=np.column_stack([q_w, q_o, q_gf + q_gd]), q_g_free=q_gf, q_g_diss=q_gd)


def flux_divergence(model: ReservoirModel, fluxes: FaceFluxes) -> np.ndarray:
    """Return the (n_cells, 3) net outflow: +q for the left cell, -q for the right cell."""
    g, n = model.grid, model.grid.n_cells
    out = np.empty((n, NVAR))
    for a in range(NVAR):
        out[:, a] = (np.bincount(g.face_left, weights=fluxes.q[:, a], minlength=n)
                     - np.bincount(g.face_right, weights=fluxes.q[:, a], minlength=n))
    return out


def well_source(model: ReservoirModel, f: CellFields) -> tuple[np.ndarray, Optional[WellTerms]]:
    """Return the (n_cells, 3) net source Q (positive = injection) and the well terms."""
    q = np.zeros((model.grid.n_cells, NVAR))
    if model.wells is None:
        return q, None
    terms = well_terms(model.wells, model.pvt, model.relperm, f.p, f.sw, f.sg)
    np.add.at(q, terms.producer_cells, terms.q_prod)
    np.add.at(q, terms.injector_cells, terms.q_inj)
    return q, terms


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ResidualEvaluation:
    """Residual plus the intermediate quantities (useful for diagnostics/tests)."""

    residual: np.ndarray          # (3 n,) cell-blocked
    r: np.ndarray                 # (n, 3)
    m_new: np.ndarray             # (n, 3)
    divergence: np.ndarray        # (n, 3)
    source: np.ndarray            # (n, 3)
    fields: CellFields
    upstream: tuple               # (water, oil, gas) FaceUpstream
    fluxes: FaceFluxes
    terms: Optional[WellTerms]


def evaluate_residual(model: ReservoirModel, x: np.ndarray, m_old: np.ndarray,
                      dt: float) -> ResidualEvaluation:
    """Evaluate the residual at ``x`` for old masses ``m_old`` and time step ``dt`` [day]."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if np.shape(m_old) != (model.grid.n_cells, NVAR):
        raise ValueError("m_old must have shape (n_cells, 3)")
    f = evaluate_cells(model, x)
    ups = phase_upstream(model, f.p)
    fluxes = compute_face_fluxes(model, f, ups)
    div = flux_divergence(model, fluxes)
    src, terms = well_source(model, f)
    m_new = accumulation_from_fields(model, f)
    r = (m_new - m_old) / dt + div - src
    return ResidualEvaluation(residual=r.reshape(-1), r=r, m_new=m_new, divergence=div,
                              source=src, fields=f, upstream=ups, fluxes=fluxes, terms=terms)


def compute_residual(model: ReservoirModel, x: np.ndarray, m_old: np.ndarray,
                     dt: float) -> np.ndarray:
    """Return the (3 n,) cell-blocked residual vector."""
    return evaluate_residual(model, x, m_old, dt).residual


# ---------------------------------------------------------------------------
# Scaling and norms (README section 5.2)
# ---------------------------------------------------------------------------
def residual_scales(model: ReservoirModel, m_old: np.ndarray, dt: float) -> np.ndarray:
    """Return the (3 n,) per-equation scale = inventory throughput per day (see module doc)."""
    floor = 1.0e-3 * np.maximum(m_old.mean(axis=0, keepdims=True), 1.0e-30)
    scale = np.maximum(np.maximum(m_old, floor), 1.0e-30) / dt
    return np.maximum(scale, model.cfg.newton.char_rate_floor).reshape(-1)


def scaled_residual_norm(residual: np.ndarray, scales: np.ndarray) -> float:
    """Return max_i |R_i / scale_i| (the scaled infinity norm)."""
    return float(np.max(np.abs(residual) / scales))


# ---------------------------------------------------------------------------
# Well diagnostics
# ---------------------------------------------------------------------------
def well_rates(model: ReservoirModel, x: np.ndarray) -> Dict[str, float]:
    """Return current surface well rates and diagnostics.

    Keys: q_inj_w, q_prod_w, q_prod_o, q_prod_g_free, q_prod_g_total (STB/d, SCF/d),
    water_cut, gor (SCF/STB), producer_bhp, injector_bhp (diagnostic), p_prod_cell, p_inj_cell.
    """
    if model.wells is None:
        raise ValueError("model has no wells")
    _check_state(model, x)
    p, sw, sg = unpack_state(x)
    w = model.wells
    _, terms = well_source(model, evaluate_cells(model, x))
    assert terms is not None
    r = terms.rates
    liquid = r.q_w + r.q_o
    return {
        "q_inj_w": w.injector_rate,
        "q_prod_w": r.q_w, "q_prod_o": r.q_o,
        "q_prod_g_free": r.q_g_free, "q_prod_g_total": r.q_g_total,
        "water_cut": r.q_w / liquid if liquid > 0.0 else 0.0,
        "gor": r.q_g_total / r.q_o if r.q_o > 0.0 else 0.0,
        "producer_bhp": w.producer_bhp,
        "injector_bhp": injector_bhp(w, model.pvt, model.relperm, p[w.injector_cells],
                                     sw[w.injector_cells], sg[w.injector_cells]),
        "p_prod_cell": float(p[w.producer_cell]), "p_inj_cell": float(p[w.injector_cell]),
    }


def main() -> None:
    """Self-checks: stationarity, conservation, hand-computed two-cell fluxes, timing."""
    import time

    cfg = default_config()

    # 1. Uniform state, no wells: residual identically zero
    closed = build_model(cfg, with_wells=False)
    x0 = uniform_state(closed)
    m0 = accumulation(closed, x0)
    r0 = compute_residual(closed, x0, m0, 1.0)
    assert np.max(np.abs(r0)) == 0.0, "uniform closed state must have zero residual"
    print("stationary uniform state, no wells: |R|_inf = 0")

    # 2. Well-only source: R = -Q for a stationary state with wells
    wm = build_model(cfg)
    ev = evaluate_residual(wm, uniform_state(wm), accumulation(wm, uniform_state(wm)), 1.0)
    assert np.isclose(ev.r[wm.wells.injector_cell, 0], -cfg.wells.injector_rate)
    assert ev.r[wm.wells.producer_cell, 1] > 0.0
    assert np.isclose(ev.r.sum(axis=0)[0], -cfg.wells.injector_rate + ev.terms.rates.q_w)

    # 3. Conservation: flux divergence sums to zero for random heterogeneous states
    rng = np.random.default_rng(3)
    n = closed.grid.n_cells
    xr = pack_state(rng.uniform(1500.0, 4500.0, n), rng.uniform(0.25, 0.6, n), rng.uniform(0.0, 0.1, n))
    div = evaluate_residual(closed, xr, accumulation(closed, xr), 1.0).divergence
    for a, name in enumerate(("water", "oil", "gas")):
        rel = abs(div[:, a].sum()) / np.abs(div[:, a]).sum()
        assert rel < 1e-12, f"{name} flux divergence does not sum to zero: {rel:.2e}"
    print("flux divergence sums to zero for all phases (relative < 1e-12)")

    # 4. Hand-computed two-cell case with dissolved gas
    two = build_model(override(cfg, "grid", nx=2, ny=1, nz=1), with_wells=False)
    p = np.array([3000.0, 2900.0])
    sw, sg = np.array([0.40, 0.30]), np.array([0.02, 0.0])
    x2 = pack_state(p, sw, sg)
    ev2 = evaluate_residual(two, x2, accumulation(two, x2), 1.0)
    props, kr = two.pvt.evaluate(p), two.relperm.evaluate(sw, sg)
    dp, T = p[0] - p[1], two.trans[0]
    lam_o0 = kr.kro[0] / (props.mu_o[0] * props.bo[0])
    lam_w0 = kr.krw[0] / (props.mu_w[0] * props.bw[0])
    lam_g0 = kr.krg[0] / (props.mu_g[0] * props.bg[0])
    q_w, q_o = T * lam_w0 * dp, T * lam_o0 * dp
    q_g = T * lam_g0 * dp + props.rs[0] * q_o
    assert np.allclose(ev2.r[0], [q_w, q_o, q_g], rtol=1e-12)
    assert np.allclose(ev2.r[1], -np.array([q_w, q_o, q_g]), rtol=1e-12)
    print(f"two-cell hand check: q_w={q_w:.3f}, q_o={q_o:.3f} STB/d, q_g={q_g:.1f} SCF/d  OK")

    # 5. Accumulation enters as (M_new - M_old)/dt: halving dt doubles that contribution
    x_new = pack_state(p + 1.0, sw, sg)
    m_old = accumulation(two, x2)
    r_dt1 = evaluate_residual(two, x_new, m_old, 1.0)
    r_dt2 = evaluate_residual(two, x_new, m_old, 2.0)
    assert np.allclose(r_dt1.r - r_dt2.r, 0.5 * (r_dt1.m_new - m_old), rtol=1e-10, atol=1e-8)

    # scaling
    scales = residual_scales(closed, m0, 1.0)
    assert scales.shape == (3 * n,) and np.all(scales > 0.0)

    # timing
    t0 = time.perf_counter()
    for _ in range(20):
        compute_residual(wm, xr, accumulation(wm, xr), 1.0)
    print(f"residual evaluation (900 cells): {(time.perf_counter() - t0) / 20 * 1e3:.2f} ms")
    print("Residual self-checks: PASSED")


if __name__ == "__main__":
    main()