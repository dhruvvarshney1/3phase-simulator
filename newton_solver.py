"""
newton_solver.py
=================

Newton-Raphson nonlinear solver for the fully implicit backward-Euler
residual defined in ``residual.py`` (README §5.4).

Algorithm (one call = one attempted time step of size ``dt``):

    1. Evaluate R(x) = compute_residual(x, x_prev, ctx, dt).
    2. Check convergence: scaled inf-norm of R below tolerance (README §5.2).
    3. If not converged: assemble J = build_jacobian(x, x_prev, ctx, dt),
       solve J*dx = -R via scipy.sparse.linalg.spsolve.
    4. Damp + clip the update (README §5.4):
       - start alpha = 1; halve (floor 0.1) while the *undamped* update
         would violate physical bounds "substantially", or while the
         resulting (clipped) residual norm is larger than the current one;
       - after the damping search, hard-clip: p >= p_min, 0 <= Sw <= 1,
         0 <= Sg <= 1, Sw + Sg <= 1 - S_or (proportional rescale).
    5. Accept the damped/clipped state, recompute R, go to 2.
    6. Converged when BOTH the scaled residual criterion (a) AND the
       increment criteria (b) max|dp|/p_ref < tol_p and max|dS| < tol_s
       hold simultaneously (README §5.2) -- this guards against the case
       where a small residual (relative to a possibly large characteristic
       rate scale) is reached while the state is still moving non-trivially,
       which can happen on stiff early time steps.
    7. Fail (return converged=False) after ``max_iter`` iterations, or
       immediately if the linear solve is singular/produces non-finite
       output -- in either case the caller (``simulator.py``) is expected to
       halve dt and retry (README §5.4).

NOTE on damping (documented per README §5.4): this is a pragmatic,
"clip-and-check" educational damping strategy, not a rigorous trust-region
or line-search method with guaranteed global convergence. A production
simulator would typically use a proper line search (e.g. Armijo) or a
trust-region Newton method with a merit function; that is noted here as a
future-work item (see README §11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.sparse.linalg import spsolve

from residual import ReservoirModel, accumulation, compute_residual, residual_scales, scaled_residual_norm, unpack_state
from jacobian import build_jacobian

__all__ = [
    "NewtonParams",
    "NewtonResult",
    "newton_solve",
    "compute_convergence_scale",
    "clip_state",
]


# ---------------------------------------------------------------------------
# Parameters / result containers
# ---------------------------------------------------------------------------
@dataclass
class NewtonParams:
    """Newton solver settings (README §5.4, §12 default table)."""

    tol: float = 1.0e-6          # scaled residual inf-norm tolerance (criterion a)
    max_iter: int = 12           # max Newton iterations per time step
    alpha_min: float = 0.1       # minimum damping factor
    p_min: float = 100.0         # hard pressure floor [psi] (README §5.4)
    dp_rel_tol: float = 1.0e-6   # criterion (b): max|dp|/p_ref
    ds_tol: float = 1.0e-6       # criterion (b): max|dS|


@dataclass
class NewtonResult:
    """Outcome of a single ``newton_solve`` call."""

    x: np.ndarray
    converged: bool
    n_iterations: int
    residual_history: List[float] = field(default_factory=list)
    scaled_residual_history: List[float] = field(default_factory=list)
    final_scaled_residual: float = float("nan")
    alpha_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Convergence scaling (README §5.2)
# ---------------------------------------------------------------------------
def compute_convergence_scale(ctx: ReservoirModel) -> float:
    """
    Characteristic flow-rate scale [STB/day] used to non-dimensionalize the
    residual inf-norm before comparing it against ``tol`` (README §5.2).

    Rationale: residual equations are balances of surface-volume rates
    (STB/day). Their natural magnitude depends entirely on how fast fluid is
    actually moving through the reservoir, which varies enormously between
    a closed-reservoir material-balance test (near-zero net rates) and an
    actively produced/injected waterflood (hundreds of STB/day). A single
    fixed absolute tolerance would therefore be either far too loose (missed
    convergence on an active case) or far too tight (many wasted iterations
    on a near-static case). We instead scale by:

        scale = max(total injection rate [STB/day],
                    (total reservoir pore volume [bbl]) / 1000 days,
                    1.0 STB/day floor)

    The second term is a "slow throughput" floor: a full pore-volume
    exchanged over the ~1000-day default simulation horizon, so that even a
    completely shut-in (no-well) reservoir gets a sensible, non-zero,
    physically-motivated convergence scale (this exercises directly in
    ``test_closed_reservoir_material_balance`` and
    ``test_initial_state_stationary``).
    """
    total_inj_rate = float(ctx.wells.injector_rate) if ctx.wells is not None else 0.0
    total_pv_bbl = float(np.sum(ctx.grid.bulk_volume) / 5.615 * np.mean(ctx.rock.phi0))
    pv_throughput_per_day = total_pv_bbl / 1000.0

    return max(total_inj_rate, pv_throughput_per_day, 1.0)


# ---------------------------------------------------------------------------
# Bound handling (README §5.4)
# ---------------------------------------------------------------------------
def _bound_violation_raw(x_raw: np.ndarray, p_ref: float) -> bool:
    """
    Cheap pre-clip check for "substantial" bound violation of the raw
    (undamped) trial state, used to decide whether to keep halving alpha
    before even bothering to clip and evaluate the residual. Thresholds are
    deliberately loose (order-of-magnitude overshoot) -- the hard clip in
    :func:`clip_state` handles the final, precise bound enforcement.
    """
    p = x_raw[0::3]
    sw = x_raw[1::3]
    sg = x_raw[2::3]
    return bool(
        np.any(p < -0.5 * p_ref)
        or np.any(sw < -0.5)
        or np.any(sw > 1.5)
        or np.any(sg < -0.5)
        or np.any(sg > 1.5)
    )


def clip_state(x: np.ndarray, ctx: ReservoirModel, params: NewtonParams) -> np.ndarray:
    """
    Hard physical-bound enforcement (README §5.4):

        p    >= p_min
        0    <= Sw <= 1
        0    <= Sg <= 1
        Sw + Sg <= 1 - S_or   (proportional rescale of Sw, Sg if violated)

    This is applied *after* the damping search settles on a step length, as
    a final safety net. It is a pragmatic engineering choice (not
    thermodynamically derived) -- documented as such per README §5.4.
    """
    x_clipped = x.copy()
    p = x_clipped[0::3]
    sw = x_clipped[1::3]
    sg = x_clipped[2::3]

    np.clip(p, params.p_min, None, out=p)
    np.clip(sw, 0.0, 1.0, out=sw)
    np.clip(sg, 0.0, 1.0, out=sg)

    sor = float(ctx.relperm.cfg.sor)
    max_total = 1.0 - sor
    total = sw + sg
    over = total > max_total
    if np.any(over):
        safe_total = np.where(total > 1.0e-12, total, 1.0)
        factor = np.where(over, max_total / safe_total, 1.0)
        sw *= factor
        sg *= factor

    x_clipped[0::3] = p
    x_clipped[1::3] = sw
    x_clipped[2::3] = sg
    return x_clipped


def _damped_trial(
    x: np.ndarray,
    dx: np.ndarray,
    res_norm_current: float,
    x_prev: np.ndarray,
    ctx: ReservoirModel,
    dt: float,
    params: NewtonParams,
):
    """
    Backtracking damping search (README §5.4): halve alpha (floor
    ``params.alpha_min``) while the raw update substantially violates
    bounds, or while the resulting (clipped) residual norm exceeds the
    current one. Accept once neither condition holds, or once the floor is
    reached (the floor is always accepted, matching "halve (min 0.1)").

    Returns
    -------
    x_clip, R_trial, res_norm_trial, alpha_used
    """
    p_ref = float(getattr(ctx.pvt, "p_ref", 4000.0))
    alpha = 1.0
    while True:
        x_raw = x + alpha * dx
        violation = _bound_violation_raw(x_raw, p_ref)
        x_clip = clip_state(x_raw, ctx, params)
        R_trial = compute_residual(ctx, x_clip, accumulation(ctx, x_prev), dt)
        res_norm_trial = float(np.max(np.abs(R_trial)))
        residual_increased = res_norm_trial > res_norm_current

        if (not violation and not residual_increased) or alpha <= params.alpha_min + 1.0e-12:
            return x_clip, R_trial, res_norm_trial, alpha

        alpha = max(alpha * 0.5, params.alpha_min)


# ---------------------------------------------------------------------------
# Main Newton loop
# ---------------------------------------------------------------------------
def newton_solve(
    x0: np.ndarray,
    x_prev: np.ndarray,
    ctx: ReservoirModel,
    dt: float,
    params: NewtonParams,
    jacobian_method: str = "analytical",
) -> NewtonResult:
    """
    Solve the nonlinear backward-Euler system for one time step of size
    ``dt``, starting from initial Newton guess ``x0`` (README §5.4).

    Parameters
    ----------
    x0 : ndarray, shape (3*nc,)
        Newton initial guess (either the previous converged state -- the
        default "warm start" -- or an RF-surrogate prediction; see
        README §9).
    x_prev : ndarray, shape (3*nc,)
        Converged state at time level n (fixed data defining the backward-
        Euler accumulation term).
    ctx : ReservoirModel
    Bundled grid/rock/pvt/relperm/wells model.
    dt : float
        Time-step size [days].
    params : NewtonParams
        Solver tolerances / iteration limits.
    jacobian_method : {"analytical", "numerical"}
        Passed through to ``jacobian.build_jacobian``; "analytical" for all
        production use, "numerical" only for small-grid debugging.

    Returns
    -------
    NewtonResult
    """
    x = x0.copy()
    nc = ctx.grid.n_cells
    p_ref = float(ctx.pvt.cfg.p_ref)
    m_old = accumulation(ctx, x_prev)
    scale = residual_scales(ctx, m_old, dt)

    residual_history: List[float] = []
    scaled_residual_history: List[float] = []
    alpha_history: List[float] = []

    R = compute_residual(ctx, x, m_old, dt)
    res_norm = scaled_residual_norm(R, scale)
    residual_history.append(res_norm)
    scaled_residual_history.append(res_norm)

    n_iter = 0
    converged = res_norm < params.tol

    while (not converged) and (n_iter < params.max_iter):
        n_iter += 1

        J = build_jacobian(ctx, x, m_old, dt, method=jacobian_method)

        try:
            dx = spsolve(J.tocsc(), -R)
        except Exception:
            # Singular / near-singular Jacobian: signal non-convergence so
            # the caller can cut dt and retry (README §5.4).
            break

        if dx is None or (not np.all(np.isfinite(dx))):
            break

        x_new, R_new, res_norm_new, alpha = _damped_trial(
            x, dx, res_norm, x_prev, ctx, dt, params
        )
        alpha_history.append(alpha)

        p_old, sw_old, sg_old = unpack_state(x)
        p_new, sw_new, sg_new = unpack_state(x_new)

        max_dp_rel = float(np.max(np.abs(p_new - p_old)) / p_ref)
        max_ds = float(
            max(np.max(np.abs(sw_new - sw_old)), np.max(np.abs(sg_new - sg_old)))
        )

        x, R, res_norm = x_new, R_new, res_norm_new
        residual_history.append(res_norm)
        scaled_res = scaled_residual_norm(R, scale)
        scaled_residual_history.append(scaled_res)

        converged = (
            (scaled_res < params.tol)
            and (max_dp_rel < params.dp_rel_tol)
            and (max_ds < params.ds_tol)
        )

    return NewtonResult(
        x=x,
        converged=converged,
        n_iterations=n_iter,
        residual_history=residual_history,
        scaled_residual_history=scaled_residual_history,
        final_scaled_residual=scaled_residual_history[-1],
        alpha_history=alpha_history,
    )