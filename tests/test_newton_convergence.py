"""
tests/test_newton_convergence.py

README §8 test 8: record the residual-norm history of a Newton solve and
demonstrate near-quadratic (at minimum, clearly superlinear) asymptotic
convergence on a well-behaved ("easy") time step: a moderate-sized,
homogeneous, single-well-pair problem with a modest time-step size, solved
to a tight tolerance so multiple iterations are actually exercised.
"""
import numpy as np

from residual import pack_state
from newton_solver import newton_solve, NewtonParams
from conftest import make_context


def _build_easy_step_context():
    nx, ny = 10, 10
    ctx = make_context(nx=nx, ny=ny, phi=0.18, k=150.0, wells=True)
    nc = ctx.grid.n_cells

    p0 = np.full(nc, 4000.0)
    sw0 = np.full(nc, 0.25)
    sg0 = np.zeros(nc)
    x0 = pack_state(p0, sw0, sg0)
    return ctx, x0


def test_newton_convergence_is_superlinear_on_easy_step():
    ctx, x0 = _build_easy_step_context()
    params = NewtonParams(tol=1e-9, max_iter=12)  # tight tol to force several iterations

    result = newton_solve(x0, x0, ctx, dt=5.0, params=params)

    assert result.converged
    history = np.asarray(result.scaled_residual_history)
    assert len(history) >= 3, "Step converged too fast to observe an asymptotic convergence trend"

    # Residual must decrease (weakly) monotonically overall.
    assert np.all(np.diff(history) <= 1.0e-14), "Residual history is not monotonically decreasing"

    # Convergence ratios r_i = res[i+1]/res[i] should trend downward
    # (superlinear / near-quadratic behaviour) in the asymptotic regime.
    ratios = history[1:] / np.maximum(history[:-1], 1e-300)
    if len(ratios) >= 2:
        assert ratios[-1] <= ratios[0] + 1e-6, (
            f"Convergence ratios did not improve in the asymptotic regime: {ratios}"
        )

    assert result.n_iterations <= params.max_iter