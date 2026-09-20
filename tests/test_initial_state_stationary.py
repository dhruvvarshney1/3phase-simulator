"""
tests/test_initial_state_stationary.py

README §8 test 3: for a spatially uniform initial state (constant p, Sw,
Sg) with no wells, the residual must be identically zero -- there is no
pressure gradient (hence no flux) and no accumulation change (x == x_prev,
no wells). Newton must therefore "converge" in zero iterations, since the
initial residual already satisfies the convergence tolerance.
"""
import numpy as np

from residual import accumulation, pack_state, compute_residual
from newton_solver import newton_solve, NewtonParams
from conftest import make_context


def test_stationary_state_zero_residual():
    ctx = make_context(nx=6, ny=6, phi=0.18, k=100.0)
    nc = ctx.grid.n_cells
    p0 = np.full(nc, 4000.0)
    sw0 = np.full(nc, 0.25)
    sg0 = np.full(nc, 0.0)
    x0 = pack_state(p0, sw0, sg0)

    R = compute_residual(ctx, x0, accumulation(ctx, x0), dt=1.0)
    np.testing.assert_allclose(R, 0.0, atol=1e-9)


def test_stationary_state_newton_converges_immediately():
    ctx = make_context(nx=6, ny=6, phi=0.18, k=100.0)
    nc = ctx.grid.n_cells
    p0 = np.full(nc, 4000.0)
    sw0 = np.full(nc, 0.25)
    sg0 = np.full(nc, 0.0)
    x0 = pack_state(p0, sw0, sg0)

    result = newton_solve(x0, x0, ctx, dt=1.0, params=NewtonParams())
    assert result.converged
    assert result.n_iterations == 0
    np.testing.assert_allclose(result.x, x0, atol=1e-12)