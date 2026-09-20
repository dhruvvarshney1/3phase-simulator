"""Validate the analytic sparse Jacobian against finite differences."""

import numpy as np

from conftest import make_context
from jacobian import build_jacobian, jacobian_relative_error
from residual import accumulation, pack_state


def test_analytic_jacobian_matches_numerical_without_wells():
    context = make_context(nx=3, ny=2, wells=None)
    cells = context.grid.n_cells
    state = pack_state(
        np.linspace(3200.0, 3600.0, cells),
        np.full(cells, 0.30),
        np.full(cells, 0.08),
    )
    old_mass = accumulation(context, state)
    analytic = build_jacobian(context, state, old_mass, 2.0, method="analytical")
    numerical = build_jacobian(context, state, old_mass, 2.0, method="numerical")
    assert jacobian_relative_error(analytic, numerical) < 1.0e-5


def test_analytic_jacobian_matches_numerical_with_wells():
    context = make_context(nx=6, ny=6, wells=True)
    cells = context.grid.n_cells
    state = pack_state(
        np.linspace(2800.0, 4200.0, cells),
        np.full(cells, 0.32),
        np.full(cells, 0.07),
    )
    old_mass = accumulation(context, state)
    analytic = build_jacobian(context, state, old_mass, 1.0, method="analytical")
    numerical = build_jacobian(context, state, old_mass, 1.0, method="numerical")
    assert jacobian_relative_error(analytic, numerical) < 1.0e-5
