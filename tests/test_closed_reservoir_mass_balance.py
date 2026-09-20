"""
tests/test_closed_reservoir_material_balance.py

README §8 test 2: for a closed reservoir (no wells) with a heterogeneous
(fractal) porosity/permeability field and a non-trivial initial pressure
perturbation (so internal flow actually occurs while the pressure
equilibrates), the total in-place water, oil, and gas (including
dissolved gas) must be conserved to within a relative change of < 1e-8
over 200 days.

A tight Newton tolerance (1e-10, well below the production default of
1e-6) is used deliberately here: this test validates the EXACT algebraic
conservation property of the discretization (the antisymmetric face-flux
assembly R_L += q, R_R -= q in residual.py/jacobian.py), which holds to
within the summed Newton residual tolerance over all cells. The
production default tolerance (1e-6, scaled) is validated separately for
iteration-count performance (test_newton_convergence.py).
"""
import numpy as np

from residual import pack_state
from newton_solver import newton_solve, NewtonParams
from fractal_porosity import generate_porosity
from config import FractalConfig, RockConfig
from rock import permeability_from_porosity
from postprocess import compute_fluid_in_place
from conftest import make_context


def test_closed_reservoir_conserves_mass():
    nx, ny = 8, 8
    phi = generate_porosity(nx, ny, FractalConfig(beta=3.0, seed=42))
    k = permeability_from_porosity(phi, RockConfig())

    ctx = make_context(nx=nx, ny=ny, phi=phi, k=k)
    nc = ctx.grid.n_cells

    rng = np.random.default_rng(7)
    p0 = np.clip(4000.0 + 100.0 * rng.standard_normal(nc), 3500.0, 4500.0)
    sw0 = np.full(nc, 0.25)
    sg0 = np.full(nc, 0.0)
    x = pack_state(p0, sw0, sg0)

    fip_initial = compute_fluid_in_place(x, ctx)

    tight_params = NewtonParams(tol=1e-10, max_iter=20)
    dt = 10.0
    n_steps = 20  # 200 days
    for _ in range(n_steps):
        result = newton_solve(x, x, ctx, dt, tight_params)
        assert result.converged, "Newton failed to converge on closed-reservoir equilibration step"
        x = result.x

    fip_final = compute_fluid_in_place(x, ctx)

    for key in ("oil_stb", "water_stb", "gas"):
        initial = fip_initial[key]
        final = fip_final[key]
        rel_change = abs(final - initial) / max(abs(initial), 1e-12)
        assert rel_change < 1e-8, f"{key} relative change {rel_change:.3e} exceeds 1e-8"