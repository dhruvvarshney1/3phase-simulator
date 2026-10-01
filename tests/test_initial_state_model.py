"""Simulator driven by the ``_initial_state_`` model: gravity, faults, anisotropy,
per-cell S_wc and multi-perforation wells."""
import dataclasses

import numpy as np
import pytest

import initial_state_model as ism
from jacobian import analytical_jacobian, jacobian_relative_error, numerical_jacobian
from residual import accumulation, evaluate_residual


def small_model():
    cfg = ism.isr.ReservoirConfig(
        Nx=6, Ny=5, Nz=4, Lx=150.0, Ly=125.0, Lz=20.0, layer_sequence=[0, 2, 2, 1],
        dome_center=(75.0, 60.0), channels=[], fractures=[
            dict(name="FR", xc=75.0, yc=60.0, strike_deg=30.0, length=80.0, width=30.0, layers=[1, 2],
                 aperture_mm=0.5, k_mult_along=20.0, k_mult_across=3.0, kz_mult=5.0)],
        faults=[dict(name="F", x0=80.0, y0=0.0, z0=2000.0, strike_deg=80.0, dip_deg=75.0,
                     curvature=0.0, trans_mult=0.05, thickness=5.0)],
        owc_depth=2014.0, depth_ref=2014.0, p_ref=17.0e6,
        wells=[dict(name="I", type="injector", x=20.0, y=20.0, control="rate", value=50.0,
                    units="m3/day", rw=0.1, skin=0.0, perforate="reservoir"),
               dict(name="P", type="producer", x=130.0, y=105.0, control="bhp", value=15.0e6,
                    units="Pa", rw=0.1, skin=0.0, perforate="reservoir")])
    return ism.build_from_initial_state(ism.isr.build_reservoir(cfg))


def test_initial_oil_leg_is_static():
    model, x0 = small_model()
    closed = dataclasses.replace(model, wells=None)
    ev = evaluate_residual(closed, x0, accumulation(closed, x0), 1.0)
    assert np.max(np.abs(ev.fluxes.q[:, 1])) < 1e-8


@pytest.mark.parametrize("dp_noise", [150.0, 2.0])   # 2 psi < gravity head: phases upwind differently
def test_jacobian_matches_numerical_with_gravity_and_wells(dp_noise):
    model, x0 = small_model()
    assert model.wells.producer_cells.size > 1
    rng = np.random.default_rng(0)
    n = model.grid.n_cells
    x = x0.copy()
    x[0::3] = x0[0::3] - 150.0 + rng.uniform(-dp_noise, dp_noise, n)   # hydrostatic, below p_bubble
    ups = __import__("residual").phase_upstream(model, x[0::3])
    if dp_noise < 5.0:
        assert np.any(ups[0].left_is_up != ups[1].left_is_up)
    x[1::3] = rng.uniform(0.3, 0.6, n)
    x[2::3] = rng.uniform(0.06, 0.12, n)
    m_old = accumulation(model, x0)
    err = jacobian_relative_error(analytical_jacobian(model, x, m_old, 5.0),
                                  numerical_jacobian(model, x, m_old, 5.0))
    assert err < 1e-5, err   # same tolerance as test_jacobian_matching_numerical (FD across upwind kinks)
