"""Small, direct regressions for the synthetic-to-flow adapter."""

import dataclasses

import numpy as np

import initial_state_model as ism
from residual import accumulation, evaluate_residual


def _case(fractures=None):
    cfg = _case_config(fractures)
    return ism.build_from_initial_state(ism.isr.build_reservoir(cfg))


def test_geological_ordering_and_initial_state_are_used():
    model, x0 = _case()
    res = ism.isr.build_reservoir(_case_config())
    for i, j, k in ((0, 0, 0), (3, 2, 1), (5, 4, 3)):
        c = model.grid.cell_index(i, j, k)
        assert np.isclose(x0[3 * c] * ism.PSI_TO_PA, res["initial_state"]["pressure"][i, j, k])
        assert np.isclose(model.grid.depth[c] * ism.FT_TO_M, res["grid"]["depth"][i, j, k])


def _case_config(fractures=None):
    cfg = ism.isr.ReservoirConfig(Nx=6, Ny=5, Nz=4, Lx=150.0, Ly=125.0, Lz=20.0,
                                  layer_sequence=[0, 2, 2, 1], channels=[],
                                  fractures=fractures if fractures is not None else [
                                      dict(name="FR", xc=75.0, yc=60.0, strike_deg=30.0, length=80.0, width=30.0,
                                           layers=[1, 2], aperture_mm=0.5, k_mult_along=20.0,
                                           k_mult_across=3.0, kz_mult=5.0)],
                                  faults=[dict(name="F", x0=80.0, y0=0.0, z0=2000.0, strike_deg=80.0,
                                               dip_deg=75.0, curvature=0.0, trans_mult=0.05, thickness=5.0)],
                                  owc_depth=2014.0, depth_ref=2014.0, p_ref=17.0e6,
                                  wells=[dict(name="I", type="injector", x=20.0, y=20.0, control="rate", value=50.0,
                                              units="m3/day", rw=0.1, skin=0.0, perforate="reservoir"),
                                         dict(name="P", type="producer", x=130.0, y=105.0, control="bhp", value=15.0e6,
                                              units="Pa", rw=0.1, skin=0.0, perforate="reservoir")])
    return cfg


def test_directional_fault_and_fracture_properties_change_flux():
    model, x0 = _case()
    assert not np.allclose(model.rock.directional_perm[:, 0], model.rock.directional_perm[:, 1])
    assert not np.allclose(model.rock.directional_perm[:, 0], model.rock.directional_perm[:, 2])
    p = x0[0::3].copy()
    p[:] = np.linspace(1800.0, 1200.0, model.grid.n_cells)
    x = x0.copy(); x[0::3] = p
    base = evaluate_residual(dataclasses.replace(model, wells=None), x,
                             accumulation(dataclasses.replace(model, wells=None), x), 1.0)
    no_fault = dataclasses.replace(model, trans=model.trans.copy())
    fault_faces = model.trans < 0.05 * model.trans.max()
    assert np.any(fault_faces)
    no_fault = dataclasses.replace(no_fault, trans=np.where(fault_faces, model.trans / 0.05, model.trans))
    open_flux = evaluate_residual(dataclasses.replace(no_fault, wells=None), x,
                                  accumulation(dataclasses.replace(no_fault, wells=None), x), 1.0)
    assert not np.allclose(base.fluxes.q, open_flux.fluxes.q)

    matrix, _ = _case(fractures=[])
    assert np.any(model.trans != matrix.trans)


def test_multiperforation_wells_contribute_and_allocate_injection():
    model, x0 = _case()
    assert model.wells.producer_cells.size > 1
    assert model.wells.injector_cells.size > 1
    assert np.isclose(model.wells.inj_frac.sum(), 1.0)
    ev = evaluate_residual(model, x0, accumulation(model, x0), 1.0)
    assert np.isclose(ev.source[:, 0].sum(), model.wells.injector_rate - ev.terms.rates.q_w)
