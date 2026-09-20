"""Small end-to-end waterflood physics checks."""

import numpy as np

from config import default_config, override
from simulator import run_simulation


def test_small_waterflood_injects_water_and_changes_saturation():
    cfg = default_config()
    cfg = override(cfg, "grid", nx=6, ny=6)
    cfg = override(cfg, "rock", use_fractal=False, phi_const=0.18)
    cfg = override(cfg, "wells", injector_ij=(1, 1), producer_ij=(4, 4))
    cfg = override(cfg, "timestep", t_end=20.0,
                   stages=((0.0, 5.0, 1.0, 1.0), (5.0, 20.0, 5.0, 5.0)),
                   report_times=(5.0, 20.0))
    output = run_simulation(cfg, verbose=False)

    initial = output["snapshots"][0.0]
    final = output["summary"]["final_state"]
    context = output["context"]
    from residual import unpack_state

    pressure, water, gas = unpack_state(final)
    initial_water = initial["Sw"]
    injector = context.wells.injector_cell
    assert output["timeseries"]["cum_water_inj_stb"].iloc[-1] > 0.0
    assert water[injector] > initial_water[injector]
    assert np.all(np.isfinite(pressure))
    assert np.all(np.isfinite(gas))
    assert np.all(output["timeseries"]["n_newton_iter"] <= cfg.newton.max_iter)
