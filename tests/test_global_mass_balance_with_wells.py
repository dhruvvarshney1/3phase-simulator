"""
tests/test_global_mass_balance_with_wells.py

README §8 test 9 / §12 acceptance checklist: over the full default MVP
waterflood run (30x30 grid, 1000 days, fractal heterogeneity, producer +
injector), for each conserved component:

    (in-place change) = (injected - produced)

with maximum relative error < 1e-6.

NOTE on tolerance: the per-step Newton tolerance is tightened to 1e-9 for
this specific test (vs. the README §12 production default of 1e-6, scaled
by the characteristic injection-rate/pore-volume-throughput scale in
newton_solver.compute_convergence_scale). The production default is tuned
for fast, robust field-scale time-stepping; the *aggregate* whole-run mass
balance error sums per-cell residuals (each individually bounded by
tol*scale) over hundreds of cells and time steps, so validating the exact
conservation PROPERTY of the discretization (the antisymmetric internal-
flux assembly in residual.py/jacobian.py) in isolation requires a tighter
per-step tolerance than the engineering-tuned production default.
Iteration-count performance at the production default tolerance is
validated separately (test_newton_convergence.py, run_mvp.py).
"""
import pytest

from config import default_config, override
from simulator import run_simulation
from postprocess import mass_balance_summary
from residual import pack_state


@pytest.mark.slow
def test_global_mass_balance_over_full_mvp_run():
    cfg = default_config()
    cfg = override(cfg, "newton", tol=1.0e-9, max_iter=20)

    out = run_simulation(cfg, init_guess_mode="previous", jacobian_method="analytical", verbose=False)

    ctx = out["context"]
    df = out["timeseries"]
    snapshots = out["snapshots"]

    s0 = snapshots[0.0]
    x_initial = pack_state(s0["p"], s0["Sw"], s0["Sg"])
    x_final = out["summary"]["final_state"]

    mb = mass_balance_summary(x_initial, x_final, ctx, df)

    worst = max(mb["oil_rel_error"], mb["water_rel_error"], mb["gas_rel_error"])
    assert worst < 1.0e-6, f"Global mass-balance relative error {worst:.3e} exceeds 1e-6: {mb}"

    assert df["n_newton_iter"].max() <= cfg.newton.max_iter