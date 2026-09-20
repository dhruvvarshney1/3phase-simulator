"""
tests/test_gas_liberation.py

README §8 test 7: with no injector and a producer BHP set well below the
bubble-point pressure (2500 psi, README §4.4/§12), and no pressure support,
the reservoir must decline below the bubble point (at least locally near
the producer). Below the bubble point Rs(p) decreases, free gas must
appear (Sg > 0), and the produced GOR must rise above the initial
solution GOR (Rs at 4000 psi = 600 SCF/STB, since 4000 > 2500 psi).
"""
import numpy as np
import pytest

from config import smoke_test_config, override
from simulator import run_simulation


@pytest.fixture(scope="module")
def depletion_result():
    cfg = smoke_test_config()
    cfg = override(cfg, "rock", use_fractal=False, phi_const=0.18)
    cfg = override(cfg, "wells", injector_rate=0.0, producer_bhp=1000.0)
    cfg = override(cfg, "timestep", t_end=365.0, stages=((0.0, 30.0, 1.0, 1.0), (30.0, 180.0, 5.0, 5.0), (180.0, 365.0, 10.0, 30.0)), report_times=(30.0, 180.0, 365.0))

    out = run_simulation(cfg, init_guess_mode="previous", jacobian_method="analytical", verbose=False)
    return out


def test_free_gas_appears_below_bubble_point(depletion_result):
    snapshots = depletion_result["snapshots"]
    t_final = max(snapshots.keys())
    sg_final = snapshots[t_final]["Sg"]
    assert np.max(sg_final) > 1.0e-6, "No free gas liberated despite depletion below bubble point"


def test_produced_gor_exceeds_initial_solution_gor(depletion_result):
    df = depletion_result["timeseries"]
    initial_rs = 600.0  # README §4.4/§12: Rs at p >= 2500 psi bubble point
    assert df["gor"].max() > initial_rs, "Produced GOR never exceeded the initial solution GOR"


def test_pressure_declines_below_bubble_point_near_producer(depletion_result):
    ctx = depletion_result["context"]
    snapshots = depletion_result["snapshots"]
    t_final = max(snapshots.keys())
    prod_cell = ctx.wells.producer_cell
    p_final_prod = snapshots[t_final]["p"][prod_cell]
    assert p_final_prod < 2500.0, "Producer-cell pressure never dropped below the bubble point"