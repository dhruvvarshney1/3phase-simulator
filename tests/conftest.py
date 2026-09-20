"""
tests/conftest.py
==================

Shared pytest fixtures and a minimal model-construction helper used
across the validation suite (README §8). ``make_context`` builds a
self-contained ``ReservoirModel`` from the current configuration and
property modules so that
unit-level tests (material balance, Jacobian, 1D Darcy, stationary state)
do not need to go through the full ``config.py`` / ``simulator.py``
machinery. Higher-level tests exercise ``simulator.run_simulation`` end to end.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Ensure the project root is importable when running `pytest` from any cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import default_config, override
from residual import ReservoirModel, build_model


def make_context(
    nx: int,
    ny: int,
    nz: int = 1,
    dx: float = 50.0,
    dy: float = 50.0,
    dz: float = 20.0,
    phi=0.18,
    k=100.0,
    pvt=None,
    relperm=None,
    wells=None,
    compressible: bool = False,
) -> ReservoirModel:
    """Build a small current-API model for focused tests."""
    cfg = default_config()
    cfg = override(cfg, "grid", nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)
    cfg = override(cfg, "rock", use_fractal=False,
                   phi_const=float(phi) if np.isscalar(phi) else 0.18,
                   pressure_dependent=compressible,
                   k_ref=float(k) if np.isscalar(k) else 100.0)
    cfg = override(cfg, "wells", producer_ij=(nx - 3, ny - 3), injector_ij=(2, 2))
    return build_model(cfg, phi_field=None if np.isscalar(phi) else np.asarray(phi),
                       with_wells=wells is not None)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)