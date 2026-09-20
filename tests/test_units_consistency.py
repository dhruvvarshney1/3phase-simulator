"""
tests/test_units_consistency.py

README §8 test 1: validates that the field-unit conversion constants in
units.py are internally consistent and correctly reproduce the flux and
accumulation formulas of README §2 / §4.7 / §5.1:

    Flux:         q [bbl/day] = 0.001127 * k[md] * A[ft^2] / L[ft] * dp[psi] / mu[cp]
    Accumulation: pore volume [bbl] = Vb[ft^3] / 5.615
    Peaceman:     WI = 0.001127 * 2*pi * sqrt(kx*ky) * dz / (ln(re/rw) + skin)
"""
import numpy as np
import pytest

import units


def test_transmissibility_constant_value():
    """The Darcy transmissibility constant must equal 0.001127 (README §2)."""
    assert units.DARCY_BBL_PER_DAY == pytest.approx(0.001127, abs=1e-12)


def test_bbl_ft3_conversion_constant_value():
    """1 bbl = 5.615 ft^3 (README §2)."""
    assert units.BBL_FT3 == pytest.approx(5.615, abs=1e-12)


def test_flux_formula_dimensional_consistency():
    """
    Reproduce the Darcy flux formula (README §4.7) manually with the raw
    literal constant and cross-check against units.TRANS_CONST for a
    randomized set of (k, A, L, dp, mu) values.
    """
    rng = np.random.default_rng(0)
    k = rng.uniform(10, 500, size=20)      # md
    A = rng.uniform(500, 2000, size=20)    # ft^2
    L = rng.uniform(20, 200, size=20)      # ft
    dp = rng.uniform(-500, 500, size=20)   # psi
    mu = rng.uniform(0.3, 3.0, size=20)    # cp

    q_manual = 0.001127 * k * A / L * dp / mu
    q_from_units = units.DARCY_BBL_PER_DAY * k * A / L * dp / mu

    np.testing.assert_allclose(q_from_units, q_manual, rtol=1e-12)


def test_accumulation_formula_dimensional_consistency():
    """
    M = (Vb/5.615) * phi * S / B [STB] must use the same bbl<->ft^3
    conversion constant as the flux formula's companion accumulation term
    (README §2).
    """
    rng = np.random.default_rng(1)
    Vb = rng.uniform(1.0e4, 1.0e5, size=20)   # ft^3
    phi = rng.uniform(0.1, 0.3, size=20)
    S = rng.uniform(0.1, 0.9, size=20)
    B = rng.uniform(1.0, 1.3, size=20)

    M_manual = (Vb / 5.615) * phi * S / B
    M_from_units = (Vb / units.BBL_FT3) * phi * S / B

    np.testing.assert_allclose(M_from_units, M_manual, rtol=1e-12)


def test_peaceman_constant_consistency():
    """Peaceman well-index formula (README §6) uses 2*pi and the same
    0.001127 transmissibility constant; verify self-consistency."""
    two_pi = getattr(units, "TWO_PI", 2.0 * np.pi)
    assert two_pi == pytest.approx(2.0 * np.pi, rel=1e-12)

    kx, ky, dz_ = 150.0, 150.0, 20.0
    re, rw, skin = 0.14 * np.hypot(50.0, 50.0), 0.25, 0.0
    WI_manual = 0.001127 * 2.0 * np.pi * np.sqrt(kx * ky) * dz_ / (np.log(re / rw) + skin)
    WI_from_units = units.DARCY_BBL_PER_DAY * two_pi * np.sqrt(kx * ky) * dz_ / (np.log(re / rw) + skin)
    assert WI_from_units == pytest.approx(WI_manual, rel=1e-12)