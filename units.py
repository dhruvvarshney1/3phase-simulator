"""units.py -- Petroleum field-unit conversion constants and helpers.

Every conversion factor used anywhere in the simulator is defined here, with
its derivation. Nothing else in the code base may hard-code 0.001127, 5.615
or 2*pi.

Unit system (petroleum field units)
-----------------------------------
    pressure        psi
    permeability    md
    length          ft
    time            day
    viscosity       cp
    volumes         bbl (reservoir), STB (stock-tank barrel), SCF (gas)
    rates           STB/day (oil, water), SCF/day (gas)

Derivation of the Darcy constant (README Eq. 5.1)
-------------------------------------------------
Darcy's law in SI:  q [m^3/s] = k[m^2] * A[m^2] * dp[Pa] / (mu[Pa.s] * L[m]).

For k in md, A in ft^2, dp in psi, mu in cp and L in ft, the factor that
converts the product ``k*A*dp/(mu*L)`` to ft^3/day is

    1 md  = 9.869233e-16 m^2
    1 ft  = 0.3048 m
    1 psi = 6894.757 Pa
    1 cp  = 1.0e-3 Pa.s
    1 day = 86400 s
    1 ft^3 = 0.0283168 m^3

    (9.869233e-16 * 0.3048 * 6894.757 / 1.0e-3) * 86400 / 0.0283168
        = 0.0063283 ft^3/day per (md * ft * psi / cp)

Dividing by 5.615 ft^3/bbl gives 0.0011271 bbl/day, i.e. the familiar
0.001127. Using the *same* two constants (0.001127 in fluxes, 5.615 in
accumulation) everywhere keeps flux and accumulation dimensionally
consistent, which is what makes discrete material balance exact.

Radial flow (Peaceman well index) uses the same factor, times 2*pi:

    q = 0.001127 * 2*pi * k * h / (mu * (ln(re/rw) + s)) * dp

Run ``python units.py`` to print the derivation table and self-check.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# SI reference values used only to *derive/verify* the field-unit constants
# ---------------------------------------------------------------------------
MD_TO_M2: float = 9.869233e-16
FT_TO_M: float = 0.3048
PSI_TO_PA: float = 6894.757293168
CP_TO_PA_S: float = 1.0e-3
DAY_TO_S: float = 86400.0
BBL_TO_M3: float = 0.158987294928
FT3_TO_M3: float = FT_TO_M ** 3
BBL_FT3_EXACT: float = BBL_TO_M3 / FT3_TO_M3  # ~5.614583 ft^3/bbl

# ---------------------------------------------------------------------------
# Constants used by the simulator (specification values)
# ---------------------------------------------------------------------------
DARCY_FT3_PER_DAY: float = 0.006328   # ft^3/day per (md*ft*psi/cp)
BBL_FT3: float = 5.615                # ft^3 per bbl (pore-volume conversion)
DARCY_BBL_PER_DAY: float = 0.001127   # bbl/day per (md*ft*psi/cp) = 0.006328/5.615
TWO_PI: float = 2.0 * math.pi
WELL_INDEX_CONST: float = DARCY_BBL_PER_DAY * TWO_PI  # radial-flow constant


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------
def derive_darcy_ft3_per_day() -> float:
    """Derive the Darcy constant in ft^3/day from SI reference values."""
    m3_per_s = MD_TO_M2 * FT_TO_M * PSI_TO_PA / CP_TO_PA_S
    return m3_per_s * DAY_TO_S / FT3_TO_M3


def derive_darcy_bbl_per_day() -> float:
    """Derive the Darcy constant in bbl/day from SI reference values."""
    return derive_darcy_ft3_per_day() / BBL_FT3_EXACT


# ---------------------------------------------------------------------------
# Helper functions (vectorized)
# ---------------------------------------------------------------------------
def transmissibility_geom(k_md: np.ndarray | float,
                          area_ft2: np.ndarray | float,
                          length_ft: np.ndarray | float) -> np.ndarray | float:
    """Return T = 0.001127 * k * A / L  [bbl*cp/(day*psi)]  (README Eq. 5.2).

    Multiplying T by (dp / mu) gives the flux in bbl/day; multiplying by
    (k_r / (mu * B)) * dp gives the surface flux in STB/day.
    """
    return DARCY_BBL_PER_DAY * np.asarray(k_md) * np.asarray(area_ft2) / np.asarray(length_ft)


def darcy_flux_bbl_per_day(k_md: np.ndarray | float,
                           area_ft2: np.ndarray | float,
                           length_ft: np.ndarray | float,
                           dp_psi: np.ndarray | float,
                           mu_cp: np.ndarray | float) -> np.ndarray | float:
    """Return q [bbl/day] = 0.001127 * k * A / L * dp / mu  (README Eq. 5.1)."""
    return transmissibility_geom(k_md, area_ft2, length_ft) * np.asarray(dp_psi) / np.asarray(mu_cp)


def pore_volume_bbl(bulk_volume_ft3: np.ndarray | float,
                    porosity: np.ndarray | float) -> np.ndarray | float:
    """Return pore volume in bbl = Vb[ft^3] * phi / 5.615  (README Eq. 5.3)."""
    return np.asarray(bulk_volume_ft3) * np.asarray(porosity) / BBL_FT3


def stb_in_place(bulk_volume_ft3: np.ndarray | float,
                 porosity: np.ndarray | float,
                 saturation: np.ndarray | float,
                 formation_volume_factor: np.ndarray | float) -> np.ndarray | float:
    """Return in-place surface volume M = (Vb/5.615) * phi * S / B  [STB].

    For free gas the result is in (bbl-equivalent of surface gas)/B_g units
    consistent with the simulator's B_g [bbl/STB-equivalent] convention.
    """
    return pore_volume_bbl(bulk_volume_ft3, porosity) * np.asarray(saturation) / np.asarray(
        formation_volume_factor)


def well_index_radial(k_md: float, h_ft: float, r_e_ft: float,
                      r_w_ft: float, skin: float = 0.0) -> float:
    """Return the Peaceman well index WI [bbl*cp/(day*psi)] (README Eq. 6.1).

    WI = 0.001127 * 2*pi * k * h / (ln(r_e / r_w) + skin)
    """
    return WELL_INDEX_CONST * k_md * h_ft / (math.log(r_e_ft / r_w_ft) + skin)


# ---------------------------------------------------------------------------
# Self-consistency check
# ---------------------------------------------------------------------------
def unit_consistency_report() -> Dict[str, float]:
    """Return relative errors that certify the constants are consistent.

    Keys
    ----
    darcy_ft3_rel_err     : 0.006328 vs SI-derived value
    bbl_ft3_rel_err       : 5.615 vs exact ft^3/bbl
    darcy_bbl_rel_err     : 0.001127 vs SI-derived value
    flux_vs_accum_rel_err : (0.006328 / 5.615) vs 0.001127, i.e. flux and
                            accumulation constants agree with each other
    well_index_rel_err    : WELL_INDEX_CONST vs 2*pi*0.001127
    """
    darcy_ft3 = derive_darcy_ft3_per_day()
    darcy_bbl = derive_darcy_bbl_per_day()
    return {
        "darcy_ft3_rel_err": abs(DARCY_FT3_PER_DAY - darcy_ft3) / darcy_ft3,
        "bbl_ft3_rel_err": abs(BBL_FT3 - BBL_FT3_EXACT) / BBL_FT3_EXACT,
        "darcy_bbl_rel_err": abs(DARCY_BBL_PER_DAY - darcy_bbl) / darcy_bbl,
        "flux_vs_accum_rel_err": abs(DARCY_FT3_PER_DAY / BBL_FT3 - DARCY_BBL_PER_DAY)
        / DARCY_BBL_PER_DAY,
        "well_index_rel_err": abs(WELL_INDEX_CONST - TWO_PI * DARCY_BBL_PER_DAY)
        / WELL_INDEX_CONST,
    }


def assert_unit_consistency(rtol: float = 1.0e-3) -> None:
    """Raise AssertionError if any constant deviates from its derivation by > rtol."""
    for name, err in unit_consistency_report().items():
        assert err < rtol, f"unit constant check failed: {name} = {err:.3e} >= {rtol:.1e}"


def main() -> None:
    """Print the constant table with derived values and run the self-check."""
    print("Unit constants (petroleum field units)")
    print("-" * 78)
    print(f"{'constant':<26}{'used':>14}{'derived':>16}{'meaning':>22}")
    print(f"{'DARCY_FT3_PER_DAY':<26}{DARCY_FT3_PER_DAY:>14.6f}"
          f"{derive_darcy_ft3_per_day():>16.6f}{'ft^3/day':>22}")
    print(f"{'BBL_FT3':<26}{BBL_FT3:>14.6f}{BBL_FT3_EXACT:>16.6f}{'ft^3/bbl':>22}")
    print(f"{'DARCY_BBL_PER_DAY':<26}{DARCY_BBL_PER_DAY:>14.6f}"
          f"{derive_darcy_bbl_per_day():>16.6f}{'bbl/day':>22}")
    print(f"{'TWO_PI':<26}{TWO_PI:>14.6f}{2 * math.pi:>16.6f}{'radial flow':>22}")
    print(f"{'WELL_INDEX_CONST':<26}{WELL_INDEX_CONST:>14.6f}"
          f"{TWO_PI * derive_darcy_bbl_per_day():>16.6f}{'bbl/day (radial)':>22}")
    print("-" * 78)
    for name, err in unit_consistency_report().items():
        print(f"{name:<26}{err:>14.3e}")
    assert_unit_consistency()
    print("Unit consistency check: PASSED")


if __name__ == "__main__":
    main()