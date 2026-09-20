"""pvt.py -- Black-oil PVT functions with analytic pressure derivatives.

All functions are vectorized over pressure ``p`` [psi] (README section 4.4):

    B_o(p)  = bo_ref * exp(-bo_comp * (p - p_ref))          [bbl/STB]
    B_w(p)  = bw_ref * exp(-bw_comp * (p - p_ref))          [bbl/STB]
    B_g(p)  = bg_coeff * bg_p_ref / max(p, bg_p_floor)      [bbl/STB-equivalent]
    mu_o(p) = muo_ref * exp(-muo_slope * (p - p_ref))       [cp]
    mu_w, mu_g constant                                     [cp]
    R_s(p)  = rs_bubble                       for p >= p_bubble
            = rs_bubble * p / p_bubble        for p <  p_bubble   [SCF/STB]

Analytic derivatives (used by the analytical Jacobian):

    dB_o/dp  = -bo_comp * B_o           dB_w/dp = -bw_comp * B_w
    dB_g/dp  = -bg_coeff*bg_p_ref/p^2 for p > bg_p_floor, else 0
    dmu_o/dp = -muo_slope * mu_o
    dR_s/dp  = rs_bubble / p_bubble (= 0.24 SCF/STB/psi) below the bubble point, else 0

Note: R_s has a kink at the bubble point. At p == p_bubble the right-hand
derivative (0) is returned. Newton still converges across the kink because
the kink is a jump in the *derivative*, not in the function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import PVTConfig, default_config


@dataclass(frozen=True, eq=False)
class PVTProperties:
    """Bundle of PVT properties and their pressure derivatives (all arrays)."""

    bo: np.ndarray
    dbo_dp: np.ndarray
    bw: np.ndarray
    dbw_dp: np.ndarray
    bg: np.ndarray
    dbg_dp: np.ndarray
    mu_o: np.ndarray
    dmu_o_dp: np.ndarray
    mu_w: np.ndarray
    mu_g: np.ndarray
    rs: np.ndarray
    drs_dp: np.ndarray


class PVT:
    """Vectorized black-oil PVT model built from a :class:`PVTConfig`."""

    def __init__(self, cfg: PVTConfig) -> None:
        if min(cfg.bo_ref, cfg.bw_ref, cfg.bg_coeff, cfg.muo_ref, cfg.muw, cfg.mug) <= 0.0:
            raise ValueError("PVT reference values must be positive")
        if cfg.p_bubble <= 0.0 or cfg.bg_p_floor <= 0.0:
            raise ValueError("p_bubble and bg_p_floor must be positive")
        self.cfg = cfg

    # ---- oil ---------------------------------------------------------
    def bo(self, p: np.ndarray | float) -> np.ndarray:
        """Oil formation volume factor B_o(p) [bbl/STB]."""
        c = self.cfg
        return c.bo_ref * np.exp(-c.bo_comp * (np.asarray(p, dtype=float) - c.p_ref))

    def dbo_dp(self, p: np.ndarray | float) -> np.ndarray:
        """dB_o/dp [bbl/STB/psi]."""
        return -self.cfg.bo_comp * self.bo(p)

    def mu_o(self, p: np.ndarray | float) -> np.ndarray:
        """Oil viscosity mu_o(p) [cp]."""
        c = self.cfg
        return c.muo_ref * np.exp(-c.muo_slope * (np.asarray(p, dtype=float) - c.p_ref))

    def dmu_o_dp(self, p: np.ndarray | float) -> np.ndarray:
        """dmu_o/dp [cp/psi]."""
        return -self.cfg.muo_slope * self.mu_o(p)

    # ---- water -------------------------------------------------------
    def bw(self, p: np.ndarray | float) -> np.ndarray:
        """Water formation volume factor B_w(p) [bbl/STB]."""
        c = self.cfg
        return c.bw_ref * np.exp(-c.bw_comp * (np.asarray(p, dtype=float) - c.p_ref))

    def dbw_dp(self, p: np.ndarray | float) -> np.ndarray:
        """dB_w/dp [bbl/STB/psi]."""
        return -self.cfg.bw_comp * self.bw(p)

    def mu_w(self, p: np.ndarray | float) -> np.ndarray:
        """Water viscosity [cp] (constant, broadcast to the shape of ``p``)."""
        return np.full_like(np.asarray(p, dtype=float), self.cfg.muw)

    # ---- gas ---------------------------------------------------------
    def bg(self, p: np.ndarray | float) -> np.ndarray:
        """Gas formation volume factor B_g(p) with pressure floor [bbl/STB]."""
        c = self.cfg
        return c.bg_coeff * c.bg_p_ref / np.maximum(np.asarray(p, dtype=float), c.bg_p_floor)

    def dbg_dp(self, p: np.ndarray | float) -> np.ndarray:
        """dB_g/dp; zero on the flat (floored) part p <= bg_p_floor."""
        c = self.cfg
        pa = np.asarray(p, dtype=float)
        safe = np.maximum(pa, c.bg_p_floor)
        return np.where(pa > c.bg_p_floor, -c.bg_coeff * c.bg_p_ref / safe ** 2, 0.0)

    def mu_g(self, p: np.ndarray | float) -> np.ndarray:
        """Gas viscosity [cp] (constant, broadcast to the shape of ``p``)."""
        return np.full_like(np.asarray(p, dtype=float), self.cfg.mug)

    # ---- solution gas ------------------------------------------------
    def rs(self, p: np.ndarray | float) -> np.ndarray:
        """Solution GOR R_s(p) [SCF/STB]; saturated line below p_bubble."""
        c = self.cfg
        pa = np.asarray(p, dtype=float)
        return np.where(pa >= c.p_bubble, c.rs_bubble, c.rs_bubble * pa / c.p_bubble)

    def drs_dp(self, p: np.ndarray | float) -> np.ndarray:
        """dR_s/dp [SCF/STB/psi]: rs_bubble/p_bubble below bubble point, else 0."""
        c = self.cfg
        pa = np.asarray(p, dtype=float)
        return np.where(pa >= c.p_bubble, 0.0, c.rs_bubble / c.p_bubble)

    # ---- bundle ------------------------------------------------------
    def evaluate(self, p: np.ndarray | float) -> PVTProperties:
        """Evaluate every property and derivative at pressure ``p``."""
        pa = np.asarray(p, dtype=float)
        return PVTProperties(
            bo=self.bo(pa), dbo_dp=self.dbo_dp(pa),
            bw=self.bw(pa), dbw_dp=self.dbw_dp(pa),
            bg=self.bg(pa), dbg_dp=self.dbg_dp(pa),
            mu_o=self.mu_o(pa), dmu_o_dp=self.dmu_o_dp(pa),
            mu_w=self.mu_w(pa), mu_g=self.mu_g(pa),
            rs=self.rs(pa), drs_dp=self.drs_dp(pa),
        )


def main() -> None:
    """Print a property table and verify analytic derivatives by central differences."""
    pvt = PVT(default_config().pvt)
    p = np.array([500.0, 1500.0, 2400.0, 2600.0, 4000.0, 5000.0])
    props = pvt.evaluate(p)
    print(f"{'p':>7}{'Bo':>10}{'Bw':>10}{'Bg':>11}{'mu_o':>9}{'Rs':>9}{'dRs/dp':>9}")
    for k in range(p.size):
        print(f"{p[k]:>7.0f}{props.bo[k]:>10.5f}{props.bw[k]:>10.5f}{props.bg[k]:>11.6f}"
              f"{props.mu_o[k]:>9.4f}{props.rs[k]:>9.2f}{props.drs_dp[k]:>9.3f}")

    h = 1.0e-2
    pairs = [("bo", pvt.bo, pvt.dbo_dp), ("bw", pvt.bw, pvt.dbw_dp), ("bg", pvt.bg, pvt.dbg_dp),
             ("mu_o", pvt.mu_o, pvt.dmu_o_dp), ("rs", pvt.rs, pvt.drs_dp)]
    for name, f, df in pairs:
        fd = (f(p + h) - f(p - h)) / (2.0 * h)
        err = np.max(np.abs(fd - df(p)) / (np.abs(df(p)) + 1.0e-12))
        assert err < 1.0e-6, f"derivative of {name} inconsistent (rel err {err:.2e})"
        print(f"d{name}/dp analytic vs finite difference: max rel err = {err:.2e}")
    assert np.isclose(pvt.drs_dp(np.array([2000.0]))[0], 0.24)
    print("PVT self-checks: PASSED")


if __name__ == "__main__":
    main()