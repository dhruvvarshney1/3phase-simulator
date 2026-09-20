"""relperm.py -- Corey three-phase relative permeability with analytic derivatives.

Model (README section 4.5)
--------------------------
Normalized (effective) saturations, with D = 1 - S_wc - S_or - S_gc:

    S_we = clip((S_w - S_wc) / D, 0, 1)
    S_ge = clip((S_g - S_gc) / D, 0, 1)
    S_oe = clip(1 - S_we - S_ge, 0, 1)

    k_rw = krw_max * S_we^nw       (default 0.30 * S_we^3)
    k_ro = kro_max * S_oe^no       (default 0.90 * S_oe^2)
    k_rg = krg_max * S_ge^ng       (default 0.80 * S_ge^2)

Each phase depends only on its own effective saturation except k_ro, which
depends on both S_w and S_g through S_oe. Hence six derivatives are nonzero:
dk_rw/dS_w, dk_ro/dS_w, dk_ro/dS_g, dk_rg/dS_g (the other two are exactly 0).
They are returned as full arrays so the Jacobian can treat all phases alike.

Where a clip is active the derivative of the clipped saturation is 0. Because
D contains S_gc, S_we saturates at 1 for S_w > 1 - S_or - S_gc (0.75 by default)
even with no gas -- this is a property of the specified model, not a bug.

Capillary pressure
------------------
P_c is neglected: p_w = p_g = p_o. If added later, it would enter in
``residual.py`` when phase pressures are formed: p_w = p_o - P_cow(S_w) and
p_g = p_o + P_cgo(S_g), which changes the upstream-weighting potential
differences and adds dP_c/dS terms to the Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import RelPermConfig, default_config


@dataclass(frozen=True, eq=False)
class RelPermResult:
    """Relative permeabilities and their partial derivatives (all arrays)."""

    krw: np.ndarray
    kro: np.ndarray
    krg: np.ndarray
    dkrw_dsw: np.ndarray
    dkrw_dsg: np.ndarray   # identically 0
    dkro_dsw: np.ndarray
    dkro_dsg: np.ndarray
    dkrg_dsw: np.ndarray   # identically 0
    dkrg_dsg: np.ndarray


class RelPerm:
    """Vectorized Corey relative permeability model."""

    def __init__(self, cfg: RelPermConfig) -> None:
        if min(cfg.krw_exp, cfg.kro_exp, cfg.krg_exp) < 1.0:
            raise ValueError("Corey exponents must be >= 1 (smooth at zero saturation)")
        self.cfg = cfg
        self.denom = 1.0 - cfg.swc - cfg.sor - cfg.sgc
        if self.denom <= 0.0:
            raise ValueError("relperm endpoints leave no mobile saturation range")

    def effective_saturations(self, sw: np.ndarray, sg: np.ndarray):
        """Return (S_we, S_ge, S_oe, dS_we/dS_w, dS_ge/dS_g, mask_oe).

        ``mask_oe`` is 1 where S_oe is not clipped (so dS_oe = -dS_we - dS_ge).
        """
        c = self.cfg
        sw_a = np.asarray(sw, dtype=float)
        sg_a = np.asarray(sg, dtype=float)
        swe_raw = (sw_a - c.swc) / self.denom
        sge_raw = (sg_a - c.sgc) / self.denom
        swe = np.clip(swe_raw, 0.0, 1.0)
        sge = np.clip(sge_raw, 0.0, 1.0)
        soe_raw = 1.0 - swe - sge
        soe = np.clip(soe_raw, 0.0, 1.0)
        dswe = np.where((swe_raw > 0.0) & (swe_raw < 1.0), 1.0 / self.denom, 0.0)
        dsge = np.where((sge_raw > 0.0) & (sge_raw < 1.0), 1.0 / self.denom, 0.0)
        mask_oe = np.where(soe_raw > 0.0, 1.0, 0.0)
        return swe, sge, soe, dswe, dsge, mask_oe

    def evaluate(self, sw: np.ndarray | float, sg: np.ndarray | float) -> RelPermResult:
        """Evaluate k_rw, k_ro, k_rg and all partial derivatives at (S_w, S_g)."""
        c = self.cfg
        swe, sge, soe, dswe, dsge, mask_oe = self.effective_saturations(sw, sg)

        krw = c.krw_max * swe ** c.krw_exp
        kro = c.kro_max * soe ** c.kro_exp
        krg = c.krg_max * sge ** c.krg_exp

        dkrw_dswe = c.krw_max * c.krw_exp * swe ** (c.krw_exp - 1.0)
        dkro_dsoe = c.kro_max * c.kro_exp * soe ** (c.kro_exp - 1.0)
        dkrg_dsge = c.krg_max * c.krg_exp * sge ** (c.krg_exp - 1.0)

        zero = np.zeros_like(krw)
        return RelPermResult(
            krw=krw, kro=kro, krg=krg,
            dkrw_dsw=dkrw_dswe * dswe,
            dkrw_dsg=zero,
            dkro_dsw=-dkro_dsoe * dswe * mask_oe,
            dkro_dsg=-dkro_dsoe * dsge * mask_oe,
            dkrg_dsw=zero.copy(),
            dkrg_dsg=dkrg_dsge * dsge,
        )


def main() -> None:
    """Verify endpoints, monotonicity and derivatives against central differences."""
    cfg = default_config().relperm
    rp = RelPerm(cfg)

    # endpoints
    r = rp.evaluate(np.array([cfg.swc]), np.array([0.0]))
    assert r.krw[0] == 0.0 and r.krg[0] == 0.0 and np.isclose(r.kro[0], cfg.kro_max)
    sw = np.linspace(cfg.swc, 1.0 - cfg.sor, 9)
    r = rp.evaluate(sw, np.zeros_like(sw))
    print(f"{'S_w':>6}{'k_rw':>10}{'k_ro':>10}{'k_rg':>10}")
    for k in range(sw.size):
        print(f"{sw[k]:>6.3f}{r.krw[k]:>10.5f}{r.kro[k]:>10.5f}{r.krg[k]:>10.5f}")
    assert np.all(np.diff(r.krw) >= -1e-15), "k_rw must be nondecreasing in S_w"
    assert np.all(np.diff(r.kro) <= 1e-15), "k_ro must be nonincreasing in S_w"

    # derivatives at random points in the smooth region (S_ge>0, S_we<1, S_oe>0)
    rng = np.random.default_rng(0)
    sw_t = rng.uniform(0.25, 0.65, 200)
    sg_t = rng.uniform(0.06, 0.10, 200)
    h = 1.0e-6
    base = rp.evaluate(sw_t, sg_t)
    dsw = (rp.evaluate(sw_t + h, sg_t), rp.evaluate(sw_t - h, sg_t))
    dsg = (rp.evaluate(sw_t, sg_t + h), rp.evaluate(sw_t, sg_t - h))
    for name in ("krw", "kro", "krg"):
        fd_w = (getattr(dsw[0], name) - getattr(dsw[1], name)) / (2 * h)
        fd_g = (getattr(dsg[0], name) - getattr(dsg[1], name)) / (2 * h)
        an_w, an_g = getattr(base, f"d{name}_dsw"), getattr(base, f"d{name}_dsg")
        err_w = np.max(np.abs(fd_w - an_w) / (np.abs(an_w) + 1e-8))
        err_g = np.max(np.abs(fd_g - an_g) / (np.abs(an_g) + 1e-8))
        assert err_w < 1e-5 and err_g < 1e-5, f"d{name} mismatch ({err_w:.2e}, {err_g:.2e})"
        print(f"d{name}/dSw, d{name}/dSg vs finite difference: rel err {err_w:.1e}, {err_g:.1e}")
    print("Relative permeability self-checks: PASSED")


if __name__ == "__main__":
    main()