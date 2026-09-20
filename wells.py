"""wells.py -- Peaceman well model: BHP-controlled producer, rate-controlled injector.

Peaceman well index (README section 6, Eq. 6.1)
-----------------------------------------------
    WI = 0.001127 * 2*pi * sqrt(kx*ky) * dz / (ln(r_e / r_w) + skin)   [bbl*cp/(day*psi)]

Equivalent radius. Isotropic form used here:

    r_e = 0.14 * sqrt(dx^2 + dy^2)

Anisotropic extension (implemented in :func:`peaceman_re_anisotropic`, not used
by default because permeability is isotropic):

    r_e = 0.28 * sqrt( sqrt(ky/kx)*dx^2 + sqrt(kx/ky)*dy^2 ) / ( (ky/kx)^(1/4) + (kx/ky)^(1/4) )

which reduces to 0.14*sqrt(dx^2+dy^2) when kx = ky.

Producer (BHP control, Eq. 6.2)
-------------------------------
With drawdown dd = max(p_i - p_bh, 0) (flow reversal into the producer is clamped):

    q_w = WI * lambda_w * dd                  [STB/day]
    q_o = WI * lambda_o * dd                  [STB/day]
    q_g,free = WI * lambda_g * dd             [SCF/day]
    q_g,total = q_g,free + R_s(p_i) * q_o     [SCF/day]  (dissolved gas)

with lambda_a = k_ra/(mu_a*B_a) evaluated in the reservoir (upstream) cell.
B_g is treated as bbl/SCF so that S_g/B_g and R_s*S_o/B_o are both SCF per
reservoir barrel of pore volume, i.e. gas rates are SCF/day.

Injector (water rate control, Eq. 6.3)
--------------------------------------
Water enters the water equation of its cell as a constant source. The BHP is
diagnostic only (not an unknown in this MVP):

    p_bh,inj = p_i + q_inj * B_w / (WI * lambda_t,vol),   lambda_t,vol = sum_a k_ra/mu_a

Extension: to add a rate-controlled *producer* (or a BHP limit on the
injector) one appends p_bh as an extra unknown per well plus a control
equation, and switches the active control when a limit is violated
(variable switching); the well rows/columns are then added to the Jacobian.

Sign convention
---------------
The cell residual is ``accumulation + sum(face fluxes) - Q`` where Q is the
net source (positive = injection). Hence producer terms are negative: Q = -q.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from config import SimulationConfig, default_config, override
from grid import Grid, build_grid
from pvt import PVT
from relperm import RelPerm
from rock import Rock, build_rock
from units import well_index_radial


def peaceman_re_isotropic(dx: float, dy: float, factor: float = 0.14) -> float:
    """Return the isotropic Peaceman equivalent radius factor*sqrt(dx^2 + dy^2) [ft]."""
    return factor * math.sqrt(dx * dx + dy * dy)


def peaceman_re_anisotropic(dx: float, dy: float, kx: float, ky: float) -> float:
    """Return the anisotropic Peaceman equivalent radius [ft] (extension)."""
    a = math.sqrt(ky / kx)
    numerator = math.sqrt(a * dx * dx + (1.0 / a) * dy * dy)
    denominator = (ky / kx) ** 0.25 + (kx / ky) ** 0.25
    return 0.28 * numerator / denominator


def peaceman_well_index(kx: float, ky: float, dz: float, r_e: float, r_w: float,
                        skin: float = 0.0) -> float:
    """Return WI [bbl*cp/(day*psi)] with k = sqrt(kx*ky)."""
    if r_e <= r_w:
        raise ValueError("equivalent radius must exceed the wellbore radius")
    return well_index_radial(math.sqrt(kx * ky), dz, r_e, r_w, skin)


@dataclass(frozen=True, eq=False)
class Wells:
    """Static well data: locations, well indices and controls."""

    producer_cell: int
    injector_cell: int
    wi_producer: float
    wi_injector: float
    r_e: float
    producer_bhp: float          # psi
    injector_rate: float         # STB/day water
    no_backflow: bool


@dataclass(frozen=True)
class ProducerRates:
    """Producer surface rates (positive = production) and their derivatives.

    ``jac`` is a (3, 3) array: rows = (water, oil, total gas), columns =
    derivatives with respect to the producer-cell unknowns (p, S_w, S_g).
    """

    q_w: float
    q_o: float
    q_g_free: float
    q_g_total: float
    jac: np.ndarray


@dataclass(frozen=True)
class WellTerms:
    """Net source terms Q (positive = injection) of both wells.

    ``q_prod`` = (-q_w, -q_o, -q_g_total) at the producer cell with
    ``dq_prod`` its (3, 3) derivative w.r.t. (p, S_w, S_g) of that cell.
    ``q_inj`` = (rate, 0, 0) at the injector cell (constant, no derivatives).
    """

    producer_cell: int
    injector_cell: int
    q_prod: np.ndarray
    dq_prod: np.ndarray
    q_inj: np.ndarray
    rates: ProducerRates


def build_wells(cfg: SimulationConfig, grid: Grid, rock: Rock) -> Wells:
    """Construct the :class:`Wells` for a configuration, grid and rock."""
    wc = cfg.wells
    pi, pj = wc.producer_cell(grid.nx, grid.ny)
    ii, ij = wc.injector_ij
    prod = int(grid.cell_index(pi, pj))
    inj = int(grid.cell_index(ii, ij))
    if prod == inj:
        raise ValueError("producer and injector must be in different cells")
    r_e = peaceman_re_isotropic(grid.dx, grid.dy, wc.r_e_factor)
    return Wells(
        producer_cell=prod, injector_cell=inj,
        wi_producer=peaceman_well_index(rock.perm[prod], rock.perm[prod], grid.dz, r_e,
                                        wc.r_w, wc.skin),
        wi_injector=peaceman_well_index(rock.perm[inj], rock.perm[inj], grid.dz, r_e,
                                        wc.r_w, wc.skin),
        r_e=r_e, producer_bhp=float(wc.producer_bhp), injector_rate=float(wc.injector_rate),
        no_backflow=bool(wc.producer_no_backflow),
    )


def producer_rates(wells: Wells, pvt: PVT, relperm: RelPerm,
                   p: float, sw: float, sg: float) -> ProducerRates:
    """Return producer surface rates and derivatives at cell state (p, S_w, S_g)."""
    pa, swa, sga = np.array(p, dtype=float), np.array(sw, dtype=float), np.array(sg, dtype=float)
    props = pvt.evaluate(pa)
    kr = relperm.evaluate(swa, sga)

    raw_dd = float(pa) - wells.producer_bhp
    if wells.no_backflow:
        dd = max(raw_dd, 0.0)
        d_dd = 1.0 if raw_dd > 0.0 else 0.0
    else:
        dd, d_dd = raw_dd, 1.0

    bw, bo, bg = float(props.bw), float(props.bo), float(props.bg)
    muw, muo, mug = float(props.mu_w), float(props.mu_o), float(props.mu_g)
    rs, drs = float(props.rs), float(props.drs_dp)

    lam_w, lam_o, lam_g = float(kr.krw) / (muw * bw), float(kr.kro) / (muo * bo), \
        float(kr.krg) / (mug * bg)

    # derivatives of mobilities: columns (p, S_w, S_g)
    dlam_w = np.array([-lam_w * float(props.dbw_dp) / bw,
                       float(kr.dkrw_dsw) / (muw * bw), float(kr.dkrw_dsg) / (muw * bw)])
    dlam_o = np.array([-lam_o * (float(props.dmu_o_dp) / muo + float(props.dbo_dp) / bo),
                       float(kr.dkro_dsw) / (muo * bo), float(kr.dkro_dsg) / (muo * bo)])
    dlam_g = np.array([-lam_g * float(props.dbg_dp) / bg,
                       float(kr.dkrg_dsw) / (mug * bg), float(kr.dkrg_dsg) / (mug * bg)])

    wi = wells.wi_producer
    ddd = np.array([d_dd, 0.0, 0.0])           # d(dd)/d(p, S_w, S_g)
    q_w, q_o, q_gf = wi * lam_w * dd, wi * lam_o * dd, wi * lam_g * dd
    dq_w = wi * (dlam_w * dd + lam_w * ddd)
    dq_o = wi * (dlam_o * dd + lam_o * ddd)
    dq_gf = wi * (dlam_g * dd + lam_g * ddd)

    q_gt = q_gf + rs * q_o
    dq_gt = dq_gf + rs * dq_o + np.array([drs * q_o, 0.0, 0.0])

    return ProducerRates(q_w=q_w, q_o=q_o, q_g_free=q_gf, q_g_total=q_gt,
                         jac=np.vstack([dq_w, dq_o, dq_gt]))


def well_terms(wells: Wells, pvt: PVT, relperm: RelPerm,
               p: np.ndarray, sw: np.ndarray, sg: np.ndarray) -> WellTerms:
    """Return the net well source terms for the current global state."""
    c = wells.producer_cell
    rates = producer_rates(wells, pvt, relperm, float(p[c]), float(sw[c]), float(sg[c]))
    return WellTerms(
        producer_cell=c, injector_cell=wells.injector_cell,
        q_prod=-np.array([rates.q_w, rates.q_o, rates.q_g_total]),
        dq_prod=-rates.jac,
        q_inj=np.array([wells.injector_rate, 0.0, 0.0]),
        rates=rates,
    )


def injector_bhp(wells: Wells, pvt: PVT, relperm: RelPerm,
                 p: float, sw: float, sg: float) -> float:
    """Return the diagnostic injector bottom-hole pressure [psi].

    p_bh = p_i + q_inj * B_w / (WI * lambda_t,vol) with the total volumetric
    mobility lambda_t,vol = k_rw/mu_w + k_ro/mu_o + k_rg/mu_g of the cell.
    """
    props = pvt.evaluate(np.array(p, dtype=float))
    kr = relperm.evaluate(np.array(sw, dtype=float), np.array(sg, dtype=float))
    lam_t = float(kr.krw) / float(props.mu_w) + float(kr.kro) / float(props.mu_o) \
        + float(kr.krg) / float(props.mu_g)
    if lam_t <= 0.0:
        raise ValueError("total mobility is zero at the injector cell")
    return float(p) + wells.injector_rate * float(props.bw) / (wells.wi_injector * lam_t)


def main() -> None:
    """Verify well index, rates, derivatives, clamping and injector BHP."""
    cfg = default_config()
    grid = build_grid(cfg.grid)
    rock = build_rock(override(cfg, "rock", use_fractal=False), grid)   # k = 100 md everywhere
    wells = build_wells(cfg, grid, rock)
    pvt, rp = PVT(cfg.pvt), RelPerm(cfg.relperm)

    # hand calculation of the well index for k = 100 md
    r_e = 0.14 * math.sqrt(50.0 ** 2 + 50.0 ** 2)
    wi_hand = 0.001127 * 2.0 * math.pi * 100.0 * 20.0 / math.log(r_e / 0.25)
    assert math.isclose(wells.wi_producer, wi_hand, rel_tol=1e-12)
    print(f"r_e = {wells.r_e:.4f} ft, WI = {wells.wi_producer:.4f} bbl*cp/(day*psi); "
          f"producer cell {wells.producer_cell}, injector cell {wells.injector_cell}")
    assert math.isclose(peaceman_re_anisotropic(50.0, 50.0, 100.0, 100.0), r_e, rel_tol=1e-12)
    assert wells.producer_cell == grid.cell_index(27, 27) and wells.injector_cell == grid.cell_index(2, 2)

    # rates, dissolved gas, derivatives against central differences
    for (p, sw, sg) in [(3000.0, 0.45, 0.0), (2000.0, 0.40, 0.08), (1800.0, 0.30, 0.15)]:
        r = producer_rates(wells, pvt, rp, p, sw, sg)
        rs = float(pvt.rs(np.array(p)))
        assert math.isclose(r.q_g_total, r.q_g_free + rs * r.q_o, rel_tol=1e-12)
        base = np.array([p, sw, sg])
        steps = np.array([1.0e-2, 1.0e-6, 1.0e-6])
        for col in range(3):
            up, dn = base.copy(), base.copy()
            up[col] += steps[col]
            dn[col] -= steps[col]
            ru, rd = producer_rates(wells, pvt, rp, *up), producer_rates(wells, pvt, rp, *dn)
            fd = np.array([ru.q_w - rd.q_w, ru.q_o - rd.q_o, ru.q_g_total - rd.q_g_total]) \
                / (2.0 * steps[col])
            an = r.jac[:, col]
            err = np.max(np.abs(fd - an) / (np.abs(an) + 1e-8))
            assert err < 1e-5, f"producer derivative mismatch (col {col}): {err:.2e}"
        print(f"p={p:>6.0f} Sw={sw:.2f} Sg={sg:.2f}: qw={r.q_w:9.3f} qo={r.q_o:9.3f} STB/d, "
              f"free gas={r.q_g_free:11.1f}, total gas={r.q_g_total:11.1f} SCF/d, "
              f"GOR={r.q_g_total / r.q_o:7.1f}")

    # flow-reversal clamp
    r0 = producer_rates(wells, pvt, rp, 1400.0, 0.4, 0.0)
    assert r0.q_w == r0.q_o == r0.q_g_total == 0.0 and np.all(r0.jac == 0.0)

    # net well terms and sign convention
    n = grid.n_cells
    terms = well_terms(wells, pvt, rp, np.full(n, 3000.0), np.full(n, 0.4), np.zeros(n))
    assert np.all(terms.q_prod[:3] <= 0.0) and terms.q_inj[0] == cfg.wells.injector_rate

    # diagnostic injector BHP exceeds the cell pressure
    bhp = injector_bhp(wells, pvt, rp, 4000.0, 0.25, 0.0)
    assert bhp > 4000.0
    print(f"diagnostic injector BHP at p_cell = 4000 psi: {bhp:.1f} psi")
    print("Wells self-checks: PASSED")


if __name__ == "__main__":
    main()