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
Per perforation k, with drawdown dd = max(p_k - (p_bh + head_k), 0)
(flow reversal into the producer is clamped):

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
    """Static well data: perforated cells, per-perforation well indices and controls.

    A well may be completed in several cells (vertical well through several
    layers). The BHP datum is the shallowest perforation; the wellbore pressure
    at perforation k is p_bh + head_k with a static wellbore head
    head_k = gamma * (D_k - D_datum) (gamma_o for the producer, gamma_w for the
    injector; zero when gravity is off).

    The injector rate is split between perforations in proportion to WI_k
    (k*h allocation).
    """

    producer_cells: np.ndarray   # (n_perf_prod,) int
    injector_cells: np.ndarray   # (n_perf_inj,) int
    wi_prod: np.ndarray          # (n_perf_prod,) [bbl*cp/(day*psi)]
    wi_inj: np.ndarray           # (n_perf_inj,)
    prod_head: np.ndarray        # (n_perf_prod,) psi
    inj_head: np.ndarray         # (n_perf_inj,) psi
    r_e: float
    producer_bhp: float          # psi at the datum
    injector_rate: float         # STB/day water
    no_backflow: bool

    # ponytail: kh-weighted injector allocation; add p_bh,inj as an unknown
    # with a rate constraint if mobility-weighted allocation matters.
    @property
    def inj_frac(self) -> np.ndarray:
        return self.wi_inj / self.wi_inj.sum()

    # single-cell views kept for diagnostics / legacy callers (first = datum perforation)
    @property
    def producer_cell(self) -> int:
        return int(self.producer_cells[0])

    @property
    def injector_cell(self) -> int:
        return int(self.injector_cells[0])

    @property
    def wi_producer(self) -> float:
        return float(self.wi_prod.sum())

    @property
    def wi_injector(self) -> float:
        return float(self.wi_inj.sum())


@dataclass(frozen=True)
class ProducerRates:
    """Producer surface rates (positive = production) and their derivatives.

    ``q`` is (n_perf, 3): per-perforation (water, oil, total gas) rates.
    ``jac`` is (n_perf, 3, 3): d q[k, eq] / d (p, S_w, S_g) of perforation cell k.
    The scalar fields are well totals.
    """

    q_w: float
    q_o: float
    q_g_free: float
    q_g_total: float
    q: np.ndarray
    jac: np.ndarray


@dataclass(frozen=True)
class WellTerms:
    """Net source terms Q (positive = injection) of both wells.

    ``q_prod`` (n_perf_prod, 3) = -(q_w, q_o, q_g_total) per producer perforation,
    ``dq_prod`` its (n_perf_prod, 3, 3) derivative w.r.t. that cell's (p, S_w, S_g).
    ``q_inj`` (n_perf_inj, 3) = (rate share, 0, 0) (constant, no derivatives).
    """

    producer_cells: np.ndarray
    injector_cells: np.ndarray
    q_prod: np.ndarray
    dq_prod: np.ndarray
    q_inj: np.ndarray
    rates: ProducerRates


def build_wells(cfg: SimulationConfig, grid: Grid, rock: Rock) -> Wells:
    """Construct single-perforation :class:`Wells` from ``cfg.wells`` (legacy MVP setup)."""
    wc = cfg.wells
    pi, pj, pk = wc.producer_cell(grid.nx, grid.ny, grid.nz)
    ii, ij, ik = wc.injector_cell(grid.nz)
    prod = int(grid.cell_index(pi, pj, pk))
    inj = int(grid.cell_index(ii, ij, ik))
    if prod == inj:
        raise ValueError("producer and injector must be in different cells")
    r_e = peaceman_re_isotropic(grid.dx, grid.dy, wc.r_e_factor)

    def wi(c: int) -> float:
        return peaceman_well_index(rock.perm[c], rock.perm[c], grid.dz, r_e, wc.r_w, wc.skin)

    return Wells(
        producer_cells=np.array([prod]), injector_cells=np.array([inj]),
        wi_prod=np.array([wi(prod)]), wi_inj=np.array([wi(inj)]),
        prod_head=np.zeros(1), inj_head=np.zeros(1),
        r_e=r_e, producer_bhp=float(wc.producer_bhp), injector_rate=float(wc.injector_rate),
        no_backflow=bool(wc.producer_no_backflow),
    )


def _perf_arrays(n: int, *values):
    return [np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in values]


def producer_rates(wells: Wells, pvt: PVT, relperm: RelPerm, p, sw, sg) -> ProducerRates:
    """Return producer rates and derivatives at the perforation-cell states.

    ``p, sw, sg`` are arrays over ``wells.producer_cells`` (scalars allowed for
    a single perforation).
    """
    n = wells.producer_cells.size
    pa, swa, sga = _perf_arrays(n, p, sw, sg)
    props = pvt.evaluate(pa)
    kr = relperm.evaluate(swa, sga, cells=wells.producer_cells)

    raw_dd = pa - (wells.producer_bhp + wells.prod_head)
    if wells.no_backflow:
        dd = np.maximum(raw_dd, 0.0)
        d_dd = (raw_dd > 0.0).astype(float)
    else:
        dd, d_dd = raw_dd, np.ones(n)

    bw, bo, bg = props.bw, props.bo, props.bg
    muw, muo, mug = props.mu_w, props.mu_o, props.mu_g
    lam_w, lam_o, lam_g = kr.krw / (muw * bw), kr.kro / (muo * bo), kr.krg / (mug * bg)

    # derivatives of mobilities: columns (p, S_w, S_g)
    dlam_w = np.column_stack([-lam_w * props.dbw_dp / bw,
                              kr.dkrw_dsw / (muw * bw), kr.dkrw_dsg / (muw * bw)])
    dlam_o = np.column_stack([-lam_o * (props.dmu_o_dp / muo + props.dbo_dp / bo),
                              kr.dkro_dsw / (muo * bo), kr.dkro_dsg / (muo * bo)])
    dlam_g = np.column_stack([-lam_g * props.dbg_dp / bg,
                              kr.dkrg_dsw / (mug * bg), kr.dkrg_dsg / (mug * bg)])

    wi = wells.wi_prod
    ddd = np.zeros((n, 3))
    ddd[:, 0] = d_dd                                   # d(dd)/d(p, S_w, S_g)
    q_w, q_o, q_gf = wi * lam_w * dd, wi * lam_o * dd, wi * lam_g * dd
    dq_w = wi[:, None] * (dlam_w * dd[:, None] + lam_w[:, None] * ddd)
    dq_o = wi[:, None] * (dlam_o * dd[:, None] + lam_o[:, None] * ddd)
    dq_gf = wi[:, None] * (dlam_g * dd[:, None] + lam_g[:, None] * ddd)

    q_gt = q_gf + props.rs * q_o
    dq_gt = dq_gf + props.rs[:, None] * dq_o
    dq_gt[:, 0] += props.drs_dp * q_o

    return ProducerRates(q_w=float(q_w.sum()), q_o=float(q_o.sum()), q_g_free=float(q_gf.sum()),
                         q_g_total=float(q_gt.sum()), q=np.column_stack([q_w, q_o, q_gt]),
                         jac=np.stack([dq_w, dq_o, dq_gt], axis=1))


def well_terms(wells: Wells, pvt: PVT, relperm: RelPerm,
               p: np.ndarray, sw: np.ndarray, sg: np.ndarray) -> WellTerms:
    """Return the net well source terms for the current global state."""
    c = wells.producer_cells
    rates = producer_rates(wells, pvt, relperm, p[c], sw[c], sg[c])
    q_inj = np.zeros((wells.injector_cells.size, 3))
    q_inj[:, 0] = wells.injector_rate * wells.inj_frac
    return WellTerms(
        producer_cells=c, injector_cells=wells.injector_cells,
        q_prod=-rates.q, dq_prod=-rates.jac, q_inj=q_inj, rates=rates,
    )


def injector_bhp(wells: Wells, pvt: PVT, relperm: RelPerm, p, sw, sg) -> float:
    """Return the diagnostic injector bottom-hole pressure at the datum [psi].

    Per perforation p_bh = p_k + q_k * B_w / (WI_k * lambda_t,vol) - head_k with
    lambda_t,vol = k_rw/mu_w + k_ro/mu_o + k_rg/mu_g; the largest (controlling)
    value is returned. ``p, sw, sg`` are arrays over ``wells.injector_cells``.
    """
    n = wells.injector_cells.size
    pa, swa, sga = _perf_arrays(n, p, sw, sg)
    props = pvt.evaluate(pa)
    kr = relperm.evaluate(swa, sga, cells=wells.injector_cells)
    lam_t = kr.krw / props.mu_w + kr.kro / props.mu_o + kr.krg / props.mu_g
    if np.any(lam_t <= 0.0):
        raise ValueError("total mobility is zero at an injector perforation")
    q = wells.injector_rate * wells.inj_frac
    return float(np.max(pa + q * props.bw / (wells.wi_inj * lam_t) - wells.inj_head))


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
    assert wells.producer_cell == grid.cell_index(27, 27, cfg.wells.producer_k) and wells.injector_cell == grid.cell_index(2, 2, cfg.wells.injector_k)

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
            an = r.jac[0][:, col]
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
    assert np.all(terms.q_prod <= 0.0) and terms.q_inj[0, 0] == cfg.wells.injector_rate

    # diagnostic injector BHP exceeds the cell pressure
    bhp = injector_bhp(wells, pvt, rp, 4000.0, 0.25, 0.0)
    assert bhp > 4000.0
    print(f"diagnostic injector BHP at p_cell = 4000 psi: {bhp:.1f} psi")
    print("Wells self-checks: PASSED")


if __name__ == "__main__":
    main()