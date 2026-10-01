"""initial_state_model.py -- Build a simulator model + initial state from ``_initial_state_.py``.

``_initial_state_.build_reservoir()`` produces the geological model and the
hydrostatic initial state in SI units on (Nx, Ny, Nz) arrays. The simulator
works in field units on flat arrays ordered ``k*ny*nx + j*nx + i``. This module
is the only place where the two meet:

    arrays   (Nx, Ny, Nz) --transpose(2, 1, 0).ravel()--> flat cell order
    faces    fault trans_mult_x/y/z (internal faces)  --> same flat face order as grid.py
    length   m   -> ft        pressure  Pa -> psi
    rate     m3/day water -> STB/day     density  rho*g [Pa/m] -> gamma [psi/ft]

Physics carried over from the initial-state model
-------------------------------------------------
* depth per cell (structure: dome + dip) -> gravity in the fluxes
* kx, ky, kz (matrix + fracture corridors) -> directional TPFA transmissibility
* fault face multipliers -> transmissibility multipliers
* facies connate water -> per-cell k_rw endpoint (water at S_wc is immobile)
* rho_w, rho_o -> gamma_w, gamma_o, the SAME constants that built the initial
  pressure, so the oil and water legs are in exact hydrostatic equilibrium
* vertical wells with their perforated layers

Known limitation (P_c = 0): cells where water is mobile above the OWC (transition
zone, shale with S_w = 1) are not in equilibrium on the oil gradient and relax
slowly under gravity. ``python initial_state_model.py`` measures that drift.
"""

from __future__ import annotations

import dataclasses

import importlib

import numpy as np

from config import SimulationConfig, default_config, override
from grid import build_grid, pack_state
from pvt import PVT
from relperm import RelPerm
from residual import ReservoirModel
from rock import Rock
from transmissibility import face_transmissibility
from units import BBL_FT3, BBL_TO_M3, FT_TO_M, PSI_TO_PA
from wells import Wells, peaceman_re_anisotropic, peaceman_well_index

isr = importlib.import_module("_initial_state_")

GRAVITY = isr.G


def _flat(a: np.ndarray) -> np.ndarray:
    """(Nx, Ny, Nz) -> flat simulator order k*ny*nx + j*nx + i."""
    return np.ascontiguousarray(np.asarray(a).transpose(2, 1, 0)).ravel()


def gamma_psi_per_ft(rho: float) -> float:
    """Hydrostatic gradient rho*g in psi/ft."""
    return rho * GRAVITY * FT_TO_M / PSI_TO_PA


def build_from_initial_state(res: dict | None = None, cfg: SimulationConfig | None = None,
                             rho_g: float = 150.0) -> tuple[ReservoirModel, np.ndarray]:
    """Return (model, x0) for the reservoir ``res`` from ``_initial_state_.build_reservoir``.

    ``cfg`` supplies PVT, relperm, Newton and time-step settings (grid and
    gravity are overwritten). ``rho_g`` [kg/m3] is the free-gas density used for
    the gas gravity head (not defined by the initial-state model; illustrative).
    """
    cfg = cfg or default_config()
    if res is None:
        # The primary flow path uses the geological fractal field. Explicit
        # ``res`` remains available for controlled fixtures and saved cases.
        geo_cfg = isr.ReservoirConfig(spectrum="fractal", fractal_beta=cfg.fractal.beta,
                                      seed=cfg.fractal.seed)
        res = isr.build_reservoir(geo_cfg)
    isr.validate_reservoir(res)
    rc, g = res["config"], res["grid"]
    shape = (g["Nx"], g["Ny"], g["Nz"])
    for name in ("depth", "facies", "phi", "kx", "ky", "kz"):
        if np.shape(res["grid"].get(name, res.get(name))) != shape:
            raise ValueError(f"geological {name} must have shape {shape}")
    st = res["initial_state"]
    if not all(np.shape(st[name]) == shape for name in ("pressure", "Sw", "So", "Sg")):
        raise ValueError("geological initial-state arrays must share grid shape")
    if not np.allclose(st["So"], 1.0 - st["Sw"] - st["Sg"], atol=1e-12):
        raise ValueError("geological initial state violates So = 1 - Sw - Sg")
    if np.any((st["Sw"] < 0) | (st["Sw"] > 1) | (st["So"] < 0) | (st["So"] > 1) |
              (st["Sg"] < 0) | (st["Sg"] > 1)):
        raise ValueError("geological initial saturations must lie in [0, 1]")
    m2ft = 1.0 / FT_TO_M

    cfg = dataclasses.replace(cfg, name="synthetic")
    cfg = override(cfg, "grid", nx=g["Nx"], ny=g["Ny"], nz=g["Nz"],
                   dx=g["dx"] * m2ft, dy=g["dy"] * m2ft, dz=g["dz"] * m2ft)
    cfg = override(cfg, "pvt", gamma_w=gamma_psi_per_ft(rc.rho_w), gamma_o=gamma_psi_per_ft(rc.rho_o),
                   gamma_g=gamma_psi_per_ft(rho_g))
    grid = build_grid(cfg.grid, depth=_flat(g["depth"]) * m2ft)

    kx, ky, kz = _flat(res["kx"]), _flat(res["ky"]), _flat(res["kz"])
    phi = _flat(res["phi"])
    rock = Rock(phi0=phi, perm=kx, perm_y=_flat(res["ky"]), perm_z=_flat(res["kz"]),
                pressure_dependent=cfg.rock.pressure_dependent,
                c_r=cfg.rock.c_r, p_ref=cfg.rock.p_ref)
    fl = res["faults"]
    mult_parts = (_flat(fl["trans_mult_x"]), _flat(fl["trans_mult_y"]), _flat(fl["trans_mult_z"]))
    expected_faces = ((g["Nx"] - 1) * g["Ny"] * g["Nz"] +
                      g["Nx"] * (g["Ny"] - 1) * g["Nz"] +
                      g["Nx"] * g["Ny"] * (g["Nz"] - 1))
    if sum(part.size for part in mult_parts) != expected_faces:
        raise ValueError("fault multiplier arrays do not match internal grid faces")
    if any(np.any((part <= 0) | (part > 1)) for part in mult_parts):
        raise ValueError("fault multipliers must lie in (0, 1]")
    mult = np.concatenate(mult_parts)
    trans = face_transmissibility(grid, np.column_stack([kx, ky, kz]), mult)

    table = res["facies_table"]
    # non-net facies carry swc = 1 as a "no oil" marker, which has no Corey range;
    # they use the table swc instead (k ~ 0.01 md, so their water barely moves).
    swc = np.array([table[int(c)].swc if table[int(c)].net else cfg.relperm.swc
                    for c in _flat(res["facies"])])
    relperm = RelPerm(cfg.relperm, swc=swc)

    wells = _build_wells(res, grid, kx, ky, cfg)
    x0 = pack_state(_flat(st["pressure"]) / PSI_TO_PA, _flat(st["Sw"]), _flat(st["Sg"]))
    model = ReservoirModel(cfg=cfg, grid=grid, rock=rock, pvt=PVT(cfg.pvt), relperm=relperm,
                           wells=wells, trans=trans, pv0=grid.bulk_volume / BBL_FT3)
    return model, x0


def _build_wells(res: dict, grid, kx: np.ndarray, ky: np.ndarray, cfg: SimulationConfig) -> Wells:
    """One BHP producer + one water-rate injector, each with all its perforations."""
    by_type = {w["type"]: [v for v in res["wells"] if v["type"] == w["type"]] for w in res["wells"]}
    if len(by_type.get("producer", [])) != 1 or len(by_type.get("injector", [])) != 1:
        raise NotImplementedError("the simulator supports exactly one producer and one injector")
    m2ft = 1.0 / FT_TO_M

    def completion(w: dict, gamma: float):
        cells = np.array([grid.cell_index(w["i"], w["j"], k) for k in w["perforated_layers"]])
        r_e = np.array([peaceman_re_anisotropic(grid.dx, grid.dy, kx[c], ky[c]) for c in cells])
        wi = np.array([peaceman_well_index(kx[c], ky[c], grid.dz, re, w["rw"] * m2ft, w["skin"])
                       for c, re in zip(cells, r_e)])
        head = gamma * (grid.depth[cells] - grid.depth[cells].min())
        return cells, wi, head, float(r_e.mean())

    prod, inj = by_type["producer"][0], by_type["injector"][0]
    if prod["control"] != "bhp" or prod["units"] != "Pa":
        raise NotImplementedError("producer must be BHP-controlled in Pa")
    if inj["control"] != "rate" or inj["units"] != "m3/day":
        raise NotImplementedError("injector must be water-rate-controlled in m3/day")
    p_cells, p_wi, p_head, r_e = completion(prod, cfg.pvt.gamma_o)   # ponytail: oil-filled wellbore head
    i_cells, i_wi, i_head, _ = completion(inj, cfg.pvt.gamma_w)
    return Wells(producer_cells=p_cells, injector_cells=i_cells, wi_prod=p_wi, wi_inj=i_wi,
                 prod_head=p_head, inj_head=i_head, r_e=r_e,
                 producer_bhp=prod["value"] / PSI_TO_PA,
                 injector_rate=inj["value"] / BBL_TO_M3,   # surface water, B_w ~ 1
                 no_backflow=cfg.wells.producer_no_backflow)


def main() -> None:
    """Self-checks: mapping, units, hydrostatic equilibrium, and no-well drift."""
    from newton_solver import NewtonParams, newton_solve
    from residual import accumulation, evaluate_residual, residual_scales, scaled_residual_norm

    res = isr.build_reservoir()
    model, x0 = build_from_initial_state(res)
    g, grid = res["grid"], model.grid
    p, sw, sg = x0[0::3], x0[1::3], x0[2::3]

    # index mapping: flat cell (i, j, k) holds the (i, j, k) value of the SI arrays
    for (i, j, k) in [(0, 0, 0), (7, 3, 5), (g["Nx"] - 1, g["Ny"] - 1, g["Nz"] - 1)]:
        c = grid.cell_index(i, j, k)
        assert np.isclose(p[c] * PSI_TO_PA, res["initial_state"]["pressure"][i, j, k])
        assert np.isclose(grid.depth[c] * FT_TO_M, g["depth"][i, j, k])
        assert sw[c] == res["initial_state"]["Sw"][i, j, k]
    # fault multipliers land on the right faces
    n_fault_faces = sum(int((res["faults"][f] < 1).sum()) for f in ("trans_mult_x", "trans_mult_y", "trans_mult_z"))
    assert int((model.trans < 0.999 * face_transmissibility(
        grid, np.column_stack([_flat(res["kx"]), _flat(res["ky"]), _flat(res["kz"])]))).sum()) == n_fault_faces
    print(f"grid {grid.nx}x{grid.ny}x{grid.nz}, depth {grid.depth.min():.1f}-{grid.depth.max():.1f} ft, "
          f"p {p.min():.1f}-{p.max():.1f} psi, gamma_w/o = {model.cfg.pvt.gamma_w:.4f}/{model.cfg.pvt.gamma_o:.4f} psi/ft")
    w = model.wells
    print(f"producer: {w.producer_cells.size} perfs, BHP {w.producer_bhp:.1f} psi; "
          f"injector: {w.injector_cells.size} perfs, {w.injector_rate:.1f} STB/d")
    assert p.min() > model.cfg.pvt.p_bubble, "initial state must be undersaturated (Sg = 0)"

    # hydrostatic equilibrium: oil and water potentials are flat wherever that phase is mobile
    closed = dataclasses.replace(model, wells=None)
    ev = evaluate_residual(closed, x0, accumulation(closed, x0), 1.0)
    q = np.abs(ev.fluxes.q)
    print(f"t=0 max |face flux|: water {q[:, 0].max():.2e}, oil {q[:, 1].max():.2e} STB/d")
    assert np.max(np.abs(ev.fluxes.q[:, 1])) < 1e-6, "oil leg must be in exact hydrostatic equilibrium"
    scale = residual_scales(closed, accumulation(closed, x0), 1.0)
    print(f"closed model, x = x0, dt = 1 d: scaled residual {scaled_residual_norm(ev.residual, scale):.2e} "
          "(non-zero only from P_c = 0 water drainage, see module doc)")

    # drift over 30 days without wells
    r = newton_solve(x0, x0, closed, 30.0, NewtonParams())
    assert r.converged
    p1, sw1 = r.x[0::3], r.x[1::3]
    print(f"30 days, no wells: max |dp| = {np.max(np.abs(p1 - p)):.3f} psi, max |dSw| = {np.max(np.abs(sw1 - sw)):.4f}")
    print("Initial-state model self-checks: PASSED")


if __name__ == "__main__":
    main()
