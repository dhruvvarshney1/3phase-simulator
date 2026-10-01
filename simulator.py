"""High-level simulation loop built on residual.ReservoirModel."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from config import SimulationConfig, validate_config
from residual import accumulation, build_model, pack_state, unpack_state
from newton_solver import NewtonParams, NewtonResult, newton_solve
from timestep import TimeStepParams, TimeStepper, cut_dt
from wells import injector_bhp, producer_rates

__all__ = ["build_context", "compute_well_rates", "run_simulation"]


def build_context(cfg: SimulationConfig, model=None, x0: Optional[np.ndarray] = None):
    """Return (model, x0, diagnostics). ``model``/``x0`` override the uniform
    config-built model (e.g. from ``initial_state_model.build_from_initial_state``)."""
    if model is None:
        validate_config(cfg)
        model = build_model(cfg)
    if x0 is None:
        x0 = pack_state(
            np.full(model.grid.n_cells, cfg.init.p_init),
            np.full(model.grid.n_cells, cfg.init.sw_init),
            np.full(model.grid.n_cells, cfg.init.sg_init),
        )
    m0 = accumulation(model, x0)
    return model, x0, {
        "ooip_stb": float(m0[:, 1].sum()),
        "owip_stb": float(m0[:, 0].sum()),
    }


def compute_well_rates(x: np.ndarray, model) -> Dict[str, Any]:
    if model.wells is None:
        return {"producers": [], "injectors": [], "totals": {"qw_prod": 0.0, "qo_prod": 0.0, "qg_prod": 0.0, "qw_inj": 0.0}}
    p, sw, sg = unpack_state(x)
    w = model.wells
    pc, ic = w.producer_cells, w.injector_cells
    rates = producer_rates(w, model.pvt, model.relperm, p[pc], sw[pc], sg[pc])
    inj_bhp = injector_bhp(w, model.pvt, model.relperm, p[ic], sw[ic], sg[ic])
    return {
        "producers": [{"cell": w.producer_cell, "qw": rates.q_w, "qo": rates.q_o, "qg": rates.q_g_total, "bhp": w.producer_bhp, "p_cell": float(p[w.producer_cell])}],
        "injectors": [{"cell": w.injector_cell, "qw": w.injector_rate, "bhp_est": inj_bhp, "p_cell": float(p[w.injector_cell])}],
        "totals": {"qw_prod": rates.q_w, "qo_prod": rates.q_o, "qg_prod": rates.q_g_total, "qw_inj": w.injector_rate},
    }


def _snapshot(x: np.ndarray) -> Dict[str, np.ndarray]:
    p, sw, sg = unpack_state(x)
    return {"p": p.copy(), "Sw": sw.copy(), "Sg": sg.copy(), "So": 1.0 - sw - sg}


def run_simulation(cfg: SimulationConfig, init_guess_mode: str = "previous", rf_model: Optional[Any] = None,
                   jacobian_method: str = "analytical", verbose: bool = True,
                   live_callback: Optional[Callable[[float, np.ndarray, Any, Dict[str, float]], None]] = None,
                   model=None, x0: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if init_guess_mode not in ("previous", "rf"):
        raise ValueError("init_guess_mode must be 'previous' or 'rf'")
    if init_guess_mode == "rf" and rf_model is None:
        raise ValueError("init_guess_mode='rf' requires rf_model")
    model, x0, diagnostics = build_context(cfg, model, x0)
    ts_cfg = cfg.timestep
    ts_params = TimeStepParams(
        dt_init=ts_cfg.stages[0][2], dt_min=ts_cfg.dt_min, dt_max=max(s[3] for s in ts_cfg.stages),
        grow_factor=ts_cfg.grow_factor, shrink_factor=ts_cfg.shrink_factor,
        iter_grow_threshold=ts_cfg.iters_easy, iter_shrink_threshold=ts_cfg.iters_hard,
        max_cuts=ts_cfg.max_cuts, phase_boundaries=[s[1] for s in ts_cfg.stages],
        phase_dt_caps=[s[3] for s in ts_cfg.stages])
    stepper = TimeStepper(ts_params, ts_cfg.t_end, ts_cfg.report_times)
    params = NewtonParams(tol=cfg.newton.tol, max_iter=cfg.newton.max_iter, alpha_min=cfg.newton.alpha_min,
                          p_min=cfg.newton.p_min, p_scale=cfg.newton.p_scale,
                          dp_rel_tol=cfg.newton.dp_tol_rel, ds_tol=cfg.newton.ds_tol)
    report_times = sorted(set(ts_cfg.report_times) | {ts_cfg.t_end})
    x, t, dt = x0.copy(), 0.0, stepper.initial_dt()
    snapshots = {0.0: _snapshot(x)}
    rows: List[Dict[str, float]] = []
    cumulative = {"qw_prod": 0.0, "qo_prod": 0.0, "qg_prod": 0.0, "qw_inj": 0.0}
    total_iters = total_cuts = steps = 0
    result: Optional[NewtonResult] = None
    while t < ts_cfg.t_end - 1e-9:
        dt_try, cuts = min(dt, ts_cfg.t_end - t), 0
        while True:
            guess = x.copy()
            if init_guess_mode == "rf":
                guess = rf_model.predict_initial_guess(x, model, dt_try, t)
            result = newton_solve(guess, x, model, dt_try, params, jacobian_method)
            if result.converged:
                break
            cuts += 1; total_cuts += 1
            if cuts > ts_params.max_cuts or dt_try <= ts_params.dt_min + 1e-12:
                raise RuntimeError(f"Newton failed at t={t:.4f} days")
            dt_try = cut_dt(dt_try, ts_params)
        rates = compute_well_rates(result.x, model)["totals"]
        for key in cumulative:
            cumulative[key] += rates[key] * dt_try
        t += dt_try; steps += 1; total_iters += result.n_iterations
        p, _, _ = unpack_state(result.x)
        qo, qw, qg = rates["qo_prod"], rates["qw_prod"], rates["qg_prod"]
        rows.append({"t": t, "dt": dt_try, "n_newton_iter": result.n_iterations,
                     "avg_pressure": float(np.mean(p)), "qo_prod": qo, "qw_prod": qw, "qg_prod": qg,
                     "qw_inj": rates["qw_inj"], "cum_oil_stb": cumulative["qo_prod"],
                     "cum_water_stb": cumulative["qw_prod"], "cum_gas": cumulative["qg_prod"],
                     "cum_water_inj_stb": cumulative["qw_inj"], "water_cut": qw / (qw + qo) if qw + qo else 0.0,
                     "gor": qg / qo if qo else 0.0, "recovery_factor": cumulative["qo_prod"] / diagnostics["ooip_stb"],
                     "final_scaled_residual": result.final_scaled_residual})
        if live_callback is not None:
            live_callback(t, result.x, model, rows[-1])
        if any(abs(t - rt) < 1e-6 for rt in report_times):
            snapshots[float(t)] = _snapshot(result.x)
        x = result.x
        dt = stepper.propose_next(t, result.n_iterations, dt_try)
        if verbose:
            print(f"[simulator] step {steps}: t={t:.2f}d dt={dt_try:.2f} iters={result.n_iterations} "
                  f"cuts={cuts} p_avg={np.mean(p):.1f} psi qo={qo:.1f} qw={qw:.1f} STB/d "
                  f"qg={qg:.0f} SCF/d", flush=True)
    return {"timeseries": pd.DataFrame(rows), "snapshots": snapshots,
            "summary": {"n_steps": steps, "total_newton_iterations": total_iters,
                         "mean_newton_iterations": total_iters / steps if steps else 0.0,
                         "total_dt_cuts": total_cuts, **diagnostics, "final_state": x.copy()},
            "context": model, "config": cfg}
