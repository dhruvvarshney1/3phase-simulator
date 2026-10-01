"""
postprocess.py
===============

Post-processing, reporting and plotting utilities for simulator runs
(README §11 "Outputs and Plots").

Produces:
    - Static rock property maps (porosity, permeability), README §7/§11.
    - Field snapshots (p, Sw, Sg, So) at requested reporting times.
    - Time-series plots (avg pressure, phase rates, water cut, GOR,
      cumulatives, recovery factor, Newton iterations, dt history) + CSV.
    - Newton convergence (residual vs. iteration, log scale).
    - Fractal-heterogeneity study comparison plots (recovery factor and
      breakthrough time vs. beta and vs. D_f).
    - Baseline-vs-RF benchmark plots (runtime, error distributions).

All figures are written as PNG to the given output directory; all
time-series data is written as CSV. Nothing in this module performs any
physics computation beyond simple, already-validated bookkeeping (fluid
in-place, from the same accumulation formulas used in ``jacobian.py``); it
is purely a consumer of ``simulator.run_simulation`` output.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Optional for mass-balance-only consumers.
    plt = None

from residual import ReservoirModel, accumulation, unpack_state

__all__ = [
    "ensure_outdir",
    "grid_dims",
    "well_markers",
    "plot_rock_maps",
    "plot_snapshots",
    "save_timeseries_csv",
    "plot_timeseries",
    "plot_newton_convergence",
    "compute_fluid_in_place",
    "mass_balance_summary",
    "save_run_config",
    "plot_fractal_study",
    "plot_benchmark_results",
    "plot_rf_error_distributions",
    "generate_mvp_report",
]


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def ensure_outdir(outdir: str) -> str:
    """Create ``outdir`` (and parents) if needed; return it unchanged."""
    os.makedirs(outdir, exist_ok=True)
    return outdir


def grid_dims(grid: Any) -> Tuple[int, int, float, float]:
    """
    Robustly extract (nx, ny, dx, dy) from a ``CartesianGrid`` instance.

    Primarily reads the standard public attributes; falls back to inferring
    nx/ny from the number of distinct cell-center coordinates if the
    attributes are named differently, so this module degrades gracefully.
    """
    nx = getattr(grid, "nx", None)
    ny = getattr(grid, "ny", None)
    dx = getattr(grid, "dx", None)
    dy = getattr(grid, "dy", None)

    if nx is None or ny is None:
        x_unique = np.unique(np.round(grid.x_center, 6))
        y_unique = np.unique(np.round(grid.y_center, 6))
        nx = nx or len(x_unique)
        ny = ny or len(y_unique)
    if dx is None:
        dx = float(np.min(np.diff(np.unique(np.round(grid.x_center, 6))))) if nx > 1 else 1.0
    if dy is None:
        dy = float(np.min(np.diff(np.unique(np.round(grid.y_center, 6))))) if ny > 1 else 1.0

    return int(nx), int(ny), float(dx), float(dy)


def _field_to_2d(field: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """
    Reshape a flat per-cell array (ordering ``cell_index(i,j) = j*nx+i``,
    README §4.1) into a 2D (ny, nx) array suitable for ``imshow`` with
    ``origin="lower"`` (row j = y-index, column i = x-index).
    """
    values = np.asarray(field)
    return values.reshape(-1, ny, nx)[values.size // (nx * ny) // 2]


def well_markers(ctx: ReservoirModel) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return ([(i,j) for each producer], [(i,j) for each injector])."""
    nx, _, _, _ = grid_dims(ctx.grid)
    if ctx.wells is None:
        return [], []
    prod_xy = [ctx.grid.cell_ij(ctx.wells.producer_cell)]
    inj_xy = [ctx.grid.cell_ij(ctx.wells.injector_cell)]
    return prod_xy, inj_xy


def _annotate_wells(ax, ctx: ReservoirModel, dx: float, dy: float) -> None:
    prod_xy, inj_xy = well_markers(ctx)
    for k, (i, j) in enumerate(prod_xy):
        ax.plot(
            (i + 0.5) * dx, (j + 0.5) * dy, marker="v", color="red",
            markersize=11, markeredgecolor="black", linestyle="none",
            label="Producer" if k == 0 else None,
        )
    for k, (i, j) in enumerate(inj_xy):
        ax.plot(
            (i + 0.5) * dx, (j + 0.5) * dy, marker="^", color="blue",
            markersize=11, markeredgecolor="black", linestyle="none",
            label="Injector" if k == 0 else None,
        )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)


# ---------------------------------------------------------------------------
# Rock property maps
# ---------------------------------------------------------------------------
def plot_rock_maps(ctx: ReservoirModel, outdir: str) -> List[str]:
    """
    Plot porosity and permeability maps with well locations overlaid
    (README §7, §11).

    Returns list of saved file paths.
    """
    ensure_outdir(outdir)
    nx, ny, dx, dy = grid_dims(ctx.grid)
    paths: List[str] = []

    fields = [
        ("porosity", ctx.rock.phi0, "viridis", "Porosity [-]"),
        ("permeability", ctx.rock.perm, "viridis", "Permeability [md]"),
    ]
    for name, field, cmap, label in fields:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            _field_to_2d(field, nx, ny),
            origin="lower",
            cmap=cmap,
            extent=[0, nx * dx, 0, ny * dy],
            aspect="equal",
        )
        fig.colorbar(im, ax=ax, label=label)
        _annotate_wells(ax, ctx, dx, dy)
        ax.set_xlabel("x [ft]")
        ax.set_ylabel("y [ft]")
        ax.set_title(f"{name.capitalize()} map")
        fpath = os.path.join(outdir, f"map_{name}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(fpath)
    return paths


# ---------------------------------------------------------------------------
# Field snapshots
# ---------------------------------------------------------------------------
def plot_snapshots(
    snapshots: Dict[float, Dict[str, np.ndarray]],
    ctx: ReservoirModel,
    outdir: str,
    times: Sequence[float] = (30.0, 180.0, 365.0, 1000.0),
    tol: float = 1.0e-3,
) -> List[str]:
    """
    Plot p, Sw, Sg, So heatmaps for each requested reporting time that is
    present in ``snapshots`` (README §11: "snapshots of p, Sw, Sg, So at
    t = 30, 180, 365, 1000 days").

    Silently skips any requested time with no matching snapshot (e.g. a
    smoke-test run whose ``t_end`` is shorter than 1000 days).
    """
    ensure_outdir(outdir)
    nx, ny, dx, dy = grid_dims(ctx.grid)
    paths: List[str] = []

    available = sorted(snapshots.keys())
    variables = [
        ("p", "plasma", "Pressure [psi]"),
        ("Sw", "Blues", "Water saturation [-]"),
        ("Sg", "Reds", "Gas saturation [-]"),
        ("So", "Greens", "Oil saturation [-]"),
    ]

    for t_req in times:
        match = next((t for t in available if abs(t - t_req) < tol), None)
        if match is None:
            continue
        snap = snapshots[match]

        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
        for ax, (var, cmap, label) in zip(axes, variables):
            data = _field_to_2d(snap[var], nx, ny)
            im = ax.imshow(data, origin="lower", cmap=cmap, extent=[0, nx * dx, 0, ny * dy], aspect="equal")
            fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
            _annotate_wells(ax, ctx, dx, dy)
            ax.set_xlabel("x [ft]")
            ax.set_ylabel("y [ft]")
            ax.set_title(f"{var} @ t={match:.0f} d")

        fig.tight_layout()
        fpath = os.path.join(outdir, f"snapshot_t{int(round(match)):05d}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(fpath)

    return paths


# ---------------------------------------------------------------------------
# Time-series
# ---------------------------------------------------------------------------
def save_timeseries_csv(df: pd.DataFrame, outdir: str, filename: str = "timeseries.csv") -> str:
    """Save the run's per-step time-series DataFrame to CSV."""
    ensure_outdir(outdir)
    fpath = os.path.join(outdir, filename)
    df.to_csv(fpath, index=False)
    return fpath


def plot_timeseries(df: pd.DataFrame, outdir: str) -> List[str]:
    """
    Plot the standard suite of time-series diagnostics (README §11):
    average pressure; phase rates; water cut; GOR; cumulatives (Np, Wp, Gp);
    recovery factor; Newton iterations per step; dt history.
    """
    ensure_outdir(outdir)
    paths: List[str] = []
    t = df["t"].values

    def _line(y_cols, ylabel, title, fname, labels=None, logy=False):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        cols = y_cols if isinstance(y_cols, (list, tuple)) else [y_cols]
        lbls = labels if labels is not None else cols
        for col, lbl in zip(cols, lbls):
            ax.plot(t, df[col].values, label=lbl, linewidth=1.6)
        ax.set_xlabel("Time [days]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        if len(cols) > 1:
            ax.legend()
        ax.grid(alpha=0.3)
        fpath = os.path.join(outdir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(fpath)

    _line("avg_pressure", "Average reservoir pressure [psi]", "Average Pressure vs. Time", "ts_avg_pressure.png")
    _line(
        ["qo_prod", "qw_prod", "qg_prod"],
        "Rate",
        "Phase Production Rates vs. Time",
        "ts_rates.png",
        labels=["Oil [STB/d]", "Water [STB/d]", "Gas [SCF/d equiv.]"],
    )
    _line("qw_inj", "Injection rate [STB/day]", "Water Injection Rate vs. Time", "ts_injection.png")
    _line("water_cut", "Water cut [-]", "Water Cut vs. Time", "ts_water_cut.png")
    _line("gor", "GOR [SCF/STB]", "Producing GOR vs. Time", "ts_gor.png")
    _line(
        ["cum_oil_stb", "cum_water_stb"],
        "Cumulative volume [STB]",
        "Cumulative Production vs. Time",
        "ts_cumulative_liquid.png",
        labels=["Np (oil)", "Wp (water)"],
    )
    _line("cum_gas", "Cumulative gas produced", "Cumulative Gas Production (Gp) vs. Time", "ts_cumulative_gas.png")
    _line("recovery_factor", "Recovery factor [-]", "Oil Recovery Factor vs. Time", "ts_recovery_factor.png")
    _line("n_newton_iter", "Newton iterations", "Newton Iterations per Accepted Step", "ts_newton_iterations.png")
    _line("dt", "Time step size [days]", "Adaptive Time-Step History", "ts_dt_history.png", logy=True)

    return paths


# ---------------------------------------------------------------------------
# Newton convergence diagnostics
# ---------------------------------------------------------------------------
def plot_newton_convergence(
    scaled_residual_history: Sequence[float],
    outdir: str,
    filename: str = "newton_convergence.png",
    title: str = "Newton Convergence (single step)",
) -> str:
    """
    Semilog plot of scaled residual inf-norm vs. Newton iteration index
    (README §5.2 scaling, §11, and used by ``test_newton_convergence``).
    """
    ensure_outdir(outdir)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    iters = np.arange(len(scaled_residual_history))
    ax.semilogy(iters, np.asarray(scaled_residual_history) + 1.0e-300, marker="o")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("Scaled residual inf-norm [-]")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    fpath = os.path.join(outdir, filename)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


# ---------------------------------------------------------------------------
# Material balance bookkeeping (also used by tests)
# ---------------------------------------------------------------------------
def compute_fluid_in_place(x: np.ndarray, ctx: ReservoirModel) -> Dict[str, float]:
    """
    In-place fluid volumes for the given state, using the SAME accumulation
    formulas as the residual/Jacobian (README §4.3), evaluated as a state
    quantity rather than a time-difference:

        N (oil, STB)  = sum_i (Vb_i/5.615) * phi_i * So_i / Bo_i
        W (water, STB)= sum_i (Vb_i/5.615) * phi_i * Sw_i / Bw_i
        G (gas)       = sum_i (Vb_i/5.615) * phi_i * (Sg_i/Bg_i + Rs_i*So_i/Bo_i)

    (gas is reported in the same surface-volume gas unit implied by the
    Rs*So/Bo dissolved-gas term, consistent with README §4.3/§4.7).
    """
    masses = accumulation(ctx, x)
    W = float(masses[:, 0].sum())
    N = float(masses[:, 1].sum())
    G = float(masses[:, 2].sum())

    return {"oil_stb": N, "water_stb": W, "gas": G}


def mass_balance_summary(
    x_initial: np.ndarray,
    x_final: np.ndarray,
    ctx: ReservoirModel,
    df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Global mass-balance check over a full run (README §8, test 9 and
    §12 acceptance checklist): for each component,

        (in-place change) should equal (injected - produced)

    Returns a dict with in-place deltas, cumulative injected/produced
    volumes, absolute mismatch and relative error per component (oil,
    water, gas). Water injection contributes only to the water balance
    (README §6: injector is a source in the water residual only).
    """
    fip0 = compute_fluid_in_place(x_initial, ctx)
    fip1 = compute_fluid_in_place(x_final, ctx)

    cum_oil_prod = float(df["cum_oil_stb"].iloc[-1]) if len(df) else 0.0
    cum_water_prod = float(df["cum_water_stb"].iloc[-1]) if len(df) else 0.0
    cum_gas_prod = float(df["cum_gas"].iloc[-1]) if len(df) else 0.0
    cum_water_inj = float(df["cum_water_inj_stb"].iloc[-1]) if len(df) else 0.0

    d_oil = fip1["oil_stb"] - fip0["oil_stb"]
    d_water = fip1["water_stb"] - fip0["water_stb"]
    d_gas = fip1["gas"] - fip0["gas"]

    expected_d_oil = -cum_oil_prod
    expected_d_water = cum_water_inj - cum_water_prod
    expected_d_gas = -cum_gas_prod

    def _rel_err(actual: float, expected: float) -> float:
        denom = max(abs(expected), abs(actual), 1.0)
        return abs(actual - expected) / denom

    return {
        "oil_in_place_initial": fip0["oil_stb"],
        "oil_in_place_final": fip1["oil_stb"],
        "oil_delta": d_oil,
        "oil_cum_produced": cum_oil_prod,
        "oil_rel_error": _rel_err(d_oil, expected_d_oil),
        "water_in_place_initial": fip0["water_stb"],
        "water_in_place_final": fip1["water_stb"],
        "water_delta": d_water,
        "water_cum_produced": cum_water_prod,
        "water_cum_injected": cum_water_inj,
        "water_rel_error": _rel_err(d_water, expected_d_water),
        "gas_in_place_initial": fip0["gas"],
        "gas_in_place_final": fip1["gas"],
        "gas_delta": d_gas,
        "gas_cum_produced": cum_gas_prod,
        "gas_rel_error": _rel_err(d_gas, expected_d_gas),
    }


# ---------------------------------------------------------------------------
# Config / provenance dump
# ---------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    """Best-effort recursive conversion of a (possibly nested) config object
    into a JSON-serializable structure. Falls back to ``str(obj)`` for
    anything that cannot be represented natively (e.g. PVT/relperm param
    objects holding numpy arrays or callables)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= 64 else f"<ndarray shape={obj.shape}>"
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(obj).items()}
    return str(obj)


def save_run_config(cfg: Any, outdir: str, seed: Optional[int] = None, filename: str = "run_config.json") -> str:
    """
    Persist the run configuration (and, if provided, the top-level RNG
    seed) alongside results, per the reproducibility requirement in
    README §2 ("save config + seeds with results").
    """
    ensure_outdir(outdir)
    payload = {"config": _jsonable(cfg), "seed": seed}
    fpath = os.path.join(outdir, filename)
    with open(fpath, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return fpath


# ---------------------------------------------------------------------------
# Fractal study comparison plots (README §7, §11)
# ---------------------------------------------------------------------------
def plot_fractal_study(study_df: pd.DataFrame, outdir: str) -> List[str]:
    """
    Compare recovery factor, breakthrough time, water cut and average
    pressure across the swept beta values of ``run_fractal_study.py``
    (README §7).

    Expects ``study_df`` with (at least) columns: "beta", "recovery_factor",
    "breakthrough_time_days", "water_cut_final", "avg_pressure_final".
    Adds/uses the fractal-dimension convention D_f = (8 - beta) / 2
    (README §7).
    """
    ensure_outdir(outdir)
    paths: List[str] = []
    df = study_df.sort_values("beta").copy()
    df["Df"] = (8.0 - df["beta"]) / 2.0

    def _scatter(xcol, xlabel, ycol, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(df[xcol], df[ycol], marker="o", linewidth=1.6)
        for _, row in df.iterrows():
            ax.annotate(f"β={row['beta']:.1f}", (row[xcol], row[ycol]), fontsize=8,
                        textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        fpath = os.path.join(outdir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(fpath)

    _scatter("beta", "Spectral exponent β", "recovery_factor", "Recovery factor [-]",
              "Recovery Factor vs. β", "study_rf_vs_beta.png")
    _scatter("Df", "Fractal dimension D_f = (8-β)/2", "recovery_factor", "Recovery factor [-]",
              "Recovery Factor vs. D_f", "study_rf_vs_Df.png")
    _scatter("beta", "Spectral exponent β", "breakthrough_time_days", "Breakthrough time [days]",
              "Water Breakthrough Time vs. β", "study_breakthrough_vs_beta.png")
    _scatter("beta", "Spectral exponent β", "water_cut_final", "Final water cut [-]",
              "Final Water Cut vs. β", "study_watercut_vs_beta.png")
    _scatter("beta", "Spectral exponent β", "avg_pressure_final", "Final average pressure [psi]",
              "Final Average Pressure vs. β", "study_pressure_vs_beta.png")

    return paths


# ---------------------------------------------------------------------------
# Benchmark (baseline vs RF) plots (README §10, §11)
# ---------------------------------------------------------------------------
def plot_benchmark_results(benchmark_df: pd.DataFrame, outdir: str) -> List[str]:
    """
    Box/bar plots comparing baseline (previous-state init) vs. RF-init
    Newton performance across held-out benchmark cases (README §10).

    Expects ``benchmark_df`` in "long" form with columns:
        "case_id", "mode" ("baseline"/"rf"), "mean_newton_iterations",
        "total_newton_iterations", "wall_time_s", "n_dt_cuts".
    """
    ensure_outdir(outdir)
    paths: List[str] = []
    modes = sorted(benchmark_df["mode"].unique())

    def _box(col, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        data = [benchmark_df.loc[benchmark_df["mode"] == m, col].values for m in modes]
        ax.boxplot(data, labels=modes, showmeans=True)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        fpath = os.path.join(outdir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(fpath)

    _box("mean_newton_iterations", "Mean Newton iterations/step",
         "Newton Iterations: Baseline vs. RF-Initialized", "bench_newton_iters.png")
    _box("wall_time_s", "Wall time [s]",
         "Wall Time: Baseline vs. RF-Initialized (incl. RF inference)", "bench_wall_time.png")
    _box("n_dt_cuts", "Time-step cuts", "Time-Step Failures: Baseline vs. RF-Initialized",
         "bench_dt_cuts.png")

    if {"case_id"}.issubset(benchmark_df.columns) and set(modes) == {"baseline", "rf"}:
        pivot = benchmark_df.pivot_table(index="case_id", columns="mode", values="wall_time_s")
        pivot["speedup"] = pivot["baseline"] / pivot["rf"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(pivot.index.astype(str), pivot["speedup"].values, color="steelblue")
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Speedup = 1x (break-even)")
        ax.set_xlabel("Benchmark case")
        ax.set_ylabel("Speedup (t_baseline / t_RF)")
        ax.set_title("RF-Initialized Newton Speedup by Case")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fpath = os.path.join(outdir, "bench_speedup.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(fpath)

    return paths


def plot_rf_error_distributions(errors: Dict[str, np.ndarray], outdir: str) -> List[str]:
    """
    Histograms of per-cell RF prediction errors for each target
    (p, Sw, Sg), used by both ``train_rf.py`` (test-set errors) and
    ``run_benchmark.py`` (solution deltas vs. baseline), README §9.4/§10.
    """
    ensure_outdir(outdir)
    paths: List[str] = []
    for name, arr in errors.items():
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.hist(np.asarray(arr).ravel(), bins=60, color="slategray", edgecolor="black", alpha=0.8)
        ax.set_xlabel(f"{name} error")
        ax.set_ylabel("Count")
        ax.set_title(f"RF Prediction Error Distribution: {name}")
        ax.grid(alpha=0.3)
        fpath = os.path.join(outdir, f"rf_error_hist_{name}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(fpath)
    return paths


# ---------------------------------------------------------------------------
# Top-level orchestration for a single MVP-style run
# ---------------------------------------------------------------------------
def generate_mvp_report(
    sim_output: Dict[str, Any],
    outdir: str,
    snapshot_times: Sequence[float] = (30.0, 180.0, 365.0, 1000.0),
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience entry point: given the dict returned by
    ``simulator.run_simulation``, produce the full standard set of plots,
    CSVs, and a mass-balance report (README §11), all under ``outdir``.

    Returns a dict summarizing everything written (paths + mass-balance
    numbers) for logging / assertions in ``run_mvp.py``.
    """
    ensure_outdir(outdir)
    ctx = sim_output["context"]
    df = sim_output["timeseries"]
    snapshots = sim_output["snapshots"]
    summary = sim_output["summary"]
    cfg = sim_output["config"]

    written: Dict[str, Any] = {}
    if plt is None:
        written["rock_maps"] = []
        written["snapshots"] = []
        written["timeseries_plots"] = []
    else:
        written["rock_maps"] = plot_rock_maps(ctx, outdir)
        written["snapshots"] = plot_snapshots(snapshots, ctx, outdir, times=snapshot_times)
        written["timeseries_plots"] = plot_timeseries(df, outdir)
    written["timeseries_csv"] = save_timeseries_csv(df, outdir)
    written["config_json"] = save_run_config(cfg, outdir, seed=seed)

    x0 = None
    if 0.0 in snapshots:
        s0 = snapshots[0.0]
        from residual import pack_state

        x0 = pack_state(s0["p"], s0["Sw"], s0["Sg"])
    x_final = summary["final_state"]

    if x0 is not None:
        mb = mass_balance_summary(x0, x_final, ctx, df)
        written["mass_balance"] = mb
        mb_path = os.path.join(outdir, "mass_balance_report.json")
        with open(mb_path, "w") as fh:
            json.dump(mb, fh, indent=2)
        written["mass_balance_json"] = mb_path

    return written