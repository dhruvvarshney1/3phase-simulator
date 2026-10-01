"""config.py -- Centralized configuration for the black-oil simulator.

All tunable parameters live here as immutable (frozen) dataclasses. Modify a
configuration with :func:`override` (or :func:`dataclasses.replace`) rather than
mutating it, so that every run is reproducible and can be saved with its
seeds via :func:`save_config`.

Factories
---------
``default_config()``     -- legacy MVP configuration (30x30, 1000 days).
``synthetic_config()``   -- primary geological-flow configuration.
``smoke_test_config()``  -- 15x15, 200 days, fewer RF cases (< ~60 s end-to-end).
``case_config(base, params)`` -- apply one Latin-hypercube sample to a base config.

Run ``python config.py [--smoke] [--save path.json]`` to print/validate/save.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, get_type_hints


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GridConfig:
    """Cartesian grid (README section 4.1). Uniform spacing; k = 0 is the top layer."""

    nx: int = 30
    ny: int = 30
    nz: int = 3
    dx: float = 50.0   # ft
    dy: float = 50.0   # ft
    dz: float = 20.0   # ft


@dataclass(frozen=True)
class InitialStateConfig:
    """Uniform initial state."""

    p_init: float = 4000.0   # psi
    sw_init: float = 0.25
    sg_init: float = 0.0


@dataclass(frozen=True)
class PVTConfig:
    """PVT coefficients (README section 4.4).

    B_o = bo_ref * exp(-bo_comp * (p - p_ref))
    B_w = bw_ref * exp(-bw_comp * (p - p_ref))
    B_g = bg_coeff * (bg_p_ref / max(p, bg_p_floor))
    mu_o = muo_ref * exp(-muo_slope * (p - p_ref))
    R_s = rs_bubble for p >= p_bubble, else rs_bubble * p / p_bubble
    """

    p_ref: float = 4000.0
    bo_ref: float = 1.20
    bo_comp: float = 1.0e-5
    bw_ref: float = 1.00
    bw_comp: float = 3.0e-6
    bg_coeff: float = 0.005
    bg_p_ref: float = 4000.0
    bg_p_floor: float = 100.0
    muo_ref: float = 1.5
    muo_slope: float = 2.0e-5
    muw: float = 0.5
    mug: float = 0.02
    p_bubble: float = 2500.0
    rs_bubble: float = 600.0   # SCF/STB
    # Constant phase gravity gradients rho*g [psi/ft]. 0 = gravity off (legacy
    # uniform-state configs); initial_state_model.py sets them from the densities
    # used to build the hydrostatic initial state so that state is in equilibrium.
    gamma_w: float = 0.0
    gamma_o: float = 0.0
    gamma_g: float = 0.0


@dataclass(frozen=True)
class RelPermConfig:
    """Corey relative permeability (README section 4.5)."""

    swc: float = 0.20
    sor: float = 0.20
    sgc: float = 0.05
    krw_max: float = 0.30
    krw_exp: float = 3.0
    kro_max: float = 0.90
    kro_exp: float = 2.0
    krg_max: float = 0.80
    krg_exp: float = 2.0


@dataclass(frozen=True)
class RockConfig:
    """Rock properties (README section 4.6 and 7).

    Permeability from porosity: k = k_ref * (phi / phi_ref)**k_exp, clipped.
    """

    use_fractal: bool = True
    phi_const: float = 0.18
    pressure_dependent: bool = False   # phi(p) = phi_ref*(1 + c_r*(p - p_ref))
    c_r: float = 3.0e-6                # 1/psi
    p_ref: float = 4000.0              # psi
    k_ref: float = 100.0               # md at phi_ref
    phi_ref: float = 0.18
    k_exp: float = 4.0
    k_min: float = 0.1                 # md
    k_max: float = 5000.0              # md


@dataclass(frozen=True)
class FractalConfig:
    """Spectral fBm porosity field (README section 6)."""

    phi_mean: float = 0.18
    phi_std: float = 0.035
    beta: float = 3.0
    seed: int = 42
    phi_min: float = 0.05
    phi_max: float = 0.30


@dataclass(frozen=True)
class WellConfig:
    """Wells (README section 6). Cell indices are zero-based (i, j)."""

    producer_bhp: float = 1500.0                      # psi
    producer_ij: Tuple[int, int] | None = None        # default (nx-3, ny-3)
    producer_k: int = 1                               # zero-based layer
    injector_rate: float = 400.0                      # STB/day water
    injector_ij: Tuple[int, int] = (2, 2)
    injector_k: int = 1                               # zero-based layer
    r_w: float = 0.25                                 # ft
    skin: float = 0.0
    r_e_factor: float = 0.14                          # r_e = factor*sqrt(dx^2+dy^2)
    producer_no_backflow: bool = True                 # clamp flow reversal

    def producer_cell(self, nx: int, ny: int, nz: int = 1) -> Tuple[int, int, int]:
        """Return the producer (i, j, k), defaulting near the far corner."""
        i, j = self.producer_ij if self.producer_ij is not None else (nx - 3, ny - 3)
        return i, j, min(max(self.producer_k, 0), nz - 1)

    def injector_cell(self, nz: int = 1) -> Tuple[int, int, int]:
        """Return the injector (i, j, k) location."""
        i, j = self.injector_ij
        return i, j, min(max(self.injector_k, 0), nz - 1)


@dataclass(frozen=True)
class NewtonConfig:
    """Newton solver (README section 5.4) and convergence scaling (section 5.2)."""

    tol: float = 1.0e-6                # scaled residual inf-norm tolerance
    max_iter: int = 12
    alpha_init: float = 1.0
    alpha_min: float = 0.1
    alpha_shrink: float = 0.5
    dp_tol_rel: float = 1.0e-6         # max|dp| / p_scale
    ds_tol: float = 1.0e-6             # max|dS|
    p_scale: float = 4000.0            # psi, reference for pressure update test
    p_min: float = 100.0               # psi, hard lower bound
    viol_tol_p_frac: float = 0.10      # "substantial" pressure bound violation
    viol_tol_sat: float = 0.05         # "substantial" saturation bound violation
    char_rate_floor: float = 1.0       # STB/day floor of the characteristic rate
    jacobian: str = "analytical"       # "analytical" | "numerical"


@dataclass(frozen=True)
class TimeStepConfig:
    """Schedule and adaptive time stepping (README section 5.5).

    ``stages`` rows are (t_start, t_stop, dt_init, dt_max) in days.
    """

    t_end: float = 1000.0
    stages: Tuple[Tuple[float, float, float, float], ...] = (
        (0.0, 30.0, 1.0, 1.0),
        (30.0, 180.0, 5.0, 5.0),
        (180.0, 1000.0, 10.0, 30.0),
    )
    report_times: Tuple[float, ...] = (30.0, 180.0, 365.0, 1000.0)
    dt_min: float = 0.01
    max_cuts: int = 5
    cut_factor: float = 0.5
    grow_factor: float = 1.5
    shrink_factor: float = 0.5
    iters_easy: int = 4      # converged in <= iters_easy -> grow dt
    iters_hard: int = 8      # converged in >= iters_hard -> shrink dt

    def stage_for_time(self, t: float) -> Tuple[float, float]:
        """Return (dt_init, dt_max) of the stage containing time ``t``."""
        eps = 1.0e-12
        for t0, t1, dt_init, dt_max in self.stages:
            if t0 - eps <= t < t1 - eps:
                return dt_init, dt_max
        last = self.stages[-1]
        return last[2], last[3]


@dataclass(frozen=True)
class RFBounds:
    """Latin-hypercube bounds (README section 9.2), each as (low, high)."""

    p_init: Tuple[float, float] = (3500.0, 4500.0)
    beta: Tuple[float, float] = (2.0, 5.0)
    phi_mean: Tuple[float, float] = (0.15, 0.22)
    k_ref: Tuple[float, float] = (50.0, 300.0)
    inj_rate: Tuple[float, float] = (300.0, 500.0)
    bhp: Tuple[float, float] = (1000.0, 2500.0)

    def ordered(self) -> List[Tuple[str, float, float]]:
        """Return [(name, low, high), ...] in the canonical sampling order."""
        names = ("p_init", "beta", "phi_mean", "k_ref", "inj_rate", "bhp")
        return [(n, getattr(self, n)[0], getattr(self, n)[1]) for n in names]


@dataclass(frozen=True)
class RFConfig:
    """Random-Forest surrogate settings (README section 9)."""

    n_train: int = 60
    n_bench: int = 10
    lhs_seed: int = 2024
    bench_seed: int = 9001
    bounds: RFBounds = field(default_factory=RFBounds)
    n_estimators: int = 200
    max_depth: int = 20
    random_state: int = 42
    test_fraction: float = 0.2
    dataset_path: str = "data/rf_dataset.npz"
    model_dir: str = "results/rf_models"


@dataclass(frozen=True)
class OutputConfig:
    """Output locations and figure settings."""

    results_dir: str = "results"
    data_dir: str = "data"
    dpi: int = 150
    save_config: bool = True


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level configuration aggregating every section."""

    name: str = "mvp"
    init_mode: str = "previous"        # Newton initial guess: "previous" | "rf"
    grid: GridConfig = field(default_factory=GridConfig)
    init: InitialStateConfig = field(default_factory=InitialStateConfig)
    pvt: PVTConfig = field(default_factory=PVTConfig)
    relperm: RelPermConfig = field(default_factory=RelPermConfig)
    rock: RockConfig = field(default_factory=RockConfig)
    fractal: FractalConfig = field(default_factory=FractalConfig)
    wells: WellConfig = field(default_factory=WellConfig)
    newton: NewtonConfig = field(default_factory=NewtonConfig)
    timestep: TimeStepConfig = field(default_factory=TimeStepConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def default_config() -> SimulationConfig:
    """Return the small, uniform-state MVP configuration used by unit tests."""
    return SimulationConfig()


def synthetic_config() -> SimulationConfig:
    """Return solver settings for the primary synthetic geological case.

    The geological grid and state are supplied by ``initial_state_model``;
    this config carries the black-oil, Newton, timestep, and output settings.
    """
    return SimulationConfig(name="synthetic")


def smoke_test_config() -> SimulationConfig:
    """Return a fast configuration: 15x15 grid, 200 days, fewer RF cases."""
    base = SimulationConfig(name="smoke")
    return dataclasses.replace(
        base,
        grid=dataclasses.replace(base.grid, nx=15, ny=15),
        timestep=dataclasses.replace(
            base.timestep,
            t_end=200.0,
            stages=((0.0, 30.0, 1.0, 1.0), (30.0, 180.0, 5.0, 5.0), (180.0, 200.0, 10.0, 30.0)),
            report_times=(30.0, 180.0, 200.0),
        ),
        rf=dataclasses.replace(
            base.rf,
            n_train=12,
            n_bench=3,
            n_estimators=50,
            max_depth=16,
            dataset_path="data/rf_dataset_smoke.npz",
            model_dir="results/rf_models_smoke",
        ),
        output=dataclasses.replace(base.output, results_dir="results/smoke"),
    )


def override(cfg: SimulationConfig, section: str, **kwargs: Any) -> SimulationConfig:
    """Return a copy of ``cfg`` with fields of one section replaced.

    Example: ``override(cfg, "wells", injector_rate=300.0)``. Use
    ``section=""`` to replace top-level fields (``name``, ``init_mode``).
    """
    if section == "":
        return dataclasses.replace(cfg, **kwargs)
    if not hasattr(cfg, section):
        raise KeyError(f"unknown config section '{section}'")
    new_section = dataclasses.replace(getattr(cfg, section), **kwargs)
    return dataclasses.replace(cfg, **{section: new_section})


def case_config(base: SimulationConfig, params: Mapping[str, float],
                name: str | None = None) -> SimulationConfig:
    """Apply one Latin-hypercube sample to ``base`` (README section 9.2).

    ``params`` keys: p_init, beta, phi_mean, k_ref, inj_rate, bhp.
    """
    known = {"p_init", "beta", "phi_mean", "k_ref", "inj_rate", "bhp"}
    unknown = set(params) - known
    if unknown:
        raise KeyError(f"unknown case parameters: {sorted(unknown)}")
    cfg = base
    if "p_init" in params:
        cfg = override(cfg, "init", p_init=float(params["p_init"]))
    fractal_kw = {}
    if "beta" in params:
        fractal_kw["beta"] = float(params["beta"])
    if "phi_mean" in params:
        fractal_kw["phi_mean"] = float(params["phi_mean"])
    if fractal_kw:
        cfg = override(cfg, "fractal", **fractal_kw)
    if "k_ref" in params:
        cfg = override(cfg, "rock", k_ref=float(params["k_ref"]))
    well_kw = {}
    if "inj_rate" in params:
        well_kw["injector_rate"] = float(params["inj_rate"])
    if "bhp" in params:
        well_kw["producer_bhp"] = float(params["bhp"])
    if well_kw:
        cfg = override(cfg, "wells", **well_kw)
    if name is not None:
        cfg = override(cfg, "", name=name)
    return cfg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_config(cfg: SimulationConfig, check_wells: bool = True) -> None:
    """Raise ValueError if the configuration is inconsistent.

    ``check_wells=False`` skips the well checks (used for well-less test models).
    """
    g, s, rp, w, ts, nw = cfg.grid, cfg.init, cfg.relperm, cfg.wells, cfg.timestep, cfg.newton

    if g.nx < 1 or g.ny < 1 or g.nz < 1:
        raise ValueError("nx, ny, and nz must be >= 1")
    if min(g.dx, g.dy, g.dz) <= 0.0:
        raise ValueError("cell dimensions must be positive")

    if s.p_init <= nw.p_min:
        raise ValueError("p_init must exceed the pressure lower bound")
    if not (rp.swc <= s.sw_init <= 1.0) or s.sg_init < 0.0:
        raise ValueError("initial saturations out of range")
    if s.sw_init + s.sg_init > 1.0 - rp.sor + 1e-12:
        raise ValueError("initial S_w + S_g must not exceed 1 - S_or")
    if rp.swc + rp.sor + rp.sgc >= 1.0:
        raise ValueError("relperm endpoints leave no mobile saturation range")

    f = cfg.fractal
    if not (f.phi_min < f.phi_max) or f.phi_std < 0.0 or f.beta <= 0.0:
        raise ValueError("invalid fractal porosity parameters")
    if cfg.rock.k_min >= cfg.rock.k_max or cfg.rock.k_ref <= 0.0:
        raise ValueError("invalid permeability bounds")

    if check_wells:
        pi, pj, pk = w.producer_cell(g.nx, g.ny, g.nz)
        ii, ij, ik = w.injector_cell(g.nz)
        for label, (ci, cj, ck) in (("producer", (pi, pj, pk)), ("injector", (ii, ij, ik))):
            if not (0 <= ci < g.nx and 0 <= cj < g.ny and 0 <= ck < g.nz):
                raise ValueError(f"{label} cell ({ci}, {cj}, {ck}) lies outside the grid")
        if (pi, pj, pk) == (ii, ij, ik):
            raise ValueError("producer and injector must be in different cells")
        if w.producer_bhp <= nw.p_min or w.injector_rate < 0.0:
            raise ValueError("invalid well controls")
        if w.r_w <= 0.0:
            raise ValueError("wellbore radius must be positive")

    if ts.t_end <= 0.0 or not ts.stages:
        raise ValueError("invalid time schedule")
    if abs(ts.stages[0][0]) > 1e-12 or abs(ts.stages[-1][1] - ts.t_end) > 1e-9:
        raise ValueError("schedule stages must start at 0 and end at t_end")
    for a, b in zip(ts.stages[:-1], ts.stages[1:]):
        if abs(a[1] - b[0]) > 1e-9:
            raise ValueError("schedule stages must be contiguous")
    for t0, t1, dt_init, dt_max in ts.stages:
        if not (t1 > t0 and 0.0 < dt_init <= dt_max):
            raise ValueError("invalid schedule stage")
    rts = list(ts.report_times)
    if rts != sorted(rts) or any(t <= 0.0 or t > ts.t_end + 1e-9 for t in rts):
        raise ValueError("report_times must be increasing and within (0, t_end]")
    if not (0.0 < ts.dt_min < ts.stages[0][2]):
        raise ValueError("dt_min must be positive and below the first time step")

    if nw.jacobian not in ("analytical", "numerical"):
        raise ValueError("newton.jacobian must be 'analytical' or 'numerical'")
    if not (0.0 < nw.alpha_min <= nw.alpha_init <= 1.0):
        raise ValueError("damping factors must satisfy 0 < alpha_min <= alpha_init <= 1")
    if cfg.init_mode not in ("previous", "rf"):
        raise ValueError("init_mode must be 'previous' or 'rf'")

    for name, lo, hi in cfg.rf.bounds.ordered():
        if not lo < hi:
            raise ValueError(f"RF bound '{name}' must satisfy low < high")
    if not (0.0 < cfg.rf.test_fraction < 1.0):
        raise ValueError("rf.test_fraction must lie in (0, 1)")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def config_to_dict(cfg: SimulationConfig) -> Dict[str, Any]:
    """Return a JSON-serializable nested dict of ``cfg``."""
    return dataclasses.asdict(cfg)


def _tuplify(value: Any) -> Any:
    """Recursively convert lists (from JSON) back to tuples."""
    if isinstance(value, list):
        return tuple(_tuplify(v) for v in value)
    return value


def _build(cls: type, data: Mapping[str, Any]) -> Any:
    """Recursively construct dataclass ``cls`` from a nested mapping."""
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints[f.name]
        if dataclasses.is_dataclass(hint) and isinstance(value, Mapping):
            kwargs[f.name] = _build(hint, value)
        else:
            kwargs[f.name] = _tuplify(value)
    return cls(**kwargs)


def config_from_dict(data: Mapping[str, Any]) -> SimulationConfig:
    """Rebuild a :class:`SimulationConfig` from :func:`config_to_dict` output."""
    return _build(SimulationConfig, data)


def config_fingerprint(cfg: SimulationConfig) -> str:
    """Return a short SHA-256 fingerprint of the configuration."""
    blob = json.dumps(config_to_dict(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def save_config(cfg: SimulationConfig, path: str | Path) -> Path:
    """Write ``cfg`` (including all seeds) plus its fingerprint to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fingerprint": config_fingerprint(cfg), "config": config_to_dict(cfg)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_config(path: str | Path) -> SimulationConfig:
    """Load a configuration written by :func:`save_config`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    data = payload["config"] if "config" in payload else payload
    return config_from_dict(data)


def ensure_dirs(cfg: SimulationConfig) -> None:
    """Create the results and data directories named in ``cfg``."""
    Path(cfg.output.results_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output.data_dir).mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Validate, print and optionally save the default or smoke configuration."""
    parser = argparse.ArgumentParser(description="Show/validate/save a simulator config")
    parser.add_argument("--smoke", action="store_true", help="use the smoke-test config")
    parser.add_argument("--save", type=str, default=None, help="write config JSON to this path")
    args = parser.parse_args()

    cfg = smoke_test_config() if args.smoke else default_config()
    validate_config(cfg)
    print(json.dumps(config_to_dict(cfg), indent=2))
    print(f"fingerprint: {config_fingerprint(cfg)}")
    if args.save:
        print(f"saved to {save_config(cfg, args.save)}")


if __name__ == "__main__":
    main()
