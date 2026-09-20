"""
run_mvp.py
==========

Command-line entry point for the default MVP waterflood simulation
(README §12), and for the fast smoke-test configuration used for CI /
quick sanity checks (README §12: "runs end-to-end in under ~60 seconds").

Usage
-----
    python run_mvp.py                              # full MVP, previous-state init
    python run_mvp.py --mode smoke                 # fast smoke test
    python run_mvp.py --init rf --rf-dir results/rf_models
    python run_mvp.py --jacobian numerical --mode smoke   # debugging only

This script is intentionally the single "does everything work end to end"
integration entry point referenced by the acceptance checklist:
"run_mvp.py completes 1000 days with global mass-balance error < 1e-6
relative" and "Newton converges within 12 iterations on every step ...
average <= 6".
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Optional

import numpy as np

from config import default_config, smoke_test_config, override
from simulator import run_simulation
from postprocess import generate_mvp_report


def _load_rf_model(rf_dir: str) -> Any:
    """
    Lazily import and load the trained RF surrogate (README §9). Import is
    deferred here (rather than at module scope) so that running the default
    ``--init previous`` MVP mode never requires scikit-learn / a trained
    model directory to be present.
    """
    from rf_surrogate import RFSurrogate

    return RFSurrogate.load(rf_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the black-oil MVP waterflood simulation (README §12)."
    )
    parser.add_argument(
        "--mode", choices=["mvp", "smoke"], default="mvp",
        help="'mvp': full 30x30, 1000-day default run. "
             "'smoke': fast 15x15, 200-day sanity-check run (README §12).",
    )
    parser.add_argument(
        "--init", choices=["previous", "rf"], default="previous",
        help="Newton initial-guess strategy (README §9). 'previous': warm "
             "start from x^n (baseline). 'rf': Random Forest surrogate "
             "prediction (requires --rf-dir pointing at trained models).",
    )
    parser.add_argument(
        "--rf-dir", type=str, default="results/rf_models",
        help="Directory containing trained RF surrogate models "
             "(joblib files produced by scripts/train_rf.py).",
    )
    parser.add_argument(
        "--jacobian", choices=["analytical", "numerical"], default="analytical",
        help="Jacobian assembly method (README §5.3). 'numerical' is for "
             "debugging only and is far too slow for the full MVP grid.",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory for plots/CSVs/reports "
             "(default: results/mvp_<mode>_<init>).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the config's default RNG seed (fractal field, RF, etc.).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-step progress logging.",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = default_config() if args.mode == "mvp" else smoke_test_config()

    seed = args.seed if args.seed is not None else getattr(cfg, "seed", 42)
    cfg = override(cfg, "fractal", seed=seed)
    rng = np.random.default_rng(seed)  # noqa: F841  (kept for provenance / future use)

    outdir = args.outdir or f"results/mvp_{args.mode}_{args.init}"

    rf_model = None
    if args.init == "rf":
        try:
            rf_model = _load_rf_model(args.rf_dir)
        except Exception as exc:
            print(
                f"[run_mvp] ERROR: could not load RF surrogate from '{args.rf_dir}': {exc}\n"
                "Train it first with:\n"
                "    python scripts/generate_rf_dataset.py\n"
                "    python scripts/train_rf.py",
                file=sys.stderr,
            )
            return 1

    print(f"[run_mvp] mode={args.mode} init={args.init} jacobian={args.jacobian} seed={seed}")
    t_start = time.perf_counter()

    sim_output = run_simulation(
        cfg,
        init_guess_mode=args.init,
        rf_model=rf_model,
        jacobian_method=args.jacobian,
        verbose=not args.quiet,
    )

    wall_time = time.perf_counter() - t_start
    summary = sim_output["summary"]

    print(f"[run_mvp] Completed {summary['n_steps']} time steps in {wall_time:.2f} s wall time.")
    print(
        f"[run_mvp] Total Newton iterations: {summary['total_newton_iterations']} "
        f"(mean {summary['mean_newton_iterations']:.2f}/step); "
        f"dt cuts: {summary['total_dt_cuts']}"
    )

    report = generate_mvp_report(sim_output, outdir, seed=seed)

    if "mass_balance" in report:
        mb = report["mass_balance"]
        print(
            "[run_mvp] Global mass balance relative errors: "
            f"oil={mb['oil_rel_error']:.3e}  water={mb['water_rel_error']:.3e}  "
            f"gas={mb['gas_rel_error']:.3e}"
        )
        worst = max(mb["oil_rel_error"], mb["water_rel_error"], mb["gas_rel_error"])
        if worst < 1.0e-6:
            print("[run_mvp] Mass balance check PASSED (< 1e-6 relative error).")
        else:
            print(
                f"[run_mvp] WARNING: mass balance relative error {worst:.3e} "
                "exceeds the 1e-6 target (README §12 acceptance checklist)."
            )

    print(f"[run_mvp] All outputs written to: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())