# Three-Phase Black-Oil Reservoir Simulator

A Python 3 fully implicit, three-phase black-oil waterflood simulator. The primary scientific path starts from a synthetic geological reservoir and carries structure, facies, heterogeneous porosity, anisotropic permeability, faults, equivalent-continuum fractures, hydrostatic initialization, and geological multi-perforation wells into the flow solver.

The numerical flow model uses backward Euler, analytical Jacobians, Newton-Raphson iteration, adaptive timesteps, Corey relative permeability, simple black-oil PVT, gravity, Peaceman wells, and surface-volume material balance.

This is a synthetic research and validation codebase. The geological ranges are illustrative and are not calibrated to a real reservoir.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- pandas
- matplotlib
- scikit-learn and joblib for the optional RF workflow
- pytest for tests

## Setup

From the repository root:

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either run the commands from `cmd.exe` or use:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the installation:

```bash
python -m pytest -q
```

The full suite includes one deliberately slow full-MVP material-balance test. A passing run currently contains 23 tests.

## Quick start

The default command runs the synthetic geological configuration:

```bash
python run_mvp.py
```

For a short validation run:

```bash
python run_mvp.py --days 1 --quiet --outdir results/quick_synthetic
```

The command reports timestep count, Newton iterations, timestep cuts, and global oil/water/gas balance errors. Results are written below the selected output directory.

## Command-line options

```text
--mode {synthetic,mvp,smoke}
--init {previous,rf}
--rf-dir PATH
--jacobian {analytical,numerical}
--outdir PATH
--seed INTEGER
--days NUMBER
--quiet
--live
```

Examples:

```bash
# Primary geological waterflood
python run_mvp.py --mode synthetic --days 7

# Fast simplified sanity check
python run_mvp.py --mode smoke --days 20 --quiet

# Legacy uniform/fractal MVP path
python run_mvp.py --mode mvp --days 30

# Debugging only: finite-difference Jacobian
python run_mvp.py --mode smoke --jacobian numerical --days 1

# Reproducible geological realization
python run_mvp.py --mode synthetic --seed 123 --days 7

# Live saturation display; requires a graphical environment
python run_mvp.py --mode synthetic --days 7 --live
```

The numerical Jacobian is intended for small debugging cases. It is much slower than the analytical Jacobian and should not be used for production-size runs.

## Scientific execution path

```text
_initial_state_.py
  → structure/depth and facies
  → porosity and kx/ky/kz
  → fault face multipliers
  → equivalent-continuum fracture enhancement
  → hydrostatic pressure and OWC saturations
  → geological wells and perforations
  → initial_state_model.py unit/order adapter
  → residual.py fully implicit black-oil model
  → jacobian.py analytical Jacobian
  → newton_solver.py
  → timestep.py adaptive timestep control
  → simulator.py and postprocess.py
```

The geological generator uses arrays shaped `(Nx, Ny, Nz)`. The flow grid stores cells in flat order:

```text
cell_index(i, j, k) = k * Ny * Nx + j * Nx + i
```

The adapter performs the mapping with:

```python
array.transpose(2, 1, 0).ravel()
```

The geological model is expressed primarily in SI units. The flow solver uses field units. Conversion occurs at `initial_state_model.py`, including metres/feet, pascals/psi, cubic metres/day/STB/day, and hydrostatic density gradients/psi per foot. Permeability remains in mD for the field-unit Darcy equations.

## Configuration

Configuration is defined by frozen dataclasses in `config.py`.

Useful factories:

```python
from config import default_config, smoke_test_config, synthetic_config, override

cfg = synthetic_config()
cfg = override(cfg, "fractal", beta=4.0, seed=123)
cfg = override(cfg, "timestep", t_end=30.0)
```

Inspect or validate the legacy/smoke configuration:

```bash
python config.py
python config.py --smoke
python config.py --smoke --save results/smoke_config.json
```

Important solver controls include:

- `cfg.newton.tol`: scaled residual tolerance
- `cfg.newton.max_iter`: Newton iteration limit
- `cfg.newton.p_scale`: pressure-update reference
- `cfg.newton.jacobian`: `analytical` or `numerical`
- `cfg.timestep.stages`: `(start, stop, initial_dt, max_dt)` schedule
- `cfg.timestep.max_cuts`: retries after Newton failure
- `cfg.fractal.beta`, `phi_mean`, `phi_std`, and `seed`
- `cfg.wells.injector_rate` and `producer_bhp`

The primary synthetic path passes `cfg.fractal.beta` and `cfg.fractal.seed` to the geological fractal realization when it builds a reservoir internally. For controlled tests, an explicit geological reservoir dictionary can be passed to `initial_state_model.build_from_initial_state`.

## Generated outputs

`run_mvp.py` writes figures, CSV data, configuration data, and reports under the output directory. Typical outputs include:

- `timeseries.csv`
- `run_config.json`
- `mass_balance_report.json`
- pressure, saturation, permeability, and porosity figures
- cumulative oil/water/gas plots
- water cut, GOR, recovery factor, timestep, and Newton-iteration plots

The in-memory simulator result contains:

- `timeseries`: pandas table of rates and cumulative metrics
- `snapshots`: saved states at report times
- `summary`: timestep, Newton, mass, and final-state diagnostics
- `context`: the assembled `ReservoirModel`

## Synthetic geological model tools

Run the geological model self-check directly:

```bash
python initial_state_model.py
```

The self-check validates cell ordering, unit conversion, fault-face placement, hydrostatic oil equilibrium, saturation bounds, and a no-well Newton step.

The lower-level generator can also create serialized geological artifacts through its own functions in `_initial_state_.py`, including NPZ arrays and JSON metadata. It explicitly labels fractures as equivalent-continuum corridors, not a discrete-fracture model.

## Tests

Run all tests:

```bash
python -m pytest -q
```

Run focused checks:

```bash
python -m pytest -q tests/test_geological_coupling.py
python -m pytest -q tests/test_initial_state_model.py
python -m pytest -q tests/test_jacobian_matching_numerical.py
python -m pytest -q tests/test_single_phase_1d.py
```

Show slow tests:

```bash
python -m pytest -q --durations=5
```

The tests cover mass conservation, Newton convergence and retries, analytical-vs-finite-difference Jacobians, Darcy flow, units, waterflood behavior, cell ordering, hydrostatic initialization, OWC, anisotropy, fault effects, fracture effects, and multi-perforation wells.

## RF surrogate workflow

The RF model is an optional Newton initializer. It does not replace the fully implicit solve: every RF prediction is clipped and passed through Newton correction.

Generate a dataset:

```bash
python scripts/generate_rf_dataset.py --cases 12 --output data/rf_dataset.npz
```

Train with simulation-level held-out groups rather than a row-wise random split:

```bash
python scripts/train_rf.py \
  --dataset data/rf_dataset.npz \
  --output-dir results/rf_models
```

Use a trained model for a compatible simulation:

```bash
python run_mvp.py --mode smoke --init rf --rf-dir results/rf_models
```

The RF feature vector contains current pressure/saturations, rock properties, well distances and controls, timestep, time, and neighbor averages. Temporal-difference features are intentionally absent because inference does not receive a separate previous state.

`scripts/benchmark_rf.py` reports prediction-level error and timing. Do not interpret that as a simulator speedup. A valid speedup claim requires a held-out end-to-end comparison of previous-state versus RF initialization, including runtime, Newton iterations, timestep cuts, production differences, and cumulative-volume differences.

## Developer notes

- Keep the residual, Jacobian, Newton, and timestep architecture intact when adding geological features.
- Do not silently mix SI and field units; put conversions at the geological/flow boundary.
- Use directional permeability: x faces use `kx`, y faces use `ky`, and z faces use `kz` with harmonic averaging.
- Fault multipliers operate on internal face transmissibilities.
- Capillary pressure is currently neglected (`Pc = 0`).
- Gas is handled with simple black-oil dissolved-gas/free-gas terms; this is not a compositional simulator.
- The model is not thermal, geomechanical, GPU-accelerated, or a discrete-fracture simulator.

## Repository map

| File | Purpose |
|---|---|
| `_initial_state_.py` | Synthetic geological reservoir generator and validation |
| `initial_state_model.py` | Geological-to-flow adapter and unit/order boundary |
| `grid.py` | Cartesian flow grid and cell ordering |
| `rock.py` | Porosity and rock property container |
| `transmissibility.py` | Directional TPFA transmissibility and upstream selection |
| `pvt.py` | Black-oil PVT relations |
| `relperm.py` | Corey relative permeability |
| `wells.py` | Peaceman wells and multi-perforation terms |
| `residual.py` | Fully implicit mass-balance residual |
| `jacobian.py` | Analytical and finite-difference Jacobians |
| `newton_solver.py` | Damped Newton-Raphson solve |
| `timestep.py` | Adaptive timestep policy and retry cuts |
| `simulator.py` | Time integration loop and diagnostics |
| `postprocess.py` | Reports, plots, and output files |
| `run_mvp.py` | Main command-line entry point |
| `rf_surrogate.py` | Optional RF initializer and dataset format |
| `tests/` | Regression and scientific validation tests |

## License and scientific use

No license file is currently included. Treat the repository as internal or research code unless a license is added. Results from the synthetic model should be described as numerical experiments, not field predictions.
