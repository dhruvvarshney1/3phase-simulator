# 3-Phase Black-Oil Reservoir Simulator — Independent Audit of Current Files

**Scope:** current uploaded files only. No prior versions, README claims, or assumptions used as evidence.
**Method:** read all simulator modules, `scripts/`, `tests/`, `_initial_state_.py`, `initial_state_model.py`, `create_synthetic_reservoir.py`, docs/roadmap/plan as context only.
**Date (UTC):** 2026-10-01.

**Vocabulary used strictly below:**

- `implemented`: code exists for the feature.
- `implemented correctly`: code exists and matches the governing math/units for the tested non-degenerate case.
- `tested`: an automated test executes it.
- `verified`: a test/derivation actually establishes the claimed property within stated tolerance/coverage.
- `claimed`: comment/docstring says it, but code/test does not establish it.

---

## A. Actual architecture map

### A.1 Connected production path: default MVP / smoke waterflood

```text
config.SimulationConfig
  -> grid.build_grid(cfg.grid)
  -> rock.build_rock(cfg, grid[, phi_field])
       uses fractal_porosity.generate_porosity(nx, ny, cfg.fractal, nz=nz)
       then permeability_from_porosity(phi)
  -> residual.build_model(cfg[, phi_field][, with_wells])
       grid + rock + PVT(cfg.pvt) + RelPerm(cfg.relperm)
       + wells.build_wells(cfg, grid, rock)
       + transmissibility.face_transmissibility(grid, rock.perm, mult=None)
       + pv0 = bulk_volume / 5.615
  -> residual.evaluate_cells / accumulation / phase_upstream /
     compute_face_fluxes / flux_divergence / well_source /
     evaluate_residual/compute_residual
  -> jacobian.build_jacobian(analytical | numerical)
  -> newton_solver.newton_solve
  -> timestep.TimeStepper + cut_dt
  -> simulator.build_context/run_simulation/compute_well_rates
  -> postprocess.generate_mvp_report / plots / CSV / mass_balance_summary
```

RF path is a side-loop, not a physics replacement:

```text
simulator.run_simulation(..., init_guess_mode="rf", rf_model=...)
  each attempted step:
    x_guess = rf_model.predict_initial_guess(x_current, model, dt_try, t)
    newton_solve(x_guess, x_current, model, dt_try, ...)
```

`run_mvp.py` is the integration entry point for this path.

### A.2 Connected geological path: synthetic-model run

```text
_initial_state_.build_reservoir([ReservoirConfig])
  grid + facies + phi + kx/ky/kz + faults + fractures + hydrostatic state + wells
  -> initial_state_model.build_from_initial_state(res, cfg)
       SI -> field conversion
       directional trans + fault multipliers
       per-cell swc + multi-perf wells + hydrostatic-consistent gamma
       (model, x0)
  -> simulator.run_simulation(cfg, model=model, x0=x0)
```

Only `run_mvp.py --mode synthetic` uses this path. It explicitly rejects `--init rf`.

### A.3 Actually independent / disconnected modules

| Module | Status |
|---|---|
| `create_synthetic_reservoir.py` | Independent toy `2x2x1` uniform random porosity demo with plots. Not imported anywhere. |
| `synthetic_reservoir_report.md` | Documents only the toy above, not the flow simulator. |
| `fluids.py` | Empty file. Dead. |
| `docs/plan.md`, `docs/roadmap.md`, PDFs | Context/spec only; not executed. |
| `notebook/` | Not part of automated flow. |
| `live_visualization.py` | Optional callback only. |
| Fault/fracture/structure/gravity/OWC/facies physics | Connected **only** in A.2, independent of A.1. |
| `newton_solver.compute_convergence_scale` | Defined; convergence path actually uses `residual.residual_scales`. No caller found in inspected simulator/scripts call paths. Treat as dead/misleading. |
| `conftest.make_context(pvt=..., relperm=...)` | Arguments accepted but never used. |
| RF dataset/training/benchmark scripts | Connected to each other, but their coverage does not match `config.RFConfig` bounds/case machinery. |

So there are **two separate systems sharing residual/Jacobian/Newton code**:

1. Default system: flat grid, uniform initial state, fractal-or-constant porosity, deterministic isotropic `k(phi)`, no faults/fractures/gravity.
2. Synthetic geological system: structured depth, facies, anisotropic `kx/ky/kz`, fault multipliers, fracture-enhanced permeability, hydrostatic OWC state, multi-perf wells.

---

## B. What simulator is ACTUALLY running

### B.1 Grid, units, state

- Default grid: `30 x 30 x 3`, `dx=50`, `dy=50`, `dz=20 ft`.
- Smoke grid: `15 x 15 x 3`, same spacing.
- Ordering: `cell_index(i,j,k)=k*ny*nx+j*nx+i`; unknowns cell-blocked `x[3c:3c+3]=(p,Sw,Sg)`.
- Units: field units everywhere: `psi`, `md`, `ft`, `day`, `cp`, `STB`, `STB/day`, gas `SCF`, `SCF/day`.
- Constants: `0.001127 bbl/day per md-ft-psi/cp`, `5.615 ft^3/bbl`, radial `0.001127*2*pi`.
- Boundaries: no-flow; only interior faces exist.

### B.2 Porosity source

`rock.build_rock`:

1. explicit `phi_field`, if supplied;
2. else fractal field if `rock.use_fractal=True`;
3. else constant `rock.phi_const`.

Default `use_fractal=True`, so default runs use fractal porosity. Tests often override to constant or hand-made fields.

### B.3 Permeability source

Default path:

```text
k = k_ref * (phi0 / phi_ref)^k_exp
```

clipped to `[k_min,k_max]`, defaults `100*(phi/0.18)^4`, clip `[0.1,5000] md`.

- Isotropic in default path: `Rock.perm` is `(n_cells,)`.
- `face_transmissibility` supports `(n_cells,3)` directional input, but `residual.build_model` never supplies it.
- Directional `kx/ky/kz` exists only in A.2.
- `pressure_dependent=False` by default, so `phi(p)=phi0`; optional linear `phi0*(1+cr*(p-p_ref))` is implemented but off.

### B.4 Correlation, anisotropy, fractal use

- Porosity and permeability are correlated by construction in default path because `k=f(phi)`.
- They are **not** independently scattered: `_initial_state_.py` has `logk_sigma` scatter, default simulator does not.
- Isotropic, not anisotropic, in default runs.
- Fractal fields **are used** when enabled, but only as porosity input; beta acts through porosity roughness, then through deterministic `k(phi)`.
- `create_synthetic_reservoir.py` uniform `[0.10,0.30]` porosity is **not** used by the simulator.

### B.5 Faults

- Fault face multipliers are implemented in generation and in `face_transmissibility(..., mult=...)`.
- `residual.build_model` always calls transmissibility with `mult=None`.
- Therefore faults do **not** affect transmissibility in default MVP/smoke runs.
- They do affect flow only in synthetic mode via `initial_state_model`.

### B.6 Fractures

- `_initial_state_.build_fractures` is equivalent-continuum permeability multipliers inside corridors, not DFM; aperture is metadata.
- Same connectivity conclusion as faults: no effect in default runs; effective only in synthetic mode through enhanced `kx/ky/kz`.

### B.7 Geological initial state and wells

- Default initial state is uniform: `p_init=4000 psi`, `Sw=0.25`, `Sg=0`.
- Geological depth, OWC, hydrostatic pressure, facies `swc`, perforation selection are ignored in default runs.
- Default wells are single-perforation: BHP producer `1500 psi`, water-rate injector `400 STB/day`.
- Synthetic mode uses multi-perf vertical wells, WI-weighted injector split, wellbore head, anisotropic Peaceman radii, SI->field-converted BHP/rate.

**Two-system verdict:** default scientific runs study fractal-porosity-derived isotropic heterogeneity under waterflood. They do **not** study faults, fractures, structure, gravity, OWC, facies endpoints, or the full synthetic geology unless `--mode synthetic` is used.

---

## C. Physics verification

Sign convention used below: face flux `q` positive left-cell to right-cell; well source `Q` positive for injection; residual row is `accumulation + outflow - source`.

### C.1 Conservation

Implemented residual:

```text
R_w = (M_w^{n+1}-M_w^n)/dt + div(q_w) - Q_w
R_o = (M_o^{n+1}-M_o^n)/dt + div(q_o) - Q_o
R_g = (M_g^{n+1}-M_g^n)/dt + div(q_g,total) - Q_g
```

This is **implemented correctly** for the stated convention:

- closed uniform state gives exactly zero residual; checked directly.
- injector-only stationary state gives water residual `-injector_rate`; checked directly.
- `flux_divergence` adds `+q` to left and `-q` to right, so interior fluxes cancel globally.
- Closed-reservoir and global mass tests verify aggregate conservation under tight Newton tolerance.

### C.2 Accumulation

```text
PV0 = Vb / 5.615
M_w = PV0 * phi * Sw / Bw
M_o = PV0 * phi * So / Bo
M_g = PV0 * phi * (Sg / Bg + Rs*So / Bo)
So  = 1 - Sw - Sg
```

**Implemented correctly.** Gas includes free plus dissolved terms. There is no fourth saturation-constraint equation because `So` is eliminated; `Sw+Sg<=1` is enforced by clipping, not as a residual. That is a valid choice for `Pc=0`, but it means `Sor` only affects mobility, not the admissible saturation volume.

### C.3 Darcy flow

```text
q_a,ij = T_ij * lambda_a,up * dPhi_a
lambda_a = kr_a / (mu_a*B_a)
dPhi_a = (pL-pR) - gamma_a*(DL-DR)
```

`T_ij=0.001127*k_ij*A/L`, harmonic `k_ij=2kLkR/(kL+kR)`. Each phase is upwinded on its own potential; ties choose left, where flux is zero anyway.

**Implemented correctly** for TPFA with diagonal permeability and `Pc=0`. Dissolved gas uses oil-upstream `Rs`:

```text
q_g,total = q_g,free + Rs,up(oil) * q_o
```

Verified by the two-cell hand calculation in `residual.main`.

Gravity defaults to zero in default configs; synthetic configs set `gamma_w/gamma_o` from the same densities used to build hydrostatic pressure, with illustrative `gamma_g`.

### C.4 Relative permeability

Corey model:

```text
D = 1 - Swc - Sor - Sgc
Swe = clip((Sw-Swc)/D,0,1)
Sge = clip((Sg-Sgc)/D,0,1)
Soe = clip(1-Swe-Sge,0,1)
krw = 0.30*Swe^3
kro = 0.90*Soe^2
krg = 0.80*Sge^2
```

Nonzero analytic derivatives:

```text
dkrw/dSw, dkro/dSw, dkro/dSg, dkrg/dSg
```

with the other two identically zero. This matches the functional dependence. Endpoint/monotonicity checks are correct. One modeling quirk is explicitly documented: because `D` contains `Sgc`, `Swe` can saturate at 1 while oil remains mobile in the formalism; that is a property of the chosen normalization, not a coding error, but it is nonstandard.

Derivative edge issue: `Soe` clipping uses only `soe_raw>0` for the nonzero-derivative mask. The upper clip is practically unreachable except at exactly zero effective saturations, where the underlying `dswe/dsge` are already zero, so no material error was found in smooth states. Kinks at clip boundaries remain non-differentiable and are not covered by Jacobian tests.

### C.5 PVT

```text
Bo = bo_ref*exp(-bo_comp*(p-p_ref))
Bw = bw_ref*exp(-bw_comp*(p-p_ref))
Bg = bg_coeff*bg_p_ref/max(p,bg_p_floor)
mu_o = muo_ref*exp(-muo_slope*(p-p_ref))
mu_w, mu_g = constants
Rs = rs_bubble for p>=p_bubble else rs_bubble*p/p_bubble
```

Derivatives:

```text
dBo/dp=-bo_comp*Bo
dBw/dp=-bw_comp*Bw
dBg/dp=-coeff*p_ref/p^2 for p>floor else 0
dmu_o/dp=-slope*mu_o
dRs/dp=rs_bubble/p_bubble below bubble else 0
```

**Implemented correctly away from kinks.** At `p=p_bubble` and `p=bg_p_floor`, the code returns one-sided derivatives. That is a legitimate engineering choice, but finite differences straddling those points will not match, and this regime is not covered by Jacobian tests.

### C.6 Wells

Peaceman:

```text
WI = 0.001127*2*pi*sqrt(kx*ky)*dz/(ln(re/rw)+skin)
re = 0.14*sqrt(dx^2+dy^2) [isotropic default]
```

Producer per perforation:

```text
dd = max(p-(pbh+head),0) if no_backflow else p-(pbh+head)
q_w = WI*lambda_w*dd
q_o = WI*lambda_o*dd
q_gf = WI*lambda_g*dd
q_g,total = q_gf + Rs(p)*q_o
```

Injector is a prescribed water source split by `WI/sum(WI)`; injector BHP is diagnostic only. Net terms are `q_prod=-rates.q`, `q_inj=(rate share,0,0)`, hence production negative and injection positive, consistent with `R=...-Q`.

Well derivatives use mobility derivatives plus `dRs/dp*q_o`; central-difference self-checks pass for flowing states. The backflow clamp is non-differentiable at `dd=0`; that regime is not covered by Jacobian tests.

### C.7 Fully implicit formulation

All nonlinear terms use `x^{n+1}`:

- accumulation from new state,
- mobilities/PVT/relperm/porosity from new state,
- upstream selections from new pressure/potentials,
- wells from new state,
- `m_old` is fixed data from `x^n`.

No IMPES/explicit mobility remains. **Implemented correctly.**

### C.8 Newton-Raphson

Linear solve is:

```text
J*dx = -R
x <- clip(x + alpha*dx)
```

with backtracking/halving to `alpha_min=0.1`, substantial-bound precheck, raw-residual non-increase check, hard clipping `p>=p_min`, `Sw/Sg in [0,1]`, proportional rescale if `Sw+Sg>1`. Convergence requires scaled residual **and** `max|dp|/p_ref` **and** `max|dS|` simultaneously.

The linear algebra is correct, but two Newton bookkeeping issues are real:

1. Damping compares raw `max|R|`, while convergence uses scaled `max|R/scale|`. Gas residuals are `SCF/day`, oil/water `STB/day`, so damping can be gas-dominated even after scaling makes equations comparable.
2. `_damped_trial` uses `getattr(ctx.pvt,"p_ref",4000)` although pressure reference lives in `ctx.pvt.cfg.p_ref`; custom `p_ref` therefore does not affect the loose precheck.

Neither issue makes the tested smooth cases diverge, but they are inconsistencies, not claimed behavior.

---

## D. Jacobian correctness

### D.1 What is compared

`jacobian.numerical_jacobian` is central FD around the current residual, with `eps_p=1e-2`, `eps_s=1e-6`, and saturation perturbations clipped to `[0,1]`. `analytical_jacobian` includes:

- accumulation derivatives including optional `dphi/dp`,
- per-phase flux derivatives with fixed upstream selection,
- mobility derivatives from PVT plus relperm,
- `Rs(up)*q_o` plus `dRs/dp` contribution,
- producer derivatives; injector has none because it is constant.

### D.2 Test outcome

Passing tests establish:

- without wells, `3x2` heterogeneous pressures: relative error `<1e-5`;
- with wells, `6x6`: relative error `<1e-5`;
- synthetic small model with gravity/directional perm/faults/multi-perf wells at two noise levels: relative error `<1e-5`.

Relative error here is `max|J_a-J_n|/max|J_n|`. That proves the analytic implementation matches the residual's smooth-branch derivatives for those states.

### D.3 What is NOT proved

- Upstream-switch boundaries: analytic Jacobian freezes upstream; FD across a switch measures a kink. Tests avoid near-zero potentials by construction.
- Relperm clip boundaries.
- `Rs` bubble-point kink.
- `Bg` pressure floor.
- Producer `dd=0` clamp.
- Small Jacobian entries: normalization by global maximum can hide relatively large errors in tiny entries.
- Rock compressibility “on” path is implemented and has a unit self-check, but Jacobian tests use the default “off” path.

So: **trustworthy for smooth flowing states away from kinks**, including gravity/directional/fault/multi-perf geometry. **Not verified at non-differentiable operating points.**

---

## E. Test inspection

| Test | What it actually checks | Valid? | Current or stale? | Missing aspect |
|---|---|---|---|---|
| `test_units_consistency` – 5 tests | Literal constants and helper formulas equal hand formulas. | Yes, but narrow | Current | Does not test that every module uses the helpers consistently. |
| `test_initial_state_stationary` – 2 tests | Closed uniform state has zero residual; Newton needs 0 iterations. | Yes | Current | Only no-well uniform case. |
| `test_closed_reservoir_mass_balance` | Closed heterogeneous equilibration conserves water/oil/gas `<1e-8` over 200 days at tight tolerance. | Yes for conservation property | Partly stale fixture: computes `k` from porosity but `make_context` ignores array `k`, so the intended permeability field is not used. Conservation still holds for the permeability actually used. | Accuracy, not just conservation. |
| `test_single_phase_1d_darcy_steady_state` | Custom Dirichlet solve stays finite/bounded and produces positive oil flux. | Weak | Stale relative to docstring: docstring promises linear Darcy profile and `<1%` flux match, but code never asserts either. | The actual Darcy accuracy claim is untested. |
| `test_jacobian_matching_numerical` – 2 tests | Analytic vs FD Jacobian agreement. | Yes, within smooth-state limits | Current | Kinks, small entries, compressibility-on. |
| `test_newton_convergence` | Easy step converges, residual decreases, ratios improve. | Yes for easy regime | Current | Says nothing about hard steps, damping, timestep cuts, or global convergence. |
| `test_waterflood_physics` | Small run injects water, injector-block `Sw` rises, states finite, iterations bounded. | Yes as smoke test | Current | No front, breakthrough, recovery, or pressure-pattern validation. |
| `test_gas_liberation` – 3 tests | Depletion below bubble point creates free gas, raises GOR above initial `Rs`, drops producer pressure below bubble point. | Yes end-to-end | Current | Quantitative liberation rate/GOR shape not checked. |
| `test_global_mass_balance_with_wells` | Full MVP mass change equals injected-minus-produced `<1e-6` at tight tolerance. | Property is correct if it passes; run timed out in this audit environment | Current but expensive | Does not validate mass error at production tolerance `1e-6`; uses `1e-9`. |
| `test_initial_state_model` – 3 tests | Synthetic oil leg has negligible oil flux; analytic/numerical Jacobian agree with gravity/wells. | Yes | Current | Water-leg drift from `Pc=0` is acknowledged but not bounded as a test; no fault-seal flow-effect test. |

Critical missing tests:

- fault multiplier changes the simulated flow solution;
- fracture corridor changes the solution;
- injector diagnostic BHP and multi-perf allocation;
- timestep cut/retry behavior;
- production-tolerance mass balance;
- RF train/inference consistency and leakage-free generalization;
- saturation/pressure clipping and `Rs`/relperm kink behavior.

---

## F. Geological initial model audit

Generator: `_initial_state_.py`. Bridge: `initial_state_model.py`.

| Feature | Verdict | Evidence/notes |
|---|---|---|
| Structural geometry | Implemented correctly | Dome Gaussian plus linear dip; depth positive down; `k=0` top. |
| Depth convention | Implemented correctly | Explicit `depth positive downward`; pressure increases downward. |
| Stratigraphy | Implemented | `layer_sequence`, lateral correlated variability, sinusoidal channels. Simplified but functional. |
| Facies architecture | Implemented | 4 facies, channel overwrite, masks retained. |
| Correlated heterogeneity | Implemented, simplified | FFT Gaussian fields with XY/Z correlation lengths; separate fields for facies threshold, porosity, permeability, vertical ratio. |
| Fractal heterogeneity hook | Implemented but disconnected from default flow | Generator supports `spectrum="fractal"`; default is `"gaussian"`. Separate `fractal_porosity.py` drives default flow. |
| Porosity distributions | Implemented correctly | Facies-conditioned clipped Gaussian within table bounds; validator checks bounds. |
| Permeability distributions | Implemented | `log10k=a+b*phi+sigma*Z`, clipped; independent scatter so `phi` does not uniquely determine `k`. |
| Anisotropy | Implemented in geo path | `kz` from `kv/kh` sampler; `ky=kx*0.85`; fracture projection adds directional multipliers. |
| Faults | Implemented but simplified and disconnected from default simulator | Signed-distance/curved plane, cell mask, internal-face multipliers. No throw/juxtaposition. Connected only in synthetic mode. |
| Fault transmissibility multipliers | Implemented in generator; disconnected by default | Correct face-indexed arrays; `build_model` never passes them. |
| Fracture corridors | Implemented but simplified and disconnected by default | Equivalent continuum, axis-projected multipliers; aperture metadata only. No DFM. |
| OWC | Implemented correctly | OWC depth plus transition ramp; no oil below OWC; shale `Sw=1`. |
| Hydrostatic pressure | Implemented correctly | Water gradient below OWC, oil gradient above, continuous at OWC with `Pc=0`. |
| Initial `Sw/So/Sg` | Implemented, simplified | Facies `swc` plus linear transition-zone surrogate for capillary effects; `Sg=0`. |
| Well perforations | Implemented | Facies-aware `reservoir` / `oil_reservoir` / explicit rules; depths/facies stored. |
| SI→field bridge | Implemented correctly for checked quantities | Length/pressure/rate/density-gradient conversions; index transpose `(Nx,Ny,Nz)->k*ny*nx+j*nx+i`; fault-face ordering matches grid. |
| Hydrostatic equilibrium in simulator | Partly verified | Oil-leg flux ~0 in tests. Water-leg/transition-zone drift under gravity is expected because `Pc=0`; documented, not a hidden bug. |

Status codes requested:

- Structure, depth, stratigraphy, facies, porosity, permeability, anisotropy, OWC, pressure, saturations, perforations: **1 = implemented correctly**, subject to stated synthetic simplifications.
- Geo fractal option: **2 = implemented but simplified**.
- Faults, fractures, multipliers, facies `swc`, directional perm, OWC state in default flow: **3 = implemented but disconnected from the flow simulator**.
- Fault cell mask and fracture aperture plots: **4 = visual/metadata where noted**.
- Throw, DFM, capillary pressure, compositional PVT: **5 = missing**.

---

## G. Fractal model audit

Implementation in `fractal_porosity.py`:

1. Complex white Gaussian noise in Fourier space.
2. Amplitude filter `A(k)=|k|^{-beta/2}`, `A(0)=0`.
3. Inverse FFT, real part, zero mean/unit variance.
4. `phi=phi_mean+phi_std*field`, clipped to `[phi_min,phi_max]`.

Findings:

- 2D and 3D dimensionality are implemented correctly: `ifft2` for `nz=1`, `ifftn` otherwise.
- Array ordering matches simulator: `(ny,nx)` or `(nz,ny,nx)` C-order ravel equals `k*ny*nx+j*nx+i`.
- `beta` means power exponent: `P(k) ~ k^{-beta}`. Larger beta gives smoother/longer-correlation fields.
- `Df=(8-beta)/2` is the 2D-surface convention: `beta=2H+2`, `H=(beta-2)/2`, `Df=3-H`. That is correct for a 2D map/surface.
- The same `Df` formula is reused in plots for all cases, including 3D fields. For a 3D volume the corresponding relation would differ. Do not interpret 3D beta sweeps with the 2D-surface dimension without qualification.
- Fields are actually spatially correlated for `beta>0`; this is not white noise.
- FFT periodicity matters: fields wrap at edges. The module documents generating larger fields and cropping for production work, but the simulator does not do that.
- Same seed with different beta reuses the same underlying noise for the same shape, so beta sweeps isolate spectral slope rather than realization noise. Changing `nz`/shape changes the RNG draw sequence/shape, so cross-dimensional seed comparisons are not the same realization.
- Clipping changes realized mean/std and distorts the Gaussian/spectral ideal. Statistics should be reported after clipping for flow purposes.
- Spectral-slope self-check fits radial power spectrum and expects slope near `-beta`; tolerance is loose and not part of `pytest`.

---

## H. RF surrogate audit

### H.1 What RF predicts and uses

- Predicts packed next-cell state `(p,Sw,Sg)`; one row per cell.
- Feature list has 18 columns: current `p/Sw/Sg`, `phi`, `perm`, producer/injector distance, injector rate, producer BHP, `dt`, time, neighbor-mean `p/Sw/Sg`, and `delta_p/delta_Sw/delta_Sg`.
- `predict_initial_guess` builds features from the current state only and never supplies `x_prev`.
- Dataset generation also never supplies `x_prev`.
- Therefore train and inference are consistent, but only because delta features are always zero. Those three features are dead in both phases.

### H.2 Physics safety

- Every RF prediction is clipped before Newton: `p>=p_min`, saturations in `[0,1]`, plus a saturation-sum rescale.
- Newton then solves the governing residual. If it converges, the accepted state is physics-corrected, not raw ML output.
- So RF is only an initial guess in the simulator path. That architectural claim is true.

But the bound is inconsistent:

- RF uses `Sw+Sg <= 1-Sor` (`0.80` by default).
- Newton `clip_state` uses `Sw+Sg <= 1.0`.
- RF is therefore more restrictive than the solver. This does not break conservation by itself, but the two “physical bounds” are not the same.

### H.3 Leakage and benchmark fairness

- `generate_rf_dataset.py` samples smoke cases over hard-coded ranges and collects snapshot-to-snapshot cell rows. It does not use `RFBounds`, `case_config`, or Latin-hypercube machinery despite those existing in config.
- `train_rf.py` does a random row-wise `train_test_split`. Cells from the same case/time interval can appear on both sides, and neighbor-correlated rows leak across the split. Test MAE/RMSE are therefore optimistic for new-reservoir generalization.
- `benchmark_rf.py` predicts on the same dataset and compares RF error to `features[:,:3]` as baseline. It measures in-sample prediction error and RF inference seconds only. It does not compare Newton iterations, timestep cuts, total wall time, or final production error on held-out simulations.
- Consequently, the RF path is **not currently usable as evidence that RF reduces runtime or Newton iterations**. The mechanism exists; fair measurement does not.

---

## I. Bugs and inconsistencies

Only issues supported by current files are listed.

| Severity | File | Problem | Why it matters | Fix |
|---|---|---|---|---|
| Critical | `residual.py`, `simulator.py`, `run_mvp.py` | Default flow ignores faults, fractures, structure, gravity, OWC, facies endpoints, directional permeability, and geological initial state. | Project objective includes fractal/geological heterogeneity, faults/fractures, and waterflooding recovery, but default scientific runs include only fractal-derived isotropic porosity/permeability. | Decide the primary scientific configuration: either make synthetic geo mode the default scientific path with benchmarks, or explicitly scope default claims to “fractal porosity-derived isotropic heterogeneity.” Add a test that the selected path actually changes flow. |
| High | `tests/test_single_phase_1d.py` | Docstring promises linear Darcy profile and `<1%` flux agreement, but test asserts only finite/bounded/positive flux. | A reader may believe Darcy accuracy is verified when it is not. | Assert pressure against the analytic linear profile and face flux against Darcy prediction within the documented tolerance. |
| High | `tests/conftest.py`, `tests/test_closed_reservoir_mass_balance.py` | `make_context` accepts `pvt`, `relperm`, and array `k`, but ignores `pvt`/`relperm` and recomputes permeability from porosity with default `k_ref`. | The closed-reservoir test computes a `k` field that is silently unused; future edits may assume permeability control they do not have. | Use the supplied `k` when given, or remove unused arguments and update the test to state that permeability is derived from porosity. |
| High | `scripts/generate_rf_dataset.py`, `scripts/train_rf.py`, `scripts/benchmark_rf.py` | Random row-wise split and same-dataset benchmark create temporal/spatial leakage and unfair runtime comparison. | Reported RF error/speedup cannot support the project’s central ML claim. | Split by held-out simulation case/time-block, generate from configured LHS bounds, and benchmark full Newton runtime/iterations/solution error on unseen cases. |
| High | `rf_surrogate.py`, `newton_solver.py` | RF enforces `Sw+Sg<=1-Sor`, Newton enforces `Sw+Sg<=1`. | Two definitions of “physical” can bias initial guesses and complicate error analysis. | Choose one documented bound and use it in both places, or explicitly justify the difference. |
| High | `rf_surrogate.py`, `scripts/*` | `delta_p/delta_Sw/delta_Sg` features are always zero in both training and inference. | Dead features waste capacity and mislead anyone interpreting feature importance. | Either thread `x_prev` through dataset generation and inference, or remove those features. |
| High | `newton_solver.py` | Damping uses raw `max|R|`, convergence uses scaled `max|R/scale|`; gas is `SCF/day`, liquids `STB/day`. | Damping can be dominated by gas scale even when scaled equations are balanced. | Use the same scaled norm for damping and convergence, or document/test why raw norm is safe. |
| Medium | `newton_solver.py` | `_damped_trial` reads `ctx.pvt.p_ref`, but the value lives in `ctx.pvt.cfg.p_ref`, so it always falls back to `4000`. | Custom reference pressure silently does not affect the bound precheck. | Read `ctx.pvt.cfg.p_ref` with the same fallback. |
| Medium | `newton_solver.py` | `compute_convergence_scale` is dead; actual scaling is `residual_scales`. | Readers/tests may tune the wrong scaling function. | Delete or wire it up; do not leave two competing scales. |
| Medium | `jacobian.py`, `wells.py`, `pvt.py`, `relperm.py` | No analytic/numerical agreement coverage at upstream switches, relperm clips, bubble-point kink, `Bg` floor, or producer backflow clamp. | Production runs can hit exactly these non-smooth points. | Add targeted Jacobian tests at/near each kink and document expected one-sided behavior. |
| Medium | `postprocess.py` | `plot_rock_maps` visualizes `rock.perm` as permeability, but synthetic models store only `kx` there while flow uses `kx/ky/kz`. | Synthetic permeability visualization can mislead. | Plot directional components or label the map as `kx`. |
| Medium | `initial_state_model.py` | Gas gravity uses illustrative `rho_g=150 kg/m3`, not a geological-model quantity. | Synthetic gas-gravity behavior is illustrative, not validated. | Expose it as an explicit assumption in configs/reports or derive it from a stated PVT choice. |
| Low | `fluids.py` | Empty file. | Confusing leftover. | Delete or implement. |
| Low | `create_synthetic_reservoir.py` | Standalone toy unrelated to simulator. | May be mistaken for the geological input. | Move to examples or remove from the simulator path/docs. |
| Low | `simulator.py`, `timestep.py` | Intermediate report-time landing depends on `propose_next`; direct `dt_try=min(dt,t_end-t)` logic is less explicit. | Future schedule edits could miss snapshots. | Add an explicit clip-to-next-report operation in the time loop and test it. |

---

## J. Current project stage

| # | Component | Implemented | Partially implemented | Tested | Untested | Disconnected | Incorrect |
|---|---|---|---|---|---|---|---|
| 1 | Geological model | Yes | Simplified: no throw, equivalent-continuum fractures, linear transition zone, illustrative gas gravity | Generator validation + bridge checks | Flow effect of faults/fractures | From default flow | No |
| 2 | Grid | Yes | — | Structural self-checks + tests | — | No | No |
| 3 | Rock | Yes | Optional compressibility off by default | Self-checks + conservation tests | Compressibility-on flow path | Directional perm from default flow | No |
| 4 | PVT | Yes | Kinks use one-sided derivatives | Derivative self-checks | Kink regimes | No | No material error found |
| 5 | Relperm | Yes | Nonstandard `Swe` saturation behavior documented | Endpoint/derivative checks | Clip/kink regimes | Per-cell `swc` only used in synthetic path | Model choice is odd but not a code bug |
| 6 | Wells | Yes | Injector BHP diagnostic only; no rate-producer/variable switching | Rate/derivative/clamp checks | Multi-perf allocation and injector BHP accuracy | Multi-perf from default flow | No |
| 7 | Transmissibility | Yes | Static perm/multipliers only | Formula/resistance/upstream checks | Sealing-fault flow test | Fault multipliers from default flow | No |
| 8 | Residual | Yes | `Pc=0`, static multipliers | Stationarity, hand flux, conservation tests | Capillarity, 3D gravity segregation | No | Damping/scale inconsistency is in Newton, not residual |
| 9 | Jacobian | Yes | Frozen-upstream linearization | Smooth-state agreement | Non-smooth states | No | No smooth-state error found |
| 10 | Newton solver | Yes | Pragmatic clip-and-check damping, not rigorous line search/trust region | Easy-step convergence | Hard steps, cuts, production tolerance | No | Raw-vs-scaled norm and `p_ref` lookup are inconsistent |
| 11 | Adaptive timestep | Yes | Report-landing logic is indirect | Indirectly through runs | Explicit cut-landing test | No | No |
| 12 | Simulator | Yes | Synthetic RF combination rejected | Waterflood/gas/mass smoke tests | Full production mass test in this environment | Default vs geological mode split | No |
| 13 | Fractal heterogeneity | Yes | Periodic boundaries, clipping, 2D `Df` reused for 3D | Generator self-checks | Anisotropic/conditional/multifractal paths | Geo-generator fractal option from default flow | 3D `Df` interpretation is misleading |
| 14 | RF acceleration | Mechanism yes; validation no | Dead delta features, bound mismatch | Only in-sample fit metrics | Fair held-out runtime/iteration benchmark | Config LHS bounds from dataset script | Leakage/benchmark claims are not supported |
| 15 | Postprocessing | Yes | Synthetic perm visualization limitation | Mass-balance helpers used by tests | Plot-content validation | No | No physics error found |
| 16 | Validation | Partly | Strong conservation/numerics core; weak Darcy/fault/RF coverage | See E | See missing-tests list | — | 1D Darcy doc overclaims |

---

## K. Correct next steps

### Priority order based only on current code

1. **Fix the scientific-configuration split.** Make one documented primary path for heterogeneity/recovery claims. Either promote synthetic geology to the benchmark path or restrict default-path claims to isotropic fractal-porosity heterogeneity.
2. **Repair validation overclaims.** Strengthen the 1D Darcy test to check the promised profile/flux; fix `make_context` permeability handling; add fault/fracture flow-effect tests.
3. **Make RF evaluation leakage-free and fair.** Case-wise held-out datasets, configured parameter coverage, full Newton runtime/iteration/solution-error benchmark. Fix dead delta features and the saturation-bound mismatch first.
4. **Cover non-smooth numerics.** Jacobian and Newton behavior at upstream switches, relperm clips, bubble point, `Bg` floor, and producer clamp; unify Newton raw/scaled norms.
5. **Run the intended fractal/geological waterflood study only after 1–4.** Beta/`Df` sweeps, breakthrough/recovery comparisons, and RF speedup claims depend on validated physics and measurement.

### Required correctness fixes

- Unify `Sw+Sg` bound between RF and Newton.
- Thread or remove RF delta features.
- Fix Newton `p_ref` lookup and raw/scaled norm inconsistency.
- Remove or connect dead code: `fluids.py`, `compute_convergence_scale`, unused fixture arguments.

### Required validation

- Real 1D Darcy accuracy test.
- Fault/fracture influence tests.
- Production-tolerance mass-balance evidence.
- Held-out RF benchmark with Newton statistics.

### Project-specific scientific work

- Beta/`Df` sweep with identical seeds and fixed well controls.
- Breakthrough time, recovery factor, pressure depletion versus heterogeneity.
- Separate isotropic-fractal effects from fault/fracture/structure effects.

### Optional improvements

- Capillary pressure.
- Rigorous line search/trust region.
- Anisotropic/conditional/multifractal geology.
- GPU/autodiff Jacobian.
- Advanced surrogates beyond RF.

---

## L. Final verdict

1. **Is this currently a real fully implicit 3-phase black-oil simulator?**  
   **Yes, for the implemented `Pc=0` black-oil equations.** Accumulation, TPFA flux, dissolved gas, Peaceman wells, backward Euler, and Newton correction are present and conservation-tested. It is not a compositional, capillary-pressure, thermal, DFM, or full-tensor simulator.

2. **Is Newton-Raphson actually being used correctly?**  
   **Mostly, with bookkeeping flaws.** `JΔx=-R`, sparse solve, damping/clipping, dual residual/update convergence, and timestep cuts are implemented. Damping uses a different norm than convergence, and the pressure-reference lookup can fall back incorrectly. The method is pragmatic rather than globally guaranteed.

3. **Is the analytical Jacobian trustworthy?**  
   **Yes for smooth states, no for kinks.** Agreement with FD is verified for flowing heterogeneous, gravity, directional-permeability, fault-multiplier, and multi-perf cases away from switches/clips. Upstream changes, saturation clips, bubble point, gas floor, and backflow clamp are not verified.

4. **Is the geological initial model connected to the simulator?**  
   **Only in synthetic mode.** Default MVP/smoke runs do not use depth, faults, fractures, facies, OWC, hydrostatic pressure, directional permeability, or geological wells.

5. **Are fractals actually influencing flow?**  
   **Yes in default runs, but narrowly.** Fractal porosity changes deterministic isotropic permeability and therefore flow. Faults, facies-specific scatter, anisotropic fractal fields, and the geological generator’s fractal option are not part of that influence.

6. **Are faults actually influencing flow?**  
   **No in default runs; yes in synthetic runs.** The multiplier machinery exists, but the default path never supplies multipliers.

7. **Are fractures actually influencing flow?**  
   **No in default runs; yes in synthetic runs as equivalent-continuum permeability enhancement.** There is no discrete-fracture flow model.

8. **Is the RF model currently usable without data leakage?**  
   **No.** The initializer mechanism is physics-safe because Newton corrects it, but training has same-case row leakage, dataset coverage ignores configured bounds, delta features are dead, and the benchmark does not measure Newton runtime fairly.

9. **What parts of the CV/project description are already supported by code?**  
   Realistic synthetic geology generation; 3D Cartesian flow grid; porosity/permeability initialization; multiphase oil/water/gas flow; fully implicit formulation; Newton-Raphson; Peaceman wells; waterflooding; fractal porosity generation; RF-as-initial-guess machinery; mass-balance/postprocessing outputs.

10. **What parts are NOT yet supported?**  
    Fault/fracture influence in default scientific runs; connected geological-to-flow heterogeneity study; validated fractal-recovery/breakthrough conclusions; validated RF runtime/iteration reduction; capillary pressure; rigorous global Newton guarantees; DFM/full-tensor/anisotropic default permeability; compositional or thermal physics.

11. **What is the single most important thing to implement next?**  
    **Select and validate one primary scientific configuration before more physics or ML.** Without fixing the default-vs-synthetic split and its missing flow-effect tests, fractal, fault, fracture, and RF conclusions remain disconnected from the runs actually used for science.

