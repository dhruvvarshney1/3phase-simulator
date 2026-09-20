Below is a practical roadmap to build a **3-phase black-oil reservoir simulator** with:

- Fully implicit finite-volume discretization
- Newton–Raphson nonlinear solver
- Peaceman well model
- Waterflooding
- Fractal/fractional-Brownian porosity heterogeneity
- Random Forest surrogate used to generate good nonlinear-solver initial guesses

A good implementation strategy is:

1. Build a clean single-phase pressure solver.
2. Add multiphase flow and black-oil PVT.
3. Add fully implicit Newton iterations.
4. Add wells and water injection.
5. Add fractal heterogeneity.
6. Add RF acceleration and benchmark it carefully.

Python is generally easier for development; MATLAB is convenient for matrix operations and plotting. The numerical design is the same.

---

# 1. Define the physical model

Use a 2D Cartesian reservoir initially:

\[
N_x \times N_y \times N_z
\]

For a first version, use \(N_z=1\), such as:

```text
Nx = 30
Ny = 30
Nz = 1
```

Each grid block has:

- Pressure: \(p_o\), usually oil pressure
- Water saturation: \(S_w\)
- Gas saturation: \(S_g\)
- Oil saturation computed from closure:

\[
S_o = 1 - S_w - S_g
\]

Primary unknown vector per grid block:

\[
\mathbf{x}_i =
\begin{bmatrix}
p_{o,i} \\
S_{w,i} \\
S_{g,i}
\end{bmatrix}
\]

Total unknowns:

\[
N_{\text{unknowns}} = 3N_{\text{cells}} + N_{\text{well unknowns}}
\]

if bottom-hole pressures are unknown.

---

# 2. Organize the project

A clean Python structure could be:

```text
black_oil_simulator/
│
├── main.py
├── config.py
│
├── grid.py
├── rock.py
├── pvt.py
├── relperm.py
├── fluids.py
├── transmissibility.py
├── wells.py
├── residual.py
├── jacobian.py
├── newton_solver.py
├── timestep.py
├── fractal_porosity.py
├── rf_surrogate.py
├── postprocess.py
│
├── data/
├── results/
└── tests/
```

MATLAB equivalent:

```text
main.m
initializeGrid.m
initializeRock.m
pvtProperties.m
relativePermeability.m
computeResidual.m
computeJacobian.m
newtonSolver.m
wellModel.m
generateFractalPorosity.m
trainRandomForest.m
plotResults.m
```

---

# 3. Create the Cartesian grid

For each cell, calculate:

- Dimensions: \(\Delta x, \Delta y, \Delta z\)
- Bulk volume:

\[
V_b = \Delta x \Delta y \Delta z
\]

- Pore volume:

\[
V_p = \phi V_b
\]

Example Python grid setup:

```python
import numpy as np

Nx, Ny = 30, 30
dx, dy, dz = 50.0, 50.0, 20.0   # ft
Nc = Nx * Ny

bulk_volume = dx * dy * dz
```

You also need mappings between 2D indices and global cell indices:

```python
def cell_index(i, j, Nx):
    return j * Nx + i
```

For each cell, identify neighbors:

- East
- West
- North
- South

No-flow boundaries mean no flux through outer faces.

---

# 4. Define rock properties

Initially use constant rock properties:

```python
phi0 = 0.18
kx = 100.0   # md
ky = 100.0   # md
```

You can include compressibility later:

\[
\phi(p) = \phi_{\text{ref}} \left[1 + c_r(p-p_{\text{ref}})\right]
\]

where:

- \(c_r\): rock compressibility
- \(p_{\text{ref}}\): reference pressure

Example:

```python
cr = 3e-6      # 1/psi
p_ref = 4000.0 # psi

def porosity_pressure(phi_ref, p):
    return phi_ref * (1.0 + cr * (p - p_ref))
```

For a first working version, keep porosity pressure-independent.

---

# 5. Implement fractal-based porosity distribution

You want heterogeneity that looks more geologically realistic than random independent noise.

A common approach is generating a **fractional Brownian motion (fBm)** or spectral random field.

## Simple spectral fractal approach

1. Generate random noise in Fourier space.
2. Scale amplitudes based on wave number.
3. Transform back to real space.
4. Normalize to desired porosity range.

A useful relationship is:

\[
P(k) \propto k^{-\beta}
\]

where:

- \(k\): spatial wave number
- \(\beta\): controls roughness
- Larger \(\beta\): smoother, larger-scale heterogeneity
- Fractal dimension is related to spectral exponent

For a 2D surface, a commonly used approximation is:

\[
D_f = \frac{8-\beta}{2}
\]

or equivalently:

\[
\beta = 8 - 2D_f
\]

The precise relationship depends on convention, so document your chosen one clearly.

Example Python implementation:

```python
import numpy as np

def generate_fractal_porosity(Nx, Ny, phi_mean=0.18, phi_std=0.04,
                              phi_min=0.05, phi_max=0.30, beta=3.0,
                              seed=42):
    rng = np.random.default_rng(seed)

    kx = np.fft.fftfreq(Nx).reshape(-1, 1)
    ky = np.fft.fftfreq(Ny).reshape(1, -1)

    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0

    amplitude = k ** (-beta / 2.0)

    noise_real = rng.normal(size=(Nx, Ny))
    noise_imag = rng.normal(size=(Nx, Ny))

    spectrum = (noise_real + 1j * noise_imag) * amplitude
    field = np.fft.ifft2(spectrum).real

    field = (field - field.mean()) / field.std()

    phi = phi_mean + phi_std * field
    phi = np.clip(phi, phi_min, phi_max)

    return phi
```

Usage:

```python
phi_map = generate_fractal_porosity(
    Nx=30,
    Ny=30,
    phi_mean=0.18,
    phi_std=0.035,
    beta=3.0
)
```

Plot it:

```python
import matplotlib.pyplot as plt

plt.imshow(phi_map.T, origin="lower", cmap="viridis")
plt.colorbar(label="Porosity")
plt.title("Fractal Porosity Field")
plt.show()
```

## Convert porosity to permeability

Porosity alone does not control flow strongly enough. Generate permeability correlated with porosity.

For example:

\[
k = k_{\text{ref}}
\left(\frac{\phi}{\phi_{\text{ref}}}\right)^n
\]

Example:

```python
def porosity_to_perm(phi, phi_ref=0.18, k_ref=100.0, exponent=4.0):
    k = k_ref * (phi / phi_ref) ** exponent
    return np.clip(k, 0.1, 5000.0)
```

For unconventional reservoirs, use lower permeability values, such as \(10^{-4}\) to \(1\) md, depending on the model scale and whether fractures are represented.

---

# 6. Define black-oil PVT properties

A black-oil model typically includes:

- Oil formation volume factor: \(B_o(p)\)
- Water formation volume factor: \(B_w(p)\)
- Gas formation volume factor: \(B_g(p)\)
- Oil viscosity: \(\mu_o(p)\)
- Water viscosity: \(\mu_w(p)\)
- Gas viscosity: \(\mu_g(p)\)
- Solution gas-oil ratio: \(R_s(p)\)
- Bubble-point pressure: \(p_b\)

For an initial educational model, use simple correlations or interpolated PVT tables.

Example PVT functions:

```python
def Bo(p):
    # p in psi
    return 1.20 * np.exp(-1.0e-5 * (p - 4000.0))

def Bw(p):
    return 1.00 * np.exp(-3.0e-6 * (p - 4000.0))

def Bg(p):
    return 0.005 * (4000.0 / np.maximum(p, 100.0))

def mu_o(p):
    return 1.5 * np.exp(-2.0e-5 * (p - 4000.0))

def mu_w(p):
    return 0.5 + 0.0 * p

def mu_g(p):
    return 0.02 + 0.0 * p

def Rs(p, pb=2500.0, Rs_bubble=600.0):
    return np.where(
        p >= pb,
        Rs_bubble,
        Rs_bubble * (p / pb)
    )
```

For a stronger project, use published correlations such as:

- Standing correlation
- Vasquez–Beggs correlation
- Glaso correlation
- Beggs–Robinson viscosity correlation

Or use PVT tables with linear interpolation:

```python
np.interp(p, pressure_table, Bo_table)
```

---

# 7. Define relative permeability and capillary pressure

Use Corey-type relative permeability curves.

Effective water saturation:

\[
S_{we} =
\frac{S_w-S_{wc}}
{1-S_{wc}-S_{or}-S_{gc}}
\]

Effective gas saturation:

\[
S_{ge} =
\frac{S_g-S_{gc}}
{1-S_{wc}-S_{or}-S_{gc}}
\]

Relative permeability:

\[
k_{rw} = k_{rw0} S_{we}^{n_w}
\]

\[
k_{ro} = k_{ro0}(1-S_{we}-S_{ge})^{n_o}
\]

\[
k_{rg} = k_{rg0}S_{ge}^{n_g}
\]

Example:

```python
def relperm(Sw, Sg,
            Swc=0.20, Sor=0.20, Sgc=0.05,
            krw0=0.30, kro0=0.90, krg0=0.80,
            nw=3.0, no=2.0, ng=2.0):

    denom = 1.0 - Swc - Sor - Sgc

    Swe = np.clip((Sw - Swc) / denom, 0.0, 1.0)
    Sge = np.clip((Sg - Sgc) / denom, 0.0, 1.0)

    Soe = np.clip(1.0 - Swe - Sge, 0.0, 1.0)

    krw = krw0 * Swe**nw
    kro = kro0 * Soe**no
    krg = krg0 * Sge**ng

    return krw, kro, krg
```

Initially ignore capillary pressure:

\[
p_w = p_o, \qquad p_g = p_o
\]

Later add:

\[
p_o - p_w = P_{cow}(S_w)
\]

\[
p_g - p_o = P_{cgo}(S_g)
\]

---

# 8. Calculate phase mobilities

For each phase:

\[
\lambda_\alpha =
\frac{k_{r\alpha}}{\mu_\alpha B_\alpha}
\]

where \(\alpha = w,o,g\).

Example:

```python
def mobilities(p, Sw, Sg):
    krw, kro, krg = relperm(Sw, Sg)

    lam_w = krw / (mu_w(p) * Bw(p))
    lam_o = kro / (mu_o(p) * Bo(p))
    lam_g = krg / (mu_g(p) * Bg(p))

    return lam_w, lam_o, lam_g
```

---

# 9. Build transmissibilities

For a connection between cells \(i\) and \(j\), use harmonic averaging of permeability:

\[
k_{ij} =
\frac{2k_i k_j}{k_i+k_j}
\]

Transmissibility in the x-direction:

\[
T_{ij} =
\frac{k_{ij} A_x}{\Delta x}
\]

where:

\[
A_x = \Delta y \Delta z
\]

In field units, include the proper unit conversion constant, often:

\[
0.001127
\]

depending on the unit system.

Example:

```python
def harmonic_mean(a, b):
    return 2.0 * a * b / (a + b + 1e-30)

def transmissibility_x(k1, k2, dy, dz, dx):
    k_h = harmonic_mean(k1, k2)
    area = dy * dz
    return 0.001127 * k_h * area / dx
```

For each face:

\[
q_{\alpha,ij} =
T_{ij} \lambda_{\alpha,\text{upwind}}
(p_{\alpha,i} - p_{\alpha,j})
\]

Use upstream mobility based on pressure potential direction.

---

# 10. Write the three conservation equations

You need component conservation equations for:

1. Water component
2. Oil component
3. Gas component

## Water equation

\[
\frac{\partial}{\partial t}
\left(
\frac{\phi S_w}{B_w}
\right)
+
\nabla \cdot
\left(
\frac{\mathbf{u}_w}{B_w}
\right)
=
q_w
\]

## Oil equation

\[
\frac{\partial}{\partial t}
\left(
\frac{\phi S_o}{B_o}
\right)
+
\nabla \cdot
\left(
\frac{\mathbf{u}_o}{B_o}
\right)
=
q_o
\]

## Gas equation

The gas component exists as free gas and dissolved gas:

\[
\frac{\partial}{\partial t}
\left[
\phi
\left(
\frac{S_g}{B_g}
+
\frac{R_s S_o}{B_o}
\right)
\right]
+
\nabla \cdot
\left[
\frac{\mathbf{u}_g}{B_g}
+
R_s\frac{\mathbf{u}_o}{B_o}
\right]
=
q_g
\]

This is one of the key black-oil model features.

---

# 11. Discretize fully implicitly

Use backward Euler time discretization:

\[
\frac{M^{n+1} - M^n}{\Delta t}
+
F^{n+1}
-
Q^{n+1}
=0
\]

All properties are evaluated at the new time level \(n+1\).

For every cell:

\[
R_w = \frac{M_w^{n+1} - M_w^n}{\Delta t}
+ \sum q_w^{n+1}
- Q_w^{n+1}
\]

\[
R_o = \frac{M_o^{n+1} - M_o^n}{\Delta t}
+ \sum q_o^{n+1}
- Q_o^{n+1}
\]

\[
R_g = \frac{M_g^{n+1} - M_g^n}{\Delta t}
+ \sum q_g^{n+1}
- Q_g^{n+1}
\]

Your nonlinear problem is:

\[
\mathbf{R}(\mathbf{x}^{n+1}) = 0
\]

---

# 12. Implement the Newton–Raphson solver

Newton iteration solves:

\[
\mathbf{J}
\Delta \mathbf{x}
=
-\mathbf{R}
\]

Then update:

\[
\mathbf{x}^{k+1}
=
\mathbf{x}^{k}
+
\alpha \Delta \mathbf{x}
\]

where \(\alpha\) is a damping factor, typically between 0.1 and 1.0.

Basic algorithm:

```python
for timestep in timesteps:

    x = initial_guess_for_new_timestep()

    for iteration in range(max_newton_iterations):

        R = compute_residual(x, x_old, dt, wells, rock, grid)

        if np.linalg.norm(R, ord=np.inf) < tolerance:
            break

        J = compute_jacobian(x, x_old, dt, wells, rock, grid)

        dx = scipy.sparse.linalg.spsolve(J, -R)

        alpha = compute_damping_factor(x, dx)

        x = x + alpha * dx

        enforce_physical_bounds(x)
```

Recommended initial settings:

```python
max_newton_iterations = 12
residual_tolerance = 1e-6
update_tolerance = 1e-6
```

Use sparse matrices:

```python
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
```

A finite-difference Jacobian is acceptable for an initial prototype but becomes slow.

For a stronger project, use:

- Analytical Jacobian
- Automatic differentiation with JAX
- Automatic differentiation with PyTorch
- MATLAB symbolic/automatic differentiation tools
- MRST-style AD concepts if using MATLAB

---

# 13. Start with a numerical Jacobian

For debugging, approximate:

\[
\frac{\partial R_i}{\partial x_j}
\approx
\frac{R_i(x+\epsilon e_j)-R_i(x)}{\epsilon}
\]

Example:

```python
def numerical_jacobian(residual_func, x, eps=1e-6):
    R0 = residual_func(x)
    n = len(x)

    J = np.zeros((n, n))

    for j in range(n):
        x_pert = x.copy()
        perturbation = eps * max(abs(x[j]), 1.0)
        x_pert[j] += perturbation

        R1 = residual_func(x_pert)
        J[:, j] = (R1 - R0) / perturbation

    return J
```

This is too slow for large models, but it is extremely useful for validating residual logic before writing an analytical sparse Jacobian.

---

# 14. Add the Peaceman well model

For a well in grid block \(i\):

\[
q_\alpha =
WI \lambda_\alpha
(p_{\text{bh}} - p_i)
\]

depending on your production/injection sign convention.

The well index is:

\[
WI =
\frac{2\pi k_h h}
{\mu B
\left[
\ln(r_e/r_w)+s
\right]}
\]

For anisotropic Cartesian grids, Peaceman equivalent radius is approximately:

\[
r_e =
0.28
\frac{
\sqrt{
\sqrt{k_y/k_x}\Delta x^2 +
\sqrt{k_x/k_y}\Delta y^2
}
}{
(k_y/k_x)^{1/4}+(k_x/k_y)^{1/4}
}
\]

For isotropic permeability:

\[
r_e \approx 0.14\sqrt{\Delta x^2+\Delta y^2}
\]

Example:

```python
def peaceman_well_index(kx, ky, dx, dy, dz, rw=0.25, skin=0.0):
    re = 0.14 * np.sqrt(dx**2 + dy**2)
    kh = np.sqrt(kx * ky)

    WI = 0.001127 * 2.0 * np.pi * kh * dz / (np.log(re / rw) + skin)
    return WI
```

## Producer

A producer can operate under:

- Fixed bottom-hole pressure, BHP control
- Fixed oil rate control

For a first implementation, use BHP control:

```python
producer = {
    "cell": producer_cell,
    "type": "producer",
    "control": "bhp",
    "bhp": 1500.0,
    "rw": 0.25,
    "skin": 0.0
}
```

## Water injector

For water injection:

```python
injector = {
    "cell": injector_cell,
    "type": "injector",
    "control": "rate",
    "water_rate": 500.0
}
```

For a rate-controlled water injector, include the specified water source term directly in the water residual.

---

# 15. Define initial and boundary conditions

Example initial reservoir state:

```python
p_init = 4000.0
Sw_init = 0.25
Sg_init = 0.00
So_init = 1.0 - Sw_init - Sg_init
```

For a waterflood case:

- One water injector near one corner
- One producer near the opposite corner
- Closed external boundaries
- Injection begins at \(t=0\)

Example well placement:

```python
injector_cell = cell_index(2, 2, Nx)
producer_cell = cell_index(Nx - 3, Ny - 3, Nx)
```

---

# 16. Add time-step control

Fully implicit methods are stable, but very large time steps may cause Newton failure.

Use adaptive time stepping:

```python
dt = 1.0  # days

if newton_iterations <= 4:
    dt *= 1.5
elif newton_iterations >= 8:
    dt *= 0.5
```

Also reduce time step if:

- Newton does not converge
- Saturations become nonphysical
- Pressure changes too abruptly
- Large water breakthrough occurs

Typical simulation schedule:

```text
0–30 days:      1 day time steps
30–180 days:    5 day time steps
180–1000 days:  10–30 day time steps
```

---

# 17. Enforce physical constraints

After each Newton update, ensure:

\[
0 \leq S_w \leq 1
\]

\[
0 \leq S_g \leq 1
\]

\[
S_w + S_g \leq 1-S_{or}
\]

\[
p > 0
\]

Example:

```python
def enforce_bounds(p, Sw, Sg, Sor=0.20):
    p = np.maximum(p, 100.0)

    Sw = np.clip(Sw, 0.0, 1.0)
    Sg = np.clip(Sg, 0.0, 1.0)

    total = Sw + Sg
    max_mobile_fraction = 1.0 - Sor

    mask = total > max_mobile_fraction
    Sw[mask] *= max_mobile_fraction / total[mask]
    Sg[mask] *= max_mobile_fraction / total[mask]

    return p, Sw, Sg
```

In a production-grade simulator, use line search or trust-region damping rather than hard clipping alone.

---

# 18. Validate the simulator before adding ML

Do not add Random Forest until the base simulator is verified.

Validation sequence:

## Test 1: Single-phase flow

Set:

```text
Sw = 0
Sg = 0
```

Compare pressure response against a known analytical or semi-analytical solution.

## Test 2: No-flow material balance

Use a closed reservoir with no wells.

Expected:

\[
\text{Initial fluid in place} \approx \text{Final fluid in place}
\]

## Test 3: Water injection

Inject water into a homogeneous model.

Expected:

- Pressure rises near injector
- Pressure declines around producer
- Water saturation front moves toward producer
- Water breakthrough eventually occurs
- Water cut increases after breakthrough

## Test 4: Gas liberation

Set initial pressure above bubble point and produce until:

\[
p < p_b
\]

Expected:

- Gas saturation starts increasing
- Gas-oil ratio rises
- Oil mobility may decline
- Pressure depletion behavior changes

---

# 19. Compute production metrics

At every time step, store:

- Average reservoir pressure
- Producer oil rate
- Producer water rate
- Producer gas rate
- Injector water rate
- Cumulative oil production
- Cumulative gas production
- Cumulative water production
- Water cut
- Gas-oil ratio
- Newton iterations
- Runtime

Definitions:

\[
N_p(t) = \int_0^t q_o(\tau)d\tau
\]

\[
G_p(t) = \int_0^t q_g(\tau)d\tau
\]

\[
W_p(t) = \int_0^t q_w(\tau)d\tau
\]

Water cut:

\[
WC = \frac{q_w}{q_w+q_o}
\]

Recovery factor:

\[
RF = \frac{N_p}{OOIP}
\]

Original oil in place:

\[
OOIP =
\sum_i
\frac{
\phi_i V_i S_{o,i}
}{
B_{o,i}
}
\]

---

# 20. Run fractal-dimension sensitivity studies

Your main scientific study can compare several heterogeneity levels.

For example:

```text
Case A: Low heterogeneity / smoother field
Case B: Moderate heterogeneity
Case C: High heterogeneity / rough field
```

Example parameter sweep:

```python
beta_values = [2.0, 3.0, 4.0, 5.0]
```

Or define directly in terms of fractal dimension:

```python
fractal_dimensions = [2.1, 2.3, 2.5, 2.7]
```

For each case:

1. Generate porosity field.
2. Convert porosity to permeability.
3. Run the same waterflood schedule.
4. Compare recovery, pressure, breakthrough, and spatial saturation patterns.

Useful plots:

1. Porosity map
2. Permeability map
3. Pressure map at multiple times
4. Water saturation map at multiple times
5. Oil saturation map
6. Average pressure vs time
7. Cumulative oil vs time
8. Cumulative gas vs time
9. Water cut vs time
10. Recovery factor vs fractal dimension
11. Water breakthrough time vs fractal dimension

Interpretation examples:

- Higher heterogeneity may create preferential high-permeability channels.
- Faster water breakthrough can reduce sweep efficiency.
- Low-permeability zones can retain bypassed oil.
- Pressure depletion can become spatially nonuniform.
- High fractal dimension fields may produce stronger local gradients and more irregular fronts.

---

# 21. Add the Random Forest surrogate correctly

A Random Forest should not replace conservation equations or the Newton solver. Use it to generate **initial estimates** for nonlinear properties or the new time-step state.

Important point: PVT and relative permeability are usually deterministic functions of pressure and saturation. A Random Forest helps only if:

- You use expensive correlations/tables,
- You have complicated compositional/property calculations,
- You predict the next time-step solution,
- You use RF as a preconditioner/initializer,
- Or you use a large number of heterogeneous simulations and want a fast approximate guess.

The most useful approach is to train the RF to predict the next state:

\[
(p^{n+1}, S_w^{n+1}, S_g^{n+1})
\]

from:

\[
(p^n, S_w^n, S_g^n, \phi, k, \Delta t, \text{well controls}, \text{neighbor features})
\]

Then use the RF output as the Newton initial guess.

## Recommended RF feature set

For each grid cell and time step:

```text
Current pressure
Current water saturation
Current gas saturation
Porosity
Permeability
Distance to producer
Distance to injector
Injection rate
Producer BHP
Time-step size
Neighbor average pressure
Neighbor average water saturation
Neighbor average gas saturation
Previous pressure change
Previous saturation changes
```

Target variables:

```text
Pressure at next time step
Water saturation at next time step
Gas saturation at next time step
```

You may alternatively predict:

```text
Bo
Bw
Bg
mu_o
mu_w
mu_g
krw
kro
krg
```

but these are generally easier and safer to compute with deterministic PVT and relative-permeability functions.

---

# 22. Generate training data for the RF model

Run many conventional simulator cases first.

Vary:

- Initial pressure
- Fractal dimension
- Mean porosity
- Permeability scale
- Injector rate
- Producer BHP
- Well location
- Time step size
- Relative permeability endpoints
- Bubble-point pressure

For example:

```text
100–500 simulation cases
20–100 time steps per case
900 cells for a 30 × 30 model
```

This produces a large training set.

Example table:

| p_n | Sw_n | Sg_n | phi | k | injector_rate | bhp | dt | p_n+1 | Sw_n+1 | Sg_n+1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Train three regressors:

```python
from sklearn.ensemble import RandomForestRegressor

rf_pressure = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    n_jobs=-1,
    random_state=42
)

rf_sw = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    n_jobs=-1,
    random_state=42
)

rf_sg = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    n_jobs=-1,
    random_state=42
)
```

Train:

```python
rf_pressure.fit(X_train, y_pressure_train)
rf_sw.fit(X_train, y_sw_train)
rf_sg.fit(X_train, y_sg_train)
```

---

# 23. Use RF predictions as Newton initial guesses

Without RF:

```python
x_guess = x_old.copy()
```

With RF:

```python
p_guess = rf_pressure.predict(features)
Sw_guess = rf_sw.predict(features)
Sg_guess = rf_sg.predict(features)

x_guess = pack_unknown_vector(p_guess, Sw_guess, Sg_guess)
```

Then Newton corrects the approximate solution:

```python
x_solution = newton_solver(
    x_initial=x_guess,
    x_old=x_old,
    dt=dt,
    wells=wells
)
```

This preserves physical conservation because the final solution still comes from the fully implicit residual equations.

A good workflow is:

```text
RF prediction → bound enforcement → Newton correction → accepted physical solution
```

---

# 24. Benchmark the RF acceleration

Measure both runtime and solution quality.

Compare:

```text
Case 1: Standard Newton initialization
Case 2: Random Forest initialization
```

Track:

- Average Newton iterations per time step
- Total nonlinear iterations
- Total runtime
- Number of failed time steps
- Pressure error
- Saturation error
- Cumulative oil error
- Cumulative gas error
- Water breakthrough time error

Runtime speedup:

\[
\text{Speedup} =
\frac{t_{\text{baseline}}}
{t_{\text{RF-assisted}}}
\]

To justify a statement like “3× faster,” report something like:

```text
Baseline runtime: 180 seconds
RF-assisted runtime: 60 seconds
Speedup: 3.0×
```

Also show that recovery predictions remain close:

\[
\text{Relative error} =
\frac{|N_{p,\text{RF}} - N_{p,\text{baseline}}|}
{N_{p,\text{baseline}}}
\times 100
\]

A reasonable target is:

```text
Cumulative oil error < 1–3%
Pressure error < 1%
Water breakthrough-time error < 5%
```

---

# 25. Suggested development sequence

## Phase 1: Basic numerical framework

1. Create Cartesian grid.
2. Create homogeneous porosity and permeability.
3. Implement single-phase pressure flow.
4. Add a producer with fixed BHP.
5. Validate pressure decline.

## Phase 2: Two-phase water-oil model

6. Add water saturation equation.
7. Add Corey relative permeability.
8. Add water injector.
9. Simulate waterflood.
10. Plot water saturation fronts and water breakthrough.

## Phase 3: Three-phase black-oil model

11. Add gas saturation.
12. Add bubble-point behavior.
13. Add solution gas-oil ratio \(R_s\).
14. Add gas accumulation and gas flux.
15. Validate gas liberation below bubble point.

## Phase 4: Fully implicit nonlinear solver

16. Write residual equations.
17. Add Newton–Raphson iteration.
18. Start with finite-difference Jacobian.
19. Move to sparse analytical or automatic-differentiation Jacobian.
20. Add adaptive time stepping and damping.

## Phase 5: Geological heterogeneity

21. Generate fractal porosity fields.
22. Convert porosity to permeability.
23. Compare homogeneous and heterogeneous cases.
24. Perform fractal-dimension sensitivity analysis.

## Phase 6: Machine-learning acceleration

25. Generate simulation dataset.
26. Train RF next-state predictor.
27. Use predictions as Newton initial guesses.
28. Measure iteration reduction and runtime speedup.
29. Validate conservation and production accuracy.

---

# 26. Key plots for the final project

Your final report or portfolio should include:

### Reservoir maps

- Fractal porosity distribution
- Permeability distribution
- Pressure distribution at several times
- Water saturation maps
- Gas saturation maps
- Oil saturation maps

### Production plots

- Average pressure vs time
- Oil rate vs time
- Water rate vs time
- Gas rate vs time
- Cumulative oil production vs time
- Cumulative gas production vs time
- Water cut vs time
- Gas-oil ratio vs time

### Numerical performance plots

- Newton iterations per time step
- Runtime comparison: baseline vs RF-assisted
- Residual norm vs Newton iteration
- Time-step size vs simulation time
- RF prediction error distributions

### Sensitivity plots

- Recovery factor vs fractal dimension
- Water breakthrough time vs fractal dimension
- Cumulative oil recovery vs heterogeneity level
- Pressure depletion rate vs fractal dimension

---

# 27. Important implementation advice

1. **Do not begin with full 3-phase black oil.**  
   First make a one-phase solver work, then two-phase, then three-phase.

2. **Use simple PVT correlations first.**  
   Replace them with more realistic tables only after mass balance and convergence work.

3. **Keep units consistent.**  
   A common petroleum field-unit combination is:
   - Pressure: psi
   - Permeability: md
   - Length: ft
   - Time: days
   - Rates: STB/day or MSCF/day

4. **Track material balance.**  
   A simulator can appear visually correct while violating fluid conservation.

5. **Use sparse linear algebra.**  
   Dense Jacobians become unusable as grid size grows.

6. **Use RF only as an accelerator.**  
   Newton iterations must still converge the final result to the governing PDE residual tolerance.

7. **Separate physics from solver code.**  
   Keep PVT, relperm, residual assembly, wells, and Newton logic in separate functions/modules.

---

# 28. Minimum viable project configuration

A manageable first complete case:

```text
Grid:                 30 × 30 × 1
Cells:                900
Initial pressure:     4000 psi
Bubble point:         2500 psi
Initial Sw:           0.25
Initial Sg:           0.00
Producer BHP:         1500 psi
Water injection rate: 300–500 STB/day
Simulation time:      1000 days
Porosity:             fractal field around 0.18
Permeability:         porosity-correlated field
Newton tolerance:     1e-6
Max Newton iterations: 12
```

This is large enough to produce meaningful spatial results but small enough to debug.

---

# 29. Final deliverables

A strong final project should contain:

1. Source code for the simulator.
2. Governing equations and discretization description.
3. Newton solver and convergence discussion.
4. Peaceman well model derivation.
5. Fractal porosity/permeability generation method.
6. Waterflood case study.
7. Fractal dimension sensitivity study.
8. RF training methodology.
9. Runtime benchmark against conventional initialization.
10. Validation and mass-balance results.
11. Production and saturation plots.
12. Limitations and future work.

Good future extensions:

- Capillary pressure
- Gravity segregation
- Compressible rock
- Anisotropic permeability
- Local grid refinement near wells
- Hydraulic fractures
- Multilayer reservoirs
- GPU acceleration
- Automatic differentiation Jacobian
- LSTM/Transformer surrogate comparison against RF