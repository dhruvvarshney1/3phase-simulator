# Incorporating Fractal Analysis into 3D Porosity Simulation of Carbonate Reservoirs

**A methodological companion to "Porosity Simulation and 3D Porosity Modeling in Carbonate Reservoirs: A Literature Review"**

---

## 0. How to read this document

Your existing review is organised around a pipeline: *structural grid → facies → diagenesis → porosity PDF → variogram → conditional simulation → validation*. Fractal analysis is not a bolt-on chapter that sits beside that pipeline. It touches **five** of those seven stages, and it can enter at three different levels of ambition:

| Level | What fractals do | Effort |
|---|---|---|
| **Descriptive** | Measure fractal/multifractal dimensions from your data (logs, thin sections, MICP, NMR) and report them as heterogeneity indices per facies | Low |
| **Generative** | Use fractal covariance models (power-law variogram, fBm, multiplicative cascades) *as the simulation engine* to build the 3D porosity field | Medium |
| **Diagnostic** | Use fractal/multifractal spectra as **validation and loss metrics** to test whether SGS / MPS / GAN / LDM realizations reproduce the true multiscale structure | Medium |

The strongest BTP does all three and closes the loop: measure $D$ from data → simulate with a fractal-consistent model → verify $D$ is reproduced → show that non-fractal methods (plain SGS) *fail* to reproduce it. That "failure demonstration" is the single most publishable result available to you, and it is cheap to obtain.

Sections 1–8 are the technical toolbox. Section 9 gives the revised workflow. Section 10 gives three scoped project plans. Sections 11–14 cover data, software, pitfalls, and text you can adapt into your thesis.

---

## 1. Why fractals belong in carbonate porosity modeling

### 1.1 The scale problem your review already identified

Your review states that carbonate pore networks span "primary matrix microporosity, secondary intercrystalline and moldic porosity, macroscopic vugs, and brittle fractures." That is roughly **nine orders of magnitude** — micrite crystal boundaries at $10^{-8}$ m, intercrystalline pores at $10^{-6}$ m, moldic pores at $10^{-4}$ m, vugs at $10^{-2}$ m, karst conduits at $10^{0}$–$10^{3}$ m.

Classical geostatistics handles this badly. A variogram with a sill and a finite range asserts that beyond distance $a$, values are uncorrelated. Real carbonate porosity does not behave that way: measured variograms of porosity logs frequently keep rising with no clear sill, and the apparent range keeps growing as you enlarge the sampling window. This is the classic signature of a **scale-invariant (fractal) field**, not a stationary one.

Fractal geometry gives you a parameterization that is:

- **Parsimonious** — one exponent ($H$ or $D$) encodes behaviour across all decades, versus a nested sum of many variogram structures.
- **Scale-consistent** — the same model works at core-plug, log, and seismic resolution, which directly addresses the multiscale conditioning problem in your Section "Data Conditioning and Multiscale Integration."
- **Physically motivated** — the processes that create carbonate porosity (dissolution fronts, reaction-transport instabilities, fracture propagation, karst network growth) are known to be scale-invariant or nearly so.

### 1.2 Three physical arguments specific to carbonates

**Diagenesis as a scale-invariant process.** Meteoric dissolution is a reactive-infiltration instability. Fingering of the dissolution front is governed by Damköhler and Péclet numbers, and the resulting wormhole/conduit patterns are known to be fractal or multifractal over wide ranges. Similarly, karst conduit networks grow by a positive-feedback mechanism formally analogous to diffusion-limited aggregation (DLA) and invasion percolation, both of which produce well-characterised fractal dimensions ($D \approx 1.71$ for 2D DLA, $D \approx 2.5$ for 3D).

**Fracture and vug size distributions are power laws.** Fracture length frequency $n(\ell) \propto \ell^{-a}$ with $a \approx 1.5$–$3.5$, karst cave volumes, and vug size distributions in Tarim-type reservoirs are all commonly reported as power-law. A power-law size distribution is exactly what "no characteristic scale" means. Your Discrete Fractured-Vuggy Network section already needs a size distribution — making it a power law is both more realistic and more defensible than a lognormal.

**Rough surfaces.** Fracture walls and stylolite surfaces are self-affine with Hurst exponents typically $H \approx 0.8$ (in-plane) and $\approx 0.5$ (in the propagation direction). This directly controls aperture distribution and therefore fracture permeability.

### 1.3 What fractal analysis will *not* do

Be honest about this in your thesis; examiners reward it.

- Fractal dimension alone is **non-unique**. Two very different-looking fields can share the same $D$. You need lacunarity and/or the full multifractal spectrum to discriminate. Always report at least two descriptors.
- Real rocks are **not fractal over unlimited range**. They show scaling over one to three decades, often with distinct regimes separated by crossovers. Claiming a single global $D$ for a carbonate is usually wrong; piecewise/multi-regime fitting is the honest approach and is itself an interesting result.
- Fractal fields are **non-stationary** (fBm has no finite variance). This conflicts with the stationarity assumption baked into SGS and kriging. Handling this properly (Section 4.2) is a real technical contribution, not a formality.

---

## 2. The fractal toolbox: definitions and mathematics

### 2.1 Fractal dimension

For a set covered by $N(\epsilon)$ boxes of side $\epsilon$, the **box-counting (Minkowski–Bouligand) dimension** is

$$D_B = \lim_{\epsilon \to 0} \frac{\log N(\epsilon)}{\log (1/\epsilon)}$$

Estimated in practice by linear regression of $\log N(\epsilon)$ against $\log(1/\epsilon)$ over the scaling range. For a 2D thin-section pore mask, $1 < D_B < 2$; for a 3D micro-CT pore phase, $2 < D_B < 3$. Higher $D$ means the pore phase fills space more thoroughly and is more geometrically complex.

### 2.2 Hurst exponent and self-affinity

A **self-affine** field $\phi(\mathbf{x})$ satisfies, in distribution,

$$\phi(\lambda \mathbf{x}) \stackrel{d}{=} \lambda^{H} \phi(\mathbf{x})$$

where $H \in (0,1)$ is the Hurst exponent. Interpretation:

- $H = 0.5$ → uncorrelated increments (ordinary Brownian motion), no memory.
- $H > 0.5$ → **persistent**: high porosity tends to be followed by high porosity; long-range positive correlation. Most porosity logs fall here, typically $H \approx 0.7$–$0.9$.
- $H < 0.5$ → **antipersistent**: alternating/oscillatory, characteristic of thin cyclic bedding.

The connection to fractal dimension for a trace embedded in $E$-dimensional space:

$$D = E + 1 - H$$

So a 1D porosity log ($E=1$) with $H = 0.8$ has trace dimension $D = 1.2$; a 2D porosity map has surface dimension $D = 2.2$; a 3D field has $D = 3.2$ (co-dimension interpretation).

### 2.3 The three equivalent spectral / statistical signatures

These are all the same statement and you should verify at least two of them agree in your data — that consistency check is a strong methodological point.

**Power-law variogram:**
$$\gamma(h) = c \, h^{2H}, \qquad 0 < H < 1$$

Note: no sill. This is the "power model" available in most geostatistics packages.

**Power-law power spectral density:**
$$S(k) \propto k^{-\beta}, \qquad \beta = 2H + E$$

For a 1D trace, $\beta = 2H+1$; for a 2D field, $\beta = 2H+2$; for 3D, $\beta = 2H+3$.

**Rescaled range scaling:**
$$E\!\left[\frac{R(n)}{S(n)}\right] \propto n^{H}$$

where $R(n)$ is the range of cumulative deviations and $S(n)$ the standard deviation over windows of length $n$.

### 2.4 Multifractals

A monofractal has one exponent. A **multifractal** requires a continuous spectrum of exponents, which is far more appropriate for carbonates, where a tight micritic interval and a vuggy grainstone interval scale differently within the same well.

Partition the field into boxes of size $\epsilon$; let $p_i(\epsilon)$ be the measure (e.g. normalised porosity or pore-voxel count) in box $i$. The partition function:

$$\chi(q, \epsilon) = \sum_i p_i(\epsilon)^q \sim \epsilon^{\tau(q)}$$

**Mass exponent** $\tau(q)$ obtained from the slope of $\log \chi$ vs $\log \epsilon$ for each moment order $q$.

**Generalized (Rényi) dimensions:**
$$D_q = \frac{\tau(q)}{q-1}, \qquad q \neq 1; \qquad D_1 = \lim_{q\to1}\frac{\sum_i p_i \log p_i}{\log \epsilon}$$

- $D_0$ = capacity/box-counting dimension
- $D_1$ = information (entropy) dimension
- $D_2$ = correlation dimension

Monofractal ⟺ $D_q$ constant in $q$. The **degree of multifractality** is $D_0 - D_q^{max}$ or the spectrum width below.

**Singularity spectrum** via Legendre transform:
$$\alpha(q) = \frac{d\tau(q)}{dq}, \qquad f(\alpha) = q\,\alpha(q) - \tau(q)$$

Report:
- $\alpha_0$ (position of the maximum) — dominant singularity strength
- $\Delta\alpha = \alpha_{max} - \alpha_{min}$ — **spectrum width; this is your primary heterogeneity index.** Wide spectrum = strongly heterogeneous, multi-population pore system. Narrow = homogeneous.
- Asymmetry $R_\alpha = (\alpha_{max}-\alpha_0)/(\alpha_0-\alpha_{min})$ — right-skewed indicates dominance of low-porosity (tight) regions; left-skewed indicates dominance of high-porosity anomalies (vugs, dissolution).

$\Delta\alpha$ per facies is a beautiful, compact result to tabulate: it turns "this facies is heterogeneous" into a number.

### 2.5 Lacunarity — the essential companion to $D$

Because $D$ is non-unique, use the gliding-box lacunarity to characterise gappiness/texture:

$$\Lambda(\epsilon) = \frac{\langle m(\epsilon)^2 \rangle}{\langle m(\epsilon)\rangle^2} = 1 + \frac{\sigma^2_{m(\epsilon)}}{\mu^2_{m(\epsilon)}}$$

where $m(\epsilon)$ is the mass (pore voxel count) in a gliding box of size $\epsilon$. Plot $\log \Lambda$ vs $\log \epsilon$. High lacunarity at large $\epsilon$ = clustered, patchy pore distribution (vuggy carbonates). Low = homogeneous. Two fields with identical $D$ routinely have very different $\Lambda(\epsilon)$ curves.

### 2.6 Succolarity (optional, high value for flow)

Succolarity quantifies the degree to which a fractal permits percolation in a given direction. Computed by flooding the pore phase from one face and measuring occupancy-weighted pressure. It gives a *directional* connectivity descriptor that pairs naturally with the Euler characteristic already in your validation section, and it is under-used in the reservoir literature — an easy novelty angle.

---

## 3. Track A — Measuring fractal parameters from carbonate data

This is your characterization chapter. Pick the data you actually have; each subsection is independently doable.

### 3.1 From 1D well logs (highest value-to-effort ratio)

**Input:** any continuous porosity log (density-derived $\phi_D$, NMR $\phi_{NMR}$, neutron), sampled at 0.1524 m (6 in).

**Method 1 — Rescaled Range (R/S).** Split the log into non-overlapping windows of length $n$. For each window compute cumulative deviation from the mean, its range $R$, and its standard deviation $S$. Average $R/S$ across windows, plot vs $n$ in log-log, slope $= H$.

**Method 2 — Detrended Fluctuation Analysis (DFA).** Superior to R/S because it removes polynomial trends and therefore is not fooled by the non-stationarity that compaction and sequence-stratigraphic trends impose on your logs. Given your review's emphasis on trend/residual decomposition, DFA is the methodologically consistent choice.

Procedure: integrate the mean-removed series into a profile $Y(i)$; partition into windows of length $s$; fit an order-$m$ polynomial in each window and remove it; compute the fluctuation

$$F(s) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left[Y(i) - Y_{\text{fit}}(i)\right]^2} \propto s^{\alpha_{DFA}}$$

For a stationary series $H = \alpha_{DFA}$; for a non-stationary (fBm-like) series $H = \alpha_{DFA} - 1$.

**Method 3 — MF-DFA (multifractal DFA).** The generalisation that gives you the full spectrum from a log. Define

$$F_q(s) = \left\{\frac{1}{N_s}\sum_{v=1}^{N_s}\left[F^2(v,s)\right]^{q/2}\right\}^{1/q} \propto s^{h(q)}$$

with $\tau(q) = q\,h(q) - 1$ and then $\alpha, f(\alpha)$ by Legendre transform. Run $q$ from $-5$ to $+5$.

**Method 4 — Spectral.** Welch PSD of the log, fit $\beta$ from the log-log slope, convert with $H = (\beta-1)/2$.

**What to report:** $H$ per well, per stratigraphic zone, and per facies association. Then test: *does $H$ differ significantly between dolomitized and non-dolomitized intervals?* If yes — and it usually does, dolomitization tends to raise $H$ by homogenising and connecting the pore system — you have a direct, quantitative link between diagenesis and spatial statistics, which is exactly the gap your review identifies.

**Crucial control test.** Randomly shuffle the log and recompute. Shuffling destroys correlations, so $H$ must collapse to $\approx 0.5$ and $\Delta\alpha$ must shrink. Also test a phase-randomised (IAAFT) surrogate: this preserves the power spectrum but destroys nonlinear correlations, so if $\Delta\alpha$ drops relative to the original, your multifractality is genuinely from long-range correlation and not merely from a fat-tailed histogram. **Do this. It is the difference between a defensible result and a curve-fitting exercise.**

### 3.2 From capillary pressure (MICP)

The standard fractal capillary pressure model. For a fractal pore-throat size distribution, the wetting-phase saturation relates to capillary pressure as

$$S_w = \left(\frac{P_c}{P_{c,\min}}\right)^{D_f - 3}$$

so a log-log plot of $S_w$ versus $P_c$ has slope $D_f - 3$, giving

$$D_f = 3 + \frac{d(\log S_w)}{d(\log P_c)}$$

Equivalently in terms of mercury saturation $S_{Hg} = 1 - S_w \propto P_c^{D_f-3}$.

**The carbonate-specific insight:** a single straight line almost never fits carbonate MICP data. You typically see **two or three linear segments** corresponding to distinct pore systems — micropores (micrite), mesopores (intercrystalline/interparticle), macropores (moldic/vuggy). Fit each segment separately and report $D_{f,\text{micro}}$, $D_{f,\text{meso}}$, $D_{f,\text{macro}}$ with their crossover throat radii. Multi-segment fractal behaviour is *itself* the evidence for the multiscale pore system your review describes qualitatively. Physically valid dimensions must satisfy $2 < D_f < 3$; values outside that range signal either a bad segment choice or conformance/closure artefacts in the low-pressure data (always discard the first few low-pressure points).

Note that $D_f \to 3$ means an extremely rough, complex pore-throat surface with poor sorting, while $D_f \to 2$ approaches smooth, well-sorted capillary tubes. Correlate $D_f$ against permeability across your plug set — the usual result is a negative correlation, and it is a clean figure.

### 3.3 From NMR $T_2$ distributions

Under fast-diffusion, $1/T_2 = \rho_2 (S/V) \approx \rho_2 F_s / r$, so pore radius $r = \rho_2 F_s T_2$ with shape factor $F_s$ (3 for spheres, 2 for cylinders). Then the fractal relation on the cumulative $T_2$ distribution:

$$\log S_v = (3 - D_f)\log T_2 + (D_f - 3)\log T_{2,\max}$$

where $S_v$ is the cumulative pore volume fraction below $T_2$. Slope gives $D_f$ directly.

This is attractive because NMR gives you a **continuous downhole fractal dimension log** — $D_f(z)$ — computed in a moving window. That is a genuinely useful product: a vertical curve of pore-structure complexity that can be used as a conditioning variable or as an input feature to your XGBoost/PNN step.

### 3.4 From 2D images (thin sections, SEM, BSE)

1. **Segment** the pore phase. Use Otsu as a baseline but *always* run a threshold-sensitivity study: recompute $D$ over $\pm10\%$ threshold variation and report the induced spread as an error bar. Segmentation is the single largest source of error in image-based fractal dimension and reviewers know it.
2. **Box-counting** on the binary mask → $D_B$.
3. **Perimeter–area method** on individual pore objects: $P \propto A^{D_{PA}/2}$; regression of $\log P$ vs $\log A$ over all pores gives the boundary fractal dimension. Sensitive to shape complexity rather than space-filling.
4. **Two-point correlation function** $S_2(r)$; for a fractal, $S_2(r) - \phi^2 \propto r^{-(E - D)}$.
5. **Lacunarity** $\Lambda(\epsilon)$ curve.
6. **Multifractal** analysis on the greyscale image (using intensity as the measure) rather than the binary mask — this uses more information and avoids the segmentation problem entirely. Strongly recommended as a complement.

**Report per microfacies.** A table of $D_B$, $\Delta\alpha$, $\Lambda$ for grainstone / packstone / wackestone / mudstone / dolomitized facies is the backbone of your characterization chapter.

### 3.5 From 3D micro-CT

Same as above but in 3D. Use PoreSpy (Python) which has `porespy.metrics.two_point_correlation`, `chord_length_distribution`, `porosity_profile`, and `pc_curve`; box-counting in 3D is a short custom function. Public Estaillades / Ketton / Indiana Limestone volumes are freely available (Section 11) — this is the fastest path to real 3D fractal numbers if you have no proprietary data.

Additionally compute the **pore network fractal dimension** after extracting a pore network model (maximal-ball or SNOW algorithm): pore size distribution $n(r) \propto r^{-(D_f+1)}$, and coordination-number statistics.

### 3.6 From seismic

Compute the 2D radially-averaged power spectrum of an acoustic-impedance horizon slice; fit $\beta$; get lateral $H$. Do the same on vertical trace segments for vertical $H$. The ratio establishes an **independent, seismic-derived estimate of anisotropy** to compare against the variogram-derived $a_h/a_v$ ratio your review quotes at 10–100. Agreement between the two is a nice internal consistency check; disagreement is itself worth discussing.

---

## 4. Track B — Fractal simulation: generating the 3D porosity field

This is the core of the "3D modeling" part of your title.

### 4.1 Option 1 — Spectral synthesis (Fourier filtering method)

The simplest and fastest way to make a 3D fractal porosity field.

1. Generate a 3D array of i.i.d. $N(0,1)$ white noise, $W(\mathbf{x})$.
2. FFT to $\hat{W}(\mathbf{k})$.
3. Multiply by the fractal filter $|\mathbf{k}|^{-\beta/2}$ with $\beta = 2H + 3$ for a 3D field. For anisotropy, use an anisotropic wavenumber $|\mathbf{k}|_a = \sqrt{(k_x/a_x)^2 + (k_y/a_y)^2 + (k_z/a_z)^2}$, and optionally direction-dependent $H_x, H_y, H_z$ (self-affine rather than self-similar scaling — essential for layered carbonates).
4. Inverse FFT, take the real part → fBm field $Z(\mathbf{x})$.
5. Normal-score back-transform $Z$ to your target bounded porosity distribution (Beta or GMM, exactly as in your existing Section on bounded random fields):
   $$\phi(\mathbf{x}) = F_\phi^{-1}\!\left[\Phi\!\left(\frac{Z(\mathbf{x}) - \mu_Z}{\sigma_Z}\right)\right]$$

This last step is important and elegant: **it lets you keep everything your review already says about bounded random fields and Beta/GMM parameterization, and simply swap the underlying Gaussian random field from a short-range stationary one to a fractal one.** Minimal disruption, maximum gain.

Caveats: periodic boundary conditions (generate on a grid 2–4× larger and crop); low-wavenumber divergence (set $\hat{Z}(\mathbf{0}) = 0$ and optionally apply a low-wavenumber cutoff = "truncated power law").

### 4.2 Option 2 — Fractal variogram inside conventional SGS (recommended primary route)

You do not have to abandon SGS. You have to change its covariance model.

**The power model.** Most packages (GSLIB, SGeMS, GeostatsPy, gstools, Petrel) support a power variogram:
$$\gamma(h) = c\, h^{\omega}, \qquad \omega = 2H, \quad 0 < \omega < 2$$

This has no sill, which is exactly the point — but it also means the process is intrinsic, not second-order stationary. In practice this is handled by using **ordinary kriging** (which only requires intrinsic stationarity) rather than simple kriging inside the SGS loop, and by imposing a maximum lag / search radius. Discuss this explicitly in your methodology; it shows you understand the theory rather than just clicking a menu.

**The truncated power law (TPL) model — the practical compromise.** Restores a sill while keeping fractal scaling over a finite band:
$$\gamma_{TPL}(h) = \int_{\ell_{\min}}^{\ell_{\max}} \gamma_0(h;\ell)\, \ell^{-1-2H} \, d\ell$$
i.e. a superposition of elementary structures with a power-law-weighted range distribution. `gstools` implements `TPLGaussian`, `TPLExponential`, `TPLStable` directly. This is arguably the *most defensible* model for carbonates: real rocks are fractal between a lower cutoff (grain/crystal size) and an upper cutoff (bed thickness or platform cycle thickness), not to infinity. Estimating $\ell_{\min}$ and $\ell_{\max}$ from your data — and tying them to grain size and parasequence thickness respectively — is a genuinely good geological result.

**The Matérn bridge — connect to what you already wrote.** Your review already includes the Matérn class. Use this: for the Matérn covariance with smoothness $\nu$, the fractal dimension of the field is

$$D = E + 1 - \nu \quad (0 < \nu < 1), \qquad H = \nu$$

while the range parameter independently controls the correlation scale. This is the Gneiting–Schlather result that Matérn **decouples local fractal roughness from global correlation length** — which is precisely why it is the right model for carbonates, where a dolomitized grainstone might have long lateral continuity (large range) but a very rough local pore fabric (low $\nu$, high $D$). Fitting $\nu$ from your data and reporting the implied $D$, then cross-checking against the DFA-derived $H$ from Section 3.1, is an excellent verification loop and costs you almost nothing extra.

### 4.3 Option 3 — Successive Random Addition / midpoint displacement

Classical fBm generator on a $2^n+1$ grid. Recursively subdivide, interpolate midpoints, and add Gaussian noise scaled by $\sigma \cdot (1/2)^{nH}$. Fast, intuitive, easy to code from scratch (a plus for a BTP — you can show your own implementation), but produces grid artefacts along the recursion hierarchy and is awkward to condition. Use it for illustration and sensitivity studies, not as your production simulator.

### 4.4 Option 4 — Multiplicative cascades (the multifractal generator)

If your Track-A analysis shows genuine multifractality (and in carbonates it usually will), a monofractal fBm field is the wrong generator. Use a cascade.

**Discrete $p$-model / binomial cascade.** Start with a cube of unit measure. Subdivide into $2^3 = 8$ sub-cubes. Redistribute the parent measure among children with multipliers $W_i$ drawn from a chosen generator (e.g. $W \in \{p, 1-p\}$, or log-normal $\log W \sim N(-\sigma^2/2, \sigma^2)$) subject to $E[W] = 1$ for conservation. Repeat $n$ times. The resulting measure is multifractal with an analytically known $\tau(q)$, so you can **tune the generator to match the $f(\alpha)$ spectrum you measured from the logs.** That inversion — measured spectrum → cascade parameters → 3D field → verified spectrum — is a complete, self-contained, thesis-worthy contribution.

**Universal multifractals.** Parameterized by three numbers: $\alpha_{UM} \in (0,2]$ (Lévy index, degree of multifractality), $C_1$ (co-dimension of the mean singularity, intermittency), and $H$ (degree of non-conservation / smoothing). Generated via fractionally-integrated Lévy noise. More elegant and only three parameters, but heavier to implement.

**Anisotropic cascades.** Use a $2\times2\times1$ or $4\times4\times1$ subdivision to impose layer-cake anisotropy consistent with carbonate bedding — a small change with a large effect on realism.

### 4.5 Option 5 — Conditioning a fractal field to well data

Unconditional fractal fields are easy; conditional ones are the actual requirement. Two standard routes:

**(a) Kriging residual substitution (post-conditioning).** Given hard data $\phi(\mathbf{x}_\alpha)$:
$$\phi_c(\mathbf{x}) = \phi_k(\mathbf{x}) + \left[\phi_u(\mathbf{x}) - \phi_{uk}(\mathbf{x})\right]$$
where $\phi_k$ is the kriged estimate from the real data, $\phi_u$ is an unconditional fractal realization, and $\phi_{uk}$ is the kriged estimate built from $\phi_u$ sampled at the same well locations. Kriging must use the *same* fractal (power/TPL/Matérn) variogram. This exactly honours hard data and preserves the fractal covariance. It is simple, provable, and works with the spectral method of 4.1.

**(b) Fractal SGS (fSGS).** Run standard SGS but with the fractal variogram and ordinary kriging. Conditioning is intrinsic to the algorithm. Slower but cleaner, and gives you the sequential-simulation framing your review already builds.

For cascades, conditioning is harder — the usual approach is to condition the coarse levels of the cascade to well-derived block averages and let the fine levels run free, or to use a simulated-annealing post-processing step that swaps values to honour wells while minimising deviation of $f(\alpha)$ from the target.

### 4.6 Option 6 — Fractal object placement for vugs, karst, and fractures

Direct extension of your existing object-based and DFN sections. Instead of Gaussian or lognormal object sizes:

- **Vug/cave volumes:** draw from a power law $p(V) \propto V^{-b}$ over $[V_{\min}, V_{\max}]$; sample via inverse transform $V = V_{\min}(1-u)^{-1/(b-1)}$.
- **Fracture lengths:** $n(\ell) \propto \ell^{-a}$, with the well-known consequence that $a < 2$ means the largest fractures dominate connectivity and $a > 3$ means the small ones do — a directly interpretable, flow-relevant statement.
- **Fracture aperture:** scale with length, $e \propto \ell^{\,c}$ with $c \approx 0.5$–$1.0$; then aperture *field* on each fracture plane as a self-affine surface with $H \approx 0.8$, giving realistic channelised in-plane flow.
- **Fracture centre positions:** rather than Poisson (which is spatially uniform), place centres on a fractal point set of dimension $D_c < 3$ (e.g. via a Lévy-flight walk or a cascade-weighted thinning), producing realistic fracture clustering around faults. This is a well-known improvement over Poissonian DFNs and directly addresses the clustering that karstified carbonates exhibit.
- **Karst conduit growth:** DLA or invasion percolation seeded at the unconformity surface, grown downward with a probability field weighted by your dissolution-intensity mask. This gives geologically plausible, connected, dendritic conduit geometries that neither SGS nor MPS produces well, and it plugs straight into the "distance-to-unconformity" diagenetic module you already describe.

---

## 5. Track C — Fractal metrics as validation criteria

Your review's validation section currently lists histogram reproduction, variogram reproduction, VPC matching, P10/P50/P90, and connectivity. Add a fractal validation battery. This is low-effort, high-impact, and it is where the "failure demonstration" lives.

**The core experiment.** Generate matched ensembles with (i) standard SGS with a spherical variogram, (ii) SGS with a TPL/fractal variogram, (iii) MPS, and if you get there (iv) a GAN or LDM. All conditioned to the same wells, all reproducing the same histogram. Then compute, for every realization:

| Metric | What it tests |
|---|---|
| $H$ from DFA of vertical traces | Long-range vertical correlation |
| $H$ from radial PSD of horizontal slices | Lateral scaling |
| $D_0, D_1, D_2$ | Space-filling, entropy, correlation structure |
| $\Delta\alpha$, $\alpha_0$, asymmetry | Multifractal heterogeneity |
| $\Lambda(\epsilon)$ curve | Clustering / gappiness |
| $S_2(r)$ tail exponent | Two-point scaling |
| Euler characteristic vs. threshold | Topology / connectivity |
| Percolation threshold & backbone fraction | Flow connectivity |

Compare each against the well-data-derived target with a box plot across the ensemble.

**Expected and reportable result:** plain SGS with a spherical variogram will systematically *under*-estimate $H$ and severely *under*-estimate $\Delta\alpha$ — it produces near-monofractal fields because, as your review itself notes, it maximises entropy for a given two-point covariance. MPS will do better on $\Delta\alpha$ but is bounded by the training image's own spectrum. The fractal/TPL simulation and the cascade will hit the target by construction. Quantifying that gap, in numbers, on real data, is your headline figure.

**Flow consequence.** Take the ensembles into a simple two-phase or tracer flow simulation (even a streamline or a simple finite-difference single-phase tracer). Report breakthrough time and sweep efficiency distributions. The standard, well-documented outcome is that fractal/long-range-correlated fields give **earlier breakthrough and lower sweep efficiency** than short-range fields with identical histogram and identical variogram sill, because the persistent long-range correlation creates continuous high-permeability pathways. If you can show a P10–P90 recovery band that shifts materially between the two model families, you have demonstrated that fractal structure has economic consequence — which is the argument that makes the whole thesis matter.

---

## 6. Track D — Fractals and machine learning

Your review already has a substantial ML/generative section. Four clean integration points:

**6.1 Fractal descriptors as ML features.** Append $D_f$ (from NMR, windowed), $H$ (windowed DFA), local $\Delta\alpha$, and windowed lacunarity to your feature vector for XGBoost/PNN permeability or porosity prediction. Rationale: permeability depends on pore-structure complexity, not just porosity magnitude, and $D_f$ is a direct proxy for it. Report SHAP values to show whether the fractal features actually earn their place — an honest negative result here is still a result.

**6.2 Fractal spectral loss for generative models.** This is the most novel item on this list and directly targets the "physics-informed multi-scale generative models" gap your review identifies. Add to your GAN/LDM training objective a term penalising deviation of the generated field's radially-averaged power spectrum from the target power law:

$$\mathcal{L}_{\text{spec}} = \sum_{k} w_k \left\| \log \hat{S}_{\text{gen}}(k) - \log S_{\text{target}}(k) \right\|^2$$

or, more directly, a multifractal loss $\mathcal{L}_{MF} = \| f_{\text{gen}}(\alpha) - f_{\text{data}}(\alpha)\|_2^2$ evaluated on each minibatch. Total loss $\mathcal{L} = \mathcal{L}_{adv} + \lambda_1 \mathcal{L}_{\text{spec}} + \lambda_2 \mathcal{L}_{MF}$. This is a concrete, implementable answer to a stated open problem, and it is exactly the kind of scoped novelty a BTP should aim at. Known GAN failure modes include spectral bias (over-smoothing high wavenumbers) and mode collapse; a spectral loss addresses the first directly.

**6.3 Fractal fields as training data.** Cascade and fBm generators can produce thousands of geologically plausible 3D volumes essentially for free, which solves the "large training ensembles" data requirement your comparison table lists for GANs/LDMs. Pre-train on synthetic fractal volumes, fine-tune on the few real ones.

**6.4 Fractal super-resolution / downscaling.** Given a coarse seismic-resolution impedance-derived porosity volume, add fine-scale detail by superposing a conditional fractal field whose spectrum continues the observed power law below the seismic wavenumber cutoff. This is the classic "fractal interpolation" idea and it is a physically-principled alternative to a black-box super-resolution CNN — or you can do both and compare. It maps precisely onto the multiscale integration problem (seismic ~10 m vs. log ~0.15 m) that your review flags.

---

## 7. Fractals in the fracture–vug–karst system (expanding your Section on complex pore systems)

Beyond object placement (4.6), fractal theory reshapes the flow formulations themselves:

**Fractal dual-porosity.** The classical Warren–Root model assumes a uniform orthogonal matrix-block array. In a fractal fracture network the fracture density itself scales, giving a modified pressure-transient response. The fractal reservoir model (Chang & Yortsos formulation) replaces constant properties with

$$\phi(r) = \phi_0 \left(\frac{r}{r_w}\right)^{d_f - d}, \qquad k(r) = k_0 \left(\frac{r}{r_w}\right)^{d_f - d - \theta}$$

where $d_f$ is the mass fractal dimension of the network, $d$ the Euclidean dimension, and $\theta$ the anomalous diffusion (conductivity) exponent. The diffusivity equation becomes fractional, and the resulting pressure derivative shows a characteristic power-law slope rather than the flat radial-flow plateau — a directly testable well-test signature. Presenting this alongside Warren–Root in your dual-porosity section is a natural, well-cited upgrade.

**Anomalous / non-Fickian transport.** In fractal media, mean-square displacement scales as $\langle r^2 \rangle \propto t^{2/(2+\theta)}$ rather than $\propto t$. This motivates continuous-time random walk (CTRW) or fractional-derivative transport models, which explain the long tracer tails routinely observed in karstified carbonates and which standard dual-porosity models fit only by tuning shape factors. Worth at least a paragraph; a full implementation is beyond a BTP but the framing is valuable.

**Stylolites.** Self-affine surfaces with a documented crossover between two scaling regimes (roughly $H \approx 1.0$ at small scale where surface energy dominates, $H \approx 0.5$ at large scale where elastic energy dominates). Since your review already treats stylolites as flow barriers/conduits, generating them as self-affine surfaces rather than planes is a small, cheap realism gain.

---

## 8. Fractal permeability, capillary pressure, and upscaling

A useful bridge from geometry to flow — and a way to make the fractal work *do something* rather than just describe.

**Fractal capillary bundle permeability (Yu–Cheng type).** For a porous medium with pore diameters following a fractal distribution between $\lambda_{\min}$ and $\lambda_{\max}$, with pore-area fractal dimension $D_f$ and tortuosity fractal dimension $D_T$:

$$k = \frac{\pi}{128}\cdot\frac{L_0^{1-D_T}}{A}\cdot\frac{D_f}{3+D_T-D_f}\cdot \lambda_{\max}^{3+D_T}\left[1 - \left(\tfrac{\lambda_{\min}}{\lambda_{\max}}\right)^{3+D_T-D_f}\right]$$

with $D_T = 1 + \ln \bar{\tau} / \ln(L_0/\bar{\lambda})$. Every parameter is measurable from MICP or image analysis. Compare its predictions against measured core permeability and against a conventional Kozeny–Carman or Lucia rock-fabric prediction — a clean, quantitative comparison chapter that directly extends the Lucia framework already in your review.

**Fractal relative permeability and $P_c$ curves.** Because the fractal pore-size distribution is analytic, you can derive Brooks–Corey-type curves with the exponent tied to $D_f$: the Brooks–Corey pore size distribution index $\lambda_{BC} = 3 - D_f$. That means **one measured fractal dimension gives you the whole saturation-function family**, which is a very economical way to populate facies-dependent SCAL curves in a dynamic model. Strong practical selling point.

**Scale-dependent upscaling.** In a fractal permeability field, effective permeability depends on the averaging volume as a power law, $k_{\text{eff}}(L) \propto L^{\,\eta}$, rather than converging to a constant REV value. This is the theoretical explanation for the long-standing observation that carbonate permeability measured on plugs, whole cores, well tests, and interference tests systematically disagree. Demonstrating this scaling on your simulated fields — plot $k_{\text{eff}}$ vs. block size in log-log for a fractal field and for a conventional SGS field, and show one is a power law and one plateaus — is a compact, striking, and cheap result. It also gives you a principled answer to the "no REV in carbonates" problem.

---

## 9. Revised integrated workflow

Your seven stages become nine. Additions in **bold**.

1. **Fractal characterization of input data.** DFA/MF-DFA on all porosity logs; MICP fractal dimensions (multi-segment); NMR-derived $D_f(z)$ log; image-based $D_B$, $\Delta\alpha$, $\Lambda$ per microfacies; seismic spectral $H$. Surrogate-data significance tests throughout. Output: a calibrated table of $(H, D_f, \Delta\alpha, \Lambda)$ per facies and per diagenetic domain.

2. 3D structural corner-point grid from sequence-stratigraphic horizons. *(unchanged)*

3. Facies simulation via TPG/MPS conditioned to VPCs. **Optionally replace the latent Gaussian fields in TPG with fractal (fBm/TPL) latent fields — a one-line change that makes facies bodies fractally rough rather than artificially smooth, and is a nice, cheap novelty.**

4. Diagenetic overprinting modules → RRTs. **Karst/dissolution masks generated by DLA or invasion percolation seeded at unconformities; dolomitization fronts as fractal-rough surfaces propagating from faults.**

5. Facies-specific bounded porosity PDFs (Beta / GMM) within $[\phi_{\min}, \phi_{\max}]$. *(unchanged — and this stage is exactly where fractal fields plug in via normal-score back-transform)*

6. **Spatial structure: fit a fractal covariance model.** Truncated power law or Matérn with $\nu$ inverted from the measured $H$; direction-dependent $H_x, H_y, H_z$ with $\ell_{\min}$ tied to grain/crystal size and $\ell_{\max}$ tied to parasequence thickness. Report the implied $D$ and cross-check against Stage 1.

7. **Conditional fractal simulation.** fSGS with the fractal variogram, or spectral synthesis + kriging residual substitution, or a conditioned multiplicative cascade if Stage 1 showed strong multifractality. Conditioned to hard logs and to soft seismic via external drift / collocated co-kriging exactly as before.

8. **Fractal-aware validation.** Standard battery (histogram, variogram, VPC, P10/P50/P90) **plus** the fractal battery from Section 5, benchmarked against a plain-SGS control ensemble.

9. **Flow response and fractal upscaling.** Streamline or two-phase simulation on the ensembles; $k_{\text{eff}}(L)$ scaling analysis; recovery P10/P50/P90 comparison between fractal and conventional model families.

---

## 10. Three scoped project plans

Pick one honestly, based on your data access and remaining time. A well-executed Plan A beats a half-finished Plan C every time.

### Plan A — "Fractal characterization + validation" (safe, ~8–10 weeks, no proprietary data needed)

**Data:** public micro-CT carbonates (Estaillades, Ketton, Indiana Limestone) + any public carbonate well logs, or the open-source carbonate benchmark model.

1. Weeks 1–2: implement box-counting, DFA, MF-DFA, lacunarity, two-point correlation in Python. Validate every routine against synthetic fields of *known* $H$ (fBm generated by spectral synthesis) — this validation table is mandatory and examiners love it.
2. Weeks 3–4: measure $D$, $H$, $\Delta\alpha$, $\Lambda$ on real carbonate images/logs. Surrogate tests.
3. Weeks 5–6: build three ensembles (spherical SGS, TPL-fractal SGS, and an fBm spectral field) with GeostatsPy/gstools.
4. Weeks 7–8: fractal validation battery; the "SGS under-estimates $\Delta\alpha$" figure.
5. Weeks 9–10: single-phase tracer flow; breakthrough comparison; write-up.

**Deliverable:** a quantitative demonstration that conventional two-point geostatistics fails to reproduce measured carbonate multifractality, and that a fractal covariance fixes it, with flow consequences.

### Plan B — "Fractal simulator" (moderate, ~12 weeks)

Plan A plus: implement a 3D anisotropic multiplicative cascade generator; invert cascade parameters to match a measured $f(\alpha)$; implement conditioning via kriging residual substitution; add the fractal permeability model of Section 8 and compare against Kozeny–Carman and Lucia rock-fabric predictions on real plug data.

**Deliverable:** a working conditional multifractal 3D carbonate porosity simulator with a calibration-to-data inversion loop.

### Plan C — "Fractal-constrained generative model" (ambitious, ~14–16 weeks, needs GPU)

Plan A plus: train a 3D WGAN-GP or a small latent diffusion model on synthetic fractal volumes (Section 6.3), with and without the spectral/multifractal loss of Section 6.2. Show the loss term measurably improves spectral fidelity and $\Delta\alpha$ reproduction versus the vanilla model.

**Deliverable:** a direct, scoped attack on the "physics-informed generative models" gap your review identifies. Highest risk, highest ceiling. Only choose this if you already have GPU access and PyTorch fluency; the fractal analysis code (Plan A) is a prerequisite anyway, so you can start with A and escalate.

---

## 11. Data and software

**Public 3D carbonate images:** Digital Rocks Portal (digitalrocksportal.org) — Estaillades limestone, Ketton oolite, Indiana limestone, Mt. Gambier; Imperial College Pore-Scale Modelling group datasets.

**Public reservoir models:** UNISIM-II (Brazilian pre-salt carbonate benchmark, UNISIM/UNICAMP); the open-source carbonate reservoir model from *Petroleum Geoscience* (petgeo2021-067) already in your literature table; SPE Comparative Solution Projects (SPE10 is clastic but useful for method validation).

**Public well logs:** Kansas Geological Survey (extensive carbonate logs, free); USGS core research center; Volve (North Sea, clastic but fully open including seismic).

**Python:**
- `numpy`, `scipy`, `matplotlib` — base
- `gstools` — variogram models including `TPLGaussian`/`TPLStable`, SRF generation, Krige; **your best single tool for Section 4.2**
- `GeostatsPy` (Michael Pyrcz) — GSLIB-equivalent SGS, easy to modify
- `scikit-image` — segmentation, morphology
- `porespy` — 3D porous-media metrics, two-point correlation, chord lengths, PNM extraction
- `openpnm` — pore network modelling and network permeability
- `MFDFA` (pip) or `nolds` / `hurst` — DFA, MF-DFA, R/S, correlation dimension
- `pyfracval` / custom — box counting (10 lines; write it yourself and validate)
- `PyTorch` — for Plan C
- `MRST` (MATLAB) or `OPM Flow` (open source) — flow simulation

**Commercial (if your department has licences):** Petrel (has power variogram model and object modelling), SGeMS (free, has SNESIM/FILTERSIM), Ipsom/Isatis.

---

## 12. Pitfalls and statistical rigour

These are the things an examiner will probe. Address each explicitly in a "Limitations and Validation of Method" subsection.

1. **Scaling range.** You need at least 1.5–2 decades of $\epsilon$ with $R^2 > 0.98$ before claiming fractality. Report the fitted range and $R^2$ for every $D$ you quote. A short log or a small image simply cannot support a fractal claim.
2. **Never fit a single line to a curve that has crossovers.** Use a segmented regression with objective breakpoint detection, and interpret the crossovers physically (they are pore-system boundaries — that is a result, not an inconvenience).
3. **Segmentation sensitivity.** Vary the threshold ±10% and report the induced range in $D$ as an error bar.
4. **Finite-size and edge effects.** Box counting biases at both $\epsilon \to 1$ pixel and $\epsilon \to$ image size. Discard the extreme two or three box sizes at each end.
5. **Validate your code on known fields.** Generate fBm with $H = 0.3, 0.5, 0.7, 0.9$ by spectral synthesis, recover $H$ with each estimator, report the recovery error. Do the same for a binomial cascade with a known analytic $f(\alpha)$. This table belongs in your methods chapter.
6. **Surrogate tests for multifractality.** Shuffled surrogates and IAAFT phase-randomised surrogates, as in Section 3.1. Multifractality from fat tails alone is not multifractality from correlation structure, and conflating them is the most common error in this literature.
7. **Self-affine ≠ self-similar.** Carbonates are layered. Never apply an isotropic box-counting dimension to a vertically-anisotropic field without first rescaling the axes, or you will measure an artefact. Use direction-dependent $H$.
8. **Non-stationarity vs. fractality.** A trend (e.g. compaction-driven porosity decrease with depth) mimics long-range correlation. DFA of sufficient polynomial order handles this; R/S does not. Detrend first, and say so.
9. **The intrinsic-stationarity issue in kriging.** Power-model variograms have no sill; use ordinary kriging, not simple kriging, and note the consequence for the SGS conditional-variance argument in your existing text.
10. **Don't over-claim.** "Porosity exhibits multifractal scaling over 1.8 decades in the interval 1450–1620 m with $\Delta\alpha = 0.94$" is a good sentence. "Carbonate reservoirs are fractal" is not.

---

## 13. Additions to your literature catalogue

Rows to add to your comparative table, following your existing column structure. **Verify every DOI and page number yourself before submission** — treat the list below as a search index, not as verified citations.

| Theme | Anchor works to locate and cite |
|---|---|
| Foundational fractal geometry | Mandelbrot, *The Fractal Geometry of Nature* (1982); Feder, *Fractals* (1988); Turcotte, *Fractals and Chaos in Geology and Geophysics* (2nd ed., 1997) |
| Fractals in reservoir description | Hewett, "Fractal distributions of reservoir heterogeneity and their influence on fluid transport," SPE Annual Technical Conference, 1986 (SPE-15386) — **the origin paper for this entire application; cite it prominently** |
| Fractal pore space in rocks | Katz & Thompson, "Fractal sandstone pores," *Phys. Rev. Lett.* (1985); Krohn, "Fractal measurements of sandstones, shales and carbonates," *J. Geophys. Res.* (1988) — Krohn is your key carbonate-specific anchor |
| Fractal capillary pressure | Friesen & Mikula (1987) on MICP fractal analysis; Li, "Analytical derivation of Brooks–Corey type capillary pressure models using fractal geometry," *SPE J.* / *Transport in Porous Media* (2010) |
| Fractal permeability | Yu & Cheng, "A fractal permeability model for bi-dispersed porous media," *Int. J. Heat Mass Transfer* (2002); Yu & Li on tortuosity fractal dimension (2004) |
| NMR fractal | Numerous recent Chinese-journal papers on $T_2$-derived fractal dimensions in tight carbonates (search: "NMR $T_2$ fractal dimension carbonate reservoir") |
| Multifractal well logs | Muller & McCauley, "Implication of fractal geometry for fluid flow properties of sedimentary rocks," *Transport in Porous Media* (1992); Kantelhardt et al., "Multifractal detrended fluctuation analysis," *Physica A* (2002) — the MF-DFA method paper, essential |
| Matérn ↔ fractal dimension | Gneiting & Schlather, "Stochastic models that separate fractal dimension and the Hurst effect," *SIAM Review* (2004) — **the cleanest bridge to your existing Matérn section** |
| Fractal / TPL variograms | Di Federico & Neuman on truncated power-law variograms and scale-dependent dispersion, *Water Resources Research* (1997–1998) |
| Fractal fracture networks | Bonnet et al., "Scaling of fracture systems in geological media," *Rev. Geophysics* (2001) — the definitive review of power-law fracture statistics |
| Fractal well testing | Chang & Yortsos, "Pressure transient analysis of fractal reservoirs," *SPE Formation Evaluation* (1990); Beier (1994) |
| Stylolite roughness | Schmittbuhl, Renard, Gratier and co-workers on self-affine stylolite scaling, *Phys. Rev. Lett.* / *J. Geophys. Res.* (2004 onward) |
| Karst as DLA/percolation | Literature on speleogenesis modelling and invasion percolation in karst aquifers |
| Anomalous transport | Berkowitz & Scher on CTRW in fractured/heterogeneous media, *Water Resources Research* |
| Universal multifractals | Schertzer & Lovejoy (1987 onward) |
| Fractal + deep learning | Recent work on spectral bias in GANs and on physics-constrained generative models for porous media — search "spectral loss GAN porous media", "multifractal loss generative reservoir" |

---

## 14. Suggested new sections for your thesis document

To slot into your existing structure with minimum disruption:

- Insert **"Fractal and Multifractal Characterization of Carbonate Pore Systems"** as a new major section immediately after *Carbonate-Specific Geological Complexity and Diagenetic Overprinting* — it is the natural bridge between geology and statistics.
- Insert **"Fractal and Scale-Invariant Covariance Models"** as a subsection of your existing *Spatial Correlation Models and Anisotropy Characterization*, immediately after the Matérn paragraph, using the Gneiting–Schlather link.
- Insert **"Fractal Object and Network Models"** as a subsection of *Representation of Complex Pore Systems: Fractures and Vugs*.
- Add a **"Fractal Validation Metrics"** subsection to *Statistical Validation Metrics*.
- Add a **"Fractal"** row to your methodological comparison table (see below).
- Rewrite Research Gap #1 ("Controllable Bounded Random Fields with Non-Stationary Bounds") to note that spatially-varying $H(\mathbf{x})$ — a *multifractional* Brownian field — is the natural formalism for spatially-varying heterogeneity, alongside spatially-varying bounds. That connects your existing gap statement to the fractal framework and makes the whole thing cohere.

**Proposed new row for your comparison table:**

| Method | Data Requirements | Captures Spatial Correlation | Captures Geological Patterns | Handles Facies | Computational Cost | Main Limitations |
|---|---|---|---|---|---|---|
| **Fractal / Multifractal Simulation (fBm, TPL-SGS, cascades)** | Logs or images sufficient for $H$/$f(\alpha)$ estimation; 1.5+ decades of scaling range | Power-law covariance across all scales; reproduces long-range persistence | Moderate–high for texture and heterogeneity; poor for specific curvilinear bodies (no shape control) | Indirectly, via facies-specific $H$ and $\Delta\alpha$; can be nested inside TPG | Very low for spectral synthesis ($O(N\log N)$); low–moderate for cascades and fSGS | Non-stationarity conflicts with kriging assumptions; $D$ non-unique; requires a genuine scaling range; conditioning cascades is non-trivial |

---

## 15. What to do first, concretely

If you want a single starting task this week:

1. Take one porosity log. Run DFA and MF-DFA on it. Get $H$ and $\Delta\alpha$.
2. Generate an fBm 1D trace with that same $H$ by spectral synthesis, and an SGS trace with a fitted spherical variogram.
3. Run DFA and MF-DFA on both synthetics.
4. Plot the three $f(\alpha)$ spectra on one axis.

That single figure — real log, fractal simulation, conventional simulation — will tell you within a day whether the whole thesis direction has legs, and if the spectra separate the way they usually do, it becomes Figure 1 of your results chapter.

---

*Note on sources: the reference list in Section 13 is a search index compiled from general knowledge of this literature, not from verified retrieval. Confirm authors, years, venues, and DOIs before citing. The equations in Sections 2, 4, and 8 are standard forms; check the original papers for exact conventions, particularly the sign and definition of $\beta$ in the PSD relations and of $D_T$ in the Yu–Cheng permeability model, which vary between authors.*