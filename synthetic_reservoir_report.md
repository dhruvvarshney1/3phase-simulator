# Synthetic 3D Reservoir Grid Report

**Project:** Incorporating Fractal Analysis into 3D Porosity Simulation of Carbonate Reservoirs  
**Script:** `create_synthetic_reservoir.py`

---

## 1. Overview

This report documents the creation of a synthetic $2 \times 2 \times 1$ 3D reservoir grid with randomly assigned porosity values. The grid serves as a foundational test case for subsequent fractal analysis and simulation workflows described in the project plan.

## 2. Grid Specification

| Parameter | Value |
|---|---|
| Grid dimensions | $N_x = 2$, $N_y = 2$, $N_z = 1$ |
| Total blocks | 4 |
| Block dimensions | $\Delta x = 20\text{ ft}$, $\Delta y = 20\text{ ft}$, $\Delta z = 20\text{ ft}$ |
| Seed | 42 (reproducible) |

Spatial coordinates (block centers) range from 10 ft to 30 ft in X and Y, and 10 ft in Z.

## 3. Porosity Distribution

Random porosity $\phi$ assigned to each block as a uniform decimal in $[0.10, 0.30]$:

| Block (i,j,k) | X (ft) | Y (ft) | Z (ft) | Porosity $\phi$ |
|---|---|---|---|---|
| (0,0,0) | 10.0 | 10.0 | 10.0 | 0.1749 |
| (0,1,0) | 10.0 | 30.0 | 10.0 | 0.2901 |
| (1,0,0) | 30.0 | 10.0 | 10.0 | 0.2464 |
| (1,1,0) | 30.0 | 30.0 | 10.0 | 0.2197 |

**Summary statistics:**
- Porosity range: 0.17 – 0.29
- Mean porosity: 0.2328

## 4. Data Representation

Properties stored in a pandas DataFrame with columns:
- Block indices: $i, j, k$
- Spatial coordinates: $X, Y, Z$ (ft)
- Porosity: $\phi$ (decimal)

The DataFrame enables straightforward export to CSV, integration with geostatistical packages, and serves as input for subsequent fractal simulation and validation workflows.

## 5. Visualization

- **2D top view** ($\phi$ layout at $k=0$) displayed via `imshow` with colorbar
- **3D scatter plot** of block centers colored by porosity (matplotlib `Axes3D`)
- Text labels on each block showing indices and porosity values

## 6. Context within Project

This synthetic grid represents the simplest possible test case ($2\times2\times1$) for:
- Validating grid generation pipelines
- Testing fractal parameter estimation routines (DFA, MF-DFA, lacunarity)
- Benchmarking against conventional SGS and fractal covariance models
- Serving as a minimal "Hello World" before progressing to larger $N_x \times N_y \times N_z$ realizations

The fixed random seed (42) ensures reproducibility across experiments, consistent with the project's emphasis on surrogate data significance tests and controlled methodology.

## 7. Output Files

- `create_synthetic_reservoir.py` — Python script generating the grid
- Console output containing the structured DataFrame and summary statistics
- Matplotlib figures (2D layout + 3D scatter when run with display)