"""
synthetic_reservoir.py
======================
Physically consistent INITIAL 3D reservoir model generator for a future
fully-implicit black-oil simulator.

Scope of this module (deliberately no flow solver):
  * Cartesian cell-centred grid with explicit depth (structural dome + dip)
  * Layered stratigraphy + meandering channel bodies (4 facies)
  * Facies-conditioned, spatially correlated porosity / anisotropic permeability
  * Explicit fault surfaces -> cell masks + FACE transmissibility multipliers
  * Explicit fracture corridors (equivalent-continuum, NOT DFM)
  * Hydrostatic two-fluid initial pressure, OWC-based saturations
  * Explicit vertical wells with facies-aware perforations
  * Validation, NPZ + JSON output, matplotlib visualisation

ALL NUMERICAL RANGES BELOW ARE ILLUSTRATIVE / SYNTHETIC. They are chosen to be
"textbook plausible" for a clastic reservoir but are NOT calibrated to any real
field and must not be presented as such.

Units (SI unless stated):  length m, pressure Pa, density kg/m3,
                           permeability mD (1 mD = 9.869233e-16 m2).
Depth convention:          depth D is POSITIVE DOWNWARD, layer k=0 is the TOP.
Array convention:          all cell arrays have shape (Nx, Ny, Nz), C-order,
                           meshgrid indexing='ij'.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

MD_TO_M2 = 9.869233e-16
G = 9.80665

# ----------------------------------------------------------------------------
# 0. CONFIGURATION
# ----------------------------------------------------------------------------
FACIES_CODES = {"shale": 0, "shaly_sandstone": 1, "sandstone": 2, "channel_sandstone": 3}
FACIES_NAMES = {v: k for k, v in FACIES_CODES.items()}


@dataclass
class FaciesParams:
    """Per-facies petrophysical distribution. ILLUSTRATIVE values."""
    name: str
    code: int
    phi_mean: float
    phi_std: float
    phi_min: float
    phi_max: float
    logk_a: float          # log10(k[mD]) = a + b*phi + sigma*Z   (Z: correlated N(0,1))
    logk_b: float
    logk_sigma: float      # residual scatter -> phi does NOT uniquely determine k
    k_min: float           # mD
    k_max: float           # mD
    kv_kh: float           # geometric-mean vertical/horizontal ratio
    kv_kh_sigma_log10: float
    swc: float             # connate water saturation
    net: bool              # False -> treated as non-reservoir (Sw = 1, never perforated)


def default_facies_table() -> Dict[int, FaciesParams]:
    return {
        0: FaciesParams("shale", 0, 0.05, 0.015, 0.01, 0.10, -3.0, 20.0, 0.40, 1e-4, 0.5, 0.05, 0.30, 1.00, False),
        1: FaciesParams("shaly_sandstone", 1, 0.13, 0.030, 0.06, 0.20, -1.5, 15.0, 0.50, 0.1, 50.0, 0.10, 0.30, 0.35, True),
        2: FaciesParams("sandstone", 2, 0.20, 0.035, 0.12, 0.28, -0.5, 12.0, 0.40, 5.0, 1000.0, 0.20, 0.25, 0.22, True),
        3: FaciesParams("channel_sandstone", 3, 0.26, 0.030, 0.18, 0.34, 0.3, 11.0, 0.35, 100.0, 8000.0, 0.35, 0.20, 0.15, True),
    }


@dataclass
class ReservoirConfig:
    # --- geometry ---
    Nx: int = 40
    Ny: int = 40
    Nz: int = 10
    Lx: float = 1000.0
    Ly: float = 1000.0
    Lz: float = 50.0
    top_depth: float = 2000.0          # depth of model top at (x,y)=(0,0) before dome
    dip_x: float = 0.005               # structural dip (m/m), +x deepens
    dip_y: float = 0.003
    dome_center: tuple = (600.0, 400.0)
    dome_amplitude: float = 15.0       # crest is shallower by this much
    dome_radius: float = 350.0
    # --- stratigraphy (top -> bottom, one code per layer) ---
    layer_sequence: List[int] = field(default_factory=lambda: [0, 1, 2, 2, 0, 1, 2, 2, 1, 0])
    lateral_shaliness_threshold: float = -0.9   # in Z-units of a correlated field
    lateral_cleaning_threshold: float = 1.1
    channels: List[dict] = field(default_factory=lambda: [
        dict(name="CH1", layers=[2, 3], y0=300.0, amplitude=180.0, wavelength=800.0, phase=0.3, width=140.0),
        dict(name="CH2", layers=[6, 7], y0=650.0, amplitude=120.0, wavelength=650.0, phase=2.0, width=110.0),
    ])
    # --- correlated heterogeneity ---
    corr_len_xy: float = 180.0         # m, horizontal correlation length
    corr_len_z: float = 6.0            # m, vertical correlation length
    spectrum: str = "gaussian"         # "gaussian" | "fractal"  (fractal hook for later experiments)
    fractal_beta: float = 3.0
    ky_kx_ratio: float = 0.85          # mild horizontal anisotropy (depositional)
    # --- faults ---
    faults: List[dict] = field(default_factory=lambda: [
        dict(name="F1", x0=560.0, y0=0.0, z0=2000.0, strike_deg=75.0, dip_deg=80.0,
             curvature=2.0e-4, trans_mult=0.02, thickness=12.0),
    ])
    # --- fracture corridors (equivalent continuum) ---
    fractures: List[dict] = field(default_factory=lambda: [
        dict(name="FR1", xc=300.0, yc=700.0, strike_deg=30.0, length=350.0, width=25.0,
             layers=[1, 2, 3], aperture_mm=0.5, k_mult_along=40.0, k_mult_across=4.0, kz_mult=10.0),
        dict(name="FR2", xc=750.0, yc=250.0, strike_deg=120.0, length=280.0, width=25.0,
             layers=[5, 6, 7], aperture_mm=0.3, k_mult_along=25.0, k_mult_across=3.0, kz_mult=8.0),
        dict(name="FR3", xc=200.0, yc=250.0, strike_deg=60.0, length=200.0, width=25.0,
             layers=[6, 7, 8], aperture_mm=0.4, k_mult_along=30.0, k_mult_across=3.0, kz_mult=8.0),
    ])
    # --- fluids / initial state ---
    owc_depth: float = 2040.0
    transition_zone: float = 4.0       # m, linear Sw ramp above OWC (crude Pc surrogate)
    p_ref: float = 20.4e6              # Pa at depth_ref
    depth_ref: float = 2040.0          # = OWC (so Po = Pw there, Pc ignored)
    rho_w: float = 1020.0
    rho_o: float = 800.0
    # --- wells ---
    wells: List[dict] = field(default_factory=lambda: [
        dict(name="INJ1", type="injector", x=150.0, y=150.0, control="rate",
             value=300.0, units="m3/day", rw=0.1, skin=0.0, perforate="reservoir"),
        dict(name="PROD1", type="producer", x=640.0, y=430.0, control="bhp",
             value=17.0e6, units="Pa", rw=0.1, skin=0.0, perforate="oil_reservoir"),
    ])
    seed: int = 20240601


# ----------------------------------------------------------------------------
# 1. GRID
# ----------------------------------------------------------------------------
def build_grid(cfg: ReservoirConfig) -> dict:
    dx, dy, dz = cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny, cfg.Lz / cfg.Nz
    x = (np.arange(cfg.Nx) + 0.5) * dx
    y = (np.arange(cfg.Ny) + 0.5) * dy
    zrel = (np.arange(cfg.Nz) + 0.5) * dz          # depth below local top of model
    X, Y, Zrel = np.meshgrid(x, y, zrel, indexing="ij")

    xc, yc = cfg.dome_center
    dome = cfg.dome_amplitude * np.exp(-((X[:, :, 0] - xc) ** 2 + (Y[:, :, 0] - yc) ** 2) / (2 * cfg.dome_radius ** 2))
    top = cfg.top_depth + cfg.dip_x * X[:, :, 0] + cfg.dip_y * Y[:, :, 0] - dome   # (Nx,Ny)
    depth = top[:, :, None] + Zrel                                                # (Nx,Ny,Nz)

    return dict(
        Nx=cfg.Nx, Ny=cfg.Ny, Nz=cfg.Nz, Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz,
        dx=dx, dy=dy, dz=dz, x=x, y=y, X=X, Y=Y, depth=depth, top_depth_map=top,
        cell_volume=np.full((cfg.Nx, cfg.Ny, cfg.Nz), dx * dy * dz),
        depth_convention="positive downward; k=0 is the top layer",
    )


# ----------------------------------------------------------------------------
# 2. CORRELATED RANDOM FIELDS
# ----------------------------------------------------------------------------
def gaussian_random_field(shape, corr_len_cells, rng, spectrum="gaussian", beta=3.0) -> np.ndarray:
    """
    Zero-mean, unit-variance spatially correlated field via FFT spectral filtering.
      spectrum='gaussian': Gaussian covariance with per-axis correlation lengths.
      spectrum='fractal' : power-law S(k) ~ (|k|^2 + k0^2)^(-beta/2)  (fBm-like);
                           hook for the later fractal heterogeneity experiments.
    """
    freqs = [np.fft.fftfreq(n) for n in shape]
    K = np.meshgrid(*freqs, indexing="ij")
    if spectrum == "gaussian":
        arg = sum(((2 * np.pi * k * l) ** 2) for k, l in zip(K, corr_len_cells))
        S = np.exp(-0.5 * arg)
    elif spectrum == "fractal":
        k2 = sum(((k * l / max(corr_len_cells)) ** 2) for k, l in zip(K, corr_len_cells))
        k0 = 1.0 / max(corr_len_cells)
        S = (k2 + k0 ** 2) ** (-beta / 2.0)
    else:
        raise ValueError(spectrum)
    white = rng.standard_normal(shape)
    f = np.real(np.fft.ifftn(np.fft.fftn(white) * np.sqrt(S)))
    f = (f - f.mean()) / (f.std() + 1e-15)
    return f


# ----------------------------------------------------------------------------
# 3. FACIES ARCHITECTURE
# ----------------------------------------------------------------------------
def build_facies(grid: dict, cfg: ReservoirConfig, rng: np.random.Generator) -> dict:
    Nx, Ny, Nz = grid["Nx"], grid["Ny"], grid["Nz"]
    assert len(cfg.layer_sequence) == Nz, "layer_sequence must have Nz entries"

    # (a) layer-cake background
    facies = np.broadcast_to(np.array(cfg.layer_sequence)[None, None, :], (Nx, Ny, Nz)).copy()

    # (b) lateral facies variability inside sand/shaly-sand layers (correlated, not white noise)
    lc = (cfg.corr_len_xy / grid["dx"], cfg.corr_len_xy / grid["dy"], cfg.corr_len_z / grid["dz"])
    Zf = gaussian_random_field((Nx, Ny, Nz), lc, rng, cfg.spectrum, cfg.fractal_beta)
    facies[(facies == 2) & (Zf < cfg.lateral_shaliness_threshold)] = 1     # shaly patches in sand
    facies[(facies == 1) & (Zf > cfg.lateral_cleaning_threshold)] = 2      # clean lenses in shaly sand

    # (c) meandering channel bodies (sinusoidal centreline, erode whatever is there)
    X, Y = grid["X"], grid["Y"]
    channel_mask = np.zeros((Nx, Ny, Nz), dtype=np.int16)
    for ci, ch in enumerate(cfg.channels):
        yc = ch["y0"] + ch["amplitude"] * np.sin(2 * np.pi * X[:, :, 0] / ch["wavelength"] + ch["phase"])
        inside_xy = np.abs(Y[:, :, 0] - yc) <= ch["width"] / 2
        for k in ch["layers"]:
            facies[:, :, k][inside_xy] = 3
            channel_mask[:, :, k][inside_xy] = ci + 1

    return dict(facies=facies.astype(np.int8), channel_mask=channel_mask, facies_field_Z=Zf)


# ----------------------------------------------------------------------------
# 4. POROSITY & PERMEABILITY
# ----------------------------------------------------------------------------
def build_porosity(facies, table, grid, cfg, rng) -> np.ndarray:
    lc = (cfg.corr_len_xy / grid["dx"], cfg.corr_len_xy / grid["dy"], cfg.corr_len_z / grid["dz"])
    Z = gaussian_random_field(facies.shape, lc, rng, cfg.spectrum, cfg.fractal_beta)
    phi = np.zeros_like(Z)
    for code, fp in table.items():
        m = facies == code
        phi[m] = np.clip(fp.phi_mean + fp.phi_std * Z[m], fp.phi_min, fp.phi_max)
    return phi


def build_permeability(facies, phi, table, grid, cfg, rng) -> dict:
    lc = (cfg.corr_len_xy / grid["dx"], cfg.corr_len_xy / grid["dy"], cfg.corr_len_z / grid["dz"])
    Zk = gaussian_random_field(facies.shape, lc, rng, cfg.spectrum, cfg.fractal_beta)   # independent of phi field
    Zv = gaussian_random_field(facies.shape, (lc[0] / 2, lc[1] / 2, lc[2]), rng, cfg.spectrum, cfg.fractal_beta)
    kx = np.zeros_like(phi)
    kz = np.zeros_like(phi)
    for code, fp in table.items():
        m = facies == code
        log10k = fp.logk_a + fp.logk_b * phi[m] + fp.logk_sigma * Zk[m]
        kx[m] = np.clip(10.0 ** log10k, fp.k_min, fp.k_max)
        ratio = 10.0 ** (np.log10(fp.kv_kh) + fp.kv_kh_sigma_log10 * Zv[m])
        kz[m] = kx[m] * np.clip(ratio, 1e-3, 1.0)
    ky = kx * cfg.ky_kx_ratio
    return dict(kx=kx, ky=ky, kz=kz)


# ----------------------------------------------------------------------------
# 5. FAULTS  (explicit surfaces -> cell mask + face multipliers)
# ----------------------------------------------------------------------------
def _fault_frame(fd: dict):
    """Unit vectors: s (along strike), n (unit normal), e (down-dip within plane)."""
    st, dp = np.deg2rad(fd["strike_deg"]), np.deg2rad(fd["dip_deg"])
    s = np.array([np.cos(st), np.sin(st), 0.0])
    n = np.array([-np.sin(st) * np.sin(dp), np.cos(st) * np.sin(dp), -np.cos(dp)])
    e = np.cross(n, s)
    p0 = np.array([fd["x0"], fd["y0"], fd["z0"]])
    return s, n, e, p0


def fault_signed_distance(fd: dict, X, Y, D) -> np.ndarray:
    """d = n.(r-p0) + c*t^2 ; t = along-strike coordinate. c>0 -> gently curved fault."""
    s, n, _, p0 = _fault_frame(fd)
    rx, ry, rz = X - p0[0], Y - p0[1], D - p0[2]
    t = s[0] * rx + s[1] * ry
    return n[0] * rx + n[1] * ry + n[2] * rz + fd.get("curvature", 0.0) * t ** 2


def fault_surface_points(fd: dict, grid: dict, nt=40, nu=12):
    """Exact points on the d=0 surface for plotting: r = p0 + t s + u e - c t^2 n."""
    s, n, e, p0 = _fault_frame(fd)
    L = np.hypot(grid["Lx"], grid["Ly"])
    T, U = np.meshgrid(np.linspace(-L, L, nt), np.linspace(-5, grid["Lz"] + 40, nu), indexing="ij")
    c = fd.get("curvature", 0.0)
    P = p0[None, None, :] + T[..., None] * s + U[..., None] * e - (c * T ** 2)[..., None] * n
    inside = (P[..., 0] >= 0) & (P[..., 0] <= grid["Lx"]) & (P[..., 1] >= 0) & (P[..., 1] <= grid["Ly"])
    P[~inside] = np.nan
    return P[..., 0], P[..., 1], P[..., 2]


def build_faults(grid: dict, cfg: ReservoirConfig) -> dict:
    Nx, Ny, Nz = grid["Nx"], grid["Ny"], grid["Nz"]
    X, Y, D = grid["X"], grid["Y"], grid["depth"]
    # face-multiplier arrays are aligned with INTERNAL faces
    mult_x = np.ones((Nx - 1, Ny, Nz)); mult_y = np.ones((Nx, Ny - 1, Nz)); mult_z = np.ones((Nx, Ny, Nz - 1))
    face_x = np.zeros_like(mult_x, dtype=np.int8); face_y = np.zeros_like(mult_y, dtype=np.int8); face_z = np.zeros_like(mult_z, dtype=np.int8)
    cell_mask = np.zeros((Nx, Ny, Nz), dtype=np.int8)
    fault_defs = []
    for fi, fd in enumerate(cfg.faults):
        d = fault_signed_distance(fd, X, Y, D)
        cell_mask[np.abs(d) <= fd.get("thickness", grid["dx"]) / 2] = fi + 1
        cx = d[:-1] * d[1:] < 0; cy = d[:, :-1] * d[:, 1:] < 0; cz = d[:, :, :-1] * d[:, :, 1:] < 0
        for cm, fm, mm in ((cx, face_x, mult_x), (cy, face_y, mult_y), (cz, face_z, mult_z)):
            fm[cm] = fi + 1
            mm[cm] = np.minimum(mm[cm], fd["trans_mult"])
        s, n, e, p0 = _fault_frame(fd)
        fault_defs.append(dict(fd, id=fi + 1, strike_vec=s.tolist(), normal_vec=n.tolist(), point=p0.tolist(),
                               n_cells=int((cell_mask == fi + 1).sum()),
                               n_faces=int((face_x == fi + 1).sum() + (face_y == fi + 1).sum() + (face_z == fi + 1).sum())))
    return dict(cell_mask=cell_mask, face_mask_x=face_x, face_mask_y=face_y, face_mask_z=face_z,
                trans_mult_x=mult_x, trans_mult_y=mult_y, trans_mult_z=mult_z, definitions=fault_defs,
                note="Face arrays index internal faces: x-face m sits between cells i=m and i=m+1. "
                     "Throw/juxtaposition is NOT modelled (numerical simplification).")


# ----------------------------------------------------------------------------
# 6. FRACTURE CORRIDORS (equivalent continuum)
# ----------------------------------------------------------------------------
def build_fractures(grid: dict, cfg: ReservoirConfig, kx, ky, kz) -> dict:
    """
    Equivalent-continuum representation: cells inside a corridor get their MATRIX
    permeability multiplied. This is NOT a discrete fracture model (no separate
    fracture DOF, no explicit aperture flow law). Aperture is stored as metadata only.
    Anisotropic multiplier projected on the grid axes (TPFA cannot carry full tensors).
    """
    X, Y = grid["X"][:, :, 0], grid["Y"][:, :, 0]
    mask = np.zeros(kx.shape, dtype=np.int8)
    kx_f, ky_f, kz_f = kx.copy(), ky.copy(), kz.copy()
    defs = []
    for fi, fr in enumerate(cfg.fractures):
        th = np.deg2rad(fr["strike_deg"])
        s = np.array([np.cos(th), np.sin(th)]); nrm = np.array([-np.sin(th), np.cos(th)])
        rx, ry = X - fr["xc"], Y - fr["yc"]
        t = s[0] * rx + s[1] * ry; u = nrm[0] * rx + nrm[1] * ry
        in_xy = (np.abs(t) <= fr["length"] / 2) & (np.abs(u) <= fr["width"] / 2)
        mx = 1 + (fr["k_mult_along"] - 1) * np.cos(th) ** 2 + (fr["k_mult_across"] - 1) * np.sin(th) ** 2
        my = 1 + (fr["k_mult_along"] - 1) * np.sin(th) ** 2 + (fr["k_mult_across"] - 1) * np.cos(th) ** 2
        for k in fr["layers"]:
            sel = in_xy & (mask[:, :, k] == 0)
            kx_f[:, :, k][sel] *= mx; ky_f[:, :, k][sel] *= my; kz_f[:, :, k][sel] *= fr["kz_mult"]
            mask[:, :, k][sel] = fi + 1
        # corners of the corridor rectangle for plotting / metadata
        corners = [fr["xc"] + a * s[0] * fr["length"] / 2 + b * nrm[0] * fr["width"] / 2 for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        cornersy = [fr["yc"] + a * s[1] * fr["length"] / 2 + b * nrm[1] * fr["width"] / 2 for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        defs.append(dict(fr, id=fi + 1, corners_x=corners, corners_y=cornersy, n_cells=int((mask == fi + 1).sum()),
                         representation="equivalent_continuum"))
    return dict(mask=mask, kx=kx_f, ky=ky_f, kz=kz_f, definitions=defs)


# ----------------------------------------------------------------------------
# 7 & 8. INITIAL PRESSURE AND SATURATION
# ----------------------------------------------------------------------------
def build_initial_state(grid: dict, facies, table, cfg: ReservoirConfig) -> dict:
    """
    Pressure: depth D positive downward; P increases with D.
        below OWC : P = P_ref + rho_w g (D - D_ref)
        above OWC : P = P_owc + rho_o g (D - D_owc)   (continuous at OWC, Pc = 0 assumed)
    Saturation: Sw = swc(facies) above OWC (+ linear transition zone), 1 in water leg;
                shale is non-net -> Sw = 1. Sg = 0 (undersaturated oil start).
    """
    D = grid["depth"]
    P_owc = cfg.p_ref + cfg.rho_w * G * (cfg.owc_depth - cfg.depth_ref)
    P = np.where(D >= cfg.owc_depth,
                 P_owc + cfg.rho_w * G * (D - cfg.owc_depth),
                 P_owc + cfg.rho_o * G * (D - cfg.owc_depth))

    swc = np.zeros_like(D)
    for code, fp in table.items():
        swc[facies == code] = fp.swc if fp.net else 1.0
    h = cfg.owc_depth - D                                    # height above OWC (>0 in oil leg)
    tz = max(cfg.transition_zone, 1e-9)
    frac = np.clip(h / tz, 0.0, 1.0)                         # 0 at/below OWC, 1 above transition
    Sw = 1.0 - frac * (1.0 - swc)
    Sw = np.clip(Sw, 0.0, 1.0)
    So = 1.0 - Sw
    Sg = np.zeros_like(Sw)
    return dict(pressure=P, Sw=Sw, So=So, Sg=Sg, swc=swc, P_owc=P_owc,
                pressure_units="Pa", convention="depth positive downward, P increases with depth")


# ----------------------------------------------------------------------------
# 9. WELLS
# ----------------------------------------------------------------------------
def build_wells(grid: dict, facies, So, table, cfg: ReservoirConfig) -> List[dict]:
    wells = []
    for w in cfg.wells:
        i = int(np.clip(w["x"] // grid["dx"], 0, grid["Nx"] - 1))
        j = int(np.clip(w["y"] // grid["dy"], 0, grid["Ny"] - 1))
        col_f = facies[i, j, :]; col_so = So[i, j, :]
        net = np.array([table[int(c)].net for c in col_f])
        if w["perforate"] == "reservoir":
            perf = np.where(net)[0]
        elif w["perforate"] == "oil_reservoir":
            perf = np.where(net & (col_so > 0.05))[0]
        elif w["perforate"] == "all":
            perf = np.arange(grid["Nz"])
        else:                                # explicit list of layers
            perf = np.array(w["perforate"], dtype=int)
        wells.append(dict(
            name=w["name"], type=w["type"], x=w["x"], y=w["y"], i=i, j=j,
            cell_x=float(grid["x"][i]), cell_y=float(grid["y"][j]),
            perforated_layers=[int(k) for k in perf],
            perforation_depths=[float(grid["depth"][i, j, k]) for k in perf],
            perforated_facies=[FACIES_NAMES[int(col_f[k])] for k in perf],
            control=w["control"], value=w["value"], units=w["units"],
            rw=w["rw"], skin=w["skin"], geometry="vertical", perforation_rule=w["perforate"],
        ))
    return wells


# ----------------------------------------------------------------------------
# 10. ASSEMBLY
# ----------------------------------------------------------------------------
def build_reservoir(cfg: Optional[ReservoirConfig] = None) -> dict:
    cfg = cfg or ReservoirConfig()
    rng = np.random.default_rng(cfg.seed)
    table = default_facies_table()

    grid = build_grid(cfg)
    fac = build_facies(grid, cfg, rng)
    phi = build_porosity(fac["facies"], table, grid, cfg, rng)
    perm = build_permeability(fac["facies"], phi, table, grid, cfg, rng)
    faults = build_faults(grid, cfg)
    frac = build_fractures(grid, cfg, perm["kx"], perm["ky"], perm["kz"])
    state = build_initial_state(grid, fac["facies"], table, cfg)
    wells = build_wells(grid, fac["facies"], state["So"], table, cfg)

    return {
        "config": cfg,
        "facies_table": table,
        "grid": grid,
        "facies": fac["facies"],
        "channel_mask": fac["channel_mask"],
        "phi": phi,
        "kx_matrix": perm["kx"], "ky_matrix": perm["ky"], "kz_matrix": perm["kz"],   # before fracture enhancement
        "kx": frac["kx"], "ky": frac["ky"], "kz": frac["kz"],                          # effective (matrix + corridors)
        "faults": faults,
        "fractures": dict(mask=frac["mask"], definitions=frac["definitions"]),
        "initial_state": dict(pressure=state["pressure"], Sw=state["Sw"], So=state["So"], Sg=state["Sg"]),
        "swc": state["swc"],
        "wells": wells,
        "units": dict(length="m", pressure="Pa", permeability="mD", density="kg/m3", md_to_m2=MD_TO_M2),
    }


# ----------------------------------------------------------------------------
# 11. VALIDATION
# ----------------------------------------------------------------------------
def validate_reservoir(res: dict, raise_on_fail=True) -> List[str]:
    cfg, table, g = res["config"], res["facies_table"], res["grid"]
    fac, phi = res["facies"], res["phi"]
    st = res["initial_state"]
    problems = []

    def chk(cond, msg):
        if not cond:
            problems.append(msg)

    for code, fp in table.items():
        m = fac == code
        if m.any():
            chk(phi[m].min() >= fp.phi_min - 1e-12 and phi[m].max() <= fp.phi_max + 1e-12, f"phi out of range in {fp.name}")
            km = res["kx_matrix"][m]
            chk(km.min() >= fp.k_min - 1e-12 and km.max() <= fp.k_max + 1e-12, f"kx_matrix out of range in {fp.name}")
    for k in ("kx", "ky", "kz", "kx_matrix", "ky_matrix", "kz_matrix"):
        chk(np.all(np.isfinite(res[k])) and np.all(res[k] > 0), f"{k} must be finite and > 0")
    chk(np.all(res["kz"] <= res["kx"] * 1.0001 + 1e-12) or True, "")  # kz<=kx expected in matrix; corridors may break it
    chk(np.all(res["kz_matrix"] <= res["kx_matrix"] + 1e-12), "kz_matrix > kx_matrix somewhere")

    S = st["Sw"] + st["So"] + st["Sg"]
    chk(np.allclose(S, 1.0, atol=1e-10), "Sw+So+Sg != 1")
    for k in ("Sw", "So", "Sg"):
        chk(np.all((st[k] >= 0) & (st[k] <= 1)), f"{k} outside [0,1]")
    D = g["depth"]
    chk(np.all(st["Sw"][D >= cfg.owc_depth] == 1.0), "oil found below OWC")
    net = np.isin(fac, [c for c, fp in table.items() if fp.net])
    chk(np.all(st["Sw"][~net] == 1.0), "non-net (shale) cells must have Sw = 1")

    # pressure monotonic with depth in every column
    dP = np.diff(st["pressure"], axis=2)
    chk(np.all(dP > 0), "pressure does not increase downward in every column")
    chk(np.all(np.diff(D, axis=2) > 0), "depth not increasing with k")
    # gradient check against expected water gradient in the water leg
    wl = (D[:, :, :-1] >= cfg.owc_depth)
    if wl.any():
        grad = dP[wl] / np.diff(D, axis=2)[wl]
        chk(np.allclose(grad, cfg.rho_w * G, rtol=1e-6), "water-leg gradient != rho_w g")

    for w in res["wells"]:
        chk(0 <= w["i"] < g["Nx"] and 0 <= w["j"] < g["Ny"], f"{w['name']} outside grid")
        chk(0 <= w["x"] <= g["Lx"] and 0 <= w["y"] <= g["Ly"], f"{w['name']} physical location outside reservoir")
        chk(len(w["perforated_layers"]) > 0, f"{w['name']} has no perforations")
        for k, fn in zip(w["perforated_layers"], w["perforated_facies"]):
            chk(0 <= k < g["Nz"], f"{w['name']} perforation layer {k} out of range")
            if w["perforation_rule"] in ("reservoir", "oil_reservoir"):
                chk(table[FACIES_CODES[fn]].net, f"{w['name']} perforates non-net facies at k={k}")
            if w["perforation_rule"] == "oil_reservoir":
                chk(st["So"][w["i"], w["j"], k] > 0.05, f"{w['name']} perforates water-bearing layer k={k}")

    fl = res["faults"]
    for fd in fl["definitions"]:
        chk(fd["n_faces"] > 0, f"fault {fd['name']} crosses no cell faces (invalid geometry)")
        chk(0 < fd["trans_mult"], f"fault {fd['name']} multiplier must be > 0")
    for arr in (fl["trans_mult_x"], fl["trans_mult_y"], fl["trans_mult_z"]):
        chk(np.all((arr > 0) & (arr <= 1.0)), "fault multipliers must be in (0,1] (increase later if desired)")
    chk(fl["trans_mult_x"].shape == (g["Nx"] - 1, g["Ny"], g["Nz"]), "trans_mult_x wrong shape")
    for fr in res["fractures"]["definitions"]:
        chk(fr["n_cells"] > 0, f"fracture {fr['name']} contains no cells")
        chk(all(0 <= k < g["Nz"] for k in fr["layers"]), f"fracture {fr['name']} layer out of range")
    chk(np.all(res["fractures"]["mask"][res["kx"] > res["kx_matrix"]] > 0), "perm enhanced outside fracture mask")

    if problems and raise_on_fail:
        raise AssertionError("Validation failed:\n  - " + "\n  - ".join(problems))
    return problems


# ----------------------------------------------------------------------------
# 12. OUTPUT
# ----------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def save_reservoir(res: dict, outdir: str, name="synthetic_reservoir") -> None:
    os.makedirs(outdir, exist_ok=True)
    g, fl = res["grid"], res["faults"]
    np.savez_compressed(
        os.path.join(outdir, f"{name}.npz"),
        x=g["x"], y=g["y"], depth=g["depth"], top_depth_map=g["top_depth_map"], cell_volume=g["cell_volume"],
        facies=res["facies"], channel_mask=res["channel_mask"], phi=res["phi"],
        kx=res["kx"], ky=res["ky"], kz=res["kz"],
        kx_matrix=res["kx_matrix"], ky_matrix=res["ky_matrix"], kz_matrix=res["kz_matrix"],
        fault_cell_mask=fl["cell_mask"], fault_face_mask_x=fl["face_mask_x"], fault_face_mask_y=fl["face_mask_y"],
        fault_face_mask_z=fl["face_mask_z"], fault_trans_mult_x=fl["trans_mult_x"],
        fault_trans_mult_y=fl["trans_mult_y"], fault_trans_mult_z=fl["trans_mult_z"],
        fracture_mask=res["fractures"]["mask"], swc=res["swc"],
        pressure=res["initial_state"]["pressure"], Sw=res["initial_state"]["Sw"],
        So=res["initial_state"]["So"], Sg=res["initial_state"]["Sg"],
    )
    cfg = res["config"]
    meta = dict(
        disclaimer="SYNTHETIC model. All ranges are illustrative; not calibrated to any real field.",
        grid=dict(Nx=g["Nx"], Ny=g["Ny"], Nz=g["Nz"], Lx=g["Lx"], Ly=g["Ly"], Lz=g["Lz"],
                  dx=g["dx"], dy=g["dy"], dz=g["dz"], depth_convention=g["depth_convention"],
                  array_shape="(Nx,Ny,Nz) C-order, indexing='ij'"),
        structure=dict(top_depth=cfg.top_depth, dip_x=cfg.dip_x, dip_y=cfg.dip_y,
                       dome_center=cfg.dome_center, dome_amplitude=cfg.dome_amplitude, dome_radius=cfg.dome_radius),
        seed=cfg.seed,
        facies_codes=FACIES_CODES,
        facies_definitions={fp.name: asdict(fp) for fp in res["facies_table"].values()},
        layer_sequence=[FACIES_NAMES[c] for c in cfg.layer_sequence],
        channels=cfg.channels,
        heterogeneity=dict(corr_len_xy=cfg.corr_len_xy, corr_len_z=cfg.corr_len_z, spectrum=cfg.spectrum,
                           fractal_beta=cfg.fractal_beta, ky_kx_ratio=cfg.ky_kx_ratio),
        property_ranges=dict(phi=[float(res["phi"].min()), float(res["phi"].max())],
                             kx=[float(res["kx"].min()), float(res["kx"].max())],
                             kz=[float(res["kz"].min()), float(res["kz"].max())]),
        faults=fl["definitions"], fault_note=fl["note"],
        fractures=res["fractures"]["definitions"],
        fracture_note="Equivalent-continuum corridors (matrix perm multipliers). Not a DFM.",
        fluids=dict(owc_depth=cfg.owc_depth, transition_zone=cfg.transition_zone, p_ref=cfg.p_ref,
                    depth_ref=cfg.depth_ref, rho_w=cfg.rho_w, rho_o=cfg.rho_o,
                    pressure_convention="P = P_ref + rho g (D - D_ref); D positive downward"),
        wells=res["wells"],
        units=res["units"],
    )
    with open(os.path.join(outdir, f"{name}.json"), "w") as f:
        json.dump(_jsonable(meta), f, indent=2)


def load_reservoir(outdir: str, name="synthetic_reservoir"):
    arrays = dict(np.load(os.path.join(outdir, f"{name}.npz")))
    with open(os.path.join(outdir, f"{name}.json")) as f:
        meta = json.load(f)
    return arrays, meta


# ----------------------------------------------------------------------------
# 13. VISUALISATION
# ----------------------------------------------------------------------------
def _mpl():
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    return matplotlib, plt


FACIES_COLORS = {0: "#6b6b6b", 1: "#c9b37e", 2: "#f2d16b", 3: "#e07b39"}


def _draw_wells(ax, res, zkey="depth"):
    g = res["grid"]
    for w in res["wells"]:
        col = "blue" if w["type"] == "injector" else "red"
        ztop, zbot = g["depth"][w["i"], w["j"], 0] - g["dz"], g["depth"][w["i"], w["j"], -1] + g["dz"]
        ax.plot([w["cell_x"]] * 2, [w["cell_y"]] * 2, [ztop, zbot], color=col, lw=2.5)
        ax.scatter([w["cell_x"]] * len(w["perforation_depths"]), [w["cell_y"]] * len(w["perforation_depths"]),
                   w["perforation_depths"], color=col, s=35, marker="s", edgecolor="k", zorder=10)
        ax.text(w["cell_x"], w["cell_y"], ztop - 3, w["name"], color=col, fontsize=9, weight="bold")


def _style3d(ax, res, title):
    g = res["grid"]
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("depth [m]")
    ax.set_xlim(0, g["Lx"]); ax.set_ylim(0, g["Ly"])
    ax.set_zlim(np.nanmax(g["depth"]) + 5, np.nanmin(g["depth"]) - 5)   # depth increases downward
    ax.set_box_aspect((1, 1, 0.35)); ax.set_title(title)


def plot_facies_3d(res, fname=None):
    _, plt = _mpl()
    g, fac = res["grid"], res["facies"]
    fig = plt.figure(figsize=(10, 7)); ax = fig.add_subplot(111, projection="3d")
    X, Y, D = g["X"], g["Y"], g["depth"]
    # plot outer faces + every layer as point cloud, skipping shale to keep channels visible
    for code, colr in FACIES_COLORS.items():
        m = (fac == code) & (fac != 0)
        if m.any():
            ax.scatter(X[m], Y[m], D[m], c=colr, s=6, alpha=0.5, label=FACIES_NAMES[code], depthshade=False)
    _draw_wells(ax, res); _style3d(ax, res, "3D facies (shale hidden for clarity)")
    ax.legend(loc="upper left", markerscale=3)
    if fname: fig.savefig(fname, dpi=140, bbox_inches="tight")
    return fig


def plot_property_slice(res, prop="phi", k=2, j=None, log=False, fname=None):
    _, plt = _mpl()
    g = res["grid"]; arr = res[prop]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    data = np.log10(arr[:, :, k]) if log else arr[:, :, k]
    im = axes[0].pcolormesh(g["x"], g["y"], data.T, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=axes[0], label=("log10 " if log else "") + prop)
    axes[0].contour(g["x"], g["y"], res["faults"]["cell_mask"][:, :, k].T, levels=[0.5], colors="k", linewidths=2)
    axes[0].contour(g["x"], g["y"], res["fractures"]["mask"][:, :, k].T, levels=[0.5], colors="magenta", linewidths=1.5)
    for w in res["wells"]:
        axes[0].plot(w["cell_x"], w["cell_y"], "o", ms=10, mfc="blue" if w["type"] == "injector" else "red", mec="k")
        axes[0].annotate(w["name"], (w["cell_x"], w["cell_y"]), xytext=(6, 6), textcoords="offset points")
    axes[0].set_title(f"{prop} map, layer k={k} (black=fault cells, magenta=fracture corridors)")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]"); axes[0].set_aspect("equal")
    j = res["wells"][1]["j"] if j is None else j
    data = np.log10(arr[:, j, :]) if log else arr[:, j, :]
    Xs, Ds = g["X"][:, j, :], g["depth"][:, j, :]
    im = axes[1].pcolormesh(Xs, Ds, data, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=axes[1], label=("log10 " if log else "") + prop)
    axes[1].contour(Xs, Ds, res["faults"]["cell_mask"][:, j, :], levels=[0.5], colors="k", linewidths=2)
    axes[1].axhline(res["config"].owc_depth, color="cyan", ls="--", label="OWC")
    axes[1].invert_yaxis(); axes[1].set_title(f"{prop} x-z section, j={j}"); axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("depth [m]")
    axes[1].legend()
    if fname: fig.savefig(fname, dpi=140, bbox_inches="tight")
    return fig


def plot_faults_3d(res, fname=None):
    _, plt = _mpl()
    fig = plt.figure(figsize=(10, 7)); ax = fig.add_subplot(111, projection="3d")
    g = res["grid"]
    cm = res["faults"]["cell_mask"]
    for fd in res["faults"]["definitions"]:
        Xs, Ys, Zs = fault_surface_points(fd, g)
        ax.plot_surface(Xs, Ys, Zs, color="k", alpha=0.35, linewidth=0)
        m = cm == fd["id"]
        ax.scatter(g["X"][m], g["Y"][m], g["depth"][m], c="crimson", s=8, label=f"{fd['name']} cells (mult={fd['trans_mult']})")
    # translucent reservoir top surface for context
    ax.plot_surface(g["X"][:, :, 0], g["Y"][:, :, 0], g["top_depth_map"], cmap="terrain", alpha=0.35)
    _draw_wells(ax, res); _style3d(ax, res, "Fault surface(s) and flagged cells (top surface shown for structure)")
    ax.legend(loc="upper left")
    if fname: fig.savefig(fname, dpi=140, bbox_inches="tight")
    return fig


def plot_fractures_3d(res, fname=None):
    _, plt = _mpl()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(10, 7)); ax = fig.add_subplot(111, projection="3d")
    g = res["grid"]; mk = res["fractures"]["mask"]
    cmap = plt.get_cmap("cool")
    for fr in res["fractures"]["definitions"]:
        m = mk == fr["id"]
        ztop, zbot = g["depth"][m].min() - g["dz"] / 2, g["depth"][m].max() + g["dz"] / 2
        th = np.deg2rad(fr["strike_deg"]); L = fr["length"] / 2
        p1 = (fr["xc"] - L * np.cos(th), fr["yc"] - L * np.sin(th)); p2 = (fr["xc"] + L * np.cos(th), fr["yc"] + L * np.sin(th))
        verts = [[(p1[0], p1[1], ztop), (p2[0], p2[1], ztop), (p2[0], p2[1], zbot), (p1[0], p1[1], zbot)]]
        c = cmap(fr["id"] / max(1, len(res["fractures"]["definitions"])))
        ax.add_collection3d(Poly3DCollection(verts, facecolor=c, alpha=0.5, edgecolor="k"))
        ax.scatter(g["X"][m], g["Y"][m], g["depth"][m], color=c, s=10, label=f"{fr['name']} (x{fr['k_mult_along']})")
    _draw_wells(ax, res); _style3d(ax, res, "Fracture corridors (equivalent continuum cells + nominal planes)")
    ax.legend(loc="upper left")
    if fname: fig.savefig(fname, dpi=140, bbox_inches="tight")
    return fig


def plot_wells_3d(res, fname=None):
    _, plt = _mpl()
    fig = plt.figure(figsize=(10, 7)); ax = fig.add_subplot(111, projection="3d")
    g = res["grid"]
    net = res["facies"] > 0
    oil = res["initial_state"]["So"] > 0.05
    ax.scatter(g["X"][net & oil], g["Y"][net & oil], g["depth"][net & oil], c="palegreen", s=4, alpha=0.25, label="oil-bearing net cells")
    ax.plot_surface(g["X"][:, :, 0], g["Y"][:, :, 0], np.full_like(g["top_depth_map"], res["config"].owc_depth), color="cyan", alpha=0.2)
    _draw_wells(ax, res); _style3d(ax, res, "Wells (blue=injector, red=producer, squares=perforations), cyan plane=OWC")
    ax.legend(loc="upper left")
    if fname: fig.savefig(fname, dpi=140, bbox_inches="tight")
    return fig


def plot_pressure_depth(res, fname=None):
    _, plt = _mpl()
    g, P = res["grid"], res["initial_state"]["pressure"]
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.scatter(P.ravel() / 1e6, g["depth"].ravel(), s=3, c=res["initial_state"]["Sw"].ravel(), cmap="coolwarm_r")
    ax.axhline(res["config"].owc_depth, color="k", ls="--", label="OWC")
    ax.invert_yaxis(); ax.set_xlabel("pressure [MPa]"); ax.set_ylabel("depth [m]")
    ax.set_title("Initial hydrostatic pressure (colour: Sw)\noil gradient above OWC, water gradient below")
    ax.legend(); ax.grid(alpha=0.3)
    if fname: fig.savefig(fname, dpi=140, bbox_inches="tight")
    return fig


def plot_saturation(res, j=None, fname=None):
    _, plt = _mpl()
    g = res["grid"]; Sw = res["initial_state"]["Sw"]
    j = res["wells"][1]["j"] if j is None else j
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    Xs, Ds = g["X"][:, j, :], g["depth"][:, j, :]
    im = axes[0].pcolormesh(Xs, Ds, Sw[:, j, :], shading="auto", cmap="coolwarm_r", vmin=0, vmax=1)
    fig.colorbar(im, ax=axes[0], label="Sw"); axes[0].axhline(res["config"].owc_depth, color="k", ls="--", label="OWC")
    axes[0].invert_yaxis(); axes[0].set_title(f"Initial Sw, x-z section j={j}"); axes[0].legend()
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("depth [m]")
    So_col = res["initial_state"]["So"].sum(axis=2) * g["dz"]
    im = axes[1].pcolormesh(g["x"], g["y"], So_col.T, shading="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=axes[1], label="sum(So*dz) [m]")
    axes[1].contour(g["x"], g["y"], res["faults"]["cell_mask"].max(axis=2).T, levels=[0.5], colors="k")
    for w in res["wells"]:
        axes[1].plot(w["cell_x"], w["cell_y"], "o", ms=10, mfc="blue" if w["type"] == "injector" else "red", mec="k")
    axes[1].set_title("Oil column thickness (structure + OWC control)"); axes[1].set_aspect("equal")
    if fname: fig.savefig(fname, dpi=140, bbox_inches="tight")
    return fig


def make_all_figures(res, outdir):
    os.makedirs(outdir, exist_ok=True)
    _, plt = _mpl()
    plot_facies_3d(res, os.path.join(outdir, "01_facies_3d.png"))
    plot_property_slice(res, "phi", k=2, fname=os.path.join(outdir, "02_porosity_slice.png"))
    plot_property_slice(res, "kx", k=2, log=True, fname=os.path.join(outdir, "03_permeability_slice.png"))
    plot_faults_3d(res, os.path.join(outdir, "04_faults_3d.png"))
    plot_fractures_3d(res, os.path.join(outdir, "05_fractures_3d.png"))
    plot_wells_3d(res, os.path.join(outdir, "06_wells_3d.png"))
    plot_pressure_depth(res, os.path.join(outdir, "07_pressure_depth.png"))
    plot_saturation(res, fname=os.path.join(outdir, "08_saturation_owc.png"))
    plt.close("all")


# ----------------------------------------------------------------------------
# 14. SUMMARY / MAIN
# ----------------------------------------------------------------------------
def summarize(res):
    g, fac = res["grid"], res["facies"]
    print(f"Grid {g['Nx']}x{g['Ny']}x{g['Nz']}  cell {g['dx']:.1f}x{g['dy']:.1f}x{g['dz']:.1f} m  "
          f"depth {g['depth'].min():.1f}-{g['depth'].max():.1f} m")
    for code, name in FACIES_NAMES.items():
        m = fac == code
        if m.any():
            print(f"  {name:18s} {100*m.mean():5.1f}%  phi {res['phi'][m].mean():.3f}  "
                  f"kx(geomean) {10**np.log10(res['kx'][m]).mean():8.2f} mD  kz/kx {np.median(res['kz'][m]/res['kx'][m]):.3f}")
    print(f"Faults: {[ (f['name'], f['n_cells'], f['n_faces']) for f in res['faults']['definitions']]}")
    print(f"Fracture corridors: {[ (f['name'], f['n_cells']) for f in res['fractures']['definitions']]}")
    P = res["initial_state"]["pressure"]
    print(f"Pressure {P.min()/1e6:.2f}-{P.max()/1e6:.2f} MPa; OWC {res['config'].owc_depth} m; "
          f"oil-in-place (pore volume) {(res['phi']*res['initial_state']['So']*g['cell_volume']).sum():.3e} m3")
    for w in res["wells"]:
        print(f"  {w['name']:6s} {w['type']:9s} cell ({w['i']},{w['j']}) perfs k={w['perforated_layers']} "
              f"{w['perforated_facies']} control={w['control']}:{w['value']} {w['units']}")


if __name__ == "__main__":
    res = build_reservoir(ReservoirConfig())
    validate_reservoir(res)
    summarize(res)
    save_reservoir(res, "output")
    make_all_figures(res, "output/figures")
    print("Saved to ./output")