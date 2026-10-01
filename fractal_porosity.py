"""fractal_porosity.py -- Spectral fractional-Brownian-motion porosity fields.

Method (README section 6)
-------------------------
1. Draw complex white Gaussian noise on the (ny, nx) grid in Fourier space.
2. Multiply by an amplitude filter A(k) = |k|^(-beta/2) (A(0) = 0), with
   ``k`` from ``np.fft.fftfreq`` (cycles per cell).
3. Inverse FFT, keep the real part, normalize to zero mean and unit std.
4. phi = phi_mean + phi_std * field, clipped to [phi_min, phi_max].

The power spectrum then follows P(k) ~ k^(-beta): larger beta gives smoother,
more strongly correlated fields.

Fractal-dimension convention
----------------------------
For a 2D fBm surface (a height field embedded in 3D) with spectral exponent
beta, the Hurst exponent is H = (beta - 2)/2 and the surface fractal dimension
is D_f = 3 - H = (8 - beta)/2. Other conventions exist (e.g. D_f of a profile
= (5 - beta)/2 for a 1D trace, or defining beta from the amplitude rather than
the power spectrum); all plots in this project use D_f = (8 - beta)/2.

Reproducibility and caveats
---------------------------
* All randomness is ``np.random.default_rng(seed)``. The same seed with a
  different beta reuses the *same noise*, so a beta sweep isolates the effect
  of spectral slope from the effect of the random realization.
* The FFT is periodic, so the field wraps at the grid edges; for a
  production study one would generate a larger field and crop.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from config import FractalConfig


def fractal_dimension(beta: float) -> float:
    """Return the 2D-surface fractal dimension D_f = (8 - beta) / 2."""
    return (8.0 - float(beta)) / 2.0


def beta_from_fractal_dimension(d_f: float) -> float:
    """Inverse of :func:`fractal_dimension`: beta = 8 - 2*D_f."""
    return 8.0 - 2.0 * float(d_f)


def generate_fbm_field(
    nx: int,
    ny: int,
    beta: float,
    seed: int,
    nz: int = 1,
) -> np.ndarray:
    """Return an (ny, nx) or (nz, ny, nx) fBm field.

    Row index is j (y), column index is i (x), so field.ravel()
    follows cell_index(i, j, k) = k*ny*nx + j*nx + i.
    """

    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if nz < 1:
        raise ValueError("nz must be >= 1")

    rng = np.random.default_rng(seed)

    shape = (ny, nx) if nz == 1 else (nz, ny, nx)

    # Complex Gaussian white noise
    noise = (
        rng.standard_normal(shape)
        + 1j * rng.standard_normal(shape)
    )

    # --------------------------------------------------------
    # Fourier frequencies
    # --------------------------------------------------------

    if nz == 1:

        kx = np.fft.fftfreq(nx)[None, :]
        ky = np.fft.fftfreq(ny)[:, None]

        k = np.hypot(kx, ky)

    else:

        kz = np.fft.fftfreq(nz)[:, None, None]
        ky = np.fft.fftfreq(ny)[None, :, None]
        kx = np.fft.fftfreq(nx)[None, None, :]

        k = np.sqrt(
            kx**2 +
            ky**2 +
            kz**2
        )

    # --------------------------------------------------------
    # Spectral amplitude filter
    # --------------------------------------------------------

    amplitude = np.zeros_like(k)

    nonzero = k > 0.0

    amplitude[nonzero] = (
        k[nonzero] ** (-beta / 2.0)
    )

    # --------------------------------------------------------
    # Inverse FFT
    # --------------------------------------------------------

    if nz == 1:
        field = np.real(
            np.fft.ifft2(noise * amplitude)
        )
    else:
        field = np.real(
            np.fft.ifftn(noise * amplitude)
        )

    # Normalize
    field -= field.mean()

    std = field.std()

    if std <= 0.0:
        return np.zeros(shape)

    return field / std


def generate_porosity(nx: int, ny: int, cfg: FractalConfig, nz: int = 1) -> np.ndarray:
    """Return an (ny, nx) or (nz, ny, nx) porosity field."""
    field = generate_fbm_field(nx, ny, cfg.beta, cfg.seed, nz=nz)
    phi = cfg.phi_mean + cfg.phi_std * field
    return np.clip(phi, cfg.phi_min, cfg.phi_max)


def estimate_spectral_slope(field: np.ndarray, n_bins: int = 15) -> float:
    """Estimate the exponent -beta of the radially averaged power spectrum.

    Fits log10 P versus log10 k over log-spaced radial bins inside the Nyquist
    circle (k <= 0.5). Returns the fitted slope (approximately -beta).
    """
    ny, nx = field.shape
    power = np.abs(np.fft.fft2(field - field.mean())) ** 2
    k = np.hypot(np.fft.fftfreq(nx)[None, :], np.fft.fftfreq(ny)[:, None])
    mask = (k > 0.0) & (k <= 0.5)
    edges = np.logspace(np.log10(k[mask].min()), np.log10(0.5), n_bins + 1)
    log_k, log_p = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = mask & (k >= lo) & (k < hi)
        if sel.any():
            log_k.append(np.log10(k[sel].mean()))
            log_p.append(np.log10(power[sel].mean()))
    slope, _ = np.polyfit(log_k, log_p, 1)
    return float(slope)


def summarize(phi: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (mean, std, min, max) of a porosity array."""
    return float(phi.mean()), float(phi.std()), float(phi.min()), float(phi.max())


def main() -> None:
    """Generate fields for beta = 2..5, print statistics and verify spectral slopes."""
    base = FractalConfig()
    print(f"{'beta':>5}{'D_f':>7}{'mean':>9}{'std':>9}{'min':>8}{'max':>8}")
    for beta in (2.0, 3.0, 4.0, 5.0):
        cfg = FractalConfig(beta=beta, seed=base.seed)
        phi = generate_porosity(30, 30, cfg)
        m, s, lo, hi = summarize(phi)
        assert cfg.phi_min <= lo and hi <= cfg.phi_max
        assert np.array_equal(phi, generate_porosity(30, 30, cfg)), "not reproducible"
        print(f"{beta:>5.1f}{fractal_dimension(beta):>7.2f}{m:>9.4f}{s:>9.4f}{lo:>8.4f}{hi:>8.4f}")

    print("\nSpectral slope check on 128x128 fields (mean over 8 seeds):")
    for beta in (2.0, 3.0, 4.0, 5.0):
        slopes = [estimate_spectral_slope(generate_fbm_field(128, 128, beta, seed))
                  for seed in range(8)]
        mean_slope = float(np.mean(slopes))
        print(f"  beta = {beta:.1f}: fitted slope {mean_slope:+.2f} (expected {-beta:+.2f})")
        assert abs(mean_slope + beta) < 0.4, "spectral slope does not match beta"

    assert np.isclose(beta_from_fractal_dimension(fractal_dimension(3.0)), 3.0)
    print("Fractal porosity self-checks: PASSED")

        # 3D field check
    phi_3d = generate_porosity(
        10,
        8,
        FractalConfig(beta=3.0, seed=42),
        nz=3,
    )

    assert phi_3d.shape == (3, 8, 10)
    assert np.all(phi_3d >= base.phi_min)
    assert np.all(phi_3d <= base.phi_max)

    phi_3d_repeat = generate_porosity(
        10,
        8,
        FractalConfig(beta=3.0, seed=42),
        nz=3,
    )

    assert np.array_equal(
        phi_3d,
        phi_3d_repeat
    ), "3D field is not reproducible"


if __name__ == "__main__":
    main()