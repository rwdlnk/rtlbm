"""Initial interface perturbations.

dalziel(): identical recipe to perturbDalziel.py used for the OpenFOAM runs
(Dalziel et al. 1999): modes with wavelengths 4dx..8dx, uniform amplitudes
sigma*sqrt(2/N), random phases from np.random.seed(seed), RMS = sigma =
0.08 dx, eta depends on x only (also in 3D, as in the OpenFOAM 3D cases).
"""
import numpy as np


def dalziel_eta(x, Lx, dx, sigma_frac=0.08, seed=123):
    sigma = sigma_frac * dx
    n_min = int(np.ceil(Lx / (8 * dx)))
    n_max = int(np.floor(Lx / (4 * dx)))
    np.random.seed(seed)
    n_modes = n_max - n_min + 1
    amp = sigma * np.sqrt(2.0 / n_modes)
    phases = np.random.uniform(0, 2 * np.pi, n_modes)
    eta = np.zeros_like(x)
    for i, n in enumerate(range(n_min, n_max + 1)):
        eta += amp * np.cos(2 * np.pi * n / Lx * x + phases[i])
    info = dict(n_min=n_min, n_max=n_max, n_modes=n_modes, sigma=sigma)
    return eta, info


def single_mode_eta(x, Lx, amplitude, mode=1):
    """eta = a cos(2 pi m x / Lx); zero slope at x=0, Lx (slip walls)."""
    return amplitude * np.cos(2 * np.pi * mode * x / Lx), dict(mode=mode, amplitude=amplitude)


def phi_from_eta(eta_x, y, y0, width):
    """phi(z,y,x) = 0.5(1+tanh((y - y0 - eta)/(width/2))); heavy (phi=1) on top.

    eta_x: (NX,) ; y: (NY,) cell centres.  Returns (NY, NX)."""
    return 0.5 * (1.0 + np.tanh((y[:, None] - y0 - eta_x[None, :]) / (0.5 * width)))
