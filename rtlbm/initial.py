"""Initial interface perturbations.

The Dalziel recipes reproduce the OpenFOAM Dalziel_1999 runs exactly. They
were checked by measuring eta at the first written time (t = 0.2) of each
OpenFOAM case and correlating it with the recipe (correlation 1.0000):

dalziel_eta() (2D): modes n = ceil(Lx/8dx) .. floor(Lx/4dx) in x, uniform
    amplitudes sigma*sqrt(2/N), random phases from np.random.seed(seed),
    RMS = sigma = 0.08 dx.  Matches every OpenFOAM-2D run except MULES
    160x200 and 320x400 (6 March, an earlier script with unequal amplitudes).
dalziel_eta_xz() (3D): the March 3D script, which is not the copy now in the
    case folders (that one was edited on 2026-07-04 to eta(x) only).
    eta(x,z) = sum_{m,n} A cos(2 pi m x/Lx + 2 pi n z/Lz + phi_mn) with m and n
    each in their own 4dx..8dx band (m outer loop, one phase draw), only
    n > 0, A = sigma*sqrt(2/N).  Matches all OpenFOAM-3D runs checked.
    This is NOT the Atwood-study (paper 2) 3D rule, which takes all (m, n) with
    |lambda| = 1/sqrt((m/Lx)^2 + (n/Lz)^2) in the band (946 vs 231 modes at
    160x200x80).  rtlbm follows the Dalziel_1999 runs it is compared with.
dalziel_eta_wall() (3D default): same spectral support as dalziel_eta_xz but
    built from cos(pi m x/Lx) cos(pi n z/Lz) modes, which meet the slip walls
    flat.  The exact OpenFOAM seed ('dalziel-of') produces spurious wall
    plumes in the LBM; see the function docstring.
In 3D, perturbation='dalziel-of' gives the exact OpenFOAM seed and
'dalziel-x' the original rtlbm behaviour (eta(x) copied through z).
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


def dalziel_eta_xz(x, z, Lx, Lz, dx, dz, sigma_frac=0.08, seed=123):
    """3D Dalziel eta on cell centres x (NX,), z (NZ,) -> (NZ, NX)."""
    sigma = sigma_frac * dx
    ms = range(int(np.ceil(Lx / (8 * dx))), int(np.floor(Lx / (4 * dx))) + 1)
    ns = range(int(np.ceil(Lz / (8 * dz))), int(np.floor(Lz / (4 * dz))) + 1)
    pairs = [(m, n) for m in ms for n in ns]
    np.random.seed(seed)
    amp = sigma * np.sqrt(2.0 / len(pairs))
    phases = np.random.uniform(0, 2 * np.pi, len(pairs))
    X, Z = np.meshgrid(x, z)
    eta = np.zeros_like(X)
    for i, (m, n) in enumerate(pairs):
        eta += amp * np.cos(2 * np.pi * m / Lx * X + 2 * np.pi * n / Lz * Z + phases[i])
    info = dict(m_range=(ms.start, ms.stop - 1), n_range=(ns.start, ns.stop - 1),
                n_modes=len(pairs), sigma=sigma)
    return eta, info


def dalziel_eta_wall(x, z, Lx, Lz, dx, dz, sigma_frac=0.08, seed=123):
    """Wall-compatible 3D Dalziel seed (rtlbm 3D default since 2026-11-02).

    eta = sum_{m,n} s_mn A cos(pi m x/Lx) cos(pi n z/Lz), s_mn = +-1 from
    np.random.seed(seed), A = 2 sigma/sqrt(N), so RMS = sigma = 0.08 dx.
    Each mode has x and z wavelengths 2Lx/m and 2Lz/n in 4dx..8dx, the same
    spectral support as dalziel_eta_xz (the OpenFOAM runs), but every mode
    has zero normal slope at the side walls.  The OpenFOAM seed meets the slip
    walls at an angle; the mirror makes a crease there, and the Allen-Cahn
    curvature term moves it, which seeded spurious wall and corner plumes
    (wall displacement 4.6x, corners 25x the interior at t = 2).
    """
    sigma = sigma_frac * dx
    ms = range(int(np.ceil(Lx / (4 * dx))), int(np.floor(Lx / (2 * dx))) + 1)
    ns = range(int(np.ceil(Lz / (4 * dz))), int(np.floor(Lz / (2 * dz))) + 1)
    pairs = [(m, n) for m in ms for n in ns]
    np.random.seed(seed)
    signs = np.where(np.random.uniform(size=len(pairs)) < 0.5, -1.0, 1.0)
    amp = 2.0 * sigma / np.sqrt(len(pairs))
    cx = {m: np.cos(np.pi * m * x / Lx) for m in ms}
    cz = {n: np.cos(np.pi * n * z / Lz) for n in ns}
    eta = np.zeros((len(z), len(x)))
    for i, (m, n) in enumerate(pairs):
        eta += signs[i] * amp * np.outer(cz[n], cx[m])
    info = dict(kind='wall-compatible', m_range=(ms.start, ms.stop - 1), n_range=(ns.start, ns.stop - 1),
                n_modes=len(pairs), sigma=sigma)
    return eta, info


def single_mode_eta(x, Lx, amplitude, mode=1):
    """eta = a cos(2 pi m x / Lx); zero slope at x=0, Lx (slip walls)."""
    return amplitude * np.cos(2 * np.pi * mode * x / Lx), dict(mode=mode, amplitude=amplitude)


def phi_from_eta(eta, y, y0, width):
    """phi = 0.5(1+tanh((y - y0 - eta)/(width/2))); heavy (phi=1) on top.

    eta: (NX,) -> returns (NY, NX);  eta: (NZ, NX) -> returns (NZ, NY, NX)."""
    if eta.ndim == 1:
        return 0.5 * (1.0 + np.tanh((y[:, None] - y0 - eta[None, :]) / (0.5 * width)))
    return 0.5 * (1.0 + np.tanh((y[None, :, None] - y0 - eta[:, None, :]) / (0.5 * width)))
