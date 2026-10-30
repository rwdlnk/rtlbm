#!/usr/bin/env python3
"""Validation 1: single-mode linear growth rate vs linear theory.

Domain lambda x 2 lambda, one cosine mode (zero slope at the slip side walls),
Dalziel fluid properties.  The amplitude is the cos(kx) Fourier coefficient of
the interface height recovered exactly from the column integral of phi:
    eta(x) = (Ly - y0) - int phi dy,
and the growth rate is the slope of ln(a) once the mode is exponential and
still linear (k a < 0.06).

Two references are printed:
  n_sharp   sharp interface, n = sqrt(A g k + nu^2 k^4) - nu k^2
  n_diffuse exact Boussinesq viscous linear eigenvalue for the *same* tanh
            density profile as the LBM interface (width W cells), i.e.
            (n^2 L - n nu L^2) w = -k^2 g (2A phi') w,  L = d2/dy2 - k^2.
The LBM should match n_diffuse closely; n_diffuse -> n_sharp as W*dx -> 0.
(At A = 0.002 and nu = 1e-6 the viscous layer sqrt(nu/n) ~ 0.8 mm is as thin
as the interface, so n_sharp is itself only approximate.)

    python validate_linear.py                      # 32..256 cells per wavelength
    python validate_linear.py --res 64 128 --M 0.005 --nu-factor 10
"""
import argparse
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import brentq

from rtlbm import RTSimulation

LAM = 0.05


def lbm_growth_rate(res, M, W, nu_factor, backend):
    sim = RTSimulation(res, 2 * res, Lx=LAM, Ly=2 * LAM, y0=LAM, W=W, M=M,
                       nu_h=1.044e-6 * nu_factor, nu_l=1.005e-6 * nu_factor,
                       t_end=3.6, write_interval=0.05, perturbation='single',
                       amplitude=1e-4 * LAM, init_width=None, backend=backend)
    k = 2 * math.pi / LAM
    xc = (np.arange(res) + 0.5) * sim.dx
    basis = np.cos(k * xc) * 2.0 / res
    ts, amps = [], []
    for _ in range(sim.n_steps // sim.steps_per_write + 1):
        phi = sim.host(sim.phi)[0].astype(np.float64)
        ts.append(sim.time)
        amps.append(float(basis @ ((sim.Ly - sim.y0) - phi.sum(axis=0) * sim.dx)))
        sim.step(sim.steps_per_write)
    ts, amps = np.array(ts), np.array(amps)
    sel = (amps > 20 * amps[0]) & (amps < 0.01 * LAM)      # exponential, k a < 0.063
    return np.polyfit(ts[sel], np.log(amps[sel]), 1)[0], int(sel.sum()), sim


def n_sharp(A, g, k, nu):
    return math.sqrt(A * g * k + nu ** 2 * k ** 4) - nu * k ** 2


def n_diffuse(A, g, k, nu, width, N=6001):
    y = np.linspace(-LAM, LAM, N)[1:-1]
    h = y[1] - y[0]
    m = len(y)
    D2 = sp.diags([np.ones(m - 1), -2 * np.ones(m), np.ones(m - 1)], [-1, 0, 1]) / h ** 2
    L = (D2 - k * k * sp.eye(m)).tocsc()
    dphi = (1 / width) / np.cosh(np.clip(2 * y / width, -300, 300)) ** 2
    B = sp.diags(k * k * g * 2 * A * dphi).tocsc()

    def mu(n):
        Aop = (n * n * L - n * nu * (L @ L)).tocsc()
        return sla.eigs(Aop, k=1, M=-B, sigma=1.0, which='LM', return_eigenvectors=False)[0].real - 1.0

    n0 = math.sqrt(A * g * k)
    return brentq(mu, 0.5 * n0, 1.05 * n0, xtol=1e-6)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--res', type=int, nargs='+', default=[32, 64, 128, 256], help='cells per wavelength')
    p.add_argument('--M', type=float, nargs='+', default=[0.002])
    p.add_argument('--W', type=float, default=3.0)
    p.add_argument('--nu-factor', type=float, default=1.0, help='scale both viscosities')
    p.add_argument('--backend', default='cuda')
    a = p.parse_args()
    print(f"{'cells/lam':>9s} {'M':>7s} {'n_sharp':>8s} {'n_diffuse':>9s} {'n_LBM':>8s} "
          f"{'LBM/diffuse':>11s} {'LBM/sharp':>9s}")
    for M in a.M:
        for r in a.res:
            n, npts, sim = lbm_growth_rate(r, M, a.W, a.nu_factor, a.backend)
            k = 2 * math.pi / LAM
            nu = 0.5 * (sim.nu_h + sim.nu_l)
            ns = n_sharp(sim.A, sim.g, k, nu)
            nd = n_diffuse(sim.A, sim.g, k, nu, a.W * sim.dx)
            print(f"{r:9d} {M:7.4f} {ns:8.4f} {nd:9.4f} {n:8.4f} {n / nd:11.4f} {n / ns:9.4f}", flush=True)


if __name__ == '__main__':
    main()
