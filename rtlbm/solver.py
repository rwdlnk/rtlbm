"""Rayleigh-Taylor lattice Boltzmann simulation driver."""
import json
import math
import os
import time

import numpy as np

from . import foamio, initial
from .kernels import build_kernels
from .lattices import lattices_for

DEFAULT_BC = dict(xlo='slip', xhi='slip', ylo='noslip', yhi='slip', zlo='slip', zhi='slip')


class RTSimulation:
    """Parameters are physical (SI) except W (cells) and M (lattice units).

    Unit mapping: dx = Lx/nx; dt chosen so that u_ref = sqrt(A g Ly) maps to
    lattice velocity u_lat (Mach control), then rounded so that the output
    interval is an integer number of steps.
    """

    def __init__(self, nx, ny, nz=1, Lx=0.4, Ly=0.5, Lz=0.2, y0=0.25,
                 rho_h=1002.2, rho_l=998.0, nu_h=1.044e-6, nu_l=1.005e-6, g=9.81,
                 W=3.0, M=0.002, c_smag=0.0, u_lat=0.04,
                 t_end=20.0, write_interval=0.2,
                 perturbation='dalziel', sigma_frac=0.08, seed=123, amplitude=None, mode=1,
                 init_width=0.005, bc=None, backend='cuda', precision='float32'):
        self.dim = 3 if nz > 1 else 2
        self.nx, self.ny, self.nz = nx, ny, nz
        self.Lx, self.Ly = Lx, Ly
        self.Lz = Lz if self.dim == 3 else Lx / nx
        self.dx = Lx / nx
        if abs(Ly / ny - self.dx) > 1e-12 * self.dx or (self.dim == 3 and abs(Lz / nz - self.dx) > 1e-12 * self.dx):
            raise ValueError('LBM needs cubic cells: Lx/nx == Ly/ny (== Lz/nz)')
        self.y0 = y0
        self.rho_h, self.rho_l, self.nu_h, self.nu_l, self.g = rho_h, rho_l, nu_h, nu_l, g
        self.A = (rho_h - rho_l) / (rho_h + rho_l)
        self.W, self.M, self.c_smag = W, M, c_smag
        self.t_end, self.write_interval = t_end, write_interval
        self.bc = dict(DEFAULT_BC, **(bc or {}))
        self.backend, self.precision = backend, precision
        self.dtype = np.float32 if precision == 'float32' else np.float64

        # ---- time step
        u_ref = math.sqrt(self.A * g * Ly)
        dt0 = u_lat * self.dx / u_ref
        self.steps_per_write = max(1, math.ceil(write_interval / dt0))
        self.dt = write_interval / self.steps_per_write
        self.n_steps = int(round(t_end / self.dt))
        self.u_conv = self.dx / self.dt          # m/s per lattice velocity unit

        # ---- lattice parameters
        self.lat_f, self.lat_h = lattices_for(self.dim)
        self.nu_lat_h = nu_h * self.dt / self.dx ** 2
        self.nu_lat_l = nu_l * self.dt / self.dx ** 2
        self.g_lat = g * self.dt ** 2 / self.dx
        self.tau_h = 0.5 + M / self.lat_h.cs2
        self.params = (self.g_lat * self.A, self.nu_lat_l, self.nu_lat_h,
                       c_smag ** 2, M, 4.0 / W, 1.0 / self.tau_h)

        # ---- initial interface
        xc = (np.arange(nx) + 0.5) * self.dx
        yc = (np.arange(ny) + 0.5) * self.dx
        if perturbation in ('dalziel', 'dalziel-of') and self.dim == 3:
            zc = (np.arange(nz) + 0.5) * self.dx
            recipe = initial.dalziel_eta_wall if perturbation == 'dalziel' else initial.dalziel_eta_xz
            eta, self.ic_info = recipe(xc, zc, Lx, self.Lz, self.dx, self.dx, sigma_frac, seed)
        elif perturbation in ('dalziel', 'dalziel-x'):
            eta, self.ic_info = initial.dalziel_eta(xc, Lx, self.dx, sigma_frac, seed)
        elif perturbation == 'single':
            eta, self.ic_info = initial.single_mode_eta(xc, Lx, amplitude, mode)
        else:
            raise ValueError(perturbation)
        width = init_width if init_width else W * self.dx
        self.init_width = width
        phi = initial.phi_from_eta(eta, yc, y0, width)
        phi0 = phi if phi.ndim == 3 else np.repeat(phi[None], nz, axis=0)
        self._init_fields(phi0)
        self.kern = build_kernels(self.lat_f, self.lat_h, self.bc, self.dtype, backend)
        self.step_count = 0

    # ------------------------------------------------------------------ setup
    def _init_fields(self, phi0):
        """Hydrostatic density, zero velocity, equilibrium populations."""
        nz, ny, nx = phi0.shape
        Fy = -self.g_lat * self.A * (2 * phi0 - 1)            # per unit lattice density
        # dp/dy = rho Fy with p = cs2 rho  ->  d ln(rho)/dy = 3 Fy, integrated
        # downward from the top cell (midpoint rule between cell centres)
        F_face = 0.5 * (Fy[:, 1:, :] + Fy[:, :-1, :])
        lnr = np.zeros_like(phi0)
        lnr[:, :-1, :] = -np.cumsum((3 * F_face)[:, ::-1, :], axis=1)[:, ::-1, :]
        rho = np.exp(lnr)
        rho /= rho.mean()                                      # mean lattice density 1
        Qf, Qh = self.lat_f.Q, self.lat_h.Q
        f = np.empty((Qf, nz, ny, nx), dtype=self.dtype)
        for q in range(Qf):
            f[q] = self.lat_f.w[q] * (rho - 1.0)
        # phase-field equilibrium including the sharpening flux
        gy, gx = np.gradient(phi0[0], axis=(0, 1)) if nz == 1 else np.gradient(phi0, axis=(1, 2))
        if nz == 1:
            gx, gy, gz = gx[None], gy[None], np.zeros_like(phi0)
        else:
            gz = np.gradient(phi0, axis=0)
        gm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-12
        s = self.M * 4.0 / self.W * phi0 * (1 - phi0) / gm
        h = np.empty((Qh, nz, ny, nx), dtype=self.dtype)
        c = self.lat_h.c
        for q in range(Qh):
            h[q] = self.lat_h.w[q] * (phi0 + (c[q, 0] * s * gx + c[q, 1] * s * gy + c[q, 2] * s * gz) / self.lat_h.cs2)
        self.phi_mass0 = float(phi0.sum())
        self._alloc(f, h, phi0)

    def _alloc(self, f, h, phi0):
        shp = phi0.shape
        z = np.zeros(shp, dtype=self.dtype)
        if self.backend == 'cuda':
            from numba import cuda
            self.f = cuda.to_device(f); self.fn = cuda.device_array_like(f)
            self.h = cuda.to_device(h); self.hn = cuda.device_array_like(h)
            self.phi = cuda.to_device(phi0.astype(self.dtype))
            self.ux, self.uy, self.uz = (cuda.to_device(z) for _ in range(3))
        else:
            self.f, self.fn = f, np.empty_like(f)
            self.h, self.hn = h, np.empty_like(h)
            self.phi = phi0.astype(self.dtype)
            self.ux, self.uy, self.uz = z.copy(), z.copy(), z.copy()

    # ------------------------------------------------------------------- step
    def step(self, n=1):
        k = self.kern
        for _ in range(n):
            # phi from post-streaming h (needed by neighbours for grad phi)
            k.compute_phi(self.h, self.phi)
            k.collide(self.f, self.h, self.phi, self.fn, self.hn, self.ux, self.uy, self.uz, self.params)
            self.f, self.fn = self.fn, self.f
            self.h, self.hn = self.hn, self.h
            self.step_count += 1
        if self.backend == 'cuda':
            from numba import cuda
            cuda.synchronize()

    @property
    def time(self):
        return self.step_count * self.dt

    def host(self, a):
        return a.copy_to_host() if self.backend == 'cuda' else np.array(a)

    # ---------------------------------------------------------- diagnostics
    def diagnostics(self):
        phi = self.host(self.phi).astype(np.float64)
        ux, uy, uz = (self.host(a) for a in (self.ux, self.uy, self.uz))
        umax = float(np.sqrt(ux.astype(np.float64) ** 2 + uy ** 2 + uz ** 2).max())
        prof = phi.mean(axis=(0, 2))
        yc = (np.arange(self.ny) + 0.5) * self.dx
        # geometric mixing width from the mean volume-fraction profile (1%-99%)
        inside = np.where((prof > 0.01) & (prof < 0.99))[0]
        h_geo = (inside[-1] - inside[0] + 1) * self.dx if inside.size else 0.0
        h_int = 2.0 * float(np.sum(np.minimum(prof, 1 - prof)) * self.dx)   # integral width
        return dict(t=self.time, step=self.step_count,
                    mass_err=float(phi.sum() / self.phi_mass0 - 1.0),
                    umax_lat=umax, umax=umax * self.u_conv,
                    phi_min=float(phi.min()), phi_max=float(phi.max()),
                    h_geo=h_geo, h_int=h_int, finite=bool(np.isfinite(phi).all()))

    def summary(self):
        return dict(dim=self.dim, nx=self.nx, ny=self.ny, nz=self.nz, dx=self.dx, dt=self.dt,
                    steps=self.n_steps, steps_per_write=self.steps_per_write,
                    A=self.A, tau_f_heavy=3 * self.nu_lat_h + 0.5, tau_f_light=3 * self.nu_lat_l + 0.5,
                    tau_h=self.tau_h, W_cells=self.W, M_lat=self.M, M_phys=self.M * self.dx ** 2 / self.dt,
                    c_smag=self.c_smag, g_lat=self.g_lat, u_conv=self.u_conv,
                    init_width=self.init_width, bc=self.bc, backend=self.backend,
                    precision=self.precision, ic=self.ic_info,
                    lattices=[self.lat_f.name, self.lat_h.name])

    # ----------------------------------------------------------------- output
    def write_case_skeleton(self, case):
        foamio.write_case_skeleton(case, self.Lx, self.Ly, self.Lz, self.nx, self.ny, self.nz,
                                   self.g, self.rho_h, self.rho_l, self.nu_h, self.nu_l)
        with open(os.path.join(case, 'lbm_params.json'), 'w') as fh:
            json.dump(self.summary(), fh, indent=2, default=str)

    def write_output(self, case, fields=True, velocity=True, interface=True):
        tn = foamio.time_name(self.time)
        phi = self.host(self.phi)
        if fields:
            foamio.write_scalar(os.path.join(case, tn, 'alpha.water'), 'alpha.water', phi)
            if velocity:
                c = self.u_conv
                foamio.write_vector(os.path.join(case, tn, 'U'), 'U',
                                    self.host(self.ux) * c, self.host(self.uy) * c, self.host(self.uz) * c)
        if interface and self.dim == 3:
            foamio.write_interface_vtk(os.path.join(case, 'postProcessing', 'extractInterface', tn,
                                                    'interface.vtk'), phi, self.dx)

    def save_checkpoint(self, path):
        np.savez(path, f=self.host(self.f), h=self.host(self.h), phi=self.host(self.phi),
                 step=self.step_count, mass0=self.phi_mass0)

    def load_checkpoint(self, path):
        d = np.load(path)
        self.step_count = int(d['step'])
        self.phi_mass0 = float(d['mass0'])
        self._alloc(d['f'].astype(self.dtype), d['h'].astype(self.dtype), d['phi'])


def run(sim, case, fields_every=1, velocity=True, interface=True, checkpoint_every=10,
        restart=None, max_steps=None, log=print):
    os.makedirs(case, exist_ok=True)
    sim.write_case_skeleton(case)
    if restart:
        sim.load_checkpoint(restart)
    log(json.dumps(sim.summary(), indent=1, default=str))
    diag_path = os.path.join(case, 'lbm_diagnostics.csv')
    new = not (restart and os.path.exists(diag_path))
    dfh = open(diag_path, 'w' if new else 'a')
    keys = ['t', 'step', 'mass_err', 'umax', 'umax_lat', 'phi_min', 'phi_max', 'h_geo', 'h_int', 'mlups']
    if new:
        dfh.write(','.join(keys) + '\n')
    n_total = sim.n_steps if max_steps is None else min(sim.n_steps, sim.step_count + max_steps)
    out_idx = sim.step_count // sim.steps_per_write
    ncell = sim.nx * sim.ny * sim.nz
    if sim.step_count == 0:
        sim.write_output(case, True, velocity, interface)
    while sim.step_count < n_total:
        nstep = min(sim.steps_per_write, n_total - sim.step_count)
        t0 = time.perf_counter()
        sim.step(nstep)
        el = time.perf_counter() - t0
        out_idx += 1
        d = sim.diagnostics()
        d['mlups'] = ncell * nstep / el / 1e6
        dfh.write(','.join(f"{d[k]:.6g}" for k in keys) + '\n'); dfh.flush()
        log(f"t={d['t']:7.3f}s step={d['step']:7d} h_geo={d['h_geo']*1e3:7.2f}mm "
            f"umax={d['umax']*1e3:6.2f}mm/s (Ma_lat {d['umax_lat']*math.sqrt(3):.3f}) "
            f"mass_err={d['mass_err']:+.1e} phi=[{d['phi_min']:.3f},{d['phi_max']:.3f}] "
            f"{d['mlups']:.0f} MLUPS")
        if not d['finite']:
            log('NaN detected -- aborting')
            break
        if d['umax_lat'] > 0.25:
            log('WARNING: lattice velocity > 0.25; reduce --u-lat')
        if sim.step_count % sim.steps_per_write == 0:
            sim.write_output(case, out_idx % fields_every == 0, velocity, interface)
            if checkpoint_every and out_idx % checkpoint_every == 0:
                sim.save_checkpoint(os.path.join(case, 'checkpoint.npz'))
    dfh.close()
    return sim
