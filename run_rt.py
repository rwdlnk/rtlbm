#!/usr/bin/env python3
"""Run a Dalziel (1999) Rayleigh-Taylor case with the lattice Boltzmann solver.

Defaults reproduce the OpenFOAM/SOLA-VOF configuration in
/media/rod/RT-Simulations/Dalziel_1999 (0.4 x 0.5 [x 0.2] m, A = 0.0021,
slip top/sides, no-slip bottom, sigma = 0.08 dx Dalziel perturbation).

Examples
  python run_rt.py --nx 160 --ny 200 --out cases/2D/160x200
  python run_rt.py --nx 160 --ny 200 --nz 80 --out cases/3D/160x200x80 --fields-every 5
  python run_rt.py --nx 640 --ny 800 --backend cpu --out cases/2D/640x800-cpu

Analyse with the existing pipeline, e.g.
  python .../FractalParameterAI/rt_analyzer/analyze_rt.py cases/2D/160x200
  python .../FractalParameterAI/rt_analyzer/analyze_rt_3d.py cases/3D/160x200x80
"""
import argparse

from rtlbm import RTSimulation, run


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nx', type=int, default=160)
    p.add_argument('--ny', type=int, default=200)
    p.add_argument('--nz', type=int, default=1, help='>1 for 3D')
    p.add_argument('--Lx', type=float, default=0.4)
    p.add_argument('--Ly', type=float, default=0.5)
    p.add_argument('--Lz', type=float, default=0.2)
    p.add_argument('--y0', type=float, default=0.25)
    p.add_argument('--t-end', type=float, default=20.0)
    p.add_argument('--write-interval', type=float, default=0.2)
    p.add_argument('--W', type=float, default=3.0, help='interface width [cells]')
    p.add_argument('--M', type=float, default=0.002, help='phase-field mobility [lattice units]')
    p.add_argument('--c-smag', type=float, default=0.0, help='Smagorinsky constant (0 = off)')
    p.add_argument('--u-lat', type=float, default=0.04,
                   help='lattice velocity of u_ref = sqrt(A g Ly); sets dt (Mach number)')
    p.add_argument('--init-width', type=float, default=0.005,
                   help='initial tanh width [m] (0.005 = OpenFOAM perturbDalziel.py; 0 = W*dx)')
    p.add_argument('--perturbation', choices=['dalziel', 'dalziel-x', 'single'], default='dalziel',
                   help="dalziel: OpenFOAM Dalziel_1999 recipe, eta(x) in 2D and eta(x,z) in 3D; "
                        "dalziel-x: eta(x) copied through z (rtlbm 3D runs before 2026-11-02)")
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--sigma-frac', type=float, default=0.08)
    p.add_argument('--amplitude', type=float, default=None, help='single-mode amplitude [m]')
    p.add_argument('--mode', type=int, default=1)
    p.add_argument('--bc', default='', help='override walls, e.g. "xlo=noslip,xhi=noslip"')
    p.add_argument('--backend', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--precision', choices=['float32', 'float64'], default='float32')
    p.add_argument('--out', required=True, help='case directory (OpenFOAM layout)')
    p.add_argument('--fields-every', type=int, default=1, help='write alpha/U every N outputs')
    p.add_argument('--no-velocity', action='store_true', help='do not write U')
    p.add_argument('--checkpoint-every', type=int, default=10, help='outputs between checkpoints')
    p.add_argument('--restart', default=None, help='checkpoint.npz to resume from')
    p.add_argument('--max-steps', type=int, default=None, help='stop after N steps (benchmarking)')
    a = p.parse_args()

    bc = dict(kv.split('=') for kv in a.bc.split(',') if kv)
    sim = RTSimulation(a.nx, a.ny, a.nz, Lx=a.Lx, Ly=a.Ly, Lz=a.Lz, y0=a.y0,
                       W=a.W, M=a.M, c_smag=a.c_smag, u_lat=a.u_lat,
                       t_end=a.t_end, write_interval=a.write_interval,
                       perturbation=a.perturbation, sigma_frac=a.sigma_frac, seed=a.seed,
                       amplitude=a.amplitude, mode=a.mode, init_width=a.init_width or None,
                       bc=bc, backend=a.backend, precision=a.precision)
    run(sim, a.out, fields_every=a.fields_every, velocity=not a.no_velocity,
        checkpoint_every=a.checkpoint_every, restart=a.restart, max_steps=a.max_steps)


if __name__ == '__main__':
    main()
