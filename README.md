# rtlbm — lattice Boltzmann Rayleigh–Taylor solver

A 2D/3D lattice Boltzmann (LBM) code for the Dalziel (1999) RT problem, built
as a fourth, independent interface method alongside SOLA-VOF, OpenFOAM MULES
and OpenFOAM MPLIC in `/media/rod/RT-Simulations/Dalziel_1999`. The defaults
reproduce that configuration. Output is written in the OpenFOAM case layout,
so the existing `FractalParameterAI` analyzers (`analyze_rt.py`,
`analyze_rt_3d.py`) read it unchanged.

## Model

| Part | Choice | Why |
|---|---|---|
| Flow | D2Q9 / D3Q19, weakly compressible, **recursive-regularized** collision (Hermite moments to 3rd order), Guo forcing | stable at τ − ½ ≈ 10⁻³ (ν = 10⁻⁶ m²/s) with no turbulence model |
| Buoyancy | Boussinesq, F = −g A (2φ − 1) ρ ŷ | A = 0.0021, so variable-inertia effects are O(10⁻³) |
| Interface | conservative Allen–Cahn phase field, D2Q5 / D3Q7, **no surface tension** | mass-conserving and handles topology changes; the analogue of an immiscible, zero-surface-tension VOF |
| Viscosity | ν(φ) blended between 1.044e-6 (heavy) and 1.005e-6 (light) | same values as the OpenFOAM `physicalProperties.*` |
| Walls | bottom no-slip (half-way bounce-back), top/sides slip (specular) | same as the OpenFOAM `0/U` |
| Initial condition | `perturbDalziel.py` recipe: modes 4–8 dx, σ = 0.08 dx, uniform amplitudes, random phases (seed 123), 5 mm tanh, η(x) only (also in 3D) | identical statistics to the OpenFOAM runs |
| Hydrostatics | density initialised in hydrostatic balance | avoids a start-up acoustic pulse |
| LES | optional Smagorinsky (`--c-smag`), **off by default** | an SGS model changes small-scale structure, and so D |

φ = 1 is the heavy fluid. It is written as `alpha.water`.

### Units and time step
dx = Lx/nx (cells must be cubic). dt is set so that u_ref = √(A g Ly) ≈ 0.1 m/s
maps to the lattice velocity `--u-lat` (default 0.04, giving a peak lattice
Mach number ≈ 0.1). dt is then rounded so the 0.2 s output interval is a whole
number of steps. Because dt ∝ dx, the phase-field mobility in physical units,
M·dx²/dt, shrinks ∝ dx under refinement.

## Parallelism
The same per-node Python source is compiled two ways by numba:

* `--backend cuda` (default): one GPU thread per lattice node, run in float32.
  RTX 3060: about 1450 MLUPS in 2D (D2Q9 + D2Q5).
* `--backend cpu`: `prange` over all cores. About 80 MLUPS on this 48-thread
  machine at 160×200; it gets better on larger grids.

GPU memory is ≈ 2 × (Q_f + Q_h) × 4 bytes per node: 0.5 GB for 1920×2400 in
2D, and ≈ 8.5 GB for 400×500×200 in 3D, which fits the 12 GB card.

## Usage

```bash
# 2D, Dalziel configuration (writes OpenFOAM-style time dirs 0, 0.2, ... 20)
python run_rt.py --nx 640 --ny 800 --out cases/2D/M0.002/640x800

# 3D (write alpha every 5th output to save disk; interface.vtk written every output)
python run_rt.py --nx 160 --ny 200 --nz 80 --no-velocity --fields-every 5 --out cases/3D/160x200x80

# resume after an interruption (checkpoint.npz is written every 10 outputs)
python run_rt.py ... --restart cases/2D/.../checkpoint.npz

# analyse with the existing pipeline
python /media/rod/ResearchIII/githubRepos/FractalParameterAI/rt_analyzer/analyze_rt.py    cases/2D/M0.002/640x800
python /media/rod/ResearchIII/githubRepos/FractalParameterAI/rt_analyzer/analyze_rt_3d.py cases/3D/160x200x80

# side-by-side with SVOF/MULES/MPLIC
python compare_codes.py cases/2D/M0.002

# linear-growth validation
python validate_linear.py
```

Each case directory also gets `lbm_params.json` (all lattice and physical
parameters) and `lbm_diagnostics.csv` (mass error, peak velocity, lattice
Mach number, mixing widths, MLUPS for every output).

Disk use: ASCII output, like OpenFOAM's. A 2D 1280×1600 run with U is about
7.4 GB. Use `--no-velocity` and `--fields-every N` to cut this down.

## Validation

### 1. Single-mode linear growth (`validate_linear.py`, results in `validation_linear.txt`)
The reference is the exact Boussinesq viscous linear eigenvalue for the same
tanh density profile the LBM carries (width W = 3 cells):

| cells/λ | n_LBM / n_diffuse | n_LBM / n_sharp |
|---|---|---|
| 32 | 0.990 | 0.864 |
| 64 | 0.997 | 0.918 |
| 128 | 0.999 | 0.943 |
| 256 | 0.999 | 0.952 |

The LBM converges to the diffuse-interface theory at about second order. The
gap to the sharp-interface formula is physical: at ν = 10⁻⁶ the viscous
interface layer, √(ν/n) ≈ 0.8 mm, is as thin as the diffuse interface. That
gap closes slowly as W·dx → 0. The result does not depend on dt (u_lat
0.01–0.04) or on float32 vs float64, and it holds over a 100× range of ν.

### 2. Dalziel multimode, cross-code (2D, M = 0.002)
Late-time fractal dimension D (mean over τ > 2) and h₁₀ at τ = 3, from
`analyze_rt.py` phase 1. Columns are D / h₁₀.

| grid | LBM | SVOF | MULES | MPLIC |
|---|---|---|---|---|
| 160×200 | 1.648 / 0.039 | 1.732 / 0.046 | 1.705 / 0.041 | 1.790 / 0.040 |
| 320×400 | 1.729 / 0.034 | 1.756 / 0.043 | 1.735 / 0.031 | 1.806 / 0.039 |
| 640×800 | 1.751 / 0.032 | 1.784 / 0.036 | 1.729 / 0.033 | 1.781 / 0.032 |
| 1280×1600 | 1.767 / 0.045 | 1.776 / 0.038 | 1.760 / 0.022 | 1.778 / 0.023 |

LBM converges from below, as SVOF and MULES do, and is inside the cross-code
range from 320×400 upward. At 1280×1600 its D (1.767) sits between MULES and
MPLIC. h₁₀ at a single instant scatters between codes at the finest grids
(0.022–0.045). Compare whole h₁₀(τ) curves, not single points.

### 3. Dalziel multimode, 3D (160×200×80, M = 0.002)
Run with `analyze_rt_3d.py` phase 1 (interface isosurface box counting).
The run took 75 s of GPU compute at 680 MLUPS, and the phase-field mass error
was 3e-7.

| | LBM | MULES | MPLIC |
|---|---|---|---|
| D₃D (τ > 2) | 2.688 | 2.667 | 2.764 |
| D₂D + 1 (same grid) | 2.648 | 2.705 | 2.790 |

Marstrand's +1 rule holds for the LBM to within 1.5%.

## Knobs that matter (and what to study)

* **Mobility M** (`--M`, default 0.002). The conservative Allen–Cahn equation
  has an implicit curvature-smoothing term with rate ~M k². That term acts like
  artificial surface tension on the 4–8 dx seed modes. At M = 0.02 it
  suppressed the Dalziel seed modes and gave D ≈ 1.57 at 160×200.
  Lowering M fixes the growth, but it lets φ overshoot by a few percent
  ([-0.07, 1.09] at M = 0.002; within ±1% at M = 0.005). The α = 0.5 contour
  used for D is insensitive to this; volume-fraction profiles are slightly
  affected. M = 0.005 and M = 0.002 give the same D at 640×800 (1.762 vs 1.760).
  A sweep over M is a natural sensitivity check for the paper.
* **Interface width W** (`--W`, default 3 cells). This is the small-scale
  cutoff of the box-counting range: filaments thinner than ~W pinch off into
  droplets. Only refinement removes it.
* **Precision**. Float32 is accurate. The rest population is set so each
  node's φ sum is exact, which keeps the phase-field mass error below about
  1e-5 over a full run. Before this fix it was about 5e-3.

## Files

```
rtlbm/lattices.py   velocity sets
rtlbm/kernels.py    single-source CPU/GPU kernels (all the physics)
rtlbm/initial.py    Dalziel and single-mode perturbations
rtlbm/foamio.py     OpenFOAM-layout writers, 3D interface isosurface VTK
rtlbm/solver.py     setup, hydrostatic init, time loop, output, checkpoints
run_rt.py           command-line driver
validate_linear.py  linear-growth validation
compare_codes.py    D / h10 comparison with the Dalziel_1999 runs
```

## Porting to Julia
Every kernel is plain scalar per-node code with no array temporaries, so a
port to `KernelAbstractions.jl` (one source for CPU threads and `CUDA.jl`) is
close to line-by-line. Do the kernels and validation first: they should
reproduce the linear-growth table above before anyone trusts multimode runs.
