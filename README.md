# rtlbm — lattice Boltzmann Rayleigh–Taylor solver

A 2D/3D lattice Boltzmann (LBM) code for the Dalziel (1999) RT problem, built
as a fourth, independent interface method alongside SOLA-VOF, OpenFOAM MULES
and OpenFOAM MPLIC in `/media/rod/RT-Simulations/Dalziel_1999`. The defaults
reproduce that configuration. Output is written in the OpenFOAM case layout,
so rtfractal and the legacy `FractalParameterAI` analyzers
(`analyze_rt.py`, `analyze_rt_3d.py`) read it unchanged.

## Model

| Part | Choice | Why |
|---|---|---|
| Flow | D2Q9 / D3Q19, weakly compressible, **recursive-regularized** collision (Hermite moments to 3rd order), Guo forcing | stable at τ − ½ ≈ 10⁻³ (ν = 10⁻⁶ m²/s) with no turbulence model |
| Buoyancy | Boussinesq, F = −g A (2φ − 1) ρ ŷ | A = 0.0021, so variable-inertia effects are O(10⁻³) |
| Interface | conservative Allen–Cahn phase field, D2Q5 / D3Q7, **no surface tension** | mass-conserving and handles topology changes; the analogue of an immiscible, zero-surface-tension VOF |
| Viscosity | ν(φ) blended between 1.044e-6 (heavy) and 1.005e-6 (light) | same values as the OpenFOAM `physicalProperties.*` |
| Walls | bottom no-slip (half-way bounce-back), top/sides slip (specular) | same as the OpenFOAM `0/U` |
| Initial condition | `perturbDalziel.py` band: modes 4–8 dx, σ = 0.08 dx, seed 123, 5 mm tanh. 2D: the OpenFOAM η(x) exactly (uniform amplitudes, random phases). 3D (default): wall-compatible η(x,z) = Σ ±A cos(πmx/Lx) cos(πnz/Lz), with x and z wavelengths each in the band | 2D is identical to the OpenFOAM Dalziel_1999 runs: η measured at their first output correlates 1.0000 with the recipe (all but MULES 2D 160/320, an earlier script). The exact OpenFOAM 3D seed (`--perturbation dalziel-of`; separable rule, m and n each in the band, 231 modes at 160×200×80) causes spurious wall plumes in the LBM (Validation §3), so 3D uses the same spectral support with modes that meet the slip walls flat. The Atwood-study (paper 2) runs used a third rule, an annulus |λ| in the band (946 modes). `--perturbation dalziel-x` reproduces 3D runs from before 2026-11-02 (η(x) copied through z) |
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

# analyse with rtfractal (replaces FractalParameterAI)
rtfractal analyze cases/2D/M0.002/640x800
rtfractal analyze cases/3D/160x200x80

# side-by-side with SVOF/MULES/MPLIC
python compare_codes.py cases/2D/M0.002

# linear-growth validation
python validate_linear.py

# view in ParaView (writes CASE/paraview/; open fields.pvd, and interface.pvd in 3D)
python -m rtlbm.to_paraview cases/3D/80x100x40-M0.002 [--every N]
```

**ParaView.** The case's `constant/polyMesh` is a stub that holds only the
grid coordinates the analyzers read. ParaView's OpenFOAM reader, including
a `.foam` file, can't open it. `rtlbm.to_paraview` writes binary VTK instead:
- `fields_<t>.vti`: alpha.water and U as cell data, float32;
- `interface_<t>.vtp`: copies of the solver's interface isosurfaces;
- `.pvd` time series for both.

It skips files that already exist; `--force` rewrites them. Output size is
about 16 bytes per cell per time with U, so `--every N` helps on big grids.
In ParaView, show `fields.pvd` as Surface (2D) or Slice/Contour (3D).

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
`analyze_rt.py` phase 1. Columns are D / h₁₀. These D use the legacy box
counting, which is biased high by about 0.05–0.13 (see rtfractal's "Known
issues"); they compare codes on an equal footing, but they aren't absolute
values. With rtfractal's defaults, LBM 160×200 gives 1.475, not 1.648.

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
**Seed history:**
- **Before 2026-11-02:** η(x) copied through z (`dalziel-x`). The flow stayed
  nearly 2D until t ≈ 10, so those results are void (D₃D = 2.688 legacy,
  2.592 with rtfractal defaults).
- **The exact OpenFOAM seed** (`dalziel-of`, the `-xz` cases). It produces
  spurious wall plumes: see below.
- **Current default: the wall-compatible seed** (the `-wc` cases). It has
  the OpenFOAM spectral band and RMS, but uses cos·cos modes that meet the
  slip walls flat.

rtfractal defaults (covering offsets, L/8 ladder), over the same τ window as
OpenFOAM (2.03–4.06, t = 10–20):

| | LBM, wall-compatible | LBM, OpenFOAM seed | MULES | MPLIC |
|---|---|---|---|---|
| mean D₃D | 2.570 | 2.373 | 2.464 | 2.575 |
| h_total_int at t = 10 / 20 [m] | 0.016 / 0.069 | 0.020 / 0.089 | 0.051 / 0.122 | 0.056 / 0.134 |
| η RMS/σ at t = 0.2 / 1 / 2 | 0.44 / 0.26 / 0.42 | 0.45 / 0.32 / 0.73 | 1.07 / 3.2 / 15 | 1.07 / 3.2 / 15 |
| wall / corner \|η\| ÷ interior RMS, t = 2 | 0.99 / 0.89 | 4.6 / 25 | 0.97 / 1.7 | 0.99 / 1.8 |

80×100×40 with the wall-compatible seed: mean D₃D = 2.559.

**Wall plumes with the exact OpenFOAM seed.** The OpenFOAM seed meets the slip
side walls at an angle. Under the mirror boundary the interface has a crease
along each wall, sharper at the vertical corners. The Allen–Cahn curvature
term moves that crease at a rate set by M. That seeds plumes that run up and
down the side walls: by t = 2 the wall displacement is 4.6× the interior RMS
and the corner displacement 25×. By t = 6 the wall |U| is 2–5× the interior
value. The effect weakens with M (2.6× and 8.7× at M = 0.0005). It is absent
with a flat interface and absent with the wall-compatible seed. OpenFOAM's
VOF has no curvature term, so the same seed is harmless there.

**The LBM damps the 3D initial perturbation,** whichever seed is used. The
column-integrated η RMS falls to about 0.45σ by t = 0.2 and 0.3σ by t = 1.
OpenFOAM grows it to 1.07σ and 3.2σ over the same times. Growth starts
about 3 s late, so the mixing zone lags throughout: about a third of
OpenFOAM's width at t = 10. The cause is again the Allen–Cahn relaxation,
which damps a mode at a rate of about M_phys·k². The 3D band includes
diagonal modes with λ ≈ 2.8–5.7 dx. At 160×200×80, M_phys·k² ≈ 10 s⁻¹ for
the shortest modes, which is faster than their RT growth rate
√(Agk) ≈ 4 s⁻¹. In 2D the band is 4–8 dx and the damping is milder: η(x)
fell to 0.7σ by t = 0.2.

Lowering M to 0.0005 (tested with the OpenFOAM seed) reduces the damping
only partly. The seed still loses 30% by t = 0.2, and the mixing zone still
lags by about 2 s. Meanwhile φ overshoot grows from 6% to 33%
([−0.32, 1.33]), which corrupts α-weighted quantities such as h_total_int.
So M = 0.002 stays the default.

The damping is a resolution limit, not a tuning problem. The seed's
shortest modes are close to the 3-cell interface width. On a finer grid,
M_phys·k² grows as 1/dx while √(Agk) grows only as dx^-1/2, so refinement
does not cure it. Compare LBM 3D with OpenFOAM at matched mixing width, not
matched t.

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
rtlbm/to_paraview.py  ParaView export (.vti/.vtp + .pvd time series)
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
