# Getting started: RT lattice Boltzmann solver and fractal analysis

You should have two tarballs:

| File | Contents |
|---|---|
| `rtlbm-code.tgz` | `LatticeBoltzmann/`: the LBM Rayleigh–Taylor solver (2D/3D, GPU or CPU) |
| `rtfractal.tgz` | `rtfractal/`: the fractal and mixing analyzer (2D/3D; LBM, OpenFOAM and SOLA-VOF output) |

**These replace the 2026-10-30 tarballs** (`rtlbm-code.tgz` and
`fractal-tools.tgz`). If you ran anything with those, two things changed:
- **The old analyzers overestimate D.** FractalParameterAI and
  3D-FractalParameterAI offset the box-counting grid so part of the interface
  was never counted. D came out about 0.05–0.13 too high in 2D and about 0.15
  in 3D, and the multifractal D_q were affected too. rtfractal fixes this.
  `--legacy` reproduces the old numbers exactly, for comparison.
- **The old solver had the wrong 3D initial condition.** It copied the 2D
  perturbation η(x) through z, so 3D runs stayed nearly 2D until t ≈ 10.
  The 3D seed is now η(x,z), identical to our OpenFOAM Dalziel runs. 2D is
  unchanged. Re-run any 3D case.

Everything below was tested on 2026-11-02 in an empty folder with a fresh
Python virtual environment on Linux.

## 1. Unpack

```bash
mkdir ~/rt && cd ~/rt
tar xzf rtlbm-code.tgz
tar xzf rtfractal.tgz
ls        # LatticeBoltzmann  rtfractal
```

## 2. Python environment

```bash
cd ~/rt
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r LatticeBoltzmann/requirements.txt
pip install -e rtfractal            # installs the `rtfractal` command
```

Check the install:
```bash
python -c "import numpy, scipy, pandas, skimage, numba; from numba import cuda; print('ok, GPU:', cuda.is_available())"
rtfractal -h
```

**GPU.** An NVIDIA GPU with a CUDA driver is optional.
- If you have one, the solver uses it by default, and rtfractal uses it for
  box counting.
- If you don't, add `--backend cpu` to every `run_rt.py` and
  `validate_linear.py` command. rtfractal picks the CPU on its own.
  It gives identical numbers either way.

## 3. Check the solver (about 1 minute on a GPU)

```bash
cd ~/rt/LatticeBoltzmann
python validate_linear.py            # add --backend cpu if no GPU
```
The `LBM/diffuse` column should be 0.990, 0.997, 0.999, 0.999. Compare with
`validation_linear.txt`.

## 4. Run a Rayleigh–Taylor case

Defaults are the Dalziel (1999) configuration we used with SOLA-VOF and
OpenFOAM: 0.4 × 0.5 m (× 0.2 m in 3D), A = 0.0021, t = 0–20 s, output every
0.2 s. The initial perturbation is the one our OpenFOAM runs used:
- modes with wavelengths 4–8 dx;
- σ = 0.08 dx, random phases, seed 123;
- a 5 mm tanh interface;
- η(x) in 2D, and η(x,z) in 3D with m and n each in the band.

```bash
cd ~/rt/LatticeBoltzmann
# 2D: ~15 s on an RTX 3060 at 160x200; ~4 min at 1280x1600
python run_rt.py --nx 160 --ny 200 --out ../cases/2D/160x200

# 3D: a few minutes at 160x200x80.  --no-velocity skips U and saves disk, but
# the 3D velocity/spectra analysis then has nothing to analyse.
python run_rt.py --nx 160 --ny 200 --nz 80 --out ../cases/3D/160x200x80
```

Each case folder uses the OpenFOAM layout (`0/`, `0.2/`, … with
`alpha.water` and `U`). It also gets:
- `lbm_params.json`: every lattice and physical parameter
- `lbm_diagnostics.csv`: mass error, peak velocity, mixing width and speed,
  one row per output

Useful options (`python run_rt.py -h` lists them all):

| Option | Meaning |
|---|---|
| `--M 0.002` | phase-field mobility; the main numerical sensitivity knob (see README) |
| `--W 3` | interface width in cells |
| `--fields-every 5` | write fields only every 5th output (saves disk) |
| `--restart <case>/checkpoint.npz` | resume an interrupted run |
| `--backend cpu` | run on CPU cores instead of the GPU |

**Disk.** Output is ASCII, as in OpenFOAM. A 2D 1280×1600 run takes about
7 GB, and a 3D 160×200×80 run about 4 GB.

**A known LBM limitation in 3D** (README, "Validation" §3). The phase field
damps the shortest 3D seed modes, at about 2.8 dx. 3D LBM growth therefore
starts 2–3 s later than in OpenFOAM, and the mixing zone and D lag
throughout. Compare 3D LBM with other codes at matched mixing width, not
matched time. 2D is much less affected.

## 5. Analyse

```bash
cd ~/rt
rtfractal analyze cases/2D/160x200           # 2D or 3D is detected automatically
rtfractal analyze cases/3D/160x200x80
```
Phases (all three by default):
1. Mixing widths and fractal dimension D(t)
2. Multifractal spectrum at selected times
3. Interface spectra and velocity statistics

Results go to `CASE/analysis/`, or to `--output-dir OUT`:

| Path | Contents |
|---|---|
| `summary/temporal_results.csv` | t, τ, mixing widths (h_10, h_geo, …), `fractal_dim` |
| `fractal/`, `interfaces/`, `profiles/` | per-snapshot box counts, contours, mean α(y) |
| `multifractal/` | D_q, f(α) |
| `spectra/`, `velocity/` | spectra and velocity statistics |
| `rtfractal_run.json` | settings used, rtfractal version and git commit |

Useful flags (`rtfractal analyze -h` lists them all):

| Flag | Meaning |
|---|---|
| `--phases 1` | run only some phases, e.g. `1` or `1,2` |
| `--mf-times 2,4,...,18` | multifractal times, in **seconds** (not τ). The default suits the Dalziel configuration only; set it for other setups |
| `--backend cpu` | force CPU box counting |
| `--no-plots` | data files only |
| `--legacy` | reproduce the old FractalParameterAI numbers exactly |

`rtfractal plot CASE` redraws every figure from an existing `analysis/`
folder without recomputing.

The late-time D we quote is the mean of `fractal_dim` over τ > 2:
```bash
python -c "import pandas as pd; d=pd.read_csv('cases/2D/160x200/analysis/summary/temporal_results.csv'); print(d[d.tau>2].fractal_dim.mean())"
```
For the tested 2D 160×200 case this prints 1.475. With `--legacy` it prints
1.648, the old analyzer's value.

**Multifractal caution.** On our runs the D_q from phase 2 almost never
survive the convexity check: τ(q) collapses to a straight line, so D_q = D_0.
Before quoting any D_q spread, check that the projected D_q differ from D_0,
and that the raw `Dq_raw` values are concave.

### Other data
rtfractal also reads:
- **OpenFOAM cases**, 2D or 3D. The folder needs `constant/polyMesh` (2D) or
  `system/blockMeshDict` (3D), plus time folders containing `alpha.water`.
  3D also needs `postProcessing/extractInterface/<t>/interface.vtk`.
- **SOLA-VOF output**: `RT<nx>x<ny>-<step>.vtk` files plus the `int*.in`
  input file in the case folder.

## 6. View in ParaView

The solver's `constant/polyMesh` is a stub, so ParaView can't open a case
directly. Convert it first:
```bash
cd ~/rt/LatticeBoltzmann
python -m rtlbm.to_paraview ../cases/3D/160x200x80 --every 5
paraview ../cases/3D/160x200x80/paraview/interface.pvd ../cases/3D/160x200x80/paraview/fields.pvd
```
This writes `CASE/paraview/` with `fields.pvd` (alpha.water and U, all
dimensions) and `interface.pvd` (3D interface surfaces). Click Apply. For
`fields.pvd`, use a Slice in 3D, or set the representation to Surface in 2D.

## 7. Comparing with our SOLA-VOF / OpenFOAM results
`LatticeBoltzmann/compare_codes.py` points at our results tree
(`/media/rod/RT-Simulations/Dalziel_1999`), which isn't included. If you get a
copy, edit the path `R` at the top of the script.

## 8. What to read next
- `LatticeBoltzmann/README.md`: the model, validation, the 2D/3D cross-code
  comparison, and the numerical knobs M and W.
- `rtfractal/README.md`: the analyzer, the box-counting fixes ("Known
  issues"), and the multifractal and spectrum conventions.
