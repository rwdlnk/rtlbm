# Getting started: RT lattice Boltzmann solver and fractal analysis tools

You should have two tarballs:

| File | Contents |
|---|---|
| `rtlbm-code.tgz` | `LatticeBoltzmann/`: the LBM Rayleigh–Taylor solver (2D/3D, GPU or CPU) |
| `fractal-tools.tgz` | `FractalParameterAI/` (2D analyzer, plus the 3D driver), `3D-FractalParameterAI/` (3D fractal library), `FractalAnalyzer/` (2D box-counting library) |

Everything below was tested on 2026-10-30 in an empty folder with a fresh
Python 3.12 virtual environment on Linux. That test covered an LBM run in 2D
and 3D, then the full 2D and 3D analysis (all phases), on both GPU and CPU.

## 1. Unpack (the layout matters)

Put **all four folders side by side in the same parent folder**. The analyzers
find their libraries by looking for sibling folders with these exact names, so
don't rename or nest them.

```bash
mkdir ~/rt && cd ~/rt
tar xzf rtlbm-code.tgz
tar xzf fractal-tools.tgz
ls        # 3D-FractalParameterAI  FractalAnalyzer  FractalParameterAI  LatticeBoltzmann
```

## 2. Python environment

```bash
cd ~/rt
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r LatticeBoltzmann/requirements.txt
```

The pins are the exact versions on the reference machine. The tools also
pass with current releases (numpy 2.5, matplotlib 3.11, pandas 3.1,
numba 0.68), checked on 2026-10-30 with identical results.

Check the install:
```bash
python -c "import numpy, scipy, pandas, matplotlib, skimage, numba, vtk; from numba import cuda; print('ok, GPU:', cuda.is_available())"
```

**GPU.** An NVIDIA GPU with a CUDA driver is optional.
- If you have one, the solver uses it by default, and so does the multifractal
  box counting in the analyzers.
- If you don't, add `--backend cpu` to every `run_rt.py` and
  `validate_linear.py` command. The analyzers switch to CPU on their own (they
  print "Using CPU box counting").

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
0.2 s.

```bash
cd ~/rt/LatticeBoltzmann
# 2D: ~15 s on an RTX 3060 at 160x200; ~4 min at 1280x1600
python run_rt.py --nx 160 --ny 200 --out ../cases/2D/160x200

# 3D: ~75 s at 160x200x80.  --no-velocity skips U and saves disk, but the
# 3D velocity/spectra phase then has nothing to analyse.
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

## 5. Analyse

Run the analyzers from the `~/rt` folder, with the venv active.

### 2D
```bash
python -u FractalParameterAI/rt_analyzer/analyze_rt.py cases/2D/160x200
```
Phases:
1. Mixing widths and fractal dimension D(t)
2. Multifractal spectrum at selected times
3. Interface spectra and velocity statistics

Useful flags:

| Flag | Meaning |
|---|---|
| `--skip-phase1` / `--skip-phase2` / `--skip-phase3` | skip a phase |
| `--mf-times 2,4,...,18` | multifractal times, in **seconds** |
| `--contour-levels 0.1,0.5,0.9` | contour levels to analyse |

Results go to `cases/2D/160x200/analysis/`:

| Folder | Contents |
|---|---|
| `summary/temporal_results.csv` | t, τ, mixing widths (h_10, h_geo, …), `fractal_dim` |
| `fractal/`, `interfaces/`, `profiles/` | per-snapshot box counts, contours, mean α(y) |
| `multifractal/` | D_q, f(α) |
| `spectra/`, `velocity/` | spectra and velocity statistics |

The late-time D we quote is the mean of `fractal_dim` over τ > 2:
```bash
python -c "import pandas as pd; d=pd.read_csv('cases/2D/160x200/analysis/summary/temporal_results.csv'); print(d[d.tau>2].fractal_dim.mean())"
```
For the tested 160×200 case this prints 1.648.

### 3D
```bash
python -u FractalParameterAI/rt_analyzer/analyze_rt_3d.py cases/3D/160x200x80
```
This uses the `postProcessing/extractInterface/<t>/interface.vtk`
isosurfaces, which `run_rt.py` writes, and the `alpha.water` and `U` fields.
The flags work as in 2D, plus `--skip-energy`. Expect 10–30 minutes at
160×200×80: 3D box counting is the slow part.

### Other data
The same analyzers read:
- **OpenFOAM cases** (2D or 3D): the folder needs `constant/polyMesh` (2D) or
  `system/blockMeshDict` (3D), plus time folders containing `alpha.water`.
- **SOLA-VOF output**: `RT<nx>x<ny>-<step>.vtk` files plus the `int*.in`
  input file in the case folder.

## 6. Comparing with our SOLA-VOF / OpenFOAM results
`LatticeBoltzmann/compare_codes.py` points at our results tree
(`/media/rod/RT-Simulations/Dalziel_1999`), which isn't included. If you get a
copy, edit the path `R` at the top of the script.

## 7. What to read next
`LatticeBoltzmann/README.md` covers:
- the model and why each choice was made
- the validation results and the 2D/3D cross-code comparison of D
- the numerical knobs: mobility M and interface width W, and what they do to D
- notes on porting to Julia
