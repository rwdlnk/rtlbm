"""Write results in the OpenFOAM case layout read by FractalParameterAI.

analyze_rt.py (2D) needs:   constant/polyMesh/points, constant/g,
                            constant/physicalProperties.{water,air},
                            <time>/alpha.water [, <time>/U]
analyze_rt_3d.py (3D) needs: system/blockMeshDict, <time>/alpha.water,
                            postProcessing/extractInterface/<time>/interface.vtk

polyMesh/points is a *stub* (only the x- and y-node lines, which is all the
analyzer parses); the case is not runnable by OpenFOAM itself.
"""
import os

import numpy as np

HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
  rtlbm (lattice Boltzmann) output in OpenFOAM layout
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{loc}";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def time_name(t):
    """OpenFOAM 'general' time format with timePrecision 6."""
    return f"{round(t, 6):.6g}"


def _write(path, cls, loc, obj, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(HEADER.format(cls=cls, loc=loc, obj=obj))
        fh.write(body)


def write_case_skeleton(case, Lx, Ly, Lz, nx, ny, nz, g, rho_h, rho_l, nu_h, nu_l):
    dim3 = nz > 1
    zt = Lz if dim3 else 0.01
    _write(os.path.join(case, 'constant', 'g'), 'uniformDimensionedVectorField', 'constant', 'g',
           f"dimensions      [0 1 -2 0 0 0 0];\nvalue           (0 {-g} 0);\n")
    for name, rho, nu in (('water', rho_h, nu_h), ('air', rho_l, nu_l)):
        _write(os.path.join(case, 'constant', f'physicalProperties.{name}'), 'dictionary', 'constant',
               f'physicalProperties.{name}',
               f"viscosityModel  constant;\nnu              {nu:.6g};\nrho             {rho:.6g};\n")
    _write(os.path.join(case, 'system', 'blockMeshDict'), 'dictionary', 'system', 'blockMeshDict',
           "scale   1;\nvertices\n(\n"
           f"    (0 0 0)\n    ({Lx} 0 0)\n    ({Lx} {Ly} 0)\n    (0 {Ly} 0)\n"
           f"    (0 0 {zt})\n    ({Lx} 0 {zt})\n    ({Lx} {Ly} {zt})\n    (0 {Ly} {zt})\n);\n"
           f"blocks\n(\n    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)\n);\n")
    xn = np.linspace(0.0, Lx, nx + 1)
    yn = np.linspace(0.0, Ly, ny + 1)
    pts = [f"({v:.9g} 0 0)" for v in xn] + [f"(0 {v:.9g} 0)" for v in yn[1:]]
    _write(os.path.join(case, 'constant', 'polyMesh', 'points'), 'vectorField', 'constant/polyMesh',
           'points', f"{len(pts)}\n(\n" + "\n".join(pts) + "\n)\n")


def write_scalar(path, name, arr):
    """arr shaped (nz, ny, nx); OpenFOAM order is x fastest, then y, then z."""
    a = np.ascontiguousarray(arr, dtype=np.float64).ravel()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(HEADER.format(cls='volScalarField', loc=os.path.basename(os.path.dirname(path)), obj=name))
        fh.write(f"dimensions      [0 0 0 0 0 0 0];\n\ninternalField   nonuniform List<scalar>\n{a.size}\n(\n")
        a.tofile(fh, sep='\n', format='%.6g')
        fh.write("\n)\n;\n\nboundaryField\n{\n    \".*\" { type zeroGradient; }\n}\n")


def write_vector(path, name, ux, uy, uz):
    n = ux.size
    v = np.empty((n, 3))
    v[:, 0] = ux.ravel(); v[:, 1] = uy.ravel(); v[:, 2] = uz.ravel()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(HEADER.format(cls='volVectorField', loc=os.path.basename(os.path.dirname(path)), obj=name))
        fh.write(f"dimensions      [0 1 -1 0 0 0 0];\n\ninternalField   nonuniform List<vector>\n{n}\n(\n")
        np.savetxt(fh, v, fmt='(%.6g %.6g %.6g)')
        fh.write(")\n;\n\nboundaryField\n{\n    \".*\" { type calculated; value uniform (0 0 0); }\n}\n")


def write_interface_vtk(path, phi, dx, level=0.5):
    """phi=0.5 isosurface of a 3D field (nz, ny, nx) as legacy VTK polydata,
    matching postProcessing/extractInterface/<t>/interface.vtk."""
    from skimage.measure import marching_cubes
    verts, faces, _, _ = marching_cubes(phi.astype(np.float32), level=level, spacing=(dx, dx, dx))
    xyz = verts[:, ::-1] + 0.5 * dx           # (z,y,x) -> (x,y,z), cell centres
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(f"# vtk DataFile Version 2.0\nsampleSurface\nASCII\nDATASET POLYDATA\nPOINTS {len(xyz)} float\n")
        np.savetxt(fh, xyz, fmt='%.6g')
        fh.write(f"POLYGONS {len(faces)} {4 * len(faces)}\n")
        np.savetxt(fh, np.column_stack([np.full(len(faces), 3), faces]), fmt='%d')
