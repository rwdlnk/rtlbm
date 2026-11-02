"""Convert an rtlbm case to ParaView files.

rtlbm writes a *stub* polyMesh (see foamio.py), so ParaView's OpenFOAM reader
cannot open the case.  This writes, inside the case:

    paraview/fields_<t>.vti     VTK ImageData (cell data alpha.water [, U]),
                                raw little-endian float32, one per time
    paraview/fields.pvd         time series of the .vti files  -> open this
    paraview/interface_<t>.vtp  3D only: binary copies of
                                postProcessing/extractInterface/<t>/interface.vtk
    paraview/interface.pvd      their time series (PVD only takes XML files)

usage:  python -m rtlbm.to_paraview CASE [CASE ...] [--every N] [--force]
"""
import argparse
import os
import re
import sys

import numpy as np


def _grid(case):
    """(nx, ny, nz, Lx, Ly, Lz) from system/blockMeshDict."""
    with open(os.path.join(case, 'system', 'blockMeshDict')) as fh:
        s = fh.read()
    nx, ny, nz = (int(v) for v in re.search(r'hex\s*\([^)]*\)\s*\((\d+)\s+(\d+)\s+(\d+)\)', s).groups())
    vb = re.search(r'vertices\s*\(([\s\S]*?)\)\s*;', s).group(1)
    v = np.array(re.findall(r'\(\s*([-0-9.eE+]+)\s+([-0-9.eE+]+)\s+([-0-9.eE+]+)\s*\)', vb), float)
    Lx, Ly, Lz = v.max(0) - v.min(0)
    return nx, ny, nz, Lx, Ly, Lz


def _internal(path, ncomp):
    with open(path) as fh:
        s = fh.read()
    m = re.search(r'internalField\s+nonuniform\s+List<\w+>\s*(\d+)\s*\(', s)
    n = int(m.group(1))
    body = s[m.end():s.index('\n)', m.end())]
    a = np.array(body.replace('(', ' ').replace(')', ' ').split(), dtype=np.float32)
    return a.reshape(n, ncomp) if ncomp > 1 else a


def _times(case):
    out = []
    for d in os.listdir(case):
        try:
            t = float(d)
        except ValueError:
            continue
        if os.path.isfile(os.path.join(case, d, 'alpha.water')):
            out.append((t, d))
    return sorted(out)


def write_vti(path, nx, ny, nz, dx, dy, dz, arrays):
    """arrays: [(name, float32 array of n cells or (n, 3))], x fastest."""
    head, blobs, off = [], [], 0
    for name, a in arrays:
        a = np.ascontiguousarray(a, dtype='<f4')
        nc = 1 if a.ndim == 1 else a.shape[1]
        head.append(f'<DataArray type="Float32" Name="{name}" NumberOfComponents="{nc}" '
                    f'format="appended" offset="{off}"/>')
        blobs.append(a)
        off += 8 + a.nbytes
    with open(path, 'wb') as fh:
        fh.write((f'<?xml version="1.0"?>\n<VTKFile type="ImageData" version="1.0" '
                  f'byte_order="LittleEndian" header_type="UInt64">\n'
                  f'<ImageData WholeExtent="0 {nx} 0 {ny} 0 {nz}" Origin="0 0 0" '
                  f'Spacing="{dx:.9g} {dy:.9g} {dz:.9g}">\n<Piece Extent="0 {nx} 0 {ny} 0 {nz}">\n'
                  f'<CellData Scalars="alpha.water">\n' + '\n'.join(head) +
                  '\n</CellData>\n</Piece>\n</ImageData>\n<AppendedData encoding="raw">\n_').encode())
        for a in blobs:
            fh.write(np.uint64(a.nbytes).tobytes())
            fh.write(a.tobytes())
        fh.write(b'\n</AppendedData>\n</VTKFile>\n')


def legacy_vtk_to_vtp(src, dst):
    """ASCII legacy POLYDATA triangles (as foamio.write_interface_vtk) -> binary .vtp."""
    with open(src) as fh:
        lines = fh.read().split('\n')
    i = next(k for k, l in enumerate(lines) if l.startswith('POINTS'))
    n = int(lines[i].split()[1])
    pts = np.array(' '.join(lines[i + 1:i + 1 + n]).split(), dtype='<f4')
    j = next(k for k, l in enumerate(lines) if l.startswith('POLYGONS'))
    m = int(lines[j].split()[1])
    tri = np.array(' '.join(lines[j + 1:j + 1 + m]).split(), dtype=np.int64).reshape(m, -1)[:, 1:]
    conn = np.ascontiguousarray(tri.ravel(), dtype='<i8')
    offs = np.arange(1, m + 1, dtype='<i8') * tri.shape[1]
    blobs, off, head = [], 0, []
    for tag, name, a, nc in (('Points', 'Points', pts, 3), ('Polys', 'connectivity', conn, 1),
                             ('Polys', 'offsets', offs, 1)):
        typ = 'Float32' if a.dtype.kind == 'f' else 'Int64'
        head.append((tag, f'<DataArray type="{typ}" Name="{name}" NumberOfComponents="{nc}" '
                          f'format="appended" offset="{off}"/>'))
        blobs.append(a)
        off += 8 + a.nbytes
    with open(dst, 'wb') as fh:
        fh.write((f'<?xml version="1.0"?>\n<VTKFile type="PolyData" version="1.0" '
                  f'byte_order="LittleEndian" header_type="UInt64">\n<PolyData>\n'
                  f'<Piece NumberOfPoints="{n}" NumberOfPolys="{m}">\n'
                  f'<Points>{head[0][1]}</Points>\n<Polys>{head[1][1]}{head[2][1]}</Polys>\n'
                  '</Piece>\n</PolyData>\n<AppendedData encoding="raw">\n_').encode())
        for a in blobs:
            fh.write(np.uint64(a.nbytes).tobytes())
            fh.write(a.tobytes())
        fh.write(b'\n</AppendedData>\n</VTKFile>\n')


def write_pvd(path, entries):
    with open(path, 'w') as fh:
        fh.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="1.0">\n<Collection>\n')
        for t, f in entries:
            fh.write(f'  <DataSet timestep="{t:.6g}" file="{f}"/>\n')
        fh.write('</Collection>\n</VTKFile>\n')


def convert(case, every=1, force=False):
    nx, ny, nz, Lx, Ly, Lz = _grid(case)
    dx, dy = Lx / nx, Ly / ny
    dz = Lz / nz if nz > 1 else dx            # 2D: one cell thick, square cells
    out = os.path.join(case, 'paraview')
    os.makedirs(out, exist_ok=True)
    entries = []
    for i, (t, d) in enumerate(_times(case)):
        if i % every:
            continue
        name = f'fields_{d}.vti'
        entries.append((t, name))
        dst = os.path.join(out, name)
        if os.path.exists(dst) and not force:
            continue
        arrays = [('alpha.water', _internal(os.path.join(case, d, 'alpha.water'), 1))]
        up = os.path.join(case, d, 'U')
        if os.path.isfile(up):
            arrays.append(('U', _internal(up, 3)))
        write_vti(dst, nx, ny, nz, dx, dy, dz, arrays)
        print(f'  t={d}', flush=True)
    write_pvd(os.path.join(out, 'fields.pvd'), entries)
    ie = os.path.join(case, 'postProcessing', 'extractInterface')
    if os.path.isdir(ie):
        iv = sorted((float(d), d) for d in os.listdir(ie)
                    if os.path.isfile(os.path.join(ie, d, 'interface.vtk')))
        ents = []
        for i, (t, d) in enumerate(iv):
            if i % every:
                continue
            name = f'interface_{d}.vtp'
            ents.append((t, name))
            dst = os.path.join(out, name)
            if force or not os.path.exists(dst):
                legacy_vtk_to_vtp(os.path.join(ie, d, 'interface.vtk'), dst)
        write_pvd(os.path.join(out, 'interface.pvd'), ents)
    print(f'{case}: {len(entries)} times -> {out}/fields.pvd')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('cases', nargs='+')
    p.add_argument('--every', type=int, default=1, help='convert every Nth time directory')
    p.add_argument('--force', action='store_true', help='rewrite existing .vti files')
    a = p.parse_args(argv)
    for c in a.cases:
        convert(os.path.abspath(c), a.every, a.force)


if __name__ == '__main__':
    sys.exit(main())
