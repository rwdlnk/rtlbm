#!/usr/bin/env python3
"""Compare late-time D (tau > 2) and h_10 of LBM cases with the Dalziel_1999 SVOF/MULES/MPLIC runs.

    python compare_codes.py cases/2D/M0.002
"""
import os
import sys

import pandas as pd

R = '/media/rod/RT-Simulations/Dalziel_1999/'
REF = {'SVOF': R + 'sola-vof/NDIS-6/{r}', 'MULES': R + 'OpenFOAM-2D/MULES/{r}x1', 'MPLIC': R + 'OpenFOAM-2D/MPLIC/{r}x1'}


def load(case):
    f = os.path.join(case, 'analysis', 'summary', 'temporal_results.csv')
    return pd.read_csv(f) if os.path.isfile(f) else None


def main(root):
    print(f"{'grid':>10s}  " + '  '.join(f"{k:>16s}" for k in ['LBM'] + list(REF)))
    print(f"{'':>10s}  " + '  '.join(f"{'D(t>2) h10(t=3)':>16s}" for _ in range(4)))
    for r in sorted(os.listdir(root), key=lambda s: int(s.split('x')[0]) if s[0].isdigit() else 0):
        case = os.path.join(root, r)
        if not os.path.isdir(case):
            continue
        row = []
        for d in [load(case)] + [load(p.format(r=r)) for p in REF.values()]:
            if d is None:
                row.append(f"{'-':>16s}")
            else:
                h = d.iloc[(d.tau - 3).abs().argmin()].h_10
                row.append(f"{d[d.tau > 2].fractal_dim.mean():8.3f} {h:7.4f}")
        print(f"{r:>10s}  " + '  '.join(row))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'cases/2D/M0.002')
