"""Velocity sets.

Fluid: D2Q9 / D3Q19 (cs^2 = 1/3).
Phase field: D2Q5 / D3Q7 (cs^2 = 1/3 for D2Q5, 1/4 for D3Q7).

Every lattice is stored with 3 velocity components (cz = 0 in 2D) so that
the same kernels serve 2D (NZ = 1) and 3D.
"""
import numpy as np


class Lattice:
    def __init__(self, name, c, w, cs2):
        self.name = name
        self.c = np.asarray(c, dtype=np.int64)
        self.w = np.asarray(w, dtype=np.float64)
        self.cs2 = float(cs2)
        self.Q = len(self.w)
        self.opp = self._lookup(-self.c)
        # mirror tables: index of the velocity with one component negated
        self.mirror = []
        for a in range(3):
            m = self.c.copy()
            m[:, a] *= -1
            self.mirror.append(self._lookup(m))
        self._check()

    def _lookup(self, cs):
        idx = np.empty(self.Q, dtype=np.int64)
        for q, v in enumerate(cs):
            idx[q] = int(np.where((self.c == v).all(axis=1))[0][0])
        return idx

    def _check(self):
        assert abs(self.w.sum() - 1.0) < 1e-14
        assert np.allclose(self.w @ self.c, 0.0)
        assert np.allclose(np.einsum('q,qa,qb->ab', self.w, self.c, self.c)[:self.dim, :self.dim],
                           self.cs2 * np.eye(self.dim))

    @property
    def dim(self):
        return 3 if np.any(self.c[:, 2] != 0) else 2


def _d2q9():
    c = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0),
         (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0)]
    w = [4 / 9] + [1 / 9] * 4 + [1 / 36] * 4
    return Lattice('D2Q9', c, w, 1 / 3)


def _d3q19():
    c = [(0, 0, 0)]
    c += [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    c += [(1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0),
          (1, 0, 1), (-1, 0, -1), (1, 0, -1), (-1, 0, 1),
          (0, 1, 1), (0, -1, -1), (0, 1, -1), (0, -1, 1)]
    w = [1 / 3] + [1 / 18] * 6 + [1 / 36] * 12
    return Lattice('D3Q19', c, w, 1 / 3)


def _d2q5():
    c = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
    w = [1 / 3] + [1 / 6] * 4
    return Lattice('D2Q5', c, w, 1 / 3)


def _d3q7():
    c = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    w = [1 / 4] + [1 / 8] * 6
    return Lattice('D3Q7', c, w, 1 / 4)


def lattices_for(dim):
    """Return (fluid lattice, phase-field lattice) for dimension 2 or 3."""
    if dim == 2:
        return _d2q9(), _d2q5()
    if dim == 3:
        return _d3q19(), _d3q7()
    raise ValueError(dim)
