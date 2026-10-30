"""Single-source LBM kernels for CPU (numba prange) and GPU (numba.cuda).

Model
-----
Flow:   weakly compressible LBM, D2Q9 / D3Q19, recursive-regularized collision
        (Hermite moments to 3rd order where the lattice supports them),
        Guo forcing, optional Smagorinsky LES.  Populations are stored as
        deviations g_i = f_i - w_i to keep float32 accurate.
Buoyancy: Boussinesq, F = -g A (2 phi - 1) rho e_y  (A = Atwood number).
Interface: conservative Allen-Cahn phase field (Chiu & Lin 2011; Geier,
        Fakhari & Lee 2015), D2Q5 / D3Q7, with NO surface tension:
        d(phi)/dt + div(phi u) = div[ M (grad phi - (4/W) phi (1-phi) n) ].
        phi = 1 heavy fluid, phi = 0 light fluid.

Streaming is "pull": a node gathers f_i from x - c_i.  When x - c_i lies
outside the domain the wall rule applies: no-slip -> half-way bounce-back,
slip -> specular reflection.  The phase field uses the same rule (zero flux).

Every per-node routine is written with scalars only (no local arrays) so the
same Python source is compiled for both targets.  All floating constants
are captured with the storage dtype so the GPU stays in float32.
"""
import math

import numba
import numpy as np
from numba import cuda, prange

NOSLIP, SLIP = 0, 1


def build_kernels(lat_f, lat_h, bc, dtype, backend):
    """Compile the two step kernels.

    bc: dict face -> 'noslip'|'slip' for faces xlo,xhi,ylo,yhi,zlo,zhi.
    Returns an object with .compute_phi(h, phi) and
    .collide(f, h, phi, f_new, h_new, ux, uy, uz, params) launchers.
    """
    T = np.dtype(dtype).type
    D = lat_f.dim
    Qf, Qh = lat_f.Q, lat_h.Q

    # constants (closure-captured => compile-time constants with dtype T)
    CXf = lat_f.c[:, 0].astype(dtype); CYf = lat_f.c[:, 1].astype(dtype); CZf = lat_f.c[:, 2].astype(dtype)
    ICX = lat_f.c[:, 0].copy(); ICY = lat_f.c[:, 1].copy(); ICZ = lat_f.c[:, 2].copy()
    Wf = lat_f.w.astype(dtype)
    OPPf = lat_f.opp; MXf, MYf, MZf = lat_f.mirror
    HCX = lat_h.c[:, 0].astype(dtype); HCY = lat_h.c[:, 1].astype(dtype); HCZ = lat_h.c[:, 2].astype(dtype)
    IHX = lat_h.c[:, 0].copy(); IHY = lat_h.c[:, 1].copy(); IHZ = lat_h.c[:, 2].copy()
    Wh = lat_h.w.astype(dtype)
    OPPh = lat_h.opp; MXh, MYh, MZh = lat_h.mirror

    code = {'noslip': NOSLIP, 'slip': SLIP}
    BXL, BXH = code[bc['xlo']], code[bc['xhi']]
    BYL, BYH = code[bc['ylo']], code[bc['yhi']]
    BZL, BZH = code[bc['zlo']], code[bc['zhi']]

    ZERO, HALF, ONE, TWO = T(0), T(0.5), T(1), T(2)
    THREE, NINE = T(3), T(9)
    THIRD = T(1 / 3)
    C2 = T(4.5)              # 1/(2 cs^4)
    C3 = T(13.5)             # 1/(2 cs^6)
    SMAGK = T(18 * math.sqrt(2.0))
    INV_CSH2 = T(1.0 / lat_h.cs2)
    INV_CSF2 = T(3.0)
    TINY = T(1e-12)

    if backend == 'cuda':
        dev = cuda.jit(device=True, inline=True)
    else:
        dev = numba.njit(inline='always', fastmath=False)

    # ------------------------------------------------------------------ helpers
    @dev
    def pull_f(q, x, y, z, NX, NY, NZ):
        sx = x - ICX[q]; sy = y - ICY[q]; sz = z - ICZ[q]
        qs = q; bb = False
        if sx < 0:
            if BXL == NOSLIP: bb = True
            else: sx = x; qs = MXf[qs]
        elif sx >= NX:
            if BXH == NOSLIP: bb = True
            else: sx = x; qs = MXf[qs]
        if sy < 0:
            if BYL == NOSLIP: bb = True
            else: sy = y; qs = MYf[qs]
        elif sy >= NY:
            if BYH == NOSLIP: bb = True
            else: sy = y; qs = MYf[qs]
        if sz < 0:
            if BZL == NOSLIP: bb = True
            else: sz = z; qs = MZf[qs]
        elif sz >= NZ:
            if BZH == NOSLIP: bb = True
            else: sz = z; qs = MZf[qs]
        if bb:
            return OPPf[q], x, y, z
        return qs, sx, sy, sz

    @dev
    def pull_h(q, x, y, z, NX, NY, NZ):
        # phase field: always zero-flux (specular reflection keeps the
        # tangential flux, bounce-back would add none either; reflection is
        # used everywhere so the contact angle stays neutral)
        sx = x - IHX[q]; sy = y - IHY[q]; sz = z - IHZ[q]
        qs = q
        if sx < 0 or sx >= NX:
            sx = x; qs = MXh[qs]
        if sy < 0 or sy >= NY:
            sy = y; qs = MYh[qs]
        if sz < 0 or sz >= NZ:
            sz = z; qs = MZh[qs]
        return qs, sx, sy, sz

    @dev
    def phi_node(h, x, y, z, NX, NY, NZ):
        s = ZERO
        for q in range(Qh):
            qs, sx, sy, sz = pull_h(q, x, y, z, NX, NY, NZ)
            s += h[qs, sz, sy, sx]
        return s

    @dev
    def clampm(i, n):
        # zero-gradient ghost: a neighbour outside the wall mirrors to the node
        if i < 0:
            return 0
        if i >= n:
            return n - 1
        return i

    @dev
    def collide_node(f, h, phi, fn, hn, ux_a, uy_a, uz_a, x, y, z, NX, NY, NZ,
                     ga, nu_l, nu_h, csmag2, mob, beta, omh):
        ph = phi[z, y, x]

        # isotropic gradient of phi (fluid stencil)
        gx = ZERO; gy = ZERO; gz = ZERO
        for q in range(1, Qf):
            v = phi[clampm(z + ICZ[q], NZ), clampm(y + ICY[q], NY), clampm(x + ICX[q], NX)]
            wv = Wf[q] * v
            gx += wv * CXf[q]; gy += wv * CYf[q]; gz += wv * CZf[q]
        gx *= INV_CSF2; gy *= INV_CSF2; gz *= INV_CSF2

        # ---- fluid moments of pulled populations
        rho = ZERO; jx = ZERO; jy = ZERO; jz = ZERO
        pxx = ZERO; pyy = ZERO; pzz = ZERO; pxy = ZERO; pxz = ZERO; pyz = ZERO
        for q in range(Qf):
            qs, sx, sy, sz = pull_f(q, x, y, z, NX, NY, NZ)
            g = f[qs, sz, sy, sx]
            cx = CXf[q]; cy = CYf[q]; cz = CZf[q]
            rho += g
            jx += cx * g; jy += cy * g; jz += cz * g
            pxx += cx * cx * g; pyy += cy * cy * g; pxy += cx * cy * g
            if D == 3:
                pzz += cz * cz * g; pxz += cx * cz * g; pyz += cy * cz * g
        drho = rho
        rho = ONE + drho

        # Boussinesq buoyancy (gravity along -y)
        Fy = -ga * (TWO * ph - ONE) * rho
        inv_rho = ONE / rho
        ux = jx * inv_rho
        uy = (jy + HALF * Fy) * inv_rho
        uz = jz * inv_rho
        ux_a[z, y, x] = ux; uy_a[z, y, x] = uy; uz_a[z, y, x] = uz

        # non-equilibrium (Guo-corrected) second moments:
        # Pi = cs2*delta + sum c c g ;  Pi_eq = rho u u + rho cs2 delta
        axx = pxx + THIRD - rho * ux * ux - rho * THIRD
        ayy = pyy + THIRD - rho * uy * uy - rho * THIRD + uy * Fy
        axy = pxy - rho * ux * uy + HALF * ux * Fy
        azz = ZERO; axz = ZERO; ayz = ZERO
        if D == 3:
            azz = pzz + THIRD - rho * uz * uz - rho * THIRD
            axz = pxz - rho * ux * uz
            ayz = pyz - rho * uy * uz + HALF * uz * Fy

        # relaxation time (phase-weighted viscosity + Smagorinsky)
        tau0 = THREE * (nu_l + ph * (nu_h - nu_l)) + HALF
        tau = tau0
        if csmag2 > ZERO:
            qn = math.sqrt(axx * axx + ayy * ayy + azz * azz
                           + TWO * (axy * axy + axz * axz + ayz * ayz))
            tau = HALF * (tau0 + math.sqrt(tau0 * tau0 + SMAGK * csmag2 * qn * inv_rho))
        omr = ONE - ONE / tau

        # recursive 3rd-order non-equilibrium coefficients
        nxxy = uy * axx + TWO * ux * axy
        nxyy = ux * ayy + TWO * uy * axy
        exxy = rho * ux * ux * uy
        exyy = rho * ux * uy * uy
        nxxz = ZERO; nxzz = ZERO; nyyz = ZERO; nyzz = ZERO
        exxz = ZERO; exzz = ZERO; eyyz = ZERO; eyzz = ZERO
        if D == 3:
            nxxz = uz * axx + TWO * ux * axz
            nxzz = ux * azz + TWO * uz * axz
            nyyz = uz * ayy + TWO * uy * ayz
            nyzz = uy * azz + TWO * uz * ayz
            exxz = rho * ux * ux * uz
            exzz = rho * ux * uz * uz
            eyyz = rho * uy * uy * uz
            eyzz = rho * uy * uz * uz

        usq = ux * ux + uy * uy + uz * uz
        for q in range(Qf):
            cx = CXf[q]; cy = CYf[q]; cz = CZf[q]
            w = Wf[q]
            cu = cx * ux + cy * uy + cz * uz
            hxx = cx * cx - THIRD; hyy = cy * cy - THIRD; hzz = cz * cz - THIRD
            # equilibrium (minus w), 2nd order + supported 3rd-order terms
            t3e = (hxx * cy) * exxy + (cx * hyy) * exyy
            t3n = (hxx * cy) * nxxy + (cx * hyy) * nxyy
            if D == 3:
                t3e += (hxx * cz) * exxz + (cx * hzz) * exzz + (hyy * cz) * eyyz + (cy * hzz) * eyzz
                t3n += (hxx * cz) * nxxz + (cx * hzz) * nxzz + (hyy * cz) * nyyz + (cy * hzz) * nyzz
            feq = w * (drho + rho * (THREE * cu + C2 * cu * cu - HALF * THREE * usq) + C3 * t3e)
            reg = w * (C2 * (hxx * axx + hyy * ayy + hzz * azz
                             + TWO * (cx * cy * axy + cx * cz * axz + cy * cz * ayz))
                       + C3 * t3n)
            src = w * (THREE * (cy - uy) + NINE * cu * cy) * Fy
            fn[q, z, y, x] = feq + omr * reg + HALF * src

        # ---- phase field (conservative Allen-Cahn, no surface tension)
        gm = math.sqrt(gx * gx + gy * gy + gz * gz) + TINY
        sharp = mob * beta * ph * (ONE - ph) / gm
        Jx = ph * ux + sharp * gx
        Jy = ph * uy + sharp * gy
        Jz = ph * uz + sharp * gz
        # q = 0 (rest) is set last so that sum_q hn = phi exactly in float32;
        # otherwise rounding makes bulk phi = 1 drift and mass creeps.
        hsum = ZERO
        for q in range(1, Qh):
            qs, sx, sy, sz = pull_h(q, x, y, z, NX, NY, NZ)
            hv = h[qs, sz, sy, sx]
            heq = Wh[q] * (ph + INV_CSH2 * (HCX[q] * Jx + HCY[q] * Jy + HCZ[q] * Jz))
            hq = heq + (ONE - omh) * (hv - heq)
            hn[q, z, y, x] = hq
            hsum += hq
        hn[0, z, y, x] = ph - hsum

    # ---------------------------------------------------------------- drivers
    if backend == 'cuda':
        @cuda.jit
        def k_phi(h, phi):
            x, y, z = cuda.grid(3)
            NZ, NY, NX = phi.shape
            if x < NX and y < NY and z < NZ:
                phi[z, y, x] = phi_node(h, x, y, z, NX, NY, NZ)

        @cuda.jit
        def k_collide(f, h, phi, fn, hn, ux, uy, uz, ga, nu_l, nu_h, csmag2, mob, beta, omh):
            x, y, z = cuda.grid(3)
            NZ, NY, NX = phi.shape
            if x < NX and y < NY and z < NZ:
                collide_node(f, h, phi, fn, hn, ux, uy, uz, x, y, z, NX, NY, NZ,
                             ga, nu_l, nu_h, csmag2, mob, beta, omh)

        class K:
            @staticmethod
            def _grid(shape):
                NZ, NY, NX = shape
                tpb = (128, 1, 1)
                return ((NX + tpb[0] - 1) // tpb[0], NY, NZ), tpb

            @classmethod
            def compute_phi(cls, h, phi):
                g, b = cls._grid(phi.shape)
                k_phi[g, b](h, phi)

            @classmethod
            def collide(cls, f, h, phi, fn, hn, ux, uy, uz, p):
                g, b = cls._grid(phi.shape)
                k_collide[g, b](f, h, phi, fn, hn, ux, uy, uz, *[T(v) for v in p])
        return K

    @numba.njit(parallel=True)
    def c_phi(h, phi):
        NZ, NY, NX = phi.shape
        for zy in prange(NZ * NY):
            z = zy // NY; y = zy - z * NY
            for x in range(NX):
                phi[z, y, x] = phi_node(h, x, y, z, NX, NY, NZ)

    @numba.njit(parallel=True)
    def c_collide(f, h, phi, fn, hn, ux, uy, uz, ga, nu_l, nu_h, csmag2, mob, beta, omh):
        NZ, NY, NX = phi.shape
        for zy in prange(NZ * NY):
            z = zy // NY; y = zy - z * NY
            for x in range(NX):
                collide_node(f, h, phi, fn, hn, ux, uy, uz, x, y, z, NX, NY, NZ,
                             ga, nu_l, nu_h, csmag2, mob, beta, omh)

    class K:
        @staticmethod
        def compute_phi(h, phi):
            c_phi(h, phi)

        @staticmethod
        def collide(f, h, phi, fn, hn, ux, uy, uz, p):
            c_collide(f, h, phi, fn, hn, ux, uy, uz, *[T(v) for v in p])
    return K
