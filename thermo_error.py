#!/usr/bin/env python3
"""Numba/SciPy translation of thermo_error.c.

The algorithmic structure is intentionally kept close to the C version:
read PTMCMC log-likelihood chains, estimate the covariance matrix with the
threshold bootstrap, shrink the correlation matrix, then run the spline RJMCMC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import linalg
from numba import njit


IM1 = 2147483563
IM2 = 2147483399
AM = 1.0 / IM1
IMM1 = IM1 - 1
IA1 = 40014
IA2 = 40692
IQ1 = 53668
IQ2 = 52774
IR1 = 12211
IR2 = 3791
NTAB = 32
NDIV = 1 + IMM1 // NTAB
EPS = 1.2e-7
RNMX = 1.0 - EPS


@njit(cache=True)
def _init_rng(seed: int) -> tuple[np.ndarray, np.ndarray]:
    '''Initializes the RNG state for the C-style ran2 random number generator.
    Returns a state vector and an initialization vector iv used by _ran2.'''
    state = np.empty(4, dtype=np.int64)
    state[0] = seed
    state[1] = 123456789
    state[2] = 0
    state[3] = 0
    iv = np.zeros(NTAB, dtype=np.int64)
    return state, iv


@njit(cache=True)
def _ran2(state: np.ndarray, iv: np.ndarray) -> float:
    '''Implements the ran2 RNG from Numerical Recipes.
    Advances the RNG state and returns a single uniform random number in (0, 1).'''
    if state[0] <= 0:
        if -state[0] < 1:
            state[0] = 1
        else:
            state[0] = -state[0]
        state[1] = state[0]
        for j in range(NTAB + 7, -1, -1):
            k = state[0] // IQ1
            state[0] = IA1 * (state[0] - k * IQ1) - k * IR1
            if state[0] < 0:
                state[0] += IM1
            if j < NTAB:
                iv[j] = state[0]
        state[2] = iv[0]

    k = state[0] // IQ1
    state[0] = IA1 * (state[0] - k * IQ1) - k * IR1
    if state[0] < 0:
        state[0] += IM1

    k = state[1] // IQ2
    state[1] = IA2 * (state[1] - k * IQ2) - k * IR2
    if state[1] < 0:
        state[1] += IM2

    j = state[2] // NDIV
    state[2] = iv[j] - state[1]
    iv[j] = state[0]
    if state[2] < 1:
        state[2] += IMM1

    temp = AM * state[2]
    if temp > RNMX:
        return RNMX
    return temp


@njit(cache=True)
def _gasdev(state: np.ndarray, iv: np.ndarray, gas_state: np.ndarray) -> float:
    '''Generates Gaussian random variates using the Box-Muller method.
    Uses _ran2 for uniform samples and caches one normal deviate for efficiency.'''
    if state[0] < 0:
        gas_state[0] = 0.0
    if gas_state[0] == 0.0:
        while True:
            v1 = 2.0 * _ran2(state, iv) - 1.0
            v2 = 2.0 * _ran2(state, iv) - 1.0
            rsq = v1 * v1 + v2 * v2
            if rsq < 1.0 and rsq != 0.0:
                break
        fac = np.sqrt(-2.0 * np.log(rsq) / rsq)
        gas_state[1] = v1 * fac
        gas_state[0] = 1.0
        return v2 * fac
    gas_state[0] = 0.0
    return gas_state[1]


@njit(cache=True)
def _spline_y2(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    '''Computes the second derivatives for a natural cubic spline interpolation.
    Returns the array y2 for spline construction from control points x, y.'''
    y2 = np.empty(n, dtype=np.float64)
    u = np.empty(n, dtype=np.float64)
    y2[0] = 0.0
    u[0] = 0.0
    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * y2[i - 1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = ((y[i + 1] - y[i]) / (x[i + 1] - x[i])
                - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
        u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p
    y2[n - 1] = 0.0
    for k in range(n - 2, -1, -1):
        y2[k] = y2[k] * y2[k + 1] + u[k]
    return y2


@njit(cache=True)
def _splint(xa: np.ndarray, ya: np.ndarray, y2a: np.ndarray, n: int, x: float) -> float:
    '''Evaluates the cubic spline at a point x using precomputed second derivatives.
    Uses binary search to locate the interval and returns the interpolated value.'''
    klo = 0
    khi = n - 1
    while khi - klo > 1:
        k = (khi + klo) >> 1
        if xa[k] > x:
            khi = k
        else:
            klo = k
    h = xa[khi] - xa[klo]
    a = (xa[khi] - x) / h
    b = (x - xa[klo]) / h
    return (a * ya[klo] + b * ya[khi]
            + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * h * h / 6.0)


@njit(cache=True)
def _bootstrap_covariance(data: np.ndarray, bsteps: int, state: np.ndarray, iv: np.ndarray):
    '''Estimates a covariance matrix from MCMC chain data using a threshold bootstrap.
    Steps:
        compute mean and standard deviation of each chain
        choose bootstrap chunk size adaptively using sign changes
        build bootstrap replicates from chunks
        compute bootstrap standard deviations and normalized replicates
        estimate correlation matrix and its variance
        compute shrinkage factor and return a shrunk covariance matrix
    Returns: covariance matrix, mean vector, chunk statistics, shrink factor, and metadata.'''
    nc, n = data.shape
    mu = np.empty(nc, dtype=np.float64)
    var = np.empty(nc, dtype=np.float64)
    cnts = np.empty(nc, dtype=np.int64)

    for ii in range(nc):
        s1 = 0.0
        s2 = 0.0
        for i in range(n):
            v = data[ii, i]
            s1 += v
            s2 += v * v
        mu[ii] = s1 / n
        var[ii] = np.sqrt(s2 / n - mu[ii] * mu[ii])

    b = 8
    cmin = 0
    cmax = 0
    icm = 0
    icx = 0
    cc = 0.0
    vc = 0.0
    while True:
        kk = 0
        nn = 0
        cmin = 100000
        cmax = 0
        for ii in range(nc):
            cycles = 0
            so = -1
            if data[ii, 0] - mu[ii] > 0.0:
                so = 1
            cp = 0
            for i in range(n):
                sc = -1
                if data[ii, i] - mu[ii] > 0.0:
                    sc = 1
                if sc != so:
                    cp += 1
                so = sc
                if cp == 2 * b:
                    cycles += 1
                    cp = 0
            cnts[ii] = cycles
            if cycles < cmin:
                cmin = cycles
                icm = ii
            if cycles > cmax:
                cmax = cycles
                icx = ii
            kk += cycles
            nn += cycles * cycles

        cc = kk / nc
        vc = np.sqrt(nn / nc - kk * kk / (nc * nc))
        if cmin < 100:
            b //= 2
        if cmin > 1000:
            b *= 2
        if b < 2:
            b = 2
            cmin = 500
        if cmin >= 100 and cmin <= 1000:
            break

    av = 0.0
    for i in range(n):
        av += data[icm, i]
    av /= n

    cyc = np.empty(n // 2 + 1, dtype=np.int64)
    cyc[0] = -1
    cycles = 0
    so = -1
    if data[icm, 0] - av > 0.0:
        so = 1
    cp = 0
    for i in range(n):
        sc = -1
        if data[icm, i] - av > 0.0:
            sc = 1
        if sc != so:
            cp += 1
        so = sc
        if cp == 2 * b:
            cycles += 1
            cyc[cycles] = i - 1
            cp = 0

    if cycles < 1:
        raise ValueError("No bootstrap chunks were found")

    chunk_len = np.empty(cycles, dtype=np.int64)
    chunk_sum = np.zeros((cycles, nc), dtype=np.float64)
    for chunk in range(cycles):
        start = cyc[chunk] + 1
        end = cyc[chunk + 1]
        chunk_len[chunk] = end - cyc[chunk]
        for j in range(start, end + 1):
            for ii in range(nc):
                chunk_sum[chunk, ii] += data[ii, j]

    fmean = np.zeros((nc, bsteps), dtype=np.float64)
    for bs in range(bsteps):
        k = 0
        while k < n:
            chunk = int(cycles * _ran2(state, iv))
            if k + chunk_len[chunk] <= n:
                for ii in range(nc):
                    fmean[ii, bs] += chunk_sum[chunk, ii]
                k += chunk_len[chunk]
            else:
                take = n - k
                start = cyc[chunk] + 1
                for j in range(start, start + take):
                    for ii in range(nc):
                        fmean[ii, bs] += data[ii, j]
                k += take
        for ii in range(nc):
            fmean[ii, bs] /= n

    boot_sigma = np.empty(nc, dtype=np.float64)
    for ii in range(nc):
        mx = 0.0
        vv = 0.0
        for bs in range(bsteps):
            mx += fmean[ii, bs]
            vv += fmean[ii, bs] * fmean[ii, bs]
        mx /= bsteps
        vv = vv / bsteps - mx * mx
        boot_sigma[ii] = np.sqrt(vv)
        for bs in range(bsteps):
            fmean[ii, bs] = (fmean[ii, bs] - mx) / boot_sigma[ii]

    rmat = np.empty((nc, nc), dtype=np.float64)
    rvar = np.empty((nc, nc), dtype=np.float64)
    for ii in range(nc):
        for jj in range(nc):
            s = 0.0
            for bs in range(bsteps):
                s += fmean[ii, bs] * fmean[jj, bs]
            rmat[ii, jj] = s / bsteps

    denom = (bsteps - 1.0) ** 3
    for ii in range(nc):
        for jj in range(nc):
            s = 0.0
            for bs in range(bsteps):
                z = fmean[ii, bs] * fmean[jj, bs] - rmat[ii, jj]
                s += z * z
            rvar[ii, jj] = bsteps * s / denom

    x = 0.0
    y = 0.0
    for ii in range(nc):
        for jj in range(nc):
            if ii != jj:
                x += rvar[ii, jj]
                y += rmat[ii, jj] * rmat[ii, jj]
    shrink = 1.0 - x / y
    if shrink < 0.0:
        shrink = 0.0
    if shrink > 1.0:
        shrink = 1.0

    cmat = np.empty((nc, nc), dtype=np.float64)
    for ii in range(nc):
        for jj in range(nc):
            if ii == jj:
                cmat[ii, jj] = boot_sigma[ii] * boot_sigma[jj]
            else:
                cmat[ii, jj] = shrink * boot_sigma[ii] * boot_sigma[jj] * rmat[ii, jj]

    return cmat, mu, cc, vc, cmin, icm, cmax, icx, cycles, shrink


@njit(cache=True)
def _roughness_prior(points, values, y2, n, ns, smooth, ep):
    '''Computes a prior term penalizing rough spline shapes.
    Uses finite-difference approximations for second and first derivatives.
    Returns a scalar log-prior that penalizes large curvature relative to slope.'''
    logp = 0.0
    for i in range(1, ns - 1):
        x = points[i]
        dx = points[i] - points[i - 1]
        y = _splint(points, values, y2, n, x)
        y3 = _splint(points, values, y2, n, x - ep)
        y4 = _splint(points, values, y2, n, x + ep)
        secd = (y3 + y4 - 2.0 * y) / (ep * ep)
        fird = (y4 - y3) / (2.0 * ep)
        logp -= (smooth / ns) * (secd * secd * dx * dx) / (fird * fird)
        if fird < 0.0:
            logp -= 10.0
    return logp


@njit(cache=True)
def _quad_form(icov, data, model):
    '''Computes quadratic form (data - model)^T icov (data - model).
    Used to evaluate Gaussian log-likelihood terms with the inverse covariance matrix.'''
    n = data.shape[0]
    av = 0.0
    for i in range(n):
        di = data[i] - model[i]
        for j in range(n):
            av += icov[i, j] * di * (data[j] - model[j])
    return av


@njit(cache=True)
def _integrate_spline(points, values, y2, n, xmin, xmax):
    '''Integrates a spline-weighted quantity over x using fixed trapezoidal integration.
    Computes two integrals:
        trap: integral of exp(x) * spline(x)
        trp: integral of (exp(x+dx)-exp(x))/2 * (spline values) as a second measure'''
    trap = 0.0
    trp = 0.0
    x3 = xmin
    y3 = _splint(points, values, y2, n, x3)
    for i in range(1, 1001):
        x = x3
        y = y3
        x3 = xmin + (xmax - xmin) * i / 1000.0
        y3 = _splint(points, values, y2, n, x3)
        trap += 0.5 * (x3 - x) * (np.exp(x3) * y3 + np.exp(x) * y)
        trp += 0.5 * (np.exp(x3) - np.exp(x)) * (y3 + y)

#############################################################################################################################
    print(f'~~~~~ Dunno what this code does but here are trap and trp: {trap}, {trp}~~~~~')
#############################################################################################################################

    return trap, trp


@njit(cache=True)
def _run_rjmcmc(datax, datay, sigma, icov, base, steps, smooth, ep, state, iv):
    '''Runs a reversible-jump MCMC algorithm on a spline representation of the integrand.
    Uses control points at data locations and midpoints, with active/inactive spline nodes.
    Proposes either dimension changes (add/remove control points) or parameter perturbations.
    Evaluates proposals with log-likelihood and roughness prior, then accepts/rejects.
    Records:
        pchain: frequent parameter chain samples every 10 steps
        chain: summary samples every 1000 steps
        fit: final interpolated fit curve
        summary: average evidence and KL/J statistics
    Returns control points, initial and final spline values, prior references, integration estimates, chains, fit, and summary stats.'''
    nd = datax.shape[0]
    ns = 2 * nd - 1
    xmin = datax[0]
    xmax = datax[nd - 1]
    gas_state = np.zeros(2, dtype=np.float64)

    ref = np.empty(ns, dtype=np.float64)
    sprd = np.empty(ns, dtype=np.float64)
    spoints = np.empty(ns, dtype=np.float64)
    sdatax = np.empty(ns, dtype=np.float64)
    sdatay = np.empty(ns, dtype=np.float64)
    active = np.ones(ns, dtype=np.int64)
    activey = np.ones(ns, dtype=np.int64)
    tpoints = np.empty(ns, dtype=np.float64)
    tdata = np.empty(ns, dtype=np.float64)
    mdl = np.empty(nd, dtype=np.float64)
    counts = np.zeros(ns, dtype=np.float64)

    for i in range(1, nd + 1):
        odd = 2 * i - 2
        sdatax[odd] = datay[i - 1]
        spoints[odd] = datax[i - 1]
        ref[odd] = datay[i - 1]
        sprd[odd] = 2.0 * sigma[i - 1]
        even = 2 * i - 1
        if even < ns:
            sdatax[even] = 0.5 * (datay[i - 1] + datay[i])
            spoints[even] = 0.5 * (datax[i - 1] + datax[i])
            ref[even] = sdatax[even]
            sprd[even] = 4.0 * (sigma[i - 1] + sigma[i])

    initial_sdatax = sdatax.copy()
    y2 = _spline_y2(spoints, sdatax, ns)
    nx = ns
    ny = ns
    for j in range(nd):
        mdl[j] = _splint(spoints, sdatax, y2, nx, datax[j])
    loglx = -0.5 * _quad_form(icov, datay, mdl)
    logpx = _roughness_prior(spoints, sdatax, y2, nx, ns, smooth, ep)

    initial_trap, _ = _integrate_spline(spoints, sdatax, y2, nx, xmin, xmax)

    pchain_rows = steps // 10 + 1
    pchain = np.empty((pchain_rows, ns + 3), dtype=np.float64)
    chain_rows = steps // 1000 + 1
    chain = np.empty((chain_rows, 4), dtype=np.float64)
    pchain_count = 0
    chain_count = 0

    acc = 1
    scnt = 0
    avr = 0.0
    var = 0.0
    atp = 0.0
    vtp = 0.0
    ka1 = 0.0
    ka2 = 0.0
    ja = 0.0
    kvar1 = 0.0
    kvar2 = 0.0
    jvar = 0.0
    fixed_dim = 0
    ltest = 0

    for mc in range(steps):
        test = 0
        for i in range(ns):
            sdatay[i] = sdatax[i]
            activey[i] = active[i]

        alpha = _ran2(state, iv)
        q = 0.5
        if fixed_dim == 1:
            q = 10.0

        if alpha > q:
            alpha = _ran2(state, iv)
            if alpha < 0.5:
                ny = nx + 1
            else:
                ny = nx - 1

            if ny < nx:
                if ny > 1 and ny <= ns:
                    while True:
                        i = int(_ran2(state, iv) * ns)
                        if active[i] != 0:
                            break
                    activey[i] = 0
                else:
                    test = 1
            else:
                if ny >= 1 and ny < ns:
                    while True:
                        i = int(_ran2(state, iv) * ns)
                        if active[i] == 0:
                            break
                    activey[i] = 1
                    sdatay[i] = ref[i] + 10.0 * sprd[i] * (1.0 - 2.0 * _ran2(state, iv))
                else:
                    test = 1
        else:
            ny = nx
            alpha = _ran2(state, iv)
            for ii in range(ns):
                if alpha > 0.8:
                    sdatay[ii] += sprd[ii] * _gasdev(state, iv, gas_state)
                elif alpha > 0.5:
                    sdatay[ii] += sprd[ii] * 1.0e-1 * _gasdev(state, iv, gas_state)
                elif alpha > 0.2:
                    sdatay[ii] += sprd[ii] * 1.0e-2 * _gasdev(state, iv, gas_state)
                else:
                    sdatay[ii] += sprd[ii] * 1.0e-3 * _gasdev(state, iv, gas_state)

        if ny >= 1 and ny <= ns:
            for ii in range(ns):
                if activey[ii] == 1:
                    if sdatay[ii] > ref[ii] + 10.0 * sprd[ii]:
                        test = 1
                    if sdatay[ii] < ref[ii] - 10.0 * sprd[ii]:
                        test = 1
        else:
            test = 1

        logly = 0.0
        logpy = 0.0
        if test == 0:
            if ltest == 0:
                m = 0
                for ii in range(ns):
                    if activey[ii] == 1:
                        tpoints[m] = spoints[ii]
                        tdata[m] = sdatay[ii]
                        m += 1
                y2t = _spline_y2(tpoints, tdata, ny)
                for j in range(nd):
                    mdl[j] = _splint(tpoints, tdata, y2t, ny, datax[j])
                logly = -0.5 * _quad_form(icov, datay, mdl)
                logpy = _roughness_prior(tpoints, tdata, y2t, ny, ns, smooth, ep)
            else:
                logly = 0.0

        h = logly - loglx + logpy - logpx
        alpha = np.log(_ran2(state, iv))
        if h > alpha and test == 0:
            acc += 1
            loglx = logly
            logpx = logpy
            nx = ny
            for i in range(ns):
                sdatax[i] = sdatay[i]
                active[i] = activey[i]

        if mc % 10 == 0:
            pchain[pchain_count, 0] = mc // 10
            pchain[pchain_count, 1] = loglx
            pchain[pchain_count, 2] = nx
            for i in range(ns):
                pchain[pchain_count, i + 3] = sdatax[i]
            pchain_count += 1

        if mc > steps // 2:
            counts[nx - 1] += 1.0

        if mc % 1000 == 0:
            m = 0
            for ii in range(ns):
                if active[ii] == 1:
                    tpoints[m] = spoints[ii]
                    tdata[m] = sdatax[ii]
                    m += 1
            y2t = _spline_y2(tpoints, tdata, nx)
            trap, trp = _integrate_spline(tpoints, tdata, y2t, nx, xmin, xmax)
            chain[chain_count, 0] = mc // 1000
            chain[chain_count, 1] = loglx
            chain[chain_count, 2] = nx
            chain[chain_count, 3] = trap
            chain_count += 1
            scnt += 1
            avr += trap
            var += trap * trap
            atp += trp
            vtp += trp * trp
            e0 = _splint(tpoints, tdata, y2t, nx, xmin)
            e1 = _splint(tpoints, tdata, y2t, nx, xmax)
            ka1 += e1 - trap
            ka2 += trap - e0
            ja += e1 - e0
            kvar1 += (e1 - trap) * (e1 - trap)
            kvar2 += (trap - e0) * (trap - e0)
            jvar += (e1 - e0) * (e1 - e0)

    m = 0
    for ii in range(ns):
        if active[ii] == 1:
            tpoints[m] = spoints[ii]
            tdata[m] = sdatax[ii]
            m += 1
    y2t = _spline_y2(tpoints, tdata, nx)
    fit = np.empty((1000, 2), dtype=np.float64)
    for i in range(1, 1001):
        x3 = xmin + (xmax - xmin) * i / 1000.0
        fit[i - 1, 0] = x3
        fit[i - 1, 1] = _splint(tpoints, tdata, y2t, nx, x3)

    summary = np.empty(10, dtype=np.float64)
    summary[0] = avr / scnt
    summary[1] = np.sqrt(var / scnt - summary[0] * summary[0])
    summary[2] = atp / scnt
    summary[3] = np.sqrt(vtp / scnt - summary[2] * summary[2])
    summary[4] = ka1 / scnt
    summary[5] = np.sqrt(kvar1 / scnt - summary[4] * summary[4])
    summary[6] = ka2 / scnt
    summary[7] = np.sqrt(kvar2 / scnt - summary[6] * summary[6])
    summary[8] = ja / scnt
    summary[9] = np.sqrt(jvar / scnt - summary[8] * summary[8])

    return spoints, initial_sdatax, sdatax, ref, sprd, initial_trap, pchain[:pchain_count], chain[:chain_count], fit, summary


def read_chain(path: Path):
    '''Reads PTMCMC chain output from a text file using np.loadtxt.
    Expects the first row to contain beta values and remaining rows to be log-likelihood samples.
    Returns:
        datax: log(beta) reversed so hottest chain comes first
        chains: transposed log-likelihood samples aligned with datax
        nd: number of temperatures
        nl: number of samples'''
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError("input file must contain a beta row and at least one sample row")
    nd = arr.shape[1] - 1
    nl = arr.shape[0] - 1
    betas = arr[0, 1:]
    logls = arr[1:, 1:]
    datax = np.log(betas[::-1]).copy()
    chains = logls[:, ::-1].T.copy()
    return datax, chains, nd, nl


def write_columns(path: str, values: np.ndarray, fmt: str = "%.12g") -> None:
    '''Writes a NumPy array to a text file using np.savetxt.
    Used for outputting rawintegrand.dat, integrand.dat, boost.dat, etc.'''
    np.savetxt(path, values, fmt=fmt)


def main() -> None:
    '''Parses command-line arguments: datafile, --steps, --bootstrap-steps, --seed.
    Reads the input chain, computes initial integrand statistics, and writes rawintegrand.dat.
    Runs _bootstrap_covariance to estimate a covariance matrix and prints diagnostics.
    Computes integrand.dat, trapezoidal evidence estimates, and statistical errors.
    Calls _run_rjmcmc to perform spline RJMCMC and save chain outputs.
    Writes output files: boost.dat, initialfit.dat, slopes.dat, pchain.dat, chain.dat, fit.dat.
    Prints final evidence and KL/J metric summaries.'''
    parser = argparse.ArgumentParser(description="Numba/SciPy version of thermo_error.c")
    parser.add_argument("datafile", type=Path)
    parser.add_argument("--steps", type=int, default=1_000_000, help="RJMCMC steps")
    parser.add_argument("--bootstrap-steps", type=int, default=100_000, help="threshold bootstrap replicates")
    parser.add_argument("--seed", type=int, default=-8325257)
    args = parser.parse_args()

    smooth = 8.0
    ep = 0.0001

    datax, chains, nd, nl = read_chain(args.datafile)
    print(f"Inferred {nd} temperatures and {nl} samples from {args.datafile}")
    print(f"\n Hottest chain beta = {np.exp(datax[0]):e}  log(beta) = {datax[0]:f}")

    avinitial = chains.mean(axis=1)
    sinitial = np.sqrt((chains * chains).mean(axis=1) - avinitial * avinitial) / np.sqrt(nl)
    output_offset = avinitial[-1]
    base = 0.0
    write_columns("rawintegrand.dat", np.column_stack((datax, avinitial, sinitial)), fmt="%.12g")

    print("\nStarting bootstrap estimate of correlation matrix")
    state, iv = _init_rng(args.seed)
    cov, datay, cc, vc, cmin, icm, cmax, icx, cycles, shrink = _bootstrap_covariance(
        chains, args.bootstrap_steps, state, iv
    )
    print(f"average number of chunks = {int(cc)}  spread = {vc:f}")
    print(f"least chunks = {cmin} for chain {icm}")
    print(f"most  chunks = {cmax} for chain {icx}")
    print(f"chunks = {cycles}  samples per chunk = {nl / cycles:f}")
    print(f"shrink = {shrink:f}\n")

    sigma = np.sqrt(np.diag(cov))
    avg_corr = np.sum(np.diag(cov, k=1) / (sigma[:-1] * sigma[1:])) / nd
    print(f"Average correlation of neighboring chains = {avg_corr:f}\n")

    eigvals, _ = linalg.eigh(cov, lower=False)
    if np.any(eigvals < 0.0):
        raise RuntimeError("Correlation matrix not positive definite")
    icov = linalg.inv(cov)

    write_columns("integrand.dat", np.column_stack((datax, datay + output_offset, sigma)), fmt="%.12g")

    trap_beta = np.trapz(datay, x=np.exp(datax))
    print(f"Trapezoid integrand in beta {trap_beta + base:f}")

    stat_var = 0.0
    for j in range(1, nd - 1):
        b0 = 0.0 if j == 1 else datax[j - 1] - datax[j - 2]
        b1 = datax[j] - datax[j - 1]
        s0 = sigma[j - 1] * np.exp(datax[j - 1])
        s1 = sigma[j] * np.exp(datax[j])
        stat_var += 0.25 * (((s1 * s1) + (s0 * s0)) * b1 * b1 + 2.0 * s0 * s0 * b1 * b0)

    weighted = np.exp(datax) * datay
    trap_ln = np.trapz(weighted, x=datax)
    a2 = trap_ln / (1.0 - np.exp(datax[0]))
    shifted = np.exp(datax) * (datay - a2)
    trap_shifted = np.trapz(shifted, x=datax)
    if nd % 2 == 0:
        s0 = 0.5 * (datax[1] - datax[0]) * (shifted[1] + shifted[0])
        for j in range(1, nd // 2):
            s0 += 0.5 * (datax[2 * j + 1] - datax[2 * j - 1]) * (shifted[2 * j + 1] + shifted[2 * j - 1])
    else:
        s0 = 0.0
        for j in range(1, nd // 2 + 1):
            s0 += 0.5 * (datax[2 * j] - datax[2 * j - 2]) * (shifted[2 * j] + shifted[2 * j - 2])
    print(
        "Trapezoid integrand in ln(beta) "
        f"{trap_shifted + base + a2 * (1.0 - np.exp(datax[0])):f}, "
        f"statistical error {np.sqrt(stat_var):f},  discretization error {abs(trap_shifted - s0) / 4.0:f}"
    )

    spoints, initial_sdatax, final_sdatax, ref, sprd, initial_trap, pchain, chain, fit, summary = _run_rjmcmc(
        datax, datay, sigma, icov, base, args.steps, smooth, ep, state, iv
    )

    boost = []
    for i in range(1, nd + 1):
        odd = 2 * i - 2
        boost.append((spoints[odd], initial_sdatax[odd] + output_offset, sprd[odd] / 2.0))
        even = 2 * i - 1
        if even < 2 * nd - 1:
            boost.append((spoints[even], initial_sdatax[even] + output_offset, sprd[even] / 2.0))
    write_columns("boost.dat", np.array(boost), fmt="%.12g")

    # The initial fit is reconstructed from the initial spline control points.
    y2_initial = _spline_y2(spoints, initial_sdatax, 2 * nd - 1)
    xs = np.linspace(datax[0], datax[-1], 1001)[1:]
    initialfit = np.array([(x, _splint(spoints, initial_sdatax, y2_initial, 2 * nd - 1, x) + output_offset) for x in xs])
    write_columns("initialfit.dat", initialfit, fmt="%.12g")
    slopes = []
    for i in range(1, 2 * nd - 2):
        x = spoints[i]
        dx = spoints[i] - spoints[i - 1]
        y = _splint(spoints, initial_sdatax, y2_initial, 2 * nd - 1, x)
        y3 = _splint(spoints, initial_sdatax, y2_initial, 2 * nd - 1, x - ep)
        y4 = _splint(spoints, initial_sdatax, y2_initial, 2 * nd - 1, x + ep)
        secd = (y3 + y4 - 2.0 * y) / (ep * ep)
        fird = (y4 - y3) / (2.0 * ep)
        slopes.append((x, fird, (smooth / (2 * nd - 1)) * (secd * secd * dx * dx) / (fird * fird)))
    write_columns("slopes.dat", np.array(slopes), fmt="%.12g")
    write_columns("pchain.dat", pchain, fmt="%.12g")
    write_columns("chain.dat", chain, fmt="%.12g")
    fit[:, 1] += output_offset
    write_columns("fit.dat", fit, fmt="%.12g")

    print(f"Initial Spline Evidence Estimate = {initial_trap + base:f}")
    print(f"\n Average Evidence using ln(beta) {summary[0] + base:f}")
    print(f" Sigma {summary[1]:f}\n")
    print(f"\n Average Evidence using beta {summary[2] + base:f}")
    print(f" Sigma {summary[3]:f}\n")
    print(f"\n Average (Base 2) KL post-prior {1.442695041 * summary[4]:f}")
    print(f" Sigma {1.442695041 * summary[5]:f}\n")
    print(f"\n Average (Base 2) KL prior-post {1.442695041 * summary[6]:f}")
    print(f" Sigma {1.442695041 * summary[7]:f}\n")
    print(f"\n Average (Base 2) J metric {1.442695041 * summary[8]:f}")
    print(f" Sigma {1.442695041 * summary[9]:f}\n")


if __name__ == "__main__":
    main()
