"""Refit of the CORALIE radial velocities of HD 127195 with two Keplerian orbits.

The model follows the kima fit of Figueira et al. (2026, A&A 712, A143): two Keplerian orbits, a
systemic velocity, a linear trend, an offset and a jitter term for each CORALIE setup, the priors
of their Table C.1, and zero likelihood for configurations that fail the angular momentum deficit
(AMD) stability test used by kima (collision condition of Laskar & Petit 2017 and resonance-overlap
condition of Petit et al. 2017). Mean anomalies and the trend refer to the middle of the time span,
as in kima. The fit gives the mean longitudes and their uncertainties, which the published table
does not list, and posterior samples with all correlations for the N-body Monte Carlo.

Usage
    python rv_fit.py DATA_DIR            fit, save rv_posterior.npz and print the summary
    python rv_fit.py DATA_DIR --quick    short chains, to test the script

DATA_DIR holds the four HD 127195 files of the CDS catalogue J/A+A/712/A143 (rv1/HD127195_CORALIE98.dat,
_CORALIE07.dat, _CORALIE14.dat and _CORALIE24.dat), the data of the published fit.
"""
import glob
import os
import re
import sys
import time

import emcee
import numpy as np

QUICK = "--quick" in sys.argv
M_STAR = 1.34                            # stellar mass used for the AMD test (Ottoni et al. 2022)
G13 = 0.0666378476025686                 # G^(1/3) in AU, day, solar-mass units (as in kima)
MJUP2MSUN = 0.0009545942339693249
TWO_PI = 2 * np.pi
NAMES = ("P_b", "K_b", "h_b", "k_b", "lam_b", "P_c", "K_c", "h_c", "k_c", "lam_c")   # then jitters, offsets, vsys, slope
PERIOD_RANGE = ((400.0, 700.0), (700.0, 1200.0))   # keeps the planet labels fixed; far from the posterior


# ---------------------------------------------------------------- data
def load_rvs(path):
    """Return t (BJD - 2400000), rv and err (m/s), instrument index per point, and instrument names.
    Points with photon noise above 30 m/s are dropped, as in the discovery paper."""
    files = sorted(glob.glob(os.path.join(path, "HD127195_CORALIE??.dat")))
    if not files:
        sys.exit(f"no HD127195_CORALIE??.dat files found in {path}")
    sets = []
    for f in files:
        d = np.loadtxt(f, usecols=(0, 1, 2), ndmin=2)
        sets.append((d[:, 0].min(), "COR" + re.search(r"CORALIE(\d\d)", f).group(1), d))
    sets.sort()                                                # instruments in time order
    t, rv, err, inst, names = [], [], [], [], []
    for j, (_, name, d) in enumerate(sets):
        keep = d[:, 2] <= 30.0
        t += list(d[keep, 0]); rv += list(d[keep, 1]); err += list(d[keep, 2])
        inst += [j] * int(keep.sum()); names.append(name)
        if (~keep).any():
            print(f"{name}: dropped {int((~keep).sum())} points with photon noise above 30 m/s")
    return np.array(t), np.array(rv), np.array(err), np.array(inst), names


# ---------------------------------------------------------------- model
def kepler_rv(t, P, K, e, w, lam, t_ref):
    """Stellar RV of one planet, K [cos(f + w) + e cos w]; arrays broadcast over (walkers, points)."""
    M = np.mod(TWO_PI * (t - t_ref) / P + lam - w, TWO_PI)
    E = M + e * np.sin(M)
    for _ in range(8):                                          # Newton steps; e < 0.9 here
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    f = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return K * (np.cos(f + w) + e * np.cos(w))


def amd_stable(P, K, e):
    """kima's AMD test for two planets; P, K, e have shape (walkers, 2), inner planet first."""
    m = 4.919e-3 * M_STAR ** (2 / 3) * P ** (1 / 3) * K * np.sqrt(1 - e ** 2) * MJUP2MSUN / M_STAR
    a = G13 * M_STAR ** (1 / 3) * (P / TWO_PI) ** (2 / 3)
    lam_circ = m * np.sqrt(a)
    amd = np.sum(lam_circ * (1 - np.sqrt(1 - e ** 2)), axis=1)
    cx = amd / lam_circ[:, 1]
    y, al, eps = m[:, 0] / m[:, 1], a[:, 0] / a[:, 1], m[:, 0] + m[:, 1]
    lo, hi = np.zeros_like(al), np.ones_like(al)                # root of F(e) (Laskar & Petit 2017, Eq. 35)
    for _ in range(60):
        mid = (lo + hi) / 2
        F = al * mid + y * mid / np.sqrt(al * (1 - mid ** 2) + y ** 2 * mid ** 2) - 1 + al
        lo, hi = np.where(F < 0, mid, lo), np.where(F < 0, hi, mid)
    e1 = (lo + hi) / 2
    e2 = 1 - al - al * e1
    c_coll = y * np.sqrt(al) * (1 - np.sqrt(1 - e1 ** 2)) + 1 - np.sqrt(1 - e2 ** 2)
    r = 0.80199
    g = 81 * (1 - al) ** 5 / (512 * r * eps) - 32 * r * eps / (9 * (1 - al) ** 2)
    c_mmr = g ** 2 * y * np.sqrt(al) / (2 + 2 * y * np.sqrt(al))
    circular_ok = al < 1 - 1.46 * eps ** (2 / 7)
    return circular_ok & (cx < np.minimum(c_coll, c_mmr)) & (amd < lam_circ[:, 0])


class Model:
    def __init__(self, t, rv, err, inst, names):
        self.t, self.rv, self.err, self.inst, self.inst_names = t, rv, err, inst, list(names)
        self.n_inst = len(names)
        self.ref = int(np.argmax(np.bincount(inst)))           # offsets are relative to the largest set
        self.others = [j for j in range(self.n_inst) if j != self.ref]
        self.t_ref = t.min() + 0.5 * (t.max() - t.min())        # kima's M0_epoch and trend epoch
        self.span = rv.max() - rv.min()
        self.slope_sigma = 10 * self.span / (t.max() - t.min())
        self.labels = list(NAMES) + [f"jit_{n}" for n in names] + [f"off_{names[j]}" for j in self.others] + ["vsys", "slope"]
        self.ndim = len(self.labels)
        self.onehot = np.array([[1.0 if i == j else 0.0 for i in inst] for j in range(self.n_inst)])

    def unpack(self, th):
        th = np.atleast_2d(th)
        pl = []
        for k in (0, 1):
            P, K, h, kk, lam = (th[:, 5 * k + i] for i in range(5))
            pl.append((P, K, h ** 2 + kk ** 2, np.arctan2(kk, h), lam))
        j0 = 10
        jit = th[:, j0:j0 + self.n_inst]
        off = np.zeros((th.shape[0], self.n_inst))
        off[:, self.others] = th[:, j0 + self.n_inst:j0 + 2 * self.n_inst - 1]
        vsys, slope = th[:, -2], th[:, -1]
        return pl, jit, off, vsys, slope

    def log_prior(self, th):
        pl, jit, off, vsys, slope = self.unpack(th)
        lp = np.zeros(th.shape[0])
        np.seterr(invalid="ignore", divide="ignore")            # out-of-range proposals give -inf, not warnings
        for k, (P, K, e, w, lam) in enumerate(pl):
            lo, hi = PERIOD_RANGE[k]
            ok = (P > lo) & (P < hi) & (K > 0) & (K < 200) & (e < 0.9)
            lp = np.where(ok, lp - np.log(P) - np.log(K + 5.0), -np.inf)
            ee = np.clip(e, 1e-12, 0.9)
            a, b = 0.867, 3.03                                   # Kumaraswamy prior on e (kima default)
            lp += np.log(a * b) + (a - 1) * np.log(ee) + (b - 1) * np.log(1 - ee ** a)
        lp = np.where(np.all((jit > 5.0) & (jit < 50.0), axis=1), lp, -np.inf)
        lp = np.where(np.all(np.abs(off) < self.span, axis=1), lp, -np.inf)
        lp = np.where((vsys > self.rv.min()) & (vsys < self.rv.max()), lp, -np.inf)
        lp -= 0.5 * (slope / self.slope_sigma) ** 2
        return lp

    def model_rv(self, th, t=None, inst=None):
        pl, jit, off, vsys, slope = self.unpack(th)
        t = self.t if t is None else t
        mu = vsys[:, None] + slope[:, None] * (t - self.t_ref)
        if inst is None:
            mu = mu + off @ self.onehot
        for P, K, e, w, lam in pl:
            mu = mu + kepler_rv(t, P[:, None], K[:, None], e[:, None], w[:, None], lam[:, None], self.t_ref)
        return mu

    def log_prob(self, th):
        th = np.atleast_2d(th)
        lp = self.log_prior(th)
        good = np.isfinite(lp)
        if not good.any():
            return lp
        t2 = th[good]
        pl, jit, off, vsys, slope = self.unpack(t2)
        P = np.stack([pl[0][0], pl[1][0]], axis=1)
        K = np.stack([pl[0][1], pl[1][1]], axis=1)
        e = np.stack([pl[0][2], pl[1][2]], axis=1)
        stable = amd_stable(P, K, e)
        var = self.err ** 2 + (jit @ self.onehot) ** 2
        res = self.rv - self.model_rv(t2)
        ll = -0.5 * np.sum(res ** 2 / var + np.log(TWO_PI * var), axis=1)
        out = lp.copy()
        out[good] = np.where(stable, lp[good] + ll, -np.inf)
        return out


def circular_start(model, periods):
    """Linear least-squares fit with circular orbits at fixed periods: starting K and mean longitudes."""
    t, rv = model.t, model.rv
    cols = []
    for P in periods:
        x = TWO_PI * (t - model.t_ref) / P
        cols += [np.cos(x), np.sin(x)]
    cols += [model.onehot[j] for j in range(model.n_inst)] + [t - model.t_ref]
    A = np.array(cols).T
    w = 1 / np.sqrt(model.err ** 2 + 8.0 ** 2)
    c, *_ = np.linalg.lstsq(A * w[:, None], rv * w, rcond=None)
    out = []
    for k in range(len(periods)):
        a, b = c[2 * k], c[2 * k + 1]                         # K cos(x + lam) = K cos lam cos x - K sin lam sin x
        out.append((np.hypot(a, b), np.mod(np.arctan2(-b, a), TWO_PI)))
    base = c[2 * len(periods):2 * len(periods) + model.n_inst]
    return out, base, c[-1]


def run(model, n_walkers=64, n_steps=30000, burn=10000, thin=20, seed=1):
    rng = np.random.default_rng(seed)
    periods = (534.50, 836.99)
    (kb, lb), (kc, lc) = circular_start(model, periods)[0]
    base, slope0 = circular_start(model, periods)[1:]
    p0 = np.zeros(model.ndim)
    p0[0:5] = (periods[0], kb, 0.05, 0.05, lb)
    p0[5:10] = (periods[1], kc, 0.05, 0.05, lc)
    p0[10:10 + model.n_inst] = 9.0
    vsys0 = base[model.ref]
    p0[10 + model.n_inst:10 + 2 * model.n_inst - 1] = [base[j] - vsys0 for j in model.others]
    p0[-2], p0[-1] = vsys0, slope0
    scale = np.full(model.ndim, 1e-3)
    scale[[0, 5]], scale[[1, 6]], scale[[2, 3, 7, 8]], scale[[4, 9]] = 1.0, 0.5, 0.02, 0.05
    scale[10:] = 0.5
    start = p0 + scale * rng.standard_normal((n_walkers, model.ndim))
    start[:, 10:10 + model.n_inst] = np.clip(start[:, 10:10 + model.n_inst], 5.5, 49)
    sampler = emcee.EnsembleSampler(n_walkers, model.ndim, model.log_prob, vectorize=True,
                                    moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
    np.random.seed(seed)                                      # emcee's moves use numpy's global generator
    sampler.run_mcmc(start, n_steps, progress=False)
    chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
    lnp = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    try:
        tau = sampler.get_autocorr_time(discard=burn, quiet=True)
    except Exception:
        tau = np.full(model.ndim, np.nan)
    return chain, lnp, tau, sampler.acceptance_fraction.mean()


def wrap_near(x, centre):
    return centre + np.mod(x - centre + np.pi, TWO_PI) - np.pi


def summary(model, chain):
    pl, jit, off, vsys, slope = model.unpack(chain)
    q = lambda x: np.percentile(x, [16, 50, 84])
    fmt = lambda x: f"{q(x)[1]:.3f} +{q(x)[2] - q(x)[1]:.3f} -{q(x)[1] - q(x)[0]:.3f}"
    rows = {}
    for k, name in enumerate("bc"):
        P, K, e, w, lam = pl[k]
        lam_c = np.angle(np.mean(np.exp(1j * lam))) % TWO_PI       # circular mean
        lam_w = wrap_near(lam, lam_c)
        w_c = np.angle(np.mean(np.exp(1j * w))) % TWO_PI
        phi = np.mod(lam - w, TWO_PI)
        rows[name] = dict(P=fmt(P), K=fmt(K), e=fmt(e), omega=fmt(np.mod(w, TWO_PI)), phi=fmt(phi),
                          lam=fmt(lam_w), lam_mean=lam_c, lam_std=float(np.std(lam_w)),
                          corr_P_lam=float(np.corrcoef(P, lam_w)[0, 1]))
    return rows


def main():
    path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    t, rv, err, inst, names = load_rvs(path)
    model = Model(t, rv, err, inst, names)
    print(f"{len(t)} RVs from {len(names)} setups ({', '.join(f'{n}: {np.sum(inst == j)}' for j, n in enumerate(names))});"
          f" time span {t.max() - t.min():.1f} d; reference epoch {model.t_ref:.2f}")
    t0 = time.time()
    steps = (6000, 2000) if QUICK else (25000, 5000)
    chain, lnp, tau, acc = run(model, n_steps=steps[0], burn=steps[1])
    print(f"sampling done in {time.time() - t0:.0f} s; acceptance {acc:.2f}; "
          f"autocorrelation times {np.nanmin(tau):.0f}-{np.nanmax(tau):.0f} steps; {len(chain)} samples kept")
    rows = summary(model, chain)
    print("\nPOSTERIOR (median, 16-84%)             Figueira et al. Table 2")
    pub = dict(b=dict(K="13.89 +1.78 -1.83", P="534.50 +2.85 -2.92", e="0.04 +0.05 -0.03", phi="2.66 +2.53 -1.75",
                      omega="3.10 +1.97 -1.93"),
               c=dict(K="11.56 +1.73 -1.82", P="836.99 +9.35 -9.67", e="0.05 +0.05 -0.04", phi="2.83 +1.51 -1.39",
                      omega="4.12 +1.21 -2.43"))
    for k in "bc":
        for par in ("P", "K", "e", "phi", "omega"):
            print(f"  {par:5s} {k}: {rows[k][par]:28s} {pub[k][par]}")
        print(f"  lam   {k}: {rows[k]['lam']:28s} (mean longitude at the reference epoch, radians)")
        print(f"            circular mean {rows[k]['lam_mean']:.3f}, std {rows[k]['lam_std']:.3f} rad;"
              f" correlation with P: {rows[k]['corr_P_lam']:+.2f}")
    pl, jit, off, vsys, slope = model.unpack(chain)
    for j, n in enumerate(names):
        print(f"  jitter {n}: {np.median(jit[:, j]):.2f} m/s")
    print(f"  slope: {np.median(slope) * 365.25:.2f} m/s per year")
    best = chain[np.argmax(lnp)]
    res = model.rv - model.model_rv(best)[0]
    print(f"  rms of residuals at the highest-probability sample: {np.std(res):.2f} m/s")
    np.savez("rv_posterior.npz", chain=chain, lnp=lnp, labels=np.array(model.labels), t_ref=model.t_ref,
             inst_names=np.array(names), M_star=M_STAR)
    print("\nsaved rv_posterior.npz")


if __name__ == "__main__":
    main()
