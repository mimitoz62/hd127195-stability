"""Checks of the AMD stability test used in the radial-velocity refit (rv_fit.py).

1. Every saved posterior sample passes the test, so the test was active during sampling.
2. If kima is installed (pip install kima), the test gives the same verdict as kima's own compiled
   function (kima.AMDstability.AMD_stable), for the posterior samples and for random two-planet systems.
3. The test needs a stellar mass. The refit uses 1.34 Msun (Ottoni et al. 2022); kima uses 1 Msun unless
   it is set, and the discovery paper does not say which value its fit used. Only the resonance-overlap
   condition depends on it, and it is stricter with 1 Msun, so the posterior with 1 Msun is the part of
   the saved posterior that also passes with 1 Msun. The script prints how much that changes the fitted
   parameters and the outcome of the 200 Monte Carlo draws of paper_analysis.py.

Usage
    python amd_check.py                  uses results/rv_posterior.npz and results/results.json
    python amd_check.py POSTERIOR RESULTS
"""
import json
import sys
from importlib.metadata import version

import numpy as np

import rv_fit

M_STAR = rv_fit.M_STAR                  # 1.34 Msun, used in the refit
M_KIMA = 1.0                            # kima's default stellar mass
TWO_PI = 2 * np.pi
TABLE2 = dict(b=dict(P=(534.50, 2.85, 2.92), K=(13.89, 1.78, 1.83), e=(0.04, 0.05, 0.03),
                     phi=(2.66, 2.53, 1.75), omega=(3.10, 1.97, 1.93)),
              c=dict(P=(836.99, 9.35, 9.67), K=(11.56, 1.73, 1.82), e=(0.05, 0.05, 0.04),
                     phi=(2.83, 1.51, 1.39), omega=(4.12, 1.21, 2.43)))   # Figueira et al. (2026), Table 2


def passes(P, K, e, m_star):
    """The test of rv_fit.py with the stellar mass set to m_star; arrays of shape (n, 2), inner planet first."""
    saved = rv_fit.M_STAR
    rv_fit.M_STAR = m_star
    try:
        return rv_fit.amd_stable(P, K, e)
    finally:
        rv_fit.M_STAR = saved


def passes_kima(P, K, e, m_star):
    """kima's own test; it returns 0 for a stable system. Its parameter order is P, K, phi, e, w."""
    from kima.AMDstability import AMD_stable
    return np.array([AMD_stable([[p[0], k[0], 0.0, x[0], 0.0], [p[1], k[1], 0.0, x[1], 0.0]], m_star) == 0
                     for p, k, x in zip(P, K, e)])


def load(path):
    D = np.load(path)
    col = {str(n): D["chain"][:, i] for i, n in enumerate(D["labels"])}
    pair = lambda b, c: np.stack([b, c], axis=1)
    e = pair(col["h_b"] ** 2 + col["k_b"] ** 2, col["h_c"] ** 2 + col["k_c"] ** 2)
    w = pair(np.arctan2(col["k_b"], col["h_b"]), np.arctan2(col["k_c"], col["h_c"]))
    return pair(col["P_b"], col["P_c"]), pair(col["K_b"], col["K_c"]), e, w, pair(col["lam_b"], col["lam_c"])


def quantiles(P, K, e, w, lam, sel):
    out = {}
    for k, s in enumerate("bc"):
        centre = np.angle(np.mean(np.exp(1j * lam[sel, k]))) % TWO_PI
        lam_w = centre + np.mod(lam[sel, k] - centre + np.pi, TWO_PI) - np.pi
        for name, x in (("P", P[sel, k]), ("K", K[sel, k]), ("e", e[sel, k]), ("omega", np.mod(w[sel, k], TWO_PI)),
                        ("phi", np.mod(lam[sel, k] - w[sel, k], TWO_PI)), ("lam", lam_w)):
            out[(s, name)] = np.percentile(x, [16, 50, 84])
    return out


def outcome(rows):
    """Counts as in the paper: regular, chaotic, early close encounter, survivors of the long runs."""
    n = len(rows)
    reg = sum(r["regular"] for r in rows)
    early = sum(r["early_encounter"] for r in rows)
    surv = sum(bool(r["survived"]) for r in rows if not r["early_encounter"])
    return n, reg, n - reg - early, early, surv


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    post_path, res_path = (args + ["results/rv_posterior.npz", "results/results.json"][len(args):])[:2]
    P, K, e, w, lam = load(post_path)
    n = len(P)
    ok_refit = passes(P, K, e, M_STAR)
    ok_kima_mass = passes(P, K, e, M_KIMA)
    print(f"[A1] posterior samples that pass the test with {M_STAR} Msun: {ok_refit.sum()} of {n}")

    try:
        kima_version = version("kima")
    except Exception:
        kima_version = None
    if kima_version is None:
        print("[A2] kima is not installed (pip install kima); comparison with kima's own code skipped")
    else:
        rng = np.random.default_rng(1)
        m = 200_000
        Pr = np.stack([rng.uniform(500, 570, m), rng.uniform(780, 900, m)], axis=1)
        Kr, er = rng.uniform(5, 25, (m, 2)), rng.uniform(0, 0.25, (m, 2))
        diff_post = [int((passes(P, K, e, ms) != passes_kima(P, K, e, ms)).sum()) for ms in (M_STAR, M_KIMA)]
        diff_rand = [int((passes(Pr, Kr, er, ms) != passes_kima(Pr, Kr, er, ms)).sum()) for ms in (M_STAR, M_KIMA)]
        frac = passes(Pr, Kr, er, M_STAR).mean()
        print(f"[A2] disagreements with kima {kima_version}, with {M_STAR} / {M_KIMA} Msun: posterior samples "
              f"{diff_post[0]} / {diff_post[1]} of {n}; random systems {diff_rand[0]} / {diff_rand[1]} of {m} "
              f"({100 * frac:.0f}% of them stable with {M_STAR} Msun)")

    print(f"[A3] posterior samples that fail the test with {M_KIMA} Msun: {n - ok_kima_mass.sum()} of {n} "
          f"({100 * (1 - ok_kima_mass.mean()):.1f}%)")

    A = quantiles(P, K, e, w, lam, slice(None))
    B = quantiles(P, K, e, w, lam, ok_kima_mass)
    print(f"[A4] median (16-84%) with {M_STAR} Msun | passing with {M_KIMA} Msun | Table 2; "
          f"distance from Table 2 in its sigma ({M_STAR} | {M_KIMA}); change of the median in posterior sigma")
    worst, shift = [0.0, 0.0], 0.0
    for (s, name), q in A.items():
        qb = B[(s, name)]
        line = (f"     {name + '_' + s:8s} {q[1]:8.3f} +{q[2] - q[1]:.3f} -{q[1] - q[0]:.3f} | "
                f"{qb[1]:8.3f} +{qb[2] - qb[1]:.3f} -{qb[1] - qb[0]:.3f}")
        change = (qb[1] - q[1]) / ((q[2] - q[0]) / 2)
        shift = max(shift, abs(change))
        if name in TABLE2[s]:
            med, up, lo = TABLE2[s][name]
            sig = (up + lo) / 2
            d = ((q[1] - med) / sig, (qb[1] - med) / sig)
            worst = [max(worst[0], abs(d[0])), max(worst[1], abs(d[1]))]
            line += f" | {med:7.2f} +{up:.2f} -{lo:.2f}   {d[0]:+.2f} | {d[1]:+.2f}"
        else:
            line += " " * 36
        print(line + f"   {change:+.2f}")
    print(f"     largest distance from Table 2: {worst[0]:.2f} | {worst[1]:.2f} sigma; "
          f"largest change of a median: {shift:.2f} sigma")

    R = json.load(open(res_path))
    mc = R["mc"]
    idx = np.random.default_rng(42).choice(n, len(mc), replace=False)       # the draws of paper_analysis.py
    if not (np.allclose([r["P"][0] for r in mc], P[idx, 0]) and np.allclose([r["e"][1] for r in mc], e[idx, 1])):
        sys.exit("the Monte Carlo draws in the results file do not come from this posterior")
    keep = ok_kima_mass[idx]
    for tag, rows in (("all draws", mc), (f"draws that pass with {M_KIMA} Msun", [r for r, k in zip(mc, keep) if k])):
        N, reg, cha, early, surv = outcome(rows)
        print(f"[A5] {tag}: {N}; regular {reg}, chaotic {cha}, early close encounters {early}; "
              f"close encounters within 146 Myr in {N - surv} of the {N - reg} that are not regular; "
              f"survivors {surv} ({100 * surv / N:.0f}%)")
        emax = np.array([max(r["e"]) for r in rows])
        ratio = np.array([r["P"][1] / r["P"][0] for r in rows])
        alive = np.array([bool(r["survived"]) and not r["early_encounter"] for r in rows])
        part = lambda sel: f"{alive[sel].sum()} of {sel.sum()}"
        print(f"     survivors: both e < 0.05 {part(emax < 0.05)}; larger e >= 0.10 {part(emax >= 0.10)}; "
              f"P_c/P_b < 1.56 {part(ratio < 1.56)}; > 1.58 {part(ratio > 1.58)}")


if __name__ == "__main__":
    main()
