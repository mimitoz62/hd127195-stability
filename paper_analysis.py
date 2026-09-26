"""Reproduces every number and Figure 1 of the Research Note on the HD 127195 planetary system,
"N-body Stability Constraints on the Two Giant Planets of the Red Giant HD 127195".

Usage
    python paper_analysis.py            full analysis (uses all but one CPU core)
    python paper_analysis.py --quick    small grids, to test the pipeline in about two minutes
    python paper_analysis.py --figure   redraw Figure 1 from results.json and maps.npz
    WORKERS=4 python paper_analysis.py  set the number of worker processes

Outputs (in the working directory)
    hd127195_stability.pdf, hd127195_stability_preview.png   Figure 1
    results.json                                             all summary numbers
    maps.npz                                                 MEGNO and close-encounter maps
The script also prints every number quoted in the paper, labelled [N1] to [N22].
Chaotic integrations are sensitive to floating-point rounding, so counts can differ by a draw
or two between machines.

Inputs
    Period, eccentricity and argument of periastron from Figueira et al. (2026, A&A 712, A143),
    Table 2 (angles in radians), and minimum masses from their text. Mean longitudes are read
    from their Fig. C.2, and the mass uncertainties follow from the semi-amplitude errors.
    Stellar mass and distance from Ottoni et al. (2022, A&A 657, A87).
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import rebound

QUICK = "--quick" in sys.argv          # small grids, only for testing the script

# ---------------------------------------------------------------- inputs
MJUP = 9.547919e-4                     # Jupiter mass in solar masses
DAY = 1 / 365.25                       # one day in years
DIST_PC = 84.7                         # distance to HD 127195 (Table A.2, CASCADES) in parsecs
BEST = dict(M_star=1.34,
            P=(534.50, 836.99),        # days
            m=(0.67, 0.65),            # m sin i, Jupiter masses
            e=(0.04, 0.05),
            omega=(3.10, 4.12),        # radians (Table 2 medians; its degree label is a typo)
            lam=(4.31, 1.00))          # mean longitude phi + omega, radians, read from Fig. C.2
ERR = dict(M_star=0.06,
           P=((2.85 + 2.92) / 2, (9.35 + 9.67) / 2),
           m=(0.09, 0.10),             # from the semi-amplitude errors (13% and 15%)
           e=((0.05 + 0.03) / 2, (0.05 + 0.04) / 2),
           omega=(1.95, 1.82),
           lam=(0.15, 0.15))           # estimated from the width of the Fig. C.2 ridges
N_ORBITS = 20_000                      # chaos test length, in orbits of planet b
REGULAR_BELOW = 2.2                    # MEGNO below this = regular


# ---------------------------------------------------------------- simulation tools
def make_sim(p, mass_factor=1.0, dt_frac=1 / 30, integrator="whfast"):
    """Star + two coplanar planets. mass_factor = 1/sin(i) scales both planet masses."""
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.add(m=p["M_star"])
    for k in (0, 1):
        sim.add(m=p["m"][k] * MJUP * mass_factor, P=p["P"][k] * DAY, e=p["e"][k],
                omega=p["omega"][k], l=p["lam"][k])
    sim.move_to_com()
    sim.integrator = integrator
    if integrator == "whfast":
        sim.dt = sim.particles[1].P * dt_frac
    hill = [q.a * (q.m / (3 * p["M_star"])) ** (1 / 3) for q in sim.particles[1:]]
    sim.exit_min_distance = min(hill)                                   # "close encounter"
    sim.exit_max_distance = 100 * max(q.a for q in sim.particles[1:])  # "ejected"
    return sim


def evaluate(job):
    """job = (params, mass_factor, n_orbits, dt_frac, integrator, megno_seed) -> (MEGNO, close_encounter)"""
    p, f, n, dt_frac, integrator, seed = job
    sim = make_sim(p, f, dt_frac, integrator)
    sim.init_megno(seed=seed)
    try:
        sim.integrate(n * sim.particles[1].P, exact_finish_time=0)
        return min(sim.megno(), 8.0), False
    except (rebound.Encounter, rebound.Escape):
        return 8.0, True


def survive(job):
    """Long run without MEGNO. job = (params, n_orbits) -> (survived, relative energy error, years)."""
    p, n_orbits = job
    sim = make_sim(p)
    e0 = sim.energy()
    years = n_orbits * sim.particles[1].P
    try:
        sim.integrate(years, exact_finish_time=0)
        return True, abs((sim.energy() - e0) / e0), years
    except (rebound.Encounter, rebound.Escape):
        return False, float("nan"), years


def draw(rng):
    """One random system within the uncertainties (same draw order as notebook 03)."""
    m_star = abs(rng.normal(BEST["M_star"], ERR["M_star"]))
    P, m, e, w, lam = [], [], [], [], []
    for k in (0, 1):
        P.append(rng.normal(BEST["P"][k], ERR["P"][k]))
        m.append(abs(rng.normal(BEST["m"][k], ERR["m"][k])))
        e.append(float(np.clip(rng.normal(BEST["e"][k], ERR["e"][k]), 0, 0.9)))
        w.append(rng.normal(BEST["omega"][k], ERR["omega"][k]))
        lam.append(rng.normal(BEST["lam"][k], ERR["lam"][k]))
    return dict(M_star=m_star, P=tuple(P), m=tuple(m), e=tuple(e), omega=tuple(w), lam=tuple(lam))


def with_(p, **changes):
    q = dict(p); q.update(changes); return q


def job(p, f=1.0, n=N_ORBITS, dt_frac=1 / 30, integrator="whfast", seed=1):
    return (p, f, n, dt_frac, integrator, seed)


def amd_value(eb, ec, alpha, gamma):
    """Angular momentum deficit of coplanar orbits, in units of planet c's circular angular momentum."""
    return gamma * np.sqrt(alpha) * (1 - np.sqrt(1 - eb ** 2)) + (1 - np.sqrt(1 - ec ** 2))


def amd_critical(alpha, gamma):
    """Laskar & Petit (2017) collision criterion: smallest AMD that lets the two orbits cross."""
    e1 = np.linspace(0.0, (1 - alpha) / alpha, 200001)
    e2 = 1 - alpha - alpha * e1
    ok = (e2 >= 0) & (e2 < 1)
    return float(np.min(gamma * np.sqrt(alpha) * (1 - np.sqrt(1 - e1[ok] ** 2)) + 1 - np.sqrt(1 - e2[ok] ** 2)))


def alpha_gamma(p):
    return (p["P"][0] / p["P"][1]) ** (2 / 3), p["m"][0] / p["m"][1]


def bisect(fun, lo, hi):
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fun(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


# ---------------------------------------------------------------- the analysis
def main():
    t0 = time.time()
    workers = int(os.environ.get("WORKERS", 0)) or max(1, (os.cpu_count() or 2) - 1)
    print(f"Using {workers} cores. Quick mode: {QUICK}")
    R = {}
    n_mc = 40 if QUICK else 200
    n_full, n_angle = (20, 10) if QUICK else (100, 40)
    F = [1.0, 2.0, 3.0, 5.0, 8.0] if QUICK else \
        [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0]
    NE, NC = (10, 10) if QUICK else (40, 20)
    NPR, NEC = (11, 8) if QUICK else (41, 31)

    # --- build every job first, run them all in parallel, then read the results back
    jobs, tags = [], []

    def add(tag, j):
        tags.append(tag); jobs.append(j)

    add(("best", "P/30"), job(BEST))
    add(("best", "P/60"), job(BEST, dt_frac=1 / 60))
    add(("best", "P/100"), job(BEST, dt_frac=1 / 100))
    add(("best", "IAS15"), job(BEST, integrator="ias15"))
    add(("best", "1e5 orbits"), job(BEST, n=100_000))
    for s in (2, 3, 4):
        add(("best", f"seed {s}"), job(BEST, seed=s))
    for db in (-0.3, 0.0, 0.3):
        for dc in (-0.3, 0.0, 0.3):
            add(("lamshift", (db, dc)), job(with_(BEST, lam=(BEST["lam"][0] + db, BEST["lam"][1] + dc))))
    rng = np.random.default_rng(42)
    mc_params = [draw(rng) for _ in range(n_mc)]
    for k, p in enumerate(mc_params):
        add(("mc", k), job(p))
    for f in F:
        add(("incl_best", f), job(BEST, f=f))
        add(("incl_best_1e5", f), job(BEST, f=f, n=100_000))
        rng = np.random.default_rng(7)
        for k in range(n_angle):
            lam = tuple(rng.normal(BEST["lam"][i], ERR["lam"][i]) for i in (0, 1))
            om = tuple(rng.normal(BEST["omega"][i], ERR["omega"][i]) for i in (0, 1))
            add(("incl_angle", f, k), job(with_(BEST, lam=lam, omega=om), f=f))
        rng = np.random.default_rng(42)
        for k in range(n_full):
            add(("incl_full", f, k), job(draw(rng), f=f))
    egrid = np.linspace(0, 0.3, NE)
    for a in range(NE):
        for b in range(NE):
            add(("emap", a, b), job(with_(BEST, e=(egrid[a], egrid[b]))))
    cgrid = np.linspace(0, 0.3, NC)
    shifts = [(0.0, 0.0), (0.3, 0.0), (-0.3, 0.0), (0.0, 0.3), (0.0, -0.3)]
    for sh in shifts:
        lam = (BEST["lam"][0] + sh[0], BEST["lam"][1] + sh[1])
        for a in range(NC):
            for b in range(NC):
                add(("shift", sh, a, b), job(with_(BEST, lam=lam, e=(cgrid[a], cgrid[b]))))
    prgrid = np.linspace(1.50, 1.66, NPR)
    ecgrid = np.linspace(0, 0.3, NEC)
    for a, pr in enumerate(prgrid):
        for b, ec in enumerate(ecgrid):
            add(("prmap", a, b), job(with_(BEST, P=(BEST["P"][0], BEST["P"][0] * pr), e=(BEST["e"][0], ec))))

    n_long = 100_000 if QUICK else 10_000_000        # long run of the best fit
    n_chaos_long = 50_000 if QUICK else 1_000_000    # long runs of the chaotic Monte Carlo draws
    print(f"{len(jobs)} simulations to run, plus one long survival run...", flush=True)
    with Pool(workers) as pool:
        long_run = pool.apply_async(survive, ((BEST, n_long),))   # runs alongside the rest
        out = pool.map(evaluate, jobs, chunksize=4)
        res = dict(zip(tags, out))
        print(f"simulations done in {time.time() - t0:.0f} s", flush=True)
        # are the chaotic draws actually unstable? follow each one much longer, without MEGNO
        chaotic = [k for k in range(n_mc) if res[("mc", k)][0] >= REGULAR_BELOW and not res[("mc", k)][1]]
        print(f"following the {len(chaotic)} chaotic draws for {n_chaos_long:,} orbits each...", flush=True)
        chaos_long = pool.map(survive, [(mc_params[k], n_chaos_long) for k in chaotic], chunksize=1)
        surv = long_run.get()
    print(f"long runs done in {time.time() - t0:.0f} s", flush=True)

    # --- best fit and robustness
    R["best_megno"] = {k[1]: v[0] for k, v in res.items() if k[0] == "best"}
    R["lamshift_megno_range"] = [min(v[0] for k, v in res.items() if k[0] == "lamshift"),
                                 max(v[0] for k, v in res.items() if k[0] == "lamshift")]
    R["long_survived"], R["long_energy_error"], R["long_years"] = surv
    R["long_orbits"] = n_long
    al, ga = alpha_gamma(BEST)
    sim = make_sim(BEST)
    ab, ac = sim.particles[1].a, sim.particles[2].a
    mb, mc = sim.particles[1].m, sim.particles[2].m
    rh = ((mb + mc) / (3 * BEST["M_star"])) ** (1 / 3) * (ab + ac) / 2
    R.update(a_b=ab, a_c=ac, period_ratio=BEST["P"][1] / BEST["P"][0], mutual_hill=(ac - ab) / rh)
    ccrit = amd_critical(al, ga)
    R["amd_limit_e_equal"] = bisect(lambda e: amd_value(e, e, al, ga) - ccrit, 0, 0.9)
    R["amd_limit_e_b_alone"] = bisect(lambda e: amd_value(e, 0, al, ga) - ccrit, 0, 0.9)
    R["amd_limit_e_c_alone"] = bisect(lambda e: amd_value(0, e, al, ga) - ccrit, 0, 0.9)

    # --- Monte Carlo at the minimum masses
    mc = [res[("mc", k)] for k in range(n_mc)]
    reg = np.array([s < REGULAR_BELOW for s, _ in mc]); enc = np.array([x for _, x in mc])
    amd_ok = np.array([amd_value(p["e"][0], p["e"][1], *alpha_gamma(p)) < amd_critical(*alpha_gamma(p)) for p in mc_params])
    pr = np.array([p["P"][1] / p["P"][0] for p in mc_params])
    R.update(mc_n=n_mc, mc_counts=[int(reg.sum()), int((~reg & ~enc).sum()), int(enc.sum())],
             mc_chaotic_long_orbits=n_chaos_long, mc_chaotic_long_lost=int(sum(not s for s, _, _ in chaos_long)),
             n_angle_draws=n_angle, n_full_draws=n_full)
    R.update(mc_regular=reg.mean(), mc_chaotic_no_encounter=(~reg & ~enc).mean(), mc_encounter=enc.mean(),
             mc_amd_stable=amd_ok.mean(), mc_regular_pr_below_156=reg[pr < 1.56].mean(),
             mc_regular_pr_above_158=reg[pr > 1.58].mean())

    # --- inclination scan
    incl = []
    for f in F:
        ang = [res[("incl_angle", f, k)] for k in range(n_angle)]
        full = [res[("incl_full", f, k)] for k in range(n_full)]
        incl.append(dict(f=f, i_deg=float(np.degrees(np.arcsin(1 / f))),
                         best=res[("incl_best", f)][0], best_enc=res[("incl_best", f)][1],
                         best_1e5=res[("incl_best_1e5", f)][0], best_1e5_enc=res[("incl_best_1e5", f)][1],
                         angle_regular=np.mean([s < REGULAR_BELOW for s, _ in ang]),
                         angle_encounter=np.mean([x for _, x in ang]),
                         full_regular=np.mean([s < REGULAR_BELOW for s, _ in full]),
                         full_no_encounter=1 - np.mean([x for _, x in full])))
    R["inclination_scan"] = incl
    unstable_best = [r["f"] for r in incl if r["best_enc"] or r["best_1e5_enc"]]
    R["best_fit_unstable_from_f"] = min(unstable_best) if unstable_best else None
    # isotropic prior on i, weighted by the fraction of draws free of close encounters
    inc = np.radians([r["i_deg"] for r in incl])
    w = np.array([r["full_no_encounter"] for r in incl])
    ii = np.radians(np.linspace(0.5, 90, 4000))
    wi = np.interp(np.sin(ii), np.sin(inc)[::-1], w[::-1], left=0.0)
    post = np.sin(ii) * wi
    cdf = np.cumsum(post) / post.sum()
    R["i_min_95"] = float(np.degrees(ii[np.searchsorted(cdf, 0.05)]))
    R["i_min_99"] = float(np.degrees(ii[np.searchsorted(cdf, 0.01)]))
    R["i_min_95_prior_only"] = float(np.degrees(np.arccos(0.95)))
    R["mass_factor_max_95"] = 1 / np.sin(np.radians(R["i_min_95"]))
    R["true_mass_max_95"] = [BEST["m"][k] * R["mass_factor_max_95"] for k in (0, 1)]
    # astrometric signature alpha = (m/M) a / d, in micro-arcseconds, at minimum mass and at the 95% limit
    sig = lambda k, fac: BEST["m"][k] * MJUP * fac / BEST["M_star"] * [ab, ac][k] / DIST_PC * 1e6
    R["astrometric_uas_min"] = [sig(0, 1), sig(1, 1)]
    R["astrometric_uas_max95"] = [sig(0, R["mass_factor_max_95"]), sig(1, R["mass_factor_max_95"])]

    # --- eccentricity map
    Y = np.array([[res[("emap", a, b)][0] for a in range(NE)] for b in range(NE)])   # rows e_c, cols e_b
    E = np.array([[res[("emap", a, b)][1] for a in range(NE)] for b in range(NE)])
    EB, EC = np.meshgrid(egrid, egrid)
    regmap = Y < REGULAR_BELOW
    inside = amd_value(EB, EC, al, ga) < ccrit
    R["emap_max_ec_regular"] = float(EC[regmap].max())
    R["emap_amd_cells_regular"] = float(regmap[inside].mean())
    R["emap_amd_cells_encounter"] = float(E[inside].mean())
    R["emap_regular_below_010"] = float(regmap[(EB <= 0.1001) & (EC <= 0.1001)].mean())
    R["emap_regular_010_015"] = float(regmap[(np.maximum(EB, EC) > 0.1001) & (np.maximum(EB, EC) <= 0.1501)].mean())
    R["shift_regular_fraction"] = {f"{sh[0]:+.1f},{sh[1]:+.1f}": float(np.mean(
        [res[("shift", sh, a, b)][0] < REGULAR_BELOW for a in range(NC) for b in range(NC)])) for sh in shifts}

    # --- period-ratio map
    YP = np.array([[res[("prmap", a, b)][0] for a in range(NPR)] for b in range(NEC)])
    EP = np.array([[res[("prmap", a, b)][1] for a in range(NPR)] for b in range(NEC)])
    PRP, ECP = np.meshgrid(prgrid, ecgrid)
    regp = YP < REGULAR_BELOW
    R["prmap_max_ec_regular_156_165"] = float(ECP[regp & (PRP > 1.559) & (PRP < 1.651)].max())
    R["prmap_max_ec_regular_below_155"] = float(ECP[regp & (PRP < 1.549)].max())

    np.savez("maps.npz", egrid=egrid, Y=Y, E=E, prgrid=prgrid, ecgrid=ecgrid, YP=YP, EP=EP)
    json.dump(R, open("results.json", "w"), indent=1, default=float)
    print_summary(R)
    figure_from_files()
    print("figure saved: hd127195_stability.pdf")
    print(f"all done in {time.time() - t0:.0f} s")


def figure_from_files():
    R = json.load(open("results.json"))
    M = np.load("maps.npz")
    make_figure(M["egrid"], M["Y"], M["E"], M["prgrid"], M["ecgrid"], M["YP"], M["EP"], R)


def make_figure(egrid, Y, E, prgrid, ecgrid, YP, EP, R):
    al, ga = alpha_gamma(BEST)
    ccrit = amd_critical(al, ga)
    incl = R["inclination_scan"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 6.3, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    vmin, vmax = 1.9, 5.0
    fig = plt.figure(figsize=(7.1, 2.55))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.055, 0.32, 1.05], wspace=0.32)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[4])]
    cax = fig.add_subplot(gs[2])

    def draw_map(ax, x, y, meg, enc):
        mesh = ax.pcolormesh(x, y, np.clip(meg, vmin, vmax), vmin=vmin, vmax=vmax, cmap="viridis",
                             shading="nearest", rasterized=True)
        ax.pcolormesh(x, y, np.where(enc, 1.0, np.nan), cmap=ListedColormap(["#c8c8c8"]),
                      shading="nearest", rasterized=True)
        return mesh

    def letter(ax, s, right=False):
        ax.text(0.97 if right else 0.03, 0.97, s, transform=ax.transAxes, ha="right" if right else "left",
                va="top", fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))

    star = dict(fmt="*", ms=9, color="white", mec="black", mew=0.6, ecolor="white", elinewidth=0.9, capsize=2)
    ax = axes[0]
    mesh = draw_map(ax, egrid, egrid, Y, E)
    eb, ec = np.meshgrid(np.linspace(0, 0.3, 400), np.linspace(0, 0.3, 400))
    ax.contour(eb, ec, amd_value(eb, ec, al, ga), levels=[ccrit], colors="black", linewidths=1.0, linestyles="--")
    ax.plot([], [], ls="--", color="black", lw=1.0, label="AMD limit")
    ax.errorbar(BEST["e"][0], BEST["e"][1], xerr=[[0.03], [0.05]], yerr=[[0.04], [0.05]], label="published", **star)
    ax.set_xlabel(r"$e_b$"); ax.set_ylabel(r"$e_c$")
    ax.set_xlim(-0.004, 0.3); ax.set_ylim(-0.004, 0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    letter(ax, "(a)")

    ax = axes[1]
    draw_map(ax, prgrid, ecgrid, YP, EP)
    high = False
    for p, q in [(3, 2), (14, 9), (11, 7), (8, 5), (13, 8)]:
        if prgrid[0] - 1e-9 <= p / q < prgrid[-1]:
            ax.axvline(p / q, color="white", lw=0.6, ls=":")
            ax.text(p / q, 1.08 if high else 1.01, f"{p}:{q}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=6)
            high = not high
    pr_err = R["period_ratio"] * np.hypot(ERR["P"][0] / BEST["P"][0], ERR["P"][1] / BEST["P"][1])
    ax.errorbar(R["period_ratio"], BEST["e"][1], xerr=pr_err, yerr=[[0.04], [0.05]], **star)
    ax.set_xlabel(r"$P_c/P_b$")
    ax.tick_params(labelleft=False)                      # same e_c axis as panel (a)
    ax.set_xlim(1.498, 1.662); ax.set_ylim(-0.004, 0.3)
    letter(ax, "(b)")

    ax = axes[2]
    inc = [r["i_deg"] for r in incl]
    ax.axvspan(R["i_min_95"], 4, color="0.88", lw=0)
    y_ne = [r["full_no_encounter"] for r in incl]
    y_reg = [r["full_regular"] for r in incl]
    ax.plot(inc, y_ne, "o-", ms=3, lw=1.0, color="#D55E00", label="no close encounter")
    ax.plot(inc, y_reg, "s--", ms=2.5, lw=1.0, color="#0072B2", label="regular")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.6,
              columnspacing=1.0, borderaxespad=0.2)
    ax.set_xscale("log")
    ax.set_xticks([6, 10, 20, 30, 45, 90]); ax.set_xticklabels(["6", "10", "20", "30", "45", "90"])
    ax.minorticks_off()
    ax.set_xlim(95, 5.2); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"inclination $i$ (deg)"); ax.set_ylabel("fraction of draws")
    letter(ax, "(c)", right=True)

    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label(r"MEGNO $\langle Y\rangle$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig("hd127195_stability.pdf", bbox_inches="tight")
    fig.savefig("hd127195_stability_preview.png", dpi=200, bbox_inches="tight")


def print_summary(R):
    pct = lambda x: f"{100 * x:.0f}%"
    b = R["best_megno"]
    inc = {round(r["f"], 2): r for r in R["inclination_scan"]}
    lines = [
        ("N1", "period ratio P_c/P_b", f"{R['period_ratio']:.3f}"),
        ("N2", "separation in mutual Hill radii", f"{R['mutual_hill']:.1f}"),
        ("N3", "best-fit MEGNO (P/30, P/60, P/100, IAS15, 1e5 orbits)",
         ", ".join(f"{b[k]:.3f}" for k in ("P/30", "P/60", "P/100", "IAS15", "1e5 orbits"))),
        ("N4", "MEGNO range with mean longitudes shifted +-0.3 rad",
         f"{R['lamshift_megno_range'][0]:.2f} to {R['lamshift_megno_range'][1]:.2f}"),
        ("N5", "long run of the best fit: survived / orbits of b / years / energy error",
         f"{R['long_survived']} / {R['long_orbits']:.0e} / {R['long_years'] / 1e6:.1f} Myr / {R['long_energy_error']:.1e}"),
        ("N6", "Monte Carlo: regular / chaotic, no encounter / close encounter",
         " / ".join(str(c) for c in R["mc_counts"]) + f" of {R['mc_n']}  (" +
         " / ".join(f"{100 * c / R['mc_n']:.1f}%" for c in R["mc_counts"]) + ")"),
        ("N7", "Monte Carlo draws that pass the AMD test", pct(R["mc_amd_stable"])),
        ("N8", "regular fraction for P_c/P_b < 1.56 and > 1.58",
         f"{pct(R['mc_regular_pr_below_156'])} and {pct(R['mc_regular_pr_above_158'])}"),
        ("N9", "map: largest e_c of any regular cell", f"{R['emap_max_ec_regular']:.2f}"),
        ("N10", "map: regular fraction with both e <= 0.10 / with max e in 0.10-0.15",
         f"{pct(R['emap_regular_below_010'])} / {pct(R['emap_regular_010_015'])}"),
        ("N11", "AMD limit: e equal / e_b alone / e_c alone",
         f"{R['amd_limit_e_equal']:.2f} / {R['amd_limit_e_b_alone']:.2f} / {R['amd_limit_e_c_alone']:.2f}"),
        ("N12", "map: AMD-allowed cells that are regular / have close encounters",
         f"{pct(R['emap_amd_cells_regular'])} / {pct(R['emap_amd_cells_encounter'])}"),
        ("N13", "angles only (best P, m, e) at i = 90 deg: regular / close encounter",
         f"{round(inc[1.0]['angle_regular'] * R['n_angle_draws'])} / "
         f"{round(inc[1.0]['angle_encounter'] * R['n_angle_draws'])} of {R['n_angle_draws']}"),
        ("N14", f"no close encounter ({R['n_full_draws']} draws) at i = 90 / 30 / 19.5 / 11.5 deg",
         " / ".join(pct(inc[f]["full_no_encounter"]) for f in (1.0, 2.0, 3.0, 5.0) if f in inc)),
        ("N15", "best fit first unstable at mass factor (i)",
         f"{R['best_fit_unstable_from_f']} ({np.degrees(np.arcsin(1 / R['best_fit_unstable_from_f'])):.1f} deg)"
         if R["best_fit_unstable_from_f"] else "never in scan"),
        ("N16", "inclination lower limit, 95% / 99% (prior alone at 95%)",
         f"{R['i_min_95']:.0f} / {R['i_min_99']:.0f} deg ({R['i_min_95_prior_only']:.0f} deg)"),
        ("N17", "true-mass upper limits at 95% (b, c), Jupiter masses",
         f"{R['true_mass_max_95'][0]:.2f}, {R['true_mass_max_95'][1]:.2f} (x{R['mass_factor_max_95']:.1f})"),
        ("N18", "astrometric signature at minimum mass (b, c), micro-arcsec",
         f"{R['astrometric_uas_min'][0]:.0f}, {R['astrometric_uas_min'][1]:.0f}"),
        ("N19", "astrometric signature at the 95% mass limit (b, c), micro-arcsec",
         f"{R['astrometric_uas_max95'][0]:.0f}, {R['astrometric_uas_max95'][1]:.0f}"),
        ("N20", "regular part of the e_b-e_c plane: unshifted / with a mean longitude shifted by +-0.3 rad",
         f"{pct(R['shift_regular_fraction']['+0.0,+0.0'])} / {pct(min(R['shift_regular_fraction'].values()))} to "
         f"{pct(max(R['shift_regular_fraction'].values()))}"),
        ("N21", "period-ratio map: largest regular e_c for P_c/P_b = 1.56-1.65 / below 1.55",
         f"{R['prmap_max_ec_regular_156_165']:.2f} / {R['prmap_max_ec_regular_below_155']:.2f}"),
        ("N22", "chaotic draws with a close encounter in the long runs (orbits of b)",
         f"{R['mc_chaotic_long_lost']} of {R['mc_counts'][1]} ({R['mc_chaotic_long_orbits']:.0e} orbits)"),
    ]
    print("\nNUMBERS QUOTED IN THE PAPER")
    for tag, label, value in lines:
        print(f"[{tag}] {label}: {value}")


if __name__ == "__main__":
    if "--figure" in sys.argv:
        figure_from_files()
        print("figure redrawn: hd127195_stability.pdf")
    else:
        main()
