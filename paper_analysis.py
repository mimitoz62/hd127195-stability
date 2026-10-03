"""Reproduces every number and Figure 1 of the Research Note on the HD 127195 planetary system,
"N-body Stability Constraints on the Two Giant Planets of the Red Giant HD 127195".

Usage
    python rv_fit.py RV_FILE               first: refit of the radial velocities -> rv_posterior.npz
    python paper_analysis.py               full analysis (all but one CPU core; several hours)
    python paper_analysis.py --quick       short runs, to test the pipeline in a few minutes
    python paper_analysis.py --figure      redraw Figure 1 from results.json
    WORKERS=4 python paper_analysis.py     set the number of worker processes

Every finished simulation is saved at once (runs_<id>.jsonl), so an interrupted run resumes where
it stopped when started again.

Outputs (in the working directory)
    hd127195_stability.pdf, hd127195_stability_preview.png   Figure 1
    results.json                                             all numbers, including every draw
The script prints every number quoted in the paper, labelled [N1] to [N20]. Chaotic integrations
are sensitive to floating-point rounding, so counts can differ by a draw or two between machines.

Inputs
    Published solution: P, e and omega from Table 2 of Figueira et al. (2026, A&A 712, A143;
    angles in radians) and m sin i from their text. Mean longitudes, and every Monte Carlo draw,
    come from the refit of their radial velocities (rv_fit.py). Stellar mass and distance from
    Ottoni et al. (2022, A&A 657, A87). The angles describe the star's reflex orbit; adding pi to
    both planets' angles only rotates the whole system, so they are used as they are.
"""
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import rebound

QUICK = "--quick" in sys.argv          # short runs, only for testing the script

# ---------------------------------------------------------------- inputs and settings
MJUP = 9.547919e-4                     # Jupiter mass in solar masses
DAY = 1 / 365.25                       # one day in years
AU_PER_YR = 4740.470463533348          # m/s in one AU per year
DIST_PC = 84.7                         # distance to HD 127195 in parsecs
M_STAR, M_STAR_ERR = 1.34, 0.06        # solar masses
PUBLISHED = dict(P=(534.50, 836.99),   # days (Table 2 medians)
                 m=(0.67, 0.65),       # m sin i, Jupiter masses (their text)
                 e=(0.04, 0.05),
                 omega=(3.10, 4.12))   # radians (Table 2; its degree label is a typo)
POSTERIOR = "rv_posterior.npz"
REGULAR_BELOW = 2.2                    # MEGNO below this = regular
N_SHORT = 20_000                       # MEGNO test, orbits of planet b (29,000 yr)
N_LONG = 50_000 if QUICK else 10_000_000          # 14.6 Myr
N_VERY_LONG = 100_000 if QUICK else 100_000_000   # 146 Myr: chaotic draws and the published solution
N_MC = 24 if QUICK else 200            # Monte Carlo draws from the RV posterior
N_INC = 8 if QUICK else 50             # draws per inclination
F = (2.0, 5.0) if QUICK else (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)   # mass factors 1/sin i
LAM_SPAN, N_LAM = 1.0, (7 if QUICK else 31)       # mean-longitude map: +-1 rad around the solution
VERSION = "revision-1"


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


def short_run(p, f=1.0, n=N_SHORT, dt_frac=1 / 30, integrator="whfast", seed=1):
    """MEGNO over n orbits of planet b -> (MEGNO capped at 8, close encounter?)"""
    sim = make_sim(p, f, dt_frac, integrator)
    sim.init_megno(seed=seed)
    try:
        sim.integrate(n * sim.particles[1].P, exact_finish_time=0)
        return min(sim.megno(), 8.0), False
    except (rebound.Encounter, rebound.Escape):
        return 8.0, True


def long_run(p, f=1.0, n=N_LONG):
    """Integration without MEGNO -> (survived?, orbits of b at the encounter or None, energy error)"""
    sim = make_sim(p, f)
    e0 = sim.energy()
    pb = sim.particles[1].P
    try:
        sim.integrate(n * pb, exact_finish_time=0)
        return True, None, abs((sim.energy() - e0) / e0)
    except (rebound.Encounter, rebound.Escape):
        return False, sim.t / pb, None


def run_one(job):
    key, kind, args = job
    return key, (short_run if kind == "short" else long_run)(*args)


def msini(K, P_days, e, m_star):
    """Minimum mass (Jupiter masses) from the RV semi-amplitude, solving the mass function."""
    P = P_days * DAY
    k = K / AU_PER_YR * np.sqrt(1 - e ** 2) * (P / (2 * np.pi * 4 * np.pi ** 2)) ** (1 / 3)
    m = k * m_star ** (2 / 3)
    for _ in range(30):
        m = k * (m_star + m) ** (2 / 3)
    return m / MJUP


def with_(p, **changes):
    q = dict(p); q.update(changes); return q


# ---------------------------------------------------------------- AMD stability (as in kima)
def amd_value(eb, ec, alpha, gamma):
    """Angular momentum deficit of coplanar orbits, in units of planet c's circular angular momentum."""
    return gamma * np.sqrt(alpha) * (1 - np.sqrt(1 - eb ** 2)) + (1 - np.sqrt(1 - ec ** 2))


def amd_limits(p):
    """Critical AMD for orbit crossing (Laskar & Petit 2017) and for resonance overlap (Petit et al.
    2017), the two conditions kima applies, and the semi-major-axis ratio alpha and mass ratio gamma."""
    alpha, gamma = (p["P"][0] / p["P"][1]) ** (2 / 3), p["m"][0] / p["m"][1]
    e1 = np.linspace(0.0, (1 - alpha) / alpha, 200001)
    e2 = 1 - alpha - alpha * e1
    ok = (e2 >= 0) & (e2 < 1)
    c_coll = float(np.min(amd_value(e1[ok], e2[ok], alpha, gamma)))
    eps = (p["m"][0] + p["m"][1]) * MJUP / p["M_star"]
    r = 0.80199
    g = 81 * (1 - alpha) ** 5 / (512 * r * eps) - 32 * r * eps / (9 * (1 - alpha) ** 2)
    c_mmr = g ** 2 * gamma * np.sqrt(alpha) / (2 + 2 * gamma * np.sqrt(alpha))
    if alpha > 1 - 1.46 * eps ** (2 / 7):
        c_mmr = 0.0
    return c_coll, float(c_mmr), alpha, gamma


def amd_stable(p):
    c_coll, c_mmr, alpha, gamma = amd_limits(p)
    return amd_value(p["e"][0], p["e"][1], alpha, gamma) < min(c_coll, c_mmr)


def bisect(fun, lo, hi):
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fun(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


# ---------------------------------------------------------------- the RV posterior
def load_posterior():
    D = np.load(POSTERIOR)
    labels = [str(x) for x in D["labels"]]
    col = {n: D["chain"][:, i] for i, n in enumerate(labels)}
    post = {}
    for s in "bc":
        h, k = col[f"h_{s}"], col[f"k_{s}"]
        post[s] = dict(P=col[f"P_{s}"], K=col[f"K_{s}"], e=h ** 2 + k ** 2,
                       omega=np.mod(np.arctan2(k, h), 2 * np.pi), lam=np.mod(col[f"lam_{s}"], 2 * np.pi))
    return post, hashlib.sha1(D["chain"].tobytes()).hexdigest()[:10]


def lam_statistics(post):
    """Circular means of the mean longitudes, their standard deviations and covariance."""
    mean = [float(np.angle(np.mean(np.exp(1j * post[s]["lam"]))) % (2 * np.pi)) for s in "bc"]
    wrapped = np.array([m + np.mod(post[s]["lam"] - m + np.pi, 2 * np.pi) - np.pi for s, m in zip("bc", mean)])
    cov = np.cov(wrapped)
    corr_p = [float(np.corrcoef(post[s]["P"], wrapped[i])[0, 1]) for i, s in enumerate("bc")]
    return dict(mean=mean, std=[float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))],
                cov=cov.tolist(), corr_with_P=corr_p)


def make_draws(post, n, seed=42):
    """n random systems from the posterior, each with its own stellar mass."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(post["b"]["P"]), n, replace=False)
    m_star = np.abs(rng.normal(M_STAR, M_STAR_ERR, n))
    draws = []
    for j, i in enumerate(idx):
        g = lambda s, q: float(post[s][q][i])
        draws.append(dict(M_star=float(m_star[j]), P=(g("b", "P"), g("c", "P")),
                          m=(float(msini(g("b", "K"), g("b", "P"), g("b", "e"), m_star[j])),
                             float(msini(g("c", "K"), g("c", "P"), g("c", "e"), m_star[j]))),
                          e=(g("b", "e"), g("c", "e")), omega=(g("b", "omega"), g("c", "omega")),
                          lam=(g("b", "lam"), g("c", "lam"))))
    return draws


# ---------------------------------------------------------------- running and saving jobs
def run_all(pool, jobs, cache, label, every=None):
    done = {}
    if os.path.exists(cache):
        for line in open(cache):
            key, res = json.loads(line)
            done[key] = res
    todo = [j for j in jobs if j[0] not in done]
    print(f"{label}: {len(jobs)} runs, {len(jobs) - len(todo)} already saved, {len(todo)} to go", flush=True)
    t0, step = time.time(), every or max(1, len(todo) // 20)
    with open(cache, "a") as fh:
        for i, (key, res) in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            done[key] = res
            fh.write(json.dumps([key, res]) + "\n"); fh.flush()
            if i % step == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {(time.time() - t0) / 60:.1f} min", flush=True)
    return done


def per_orbit_seconds(p):
    sim = make_sim(p)
    t = time.time(); sim.integrate(50_000 * sim.particles[1].P, exact_finish_time=0)
    return (time.time() - t) / 50_000


# ---------------------------------------------------------------- the analysis
def main():
    t_start = time.time()
    workers = int(os.environ.get("WORKERS", 0)) or max(1, (os.cpu_count() or 2) - 1)
    post, post_id = load_posterior()
    lam = lam_statistics(post)
    BEST = dict(M_star=M_STAR, lam=tuple(lam["mean"]), **PUBLISHED)
    draws = make_draws(post, N_MC)
    settings = (VERSION, QUICK, PUBLISHED, N_SHORT, N_LONG, N_VERY_LONG, N_MC, N_INC, F, LAM_SPAN, N_LAM, post_id)
    cache = f"runs_{hashlib.sha1(repr(settings).encode()).hexdigest()[:8]}.jsonl"
    print(f"Using {workers} processes. Quick mode: {QUICK}. Saving runs to {cache}", flush=True)
    lam_off = np.linspace(-LAM_SPAN, LAM_SPAN, N_LAM)
    sb, sc = lam["std"]

    # --- stage 1: MEGNO runs
    jobs = []
    for name, kw in [("P/30", {}), ("P/60", dict(dt_frac=1 / 60)), ("P/100", dict(dt_frac=1 / 100)),
                     ("IAS15", dict(integrator="ias15")), ("1e5 orbits", dict(n=100_000)),
                     ("seed 2", dict(seed=2)), ("seed 3", dict(seed=3)), ("seed 4", dict(seed=4))]:
        a = dict(f=1.0, n=N_SHORT, dt_frac=1 / 30, integrator="whfast", seed=1); a.update(kw)
        jobs.append((f"short|best|{name}", "short", (BEST, a["f"], a["n"], a["dt_frac"], a["integrator"], a["seed"])))
    for db in (-2, 0, 2):
        for dc in (-2, 0, 2):
            p = with_(BEST, lam=(BEST["lam"][0] + db * sb, BEST["lam"][1] + dc * sc))
            jobs.append((f"short|lamshift|{db}|{dc}", "short", (p,)))
    for a, db in enumerate(lam_off):
        for b, dc in enumerate(lam_off):
            jobs.append((f"short|lammap|{a}|{b}", "short", (with_(BEST, lam=(BEST["lam"][0] + db, BEST["lam"][1] + dc)),)))
    for k, p in enumerate(draws):
        jobs.append((f"short|mc|{k}", "short", (p,)))
    for f in F:
        jobs.append((f"short|incbest|{f}", "short", (BEST, f)))
        for k in range(N_INC):
            jobs.append((f"short|inc|{f}|{k}", "short", (draws[k], f)))

    with Pool(workers) as pool:
        S = run_all(pool, jobs, cache, "Stage 1, chaos indicator")

        # --- stage 2: long runs of everything without an early close encounter
        long_jobs = [("long|best", "long", (BEST, 1.0, N_VERY_LONG))]
        for k, p in enumerate(draws):
            meg, enc = S[f"short|mc|{k}"]
            if not enc:
                long_jobs.append((f"long|mc|{k}", "long", (p, 1.0, N_LONG if meg < REGULAR_BELOW else N_VERY_LONG)))
        for f in F:
            if not S[f"short|incbest|{f}"][1]:
                long_jobs.append((f"long|incbest|{f}", "long", (BEST, f, N_LONG)))
            for k in range(N_INC):
                if not S[f"short|inc|{f}|{k}"][1]:
                    long_jobs.append((f"long|inc|{f}|{k}", "long", (draws[k], f, N_LONG)))
        long_jobs.sort(key=lambda j: -j[2][2])                     # longest first, for an even load
        sec = per_orbit_seconds(BEST)
        cost = sec * sum(j[2][2] for j in long_jobs) / workers / 3600
        print(f"Stage 2: roughly {cost:.1f} h on this machine if every run goes to the end (unstable runs stop"
              f" early). The longest runs go first and take about {sec * N_VERY_LONG / 60:.0f} min each, so the"
              f" first progress lines can take that long to appear.", flush=True)
        L = run_all(pool, long_jobs, cache, "Stage 2, long integrations", every=1)

    R = analyse(BEST, draws, lam, post, S, L, lam_off)
    json.dump(R, open("results.json", "w"), indent=1, default=float)
    print_summary(R)
    figure_from_files()
    print(f"figure saved: hd127195_stability.pdf\nall done in {(time.time() - t_start) / 3600:.2f} h")


def analyse(BEST, draws, lam, post, S, L, lam_off):
    R = dict(version=VERSION, quick=QUICK, n_short=N_SHORT, n_long=N_LONG, n_very_long=N_VERY_LONG)
    sim = make_sim(BEST)
    pb_yr = sim.particles[1].P
    ab, ac = sim.particles[1].a, sim.particles[2].a
    rh = ((sim.particles[1].m + sim.particles[2].m) / (3 * M_STAR)) ** (1 / 3) * (ab + ac) / 2
    ratio = post["c"]["P"] / post["b"]["P"]
    R.update(a_b=ab, a_c=ac, orbit_b_years=pb_yr, period_ratio=PUBLISHED["P"][1] / PUBLISHED["P"][0],
             period_ratio_std=float(np.std(ratio)), mutual_hill=(ac - ab) / rh, lam=lam,
             best=dict(BEST, lam=list(BEST["lam"])))
    q = lambda x: [float(v) for v in np.percentile(x, [16, 50, 84])]
    R["posterior"] = {s: {k: q(post[s][k]) for k in ("P", "K", "e")} for s in "bc"}

    # published solution: robustness, mean-longitude map and long run
    R["best_megno"] = {k.split("|")[2]: v[0] for k, v in S.items() if k.startswith("short|best|")}
    shift = [v[0] for k, v in S.items() if k.startswith("short|lamshift|")]
    R["lamshift_megno_range"] = [min(shift), max(shift)]
    Y = np.array([[S[f"short|lammap|{a}|{b}"][0] for a in range(N_LAM)] for b in range(N_LAM)])   # rows: lam_c
    E = np.array([[S[f"short|lammap|{a}|{b}"][1] for a in range(N_LAM)] for b in range(N_LAM)])
    DB, DC = np.meshgrid(lam_off, lam_off)
    icov = np.linalg.inv(np.array(lam["cov"]))
    d2 = icov[0, 0] * DB ** 2 + 2 * icov[0, 1] * DB * DC + icov[1, 1] * DC ** 2
    R["lam_map"] = dict(offsets=lam_off.tolist(), megno=Y.tolist(), encounter=E.tolist())
    R["lam_map_regular_within_2sigma"] = float((Y[d2 <= 6.18] < REGULAR_BELOW).mean())   # 2D 2-sigma region
    R["lam_map_regular_all"] = float((Y < REGULAR_BELOW).mean())
    ok, t_enc, err = L["long|best"]
    R["best_long"] = dict(survived=ok, orbits=N_VERY_LONG, years=N_VERY_LONG * pb_yr, energy_error=err,
                          encounter_orbits=t_enc)
    c_coll, c_mmr, alpha, gamma = amd_limits(BEST)
    crit = min(c_coll, c_mmr)
    lim = lambda c, fun: bisect(lambda e: fun(e) - c, 0, 0.9)
    funs = dict(equal=lambda e: amd_value(e, e, alpha, gamma), b_alone=lambda e: amd_value(e, 0, alpha, gamma),
                c_alone=lambda e: amd_value(0, e, alpha, gamma))
    R["amd"] = dict(c_collision=c_coll, c_overlap=c_mmr,
                    e_limit={k: lim(crit, f) for k, f in funs.items()},
                    e_limit_collision_only={k: lim(c_coll, f) for k, f in funs.items()})

    # Monte Carlo draws
    mc = []
    for k, p in enumerate(draws):
        meg, enc = S[f"short|mc|{k}"]
        row = dict(p, megno=meg, regular=bool(meg < REGULAR_BELOW and not enc), early_encounter=bool(enc),
                   amd_stable=bool(amd_stable(p)), amd_stable_at_134=bool(amd_stable(dict(p, M_star=M_STAR))))
        if enc:
            row.update(long_orbits=0, survived=False, encounter_orbits=None)
        else:
            okk, te, _ = L[f"long|mc|{k}"]
            row.update(long_orbits=N_LONG if row["regular"] else N_VERY_LONG, survived=okk, encounter_orbits=te)
        mc.append(row)
    R["mc"] = mc
    reg = np.array([r["regular"] for r in mc]); early = np.array([r["early_encounter"] for r in mc])
    cha = ~reg & ~early
    lost_by = lambda n: np.array([r["early_encounter"] or (r["encounter_orbits"] is not None and r["encounter_orbits"] <= n)
                                  for r in mc])
    R["mc_counts"] = dict(n=len(mc), regular=int(reg.sum()), chaotic=int(cha.sum()), early_encounter=int(early.sum()),
                          regular_lost_by_long=int((reg & lost_by(N_LONG)).sum()),
                          chaotic_lost_by_1e6=int((cha & lost_by(1_000_000)).sum()),
                          chaotic_lost_by_long=int((cha & lost_by(N_LONG)).sum()),
                          chaotic_lost_by_very_long=int((cha & lost_by(N_VERY_LONG)).sum()),
                          amd_stable=int(sum(r["amd_stable"] for r in mc)),
                          amd_stable_at_134=int(sum(r["amd_stable_at_134"] for r in mc)))
    pr = np.array([r["P"][1] / r["P"][0] for r in mc]); ec = np.array([r["e"][1] for r in mc])
    eb = np.array([r["e"][0] for r in mc])
    frac = lambda sel: [float(reg[sel].mean()) if sel.any() else None, int(sel.sum())]
    R["mc_regular_by"] = dict(ratio_below_156=frac(pr < 1.56), ratio_above_158=frac(pr > 1.58),
                              ec_below_005=frac(ec < 0.05), ec_005_010=frac((ec >= 0.05) & (ec < 0.10)),
                              ec_above_010=frac(ec >= 0.10), emax_below_005=frac(np.maximum(eb, ec) < 0.05),
                              emax_above_010=frac(np.maximum(eb, ec) >= 0.10))

    # inclination scan: f = 1 comes from the first N_INC Monte Carlo draws
    inc = [dict(f=1.0, i_deg=90.0, regular=float(reg[:N_INC].mean()),
                no_early_encounter=float(1 - early[:N_INC].mean()),
                no_encounter_long=float(1 - lost_by(N_LONG)[:N_INC].mean()),
                best_megno=R["best_megno"]["P/30"], best_survives_long=bool(ok))]
    for f in F:
        sh = [S[f"short|inc|{f}|{k}"] for k in range(N_INC)]
        lo = [S[f"short|inc|{f}|{k}"][1] or not L[f"long|inc|{f}|{k}"][0] for k in range(N_INC)]
        bmeg, benc = S[f"short|incbest|{f}"]
        inc.append(dict(f=f, i_deg=float(np.degrees(np.arcsin(1 / f))),
                        regular=float(np.mean([m < REGULAR_BELOW and not x for m, x in sh])),
                        no_early_encounter=float(1 - np.mean([x for _, x in sh])),
                        no_encounter_long=float(1 - np.mean(lo)), best_megno=bmeg,
                        best_survives_long=bool(not benc and L[f"long|incbest|{f}"][0])))
    R["inclination_scan"] = inc
    unstable = [r["f"] for r in inc if not r["best_survives_long"]]
    R["best_unstable_from_f"] = min(unstable) if unstable else None
    half = [r for r in inc if r["no_encounter_long"] < 0.5]
    R["half_lost_from_f"] = min(r["f"] for r in half) if half else None
    above = [r["f"] for r in inc if r["no_encounter_long"] >= 0.5]
    R["half_kept_up_to_f"] = max(above) if above else None
    fac = R["half_kept_up_to_f"] or 1.0
    sig = lambda k, x: PUBLISHED["m"][k] * MJUP * x / M_STAR * [ab, ac][k] / DIST_PC * 1e6   # micro-arcsec
    R["astrometric_uas_min"] = [sig(0, 1), sig(1, 1)]
    R["astrometric_uas_at_half"] = [sig(0, fac), sig(1, fac)]
    R["true_mass_at_half"] = [PUBLISHED["m"][k] * fac for k in (0, 1)]
    return R


# ---------------------------------------------------------------- figure
def figure_from_files():
    make_figure(json.load(open("results.json")))


def make_figure(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 6.3, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#5f5f5f"
    vmin, vmax = 1.9, 5.0
    fig = plt.figure(figsize=(7.1, 2.6))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 0.05, 0.42, 1, 0.3, 1], wspace=0.08)
    ax_a, cax, ax_b, ax_c = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3]), fig.add_subplot(gs[5])

    def letter(ax, s, right=False):
        ax.text(0.97 if right else 0.03, 0.97, s, transform=ax.transAxes, ha="right" if right else "left",
                va="top", fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))

    # (a) chaos indicator over the two mean longitudes, with the posterior 1 and 2 sigma regions
    off = np.array(R["lam_map"]["offsets"])
    Y, E = np.array(R["lam_map"]["megno"]), np.array(R["lam_map"]["encounter"], dtype=bool)
    mesh = ax_a.pcolormesh(off, off, np.clip(Y, vmin, vmax), vmin=vmin, vmax=vmax, cmap="viridis",
                           shading="nearest", rasterized=True)
    ax_a.pcolormesh(off, off, np.where(E, 1.0, np.nan), cmap=ListedColormap(["#c8c8c8"]), shading="nearest",
                    rasterized=True)
    cov = np.array(R["lam"]["cov"])
    w, v = np.linalg.eigh(cov)
    th = np.linspace(0, 2 * np.pi, 200)
    for k2, ls in ((2.30, "-"), (6.18, "--")):                  # 1 and 2 sigma for two parameters
        pts = v @ (np.sqrt(k2 * w)[:, None] * np.array([np.cos(th), np.sin(th)]))
        ax_a.plot(pts[0], pts[1], color="white", lw=0.9, ls=ls)
    ax_a.plot(0, 0, marker="*", ms=8, color="white", mec="black", mew=0.6, ls="none")
    ax_a.set_xlabel(r"$\Delta\lambda_b$ (rad)"); ax_a.set_ylabel(r"$\Delta\lambda_c$ (rad)")
    ax_a.set_xlim(off[0], off[-1]); ax_a.set_ylim(off[0], off[-1])
    letter(ax_a, "(a)")
    cb = fig.colorbar(mesh, cax=cax)
    cb.ax.set_title(r"$\langle Y\rangle$", fontsize=7, pad=3)
    cb.ax.tick_params(labelsize=6)

    # (b) the Monte Carlo draws: period ratio and eccentricity of planet c, by outcome
    mc = R["mc"]
    pr = np.array([r["P"][1] / r["P"][0] for r in mc]); ec = np.array([r["e"][1] for r in mc])
    lost = np.array([not r["survived"] for r in mc]); reg = np.array([r["regular"] for r in mc])
    for sel, kw in [(reg & ~lost, dict(marker="o", s=10, color=BLUE, label="regular")),
                    (~reg & ~lost, dict(marker="^", s=13, color=ORANGE, label="chaotic")),
                    (lost, dict(marker="x", s=14, color=GRAY, label="close encounter"))]:
        ax_b.scatter(pr[sel], ec[sel], lw=0.9 if kw["marker"] == "x" else 0.3,
                     edgecolors="white" if kw["marker"] != "x" else None, **kw)
    for p_, q_ in [(3, 2), (8, 5)]:
        if pr.min() - 0.01 < p_ / q_ < pr.max() + 0.01:
            ax_b.axvline(p_ / q_, color="0.4", lw=0.6, ls=":")
            ax_b.text(p_ / q_, 0.985, f" {p_}:{q_}", transform=ax_b.get_xaxis_transform(), ha="left", va="top",
                      fontsize=6, color="0.3")
    ax_b.set_xlabel(r"$P_c/P_b$"); ax_b.set_ylabel(r"$e_c$")
    ax_b.set_ylim(0, 1.5 * ec.max())
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handletextpad=0.2,
                columnspacing=0.8, borderaxespad=0.2, markerscale=1.1)
    letter(ax_b, "(b)")

    # (c) inclination
    inc = R["inclination_scan"]
    i_deg = [r["i_deg"] for r in inc]
    ax_c.plot(i_deg, [r["no_encounter_long"] for r in inc], "o-", ms=3, lw=1.0, color="0.15",
              label="no close encounter")
    ax_c.plot(i_deg, [r["regular"] for r in inc], "s--", ms=2.5, lw=1.0, color=BLUE, label="regular")
    if R["best_unstable_from_f"]:
        ax_c.axvline(np.degrees(np.arcsin(1 / R["best_unstable_from_f"])), color="0.5", lw=0.6, ls=":")
    ax_c.legend(loc="upper right", frameon=False, handlelength=1.6, borderaxespad=0.4)
    ax_c.set_xscale("log")
    ax_c.set_xticks([6, 10, 20, 30, 45, 90]); ax_c.set_xticklabels(["6", "10", "20", "30", "45", "90"])
    ax_c.minorticks_off()
    ax_c.set_xlim(95, 5.2); ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_xlabel(r"inclination $i$ (deg)"); ax_c.set_ylabel("fraction of draws")
    letter(ax_c, "(c)")
    fig.savefig("hd127195_stability.pdf", bbox_inches="tight")
    fig.savefig("hd127195_stability_preview.png", dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------- printed numbers
def print_summary(R):
    pct = lambda x: f"{100 * x:.0f}%"
    c = R["mc_counts"]
    b = R["best_megno"]
    inc = {round(r["f"], 2): r for r in R["inclination_scan"]}
    myr = lambda n: n * R["orbit_b_years"] / 1e6
    by = R["mc_regular_by"]
    fr = lambda x: f"{pct(x[0])} of {x[1]}" if x[0] is not None else f"none of {x[1]}"
    lines = [
        ("N1", "period ratio P_c/P_b (published) and its spread in the RV posterior",
         f"{R['period_ratio']:.3f} +- {R['period_ratio_std']:.3f}"),
        ("N2", "separation in mutual Hill radii", f"{R['mutual_hill']:.1f}"),
        ("N3", "mean longitudes from the RV refit (b, c), radians",
         f"{R['lam']['mean'][0]:.2f} +- {R['lam']['std'][0]:.2f}, {R['lam']['mean'][1]:.2f} +- {R['lam']['std'][1]:.2f}"
         f" (correlation with P: {R['lam']['corr_with_P'][0]:+.2f}, {R['lam']['corr_with_P'][1]:+.2f})"),
        ("N4", "published solution, MEGNO (P/30, P/60, P/100, IAS15, 1e5 orbits)",
         ", ".join(f"{b[k]:.3f}" for k in ("P/30", "P/60", "P/100", "IAS15", "1e5 orbits"))),
        ("N5", "MEGNO range with the mean longitudes shifted by +-2 sigma",
         f"{R['lamshift_megno_range'][0]:.2f} to {R['lamshift_megno_range'][1]:.2f}"),
        ("N6", "mean-longitude map: regular inside the 2-sigma region / whole map (+-1 rad)",
         f"{pct(R['lam_map_regular_within_2sigma'])} / {pct(R['lam_map_regular_all'])}"),
        ("N7", "published solution, long run: survived / Myr / energy error",
         f"{R['best_long']['survived']} / {myr(R['best_long']['orbits']):.0f} Myr / {R['best_long']['energy_error']}"),
        ("N8", "Monte Carlo: regular / chaotic / close encounter within 2e4 orbits",
         f"{c['regular']} / {c['chaotic']} / {c['early_encounter']} of {c['n']}"),
        ("N9", f"regular draws lost within {myr(R['n_long']):.1f} Myr", f"{c['regular_lost_by_long']} of {c['regular']}"),
        ("N10", f"chaotic draws lost within 1.5 / {myr(R['n_long']):.1f} / {myr(R['n_very_long']):.0f} Myr",
         f"{c['chaotic_lost_by_1e6']} / {c['chaotic_lost_by_long']} / {c['chaotic_lost_by_very_long']} of {c['chaotic']}"),
        ("N11", "draws that pass kima's AMD test (with their own stellar mass / with 1.34 Msun but unchanged planet masses)",
         f"{c['amd_stable']} / {c['amd_stable_at_134']} of {c['n']}"),
        ("N12", "AMD limits for the published solution: equal e / e_b alone / e_c alone (collision only)",
         " / ".join(f"{R['amd']['e_limit'][k]:.3f}" for k in ("equal", "b_alone", "c_alone")) + "  (" +
         " / ".join(f"{R['amd']['e_limit_collision_only'][k]:.3f}" for k in ("equal", "b_alone", "c_alone")) + ")"),
        ("N13", "regular fraction for P_c/P_b < 1.56 / > 1.58", f"{fr(by['ratio_below_156'])} / {fr(by['ratio_above_158'])}"),
        ("N14", "regular fraction for e_c < 0.05 / 0.05-0.10 / >= 0.10",
         f"{fr(by['ec_below_005'])} / {fr(by['ec_005_010'])} / {fr(by['ec_above_010'])}"),
        ("N15", "regular fraction with both e < 0.05 / larger e >= 0.10",
         f"{fr(by['emax_below_005'])} / {fr(by['emax_above_010'])}"),
        ("N16", f"inclination scan, no close encounter in {myr(R['n_long']):.1f} Myr ({N_INC} draws), by i",
         ", ".join(f"{r['i_deg']:.1f} deg: {pct(r['no_encounter_long'])}" for r in R["inclination_scan"])),
        ("N17", "inclination scan, regular fraction by i",
         ", ".join(f"{r['i_deg']:.1f} deg: {pct(r['regular'])}" for r in R["inclination_scan"])),
        ("N18", "published solution first unstable at mass factor (i)",
         f"{R['best_unstable_from_f']} ({np.degrees(np.arcsin(1 / R['best_unstable_from_f'])):.1f} deg)"
         if R["best_unstable_from_f"] else "never in the scan"),
        ("N19", "at least half the draws survive up to mass factor (i); true masses there (b, c)",
         f"{R['half_kept_up_to_f']} ({np.degrees(np.arcsin(1 / R['half_kept_up_to_f'])):.1f} deg); "
         f"{R['true_mass_at_half'][0]:.2f}, {R['true_mass_at_half'][1]:.2f} M_J" if R["half_kept_up_to_f"] else "none"),
        ("N20", "astrometric signatures at minimum mass and at that factor (b, c), micro-arcsec",
         f"{R['astrometric_uas_min'][0]:.0f}, {R['astrometric_uas_min'][1]:.0f}; "
         f"{R['astrometric_uas_at_half'][0]:.0f}, {R['astrometric_uas_at_half'][1]:.0f}"),
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
