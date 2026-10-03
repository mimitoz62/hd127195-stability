"""How much do the planets' mutual perturbations change the radial velocities over the 18-yr baseline?

Integrates the published solution (with the mean longitudes of the RV refit) once with both planets
together and once with each planet alone, and prints the difference in the star's radial velocity
over the time span of the CORALIE data, centred on the reference epoch of the fit.

Usage
    python interaction_check.py
"""
import numpy as np
import rebound

from paper_analysis import MJUP, PUBLISHED, load_posterior, lam_statistics

M_STAR = 1.34
SPAN_DAYS = 6737.6                                   # time span of the 81 RVs
AU_PER_DAY_IN_MS = 1.495978707e11 / 86400.0


def star_rv(planets, lam, t):
    sim = rebound.Simulation()
    sim.units = ("day", "AU", "Msun")
    sim.add(m=M_STAR)
    for k in planets:
        sim.add(m=PUBLISHED["m"][k] * MJUP, P=PUBLISHED["P"][k], e=PUBLISHED["e"][k],
                omega=PUBLISHED["omega"][k], l=lam[k])
    sim.move_to_com()
    sim.integrator = "ias15"
    out = []
    for ti in t:
        sim.integrate(ti)
        out.append(-sim.particles[0].vx * AU_PER_DAY_IN_MS)
    return np.array(out)


if __name__ == "__main__":
    lam = lam_statistics(load_posterior()[0])["mean"]
    t = np.linspace(-SPAN_DAYS / 2, SPAN_DAYS / 2, 2000)
    diff = star_rv((0, 1), lam, t) - (star_rv((0,), lam, t) + star_rv((1,), lam, t))
    print(f"Interaction signal over {SPAN_DAYS / 365.25:.1f} yr: rms {np.sqrt(np.mean(diff ** 2)):.2f} m/s, "
          f"largest {np.max(np.abs(diff)):.2f} m/s")
