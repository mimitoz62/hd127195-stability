# HD 127195 stability

N-body stability analysis of the two giant planets orbiting the red giant HD 127195, discovered by
[Figueira et al. (2026)](https://doi.org/10.1051/0004-6361/202659966). This repository holds the code for the
Research Note *N-body Stability Constraints on the Two Giant Planets of the Red Giant HD 127195*, prepared for
Research Notes of the AAS.

## Results in brief

- The published best-fit configuration (coplanar orbits, minimum masses) is regular, with MEGNO ⟨Y⟩ = 2.00, and
  survives 10⁷ orbits of planet b (14.6 Myr).
- Of 200 configurations drawn from the uncertainties, all pass the angular momentum deficit (AMD) stability criterion
  used in the discovery fit, but only 134 are regular; 9 have close encounters within 2 × 10⁴ orbits and 16 more
  within 10⁶ orbits (1.5 Myr).
- Near the best-fit period ratio (1.566), regular orbits require e_c ≲ 0.1, and chaos becomes more common toward the
  8:5 commensurability.
- If both masses scale as 1/sin i, stability requires i > 26° (95% probability), so the true masses are at most
  2.3 times the minimum masses.

![Figure 1](results/hd127195_stability.png)

*MEGNO maps over the eccentricities (a) and over the period ratio (b), and the fraction of draws that stay regular or
free of close encounters as a function of inclination (c).*

## Contents

| Path | Description |
|---|---|
| `paper_analysis.py` | Reproduces every number and Figure 1 of the Research Note |
| `notebooks/01_jupiter_saturn_validation.ipynb` | Energy conservation and MEGNO tests on the Sun–Jupiter–Saturn system |
| `notebooks/02_gladman_hill_stability.ipynb` | Reproduction of the Gladman (1993) two-planet Hill stability limit |
| `notebooks/03_hd127195_first_tests.ipynb` | Target selection from the NASA Exoplanet Archive and first stability tests of HD 127195 |
| `results/` | Output of the run used in the paper: `results.json`, `maps.npz` and Figure 1 |
| `requirements.txt` | Python dependencies |

## Reproducing the paper

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python paper_analysis.py
```

The full run takes a few minutes on a recent laptop and uses all but one CPU core
(`WORKERS=4 python paper_analysis.py` sets the number of processes). It prints every number quoted in the paper,
labelled [N1] to [N22], and writes:

- `hd127195_stability.pdf` (Figure 1) and `hd127195_stability_preview.png`
- `results.json`: all summary numbers
- `maps.npz`: the MEGNO and close-encounter maps

`python paper_analysis.py --quick` runs small grids to test the pipeline, and `python paper_analysis.py --figure`
redraws Figure 1 from the saved results. Chaotic integrations are sensitive to floating-point rounding, so counts can
differ by a draw or two between machines.

## Method

- Integrator: REBOUND 5.2.0 with WHFast and a time step of P_b/30; spot checks with P_b/60, P_b/100 and IAS15.
- Chaos indicator: MEGNO ⟨Y⟩ over 2 × 10⁴ orbits of planet b; ⟨Y⟩ < 2.2 counts as regular.
- Close encounter: the planets approach within the smaller Hill radius, or one is ejected beyond 100 a_c.
- Uncertainties: 200 draws with each parameter sampled independently from a Gaussian; negative eccentricities are set
  to zero. Chaotic draws are followed for 10⁶ orbits of planet b.
- Inclination: both masses scaled by 1/sin i, with an isotropic prior weighted by the fraction of draws free of close
  encounters.

## Data sources

- Figueira, P., et al. 2026, A&A, 712, A143: orbital solution (Table 2, text and Fig. C.2)
- Ottoni, G., et al. 2022, A&A, 657, A87: stellar mass and distance
- NASA Exoplanet Archive, Planetary Systems and Planetary Systems Composite tables (notebook 03)

## License

MIT; see `LICENSE`.
