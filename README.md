# Financial Toxicity: The Anatomy of a Medical Shock

A scrollytelling data narrative exploring why medical bills vary so wildly between patients with similar diagnoses, and using an econometric causal model to separate what actually drives the cost of a hospital stay from what looks like pure, unexplained "financial toxicity."

**Live site:** enable GitHub Pages for this repo (Settings -> Pages -> Deploy from branch -> `main`, folder `/ (root)`) and it will be live at `https://<your-username>.github.io/financial-toxicity/`.

## What it is

A single self-contained `index.html` page: five narrative scenes advance as the reader scrolls (via `IntersectionObserver`, no scrollytelling library needed), each swapping in a different Plotly chart rendered from pre-computed JSON. There is no backend and no build step -- it's meant to be served as-is by GitHub Pages.

1. **The Billing Black Box** -- distribution of billing amounts across common conditions.
2. **Where Do Shocks Happen?** -- billing variance by insurance type and admission pathway (Emergency vs. planned).
3. **The Baseline Cost (Doubly Robust AIPW Model)** -- an Augmented Inverse Probability Weighting (AIPW) causal model isolating the true daily cost of a hospital stay from confounding.
4. **The Unexplained Residuals** -- the gap between predicted and actual bills, highlighting the top 1% of statistical outliers ("Marcus," the narrative's running example).
5. **A Note on Data & Governance** -- discloses the synthetic dataset and why real patient-level billing data isn't used here.

## Data and methodology

Source data: [`Nicolybgs/healthcare_data`](https://huggingface.co/datasets/Nicolybgs/healthcare_data) on Hugging Face -- a synthetic healthcare billing dataset. No real patient data is used anywhere in this project.

`Healthcare_project.py` is the analysis pipeline that produces every chart on the page: it loads and cleans the dataset, applies a US inpatient economic calibration layer (a realistic fixed admission charge plus a per-diem marginal cost, weighted by condition acuity, severity, and admission type), fits a Doubly Robust Augmented Inverse Probability Weighting (AIPW) model to isolate the causal effect of an extended stay on total billing, computes billing residuals against each patient's AIPW counterfactual to isolate unexplained cost shocks, and writes out five Plotly figure JSON files plus a small metrics file. Re-running it regenerates everything in `scrollytelling_assets/`.

Model diagnostics: Doubly Robust ATE (Average Treatment Effect) = $29,781.40 (SE = $109.46, p < 0.0001) for an extended stay averaging 11.6 additional days, implying a marginal cost of roughly $2,556.44 per day -- within standard US acute-care inpatient benchmarks ($2,200-$3,500/day for routine-to-intermediate care). Propensity score overlap: min = 0.168, max = 0.990, satisfying the positivity assumption ($0 < P(T=1\vert W) < 1$) with no near-zero extreme weights.

### Methodological note: why AIPW instead of 2SLS

An earlier version of this project used a Two-Stage Least Squares (2SLS) instrumental-variable design, with Emergency-admission status as an instrument for length of stay. That design had a fatal flaw: for an instrument to be valid it can only affect the outcome (total bill) through the treatment (length of stay), and Emergency admission doesn't satisfy that. Emergency admissions carry their own direct cost drivers -- ED facility fees, rapid-sequence triage, immediate CT/MRI imaging, emergency physician fees, urgent labs -- that inflate the bill independently of how long the patient stays. That's a direct violation of the exclusion restriction, which invalidates the 2SLS estimate regardless of how strong the first-stage F-statistic looks (a high F-statistic only confirms the instrument is *correlated* with the treatment; it says nothing about whether the instrument is *exogenous*). Consistent with that flaw, the old model produced an implausible $134.32/day estimate -- well below basic nurse staffing or hotel lodging costs, let alone comprehensive inpatient care, and a sign the estimate was an artifact of model mis-specification rather than a real effect.

The current version replaces that instrument with AIPW, a doubly robust design: it combines a propensity-score model of who receives an extended stay with an outcome model (a Gamma GLM with a log link, standard for right-skewed billing data) of expected cost under each condition, and remains consistent if *either* model is correctly specified. This avoids the exclusion-restriction problem entirely -- there's no instrument to be invalid -- while still producing a defensible, doubly robust causal estimate, and it happens to land inside real-world inpatient cost benchmarks rather than far below them. It also makes the framing of the residuals in Scene 4 more honest: they're the gap left after conditioning on observed severity, condition, admission type, and insurance, not a leftover from a mis-specified instrument -- though, as the narrative there notes, some of that gap can still be explained by clinical detail (complications, procedures, unobserved severity) that no observational dataset like this one fully captures.

## Repository layout

- `index.html` -- the scrollytelling page. Everything (styles, scroll logic, chart rendering) is in this one file, aside from the Plotly.js library loaded from a CDN.
- `scrollytelling_assets/` -- pre-rendered Plotly figure JSON for each scene, fetched by `index.html` at runtime. **Do not put these behind Git LFS** -- GitHub Pages serves LFS-tracked files as plain-text pointer stubs, not the real content, which would silently break every chart on the live site. Committing them as regular files (as this repo does) is the correct approach here despite their size.
- `Healthcare_project.py` -- the script that generates everything in `scrollytelling_assets/` from the raw Hugging Face dataset.

## Running locally

Because the page loads its chart data with `fetch()`, opening `index.html` directly (double-clicking it, `file://...`) will not work -- browsers block `fetch()` of local files under the `file://` origin, so every chart will hang on "Loading interactive charts..." Serve the folder instead:

```bash
cd financial-toxicity
python3 -m http.server 8000
```

Then open `http://localhost:8000`. This restriction is specific to `file://`; once deployed to GitHub Pages (or any real HTTP server), the same `fetch()` calls work with no changes needed.

To regenerate the chart data from scratch:

```bash
pip install -r requirements.txt
python3 Healthcare_project.py
```

## A performance note

The four scene files (`scene1_billing_distribution.json` ~19.3MB, `scene2_insurance_variance.json` ~8.6MB, `scene3_causal_coefficients.json` ~6.8MB, `scene4_residual_outliers.json` ~31.6MB) add up to roughly 66MB fetched by a visitor's browser. None of this will block a GitHub push (GitHub's hard limit is 100MB per file), and the page will work correctly on GitHub Pages as-is -- but ~66MB is a lot for a page to load, especially on mobile. If load time turns out to matter, the fix would be in `Healthcare_project.py`: sampling the dataset before passing it to `px.histogram`/`px.scatter`, dropping unused `hover_data` columns, or switching the scatter traces to WebGL (`scattergl`) -- not something this GitHub-readiness pass changes on its own.

## Tech stack

Plotly.js (CDN), vanilla JS `IntersectionObserver` for the scroll-driven narrative, and a Python data pipeline (pandas, Plotly Express, statsmodels, scikit-learn) for the causal model and chart generation.
