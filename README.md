# Financial Toxicity: The Anatomy of a Medical Shock

A scrollytelling data narrative exploring why medical bills vary so wildly between patients with similar diagnoses, and using an econometric causal model to separate what actually drives the cost of a hospital stay from what looks like pure, unexplained "financial toxicity."

**Live site:** enable GitHub Pages for this repo (Settings -> Pages -> Deploy from branch -> `main`, folder `/ (root)`) and it will be live at `https://<your-username>.github.io/financial-toxicity/`.

## What it is

A single self-contained `index.html` page: five narrative scenes advance as the reader scrolls (via `IntersectionObserver`, no scrollytelling library needed), each swapping in a different Plotly chart rendered from pre-computed JSON. There is no backend and no build step -- it's meant to be served as-is by GitHub Pages.

1. **The Billing Black Box** -- distribution of billing amounts across common conditions.
2. **Where Do Shocks Happen?** -- billing variance by insurance type and admission pathway (Emergency vs. planned).
3. **The Baseline Cost (2SLS Model)** -- an Instrumental Variable (2SLS) causal model isolating the true daily cost of a hospital stay from confounding.
4. **The Unexplained Residuals** -- the gap between predicted and actual bills, highlighting the top 1% of statistical outliers ("Marcus," the narrative's running example).
5. **A Note on Data & Governance** -- discloses the synthetic dataset and why real patient-level billing data isn't used here.

## Data and methodology

Source data: [`Nicolybgs/healthcare_data`](https://huggingface.co/datasets/Nicolybgs/healthcare_data) on Hugging Face -- a synthetic healthcare billing dataset. No real patient data is used anywhere in this project.

`Healthcare_project.py` is the analysis pipeline that produces every chart on the page: it loads and cleans the dataset, fits an Instrumental Variable (2SLS) model using Emergency-admission status as an instrument for length of stay (to separate correlation from causation), computes billing residuals to isolate unexplained cost shocks, and writes out five Plotly figure JSON files plus a small metrics file. Re-running it regenerates everything in `scrollytelling_assets/`.

Model diagnostics: Wu-Hausman p-value = 0.0000, first-stage F-statistic = 746.54 (comfortably above the conventional weak-instrument threshold of 10), estimated causal impact of length of stay on the baseline bill = $134.32/day.

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

The three largest scene files (`scene1_billing_distribution.json` ~19MB, `scene2_insurance_variance.json` ~8.2MB, `scene4_residual_outliers.json` ~29MB) add up to roughly 57MB fetched by a visitor's browser. None of this will block a GitHub push (GitHub's hard limit is 100MB per file), and the page will work correctly on GitHub Pages as-is -- but ~57MB is a lot for a page to load, especially on mobile. If load time turns out to matter, the fix would be in `Healthcare_project.py`: sampling the dataset before passing it to `px.histogram`/`px.scatter`, dropping unused `hover_data` columns, or switching the scatter traces to WebGL (`scattergl`) -- not something this GitHub-readiness pass changes on its own.

## Tech stack

Plotly.js (CDN), vanilla JS `IntersectionObserver` for the scroll-driven narrative, and a Python data pipeline (pandas, Plotly Express, statsmodels, linearmodels) for the causal model and chart generation.
