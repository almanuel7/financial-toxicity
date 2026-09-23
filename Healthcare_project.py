"""
Financial Toxicity: The Anatomy of a Medical Shock
Module: Data Pipeline, Doubly Robust Causal Model & Editorial Scrollytelling Assets
Description:
    1. Loads and calibrates the synthetic Hugging Face healthcare billing dataset
       into realistic US acute-care inpatient economics.
    2. Fits a Doubly Robust Augmented Inverse Probability Weighting (AIPW) causal
       model to isolate the effect of an extended hospital stay on total billing.
    3. Builds five small, pre-aggregated Plotly figures -- a ridgeline, a grouped
       box plot, a two-panel causal comparison, and a density-plus-outlier scatter
       -- designed for a fast-loading, editorial scrollytelling page. Every figure
       is built from summary statistics or bin counts rather than raw per-row
       data, which is what keeps the exported JSON small (kilobytes, not tens of
       megabytes) without sacrificing interactivity.
"""

import os
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datasets import load_dataset
from scipy import stats
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

# ==========================================
# 0. EDITORIAL PALETTE
# ==========================================
# A warm paper background with a cool-to-warm sequential gradient (cheap-to-
# expensive) for the ridgeline, and two signature accents everywhere else:
# a calm teal for "baseline / explained" and a deep crimson for "shock /
# unexplained". Kept in one place so the Python-rendered figures and the
# index.html design system (see the <style> block there) stay in sync.
PAPER = '#FBF9F4'
INK = '#1C1B1A'
MUTED = '#8A8580'
GRID = '#EDE7DC'
TEAL = '#2B5F6B'
CRIMSON = '#B3282D'
RIDGE_GRADIENT_STOPS = ['#1F5F6B', '#6B8A55', '#D4A72C', '#C4622C', '#B3282D']


def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return '#' + ''.join(f'{int(round(c)):02x}' for c in rgb)


def hex_to_rgba(hex_color, alpha):
    r, g, b = _hex_to_rgb(hex_color)
    return f'rgba({r},{g},{b},{alpha})'


def sequential_gradient(stops_hex, n):
    """n colors interpolated across the given hex stops. Used so overlapping
    ridgeline fills blend into neighboring hues instead of clashing (a plain
    categorical/qualitative palette mixes into muddy colors wherever two
    semi-transparent ridges overlap)."""
    stops = [_hex_to_rgb(s) for s in stops_hex]
    k = len(stops) - 1
    out = []
    for i in range(n):
        t = i / max(n - 1, 1) * k
        lo = int(np.floor(t))
        hi = min(lo + 1, k)
        frac = t - lo
        rgb = [stops[lo][c] * (1 - frac) + stops[hi][c] * frac for c in range(3)]
        out.append(_rgb_to_hex(rgb))
    return out


# ==========================================
# 1. SETUP & DATA INGESTION
# ==========================================
print("Loading dataset from Hugging Face...")
dataset = load_dataset("Nicolybgs/healthcare_data")
df = dataset['train'].to_pandas()

# Create output directory for Scrollytelling assets
output_dir = "scrollytelling_assets"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. DATA CLEANING & STANDARDIZATION
# ==========================================
print("Standardizing messy column names...")
df.columns = df.columns.str.strip()

column_mapping = {
    'Stay (in days)': 'Length_of_Stay',
    'Stay_(in_days)': 'Length_of_Stay',
    'Admission_Deposit': 'Billing_Amount',
    'Admission Deposit': 'Billing_Amount',
    'health conditions': 'Medical_Condition',
    'health_conditions': 'Medical_Condition',
    'Type of Admission': 'Admission_Type',
    'Type_of_Admission': 'Admission_Type',
    'Age': 'Age_Group',
    'Severity of Illness': 'Severity_of_Illness',
    'Available Extra Rooms in Hospital': 'Available_Extra_Rooms_in_Hospital',
    'Visitors with Patient': 'Visitors_With_Patient'
}
df = df.rename(columns=column_mapping)

# Handle Missing Values safely
df['Medical_Condition'] = df['Medical_Condition'].fillna("Unknown")
df['Insurance'] = df['Insurance'].fillna("Unknown")
df['Age_Group'] = df['Age_Group'].fillna("Unknown")
df['Admission_Type'] = df['Admission_Type'].fillna("Unknown")
df['Severity_of_Illness'] = df.get('Severity_of_Illness', pd.Series("Moderate", index=df.index)).fillna("Moderate")

# Clean numerical length of stay
df['Length_of_Stay'] = pd.to_numeric(df['Length_of_Stay'], errors='coerce').fillna(1)
df['Length_of_Stay'] = df['Length_of_Stay'].replace(0, 1)

# ==========================================
# 3. REALISTIC HEALTHCARE ECONOMIC CALIBRATION
# ==========================================
print("Applying US Inpatient Economic Calibration Layer...")
# Calibrate baseline charges into realistic CMS IPPS scales:
# Fixed baseline admission charge ($11,500) + daily marginal per diem ($3,200/day)
# weighted by condition acuity, admission type, and log-normal billing shocks.
np.random.seed(42)

acuity_multipliers = {
    'Cancer': 1.65,
    'Diabetes': 1.15,
    'Hypertension': 1.05,
    'Asthma': 1.10,
    'Arthritis': 1.00
}
condition_mult = df['Medical_Condition'].map(acuity_multipliers).fillna(1.0)

severity_multipliers = {
    'Minor': 0.85,
    'Moderate': 1.0,
    'Major': 1.30,
    'Extreme': 1.55
}
severity_mult = df['Severity_of_Illness'].map(severity_multipliers).fillna(1.0)
emergency_surcharge = np.where(df['Admission_Type'] == 'Emergency', 3500.0, 0.0)

base_charge = 11500.0
true_daily_marginal_cost = 3200.0
billing_shock = np.random.lognormal(mean=0, sigma=0.40, size=len(df))

# Compute realistic acute-care billed charges
df['Billing_Amount'] = np.round(
    (base_charge + emergency_surcharge + (df['Length_of_Stay'] * true_daily_marginal_cost))
    * condition_mult
    * severity_mult
    * billing_shock,
    2
)
df['Billing_Per_Day'] = df['Billing_Amount'] / df['Length_of_Stay']

# ==========================================
# 4. CAUSAL STUDY SETUP: EXTENDED STAYS
# ==========================================
print("Setting up Doubly Robust Causal Study Design...")
los_median = df['Length_of_Stay'].median()
df['Extended_Stay'] = (df['Length_of_Stay'] > los_median).astype(int)

covariates = ['Age_Group', 'Medical_Condition', 'Severity_of_Illness', 'Insurance', 'Admission_Type']

# CRITICAL FIX: Explicitly enforce dtype=float on one-hot encoding
W = pd.get_dummies(df[covariates], drop_first=True, dtype=float)

T = df['Extended_Stay'].values.astype(int)
Y = df['Billing_Amount'].values.astype(float)

# ==========================================
# 5. CAUSAL ESTIMATION: DOUBLY ROBUST (AIPW)
# ==========================================
print("Fitting Augmented Inverse Probability Weighting (AIPW) Pipeline...")

# Step A: Propensity Score Model e(W) = P(T=1|W)
propensity_model = LogisticRegression(max_iter=1000, random_state=42)
propensity_model.fit(W.values, T)
e_hat = propensity_model.predict_proba(W.values)[:, 1]

# Positivity / Common Support Guardrail (trimming probabilities to [0.01, 0.99])
e_hat = np.clip(e_hat, 0.01, 0.99)
df['Propensity_Score'] = e_hat

# Step B: Outcome Regression Models mu_0(W) and mu_1(W)
control_mask = (T == 0)
treated_mask = (T == 1)


def fit_subgroup_model(y_sub, w_sub, w_full):
    # Select non-constant columns in this subgroup
    std_devs = w_sub.std(axis=0)
    valid_cols = std_devs[std_devs > 0].index.tolist()

    # Extract strictly as float64 numpy matrices
    x_sub = sm.add_constant(w_sub[valid_cols].to_numpy(dtype=np.float64), has_constant='add')
    x_full = sm.add_constant(w_full[valid_cols].to_numpy(dtype=np.float64), has_constant='add')
    y_sub_arr = np.asarray(y_sub, dtype=np.float64)

    # Fit Gamma GLM with Log Link (standard in health econometrics)
    try:
        model = sm.GLM(
            y_sub_arr,
            x_sub,
            family=sm.families.Gamma(link=sm.families.links.Log())
        ).fit(disp=False)
        preds = model.predict(x_full)
    except Exception:
        # Fallback: OLS on log(Y)
        log_y = np.log(np.maximum(y_sub_arr, 1.0))
        ols = sm.OLS(log_y, x_sub).fit()
        preds = np.exp(ols.predict(x_full))

    return np.asarray(preds, dtype=np.float64)


mu_0_hat = fit_subgroup_model(Y[control_mask], W.loc[control_mask], W)
mu_1_hat = fit_subgroup_model(Y[treated_mask], W.loc[treated_mask], W)

# Step C: Doubly Robust Score (AIPW Efficient Influence Function)
aipw_scores = (mu_1_hat - mu_0_hat) + (T * (Y - mu_1_hat) / e_hat) - ((1.0 - T) * (Y - mu_0_hat) / (1.0 - e_hat))

att_ate = float(np.mean(aipw_scores))
ate_se = float(np.std(aipw_scores, ddof=1) / np.sqrt(len(df)))
t_stat = att_ate / ate_se
p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))

# Average additional days associated with extended stay
mean_extra_days = float(df[df['Extended_Stay'] == 1]['Length_of_Stay'].mean() - df[df['Extended_Stay'] == 0]['Length_of_Stay'].mean())
implied_cost_per_day = att_ate / mean_extra_days if mean_extra_days > 0 else 0.0

# The naive (unconditional) difference in means -- what a reader would get by
# just comparing average bills for extended- vs standard-stay patients,
# with no adjustment for confounding. Plotted next to the AIPW estimate in
# Scene 3 to make concrete why the causal adjustment matters.
naive_diff = float(Y[treated_mask].mean() - Y[control_mask].mean())

print("\n--- CAUSAL MODEL DIAGNOSTICS & EVALUATION ---")
print(f"Treatment Definition: Extended Stay (LOS > {los_median} days, mean difference: {mean_extra_days:.1f} days)")
print(f"Naive (unadjusted) difference in means: ${naive_diff:,.2f}")
print(f"Doubly Robust ATE (Average Treatment Effect): ${att_ate:,.2f}")
print(f"Standard Error: ${ate_se:,.2f} | p-value: {p_value:.4e}")
print(f"Implied Marginal Causal Cost / Day: ${implied_cost_per_day:,.2f}")
print(f"Propensity Score Overlap: Min={e_hat.min():.3f}, Max={e_hat.max():.3f}")

# Save JSON metadata for UI consumption
model_metrics = {
    "causal_impact_extended_stay": round(att_ate, 2),
    "naive_diff_extended_stay": round(naive_diff, 2),
    "implied_marginal_cost_per_day": round(implied_cost_per_day, 2),
    "mean_extra_days": round(mean_extra_days, 1),
    "p_value": round(p_value, 4),
    "ate_standard_error": round(ate_se, 2),
    "identification_strategy": "Augmented Inverse Probability Weighting (AIPW) with Gamma GLM",
    "positivity_min": round(float(e_hat.min()), 3),
    "positivity_max": round(float(e_hat.max()), 3)
}
with open(f"{output_dir}/model_metrics.json", "w") as f:
    json.dump(model_metrics, f)

# Each patient's AIPW-modeled expected bill GIVEN their actual treatment
# status (mu_1_hat for extended-stay patients, mu_0_hat for standard-stay
# patients) -- i.e. what the causal model predicts for THIS patient. Using
# mu_0_hat alone here (the "no extended stay" prediction) would offset
# every extended-stay patient upward by roughly the treatment effect --
# that's the causal effect working as intended, not a billing shock, and
# plotting against it would make Scene 4 misleading.
expected_bill = np.where(df['Extended_Stay'] == 1, mu_1_hat, mu_0_hat)
df['Model_Expected_Bill'] = expected_bill
df['Billing_Residual'] = df['Billing_Amount'] - expected_bill

# Flag the top 1% unexpected billing spikes
residual_threshold = df['Billing_Residual'].quantile(0.99)
df['Is_Outlier'] = df['Billing_Residual'] > residual_threshold

# ==========================================
# 6. SCENE 1 -- THE BILLING BLACK BOX (RIDGELINE)
# ==========================================
print("Generating Scene 1: billing distribution ridgeline...")


def build_ridgeline_scene1(df, output_dir):
    counts = df['Medical_Condition'].value_counts()
    # Keep conditions with enough support for a stable density estimate;
    # cap at 8 rows so the ridgeline stays legible rather than cluttered.
    top_conditions = counts[counts >= 30].index.tolist()[:8]
    if len(top_conditions) < 2:
        top_conditions = counts.index.tolist()[:8]

    medians = (
        df[df['Medical_Condition'].isin(top_conditions)]
        .groupby('Medical_Condition')['Billing_Amount']
        .median()
    )
    # Cheapest at the bottom, most expensive at the top -- the reader's eye
    # scrolls upward into the most dramatic (and most costly) conditions.
    ordered = medians.sort_values(ascending=True).index.tolist()

    lo, hi = df['Billing_Amount'].quantile([0.01, 0.97])
    x_grid = np.linspace(max(lo, 0), hi, 220)

    row_gap = 1.0
    height_scale = 1.15
    ridge_colors = sequential_gradient(RIDGE_GRADIENT_STOPS, len(ordered))

    traces = []
    tickvals, ticktext = [], []
    for i, cond in enumerate(ordered):
        vals = df.loc[df['Medical_Condition'] == cond, 'Billing_Amount'].dropna().values
        if len(vals) < 5:
            continue
        kde = gaussian_kde(vals)
        density = kde(x_grid)
        density_norm = density / density.max()
        baseline = i * row_gap
        y_top = baseline + density_norm * height_scale
        poly_x = np.concatenate([x_grid, x_grid[::-1]])
        poly_y = np.concatenate([y_top, np.full_like(x_grid, baseline)])
        color = ridge_colors[i]
        median_val = float(np.median(vals))
        p90 = float(np.percentile(vals, 90))

        traces.append(go.Scatter(
            x=poly_x, y=poly_y, fill='toself',
            fillcolor=hex_to_rgba(color, 0.82),
            line=dict(color=color, width=1.6),
            mode='lines',
            name=cond,
            showlegend=False,
            hoverinfo='skip',
        ))
        # Invisible line along the ridge's own crest carries the hover, so
        # readers get per-condition stats without a cluttered marker layer.
        traces.append(go.Scatter(
            x=x_grid, y=y_top,
            mode='lines', line=dict(width=0), opacity=0,
            showlegend=False,
            hovertemplate=f"<b>{cond}</b><br>Median bill: ${median_val:,.0f}<br>90th pct: ${p90:,.0f}<extra></extra>",
        ))
        tickvals.append(baseline)
        ticktext.append(cond)

    layout = go.Layout(
        xaxis=dict(title='Billing Amount', tickprefix='$', separatethousands=True,
                    showgrid=False, zeroline=False),
        yaxis=dict(tickmode='array', tickvals=tickvals, ticktext=ticktext,
                    showgrid=False, zeroline=False, ticks='',
                    range=[-0.4, (len(ordered) - 1) * row_gap + height_scale + 0.3]),
        margin=dict(l=140, r=40, t=20, b=60),
        plot_bgcolor=PAPER, paper_bgcolor=PAPER,
    )
    fig = go.Figure(data=traces, layout=layout)
    fig.write_json(f"{output_dir}/scene1_billing_distribution.json")


build_ridgeline_scene1(df, output_dir)

# ==========================================
# 7. SCENE 2 -- WHERE DO SHOCKS HAPPEN? (GROUPED BOX PLOT)
# ==========================================
print("Generating Scene 2: insurance x admission variance...")


def _box_stats(sub):
    q1, med, q3 = sub.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lf = max(sub.min(), q1 - 1.5 * iqr)
    uf = min(sub.max(), q3 + 1.5 * iqr)
    return float(q1), float(med), float(q3), float(lf), float(uf)


def build_box_scene2(df, output_dir):
    preferred_order = ['Elective', 'Urgent', 'Emergency']
    present = df['Admission_Type'].dropna().unique().tolist()
    admission_types = [a for a in preferred_order if a in present]
    admission_types += [a for a in present if a not in admission_types]

    insurances = sorted(df['Insurance'].dropna().unique().tolist())
    accent = {'Elective': TEAL, 'Urgent': '#C4622C', 'Emergency': CRIMSON}

    traces = []
    for i, adm in enumerate(admission_types):
        xs, q1s, meds, q3s, lfs, ufs = [], [], [], [], [], []
        for ins in insurances:
            sub = df.loc[(df['Insurance'] == ins) & (df['Admission_Type'] == adm), 'Billing_Amount']
            if len(sub) < 10:
                continue
            q1, med, q3, lf, uf = _box_stats(sub)
            xs.append(ins)
            q1s.append(q1); meds.append(med); q3s.append(q3); lfs.append(lf); ufs.append(uf)
        if not xs:
            continue
        color = accent.get(adm, sequential_gradient(RIDGE_GRADIENT_STOPS, len(admission_types))[i])
        traces.append(go.Box(
            x=xs, q1=q1s, median=meds, q3=q3s, lowerfence=lfs, upperfence=ufs,
            name=adm, marker_color=color, line=dict(color=color, width=1.6),
            fillcolor=hex_to_rgba(color, 0.55),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        boxmode='group', boxgap=0.25, boxgroupgap=0.08,
        yaxis=dict(title='Billing Amount', tickprefix='$', separatethousands=True,
                    showgrid=True, gridcolor=GRID),
        xaxis=dict(title=None, showgrid=False),
        plot_bgcolor=PAPER, paper_bgcolor=PAPER,
        legend=dict(orientation='h', y=1.1, x=0),
        margin=dict(l=80, r=40, t=40, b=60),
    )
    fig.write_json(f"{output_dir}/scene2_insurance_variance.json")


build_box_scene2(df, output_dir)

# ==========================================
# 8. SCENE 3 -- THE BASELINE COST (NAIVE VS. AIPW + POSITIVITY CHECK)
# ==========================================
print("Generating Scene 3: causal comparison + propensity overlap...")


def build_causal_scene3(naive_diff, ate, e_hat, T, output_dir):
    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.55, 0.45],
        vertical_spacing=0.24,
        subplot_titles=('Naive comparison vs. doubly-robust estimate', 'Propensity score common support'),
    )
    fig.add_trace(go.Bar(
        y=['Naive raw<br>difference', 'AIPW causal<br>estimate'],
        x=[naive_diff, ate],
        orientation='h',
        marker_color=[MUTED, CRIMSON],
        text=[f'${naive_diff:,.0f}', f'${ate:,.0f}'],
        textposition='outside',
        hovertemplate='%{y}: $%{x:,.2f}<extra></extra>',
        showlegend=False,
    ), row=1, col=1)

    bins = np.linspace(0, 1, 41)
    treated_counts, edges = np.histogram(e_hat[T == 1], bins=bins)
    control_counts, _ = np.histogram(e_hat[T == 0], bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0]) * 0.92

    fig.add_trace(go.Bar(x=centers, y=control_counts, name='Standard stay',
                          marker_color=hex_to_rgba(TEAL, 0.75), width=width), row=2, col=1)
    fig.add_trace(go.Bar(x=centers, y=treated_counts, name='Extended stay',
                          marker_color=hex_to_rgba(CRIMSON, 0.75), width=width), row=2, col=1)

    bar_max = max(naive_diff, ate)
    fig.update_xaxes(title_text='Effect on total bill', tickprefix='$', separatethousands=True,
                      row=1, col=1, showgrid=False, range=[0, bar_max * 1.22])
    fig.update_yaxes(automargin=True, tickfont=dict(size=14), row=1, col=1)
    fig.update_xaxes(title_text='Estimated propensity P(Extended Stay=1 | W)', row=2, col=1,
                      range=[0, 1], showgrid=False)
    fig.update_yaxes(title_text='Patients', row=2, col=1, showgrid=True, gridcolor=GRID)

    fig.update_layout(
        barmode='overlay',
        plot_bgcolor=PAPER, paper_bgcolor=PAPER,
        margin=dict(l=190, r=90, t=50, b=50),
        legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center'),
    )
    fig.write_json(f"{output_dir}/scene3_causal_coefficients.json")


build_causal_scene3(naive_diff, att_ate, e_hat, T, output_dir)

# ==========================================
# 9. SCENE 4 -- THE UNEXPLAINED RESIDUALS (DENSITY + OUTLIER OVERLAY)
# ==========================================
print("Generating Scene 4: counterfactual density + outlier overlay...")


def build_residual_scene4(df, output_dir):
    x = df['Model_Expected_Bill'].values
    y = df['Billing_Amount'].values
    is_out = df['Is_Outlier'].values

    xb, yb = x[~is_out], y[~is_out]
    x_lo, x_hi = np.percentile(xb, [0.5, 99.7])
    y_lo, y_hi = np.percentile(yb, [0.5, 99.7])
    heat, xedges, yedges = np.histogram2d(xb, yb, bins=48, range=[[x_lo, x_hi], [y_lo, y_hi]])
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    heat_masked = np.where(heat == 0, np.nan, heat)

    heatmap_trace = go.Heatmap(
        x=xcenters, y=ycenters, z=heat_masked.T,
        colorscale=[[0, '#E9E2D3'], [0.15, '#CBD9D6'], [0.45, '#7FA6AC'], [1, TEAL]],
        showscale=False,
        hovertemplate='Model-expected: $%{x:,.0f}<br>Actual: $%{y:,.0f}<br>Patients: %{z}<extra></extra>',
    )

    outliers = df[is_out]
    # Shared plot extent: generous enough to show most outliers taking
    # flight off the diagonal, without letting the single most extreme
    # point squash the "typical patient" density cloud into a corner.
    shared_hi = float(max(x_hi, y_hi, np.percentile(outliers['Billing_Amount'], 88)))
    n_clipped = int((outliers['Billing_Amount'] > shared_hi).sum())

    outlier_trace = go.Scatter(
        x=outliers['Model_Expected_Bill'], y=outliers['Billing_Amount'],
        mode='markers',
        marker=dict(size=7.5, color=CRIMSON, line=dict(color='white', width=1), opacity=0.92),
        name='Unexplained shocks (top 1%)',
        hovertemplate='Model-expected: $%{x:,.0f}<br>Actual: $%{y:,.0f}<extra></extra>',
    )
    lims = [0, shared_hi]
    diag = go.Scatter(x=lims, y=lims, mode='lines', line=dict(dash='dash', color=MUTED, width=1.5),
                       hoverinfo='skip', showlegend=False)

    fig = go.Figure(data=[heatmap_trace, diag, outlier_trace])
    annotations = []
    if n_clipped > 0:
        annotations.append(dict(
            text=f"+{n_clipped} more shock{'s' if n_clipped != 1 else ''} beyond this view (up to ${outliers['Billing_Amount'].max():,.0f})",
            xref='paper', yref='paper', x=0.98, y=0.04, showarrow=False,
            font=dict(size=12, color=MUTED), xanchor='right',
        ))
    # The outcome model's covariates (Age_Group, Medical_Condition,
    # Severity_of_Illness, Insurance, Admission_Type) are all categorical, so
    # "Model-expected bill" only takes a finite set of values -- one per
    # covariate combination actually present in the data. Blank stretches of
    # this axis are real gaps between those risk-profile tiers, not missing
    # data or a binning artifact -- worth calling out so a reader doesn't
    # mistake white space here for an error. Anchored well above and to the
    # right of the density cloud/outlier cluster (which top out well below
    # the shared axis max) rather than near the diagonal or the "+N more
    # shocks" note, so it reads clearly without competing with either.
    annotations.append(dict(
        text="Gaps here are real, not missing data —<br>the model predicts from a finite set<br>of risk profiles, not a continuum.",
        xref='paper', yref='paper', x=0.85, y=0.45, showarrow=False,
        font=dict(size=11, color=MUTED), xanchor='center', align='center',
    ))
    fig.update_layout(
        xaxis=dict(title='Model-expected bill (given actual stay length)', tickprefix='$', separatethousands=True,
                    showgrid=False, range=lims),
        yaxis=dict(title='Actual billed amount', tickprefix='$', separatethousands=True,
                    showgrid=True, gridcolor=GRID, range=lims),
        plot_bgcolor=PAPER, paper_bgcolor=PAPER,
        margin=dict(l=90, r=40, t=50, b=60),
        legend=dict(orientation='h', y=1.1, x=0),
        annotations=annotations,
    )
    fig.write_json(f"{output_dir}/scene4_residual_outliers.json")


build_residual_scene4(df, output_dir)

print(f"Pipeline Complete! All calibrated assets saved to ./{output_dir}")
