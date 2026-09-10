import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datasets import load_dataset
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
import os
import json

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
print("🧹 Standardizing messy column names...")

# FIX 1: Strip any hidden leading/trailing spaces from the raw Hugging Face columns
df.columns = df.columns.str.strip()

# FIX 2: Catch multiple variations of the messy columns (spaces vs underscores)
column_mapping = {
    'Stay (in days)': 'Length_of_Stay',
    'Stay_(in_days)': 'Length_of_Stay',
    'Admission_Deposit': 'Billing_Amount',
    'Admission Deposit': 'Billing_Amount',
    'health conditions': 'Medical_Condition',
    'health_conditions': 'Medical_Condition',  # Catching the underscore variation
    'Type of Admission': 'Admission_Type',
    'Type_of_Admission': 'Admission_Type',
    'Age': 'Age_Group',
    'Severity of Illness': 'Severity_of_Illness',
    'Available Extra Rooms in Hospital':'Available_Extra_Rooms_in_Hospital',
     'Visitors with Patient':'Visitors_With_Patient'
}

df = df.rename(columns=column_mapping)

# Failsafe Print: Verify the rename worked before proceeding
required_cols = ['Medical_Condition', 'Insurance', 'Admission_Type', 'Age_Group', 'Length_of_Stay', 'Billing_Amount']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"🚨 CRITICAL ERROR: The following columns failed to rename: {missing_cols}")
    print(f"Here are the exact columns Hugging Face provided: {df.columns.tolist()}")
    raise KeyError("Column renaming failed. See console for actual column names.")

# Handle Missing Values safely
df['Medical_Condition'] = df['Medical_Condition'].fillna("Unknown")
df['Insurance'] = df['Insurance'].fillna("Unknown")
df['Age_Group'] = df['Age_Group'].fillna("Unknown")
df['Admission_Type'] = df['Admission_Type'].fillna("Unknown")


# ==========================================
# 3. FEATURE ENGINEERING (The "Marcus" Context)
# ==========================================
print("Engineering clinical and financial features...")

# Ensure numerical types and prevent division by zero
df['Length_of_Stay'] = pd.to_numeric(df['Length_of_Stay'], errors='coerce').fillna(1)
df['Length_of_Stay'] = df['Length_of_Stay'].replace(0, 1)

df['Billing_Amount'] = pd.to_numeric(df['Billing_Amount'], errors='coerce').fillna(0)

# Define the Shock: Billing Amount per Day
df['Billing_Per_Day'] = df['Billing_Amount'] / df['Length_of_Stay']

# Create Exogenous Controls (W)
# We dummify categorical variables to use in the regression
W = pd.get_dummies(df[['Age_Group', 'Medical_Condition','Severity_of_Illness', 'Insurance']], drop_first=True)
W = sm.add_constant(W) * 1.0  # Convert booleans to float and add constant

# Create the Instrument (Z): Emergency Admission
# We check if 'Emergency' exists, otherwise fallback to the first admission type dummy
adm_dummies = pd.get_dummies(df['Admission_Type'])
if 'Emergency' in adm_dummies.columns:
    Z = adm_dummies['Emergency'].astype(float)
else:
    print("'Emergency' not found in Admission_Type. Using fallback instrument.")
    Z = adm_dummies.iloc[:, 0].astype(float)

# Create Endogenous Variable (X): Length of Stay
X = df['Length_of_Stay'].astype(float)

# Dependent Variable (Y): Billing Amount
Y = df['Billing_Amount'].astype(float)


# ==========================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("Generating EDA Visualizations...")

# Scene 1: The Billing Black Box (Distribution of Costs)
fig_dist = px.histogram(
    df, x="Billing_Amount", color="Medical_Condition", 
    title="Scene 1: The Billing Distribution",
    marginal="box", hover_data=['Age_Group', 'Insurance']
)
fig_dist.write_json(f"{output_dir}/scene1_billing_distribution.json")

# Scene 2: The Shock Matrix (Insurance vs. Admission Type)
fig_box = px.box(
    df, x="Insurance", y="Billing_Amount", color="Admission_Type",
    title="Scene 2: Where do the shocks happen?"
)
fig_box.write_json(f"{output_dir}/scene2_insurance_variance.json")


# ==========================================
# 5. ECONOMETRIC MODELING (2SLS)
# ==========================================
print("Running 2-Stage Least Squares (2SLS) IV Model...")

# Testing if Length of Stay CAUSES the massive billing shocks
iv_model = IV2SLS(dependent=Y, exog=W, endog=X, instruments=Z).fit(cov_type='robust')

print("\n--- MODEL DIAGNOSTICS & EVALUATION ---")
f_stat = iv_model.first_stage.diagnostics['f.stat'].values[0]
print(f"Weak Instrument F-stat: {f_stat:.2f} (Should be > 10)")
print(f"Wu-Hausman p-value: {iv_model.wu_hausman().pval:.4f}")

# Save the model summary metrics as a JSON for the text portion of the UI
model_metrics = {
    "causal_impact_per_day": round(iv_model.params['Length_of_Stay'], 2),
    "p_value": round(iv_model.pvalues['Length_of_Stay'], 4),
    "weak_instrument_fstat": round(f_stat, 2)
}
with open(f"{output_dir}/model_metrics.json", "w") as f:
    json.dump(model_metrics, f)


# ==========================================
# 6. MODEL EXPLANATION VISUALS
# ==========================================
print("Generating Econometric Story Visuals...")

# Scene 3: The Coefficient Forest Plot
coefficients = iv_model.params
conf_ints = iv_model.conf_int()
features = coefficients.index

fig_coef = go.Figure()
fig_coef.add_trace(go.Scatter(
    x=coefficients.values,
    y=features,
    mode='markers',
    error_x=dict(type='data', 
                 symmetric=False, 
                 array=conf_ints['upper'] - coefficients, 
                 arrayminus=coefficients - conf_ints['lower']),
    marker=dict(size=10, color='crimson')
))
fig_coef.update_layout(
    title="Scene 3: The True Drivers of Financial Toxicity",
    xaxis_title="Causal Impact on Total Bill ($)",
    yaxis_title="Patient Variables",
    yaxis=dict(autorange="reversed") # Standard for forest plots
)
fig_coef.write_json(f"{output_dir}/scene3_causal_coefficients.json")

# Scene 4: The "Marcus" Outlier Highlight (Residuals)
df['Predicted_Bill'] = iv_model.predict()
df['Billing_Residual'] = df['Billing_Amount'] - df['Predicted_Bill']

# Flag the top 1% of unexpected bills
threshold = df['Billing_Residual'].quantile(0.99)
df['Is_Outlier'] = df['Billing_Residual'] > threshold

fig_resid = px.scatter(
    df, x="Predicted_Bill", y="Billing_Amount", 
    color="Is_Outlier",
    hover_data=['Age_Group', 'Medical_Condition', 'Severity_of_Illness', 'Insurance'],
    color_discrete_map={True: 'red', False: 'lightgrey'},
    title="Scene 4: Identifying 'Marcus' - The Unexplained Shocks"
)
# Add the line of perfect prediction
fig_resid.add_shape(
    type="line", line=dict(dash='dash'),
    x0=df['Predicted_Bill'].min(), y0=df['Predicted_Bill'].min(),
    x1=df['Predicted_Bill'].max(), y1=df['Predicted_Bill'].max()
)
fig_resid.write_json(f"{output_dir}/scene4_residual_outliers.json")

print(f"Pipeline Complete! All assets saved to ./{output_dir}")