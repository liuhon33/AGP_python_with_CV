# ============================================================
# UKB PROJECTION + REGRESSION SCRIPT (UPDATED)
#
# What this version guarantees:
# 1) Loads your saved FINAL RF models + training metadata
# 2) Builds UKB X with EXACT training columns (missing cols -> NaN)
# 3) Coerces to numeric (strings -> NaN)
# 4) Uses the *trained* SimpleImputer inside each saved RF pipeline:
#    - If a whole column is missing in UKB (e.g., dog/cat), it becomes NaN,
#      and is imputed to the AGP median (constant for all UKB participants).
#    - If some entries are missing, those entries get AGP-median imputation.
# 5) Predicts PC_hat for UKB
# 6) Runs OLS regression: cognition ~ PC_hat (+ optional covariates)
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm


# 0) Config
model_dir = "./RF_PCA_Trained_Models/"  # where you saved final_*.pkl and final_training_metadata.json

ukb_metadata_path = "variable_mapping/ukb_as_agp_metadata.filtered_93pct_complete.csv"
ukb_index_col = "sample_name"  # your file uses sample_name

# Your outcome column(s)
cognition_col = "fluid_intelligence_score"      # <-- change if needed
# optional alternative outcome example:
# cognition_col = "mean_match_rt_ms"

# Optional covariates to include in regression (must exist in UKB file)
# Keep [] if you want cognition ~ PCs only.
reg_covars = []    # <-- edit as you like, or set to []

out_dir = "./UKB_RF_PCA_Projection/"
os.makedirs(out_dir, exist_ok=True)

# 1) LOAD SAVED MODELS + TRAINING METADATA
final_rf_models = joblib.load(os.path.join(model_dir, "final_rf_models.pkl"))

with open(os.path.join(model_dir, "final_training_metadata.json"), "r") as f:
    train_meta = json.load(f)

train_cols = train_meta["numeric_cols_used_as_X"]          # EXACT X columns used in AGP training
n_pcs_to_predict = int(train_meta["n_pcs_to_predict"])

print(f"Loaded {len(final_rf_models)} RF models.")
print(f"Training expects {len(train_cols)} numeric metadata columns.")
print(f"Will predict {n_pcs_to_predict} PCs.")

# 2) LOAD UKB METADATA (AGP-LIKE)
ukb_df = pd.read_csv(ukb_metadata_path, index_col=ukb_index_col)
print(f"Loaded UKB-like metadata: {ukb_df.shape[0]} rows, {ukb_df.shape[1]} columns.")

if cognition_col not in ukb_df.columns:
    raise ValueError(f"cognition_col='{cognition_col}' not found. Available cols include: {list(ukb_df.columns)[:30]} ...")

missing_covars = [c for c in reg_covars if c not in ukb_df.columns]
if missing_covars:
    raise ValueError(f"Missing regression covariates in UKB dataframe: {missing_covars}")

# 3) BUILD UKB X MATRIX WITH EXACT TRAINING COLUMNS
#    - missing columns become NaN (so imputer can handle)
#    - strings are coerced to NaN

X_ukb = ukb_df.reindex(columns=train_cols)

# Coerce to numeric: any strings like "United Kingdom" -> NaN
X_ukb = X_ukb.apply(pd.to_numeric, errors="coerce")

# Report how many entire columns are missing from UKB
missing_cols = [c for c in train_cols if c not in ukb_df.columns]
print(f"UKB is missing {len(missing_cols)} / {len(train_cols)} training columns.")
if len(missing_cols) > 0:
    print("First 25 missing training columns:", missing_cols[:25])

# Report overall missingness in the matrix (after coercion)
missing_rate = float(np.mean(pd.isna(X_ukb.values)))
print(f"Overall missing rate in X_ukb (after numeric coercion): {missing_rate*100:.2f}%")

# 4) PREDICT PC_hat FOR UKB
pc_hat = pd.DataFrame(index=ukb_df.index)

for k in range(n_pcs_to_predict):
    pc_name = f"PC{k+1}"
    if pc_name not in final_rf_models:
        raise KeyError(f"Model for {pc_name} not found in final_rf_models.pkl (keys={list(final_rf_models.keys())[:5]}...)")
    pc_hat[f"{pc_name}_hat"] = final_rf_models[pc_name].predict(X_ukb)

pc_hat_path = os.path.join(out_dir, "ukb_predicted_microbiome_pcs.csv")
pc_hat.to_csv(pc_hat_path)
print(f"Saved predicted PCs to: {pc_hat_path}")

# 5) REGRESSION: cognition ~ PC_hat 
y = pd.to_numeric(ukb_df[cognition_col], errors="coerce")

X_reg = pc_hat.copy()
if len(reg_covars) > 0:
    # IMPORTANT: covars might be coded as strings in your file; coerce them too
    cov_df = ukb_df[reg_covars].apply(pd.to_numeric, errors="coerce")
    X_reg = pd.concat([X_reg, cov_df], axis=1)

X_reg = sm.add_constant(X_reg, has_constant="add")

# Drop missing rows
data = pd.concat([y.rename("y"), X_reg], axis=1).dropna()
y_clean = data["y"]
X_clean = data.drop(columns=["y"])

fit = sm.OLS(y_clean, X_clean).fit(cov_type="HC3")
print(fit.summary())

summary_path = os.path.join(out_dir, "ukb_cognition_on_predicted_pcs_summary.txt")
with open(summary_path, "w") as f:
    f.write(fit.summary().as_text())
print(f"Saved regression summary to: {summary_path}")

# 6) BASELINE MODEL (covariates only) Skip this for now as I am not putting any covariates for now
if len(reg_covars) > 0:
    X_base = ukb_df[reg_covars].apply(pd.to_numeric, errors="coerce")
    X_base = sm.add_constant(X_base, has_constant="add")

    base_data = pd.concat([y.rename("y"), X_base], axis=1).dropna()
    fit_base = sm.OLS(base_data["y"], base_data.drop(columns=["y"])).fit(cov_type="HC3")

    print("\nBaseline model (covariates only):")
    print(fit_base.summary())

    base_path = os.path.join(out_dir, "ukb_baseline_covariates_only_summary.txt")
    with open(base_path, "w") as f:
        f.write(fit_base.summary().as_text())
    print(f"Saved baseline summary to: {base_path}")

print("\nUKB projection + regression complete.")

# negative test
# NEGATIVE CONTROL: Permute cognition scores (no covariates)
# - Uses pc_hat DataFrame (PC1_hat..PC10_hat)
# - Uses existing ukb_df with cognition_col
# - Repeats OLS on permuted y many times
# - Reports:
#   1) empirical p for the JOINT effect of all PCs (Wald test)
#   2) empirical p for the MIN single-PC p-value (multiple-testing-sensitive)
#   3) how often each PC is "significant" under permutation (optional)

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---- USER EDITS ----
cognition_col = "fluid_intelligence_score"   # <-- your column
B = 2000                                    # permutations (quick: 200-500; better: 2000-10000)
seed = 42

# ---- Build analysis dataframe once (locks the sample set) ----
pc_cols = [c for c in pc_hat.columns if c.endswith("_hat")]  # expects PC1_hat..PC10_hat

df = pd.concat(
    [ukb_df[[cognition_col]].rename(columns={cognition_col: "y"}),
     pc_hat[pc_cols]],
    axis=1
).copy()

# Coerce numeric
for c in ["y"] + pc_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna()
y = df["y"]
X = df[pc_cols]
Xc = sm.add_constant(X, has_constant="add")

def fit_ols_hc3(yvec):
    return sm.OLS(yvec, Xc).fit(cov_type="HC3")

def joint_wald_stat(fit, cols_to_test):
    param_names = list(fit.params.index)
    R = np.zeros((len(cols_to_test), len(param_names)))
    for i, col in enumerate(cols_to_test):
        R[i, param_names.index(col)] = 1.0
    w = fit.wald_test(R)
    return float(np.asarray(w.statistic).squeeze())

# ---- Observed model ----
fit_obs = fit_ols_hc3(y)
obs_joint = joint_wald_stat(fit_obs, pc_cols)
obs_pvals = fit_obs.pvalues[pc_cols].copy()
obs_minp = float(obs_pvals.min())

print("Observed R2:", float(fit_obs.rsquared))
print("Observed joint Wald stat (all PCs):", obs_joint)
print("Observed min single-PC p-value:", obs_minp)
print("Observed p-values:\n", obs_pvals.sort_values())

# ---- Permutation test ----
rng = np.random.default_rng(seed)

perm_joint = np.empty(B)
perm_minp = np.empty(B)
perm_sig_counts = {c: 0 for c in pc_cols}  # optional: per-PC sig rate under null

y_vals = y.values

for b in range(B):
    y_perm = rng.permutation(y_vals)
    fit_b = fit_ols_hc3(y_perm)

    perm_joint[b] = joint_wald_stat(fit_b, pc_cols)
    pvals_b = fit_b.pvalues[pc_cols].values
    perm_minp[b] = float(np.min(pvals_b))

    # optional: count how often each PC is significant at 0.05 under null
    for col, pv in zip(pc_cols, pvals_b):
        if pv < 0.05:
            perm_sig_counts[col] += 1

# Empirical p-values
p_emp_joint = (1.0 + np.sum(perm_joint >= obs_joint)) / (B + 1.0)
p_emp_minp  = (1.0 + np.sum(perm_minp <= obs_minp)) / (B + 1.0)

print("\n--- Permutation results (y permuted) ---")
print(f"Permutations: {B}")
print(f"Empirical p (JOINT effect of PCs): {p_emp_joint:.4g}")
print(f"Empirical p (MIN single-PC p across PCs): {p_emp_minp:.4g}")

# Optional: per-PC null significance rates
sig_rates = {col: perm_sig_counts[col] / B for col in pc_cols}
sig_rates_series = pd.Series(sig_rates).sort_values(ascending=False)

print("\nNull significance rate per PC at alpha=0.05 (should be ~0.05 each):")
print(sig_rates_series)

# Optional: save permutation distributions
# pd.DataFrame({"perm_joint": perm_joint, "perm_minp": perm_minp}).to_csv("perm_null_distributions.csv", index=False)
