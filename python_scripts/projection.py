#!/usr/bin/env python
# coding: utf-8

# In[6]:


# UKB PROJECTION + REGRESSION SCRIPT (UPDATED)
# 1) Loads the saved FINAL RF models + training metadata
# 2) Builds UKB X with EXACT training columns (missing cols -> NaN)
# 3) Coerces to numeric (strings -> NaN)
# 4) Uses the *trained* SimpleImputer inside each saved RF pipeline:
#    - If a whole column is missing in UKB (e.g., dog/cat), it becomes NaN,
#      and is imputed to the AGP median (constant for all UKB participants).
#    - If some entries are missing, those entries get AGP-median imputation.
# 5) Predicts PC_hat for UKB
# 6) Runs OLS regression: cognition ~ PC_hat (+ optional covariates)

import os
import json
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


# 0) Config
model_dir = "./RF_PCA_Trained_Models/"  # where you saved final_*.pkl and final_training_metadata.json
ukb_metadata_path = "variable_mapping/ukb_as_agp_metadata.csv"
ukb_index_col = "sample_name"  # your file uses sample_name

# outcome variable
cognition_col = "fluid_intelligence_score"
# cognition_col = "mean_match_rt_ms"

# Optional covariates to include in regression
reg_covars = ["age_corrected", "race", "country_of_birth"]

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
ukb_df["age_corrected"] = pd.to_numeric(ukb_df["age_corrected"], errors="coerce")
print(f"Loaded UKB-like metadata: {ukb_df.shape[0]} rows, {ukb_df.shape[1]} columns.")

if cognition_col not in ukb_df.columns:
    raise ValueError(f"cognition_col='{cognition_col}' not found. Available cols include: {list(ukb_df.columns)[:30]} ...")

missing_covars = [c for c in reg_covars if c not in ukb_df.columns]
if missing_covars:
    raise ValueError(f"Missing regression covariates in UKB dataframe: {missing_covars}")


# In[7]:


# Build covariates ONCE (consistent across OLS/baseline/permutation/logit)

cov_num = ["age_corrected"]                 # continuous
cov_cat = ["race", "country_of_birth"]      # categorical

# ensure numeric
for c in cov_num:
    if c in ukb_df.columns:
        ukb_df[c] = pd.to_numeric(ukb_df[c], errors="coerce")

# build cov_df
cov_df_num = ukb_df[cov_num].apply(pd.to_numeric, errors="coerce")
cov_df_cat = pd.get_dummies(
    ukb_df[cov_cat].astype("string").fillna("MISSING"),
    prefix=cov_cat,
    drop_first=True,
    dtype=float
)

cov_df = pd.concat([cov_df_num, cov_df_cat], axis=1)
cov_cols = list(cov_df.columns)
cov_df.head(10)


# In[8]:


# ============================================================
# 2.5) ROW FILTERING BY MISSINGNESS
# - Decide which columns count toward per-row missingness
# - Optionally "skip" very sparse columns so they don't nuke N
# - Enforces outcome + covariates are present for regression
# ============================================================

# Columns to IGNORE in the row-missingness calculation
# (edit this list and re-run this cell to see N change)
skip_cols = [
    "sugar_sweetened_drink_frequency",
    "free_sugar_scaled_0_5",
    "artificial_sweeteners",
    "one_liter_of_water_a_day_frequency",
    "olive_oil",
    "prepared_meals_frequency",
    "ready_to_eat_meals_frequency",
    "probiotic_frequency",
    "whole_eggs",
]

# Keep rows with <= this fraction missing across "cols_check"
max_missing_frac = 0.0  # 10% missing allowed (i.e., >=90% complete)

def filter_rows_by_missingness(df, *, skip_cols=(), required_cols=(), max_missing_frac=0.10):
    skip_cols = list(skip_cols)
    required_cols = list(required_cols)

    # warn if skip columns aren't present
    missing_skip = [c for c in skip_cols if c not in df.columns]
    if missing_skip:
        print(f"[WARN] skip_cols not in dataframe (ignored): {missing_skip[:10]}{'...' if len(missing_skip)>10 else ''}")

    # columns that count toward per-row missingness
    cols_check = [c for c in df.columns if c not in set(skip_cols)]
    if len(cols_check) == 0:
        raise ValueError("After applying skip_cols, cols_check is empty. Reduce skip_cols.")

    missing_frac = df[cols_check].isna().mean(axis=1)

    n_in = df.shape[0]
    mask = missing_frac <= max_missing_frac
    df_f = df.loc[mask].copy()

    # enforce outcome/covariates present
    required_present = [c for c in required_cols if c in df_f.columns]
    df_f = df_f.dropna(subset=required_present)

    n_out = df_f.shape[0]

    report = {
        "n_in": n_in,
        "n_out": n_out,
        "dropped": n_in - n_out,
        "cols_check_n": len(cols_check),
        "missing_frac_summary": missing_frac.describe().to_dict()
    }

    return df_f, report


required_for_regression = [cognition_col] + cov_num

ukb_df, miss_report = filter_rows_by_missingness(
    ukb_df,
    skip_cols=skip_cols,
    required_cols=required_for_regression,
    max_missing_frac=max_missing_frac
)

print("\n[ROW FILTER REPORT]")
print(f"Input rows:   {miss_report['n_in']}")
print(f"Output rows:  {miss_report['n_out']}")
print(f"Dropped rows: {miss_report['dropped']}")
print(f"Cols counted toward missingness: {miss_report['cols_check_n']}")
print("Missingness fraction summary (before filtering):")
print(pd.Series(miss_report["missing_frac_summary"]))


# In[4]:


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

# 4) PREDICT PC_hat FOR UKB|
pc_hat = pd.DataFrame(index=ukb_df.index)

for k in range(n_pcs_to_predict):
    pc_name = f"PC{k+1}"
    if pc_name not in final_rf_models:
        raise KeyError(f"Model for {pc_name} not found in final_rf_models.pkl (keys={list(final_rf_models.keys())[:5]}...)")
    pc_hat[f"{pc_name}_hat"] = final_rf_models[pc_name].predict(X_ukb)

pc_hat_path = os.path.join(out_dir, "ukb_predicted_microbiome_pcs.csv")
pc_hat.to_csv(pc_hat_path)
print(f"Saved predicted PCs to: {pc_hat_path}")


# In[5]:


# ============================================================
# 5) OLS REGRESSION (HC3): cognition ~ PC_hat + covariates
# - age_corrected treated as CONTINUOUS
# - race + country_of_birth treated as CATEGORICAL (one-hot)
# - uses a single dropna() so sample is consistent
# ============================================================

# Outcome
y = pd.to_numeric(ukb_df[cognition_col], errors="coerce")

# Predicted PCs (ensure numeric)
pc_cols = [c for c in pc_hat.columns if c.endswith("_hat")]
pc_hat_num = pc_hat[pc_cols].apply(pd.to_numeric, errors="coerce")

# ---- Full design matrix: PCs + covariates ----
X_reg = pd.concat([pc_hat_num, cov_df], axis=1)

# Add intercept
X_reg = sm.add_constant(X_reg, has_constant="add")

# Clean inf / ensure numeric
X_reg = X_reg.replace([np.inf, -np.inf], np.nan)
X_reg = X_reg.apply(pd.to_numeric, errors="coerce")

# Single analysis dataframe (locks sample)
data = pd.concat([y.rename("y"), X_reg], axis=1).dropna()

print("Rows used in regression:", data.shape[0], "Cols:", data.shape[1])
if data.shape[0] == 0:
    raise ValueError("No rows left after dropna. Check missingness in y / PCs / covariates.")

y_clean = data["y"].astype(float)
X_clean = data.drop(columns=["y"]).astype(float)

fit = sm.OLS(y_clean, X_clean).fit(cov_type="HC3")
print(fit.summary())

summary_path = os.path.join(out_dir, "ukb_cognition_on_predicted_pcs_summary.txt")
with open(summary_path, "w") as f:
    f.write(fit.summary().as_text())
print(f"Saved regression summary to: {summary_path}")


# In[ ]:


tmp = pd.concat(
    [ukb_df[[cognition_col]].rename(columns={cognition_col:"y"}),
     ukb_df[["age_corrected"]]],
    axis=1
).apply(pd.to_numeric, errors="coerce").dropna()

fit_age_only = sm.OLS(tmp["y"], sm.add_constant(tmp[["age_corrected"]])).fit(cov_type="HC3")
print(fit_age_only.summary())


# In[ ]:


print(ukb_df["age_corrected"].dtype)
print(ukb_df["age_corrected"].describe())
print("nunique:", ukb_df["age_corrected"].nunique())


# In[ ]:


# 6) BASELINE MODEL (covariates only) — handles categorical covariates properly
if len(reg_covars) > 0:
    # Build covariate design matrix the SAME way as in the full model
    cov_df = pd.get_dummies(
        ukb_df[reg_covars].astype("string").fillna("MISSING"),
        prefix=reg_covars,
        drop_first=True,
        dtype=float
    )

    # Outcome
    y_base = pd.to_numeric(ukb_df[cognition_col], errors="coerce")

    # Design
    X_base = sm.add_constant(cov_df, has_constant="add")
    X_base = X_base.replace([np.inf, -np.inf], np.nan)

    base_data = pd.concat([y_base.rename("y"), X_base], axis=1).dropna()
    print("Rows used in BASELINE regression:", base_data.shape[0], "Cols:", base_data.shape[1])

    if base_data.shape[0] == 0:
        raise ValueError("Baseline model has 0 rows after dropna — check cognition_col missingness.")

    fit_base = sm.OLS(
        base_data["y"].astype(float),
        base_data.drop(columns=["y"]).astype(float)
    ).fit(cov_type="HC3")

    print("\nBaseline model (covariates only):")
    print(fit_base.summary())

    base_path = os.path.join(out_dir, "ukb_baseline_covariates_only_summary.txt")
    with open(base_path, "w") as f:
        f.write(fit_base.summary().as_text())
    print(f"Saved baseline summary to: {base_path}")


# In[ ]:


# ============================================================
# PERMUTATION TEST (Freedman–Lane): PCs add nothing beyond covariates
# - Fits baseline: y ~ covariates
# - Permutes baseline residuals
# - Re-fits full: y_perm ~ covariates + PC_hat
# - Tests joint Wald for PC terms + min single-PC p
# ============================================================

# config
B = 2000
seed = 42
rng = np.random.default_rng(seed)

pc_cols = [c for c in pc_hat.columns if c.endswith("_hat")]  # PC1_hat..PCk_hat

# ---- Build covariate dummies ONCE (must match your OLS regression encoding) ----
if len(reg_covars) > 0:
    cov_df = pd.get_dummies(
        ukb_df[reg_covars].astype("string").fillna("MISSING"),
        prefix=reg_covars,
        drop_first=True,
        dtype=float
    )
else:
    cov_df = pd.DataFrame(index=ukb_df.index)

# ---- Build one analysis dataframe (locks sample) ----
df = pd.concat(
    [
        ukb_df[[cognition_col]].rename(columns={cognition_col: "y"}),
        pc_hat[pc_cols].apply(pd.to_numeric, errors="coerce"),
        cov_df
    ],
    axis=1
).replace([np.inf, -np.inf], np.nan)

df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna()

y = df["y"].astype(float)

cov_cols = list(cov_df.columns)  # after get_dummies
X_base = sm.add_constant(df[cov_cols], has_constant="add") if cov_cols else sm.add_constant(pd.DataFrame(index=df.index), has_constant="add")
X_full = sm.add_constant(df[pc_cols + cov_cols], has_constant="add") if cov_cols else sm.add_constant(df[pc_cols], has_constant="add")

def fit_ols_hc3(yvec, Xmat):
    return sm.OLS(yvec, Xmat).fit(cov_type="HC3")

def joint_wald_stat(fit, cols_to_test):
    param_names = list(fit.params.index)
    R = np.zeros((len(cols_to_test), len(param_names)))
    for i, col in enumerate(cols_to_test):
        R[i, param_names.index(col)] = 1.0
    w = fit.wald_test(R, scalar=True)
    return float(w.statistic)

# ---- Observed (full) ----
fit_obs = fit_ols_hc3(y, X_full)
obs_joint = joint_wald_stat(fit_obs, pc_cols)
obs_pvals = fit_obs.pvalues[pc_cols].copy()
obs_minp = float(obs_pvals.min())

print("Observed R2 (full):", float(fit_obs.rsquared))
print("Observed joint Wald stat (PCs):", obs_joint)
print("Observed min single-PC p:", obs_minp)

# ---- Freedman–Lane permutations ----
# baseline fit: y ~ covariates
fit_base = fit_ols_hc3(y, X_base)
yhat = fit_base.fittedvalues
resid = y - yhat

perm_joint = np.empty(B)
perm_minp  = np.empty(B)
perm_sig_counts = {c: 0 for c in pc_cols}

resid_vals = resid.values

for b in range(B):
    resid_perm = rng.permutation(resid_vals)
    y_perm = yhat.values + resid_perm

    fit_b = fit_ols_hc3(y_perm, X_full)
    perm_joint[b] = joint_wald_stat(fit_b, pc_cols)

    pvals_b = fit_b.pvalues[pc_cols].values
    perm_minp[b] = float(np.min(pvals_b))

    for col, pv in zip(pc_cols, pvals_b):
        if pv < 0.05:
            perm_sig_counts[col] += 1

p_emp_joint = (1.0 + np.sum(perm_joint >= obs_joint)) / (B + 1.0)
p_emp_minp  = (1.0 + np.sum(perm_minp <= obs_minp)) / (B + 1.0)

print("\n--- Permutation results (Freedman–Lane) ---")
print(f"Permutations: {B}")
print(f"Empirical p (JOINT PCs | covariates): {p_emp_joint:.4g}")
print(f"Empirical p (MIN single-PC p): {p_emp_minp:.4g}")

sig_rates = pd.Series({col: perm_sig_counts[col] / B for col in pc_cols}).sort_values(ascending=False)
print("\nNull significance rate per PC at alpha=0.05 (should be ~0.05):")
print(sig_rates)


# In[ ]:


# ---- configure your binary outcomes here ----
binary_outcomes = [
    "ibs",
    "crohns_disease",
    "ulcerative_colitis",
    "dementia_alzheimers",
    "dementia_vascular",
    "dementia_other",
    "dementia_unspecified"
]

# --- Precompute predictors once (numeric PCs + encoded covariates) ---
pc_cols = [c for c in pc_hat.columns if c.endswith("_hat")]
pc_hat_num = pc_hat[pc_cols].apply(pd.to_numeric, errors="coerce")

if len(reg_covars) > 0:
    cov_df = pd.get_dummies(
        ukb_df[reg_covars].astype("string").fillna("MISSING"),
        prefix=reg_covars,
        drop_first=True,
        dtype=float
    )
    cov_cols = list(cov_df.columns)
else:
    cov_df = pd.DataFrame(index=ukb_df.index)
    cov_cols = []

# Optional: include covariates later (you already have reg_covars list)
# reg_covars = ["age_corrected", "sex", "bmi"]

def joint_wald_pvalue(fit, cols_to_test):
    param_names = list(fit.params.index)
    R = np.zeros((len(cols_to_test), len(param_names)))
    for i, col in enumerate(cols_to_test):
        if col in param_names:
            R[i, param_names.index(col)] = 1.0

    w = fit.wald_test(R, scalar=True)   # <- add this
    return float(w.pvalue)              # <- now already scalar
def fit_logit_with_fallback(y, X):
    """
    Try standard Logit; if it fails (separation, singular), fall back to L2-regularized fit.
    Returns (fit_result, used_regularized: bool)
    """
    try:
        res = sm.Logit(y, X).fit(disp=0, maxiter=200)
        # robust SE version:
        res_rob = res.get_robustcov_results(cov_type="HC3")
        return res_rob, False
    except Exception as e:
        # Regularized fallback (stabilizes rare outcomes)
        # alpha controls strength; start small
        try:
            res_reg = sm.Logit(y, X).fit_regularized(method="l1", alpha=0.0, disp=0)  # essentially no penalty
        except Exception:
            res_reg = sm.Logit(y, X).fit_regularized(method="l1", alpha=0.1, disp=0)  # mild penalty
        # fit_regularized doesn't support robust cov in the same way; return as-is
        return res_reg, True

logit_outcome_rows = []
logit_pc_rows = []

logit_outcome_rows = []
logit_pc_rows = []

def coerce_binary(series):
    """Return 0/1 int series or None if not binary."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    uniq = sorted(s.unique().tolist())
    if len(uniq) == 0:
        return None
    # accept {0,1} or {1,2} (convert to 0/1)
    if set(uniq).issubset({0, 1}):
        return pd.to_numeric(series, errors="coerce").astype(int)
    if set(uniq).issubset({1, 2}):
        return (pd.to_numeric(series, errors="coerce") - 1).astype(int)
    return None

for outcome in binary_outcomes:
    if outcome not in ukb_df.columns:
        print(f"[SKIP] {outcome} not found in ukb_df columns.")
        continue

    # Build modeling dataframe using precomputed numeric predictors
    df_log = pd.concat(
        [
            ukb_df[[outcome]].rename(columns={outcome: "y"}),
            pc_hat_num,
            cov_df
        ],
        axis=1
    ).replace([np.inf, -np.inf], np.nan)

    # Coerce y to binary
    y_bin = coerce_binary(df_log["y"])
    if y_bin is None:
        print(f"[SKIP] {outcome}: not a recognized binary coding (expected 0/1 or 1/2).")
        continue
    df_log["y"] = y_bin

    # Drop missing rows (now covariates won't be nuked)
    df_log = df_log.dropna(subset=["y"] + pc_cols + cov_cols)
    if df_log.shape[0] == 0:
        print(f"[SKIP] {outcome}: no complete rows after dropna.")
        continue

    y_bin = df_log["y"].astype(int)
    n = int(len(y_bin))
    n_cases = int(y_bin.sum())
    n_controls = n - n_cases

    if n_cases == 0 or n_controls == 0:
        print(f"[SKIP] {outcome}: degenerate (cases={n_cases}, controls={n_controls}).")
        continue

    # Design matrix: PCs + encoded covariates
    Xmat = df_log[pc_cols + cov_cols].astype(float)
    Xmat = sm.add_constant(Xmat, has_constant="add")

    # Fit
    fit_res, used_reg = fit_logit_with_fallback(y_bin, Xmat)

    # Joint PC test
    try:
        p_joint = joint_wald_pvalue(fit_res, pc_cols)
    except Exception:
        p_joint = np.nan

    logit_outcome_rows.append({
        "outcome": outcome,
        "n": n,
        "cases": n_cases,
        "controls": n_controls,
        "used_regularized_fit": used_reg,
        "p_joint_all_PCs": p_joint
    })

    # Per-PC outputs
    if hasattr(fit_res, "params"):
        for pc in pc_cols:
            if pc in fit_res.params.index:
                logit_pc_rows.append({
                    "outcome": outcome,
                    "predictor": pc,
                    "coef_log_odds": float(fit_res.params[pc]),
                    "p_value": float(fit_res.pvalues[pc]) if hasattr(fit_res, "pvalues") else np.nan
                })
# Convert to DataFrames
outcome_df = pd.DataFrame(logit_outcome_rows)
pc_df = pd.DataFrame(logit_pc_rows)

# Multiple-testing correction across outcomes (joint tests)
if outcome_df.shape[0] > 0:
    mask = outcome_df["p_joint_all_PCs"].notna()
    pvals = outcome_df.loc[mask, "p_joint_all_PCs"].values
    rej, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    outcome_df.loc[mask, "q_joint_all_PCs_BH"] = qvals
    outcome_df.loc[mask, "reject_FDR_0p05_joint"] = rej

# Optional: BH across ALL per-PC tests (outcome x PC)
if pc_df.shape[0] > 0 and pc_df["p_value"].notna().any():
    mask2 = pc_df["p_value"].notna()
    pvals2 = pc_df.loc[mask2, "p_value"].values
    rej2, qvals2, _, _ = multipletests(pvals2, alpha=0.05, method="fdr_bh")
    pc_df.loc[mask2, "q_value_BH"] = qvals2
    pc_df.loc[mask2, "reject_FDR_0p05"] = rej2

# Save results
logit_outcome_path = os.path.join(out_dir, "ukb_logistic_outcomes_joint_tests.csv")
logit_pc_path = os.path.join(out_dir, "ukb_logistic_pc_level_results.csv")

outcome_df.sort_values("p_joint_all_PCs").to_csv(logit_outcome_path, index=False)
pc_df.to_csv(logit_pc_path, index=False)

print("\n=== Logistic regression (PC_hat) complete ===")
print("Saved outcome-level joint tests to:", logit_outcome_path)
print("Saved PC-level results to:", logit_pc_path)
print("\nTop outcomes by joint p-value:")
print(outcome_df.sort_values("p_joint_all_PCs").head(10))


# In[ ]:




