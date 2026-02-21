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


# 0) Config
model_dir = "./RF_PCA_Trained_Models/"  # where you saved final_*.pkl and final_training_metadata.json

ukb_metadata_path = "variable_mapping/ukb_as_agp_metadata.filtered_90pct_complete.csv"
ukb_index_col = "sample_name"  # your file uses sample_name

# Your outcome column(s)
cognition_col = "fluid_intelligence_score"
# cognition_col = "mean_match_rt_ms"

# Optional covariates to include in regression (must exist in UKB file)
# use [] if no covariates
reg_covars = []

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

# ============================================================
# 2.5) ROW FILTERING BY MISSINGNESS
# - Decide which columns count toward per-row missingness
# - Optionally "skip" very sparse columns so they don't nuke N
# - Enforces outcome + covariates are present for regression
# ============================================================

# Columns you want to IGNORE in the row-missingness calculation
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
max_missing_frac = 0.10  # 10% missing allowed (i.e., >=90% complete)

def filter_rows_by_missingness(df, *, skip_cols=(), required_cols=(), max_missing_frac=0.10):
    skip_cols = list(skip_cols)
    required_cols = list(required_cols)

    # warn if skip columns aren't present (harmless)
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


# Decide what MUST be present for Stage 2 regression
required_for_regression = [cognition_col] + reg_covars

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

# ============================================================
# LOGISTIC REGRESSION ON BINARY OUTCOMES USING PC_hat
# - Fits: outcome ~ PC1_hat + ... + PCk_hat (+ optional covars)
# - Provides:
#   (A) Joint Wald test p-value for all PCs (recommended primary test)
#   (B) Per-PC coefficients and p-values (secondary / exploratory)
# - Adds BH-FDR correction for multiple outcomes (and optionally for all PC tests)
# ============================================================

from statsmodels.stats.multitest import multipletests

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

pc_cols = [c for c in pc_hat.columns if c.endswith("_hat")]  # PC1_hat..PCk_hat

# Optional: include covariates later (you already have reg_covars list)
# reg_covars = ["age_corrected", "sex", "bmi"]

def joint_wald_pvalue(fit, cols_to_test):
    """H0: all coefficients of cols_to_test are 0 (joint Wald test)."""
    param_names = list(fit.params.index)
    R = np.zeros((len(cols_to_test), len(param_names)))
    for i, col in enumerate(cols_to_test):
        if col in param_names:
            R[i, param_names.index(col)] = 1.0
    w = fit.wald_test(R)
    return float(np.asarray(w.pvalue).squeeze())

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

for outcome in binary_outcomes:
    if outcome not in ukb_df.columns:
        print(f"[SKIP] {outcome} not found in ukb_df columns.")
        continue

    # Build modeling dataframe
    df_log = pd.concat(
        [
            ukb_df[[outcome]].rename(columns={outcome: "y"}),
            pc_hat[pc_cols],
            ukb_df[reg_covars] if len(reg_covars) else pd.DataFrame(index=ukb_df.index)
        ],
        axis=1
    ).copy()

    # Coerce numeric
    for c in ["y"] + pc_cols + list(reg_covars):
        df_log[c] = pd.to_numeric(df_log[c], errors="coerce")

    df_log = df_log.dropna()
    if df_log.shape[0] == 0:
        print(f"[SKIP] {outcome}: no complete rows after dropna.")
        continue

    y_bin = df_log["y"].astype(int)
    n = int(len(y_bin))
    n_cases = int(y_bin.sum())
    n_controls = n - n_cases

    # Skip degenerate outcomes
    if n_cases == 0 or n_controls == 0:
        print(f"[SKIP] {outcome}: degenerate (cases={n_cases}, controls={n_controls}).")
        continue

    # Design matrix
    X_list = [df_log[pc_cols]]
    if len(reg_covars) > 0:
        X_list.append(df_log[reg_covars])
    Xmat = pd.concat(X_list, axis=1)
    Xmat = sm.add_constant(Xmat, has_constant="add")

    # Fit
    fit_res, used_reg = fit_logit_with_fallback(y_bin, Xmat)

    # Joint PC test (primary)
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

    # Per-PC outputs (secondary)
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
