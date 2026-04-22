# compare my PC projection model with the regular basic model
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# ----------------------------
# CONFIG
# ----------------------------
os.chdir('/scratch/liuhon33/parallel/AGPMicrobiomeHostPredictions')
ukb_metadata_path = "variable_mapping/ukb_as_agp_metadata.filtered_93pct_complete.csv"
ukb_index_col = "sample_name"

cognition_col = "fluid_intelligence_score"   # <-- change if needed
# cognition_col = "mean_match_rt_ms"

# If you already saved predicted PCs from the projection script:
pc_hat_path = "./UKB_RF_PCA_Projection/ukb_predicted_microbiome_pcs.csv"  # <-- change if needed

# CV
n_splits = 5
seed = 42

# ----------------------------
# LOAD UKB DATA + PC_HAT
# ----------------------------
ukb_df = pd.read_csv(ukb_metadata_path, index_col=ukb_index_col)

if cognition_col not in ukb_df.columns:
    raise ValueError(f"Outcome column not found: {cognition_col}")

# Load predicted PCs (PC1_hat..PCk_hat)
pc_hat = pd.read_csv(pc_hat_path, index_col=ukb_index_col)

# Keep only PC*_hat columns
pc_cols = [c for c in pc_hat.columns if c.startswith("PC") and c.endswith("_hat")]
if len(pc_cols) == 0:
    raise ValueError("No PC*_hat columns found in pc_hat file.")

# Align indices (just in case)
common = ukb_df.index.intersection(pc_hat.index)
ukb_df = ukb_df.loc[common].copy()
pc_hat = pc_hat.loc[common].copy()

# Coerce outcome numeric
y = pd.to_numeric(ukb_df[cognition_col], errors="coerce")

# ----------------------------
# BUILD BASELINE X (NO PCs)
#   - use all numeric metadata columns except the outcome
#   - strings become NaN and will be imputed inside pipelines
# ----------------------------
X_base = ukb_df.drop(columns=[cognition_col]).copy()

# Coerce everything to numeric where possible (strings -> NaN)
X_base = X_base.apply(pd.to_numeric, errors="coerce")

# PC_hat features
X_pc = pc_hat[pc_cols].apply(pd.to_numeric, errors="coerce")

# Lock a single analysis dataset per comparison (drop rows with missing y)
# (X missing is OK; it will be imputed inside pipelines)
mask = ~y.isna()
y = y.loc[mask]
X_base = X_base.loc[mask]
X_pc = X_pc.loc[mask]

print(f"N used (non-missing outcome): {len(y)}")
print(f"Baseline X dims: {X_base.shape} | PC_hat dims: {X_pc.shape}")

# ----------------------------
# CV helper
# ----------------------------
def cv_r2(model_pipeline, X, y, cv):
    scores = []
    for tr, te in cv.split(X):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        model_pipeline.fit(X_tr, y_tr)
        pred = model_pipeline.predict(X_te)
        scores.append(r2_score(y_te, pred))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))

cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

# ----------------------------
# MODELS
# ----------------------------
# Baseline: X -> cognition
ridge_base = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", Ridge(alpha=1.0, random_state=seed))
])

rf_base = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=10
    ))
])

# Proxy: PC_hat -> cognition
ridge_pc = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", Ridge(alpha=1.0, random_state=seed))
])

rf_pc = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=10
    ))
])

# ----------------------------
# RUN CV
# ----------------------------
results = []

m, s = cv_r2(ridge_base, X_base, y, cv); results.append(("Baseline: Ridge (X→y)", m, s))
m, s = cv_r2(rf_base,    X_base, y, cv); results.append(("Baseline: RF (X→y)",    m, s))

m, s = cv_r2(ridge_pc,   X_pc,   y, cv); results.append(("Proxy: Ridge (PĈ→y)",  m, s))
m, s = cv_r2(rf_pc,      X_pc,   y, cv); results.append(("Proxy: RF (PĈ→y)",     m, s))

res_df = pd.DataFrame(results, columns=["Model", "CV_R2_mean", "CV_R2_sd"]).sort_values("CV_R2_mean", ascending=False)
print("\n=== Cross-validated predictive performance (R^2) ===")
print(res_df.to_string(index=False))

# Optional: save
res_df.to_csv("ukb_baseline_vs_proxy_cv_r2.csv", index=False)
print("\nSaved: ukb_baseline_vs_proxy_cv_r2.csv")
