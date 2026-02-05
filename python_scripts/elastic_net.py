# ============================================================
# Nested-CV ELASTIC NET Regression to Predict Microbiome OTUs (Y)
# from Lifestyle/Metadata Features (X), with NO leakage
#
# - TRUE nested CV: outer evaluates; inner tunes (alpha, l1_ratio)
# - No pre-CV imputation: SimpleImputer inside Pipeline
# - Optional subject leakage prevention: GroupKFold if subject_id_col set
# - Saves outer-fold R2 mean/std and hyperparameter distributions
# ============================================================

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm  # if not in notebook, use: from tqdm import tqdm

from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

# -----------------------------
# 1. Configuration
# -----------------------------
otu_file_path = "./Data/Cleaned_data/AGP_Otu_Data.csv"
metadata_file_path = "./Data/Cleaned_data/processed_metadata.csv"

metadata_index_col = "sample_name"
otu_index_col = 0

output_dir = "./hongrui_result/ElasticNet_NestedCV_Results/"
os.makedirs(output_dir, exist_ok=True)

# CV config
n_splits_outer = 5
n_splits_inner = 5
random_state_cv = 42

# OTU filter config
abundance_threshold = 0.0001

# ElasticNet hyper-parameter grids
alphas_to_test = np.logspace(-3, 3, 7)          # lambda-like strength
l1_ratios_to_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # 0=L2, 1=L1
print(f"Alphas to test: {alphas_to_test}")
print(f"L1 ratios to test: {l1_ratios_to_test}")

# Optional grouping for leakage prevention
subject_id_col = None  # <-- CHANGE if available

# Target transformation choice
use_log1p_counts = True

# ElasticNet solver config (important for convergence)
enet_max_iter = 20000
enet_tol = 1e-4

# -----------------------------
# 2. Load and Prepare Data
# -----------------------------
print("--- Loading data ---")
metadata_df = pd.read_csv(metadata_file_path, index_col=metadata_index_col)
otu_df = pd.read_csv(otu_file_path, index_col=otu_index_col)

# Align samples
common_samples = metadata_df.index.intersection(otu_df.index)
metadata_df = metadata_df.loc[common_samples].copy()
otu_df = otu_df.loc[common_samples].copy()
print(f"Data aligned. Found {len(common_samples)} common samples.")
print(f"Initial number of OTUs: {otu_df.shape[1]}")

# Filter OTUs by mean relative abundance
otu_rel_abund = otu_df.div(otu_df.sum(axis=1), axis=0)
mean_rel_abund = otu_rel_abund.mean(axis=0)
otus_to_keep = mean_rel_abund[mean_rel_abund > abundance_threshold].index
otu_df_filtered = otu_df[otus_to_keep].copy()
print(f"Number of OTUs after filtering: {otu_df_filtered.shape[1]}")

# Define Y
if use_log1p_counts:
    Y = np.log1p(otu_df_filtered)
    print("Using Y = log1p(counts).")
else:
    otu_rel_abund_filt = otu_df_filtered.div(otu_df_filtered.sum(axis=1), axis=0)
    Y = np.log1p(otu_rel_abund_filt)
    print("Using Y = log1p(relative abundance).")

# Define X (numeric only)
numeric_cols = metadata_df.select_dtypes(include=np.number).columns.tolist()
X = metadata_df[numeric_cols].copy()
print(f"Using {X.shape[1]} numeric metadata features as predictors.")

# Groups (optional)
groups = None
if subject_id_col is not None:
    if subject_id_col not in metadata_df.columns:
        raise ValueError(
            f"subject_id_col='{subject_id_col}' not found in metadata_df columns. "
            f"Available cols include: {list(metadata_df.columns)[:20]} ..."
        )
    groups = metadata_df[subject_id_col].values
    print(f"Using GroupKFold with groups from column: {subject_id_col}")

# -----------------------------
# 3. Helpers
# -----------------------------
def make_pipeline(alpha: float, l1_ratio: float) -> Pipeline:
    """Pipeline that avoids leakage: impute + scale + elasticnet."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("enet", ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=enet_max_iter,
            tol=enet_tol
        ))
    ])

def get_cv_splitter(n_splits: int, use_groups: bool):
    """Return KFold or GroupKFold splitter."""
    if use_groups:
        return GroupKFold(n_splits=n_splits)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state_cv)

# -----------------------------
# 4. Nested CV per OTU
# -----------------------------
print(f"\n--- Running Nested CV ELASTIC NET (outer={n_splits_outer}, inner={n_splits_inner}) ---")
use_groups = groups is not None

outer_splitter = get_cv_splitter(n_splits_outer, use_groups=use_groups)
inner_splitter = get_cv_splitter(n_splits_inner, use_groups=use_groups)

otu_results = []

best_scatter = {
    "otu": None,
    "y_true": None,
    "y_pred": None,
    "r2": None,
    "alpha": None,
    "l1_ratio": None
}

for otu_name in tqdm(Y.columns, desc="Processing OTUs"):
    y = Y[otu_name].astype(float)

    if np.nanvar(y.values) < 1e-9:
        otu_results.append({
            "OTU": otu_name,
            "outer_r2_mean": np.nan,
            "outer_r2_std": np.nan,
            "outer_r2_median": np.nan,
            "outer_best_alpha_mode": np.nan,
            "outer_best_alpha_mean": np.nan,
            "outer_best_l1_ratio_mode": np.nan,
            "outer_best_l1_ratio_mean": np.nan,
            "outer_best_params_list": "",
            "n_samples": len(y)
        })
        continue

    outer_r2_scores = []
    outer_best_params = []  # list of (alpha, l1_ratio)

    if use_groups:
        outer_splits = list(outer_splitter.split(X, y, groups=groups))
    else:
        outer_splits = list(outer_splitter.split(X, y))

    for ofold, (tr_idx, te_idx) in enumerate(outer_splits):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        best_alpha = None
        best_l1_ratio = None
        best_inner_score = -np.inf

        if use_groups:
            g_tr = groups[tr_idx]
            inner_splits = list(inner_splitter.split(X_tr, y_tr, groups=g_tr))
        else:
            inner_splits = list(inner_splitter.split(X_tr, y_tr))

        # Grid search over (alpha, l1_ratio)
        for alpha in alphas_to_test:
            for l1r in l1_ratios_to_test:
                inner_scores = []
                for itr_idx, iva_idx in inner_splits:
                    X_itr, X_iva = X_tr.iloc[itr_idx], X_tr.iloc[iva_idx]
                    y_itr, y_iva = y_tr.iloc[itr_idx], y_tr.iloc[iva_idx]

                    pipe = make_pipeline(float(alpha), float(l1r))
                    pipe.fit(X_itr, y_itr)
                    pred_iva = pipe.predict(X_iva)
                    inner_scores.append(r2_score(y_iva, pred_iva))

                inner_mean = float(np.mean(inner_scores))
                if inner_mean > best_inner_score:
                    best_inner_score = inner_mean
                    best_alpha = float(alpha)
                    best_l1_ratio = float(l1r)

        # Fit on full outer-train with best params; evaluate on outer-test
        final_pipe = make_pipeline(best_alpha, best_l1_ratio)
        final_pipe.fit(X_tr, y_tr)
        pred_te = final_pipe.predict(X_te)
        r2_te = float(r2_score(y_te, pred_te))

        outer_r2_scores.append(r2_te)
        outer_best_params.append((best_alpha, best_l1_ratio))

        last_fold_true = y_te
        last_fold_pred = pred_te
        last_fold_r2 = r2_te
        last_fold_alpha = best_alpha
        last_fold_l1_ratio = best_l1_ratio

    # Summaries
    outer_r2_scores_arr = np.array(outer_r2_scores, dtype=float)
    r2_mean = float(np.mean(outer_r2_scores_arr))
    r2_std = float(np.std(outer_r2_scores_arr, ddof=1)) if len(outer_r2_scores_arr) > 1 else 0.0
    r2_median = float(np.median(outer_r2_scores_arr))

    params_arr = np.array(outer_best_params, dtype=float)  # shape (n_outer, 2)
    alphas_arr = params_arr[:, 0]
    l1r_arr = params_arr[:, 1]

    # "mode" for (alpha, l1_ratio) tuple across outer folds (tie-breaker: smaller alpha, then smaller l1_ratio)
    unique_rows, counts = np.unique(params_arr, axis=0, return_counts=True)
    max_count = counts.max()
    candidates = unique_rows[counts == max_count]
    candidates = candidates[np.lexsort((candidates[:, 1], candidates[:, 0]))]
    alpha_mode, l1r_mode = float(candidates[0, 0]), float(candidates[0, 1])

    otu_results.append({
        "OTU": otu_name,
        "outer_r2_mean": r2_mean,
        "outer_r2_std": r2_std,
        "outer_r2_median": r2_median,
        "outer_best_alpha_mode": alpha_mode,
        "outer_best_alpha_mean": float(np.mean(alphas_arr)),
        "outer_best_l1_ratio_mode": l1r_mode,
        "outer_best_l1_ratio_mean": float(np.mean(l1r_arr)),
        "outer_best_params_list": ",".join([f"(a={a:g}|l1r={l:g})" for a, l in outer_best_params]),
        "n_samples": len(y)
    })

    # Update best-scatter OTU
    if best_scatter["otu"] is None or r2_mean > best_scatter.get("outer_mean_r2", -np.inf):
        best_scatter["otu"] = otu_name
        best_scatter["y_true"] = last_fold_true
        best_scatter["y_pred"] = last_fold_pred
        best_scatter["r2"] = last_fold_r2
        best_scatter["alpha"] = last_fold_alpha
        best_scatter["l1_ratio"] = last_fold_l1_ratio
        best_scatter["outer_mean_r2"] = r2_mean

results_df = pd.DataFrame(otu_results)

# Save full results
results_csv_path = os.path.join(output_dir, "nestedcv_elasticnet_all_otus_results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved full results to: {results_csv_path}")

# -----------------------------
# 5. Rank and show Top 10
# -----------------------------
print("\n--- Top 10 OTUs by OUTER-CV mean R^2 (ElasticNet, nested CV) ---")
ranked = results_df.dropna(subset=["outer_r2_mean"]).sort_values("outer_r2_mean", ascending=False)
print(ranked[[
    "OTU", "outer_r2_mean", "outer_r2_std", "outer_best_alpha_mode", "outer_best_l1_ratio_mode"
]].head(10).to_string(index=False))

top10_df = ranked.head(10).copy()

# -----------------------------
# 6. Plots
# -----------------------------
print("\n--- Generating Plots ---")

# Plot 1: Top 10 bar plot with error bars
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    x="outer_r2_mean",
    y="OTU",
    data=top10_df,
    orient="h"
)
ax.errorbar(
    x=top10_df["outer_r2_mean"].values,
    y=np.arange(len(top10_df)),
    xerr=top10_df["outer_r2_std"].values,
    fmt="none",
    capsize=5
)
plt.title("Top 10 OTUs Predictable from Metadata (ElasticNet Nested CV Outer-Fold R²)")
plt.xlabel("Outer-CV Mean R² (± SD across outer folds)")
plt.ylabel("OTU")
xmin = float(min(0.0, top10_df["outer_r2_mean"].min() - top10_df["outer_r2_std"].max()))
plt.xlim(left=xmin)
plt.tight_layout()
barplot_path = os.path.join(output_dir, "top10_nestedcv_outer_r2_barplot.pdf")
plt.savefig(barplot_path)
plt.close()
print(f"Saved: {barplot_path}")

# Plot 2: Scatter for best OTU (last outer fold)
if best_scatter["otu"] is not None:
    otu_name = best_scatter["otu"]
    y_true = best_scatter["y_true"].astype(float)
    y_pred = np.array(best_scatter["y_pred"], dtype=float)
    r2_last = float(best_scatter["r2"])
    alpha_last = float(best_scatter["alpha"])
    l1r_last = float(best_scatter["l1_ratio"])
    outer_mean = float(best_scatter["outer_mean_r2"])

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.5, label="Outer-fold predictions")

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "--", linewidth=2, label="Identity (y = x)")

    plt.title(f"Best OTU: {otu_name}\nLast Outer Fold (alpha={alpha_last:g}, l1_ratio={l1r_last:g})")
    plt.xlabel("True Y")
    plt.ylabel("Predicted Y")
    plt.text(
        0.05, 0.95,
        f"Last-fold R² = {r2_last:.3f}\nOuter mean R² = {outer_mean:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="wheat", alpha=0.5)
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    scatter_path = os.path.join(output_dir, "best_otu_last_outerfold_scatter.pdf")
    plt.savefig(scatter_path)
    plt.close()
    print(f"Saved: {scatter_path}")
else:
    print("No OTU available for scatter plot.")

print("\nAnalysis complete.")

# Notes:
# - If many models converge poorly, increase enet_max_iter further.
# - If ElasticNet always prefers extreme l1_ratio values, broaden or refine the l1_ratios_to_test grid.
