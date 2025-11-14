#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- 0. Import Libraries ---
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from tqdm.notebook import tqdm  # use plain `tqdm` if running in terminal

import matplotlib.pyplot as plt
import seaborn as sns

# --- Libraries for WGCNA ---
import anndata as ad
import PyWGCNA

# --- 1. Configuration ---
otu_file_path = "./Data/Cleaned_data/AGP_Otu_Data.csv"
metadata_file_path = "./Data/Cleaned_data/processed_metadata.csv"
metadata_index_col = 'sample_name'
otu_index_col = 0

# CV and filtering
outer_cv_splits = 5
inner_cv_splits = 3
random_state_cv = 42
abundance_threshold = 0.0001  # mean relative abundance cutoff

# RF hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [5, 10, 20],
    'max_depth': [None, 10, 20]
}

# Output directory
output_dir = "./RF_on_ME_Results/"
os.makedirs(output_dir, exist_ok=True)


# --- 2. Load and Prepare Data ---
print("--- Loading data ---")
metadata_df = pd.read_csv(metadata_file_path, index_col=metadata_index_col)
otu_df = pd.read_csv(otu_file_path, index_col=otu_index_col)

# Align samples
common_samples = metadata_df.index.intersection(otu_df.index)
metadata_df = metadata_df.loc[common_samples]
otu_df = otu_df.loc[common_samples]
print(f"Data aligned. Found {len(common_samples)} common samples.")
print(f"Initial number of OTUs: {otu_df.shape[1]}")

# Filter OTU table by mean relative abundance
otu_rel_abund = otu_df.apply(lambda x: x / x.sum(), axis=1)
mean_rel_abund = otu_rel_abund.mean(axis=0)
otus_to_keep = mean_rel_abund[mean_rel_abund > abundance_threshold].index
otu_df_filtered = otu_df[otus_to_keep]
print(f"Number of OTUs after filtering: {otu_df_filtered.shape[1]}")

# Log1p transform OTU counts (your current choice)
otu_df_log_transformed = np.log1p(otu_df_filtered)
print("Applied log(x+1) transformation to OTU counts.")

# Prepare numeric metadata predictors (X)
numeric_cols = metadata_df.select_dtypes(include=np.number).columns
X_metadata = metadata_df[numeric_cols]
if X_metadata.isnull().values.any():
    print("Warning: Missing values found in metadata. Filling with column medians.")
    X_metadata = X_metadata.fillna(X_metadata.median())
print(f"Using {X_metadata.shape[1]} numeric metadata features as predictors.")


# --- 3. Run WGCNA to Get Module Labels (PyWGCNA) ---
print("\n--- 3. Running PyWGCNA Module Detection ---")
print("Converting data to AnnData format...")
adata = ad.AnnData(
    X=otu_df_log_transformed.astype(float).values,   # rows: samples, cols: OTUs
    obs=pd.DataFrame(index=otu_df_log_transformed.index),
    var=pd.DataFrame(index=otu_df_log_transformed.columns)
)

# Initialize WGCNA object (microbiome-friendly defaults)
pyWGCNA_obj = PyWGCNA.WGCNA(
    anndata=adata,
    species='unknown',         # only used for enrichment; placeholder is fine
    networkType='signed',      # or 'signed hybrid' (default); signed is fine here
    TOMType='signed',
    minModuleSize=20,          # smaller than default 50 for OTUs
    RsquaredCut=0.8,           # your threshold
    save=False,
    figureType='pdf'
)

# Preprocess (sample/gene QC & optional outlier removal)
print("Running PyWGCNA preprocessing...")
pyWGCNA_obj.preprocess()

# Pick soft-threshold power
print("Finding soft-threshold power...")
data_df = pyWGCNA_obj.datExpr.to_df()  # rows=samples, cols=OTUs
pyWGCNA_obj.pickSoftThreshold(
    data_df,
    powerVector=range(1, 21),
    RsquaredCut=0.8
)

# Build network & detect modules
print("Building network and detecting modules via runWGCNA()...")
pyWGCNA_obj.runWGCNA()

# Module labels live in datExpr.var['moduleColors'] (color per OTU)
module_labels = pyWGCNA_obj.datExpr.var['moduleColors']

# Count modules (API may return list/ndarray)
modules = pyWGCNA_obj.getModuleName()
n_modules_incl_grey = len(modules) if not hasattr(modules, "size") else modules.size
print(f"Module detection complete. Found {n_modules_incl_grey} modules (including 'grey').")
print(module_labels.value_counts())


# --- 4. Calculate Module Eigengenes (Targets, Y_mes) ---
print("\n--- 4. Calculating Module Eigengenes (MEs) ---")
wgcna_input_df = pyWGCNA_obj.datExpr.to_df()  # rows=samples, cols=OTUs

# Align color vector to expression columns; **pass a NumPy array** (not list)
colors_aligned = module_labels.reindex(wgcna_input_df.columns).to_numpy()

ME_out = PyWGCNA.WGCNA.moduleEigengenes(
    expr=wgcna_input_df,
    colors=colors_aligned,      # MUST be np.ndarray or pd.Series for vectorized comparisons
    excludeGrey=True,           # drop 'grey' if present
    nPC=1,
    scaleVar=True
)

Y_mes = ME_out["eigengenes"]  # DataFrame with columns like MEturquoise, MEblue, ...
if Y_mes.shape[1] == 0:
    raise RuntimeError("No non-grey modules found; try lowering minModuleSize or adjusting parameters.")
print(f"Created Y-target matrix with {Y_mes.shape[1]} module eigengenes (non-grey).")


# --- 5. Scale Metadata Predictors (Our X) ---
print("\n--- 5. Scaling full metadata matrix (Our X Predictors) ---")
scaler_meta = StandardScaler()
X_meta_scaled = scaler_meta.fit_transform(X_metadata)


# --- 6. Run Nested Cross-Validation: RF predicting each ME from X ---
print(f"\n--- 6. Running {outer_cv_splits}-Fold Nested CV for {Y_mes.shape[1]} Module Eigengenes ---")
nested_cv_results = {}

for me_name in Y_mes.columns:
    y_target = Y_mes[me_name].values
    print(f"\n... Processing {me_name} ...")
    start_time = time.time()

    outer_cv = KFold(n_splits=outer_cv_splits, shuffle=True, random_state=random_state_cv)
    outer_loop_scores = []
    y_true_all_folds = []
    y_pred_all_folds = []

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(outer_cv.split(X_meta_scaled), total=outer_cv_splits, desc=f"Outer CV for {me_name}")
    ):
        X_outer_train, X_outer_test = X_meta_scaled[train_idx], X_meta_scaled[test_idx]
        y_outer_train, y_outer_test = y_target[train_idx], y_target[test_idx]

        # Inner loop hyperparameter tuning
        rf = RandomForestRegressor(random_state=random_state_cv)
        inner_cv = KFold(n_splits=inner_cv_splits, shuffle=True, random_state=random_state_cv)

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_outer_train, y_outer_train)

        # Evaluate on outer test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_outer_test)

        score = r2_score(y_outer_test, y_pred)
        outer_loop_scores.append(score)

        y_true_all_folds.append(y_outer_test)
        y_pred_all_folds.append(y_pred)

    nested_cv_results[me_name] = {
        'avg_r2': float(np.mean(outer_loop_scores)),
        'y_true': np.concatenate(y_true_all_folds),
        'y_pred': np.concatenate(y_pred_all_folds)
    }

    elapsed = time.time() - start_time
    print(f"  {me_name} Average R^2: {nested_cv_results[me_name]['avg_r2']:.4f} (took {elapsed:.1f} seconds)")



# --- 7. Display & Save Final Results ---
print("\n--- 7. Final Nested Cross-Validation Results ---")
print("Average R^2 (Predictability of Microbiome Module Eigengene from Metadata):")

avg_scores = {me: results['avg_r2'] for me, results in nested_cv_results.items()}
results_df = pd.DataFrame.from_dict(avg_scores, orient='index', columns=['Nested CV R^2']).sort_values(
    by='Nested CV R^2', ascending=False
)

print(results_df)
results_csv = Path(output_dir) / "rf_me_regression_nested_cv_results.csv"
results_df.to_csv(results_csv)
print(f"\nFull results saved to '{results_csv}'")

# --- 8. Generate Plot for Best Performing ME (scatter) ---
print("\n--- 8. Generating Plot for Best ME ---")

best_me_name = results_df.index[0]
best_me_avg_r2 = results_df.loc[best_me_name, 'Nested CV R^2']
true_values = nested_cv_results[best_me_name]['y_true']
pred_values = nested_cv_results[best_me_name]['y_pred']

# Fit a regression line to OOF predictions
true_values_reshaped = true_values.reshape(-1, 1)
line_model = LinearRegression()
line_model.fit(true_values_reshaped, pred_values)

min_val = true_values.min()
max_val = true_values.max()
line_x = np.array([min_val, max_val]).reshape(-1, 1)
line_y = line_model.predict(line_x)
line_r2 = line_model.score(true_values_reshaped, pred_values)

# Plot
plt.figure(figsize=(7, 7))
plt.scatter(true_values, pred_values, alpha=0.5, label="Out-of-Fold Predictions")
plt.plot(line_x.flatten(), line_y, '--', color='red', lw=2,
         label=f"Actual Fit ($R^2$ = {line_r2:.3f})")

plt.title(f'Best Performing Model: Predict {best_me_name} from Metadata\n(Nested CV $R^2$ = {best_me_avg_r2:.3f})')
plt.xlabel("True ME Value (Test Set)")
plt.ylabel("Predicted ME Value (from RF Model)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

safe_name = "".join(c if c.isalnum() or c in ('_', '-') else "_" for c in best_me_name)
plot_path = Path(output_dir) / f"best_me_prediction_scatter_{safe_name}.pdf"
plt.savefig(plot_path)
print(f"Prediction scatter plot for best ME ({best_me_name}) saved to '{plot_path}'.")
plt.show()

print("\nAnalysis complete.")
