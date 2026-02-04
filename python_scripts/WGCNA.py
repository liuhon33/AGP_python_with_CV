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
output_dir = ".hongrui_result/RF_on_ME_Results/"
os.makedirs(output_dir, exist_ok=True)
fig_dir = Path(output_dir) / "wgcna_figs"
fig_dir.mkdir(parents=True, exist_ok=True)

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
# --- IMPROVED FILTERING ---
print(f"Initial OTUs: {otu_df.shape[1]}")

# 1. Prevalence Filter: Keep OTUs present in at least 10% of samples
# (Adjust to 0.05 if 0.10 removes too many, but 0.10 is safer for WGCNA)
min_prevalence = 0.10 * otu_df.shape[0]
present_in_samples = (otu_df > 0).sum(axis=0)
otu_prevalence_filtered = otu_df.loc[:, present_in_samples > min_prevalence]
print(f"OTUs after Prevalence filtering (10%): {otu_prevalence_filtered.shape[1]}")

# 2. Abundance Filter: Keep OTUs with mean relative abundance > 0.0001
otu_rel = otu_prevalence_filtered.div(otu_prevalence_filtered.sum(axis=1), axis=0)
mean_rel = otu_rel.mean(axis=0)
otu_final = otu_prevalence_filtered.loc[:, mean_rel > 0.0001]
print(f"Final OTUs for WGCNA: {otu_final.shape[1]}")

# 3. Apply Log Transform to this cleaner dataset
otu_df_log_transformed = np.log1p(otu_final)

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

# Initialize WGCNA object
pyWGCNA_obj = PyWGCNA.WGCNA(
    anndata=adata,
    species='unknown',         # only used for enrichment; placeholder is fine
    networkType='signed',      # or 'signed hybrid' (default); signed is fine here
    TOMType='signed',
    minModuleSize=20,          # smaller than default 50 for OTUs
    #RsquaredCut=0.8,           # your threshold
    save=False,
    figureType='pdf'
)

# Preprocess (sample/gene QC & optional outlier removal)
print("Running PyWGCNA preprocessing...")
pyWGCNA_obj.preprocess()

# Pick soft-threshold power
print("Finding soft-threshold power...")
data_df = pyWGCNA_obj.datExpr.to_df()  # rows=samples, cols=OTUs
# pyWGCNA_obj.pickSoftThreshold(
#     data_df,
#     powerVector=range(1, 21),
#     RsquaredCut=0.8
# )
pyWGCNA_obj.power = 6

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

# ================================
# 4A. WGCNA VISUALS (CORRECTED)
# ================================
print("\n--- 4A. Creating WGCNA visualizations ---")
import matplotlib.colors as mcolors # Required for the color bar fix

# Ensure output directory for figures exists
fig_dir = Path(output_dir) / "wgcna_figs"
fig_dir.mkdir(parents=True, exist_ok=True)

# 4A-1: Soft-Thresholding Plots
# NOTE: PyWGCNA generates these plots inside 'pickSoftThreshold'. 
# Accessing them manually requires inspecting the object attributes or checking the output folder 
# if 'save=True' was set in initialization. We skip the manual call to avoid the AttributeError.
print("... SFT plots are generated during the 'pickSoftThreshold' step above ...")

# 4A-2: The CUSTOM Dendrograms
try:
    from scipy.cluster.hierarchy import dendrogram
    
    print("... Plotting Custom Dendrograms ...")
    
    # Get the linkage matrix from the object
    linkage_matrix = pyWGCNA_obj.geneTree

    if linkage_matrix is not None:
        # --- PLOT A: Truncated Dendrogram (The Readable One) ---
        plt.figure(figsize=(12, 6))
        dendrogram(
            linkage_matrix,
            truncate_mode='lastp',
            p=100,
            leaf_rotation=90.,
            leaf_font_size=8.,
            show_contracted=True,
            above_threshold_color='black',
            color_threshold=0
        )
        plt.title('Summary Dendrogram (Top 100 Branches Only)')
        plt.xlabel('Cluster Size (number of OTUs in branch)')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(fig_dir / "dendrogram_truncated_READABLE.pdf")
        plt.close()

        # --- PLOT B: Full Dendrogram (High Res, Thin Lines) ---
        # FIX: dendrogram() does not accept 'linewidth'. 
        # We use rc_context to change the global line setting temporarily.
        plt.figure(figsize=(20, 8))
        with plt.rc_context({'lines.linewidth': 0.1}): 
            dendrogram(
                linkage_matrix,
                no_labels=True,
                color_threshold=0,
                above_threshold_color='blue'
            )
        plt.title('Full Gene Dendrogram (High Res, Thin Lines)')
        plt.tight_layout()
        plt.savefig(fig_dir / "dendrogram_full_hires.png", dpi=600) 
        plt.close()
    else:
        print("[warn] No geneTree found in pyWGCNA_obj.")

except Exception as e:
    print(f"[error] Custom dendrogram plotting failed: {e}")

# 4A-3: Module-Trait Heatmap
try:
    print("... Plotting Module-Trait Heatmap ...")
    
    # Update Sample Info
    md_clean = X_metadata.copy()
    md_clean = md_clean.loc[pyWGCNA_obj.datExpr.obs.index]
    
    # Update the object
    pyWGCNA_obj.updateSampleInfo(sampleInfo=md_clean)
    
    # FIX: Removed 'figureType' argument which caused the error
    pyWGCNA_obj.module_trait_relationships_heatmap(
        file_name=str(fig_dir / "module_trait_relationships")
    )
    print("Module-Trait heatmap saved.")
except Exception as e:
    print(f"[warn] Module-Trait heatmap skipped: {e}")

# 4A-4 Gene dendrogram with module colors (Color Bar Fix)
try:
    geneTree = pyWGCNA_obj.geneTree 
    if geneTree is not None:
        # Plot the dendrogram itself
        plt.figure(figsize=(12, 5)) # Increased height slightly
        dendro = dendrogram(geneTree, no_labels=True, color_threshold=0)
        plt.title('Gene Dendrogram')
        plt.tight_layout()
        plt.savefig(fig_dir / "gene_dendrogram.png", dpi=200)
        plt.close()

        # --- COLOR BAR FIX ---
        # Need leaves order to map colors
        leaves = dendro['leaves']
        ordered_cols = pyWGCNA_obj.datExpr.var.index[leaves]
        ordered_colors = module_labels.loc[ordered_cols].values
        
        # FIX: Convert string colors (e.g., 'turquoise', '#ff0000') to RGB numbers
        # imshow cannot handle an array of strings directly.
        rgb_colors = [mcolors.to_rgb(c) for c in ordered_colors]
        rgb_array = np.array(rgb_colors).reshape(1, len(rgb_colors), 3)

        fig, ax = plt.subplots(figsize=(12, 0.5))
        ax.imshow(rgb_array, aspect='auto')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title('Module Colors (ordered)')
        plt.tight_layout()
        plt.savefig(fig_dir / "gene_dendrogram_module_colors.png", dpi=200)
        plt.close()
except Exception as e:
    import traceback
    print(f"[warn] Dendrogram color bar plot skipped: {e}")
    traceback.print_exc()

# 4A-5 Module size barplot
try:
    vc = module_labels.value_counts().rename_axis('module').reset_index(name='size')
    plt.figure(figsize=(8,4))
    sns.barplot(data=vc, x='module', y='size')
    plt.title('Module Sizes (including grey)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(fig_dir / "module_sizes.png", dpi=200)
    plt.close()
except Exception as e:
    print(f"[warn] Module size plot skipped: {e}")

# 4A-6 TOM heatmap (subset)
try:
    expr = pyWGCNA_obj.datExpr.to_df()
    non_grey = module_labels[module_labels != 'grey']
    if not non_grey.empty:
        top_mods = non_grey.value_counts().head(4).index.tolist()
        sel_genes = []
        for m in top_mods:
            sel = non_grey[non_grey == m].index[:25]
            sel_genes.extend(sel)
        
        # Fallback if selection is too small
        if len(sel_genes) < 50: 
            sel_genes = list(expr.columns[:100])

        sub_expr = expr[sel_genes]
        
        # Calculate Adjacency & TOM
        adj = PyWGCNA.WGCNA.adjacency(sub_expr, adjacencyType='signed', power=int(pyWGCNA_obj.power))
        tom = PyWGCNA.WGCNA.TOMsimilarity(adj, TOMType='signed')
        
        plt.figure(figsize=(6,5))
        sns.heatmap(tom, cmap='viridis', square=True, cbar_kws={'label':'TOM'})
        plt.title('TOM Heatmap (subset)')
        plt.tight_layout()
        plt.savefig(fig_dir / "TOM_heatmap_subset.png", dpi=200)
        plt.close()
    else:
        print("[info] Skipping TOM heatmap (no non-grey modules found).")
except Exception as e:
    print(f"[warn] TOM heatmap skipped: {e}")

print("Visualizations complete.")

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

# 4A-5 Eigengene correlation heatmap
try:
    corr = Y_mes.corr()
    plt.figure(figsize=(max(6, 0.5*len(corr)), max(5, 0.5*len(corr))))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, square=True,
                cbar_kws={'label':'Pearson r'})
    plt.title('Module Eigengene Correlation')
    plt.tight_layout()
    plt.savefig(fig_dir / "eigengene_correlation_heatmap.png", dpi=200)
    plt.close()
except Exception as e:
    print(f"[warn] Eigengene correlation heatmap skipped: {e}")

# 4A-6 Variance explained per eigengene (from moduleEigengenes output)
try:
    var_exp = ME_out.get("varExplained")
    if isinstance(var_exp, pd.DataFrame):
        pc1 = var_exp.loc[1] if 1 in var_exp.index else var_exp.iloc[0]
        plt.figure(figsize=(max(6, 0.5*len(pc1)), 4))
        (pc1*100).sort_values(ascending=False).plot(kind='bar')
        plt.ylabel('% Variance Explained (PC1)')
        plt.title('Variance Explained by Module Eigengenes (PC1)')
        plt.tight_layout()
        plt.savefig(fig_dir / "eigengene_variance_explained.png", dpi=200)
        plt.close()
except Exception as e:
    print(f"[warn] Variance explained plot skipped: {e}")

# 4A-7 Module–Trait correlation heatmap (add metadata into .obs)
try:
    # Attach numeric metadata to the PyWGCNA object
    # Keep only columns without NA to avoid API errors
    md_clean = X_metadata.copy()
    md_clean = md_clean.loc[wgcna_input_df.index]
    md_clean = md_clean.loc[:, md_clean.isnull().mean() < 0.2].copy()  # drop columns with >=20% NA
    pyWGCNA_obj.updateSampleInfo(sampleInfo=md_clean)

    # Plot via API (saves a pdf by default name); we also export a png using our own draw if desired
    cols_for_heatmap = list(md_clean.columns)[: min(20, md_clean.shape[1])]  # limit for readability
    if len(cols_for_heatmap) > 0:
        pyWGCNA_obj.module_trait_relationships_heatmap(
            metaData=cols_for_heatmap,
            alternative='two-sided',
            file_name=str(fig_dir / "module_trait_relationships")
        )
    else:
        print("[info] Skipped module–trait heatmap (no clean numeric traits).")
except Exception as e:
    print(f"[warn] Module–trait heatmap skipped: {e}")

# 4A-8 kME (module membership) heatmap & hub genes
try:
    kme = pyWGCNA_obj.CalculateSignedKME()
    # Keep top 200 genes by max |kME|
    kme['max_abs_kME'] = kme.abs().max(axis=1)
    top_genes = kme.nlargest(200, 'max_abs_kME').index
    kme_top = kme.loc[top_genes].drop(columns=['max_abs_kME'])
    plt.figure(figsize=(max(8, 0.2 * kme_top.shape[1]), 10))
    sns.heatmap(kme_top, cmap='vlag', center=0)
    plt.title('Signed kME (Module Membership) — Top 200 Genes')
    plt.tight_layout()
    plt.savefig(fig_dir / "kME_heatmap_top200.png", dpi=200)
    plt.close()

    # Save hub genes (top 10 per top 3 largest modules)
    non_grey = module_labels[module_labels != 'grey']
    largest_mods = non_grey.value_counts().head(3).index.tolist()
    hub_dir = fig_dir / "hub_gene_tables"
    hub_dir.mkdir(exist_ok=True)
    for m in largest_mods:
        hub_df = pyWGCNA_obj.top_n_hub_genes(moduleName=m, n=10)
        hub_df.to_csv(hub_dir / f"top10_hubs_{m}.csv", index=False)
except Exception as e:
    print(f"[warn] kME heatmap / hub genes skipped: {e}")

# 4A-9 Intramodular connectivity distributions
try:
    # Build full adjacency once (may be heavy if many features; adjust if needed)
    expr = pyWGCNA_obj.datExpr.to_df()
    adj_full = PyWGCNA.WGCNA.adjacency(expr, adjacencyType='signed', power=int(pyWGCNA_obj.power))
    intrak = PyWGCNA.WGCNA.intramodularConnectivity(
        mat=adj_full.values,
        colors=module_labels.values,
        index=expr.columns.values
    )
    # Plot distributions for up to 4 largest non-grey modules
    non_grey = module_labels[module_labels != 'grey']
    largest_mods = non_grey.value_counts().head(4).index.tolist()
    plt.figure(figsize=(10, 6))
    for m in largest_mods:
        vals = intrak.set_index('index').loc[non_grey[non_grey == m].index, 'intra']
        sns.kdeplot(vals, label=m, fill=False)
    plt.legend(title='Module')
    plt.xlabel('Intramodular Connectivity')
    plt.title('Intramodular Connectivity Distributions (selected modules)')
    plt.tight_layout()
    plt.savefig(fig_dir / "intramodular_connectivity_distributions.png", dpi=200)
    plt.close()
except Exception as e:
    print(f"[warn] Intramodular connectivity plot skipped: {e}")

# 4A-10 Coexpression (network) HTML for largest modules
try:
    non_grey = module_labels[module_labels != 'grey']
    largest_mods = non_grey.value_counts().head(3).index.tolist()
    if len(largest_mods):
        pyWGCNA_obj.CoexpressionModulePlot(
            modules=largest_mods, numGenes=30, numConnections=200,
            minTOM=0, file_name=str(fig_dir / "coexpression_modules")
        )
except Exception as e:
    print(f"[warn] Coexpression module plot (HTML) skipped: {e}")

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
