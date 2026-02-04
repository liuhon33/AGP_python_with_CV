import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA # <-- IMPORT PCA
from sklearn.metrics import r2_score
from tqdm.notebook import tqdm # For terminal, use: from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os # <-- IMPORT OS for file paths

# --- 1. Configuration ---
# File paths
otu_file_path = "./Data/Cleaned_data/AGP_Otu_Data.csv"
metadata_file_path = "./Data/Cleaned_data/processed_metadata.csv"
metadata_index_col = 'sample_name'
otu_index_col = 0

# CV, Filtering, and Model Params
n_splits_cv = 5
random_state_cv = 42
abundance_threshold = 0.0001
alphas_to_test = np.logspace(-3, 3, 7)
n_components_pca_regression = 0.95 # For the regression
n_components_pca_explore = 10      # For the exploratory plots

# --- NEW: Set output directory ---
output_dir = "./hongrui_result/PCR_CV_Results/"
os.makedirs(output_dir, exist_ok=True)
print(f"Alphas to test: {alphas_to_test}")
print(f"PCA for regression set to capture: {n_components_pca_regression} variance")

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
# Filter OTU Table
otu_rel_abund = otu_df.apply(lambda x: x / x.sum(), axis=1)
mean_rel_abund = otu_rel_abund.mean(axis=0)
otus_to_keep = mean_rel_abund[mean_rel_abund > abundance_threshold].index
otu_df_filtered = otu_df[otus_to_keep]
print(f"Number of OTUs after filtering: {otu_df_filtered.shape[1]}")
# Log Transform OTU Data (This is our Y)
otu_df_log_transformed = np.log1p(otu_df_filtered)
print("Applied log(x+1) transformation to OTU counts.")
# Prepare metadata predictors (This is our X)
numeric_cols = metadata_df.select_dtypes(include=np.number).columns
X_metadata = metadata_df[numeric_cols]
if X_metadata.isnull().values.any():
    print("Warning: Missing values found in metadata. Filling with column medians.")
    X_metadata = X_metadata.fillna(X_metadata.median())
print(f"Using {X_metadata.shape[1]} numeric metadata features as predictors.")


# --- 3. NEW: Exploratory PCA on Full Metadata (X) ---
print(f"\n--- 3. Running Exploratory PCA on {X_metadata.shape[1]} Metadata Features ---")

# Scale the *full* metadata matrix for this exploratory analysis
scaler_full = StandardScaler()
X_meta_scaled_full = scaler_full.fit_transform(X_metadata)

# Run PCA to get the top components
pca_explore = PCA(n_components=n_components_pca_explore)
# `pca_scores` are the coordinates of each sample in the new PC space
pca_scores = pca_explore.fit_transform(X_meta_scaled_full)
# `pca_loadings` show how much each original variable contributes to each PC
pca_loadings = pca_explore.components_
# `explained_variance` shows the % of variance each PC captures
explained_variance = pca_explore.explained_variance_ratio_

print("Exploratory PCA complete.")
print(f"Variance explained by first {n_components_pca_explore} components:")
for i, var in enumerate(explained_variance):
    print(f"  PC{i+1}: {var*100:.2f}%")


# --- 4. NEW: Generate Exploratory PCA Plots ---
print("\n--- 4. Generating Exploratory PCA Plots ---")

# Plot 1: Scree Plot (Explained Variance)
plt.figure(figsize=(8, 5))
sns.barplot(x=[f'PC{i+1}' for i in range(n_components_pca_explore)], 
            y=explained_variance * 100, 
            color="steelblue")
plt.title('Scree Plot - Variance Explained by Metadata PCs')
plt.ylabel('Percent Variance Explained')
plt.xlabel('Principal Component')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "metadata_pca_scree_plot.pdf"))
print(f"Scree plot saved to '{output_dir}metadata_pca_scree_plot.pdf'")
plt.close()

# Plot 2: Scores Plot (Samples)
pca_scores_df = pd.DataFrame(pca_scores, columns=[f'PC{i+1}' for i in range(n_components_pca_explore)])
plt.figure(figsize=(8, 7))
sns.scatterplot(data=pca_scores_df, x='PC1', y='PC2', alpha=0.5)
plt.title('PCA Scores Plot (Samples in PC Space)')
plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "metadata_pca_scores_plot.pdf"))
print(f"Scores plot saved to '{output_dir}metadata_pca_scores_plot.pdf'")
plt.close()

# Plot 3: Loadings Plot (Heatmap of Features)
loadings_df = pd.DataFrame(pca_loadings, 
                           columns=numeric_cols, 
                           index=[f'PC{i+1}' for i in range(n_components_pca_explore)])
plt.figure(figsize=(10, 12)) # Taller figure to fit all variables
sns.heatmap(loadings_df.T, cmap='vlag', center=0, annot=False, fmt=".2f", linewidths=.5)
plt.title('PCA Loadings (Contribution of Metadata Features to PCs)')
plt.xlabel('Principal Component')
plt.ylabel('Original Metadata Feature')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "metadata_pca_loadings_heatmap.pdf"))
print(f"Loadings heatmap saved to '{output_dir}metadata_pca_loadings_heatmap.pdf'")
plt.close()
'''
# --- 5. Run Cross-Validated PCR (PCA + Ridge) with Alpha Tuning ---
# (This is your original Step 3, now renamed)
print(f"\n--- 5. Running {n_splits_cv}-Fold CV PCR with alpha tuning ---")
print(f"(You can comment out this section if you only want the exploratory plots)")

otu_best_r2_scores = {}
otu_best_alphas = {}
last_fold_predictions = {'otu_name': None, 'y_val': None, 'y_pred': None, 'r2': None, 'best_alpha': None}

for otu_name in tqdm(otu_df_log_transformed.columns, desc="Processing OTUs"):
    y_otu_log = otu_df_log_transformed[otu_name]
    if y_otu_log.var() < 1e-9:
        otu_best_r2_scores[otu_name] = np.nan
        continue
    alpha_results = {}
    for alpha in alphas_to_test:
        fold_scores = []
        kf = KFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state_cv)
        for fold, (train_index, val_index) in enumerate(kf.split(X_metadata)):
            X_train, X_val = X_metadata.iloc[train_index], X_metadata.iloc[val_index]
            y_train, y_val = y_otu_log.iloc[train_index], y_otu_log.iloc[val_index]
            
            # Apply StandardScaler and PCA inside the loop for the regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            pca = PCA(n_components=n_components_pca_regression) # Use regression-specific n_components
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_val_pca = pca.transform(X_val_scaled)
            
            model = Ridge(alpha=alpha)
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_val_pca)
            score = r2_score(y_val, y_pred)
            fold_scores.append(score)
        alpha_results[alpha] = np.mean(fold_scores)
    best_alpha = max(alpha_results, key=alpha_results.get)
    best_r2 = alpha_results[best_alpha]
    otu_best_r2_scores[otu_name] = best_r2
    otu_best_alphas[otu_name] = best_alpha
    
    # Update logic for the scatter plot
    current_best_otu = last_fold_predictions['otu_name']
    if current_best_otu is None or best_r2 > otu_best_r2_scores.get(current_best_otu, -np.inf):
        kf_plot = KFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state_cv)
        last_fold_train_idx, last_fold_val_idx = list(kf_plot.split(X_metadata))[-1]
        X_train_lf, X_val_lf = X_metadata.iloc[last_fold_train_idx], X_metadata.iloc[last_fold_val_idx]
        y_train_lf, y_val_lf = y_otu_log.iloc[last_fold_train_idx], y_otu_log.iloc[last_fold_val_idx]
        scaler_lf = StandardScaler()
        X_train_scaled_lf = scaler_lf.fit_transform(X_train_lf)
        X_val_scaled_lf = scaler_lf.transform(X_val_lf)
        pca_lf = PCA(n_components=n_components_pca_regression)
        X_train_pca_lf = pca_lf.fit_transform(X_train_scaled_lf)
        X_val_pca_lf = pca_lf.transform(X_val_scaled_lf)
        model_lf = Ridge(alpha=best_alpha)
        model_lf.fit(X_train_pca_lf, y_train_lf)
        y_pred_lf = model_lf.predict(X_val_pca_lf)
        r2_lf = r2_score(y_val_lf, y_pred_lf)
        last_fold_predictions = {
            'otu_name': otu_name, 'y_val': y_val_lf, 'y_pred': y_pred_lf,
            'r2': r2_lf, 'best_alpha': best_alpha
        }

# --- 6. Rank OTUs and Display Top 10 ---
print("\n--- 6. Top 10 Most Predictable OTUs by Metadata (PCR, Avg R^2 after alpha tuning) ---")
results_series = pd.Series(otu_best_r2_scores).dropna()
top_otus = results_series.sort_values(ascending=False)
print(top_otus.head(10))
print("Corresponding Best Alphas:")
print(pd.Series(otu_best_alphas).loc[top_otus.head(10).index])
print("---------------------------------------------------------------------------------")

# --- 7. Generate Plots ---
print("\n--- 7. Generating Final Regression Plots ---")
# Plot 1: Bar plot of Top 10 OTU R^2
top_10_df = top_otus.head(10).reset_index()
top_10_df.columns = ['OTU', 'Best Average R^2']
plt.figure(figsize=(10, 6))
sns.barplot(x='Best Average R^2', y='OTU', data=top_10_df, palette='viridis', orient='h')
plt.title('Top 10 Most Predictable OTUs (PCR, Best Avg R^2 from 5-Fold CV)')
plt.xlabel('Best Average R-squared')
plt.ylabel('OTU Identifier')
plt.xlim(left=max(0, top_10_df['Best Average R^2'].min() - 0.05), right=top_10_df['Best Average R^2'].max() + 0.05)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_10_otu_pcr_best_r2_barplot.pdf"))
print(f"Top 10 OTU best R^2 bar plot saved to '{output_dir}top_10_otu_pcr_best_r2_barplot.pdf'")
plt.close()

# Plot 2: Scatter plot for the single best OTU
if last_fold_predictions['otu_name']:
    top_otu_name = last_fold_predictions['otu_name']
    y_val_top = last_fold_predictions['y_val']
    y_pred_top = last_fold_predictions['y_pred']
    r2_last_fold = last_fold_predictions['r2']
    best_alpha_top = last_fold_predictions['best_alpha']
    avg_r2_top = top_otus.loc[top_otu_name]
    plt.figure(figsize=(6, 6))
    plt.scatter(y_val_top, y_pred_top, alpha=0.5)
    min_val = min(y_val_top.min(), y_pred_top.min())
    max_val = max(y_val_top.max(), y_pred_top.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', lw=1.5, label='Ideal Fit (y=x)')
    plt.title(f'Top OTU ({top_otu_name})\nPredicted vs. Actual log(Count+1) (Last CV Fold, Best Alpha={best_alpha_top:.3f})')
    plt.xlabel('True log(Count+1)')
    plt.ylabel('Predicted log(Count+1)')
    plt.text(0.05, 0.95, f'$R^2$ (this fold) = {r2_last_fold:.3f}\n$R^2$ (avg CV) = {avg_r2_top:.3f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_otu_pcr_prediction_scatter.pdf"))
    print(f"Prediction scatter plot for top OTU ({top_otu_name}) saved to '{output_dir}top_otu_pcr_prediction_scatter.pdf'")
    plt.close()
else:
    print("Could not generate scatter plot: No OTUs found after filtering or all had zero variance.")

print("\nAnalysis complete.")
'''