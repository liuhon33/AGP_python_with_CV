# --- 0. Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm.notebook import tqdm # Use 'tqdm' for regular terminal
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Configuration ---
# File paths
otu_file_path = "./Data/Cleaned_data/AGP_Otu_Data.csv"
metadata_file_path = "./Data/Cleaned_data/processed_metadata.csv"
metadata_index_col = 'sample_name'
otu_index_col = 0

# CV and Filtering
outer_cv_splits = 5
inner_cv_splits = 3
random_state_cv = 42
abundance_threshold = 0.0001
n_pcs_to_predict = 5    # How many PCs to use as targets for the RF models
n_pcs_to_explore = 10   # How many PCs to calculate for the exploratory plots

# RF Hyperparameter Grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [5, 10, 20],
    'max_depth': [None, 10, 20]
}

# Set output directory
output_dir = "./RF_on_PCA_Results/"
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

# Filter OTU Table
otu_rel_abund = otu_df.apply(lambda x: x / x.sum(), axis=1)
mean_rel_abund = otu_rel_abund.mean(axis=0)
otus_to_keep = mean_rel_abund[mean_rel_abund > abundance_threshold].index
otu_df_filtered = otu_df[otus_to_keep]
print(f"Number of OTUs after filtering: {otu_df_filtered.shape[1]}")

# Log Transform OTU Data (This is our Y-base)
otu_df_log_transformed = np.log1p(otu_df_filtered)
print("Applied log(x+1) transformation to OTU counts.")

# Prepare metadata predictors (This is our X)
numeric_cols = metadata_df.select_dtypes(include=np.number).columns
X_metadata = metadata_df[numeric_cols]
if X_metadata.isnull().values.any():
    print("Warning: Missing values found in metadata. Filling with column medians.")
    X_metadata = X_metadata.fillna(X_metadata.median())
print(f"Using {X_metadata.shape[1]} numeric metadata features as predictors.")

# --- 3. Exploratory PCA on Microbiome Data (Y-base) ---
print(f"\n--- 3. Performing PCA on {otu_df_filtered.shape[1]} log-transformed OTUs ---")
scaler_otu = StandardScaler()
otu_scaled = scaler_otu.fit_transform(otu_df_log_transformed)

# Run PCA to get ALL components
pca_otu_full = PCA(n_components=None)
pca_otu_full.fit(otu_scaled) # Fit on the scaled OTU data

# Get all explained variances
all_otu_explained_variance = pca_otu_full.explained_variance_ratio_

# --- NEW: Calculate components for 50% variance ---
cumulative_variance = np.cumsum(all_otu_explained_variance)
# Find the index of the first component that reaches or exceeds 0.50
components_for_50_percent = np.argmax(cumulative_variance >= 0.50) + 1 # +1 because index is 0-based

print("\n--- OTU PCA Variance Analysis ---")
print(f"** Components needed to explain 50% of variance: {components_for_50_percent} **")
print("---------------------------------")
# ---

# Now, get the specific slices needed for the rest of the script
# Get data for the top N components for exploration (plots)
otu_explained_variance_explore = all_otu_explained_variance[:n_pcs_to_explore]
# We need to transform the data to get the scores
pca_otu_scores_full = pca_otu_full.transform(otu_scaled)
pca_otu_scores_explore = pca_otu_scores_full[:, :n_pcs_to_explore]
pca_otu_loadings_explore = pca_otu_full.components_[:n_pcs_to_explore, :]

print("Exploratory OTU PCA complete.")
print(f"Variance explained by first {n_pcs_to_explore} OTU components:")
for i, var in enumerate(otu_explained_variance_explore):
    print(f"  PC{i+1}: {var*100:.2f}%")

# --- 4. Generate Exploratory OTU PCA Plots ---
print("\n--- 4. Generating Exploratory OTU PCA Plots ---")

# Plot 1: Scree Plot (Explained Variance)
plt.figure(figsize=(8, 5))
sns.barplot(x=[f'PC{i+1}' for i in range(n_pcs_to_explore)], 
            y=otu_explained_variance_explore * 100, 
            color="steelblue")
plt.title('Scree Plot - Variance Explained by OTU PCs')
plt.ylabel('Percent Variance Explained')
plt.xlabel('Principal Component')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "otu_pca_scree_plot.pdf"))
print(f"OTU Scree plot saved to '{output_dir}otu_pca_scree_plot.pdf'")
plt.close()

# Plot 2: Scores Plot (Samples)
pca_otu_scores_df = pd.DataFrame(pca_otu_scores_explore, columns=[f'PC{i+1}' for i in range(n_pcs_to_explore)])
plt.figure(figsize=(8, 7))
sns.scatterplot(data=pca_otu_scores_df, x='PC1', y='PC2', alpha=0.3)
plt.title('OTU PCA Scores Plot (Samples in PC Space)')
plt.xlabel(f'PC1 ({otu_explained_variance_explore[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({otu_explained_variance_explore[1]*100:.2f}%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "otu_pca_scores_plot.pdf"))
print(f"OTU Scores plot saved to '{output_dir}otu_pca_scores_plot.pdf'")
plt.close()

# Plot 3: Loadings Bar Plot (Top Features for PC1)
loadings_pc1 = pd.Series(pca_otu_loadings_explore[0, :], index=otu_df_filtered.columns)
top_loadings = pd.concat([loadings_pc1.nlargest(10), loadings_pc1.nsmallest(10)]).sort_values()

plt.figure(figsize=(10, 8))
sns.barplot(x=top_loadings.values, y=top_loadings.index, orient='h', palette='vlag')
plt.title('OTU Loadings for PC1 (Top 10 Positive & Negative Drivers)')
plt.xlabel('Loading Value')
plt.ylabel('OTU Identifier')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "otu_pca_pc1_loadings_plot.pdf"))
print(f"OTU PC1 Loadings plot saved to '{output_dir}otu_pca_pc1_loadings_plot.pdf'")
plt.close()

# --- 5. Prepare Y Targets for Regression ---
# Slice the scores from the full PCA we already ran
Y_pcs = pca_otu_scores_full[:, :n_pcs_to_predict]
print(f"\nUsing first {n_pcs_to_predict} PCs as targets for regression.")

# --- 6. Scale Metadata Predictors (X) ---
print("\n--- 6. Scaling full metadata matrix (Our X Predictors) ---")
scaler_meta = StandardScaler()
X_meta_scaled = scaler_meta.fit_transform(X_metadata)

# --- 7. Run Nested Cross-Validation for each PC ---
print(f"\n--- 7. Running {outer_cv_splits}-Fold Nested CV for {n_pcs_to_predict} Microbiome PCs ---")
print("(This is the time-consuming step. You can comment out from here.)")

nested_cv_results = {} 

for i in range(n_pcs_to_predict):
    pc_name = f'PC{i+1}'
    y_target = Y_pcs[:, i] # Target is the i-th OTU PC
    print(f"\n... Processing {pc_name} ...")
    start_time = time.time()
    
    outer_cv = KFold(n_splits=outer_cv_splits, shuffle=True, random_state=random_state_cv)
    
    outer_loop_scores = []
    y_true_all_folds = []
    y_pred_all_folds = []

    for fold, (train_idx, test_idx) in enumerate(tqdm(outer_cv.split(X_meta_scaled), total=outer_cv_splits, desc=f"Outer CV for {pc_name}")):
        X_outer_train, X_outer_test = X_meta_scaled[train_idx], X_meta_scaled[test_idx]
        y_outer_train, y_outer_test = y_target[train_idx], y_target[test_idx]

        # --- Inner Loop (Hyperparameter Tuning) ---
        rf = RandomForestRegressor(random_state=random_state_cv)
        inner_cv = KFold(n_splits=inner_cv_splits, shuffle=True, random_state=random_state_cv)
        
        grid_search = GridSearchCV(estimator=rf, 
                                   param_grid=param_grid, 
                                   cv=inner_cv, 
                                   scoring='r2', 
                                   n_jobs=-1,
                                   verbose=0) 
        
        grid_search.fit(X_outer_train, y_outer_train)
        
        # --- Evaluate on Outer Test Set ---
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_outer_test)
        
        score = r2_score(y_outer_test, y_pred)
        outer_loop_scores.append(score)
        
        y_true_all_folds.append(y_outer_test)
        y_pred_all_folds.append(y_pred)

    # Store all results for this PC
    nested_cv_results[pc_name] = {
        'avg_r2': np.mean(outer_loop_scores),
        'y_true': np.concatenate(y_true_all_folds), 
        'y_pred': np.concatenate(y_pred_all_folds)
    }
    
    end_time = time.time()
    print(f"  {pc_name} Average R^2: {nested_cv_results[pc_name]['avg_r2']:.4f} (took {end_time - start_time:.1f} seconds)")

# --- 8. Display Final Results ---
print("\n--- 8. Final Nested Cross-Validation Results ---")
print("Average R^2 (Predictability of Microbiome PC from Metadata):")
avg_scores = {pc: results['avg_r2'] for pc, results in nested_cv_results.items()}
results_df = pd.DataFrame.from_dict(avg_scores, orient='index', columns=['Nested CV R^2'])
print(results_df)

results_df.to_csv(os.path.join(output_dir, "rf_pcr_nested_cv_results.csv"))
print(f"\nFull results saved to '{output_dir}rf_pcr_nested_cv_results.csv'")


# --- 9. Generate Plot for Best Performing PC with Actual Fit Line ---
print("\n--- 9. Generating Plot for Best PC ---")

# Find the best PC
best_pc_name = max(avg_scores, key=avg_scores.get)
best_pc_avg_r2 = avg_scores[best_pc_name]
true_values = nested_cv_results[best_pc_name]['y_true']
pred_values = nested_cv_results[best_pc_name]['y_pred']

# --- Fit a linear regression model to the plot data ---
true_values_reshaped = true_values.reshape(-1, 1)
line_model = LinearRegression()
line_model.fit(true_values_reshaped, pred_values)
min_val = true_values.min()
max_val = true_values.max()
line_x = np.array([min_val, max_val]).reshape(-1, 1)
line_y = line_model.predict(line_x)
line_r2 = line_model.score(true_values_reshaped, pred_values)
# ---

# Create the scatter plot
plt.figure(figsize=(7, 7))
plt.scatter(true_values, pred_values, alpha=0.5, label="Out-of-Fold Predictions")
plt.plot(line_x.flatten(), line_y, '--', color='red', lw=2, 
         label=f"Actual Fit ($R^2$ = {line_r2:.3f})")

plt.title(f'Best Performing Model: Predict {best_pc_name} from Metadata\n(Nested CV $R^2$ = {best_pc_avg_r2:.3f})')
plt.xlabel("True PC Value (Test Set)")
plt.ylabel("Predicted PC Value (from RF Model)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"best_pc_prediction_scatter_actual_fit_{best_pc_name}.pdf"))
print(f"Prediction scatter plot for best PC ({best_pc_name}) saved.")
plt.show()

print("\nAnalysis complete.")