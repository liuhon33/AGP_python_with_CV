# --- 0. Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression # <-- IMPORTED
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
n_pcs_to_predict = 5

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

# --- 3. PCA on Microbiome Data (Y-base) ---
print("\n--- Performing PCA on full log-transformed OTU data ---")
scaler_otu = StandardScaler()
otu_scaled = scaler_otu.fit_transform(otu_df_log_transformed)

pca = PCA(n_components=n_pcs_to_predict)
Y_pcs = pca.fit_transform(otu_scaled)

print("Explained Variance Ratio by PC:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.4f} ({(var*100):.2f}%)")
print(f"  Total for {n_pcs_to_predict} PCs: {np.sum(pca.explained_variance_ratio_):.4f} ({(np.sum(pca.explained_variance_ratio_)*100):.2f}%)")

# --- 4. Scale Metadata Predictors (X) ---
scaler_meta = StandardScaler()
X_meta_scaled = scaler_meta.fit_transform(X_metadata)

# --- 5. Run Nested Cross-Validation for each PC ---
print(f"\n--- Running {outer_cv_splits}-Fold Nested CV for {n_pcs_to_predict} PCs ---")

nested_cv_results = {} 

for i in range(n_pcs_to_predict):
    pc_name = f'PC{i+1}'
    y_target = Y_pcs[:, i] 
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

# --- 6. Display Final Results ---
print("\n--- Final Nested Cross-Validation Results ---")
print("Average R^2 (Predictability of Microbiome PC from Metadata):")
avg_scores = {pc: results['avg_r2'] for pc, results in nested_cv_results.items()}
results_df = pd.DataFrame.from_dict(avg_scores, orient='index', columns=['Nested CV R^2'])
print(results_df)

results_df.to_csv(os.path.join(output_dir, "rf_pcr_nested_cv_results.csv"))
print(f"\nFull results saved to '{output_dir}rf_pcr_nested_cv_results.csv'")


# --- 7. MODIFIED: Generate Plot for Best Performing PC with Actual Fit Line ---
print("\n--- Generating Plot for Best PC ---")

# Find the best PC
best_pc_name = max(avg_scores, key=avg_scores.get)
best_pc_avg_r2 = avg_scores[best_pc_name]
true_values = nested_cv_results[best_pc_name]['y_true']
pred_values = nested_cv_results[best_pc_name]['y_pred']

# --- Fit a linear regression model to the plot data ---
# Reshape data for sklearn
true_values_reshaped = true_values.reshape(-1, 1)

# Create and fit the linear model
line_model = LinearRegression()
line_model.fit(true_values_reshaped, pred_values)

# Create x-values for the line
min_val = true_values.min()
max_val = true_values.max()
line_x = np.array([min_val, max_val]).reshape(-1, 1)

# Predict y-values for the line
line_y = line_model.predict(line_x)

# Get the R^2 score of this linear fit
line_r2 = line_model.score(true_values_reshaped, pred_values)
# ---

# Create the scatter plot
plt.figure(figsize=(7, 7))
plt.scatter(true_values, pred_values, alpha=0.5, label="Out-of-Fold Predictions")

# Plot the new "Actual Fit" line
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