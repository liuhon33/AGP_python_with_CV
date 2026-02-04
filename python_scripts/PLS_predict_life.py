# --- 0. Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.notebook import tqdm # Use 'tqdm' for regular terminal

# --- Helper Function for OTU Filtering ---
def filter_otus_by_abundance(otu_df, threshold=0.0001):
    """Filters OTUs based on mean relative abundance."""
    # Convert counts to relative abundance
    row_sums = otu_df.sum(axis=1)
    otu_rel_abund = otu_df.apply(lambda col: col / row_sums if row_sums[col.name] > 0 else col, axis=0)
    mean_rel_abund = otu_rel_abund.mean(axis=0)
    otus_to_keep = mean_rel_abund[mean_rel_abund > threshold].index
    return otu_df[otus_to_keep]


# --- 1. Configuration ---
# File paths
otu_file_path = "./Data/Cleaned_data/AGP_Otu_Data.csv"
metadata_file_path = "./Data/Cleaned_data/AGP_Metadata.csv"
metadata_index_col = 'sample_name'
otu_index_col = 0

output_dir = "./hongrui_result/PLS_CV_Predict_Lifestyle_py/" # New output directory

# Parameters
otu_abundance_threshold = 0.0001 # 0.01% threshold
n_splits_cv = 5
n_components_pls = 10 # Number of PLS components
random_state_cv = 123

# --- 2. Create Output Dir ---
os.makedirs(output_dir, exist_ok=True)

# --- 3. Load and Prepare Data (Following Ridge Script Logic) ---
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
otu_df_filtered = filter_otus_by_abundance(otu_df, threshold=otu_abundance_threshold)
print(f"Number of OTUs after filtering (> {otu_abundance_threshold*100:.3f}%): {otu_df_filtered.shape[1]}")

# Log Transform OTU Data - This is X_microbiome
X_microbiome_log = np.log1p(otu_df_filtered)
print("Applied log(x+1) transformation to filtered OTU counts (X).")

# Prepare metadata predictors - This is Y_lifestyle
numeric_cols = metadata_df.select_dtypes(include=np.number).columns
Y_lifestyle_full = metadata_df[numeric_cols]
if Y_lifestyle_full.isnull().values.any():
    print("Warning: Missing values found in metadata. Filling with column medians.")
    Y_lifestyle_full = Y_lifestyle_full.fillna(Y_lifestyle_full.median())
print(f"Using {Y_lifestyle_full.shape[1]} numeric metadata features as targets (Y).")

# Ensure same order before converting to NumPy
Y_lifestyle_full = Y_lifestyle_full.sort_index()
X_microbiome_log = X_microbiome_log.sort_index()

Y_lifestyle_full_np = Y_lifestyle_full.values
X_microbiome_full_np = X_microbiome_log.values # log-transformed counts

# --- 4. Perform K-Fold Cross-Validation for Lifestyle Prediction ---
print(f"\n--- Performing {n_splits_cv}-Fold CV for Lifestyle Prediction ---")
kf = KFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state_cv)

# Store prediction metrics for each fold (averaged across all Y variables)
prediction_metrics = []

for fold, (train_index, test_index) in enumerate(tqdm(kf.split(X_microbiome_full_np), total=n_splits_cv, desc="CV Folds")):
    X_train, X_test = X_microbiome_full_np[train_index], X_microbiome_full_np[test_index]
    Y_train, Y_test = Y_lifestyle_full_np[train_index], Y_lifestyle_full_np[test_index]

    # Scale X (microbiome) based on training data
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    # Scale Y (lifestyle) based on training data
    scaler_y = StandardScaler()
    Y_train_scaled = scaler_y.fit_transform(Y_train)
    # Y_test is kept unscaled for final comparison

    # Determine number of components dynamically
    n_samples_train = X_train_scaled.shape[0]
    n_features_x = X_train_scaled.shape[1]
    n_features_y = Y_train_scaled.shape[1]
    ncomp_fold = min(n_components_pls, n_samples_train -1, n_features_x, n_features_y)

    if ncomp_fold < 1:
        print(f"  Skipping fold {fold+1} - not enough samples/variables for PLS.")
        prediction_metrics.append({'R2_avg': np.nan, 'MSE_avg': np.nan, 'MAE_avg': np.nan})
        continue

    # Train PLS model: Predict Y_train_scaled from X_train_scaled
    pls_model = PLSRegression(n_components=ncomp_fold, scale=False)
    pls_model.fit(X_train_scaled, Y_train_scaled)

    # Predict scaled Y values for the test set using scaled X_test
    Y_pred_scaled = pls_model.predict(X_test_scaled)

    # Inverse transform the predictions back to the original lifestyle scale
    Y_pred = scaler_y.inverse_transform(Y_pred_scaled)

    # Compare the prediction (Y_pred) with the original Y_test
    # Calculate metrics for each Y variable and then average
    r2_scores_fold = []
    mse_scores_fold = []
    mae_scores_fold = []

    for j in range(Y_test.shape[1]): # Iterate through each lifestyle variable
        y_true_j = Y_test[:, j]
        y_pred_j = Y_pred[:, j]

        # Handle cases with zero variance in the true values for R2 calculation
        if np.var(y_true_j) < np.finfo(float).eps:
            r2_scores_fold.append(np.nan) # Cannot calculate R2 if true values don't vary
        else:
            r2_scores_fold.append(r2_score(y_true_j, y_pred_j))

        mse_scores_fold.append(mean_squared_error(y_true_j, y_pred_j))
        mae_scores_fold.append(mean_absolute_error(y_true_j, y_pred_j))

    # Average metrics across all predicted lifestyle variables for this fold
    fold_avg_r2 = np.nanmean(r2_scores_fold) # Use nanmean to ignore NaNs
    fold_avg_mse = np.nanmean(mse_scores_fold)
    fold_avg_mae = np.nanmean(mae_scores_fold)

    prediction_metrics.append({'R2_avg': fold_avg_r2, 'MSE_avg': fold_avg_mse, 'MAE_avg': fold_avg_mae})

# --- 5. Aggregate and Display Results ---
print("\n--- Cross-Validation Results (Predicting Lifestyle from Microbiome) ---")
metrics_df = pd.DataFrame(prediction_metrics)
metrics_df.index.name = "Fold"

average_metrics = metrics_df.mean(axis=0, skipna=True)

print("Average Metrics Across Folds (averaged over all lifestyle variables):")
print(f"  R-squared: {average_metrics['R2_avg']:.4f}")
print(f"  MSE:       {average_metrics['MSE_avg']:.4f}")
print(f"  MAE:       {average_metrics['MAE_avg']:.4f}")

# Save results
metrics_df.to_csv(os.path.join(output_dir, "pls_predict_lifestyle_metrics_all_folds.csv"))
average_metrics.to_csv(os.path.join(output_dir, "pls_predict_lifestyle_metrics_average.csv"), header=['Avg_Metric'])
print(f"\nFull prediction metrics saved to: {output_dir}")
print("Analysis complete.")

# --- 6. Optional: Visualize Results ---
plt.figure(figsize=(8, 5))
# Rename columns for better plot labels
plot_df = metrics_df.rename(columns={'R2_avg': 'Avg R-squared', 'MSE_avg': 'Avg MSE', 'MAE_avg': 'Avg MAE'})
metrics_plot = sns.boxplot(data=plot_df, palette="viridis")
plt.title("Distribution of Prediction Metrics Across Folds (PLS: Microbiome -> Lifestyle)", fontsize=14)
plt.ylabel("Average Metric Value (across Y variables)", fontsize=12)
plt.xlabel("Metric Type", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pls_predict_lifestyle_metrics_boxplot.pdf"))
print("Prediction metrics boxplot saved.")
plt.show()

'''
# --- 0. Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.notebook import tqdm # Use 'tqdm' for regular terminal

# --- Helper Function for OTU Filtering ---
def filter_otus_by_abundance(otu_df, threshold=0.0001): # Using 0.0001
    """Filters OTUs based on mean relative abundance."""
    # Convert counts to relative abundance
    # Handle zero sums in rows if necessary
    row_sums = otu_df.sum(axis=1)
    otu_rel_abund = otu_df.apply(lambda x: x / row_sums if row_sums[x.name] > 0 else x, axis=0).T

    # Calculate mean relative abundance
    mean_rel_abund = otu_rel_abund.mean(axis=0)
    # Identify OTUs above the threshold
    otus_to_keep = mean_rel_abund[mean_rel_abund > threshold].index
    return otu_df[otus_to_keep]

# --- 1. Configuration ---
# File paths (Using paths from the Ridge script)
otu_file_path = "./Data/Cleaned_data/AGP_Otu_Data.csv"
metadata_file_path = "./Data/Cleaned_data/processed_metadata.csv"
metadata_index_col = 'sample_name' # As specified in Ridge script
otu_index_col = 0 # Assuming first column is index in AGP_Otu_Data.csv

output_dir = "./PLS_CV_Reconstruction_RidgeProcessing/" # New output directory

# Parameters
otu_abundance_threshold = 0.0001 # 0.01% threshold
n_splits_cv = 5
n_components_pls = 10 # Number of PLS components for reconstruction
random_state_cv = 123

# --- 2. Create Output Dir ---
os.makedirs(output_dir, exist_ok=True)

# --- 3. Load and Prepare Data (Following Ridge Script Logic) ---
print("--- Loading data ---")
metadata_df = pd.read_csv(metadata_file_path, index_col=metadata_index_col)
otu_df = pd.read_csv(otu_file_path, index_col=otu_index_col)

# Align samples
common_samples = metadata_df.index.intersection(otu_df.index)
metadata_df = metadata_df.loc[common_samples]
otu_df = otu_df.loc[common_samples]
print(f"Data aligned. Found {len(common_samples)} common samples.")
print(f"Initial number of OTUs: {otu_df.shape[1]}")

# Filter OTU Table using the specified threshold
otu_df_filtered = filter_otus_by_abundance(otu_df, threshold=otu_abundance_threshold)
print(f"Number of OTUs after filtering (> {otu_abundance_threshold*100:.3f}%): {otu_df_filtered.shape[1]}")

# Log Transform OTU Data - This is X_microbiome
X_microbiome_log = np.log1p(otu_df_filtered)
print("Applied log(x+1) transformation to filtered OTU counts.")

# Prepare metadata predictors - This is Y_lifestyle
numeric_cols = metadata_df.select_dtypes(include=np.number).columns
Y_lifestyle_full = metadata_df[numeric_cols]
if Y_lifestyle_full.isnull().values.any():
    print("Warning: Missing values found in metadata. Filling with column medians.")
    Y_lifestyle_full = Y_lifestyle_full.fillna(Y_lifestyle_full.median())
print(f"Using {Y_lifestyle_full.shape[1]} numeric metadata features as predictors (Y).")

# Ensure same order before converting to NumPy
Y_lifestyle_full = Y_lifestyle_full.sort_index()
X_microbiome_log = X_microbiome_log.sort_index()

Y_lifestyle_full_np = Y_lifestyle_full.values
X_microbiome_full_np = X_microbiome_log.values # log-transformed counts

# --- 4. Perform K-Fold Cross-Validation for Reconstruction ---
print(f"\n--- Performing {n_splits_cv}-Fold CV for Microbiome Reconstruction ---")
kf = KFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state_cv)
reconstruction_metrics = []

for fold, (train_index, test_index) in enumerate(tqdm(kf.split(X_microbiome_full_np), total=n_splits_cv, desc="CV Folds")):
    X_train, X_test = X_microbiome_full_np[train_index], X_microbiome_full_np[test_index]
    Y_train, Y_test = Y_lifestyle_full_np[train_index], Y_lifestyle_full_np[test_index]

    # Scale Y (lifestyle) based on training data
    scaler_y = StandardScaler()
    Y_train_scaled = scaler_y.fit_transform(Y_train)
    Y_test_scaled = scaler_y.transform(Y_test)

    # Scale X (microbiome) based on training data
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test) # Scaled version needed for PLS math consistency

    # Determine number of components dynamically
    n_samples_train = X_train_scaled.shape[0]
    n_features_x = X_train_scaled.shape[1]
    n_features_y = Y_train_scaled.shape[1]
    ncomp_fold = min(n_components_pls, n_samples_train -1, n_features_x, n_features_y)

    if ncomp_fold < 1:
        print(f"  Skipping fold {fold+1} - not enough samples/variables for PLS.")
        reconstruction_metrics.append({'R2': np.nan, 'MSE': np.nan, 'MAE': np.nan})
        continue

    # Train PLS model on scaled training data
    pls_model = PLSRegression(n_components=ncomp_fold, scale=False)
    pls_model.fit(X_train_scaled, Y_train_scaled)

    # Predict X latent scores using Y_test_scaled
    if Y_test_scaled.shape[1] == pls_model.y_rotations_.shape[0]:
         predicted_X_scores = Y_test_scaled @ pls_model.y_rotations_
    else:
         print(f"  Warning: Dimension mismatch for predicting X_scores in fold {fold+1}. Skipping.")
         reconstruction_metrics.append({'R2': np.nan, 'MSE': np.nan, 'MAE': np.nan})
         continue

    # Reconstruct the *scaled* X_test using the predicted scores and X loadings
    X_reconstructed_scaled = predicted_X_scores @ pls_model.x_loadings_.T

    # Inverse transform the reconstruction back to the original log-count scale
    X_reconstructed = scaler_x.inverse_transform(X_reconstructed_scaled)

    # Compare the reconstruction (X_reconstructed) with the original X_test (log-transformed)
    r2 = r2_score(X_test, X_reconstructed)
    mse = mean_squared_error(X_test, X_reconstructed)
    mae = mean_absolute_error(X_test, X_reconstructed)

    reconstruction_metrics.append({'R2': r2, 'MSE': mse, 'MAE': mae})

# --- 5. Aggregate and Display Results ---
print("\n--- Cross-Validation Results (Microbiome Reconstruction from Lifestyle) ---")
metrics_df = pd.DataFrame(reconstruction_metrics)
metrics_df.index.name = "Fold"
average_metrics = metrics_df.mean(axis=0, skipna=True)

print("Average Metrics Across Folds:")
print(f"  R-squared: {average_metrics['R2']:.4f}")
print(f"  MSE:       {average_metrics['MSE']:.4f}")
print(f"  MAE:       {average_metrics['MAE']:.4f}")

# Save results
metrics_df.to_csv(os.path.join(output_dir, "pls_reconstruction_metrics_all_folds.csv"))
average_metrics.to_csv(os.path.join(output_dir, "pls_reconstruction_metrics_average.csv"), header=['Avg_Metric'])
print(f"\nFull reconstruction metrics saved to: {output_dir}")
print("Analysis complete.")

# --- 6. Optional: Visualize Results ---
plt.figure(figsize=(8, 5))
metrics_plot = sns.boxplot(data=metrics_df, palette="viridis")
plt.title("Distribution of Reconstruction Metrics Across Folds (PLS)", fontsize=14)
plt.ylabel("Metric Value", fontsize=12)
plt.xlabel("Metric Type", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pls_reconstruction_metrics_boxplot.pdf"))
print("Reconstruction metrics boxplot saved.")
plt.show()
'''
