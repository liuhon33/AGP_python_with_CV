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
    otu_rel_abund = otu_df.apply(lambda x: x / x.sum() if x.sum() > 0 else x, axis=1)
    mean_rel_abund = otu_rel_abund.mean(axis=0)
    otus_to_keep = mean_rel_abund[mean_rel_abund > threshold].index
    return otu_df[otus_to_keep]

# --- 1. Configuration ---
# File Paths
otu_file_path = "./Data/Cleaned_data/AGP_Otu_Data.csv"
metadata_file_path = "./Data/Cleaned_data/processed_metadata.csv"
metadata_index_col = 'sample_name'
otu_index_col = 0

output_dir = "./PLS_CV_Reconstruction_py/" # Adjusted output directory

# Parameters
otu_abundance_threshold = 0.0001
n_splits_cv = 5
n_components_pls = 10 # Number of PLS components to use for reconstruction
random_state_cv = 123

# --- 2. Create Output Dir ---
os.makedirs(output_dir, exist_ok=True)

# --- 3. Load and Preprocess OTU Data ---
print("--- Loading and preprocessing OTU data ---")
otu_df = pd.read_csv(otu_file_path, index_col=otu_index_col)
print(f"Initial number of OTUs: {otu_df.shape[1]}")

# Apply filtering
otu_df_filt = filter_otus_by_abundance(otu_df, threshold=otu_abundance_threshold)
print(f"Number of OTUs after filtering (> {otu_abundance_threshold*100:.3f}%): {otu_df_filt.shape[1]}")

# Log transform (log1p) - This is the microbiome matrix (X)
X_microbiome_log = np.log1p(otu_df_filt)
print("Applied log(x+1) transformation.")

# --- 4. Load and Preprocess Metadata ---
print("--- Loading and preprocessing Metadata ---")
metadata_df = pd.read_csv(metadata_file_path, index_col=metadata_index_col)

# --- 5. Align Dataframes ---
common_samples = X_microbiome_log.index.intersection(metadata_df.index)
print(f"Found {len(common_samples)} common samples between OTU and Metadata.")
X_microbiome_full = X_microbiome_log.loc[common_samples]
metadata_aligned = metadata_df.loc[common_samples]
# Ensure same order
metadata_aligned = metadata_aligned.sort_index()
X_microbiome_full = X_microbiome_full.sort_index()

# --- 6. Select Numeric Metadata & Prepare Y Matrix ---
numeric_cols = metadata_aligned.select_dtypes(include=np.number).columns
Y_lifestyle_full = metadata_aligned[numeric_cols]

if Y_lifestyle_full.isnull().values.any():
    print("Warning: Missing values found in metadata. Imputing with column medians.")
    Y_lifestyle_full = Y_lifestyle_full.fillna(Y_lifestyle_full.median())

Y_lifestyle_full_np = Y_lifestyle_full.values
X_microbiome_full_np = X_microbiome_full.values # log-transformed counts

print(f"Selected {Y_lifestyle_full_np.shape[1]} numeric lifestyle variables.")

# --- 7. Perform K-Fold Cross-Validation for Reconstruction ---
print(f"\n--- Performing {n_splits_cv}-Fold CV for Microbiome Reconstruction ---")
kf = KFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state_cv)

# Store reconstruction metrics for each fold
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
    # We need X_test unscaled for comparison later, but also scaled for potential PLS steps
    X_test_scaled = scaler_x.transform(X_test)

    # Determine number of components dynamically for this fold
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
    # X_reconstructed = Scores @ Loadings.T
    X_reconstructed_scaled = predicted_X_scores @ pls_model.x_loadings_.T

    # Inverse transform the reconstruction back to the original log-count scale
    X_reconstructed = scaler_x.inverse_transform(X_reconstructed_scaled)

    # Compare the reconstruction (X_reconstructed) with the original X_test (log-transformed)
    # Calculate metrics over the entire X matrix for this fold
    r2 = r2_score(X_test, X_reconstructed)
    mse = mean_squared_error(X_test, X_reconstructed)
    mae = mean_absolute_error(X_test, X_reconstructed)

    reconstruction_metrics.append({'R2': r2, 'MSE': mse, 'MAE': mae})

# --- 8. Aggregate and Display Results ---
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

# --- Optional: Visualize Results ---
plt.figure(figsize=(8, 5))
metrics_plot = sns.boxplot(data=metrics_df, palette="viridis")
plt.title("Distribution of Reconstruction Metrics Across Folds (PLS)", fontsize=14)
plt.ylabel("Metric Value", fontsize=12)
plt.xlabel("Metric Type", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pls_reconstruction_metrics_boxplot.pdf"))
print("Reconstruction metrics boxplot saved.")
plt.show()
