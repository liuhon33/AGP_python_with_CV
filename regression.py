import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm.notebook import tqdm # Use 'tqdm' if running in a regular terminal
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Configuration ---
# File paths
otu_file_path = "./Data/Cleaned_data/AGP_Otu_Data.csv"
metadata_file_path = "./Data/Cleaned_data/processed_metadata.csv"
metadata_index_col = 'sample_name'
otu_index_col = 0

# CV and Filtering
n_splits_cv = 5
random_state_cv = 42
abundance_threshold = 0.0001

# --- NEW: Define range of alpha (lambda) values to test ---
# Using a logarithmic scale is common for regularization parameters
alphas_to_test = np.logspace(-3, 3, 7) # Test 7 alphas from 0.001 to 1000
print(f"Alphas to test: {alphas_to_test}")
# ---

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

# Log Transform OTU Data
otu_df_log_transformed = np.log1p(otu_df_filtered)
print("Applied log(x+1) transformation to OTU counts.")

# Prepare metadata predictors
numeric_cols = metadata_df.select_dtypes(include=np.number).columns
X_metadata = metadata_df[numeric_cols]
if X_metadata.isnull().values.any():
    print("Warning: Missing values found in metadata. Filling with column medians.")
    X_metadata = X_metadata.fillna(X_metadata.median())
print(f"Using {X_metadata.shape[1]} numeric metadata features as predictors.")

# --- 3. Run Cross-Validated Ridge Regression with Alpha Tuning ---
print(f"\n--- Running {n_splits_cv}-Fold CV Ridge Regression with alpha tuning ---")

otu_best_r2_scores = {} # Store the BEST R2 for each OTU
otu_best_alphas = {} # Store the BEST alpha for each OTU
# Store predictions from the last fold using the BEST alpha for the top OTU plot
last_fold_predictions = {'otu_name': None, 'y_val': None, 'y_pred': None, 'r2': None, 'best_alpha': None}

for otu_name in tqdm(otu_df_log_transformed.columns, desc="Processing OTUs"):
    y_otu_log = otu_df_log_transformed[otu_name]

    if y_otu_log.var() < 1e-9:
        otu_best_r2_scores[otu_name] = np.nan
        continue

    # --- NEW: Store results for different alphas for this OTU ---
    alpha_results = {} # Key: alpha, Value: average R2 across folds

    for alpha in alphas_to_test:
        fold_scores = []
        kf = KFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state_cv)

        for fold, (train_index, val_index) in enumerate(kf.split(X_metadata)):
            X_train, X_val = X_metadata.iloc[train_index], X_metadata.iloc[val_index]
            y_train, y_val = y_otu_log.iloc[train_index], y_otu_log.iloc[val_index]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Use the current alpha
            model = Ridge(alpha=alpha)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            score = r2_score(y_val, y_pred)
            fold_scores.append(score)

        # Store the average R2 for this alpha
        alpha_results[alpha] = np.mean(fold_scores)
    # --- End Alpha Loop ---

    # --- NEW: Find the best alpha and its R2 for this OTU ---
    best_alpha = max(alpha_results, key=alpha_results.get)
    best_r2 = alpha_results[best_alpha]
    otu_best_r2_scores[otu_name] = best_r2
    otu_best_alphas[otu_name] = best_alpha
    # ---

    # --- Update logic for the scatter plot ---
    # Check if this OTU is the best overall so far
    current_best_otu = last_fold_predictions['otu_name']
    if current_best_otu is None or best_r2 > otu_best_r2_scores.get(current_best_otu, -np.inf):
        # We need to re-run the *last fold* with the *best alpha* to get the plot data
        last_fold_train_idx, last_fold_val_idx = list(kf.split(X_metadata))[-1]
        X_train_lf, X_val_lf = X_metadata.iloc[last_fold_train_idx], X_metadata.iloc[last_fold_val_idx]
        y_train_lf, y_val_lf = y_otu_log.iloc[last_fold_train_idx], y_otu_log.iloc[last_fold_val_idx]

        scaler_lf = StandardScaler()
        X_train_scaled_lf = scaler_lf.fit_transform(X_train_lf)
        X_val_scaled_lf = scaler_lf.transform(X_val_lf)

        model_lf = Ridge(alpha=best_alpha) # Use best alpha
        model_lf.fit(X_train_scaled_lf, y_train_lf)
        y_pred_lf = model_lf.predict(X_val_scaled_lf)
        r2_lf = r2_score(y_val_lf, y_pred_lf)

        last_fold_predictions['otu_name'] = otu_name
        last_fold_predictions['y_val'] = y_val_lf
        last_fold_predictions['y_pred'] = y_pred_lf
        last_fold_predictions['r2'] = r2_lf # R2 from the last fold with best alpha
        last_fold_predictions['best_alpha'] = best_alpha # Store the best alpha used

# --- 4. Rank OTUs and Display Top 10 ---
print("\n--- Top 10 Most Predictable OTUs by Metadata (Average R^2 after alpha tuning) ---")
results_series = pd.Series(otu_best_r2_scores).dropna() # Use the BEST R2 scores
top_otus = results_series.sort_values(ascending=False)
print(top_otus.head(10))
print("Corresponding Best Alphas:")
print(pd.Series(otu_best_alphas).loc[top_otus.head(10).index]) # Show best alphas for top 10
print("---------------------------------------------------------------------------------")

# --- 5. Generate Plots ---
print("\n--- Generating Plots ---")

# Plot 1: Bar plot of Top 10 OTU R^2
top_10_df = top_otus.head(10).reset_index()
top_10_df.columns = ['OTU', 'Best Average R^2']

plt.figure(figsize=(10, 6))
sns.barplot(x='Best Average R^2', y='OTU', data=top_10_df, palette='viridis', orient='h')
plt.title('Top 10 Most Predictable OTUs (Best Average R^2 from 5-Fold CV with Alpha Tuning)')
plt.xlabel('Best Average R-squared')
plt.ylabel('OTU Identifier')
plt.xlim(left=max(0, top_10_df['Best Average R^2'].min() - 0.05), right=top_10_df['Best Average R^2'].max() + 0.05)
plt.tight_layout()
plt.savefig("top_10_otu_best_r2_barplot.pdf")
print("Top 10 OTU best R^2 bar plot saved as 'top_10_otu_best_r2_barplot.pdf'")
plt.close()

# Plot 2: Scatter plot for the single best OTU (from last CV fold using best alpha)
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
    plt.savefig("top_otu_prediction_scatter_tuned.pdf")
    print(f"Prediction scatter plot for top OTU ({top_otu_name}) saved as 'top_otu_prediction_scatter_tuned.pdf'")
    plt.close()
else:
    print("Could not generate scatter plot: No OTUs found after filtering or all had zero variance.")

print("\nAnalysis complete.")