import pandas as pd
import numpy as np
import os # <-- IMPORTED
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression # <-- IMPORTED LinearRegression
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

# --- NEW: Set output directory ---
output_dir = ".hongrui_result/Ridge_CV_Results/"
os.makedirs(output_dir, exist_ok=True)

# CV and Filtering
n_splits_cv = 5
random_state_cv = 42
abundance_threshold = 0.0001
alphas_to_test = np.logspace(-3, 3, 7) # Test 7 alphas from 0.001 to 1000
print(f"Alphas to test: {alphas_to_test}")

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

otu_best_r2_scores = {} # Store the BEST mean R2
otu_best_r2_stds = {}  # <-- NEW: Store the standard deviation for the best R2
otu_best_alphas = {} # Store the BEST alpha
last_fold_predictions = {'otu_name': None, 'y_val': None, 'y_pred': None, 'r2': None, 'best_alpha': None}

for otu_name in tqdm(otu_df_log_transformed.columns, desc="Processing OTUs"):
    y_otu_log = otu_df_log_transformed[otu_name]
    if y_otu_log.var() < 1e-9:
        otu_best_r2_scores[otu_name] = np.nan
        otu_best_r2_stds[otu_name] = np.nan
        continue

    # Store results for different alphas: {'alpha': {'mean_r2': ..., 'std_r2': ...}}
    alpha_results = {} 

    for alpha in alphas_to_test:
        fold_scores = []
        kf = KFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state_cv)

        for fold, (train_index, val_index) in enumerate(kf.split(X_metadata)):
            X_train, X_val = X_metadata.iloc[train_index], X_metadata.iloc[val_index]
            y_train, y_val = y_otu_log.iloc[train_index], y_otu_log.iloc[val_index]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            model = Ridge(alpha=alpha)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            score = r2_score(y_val, y_pred)
            fold_scores.append(score)

        # Store the mean and std R2 for this alpha
        alpha_results[alpha] = {'mean_r2': np.mean(fold_scores), 'std_r2': np.std(fold_scores)}
    
    # Find the best alpha based on the mean_r2
    best_alpha = max(alpha_results, key=lambda a: alpha_results[a]['mean_r2'])
    best_r2_mean = alpha_results[best_alpha]['mean_r2']
    best_r2_std = alpha_results[best_alpha]['std_r2'] # <-- Get the corresponding std

    otu_best_r2_scores[otu_name] = best_r2_mean
    otu_best_r2_stds[otu_name] = best_r2_std # <-- Store it
    otu_best_alphas[otu_name] = best_alpha

    # --- Update logic for the scatter plot ---
    current_best_otu = last_fold_predictions['otu_name']
    if current_best_otu is None or best_r2_mean > otu_best_r2_scores.get(current_best_otu, -np.inf):
        kf_plot = KFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state_cv)
        last_fold_train_idx, last_fold_val_idx = list(kf_plot.split(X_metadata))[-1]
        X_train_lf, X_val_lf = X_metadata.iloc[last_fold_train_idx], X_metadata.iloc[last_fold_val_idx]
        y_train_lf, y_val_lf = y_otu_log.iloc[last_fold_train_idx], y_otu_log.iloc[last_fold_val_idx]
        scaler_lf = StandardScaler()
        X_train_scaled_lf = scaler_lf.fit_transform(X_train_lf)
        X_val_scaled_lf = scaler_lf.transform(X_val_lf)
        model_lf = Ridge(alpha=best_alpha)
        model_lf.fit(X_train_scaled_lf, y_train_lf)
        y_pred_lf = model_lf.predict(X_val_scaled_lf)
        r2_lf = r2_score(y_val_lf, y_pred_lf)
        
        last_fold_predictions['otu_name'] = otu_name
        last_fold_predictions['y_val'] = y_val_lf
        last_fold_predictions['y_pred'] = y_pred_lf
        last_fold_predictions['r2'] = r2_lf
        last_fold_predictions['best_alpha'] = best_alpha

# --- 4. Rank OTUs and Display Top 10 ---
print("\n--- Top 10 Most Predictable OTUs by Metadata (Average R^2 after alpha tuning) ---")
results_series = pd.Series(otu_best_r2_scores).dropna()
top_otus = results_series.sort_values(ascending=False)
print(top_otus.head(10))
print("Corresponding Best Alphas:")
print(pd.Series(otu_best_alphas).loc[top_otus.head(10).index])
print("---------------------------------------------------------------------------------")

# --- 5. Generate Plots ---
print("\n--- Generating Plots ---")

# --- MODIFIED: Plot 1: Bar plot of Top 10 OTU R^2 with Error Bars ---
top_10_df = top_otus.head(10).reset_index()
top_10_df.columns = ['OTU', 'Best Average R^2']
# Get the corresponding standard deviations
top_10_stds = pd.Series(otu_best_r2_stds).loc[top_10_df['OTU']].values
top_10_df['Std Dev R^2'] = top_10_stds

plt.figure(figsize=(10, 6))
# Create the bar plot using seaborn
ax = sns.barplot(x='Best Average R^2', y='OTU', data=top_10_df, palette='viridis', orient='h')
# Overlay the error bars
ax.errorbar(x=top_10_df['Best Average R^2'], y=np.arange(len(top_10_df)), 
            xerr=top_10_df['Std Dev R^2'], fmt='none', c='black', capsize=5,
            label='Std Dev across CV folds')

plt.title('Top 10 Most Predictable OTUs (Best Average R^2 from 5-Fold CV with Alpha Tuning)')
plt.xlabel('Best Average R-squared (with Std Dev across folds)')
plt.ylabel('OTU Identifier')
plt.xlim(left=0) # R2 shouldn't be negative in a good model
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_10_otu_best_r2_barplot_with_error.pdf"))
print(f"Top 10 OTU R^2 bar plot saved to '{output_dir}top_10_otu_best_r2_barplot_with_error.pdf'")
plt.close()

# --- MODIFIED: Plot 2: Scatter plot for the single best OTU with Actual Fit Line ---
if last_fold_predictions['otu_name']:
    top_otu_name = last_fold_predictions['otu_name']
    # Get true and predicted values from the last fold of the best OTU
    y_val_top = last_fold_predictions['y_val']
    y_pred_top = last_fold_predictions['y_pred']
    r2_last_fold = last_fold_predictions['r2']
    best_alpha_top = last_fold_predictions['best_alpha']
    avg_r2_top = top_otus.loc[top_otu_name] # Get the average R2

    plt.figure(figsize=(7, 7))
    plt.scatter(y_val_top, y_pred_top, alpha=0.5, label="Out-of-Fold Predictions")

    # --- Fit and plot the actual regression line ---
    # Reshape data for sklearn's LinearRegression
    true_values_reshaped = y_val_top.values.reshape(-1, 1)
    
    # Create and fit the linear model
    line_model = LinearRegression()
    line_model.fit(true_values_reshaped, y_pred_top)
    
    # Create x-values for the line
    line_x = np.array([true_values_reshaped.min(), true_values_reshaped.max()]).reshape(-1, 1)
    # Predict y-values for the line
    line_y = line_model.predict(line_x)
    
    # Get the R^2 score of this linear fit
    line_r2 = line_model.score(true_values_reshaped, y_pred_top)
    
    # Plot the new "Actual Fit" line
    plt.plot(line_x.flatten(), line_y, '--', color='red', lw=2, 
             label=f"Actual Fit ($R^2$ = {line_r2:.3f})")
    # ---

    plt.title(f'Top OTU ({top_otu_name})\nPredicted vs. Actual log(Count+1) (Last CV Fold, Best Alpha={best_alpha_top:.3f})')
    plt.xlabel('True log(Count+1)')
    plt.ylabel('Predicted log(Count+1)')
    # Updated text box to show the average CV R2
    plt.text(0.05, 0.95, f'Avg CV $R^2$ = {avg_r2_top:.3f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_otu_prediction_scatter_actual_fit.pdf"))
    print(f"Prediction scatter plot for top OTU saved to '{output_dir}top_otu_prediction_scatter_actual_fit.pdf'")
    plt.close()
else:
    print("Could not generate scatter plot: No OTUs found after filtering or all had zero variance.")

print("\nAnalysis complete.")