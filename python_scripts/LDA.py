# --- 0. Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. Configuration ---
otu_file_path = "./Data/Cleaned_data/AGP_Otu_Data.csv"
metadata_file_path = "./Data/Cleaned_data/processed_metadata.csv"
metadata_index_col = 'sample_name'
otu_index_col = 0

output_dir = ".hongrui_result/LDA_Final_Validation/"
os.makedirs(output_dir, exist_ok=True)

# MODEL PARAMETERS
chosen_k = 20                # Determined from your diagnostic "Spike" plots
topic_prior = 0.5            # Keep high to force community structure
n_top_topics_to_plot = 3     # How many of the best topics to visualize?

# --- 2. Load and Prepare Data ---
print("--- Loading and Aligning Data ---")
metadata_df = pd.read_csv(metadata_file_path, index_col=metadata_index_col)
otu_df = pd.read_csv(otu_file_path, index_col=otu_index_col)

# Align Data
common_samples = metadata_df.index.intersection(otu_df.index)
metadata_df = metadata_df.loc[common_samples]
otu_df = otu_df.loc[common_samples]

# Filter Rare OTUs
min_prevalence_count = int(0.10 * len(otu_df))
otu_binary = (otu_df > 0).astype(int)
otus_to_keep = otu_binary.columns[otu_binary.sum(axis=0) >= min_prevalence_count]
otu_filtered = otu_df[otus_to_keep]
X_otu_counts = otu_filtered.values.astype(int)

# Scale Metadata
numeric_cols = metadata_df.select_dtypes(include=np.number).columns
X_metadata = metadata_df[numeric_cols].fillna(metadata_df[numeric_cols].median())
scaler = StandardScaler()
X_meta_scaled = scaler.fit_transform(X_metadata)

print(f"Data Ready. N={X_otu_counts.shape[0]} samples.")

# --- 3. Fit Final LDA Model ---
print(f"\n--- Running Final LDA (k={chosen_k}) ---")
lda = LatentDirichletAllocation(
    n_components=chosen_k, 
    random_state=42, 
    n_jobs=-1,
    topic_word_prior=topic_prior
)
Y_topics = lda.fit_transform(X_otu_counts)

# --- 4. Cross-Validation Predictions (Get 'Predicted vs Actual' Data) ---
print("--- Running Cross-Validation Prediction ---")

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# We use cross_val_predict to get clean out-of-fold predictions for the whole dataset
# This is better than manual looping for plotting purposes
Y_pred_cv = cross_val_predict(rf, X_meta_scaled, Y_topics, cv=kf, n_jobs=-1)

# Calculate R2 per topic
r2_scores = r2_score(Y_topics, Y_pred_cv, multioutput='raw_values')

# Rank topics by predictability (R2)
ranked_indices = np.argsort(r2_scores)[::-1] # Descending order

print(f"\nTop 3 Best Topics (by R2):")
for i in range(3):
    idx = ranked_indices[i]
    print(f"  Topic {idx}: R2 = {r2_scores[idx]:.4f}")

# --- 5. Visualization Loop ---

# Set global style
sns.set(style="whitegrid")

for rank in range(n_top_topics_to_plot):
    topic_idx = ranked_indices[rank]
    r2_val = r2_scores[topic_idx]
    
    print(f"\nGenerating plots for Rank {rank+1}: Topic {topic_idx} (R2={r2_val:.3f})...")
    
    # Create a figure with 3 subplots (Composition, Drivers, Pred vs Actual)
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3)
    
    # --- PLOT 1: Biological Composition ---
    ax1 = fig.add_subplot(gs[0, 0])
    topic_dist = lda.components_[topic_idx]
    top_otu_indices = topic_dist.argsort()[:-11:-1] # Top 10
    top_otu_names = otu_filtered.columns[top_otu_indices]
    top_otu_weights = topic_dist[top_otu_indices] / topic_dist.sum()
    
    sns.barplot(x=top_otu_weights, y=top_otu_names, palette="viridis", ax=ax1)
    ax1.set_title(f"Biology: Topic {topic_idx}\n(Top 10 OTUs)")
    ax1.set_xlabel("Relative Abundance")
    
    # --- PLOT 2: Lifestyle Drivers ---
    ax2 = fig.add_subplot(gs[0, 1])
    # Refit RF on single topic to get specific feature importances
    rf_single = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_single.fit(X_meta_scaled, Y_topics[:, topic_idx])
    
    importances = rf_single.feature_importances_
    top_feat_indices = np.argsort(importances)[::-1][:10] # Top 10 features
    
    sns.barplot(x=importances[top_feat_indices], y=numeric_cols[top_feat_indices], palette="magma", ax=ax2)
    ax2.set_title(f"Drivers: What predicts Topic {topic_idx}?")
    ax2.set_xlabel("Feature Importance")
    
    # --- PLOT 3: Predicted vs Actual ---
    ax3 = fig.add_subplot(gs[0, 2])
    y_true = Y_topics[:, topic_idx]
    y_pred = Y_pred_cv[:, topic_idx]
    
    # Scatter plot
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, color='b', ax=ax3)
    
    # Add perfect fit line (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Add trend line (actual fit)
    sns.regplot(x=y_true, y=y_pred, scatter=False, color='k', line_kws={'linestyle':'dotted'}, ax=ax3, label='Actual Fit')

    ax3.set_title(f"Performance: Predicted vs Actual\n($R^2$ = {r2_val:.3f})")
    ax3.set_xlabel("Actual Topic Weight")
    ax3.set_ylabel("Predicted Topic Weight")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_dir + f"Topic_{topic_idx}_Full_Analysis.pdf")
    plt.show()

print("\n--- Analysis Complete ---")