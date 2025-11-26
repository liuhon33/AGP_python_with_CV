# --- 0. Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import LatentDirichletAllocation # <-- NEW MODEL
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm.notebook import tqdm 
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Configuration ---
otu_file_path = "./Data/Cleaned_data/AGP_Otu_Data.csv"
metadata_file_path = "./Data/Cleaned_data/processed_metadata.csv"
metadata_index_col = 'sample_name'
otu_index_col = 0

# CV and Filtering
outer_cv_splits = 5
inner_cv_splits = 3
random_state_cv = 42
n_topics = 5  # Number of "Microbial Topics" to find (similar to PCs)

# RF Hyperparameter Grid
param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [5, 10],
    'max_depth': [10, None]
}

output_dir = "./RF_on_LDA_Results/"
os.makedirs(output_dir, exist_ok=True)

# --- 2. Load Data ---
print("--- Loading data ---")
metadata_df = pd.read_csv(metadata_file_path, index_col=metadata_index_col)
otu_df = pd.read_csv(otu_file_path, index_col=otu_index_col)

# Align samples
common_samples = metadata_df.index.intersection(otu_df.index)
metadata_df = metadata_df.loc[common_samples]
otu_df = otu_df.loc[common_samples]
print(f"Data aligned. Found {len(common_samples)} common samples.")

# --- 3. PREVALENCE FILTERING (Crucial for LDA) ---
# Remove rare bugs to speed up LDA and remove noise
min_prevalence_count = int(0.10 * len(otu_df)) # Present in at least 10% of samples
otu_binary = (otu_df > 0).astype(int)
otus_to_keep = otu_binary.columns[otu_binary.sum(axis=0) >= min_prevalence_count]
otu_filtered = otu_df[otus_to_keep]

print(f"Initial OTUs: {otu_df.shape[1]}")
print(f"OTUs after 10% prevalence filter: {otu_filtered.shape[1]}")

# ** IMPORTANT: NO LOG TRANSFORM FOR LDA **
# LDA expects raw integer counts.
X_otu_counts = otu_filtered.values.astype(int)

# Prepare Metadata (Predictors)
numeric_cols = metadata_df.select_dtypes(include=np.number).columns
X_metadata = metadata_df[numeric_cols]
if X_metadata.isnull().values.any():
    X_metadata = X_metadata.fillna(X_metadata.median())
    
# Scale Metadata (RF doesn't strictly need it, but good practice)
scaler_meta = StandardScaler()
X_meta_scaled = scaler_meta.fit_transform(X_metadata)

# --- 4. Run LDA (Topic Modeling) ---
print(f"\n--- Running LDA to extract {n_topics} Microbial Topics ---")
print("(This may take a few minutes...)")

# Initialize LDA
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=random_state_cv,
    n_jobs=-1, # Use all cores
    learning_method='batch' 
)

# Fit LDA and transform data to get "Topic Weights" for each person
# Y_topics shape: (n_samples, n_topics)
Y_topics = lda.fit_transform(X_otu_counts)

print("LDA Complete. Target matrix shape:", Y_topics.shape)

# --- 5. Nested CV: Predict Topic Weights from Lifestyle ---
print(f"\n--- Running {outer_cv_splits}-Fold Nested CV to Predict Topics ---")

nested_cv_results = {}

for i in range(n_topics):
    topic_name = f'Topic_{i+1}'
    y_target = Y_topics[:, i] # Target is the weight of Topic i
    
    print(f"\n... Processing {topic_name} ...")
    start_time = time.time()
    
    outer_cv = KFold(n_splits=outer_cv_splits, shuffle=True, random_state=random_state_cv)
    outer_loop_scores = []
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, test_idx) in enumerate(tqdm(outer_cv.split(X_meta_scaled), total=outer_cv_splits)):
        X_train, X_test = X_meta_scaled[train_idx], X_meta_scaled[test_idx]
        y_train, y_test = y_target[train_idx], y_target[test_idx]

        # Inner Loop (Tuning)
        rf = RandomForestRegressor(random_state=random_state_cv)
        inner_cv = KFold(n_splits=inner_cv_splits, shuffle=True, random_state=random_state_cv)
        
        grid = GridSearchCV(rf, param_grid, cv=inner_cv, scoring='r2', n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
        
        # Evaluate
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        
        score = r2_score(y_test, y_pred)
        outer_loop_scores.append(score)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    avg_r2 = np.mean(outer_loop_scores)
    
    nested_cv_results[topic_name] = {
        'avg_r2': avg_r2,
        'y_true': np.array(y_true_all),
        'y_pred': np.array(y_pred_all)
    }
    
    print(f"  {topic_name} Average R^2: {avg_r2:.4f} (Time: {time.time()-start_time:.1f}s)")

# --- 6. Results Summary ---
print("\n--- Final Results: Predictability of Microbial Topics ---")
r2_dict = {k: v['avg_r2'] for k, v in nested_cv_results.items()}
results_df = pd.DataFrame.from_dict(r2_dict, orient='index', columns=['Nested CV R2'])
results_df = results_df.sort_values('Nested CV R2', ascending=False)
print(results_df)
results_df.to_csv(output_dir + "lda_rf_results.csv")

# --- 7. Interpret the Best Topic (What bugs are in it?) ---
best_topic_name = results_df.index[0]
best_topic_idx = int(best_topic_name.split('_')[1]) - 1

print(f"\n--- Inspecting the Best Performing Topic: {best_topic_name} ---")
# Get the distribution of words (OTUs) for this topic
topic_otu_dist = lda.components_[best_topic_idx]
# Sort to find top OTUs
top_otu_indices = topic_otu_dist.argsort()[:-11:-1] # Top 10
top_otus = otu_filtered.columns[top_otu_indices]
top_weights = topic_otu_dist[top_otu_indices]

# Plot Top OTUs in this Topic
plt.figure(figsize=(10, 6))
sns.barplot(x=top_weights, y=top_otus, palette='viridis')
plt.title(f"Top 10 OTUs in {best_topic_name}\n(The topic most predicted by lifestyle)")
plt.xlabel("Weight in Topic")
plt.tight_layout()
plt.savefig(output_dir + f"top_otus_{best_topic_name}.pdf")
plt.show()

# --- 8. Prediction Scatter Plot ---
best_data = nested_cv_results[best_topic_name]
plt.figure(figsize=(7, 7))
plt.scatter(best_data['y_true'], best_data['y_pred'], alpha=0.5, label="Predictions")

# Fit line
lr = LinearRegression().fit(best_data['y_true'].reshape(-1,1), best_data['y_pred'])
line_x = np.linspace(best_data['y_true'].min(), best_data['y_true'].max(), 100).reshape(-1,1)
line_y = lr.predict(line_x)
plt.plot(line_x, line_y, 'r--', lw=2, label=f"Actual Fit (R2={lr.score(best_data['y_true'].reshape(-1,1), best_data['y_pred']):.3f})")

plt.title(f"Predicting {best_topic_name} from Lifestyle")
plt.xlabel("True Topic Weight")
plt.ylabel("Predicted Topic Weight")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir + f"scatter_{best_topic_name}.pdf")
plt.show()