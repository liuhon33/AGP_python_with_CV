from pathlib import Path
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pc_dir = Path("/scratch/liuhon33/parallel/AGPMicrobiomeHostPredictions/hongrui_result/layered_network_output/SHAP/PC1")
pc_name = "PC1"

rename_dict = {
    "fruit_frequency": "fruit",
    "vegetable_frequency": "vegetables",
    "vitamin_d_supplement_frequency": "vitamin D supplement",
    "vitamin_b_supplement_frequency": "vitamin B supplement",
    "whole_grain_frequency": "whole grains",
    "salted_snacks_frequency": "salty snacks",
    "one_liter_of_water_a_day_frequency": "water intake",
    "milk_cheese_frequency": "milk/cheese",
    "milk_substitute_frequency": "milk substitute",
    "bowel_movement_frequency": "bowel movement",
}

X_imp = pd.read_csv(pc_dir / f"{pc_name}_X_imputed_wide.csv.gz", index_col=0)
shap_df = pd.read_csv(pc_dir / f"{pc_name}_shap_values_wide.csv.gz", index_col=0)

# align rows
common_idx = X_imp.index.intersection(shap_df.index)
X_imp = X_imp.loc[common_idx]
shap_df = shap_df.loc[common_idx]

# align columns
common_cols = [c for c in X_imp.columns if c in shap_df.columns]
X_imp = X_imp[common_cols]
shap_df = shap_df[common_cols]

X_plot = X_imp.rename(columns=rename_dict)

# new SHAP API: build an Explanation object
explanation = shap.Explanation(
    values=shap_df.to_numpy(),
    data=X_plot.to_numpy(),
    feature_names=list(X_plot.columns),
)

plt.figure()
shap.plots.beeswarm(
    explanation,
    max_display=27,
    show=False,
    plot_size=(8, 14),
)

fig = plt.gcf()
ax = plt.gca()

ax.set_xlabel(f"SHAP value for {pc_name} prediction", fontsize=18, fontweight="bold")
ax.set_title(f"Predictors for microbiome {pc_name}", fontsize=22, fontweight="bold", pad=12)

ax.tick_params(axis="x", labelsize=15)
ax.tick_params(axis="y", labelsize=15)

if len(fig.axes) > 1:
    cbar_ax = fig.axes[-1]
    cbar_ax.set_ylabel("Feature value", fontsize=17, fontweight="bold")
    cbar_ax.tick_params(labelsize=14)
    for tick in cbar_ax.get_yticklabels():
        tick.set_fontweight("bold")

if len(fig.axes) > 1:
    cbar_ax = fig.axes[-1]
    cbar_ax.tick_params(labelsize=12)
    cbar_ax.set_ylabel("Feature value", fontsize=14)

plt.tight_layout()
plt.savefig(pc_dir / f"{pc_name}_shap_summary_beeswarm.png", dpi=300)
plt.close()

print("done")