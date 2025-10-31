## 🧱Correlation Script (Growth increments)

This script calculates Pearson correlation coefficients between imaging techniques (RM, TML, CL)
for the **archaeological/modern dataset**, and highlights significant relationships (p < 0.05) in yellow on the heatmap.


```python
# --- Import libraries ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import os

# --- Choose dataset type (edit as needed) ---
dataset_type = "Archae"  # options: "Archae" or "Modern"

# --- Load dataset ---
file_path = fr"C:\Users\malidoostsal\LO Methods\Appendix_{dataset_type}.csv"
df = pd.read_csv(file_path)
print("Columns in file:", list(df.columns))
print(f"\n✅ Data loaded successfully! Shape: {df.shape[0]} × {df.shape[1]}\n")

# --- Define only the mean columns ---
mean_cols = ["Mean_RM", "Mean_Cl", "Mean_TML"]
for col in mean_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset!")

# --- Helper: compute correlation matrix, p-value matrix, and avg |r| ---
def corr_with_sig(df_subset):
    cols = df_subset.columns
    n = len(cols)
    corr = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
    pval = pd.DataFrame(np.ones((n, n)), columns=cols, index=cols)
    for i in range(n):
        for j in range(i + 1, n):
            valid = df_subset[[cols[i], cols[j]]].dropna()
            if len(valid) >= 3:
                r, p = pearsonr(valid[cols[i]], valid[cols[j]])
                corr.iloc[i, j] = corr.iloc[j, i] = r
                pval.iloc[i, j] = pval.iloc[j, i] = p
    mask = ~np.eye(n, dtype=bool)
    avg_r = corr.abs().where(mask).stack().mean()
    return corr, pval, avg_r

# --- Compute correlations and p-values for mean columns ---
corr_mean, p_mean, avg_r_mean = corr_with_sig(df[mean_cols])

# --- Print results ---
print("\nCorrelation (Means):")
print(corr_mean.round(3))
print("\nP-values (Means):")
print(p_mean.round(3))
print(f"\nAverage |r| (off-diagonal): {avg_r_mean:.3f}")

# --- Visualize with significance highlighting ---
alpha = 0.05
plt.figure(figsize=(5, 4))
ax = sns.heatmap(corr_mean, vmin=-1, vmax=1, cmap="coolwarm", annot=True, fmt=".2f")
plt.title(f"Correlation Among Imaging Techniques ({dataset_type} Means)", fontsize=12)

# Highlight statistically significant (p < 0.05)
for x in range(len(corr_mean)):
    for y in range(len(corr_mean)):
        if p_mean.iloc[x, y] < alpha and x != y:
            ax.add_patch(plt.Rectangle((y, x), 1, 1, fill=False, edgecolor='yellow', lw=3))

plt.tight_layout()

# 💾 Save figure as high-quality PDF
save_path = fr"C:\Users\malidoostsal\LO Methods\Correlation_{dataset_type}.pdf"
plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
plt.show()

print(f"\n✅ Figure saved to: {save_path}")

# =====================================================================
# --- Cronbach’s α (internal consistency) ---
# =====================================================================
def cronbach_alpha(df_subset):
    df_subset = df_subset.dropna(axis=0)
    k = df_subset.shape[1]
    variances = df_subset.var(axis=0, ddof=1)
    total_var = df_subset.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    alpha = (k / (k - 1)) * (1 - variances.sum() / total_var)
    return alpha

alpha_val = cronbach_alpha(df[mean_cols])
print(f"\nCronbach’s α (internal consistency): {alpha_val:.3f}")

# =====================================================================
# --- Intra-class correlation (ICC) ---
# =====================================================================
try:
    import pingouin as pg
except ImportError:
    print("\n⚠️  pingouin not installed — run `pip install pingouin` to compute ICC.")
else:
    df_long = df[mean_cols].melt(var_name='method', value_name='value', ignore_index=False).reset_index(names='subject')
    icc = pg.intraclass_corr(data=df_long, targets='subject', raters='method', ratings='value')
    icc2 = icc[icc['Type'] == 'ICC2']
    print("\nIntra-class correlation (ICC2, absolute agreement):")
    print(icc2[['Type', 'ICC', 'CI95%']])

print("\n✅ Analysis complete.")
```
