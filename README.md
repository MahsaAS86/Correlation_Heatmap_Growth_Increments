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

# --- Load dataset ---
file_path = r"C:\Users\malidoostsal\datafile.csv"
df = pd.read_csv(file_path)
print("Columns in file:", list(df.columns))
print("\n✅ Data loaded successfully!")
print(f"Shape: {df.shape[0]} × {df.shape[1]}\n")

# --- Define only the mean columns ---
mean_cols = ["TML", "CL", "RM", "Mutvei"]

for c in mean_cols:
    if c not in df.columns:
        raise ValueError(f"Column '{c}' not found in dataset!")

# --- ⚙️ Handle missing values safely ---
# Drop rows where ALL methods are NaN (feature measured by none)
df_mean = df[mean_cols].dropna(how="all")

# We do NOT impute or fill values here — each method keeps its own NaNs
print("\nMissing values per column:")
print(df_mean.isna().sum())

# --- Helper: compute correlation matrix, p-value matrix, and avg |r| ---
def corr_with_sig(df_subset):
    cols = df_subset.columns
    n = len(cols)
    corr = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
    pval = pd.DataFrame(np.ones((n, n)), columns=cols, index=cols)
    for i in range(n):
        for j in range(i+1, n):  # compute only upper triangle
            valid = df_subset[[cols[i], cols[j]]].dropna()
            if len(valid) >= 3:
                r, p = pearsonr(valid[cols[i]], valid[cols[j]])
                corr.iloc[i, j] = corr.iloc[j, i] = r
                pval.iloc[i, j] = pval.iloc[j, i] = p
    mask = ~np.eye(n, dtype=bool)
    avg_r = corr.abs().where(mask).stack().mean()
    return corr, pval, avg_r

# --- Compute correlations and p-values for mean columns ---
corr_mean, p_mean, avg_r_mean = corr_with_sig(df_mean)

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
plt.title("Correlation Among Imaging Techniques (Means)", fontsize=12)

for x in range(len(corr_mean)):
    for y in range(len(corr_mean)):
        if p_mean.iloc[x, y] < alpha and x != y:
            ax.add_patch(plt.Rectangle((y, x), 1, 1, fill=False, edgecolor='yellow', lw=3))

plt.tight_layout()
plt.savefig(r"C:\Users\malidoostsal\LO Methods\Correlation_Archae.pdf",
            dpi=600, bbox_inches='tight', transparent=True)
plt.show()

# --- Cronbach’s α ---
def cronbach_alpha(df_subset):
    df_subset = df_subset.dropna(axis=0)  # drop only rows where ANY method missing
    k = df_subset.shape[1]
    variances = df_subset.var(axis=0, ddof=1)
    total_var = df_subset.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    alpha = (k / (k - 1)) * (1 - variances.sum() / total_var)
    return alpha

alpha = cronbach_alpha(df_mean)
print(f"\nCronbach’s α (internal consistency): {alpha:.3f}")

# --- Intra-class correlation (ICC) ---
try:
    import pingouin as pg
except ImportError:
    print("\n⚠️  pingouin not installed — run `pip install pingouin` to compute ICC.")
else:
    # ⚙️ FIX: Use only complete rows (samples measured by ALL methods)
    df_complete = df_mean.dropna(axis=0)

    # Melt to long format
    df_long = df_complete.melt(var_name='method', value_name='value', ignore_index=False).reset_index(names='subject')

    # Compute ICC (no missing or unbalanced data now)
    icc = pg.intraclass_corr(data=df_long, targets='subject', raters='method', ratings='value')
    icc2 = icc[icc['Type'] == 'ICC2']

    print("\nIntra-class correlation (ICC2, absolute agreement):")
    print(icc2[['Type', 'ICC', 'CI95%']])

# --- Dataset summary ---
print("Complete rows:", df_mean.dropna().shape[0])
print(df_mean.describe().T[['mean', 'std', 'min', 'max']])
```
