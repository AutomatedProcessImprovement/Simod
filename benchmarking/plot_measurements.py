from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

measurements_path = Path(__file__).parent / "results/measurements.csv"

df = pd.read_csv(measurements_path).sort_values(by=["metric", "name", "simod_version"])

ncols = 4
nrows = df["metric"].nunique() * df["name"].nunique() / ncols

fig, axes = plt.subplots(nrows=int(nrows), ncols=int(ncols), figsize=(20, 40))

for group_name, group_df in df.groupby(["metric", "name"]):
    metric, name = group_name
    ax = axes.flatten()[list(df["metric"].unique()).index(metric) * 4 + list(df["name"].unique()).index(name)]
    ax.set_title(f"{metric} - {name}")
    sns.barplot(data=group_df, x="simod_version", y="distance", ax=ax)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "measurements.png")
