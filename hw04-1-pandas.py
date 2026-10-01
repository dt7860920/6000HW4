import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psutil


def compute_memory_usage(filepath=None):
    """
    Measure the current Python process's RAM usage in MB.
    Optionally write the measurement to a text file.
    """
    process = psutil.Process()
    mem_usage_mb = process.memory_info().rss / (1024 ** 2)
    mem_usage_str = f"{mem_usage_mb:.2f} MB"

    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as outfile:
            outfile.write(mem_usage_str)

    return mem_usage_mb


if __name__ == "__main__":
    memory_pre = compute_memory_usage("outputs/hw04-1-pandas-pre.txt")
    print(f"{memory_pre:.2f} MB used pre-computation")

    s3_uri = "s3://dsan6000-data/acled_civilian_events.parquet"

    acled_df = pd.read_parquet(s3_uri)
    acled_df = acled_df[acled_df["year"].between(2018, 2024)]

    yearly_df = (
        acled_df.groupby("year", as_index=False)["fatalities"]
        .sum()
        .sort_values("year")
    )

    sns.lineplot(
        data=yearly_df,
        x="year",
        y="fatalities",
        marker="o",
    )
    plt.title("ACLED: Yearly Fatality Counts (Pandas)")
    plt.tight_layout()
    plt.savefig("images/yearly_fatalities_pandas.svg")
    plt.close()

    memory_post = compute_memory_usage("outputs/hw04-1-pandas-post.txt")
    print(f"{memory_post:.2f} MB used post-computation")

    memory_diff = memory_post - memory_pre
    print(f"=> {memory_diff:.2f} MB added via Pandas operation")

    with open(
        "outputs/hw04-1-pandas-diff.txt", "w", encoding="utf-8"
    ) as outfile:
        outfile.write(f"{memory_diff:.2f} MB")