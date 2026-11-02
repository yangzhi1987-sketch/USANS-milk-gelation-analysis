#!/usr/bin/env python3
"""Reproduce central descriptive summaries from the synchronized dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "q_synchronised_USANS_rheology_dataset.csv"
OUT = ROOT / "reproduced_results"


def feature_metadata(columns):
    rows = []
    for column in columns:
        q = float(column.removeprefix("logI_q_"))
        length_um = 2.0 * np.pi / (q * 1.0e4)
        if length_um < 1.0:
            regime = "Small aggregates"
        elif length_um < 3.0:
            regime = "Clusters"
        else:
            regime = "Network"
        rows.append((column, q, length_um, regime))
    return pd.DataFrame(rows, columns=["feature", "q_A^-1", "length_scale_um", "regime"])


def first_downward_crossing(time, value, threshold=0.05):
    for i in range(1, len(value)):
        if value[i - 1] > threshold >= value[i]:
            fraction = (value[i - 1] - threshold) / (value[i - 1] - value[i])
            return time[i - 1] + fraction * (time[i] - time[i - 1])
    return np.nan


def main():
    df = pd.read_csv(DATA).sort_values(["sample_id", "time_min"])
    features = [c for c in df.columns if c.startswith("logI_q_")]
    metadata = feature_metadata(features)

    correlations = []
    convergence = []
    for (sample_id, label), group in df.groupby(["sample_id", "sample_label"], sort=False):
        for row in metadata.itertuples(index=False):
            correlations.append(
                {
                    "sample_id": sample_id,
                    "sample_label": label,
                    "q_A^-1": row[1],
                    "length_scale_um": row[2],
                    "regime": row[3],
                    "raw_Pearson_r": group[row[0]].corr(group["log10_Gprime"]),
                }
            )

        matrix = group[features].to_numpy(float)
        span = np.ptp(matrix, axis=0)
        span[span == 0] = 1.0
        scaled = (matrix - matrix.min(axis=0)) / span
        distance = np.sqrt(np.mean((scaled - scaled[-1]) ** 2, axis=1))
        convergence.append(
            {
                "sample_id": sample_id,
                "sample_label": label,
                "threshold": 0.05,
                "operational_crossing_min": first_downward_crossing(
                    group["time_min"].to_numpy(float), distance
                ),
            }
        )

    OUT.mkdir(exist_ok=True)
    correlation_df = pd.DataFrame(correlations)
    correlation_df.to_csv(OUT / "structure_rheology_correlations.csv", index=False)
    pd.DataFrame(convergence).to_csv(OUT / "whole_pattern_crossings.csv", index=False)
    (
        correlation_df.groupby("regime", sort=False)["raw_Pearson_r"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .to_csv(OUT / "regime_correlation_summary.csv", index=False)
    )
    metadata.to_csv(OUT / "feature_metadata.csv", index=False)
    print(f"Wrote core summaries to {OUT}")


if __name__ == "__main__":
    main()
