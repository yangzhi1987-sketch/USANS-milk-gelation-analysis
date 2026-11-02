#!/usr/bin/env python3
"""Nested leave-one-system-out validation for concise reference models."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "q_synchronised_USANS_rheology_dataset.csv"
OUT = ROOT / "reproduced_results"
SEED = 20260924


def candidate_models(n_features):
    models = []
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        models.append(("Ridge", {"alpha": alpha}, Pipeline([
            ("scale", StandardScaler()), ("model", Ridge(alpha=alpha))
        ])))
    for n_components in [2, 4, 6, 8]:
        if n_components <= n_features:
            models.append(("PLS", {"n_components": n_components}, Pipeline([
                ("scale", StandardScaler()),
                ("model", PLSRegression(n_components=n_components, scale=False)),
            ])))
    return models


def time_baseline_candidates():
    models = []
    for degree in [1, 2, 3]:
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            models.append(("Time-only baseline", {"degree": degree, "alpha": alpha}, Pipeline([
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=alpha)),
            ])))
    return models


def choose_by_inner_loso(candidates, x, y, groups):
    best = None
    for name, params, model in candidates:
        predictions = np.full(len(y), np.nan)
        for heldout in pd.unique(groups):
            train = groups != heldout
            valid = ~train
            model.fit(x[train], y[train])
            predictions[valid] = np.asarray(model.predict(x[valid])).ravel()
        rmse = mean_squared_error(y, predictions) ** 0.5
        if best is None or rmse < best[0]:
            best = (rmse, name, params, model)
    return best


def main():
    np.random.seed(SEED)
    df = pd.read_csv(DATA)
    features = [c for c in df.columns if c.startswith("logI_q_")]
    groups = df["sample_id"].to_numpy()
    y = df["log10_Gprime"].to_numpy(float)
    rows = []

    scenarios = [
        ("USANS", df[features].to_numpy(float), candidate_models(len(features))),
        ("Time", df[["rheology_time_min"]].to_numpy(float), time_baseline_candidates()),
    ]

    for scenario, x, candidates in scenarios:
        for heldout in pd.unique(groups):
            train = groups != heldout
            test = ~train
            best_rmse, name, params, model = choose_by_inner_loso(
                candidates, x[train], y[train], groups[train]
            )
            model.fit(x[train], y[train])
            prediction = np.asarray(model.predict(x[test])).ravel()
            rows.append(
                {
                    "model": name,
                    "heldout_sample_id": heldout,
                    "heldout_sample": df.loc[test, "sample_label"].iloc[0],
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    "R2": r2_score(y[test], prediction),
                    "RMSE_log10": mean_squared_error(y[test], prediction) ** 0.5,
                    "MAE_log10": mean_absolute_error(y[test], prediction),
                    "inner_best_RMSE_log10": best_rmse,
                    "best_parameters": str(params),
                }
            )

    OUT.mkdir(exist_ok=True)
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(OUT / "nested_loso_fold_metrics.csv", index=False)
    summary = fold_df.groupby("model", sort=False).agg(
        mean_R2=("R2", "mean"),
        sd_R2=("R2", "std"),
        mean_RMSE_log10=("RMSE_log10", "mean"),
        mean_MAE_log10=("MAE_log10", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "nested_loso_summary.csv", index=False)
    print(f"Wrote nested LOSO summaries to {OUT}")


if __name__ == "__main__":
    main()
