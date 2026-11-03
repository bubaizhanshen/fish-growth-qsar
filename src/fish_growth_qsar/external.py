"""Chemical-weighted evaluation of external growth NOEC predictions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = (
    "mortality_calibrated", "target_rf", "target_krr", "target_tabpfn",
    "target_lgbm", "joint_natural", "joint_balanced",
)


def evaluate(predictions, models=None, primary="mortality_calibrated",
             seed=2026110101, draws=10000):
    """Summarize paired predictions at the chemical level.

    The input has one row per measured record and fitted model, with columns
    row_index, conn, split, truth, and one prediction column for each model.
    First average over records and fits within each chemical, then give each
    chemical equal weight. Paired percentile intervals resample chemicals and
    condition on the fitted models and the selected external set.
    """
    if models is None:
        models = list(MODELS)
        if "growth_logp_calibrated" in predictions.columns:
            models.insert(1, "growth_logp_calibrated")
    models = tuple(models)
    if not models or primary not in models or len(models) != len(set(models)):
        raise ValueError("Models must be unique and include the primary model.")
    required = ["row_index", "conn", "split", "truth", *models]
    missing = sorted(set(required) - set(predictions.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    data = predictions[required].copy()
    if data.empty or data[["row_index", "conn", "split"]].isna().any().any():
        raise ValueError("Records, chemicals and fits must be identified.")
    if data.duplicated(["row_index", "split"]).any():
        raise ValueError("A record can occur only once within each fit.")
    if not np.isfinite(data[["truth", *models]].to_numpy(dtype=float)).all():
        raise ValueError("Observed and predicted values must be finite.")
    records = data.groupby("row_index", sort=False)
    if (records.conn.nunique() != 1).any() or (records.truth.nunique() != 1).any():
        raise ValueError("Chemical identity and observation must agree across fits.")
    nfits = data.split.nunique()
    if (records.split.nunique() != nfits).any():
        raise ValueError("Every record must have a prediction for every fit.")
    if draws < 1:
        raise ValueError("draws must be positive.")

    rows = []
    chemical_rows = []
    for model in models:
        error = data[model] - data.truth
        t = pd.DataFrame({
            "conn": data.conn, "bias": error, "mae": error.abs(),
            "squared_error": error ** 2, "within_tenfold": (error.abs() <= 1).astype(float),
            "noec_over_10fold": (error > 1).astype(float),
            "noec_under_10fold": (error < -1).astype(float),
        })
        chemical = t.groupby("conn", sort=True).mean()
        ranks = []
        for _, fit in data.groupby("split", sort=True):
            means = fit.groupby("conn", sort=True)[[model, "truth"]].mean()
            ranks.append(means[model].rank().corr(means.truth.rank()))
        rows.append({
            "model": model, "chemicals": len(chemical),
            "records": int(data.row_index.nunique()),
            "mae": chemical.mae.mean(), "rmse": np.sqrt(chemical.squared_error.mean()),
            "bias": chemical.bias.mean(), "within_tenfold": chemical.within_tenfold.mean(),
            "noec_over_10fold": chemical.noec_over_10fold.mean(),
            "noec_under_10fold": chemical.noec_under_10fold.mean(),
            "spearman_mean": np.nanmean(ranks),
        })
        table = chemical.reset_index()
        table.insert(1, "model", model)
        chemical_rows.append(table)
    by_chemical = pd.concat(chemical_rows, ignore_index=True)
    pivot = by_chemical.pivot(index="conn", columns="model", values="mae")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(pivot), size=(draws, len(pivot)))
    paired = []
    for comparator in models:
        if comparator == primary:
            continue
        delta = (pivot[primary] - pivot[comparator]).to_numpy()
        lo, hi = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975])
        paired.append({"comparator": comparator, "difference": delta.mean(),
                       "conditional_low95": lo, "conditional_high95": hi,
                       "chemicals": len(delta)})
    return pd.DataFrame(rows), by_chemical, pd.DataFrame(paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="External prediction table (CSV or CSV.GZ)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026110101)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--models", nargs="+", help="Prediction columns to compare; include mortality_calibrated")
    args = parser.parse_args()
    metrics, chemicals, paired = evaluate(pd.read_csv(args.input), models=args.models,
                                          seed=args.seed, draws=args.draws)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "external_metrics.csv", index=False)
    chemicals.to_csv(args.output_dir / "external_chemical_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "external_paired_mae.csv", index=False)


if __name__ == "__main__":
    main()
