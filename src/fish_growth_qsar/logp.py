#!/usr/bin/env python3
"""Refit the matched log P mortality comparator and its growth intercept."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


def fit(source):
    """Fit mortality concentrations on a molar scale with record weights."""
    categories = source[["effect", "endpoint_type"]].fillna("unknown")
    encoder = OneHotEncoder(handle_unknown="ignore").fit(categories)
    numeric = np.column_stack(
        [source.parent_logP, np.log10(source.duration_h.clip(lower=1))]
    )
    x = sparse.hstack(
        [sparse.csr_matrix(numeric), encoder.transform(categories)], format="csr"
    )
    y = source.value_for_model.to_numpy(float) - np.log10(source.reported_structure_MW) - 3
    group_size = source.groupby(["structure_key", "effect", "endpoint_type"])[
        "record_id"
    ].transform("size")
    model = LinearRegression().fit(
        x, y, sample_weight=1 / np.sqrt(group_size.to_numpy(float))
    )
    return model, encoder


def predict(model, encoder, frame):
    """Predict 96 h mortality LC50 in log10(mg/L)."""
    n = len(frame)
    categories = pd.DataFrame({"effect": ["MOR"] * n, "endpoint_type": ["LC50"] * n})
    numeric = np.column_stack([frame.parent_logP, np.full(n, np.log10(96.0))])
    x = sparse.hstack(
        [sparse.csr_matrix(numeric), encoder.transform(categories)], format="csr"
    )
    return np.asarray(model.predict(x)) + np.log10(frame.reported_structure_MW) + 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=int, choices=range(6), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.data / f"source_{args.split}.csv.gz", dtype={"record_id": str})
    growth = pd.read_csv(args.data / f"growth_{args.split}.csv", dtype={"record_id": str})
    query = pd.read_csv(args.data / "query.csv")
    if len(growth) != 20 or growth.parent_group.nunique() != 20:
        raise ValueError("Expected one growth record for each of 20 chemicals")

    oof = np.full(len(growth), np.nan)
    for fold in sorted(growth.source_fold.unique()):
        selected = growth.source_fold.eq(fold).to_numpy()
        model, encoder = fit(source[source.excluded_fold.ne(fold)])
        oof[selected] = predict(model, encoder, growth.loc[selected])
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete cross-fitted predictions")

    offset = float(np.mean(growth.value_for_model - oof))
    model, encoder = fit(source)
    result = query[["row_index", "conn"]].copy()
    result["split"] = args.split
    result["growth_logp_calibrated"] = predict(model, encoder, query) + offset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Split {args.split}: {len(source)} mortality records; intercept {offset:.6f}.")


if __name__ == "__main__":
    main()
