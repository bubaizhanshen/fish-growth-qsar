"""Apply chemical-held-out intercept calibration to paired model predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import InterceptCalibration


def apply_calibration(predictions, out_of_fold, models):
    """Calibrate each model and fit with the same growth-training chemicals.

    Out-of-fold rows contain a training chemical's observed growth NOEC and a
    prediction made without that chemical in either training endpoint. The
    caller is responsible for generating and checking those held-out fits.
    No external test outcome is used to estimate the intercepts.
    """
    models = tuple(models)
    if not models or len(set(models)) != len(models):
        raise ValueError("Supply a nonempty list of unique model columns.")
    required_predictions = {"row_index", "split", "conn", *models}
    required_oof = {"split", "model", "record_id", "parent", "observed", "oof_prediction"}
    if required_predictions - set(predictions):
        raise ValueError(
            "Missing prediction columns: "
            + ", ".join(sorted(required_predictions - set(predictions)))
        )
    if required_oof - set(out_of_fold):
        raise ValueError(
            "Missing calibration columns: " + ", ".join(sorted(required_oof - set(out_of_fold)))
        )
    pred = predictions.copy()
    oof = out_of_fold.loc[out_of_fold.model.isin(models)].copy()
    if pred.empty or oof.empty or pred.duplicated(["split", "row_index"]).any():
        raise ValueError("Prediction records must be nonempty and unique within each fit.")
    if oof.duplicated(["split", "model", "record_id"]).any():
        raise ValueError("Duplicate out-of-fold predictions for a training record.")
    if oof[["split", "model", "record_id", "parent"]].isna().any().any():
        raise ValueError("Every calibration record must have an identity.")
    if not np.isfinite(pred[list(models)].to_numpy(float)).all():
        raise ValueError("Test predictions must be finite.")
    if not np.isfinite(oof[["observed", "oof_prediction"]].to_numpy(float)).all():
        raise ValueError("Calibration values must be finite.")
    if set(pred.split) != set(oof.split):
        raise ValueError("Prediction and calibration fits differ.")

    offsets = []
    for split, group in pred.groupby("split", sort=True):
        target = oof.loc[oof.split.eq(split)]
        if set(target.model) != set(models):
            raise ValueError("A model is missing out-of-fold predictions for a fit.")
        reference = None
        for model in models:
            data = target.loc[target.model.eq(model)].sort_values("record_id")
            identity = data[["record_id", "parent", "observed"]].reset_index(drop=True)
            if reference is None:
                reference = identity
            elif not identity.equals(reference):
                raise ValueError(
                    "Models must use identical growth records and observations within each fit."
                )
            calibration = InterceptCalibration.fit(data.observed, data.oof_prediction, data.parent)
            selected = pred.split.eq(split)
            pred.loc[selected, model + "_calibrated"] = calibration.predict_log_concentration(
                group[model]
            )
            offsets.append(
                {
                    "split": split,
                    "model": model,
                    "n_growth": calibration.n_chemicals,
                    "n_records": calibration.n_records,
                    "mean_observed_minus_oof": calibration.intercept,
                }
            )
    return pred, pd.DataFrame(offsets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    predictions = pd.read_csv(args.predictions, dtype={"conn": str})
    oof = pd.read_csv(args.oof, dtype={"record_id": str})
    pred, offsets = apply_calibration(predictions, oof, args.models)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.output_dir / "calibrated_predictions.csv", index=False)
    offsets.to_csv(args.output_dir / "calibration_offsets.csv", index=False)


if __name__ == "__main__":
    main()
