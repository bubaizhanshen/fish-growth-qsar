# Fish growth QSAR

Core analysis code for **Beyond Lethality: Using Mortality QSARs to Predict Fish Growth Toxicity**.

The method fits a fish mortality QSAR, obtains chemical-level cross-fitted
predictions for growth-training compounds, and estimates an intercept from
growth no-observed-effect concentrations (NOECs). Each training chemical receives
equal total weight. Prediction for an untested chemical uses its mortality QSAR
estimate plus this intercept. Concentrations are expressed as log10(mg/L).

## Install

```bash
python -m pip install -e .
```

Python 3.10 or later is required. To fit the mortality model, also install:

```bash
python -m pip install -e '.[source]'
```

## Reproduce growth calibration

The prepared `model_inputs` in Supporting Data S1 contain the frozen
one-record selections for five or twenty growth-training chemicals. For the
affected source fits, selected cross-fitted predictions and test predictions
reflect the corrected source records. These inputs support only the supplied
one-record selections; unselected training positions are not usable for
all-record calibration.

```bash
fish-growth --inputs data/model_inputs/fels_split_1 \
  --split 1 --budget 20 --repeat 0 --regime one_record \
  --output results/predictions.csv
```

Use `--budget 5` or `--budget 20` and `--repeat 0` through `9`. Apply the
command separately to each of six partitions and three datasets. The command
writes predictions and the fitted intercept; it reads cached source predictions
and does not refit LightGBM.

## Fit the mortality source model

`source.cross_fitted_mortality` implements source training and threefold chemical
cross-fitting. It accepts curated source, growth-training, and query records,
a molecular-property table, and a sparse Morgan fingerprint matrix. Molecular
features are prepared before calling it.

```python
from fish_growth_qsar import InterceptCalibration
from fish_growth_qsar.source import cross_fitted_mortality

m_train, m_query, source_model = cross_fitted_mortality(
    source_records, growth_records, query_records,
    properties, fingerprints, descriptor_names,
    split=0, excluded_parents=held_out_candidates,
)
calibration = InterceptCalibration.fit(
    growth_records.value_for_model, m_train, growth_records.parent_group
)
predictions = calibration.predict_log_concentration(m_query)
```

Supply the full permitted growth-training pool for cross-fitting, then apply the
desired label selection when fitting `InterceptCalibration`. This preserves the
separation between source fitting and growth-data budgets.

The record schema uses `record_id`, `parent_group`, `group`, `chemical_index`,
`structure_key`, `effect`, `endpoint_type`, `duration_h`, and `value_for_model`.
Query records need only identifiers and feature indices. `parent_group` links
aliases of a standardized chemical; `group` indexes the reported structure in
the property table; `chemical_index` indexes fingerprint rows. Source records
are fish mortality LC50/EC50 observations. Growth labels must be explicitly
recovered GRO/NOEC records. Held-out candidates must be excluded from both
training pools before fitting.

The fingerprint is radius 2 with 2048 bits. The 13 properties, in the supplied
order, are parent logP, parent molecular weight, reported-structure molecular
weight, topological polar surface area, hydrogen-bond donors and acceptors,
ring count, aromatic-ring count, sp3-carbon fraction, rotatable bonds,
original-organic-component formal charge, heteroatoms, and heavy atoms.
Species is not a model input. Source rows retain their exposure durations;
queries use mortality LC50 at 96 h.

Observations originate from the [ECOTOX Knowledgebase](https://cfpub.epa.gov/ecotox/)
and the [TRIDENT release](https://doi.org/10.1126/sciadv.adk6669), with external
growth records described in the manuscript. Raw releases require endpoint,
identity, and unit harmonization. Use the supplied memberships and prepared
features for the reported analysis. Cached predictions provide the direct route
to reproducing the released calibration; numerical refits can vary with library
versions. No pretrained model checkpoint is distributed here.

## Evaluate predictions

`metrics.performance` calculates MAE, signed error, and large-error frequencies.
`metrics.comparisons` calculates paired differences using shared chemical
Bayesian-bootstrap weights. Input tables use `split`, `repeat`, `endpoint_key`,
`group`, `parent_group`, `record_id`, and observed `value_for_model`, plus one
prediction column per model.

```python
from fish_growth_qsar.metrics import performance, comparisons

scores = performance(predictions_table, models=["calibration", "direct"], seed=42)
contrasts = comparisons(
    predictions_table, primary="calibration", references=["direct"], seed=42
)
```

Run metrics separately for each dataset and training budget. Records are averaged
within chemical and endpoint group, with equal weighting of endpoint groups and
training contexts. The simultaneous limits in `comparisons` use the six-comparison
family specified in the paper; pointwise limits are also returned. Intervals
condition on fitted models and selections. Positive concentration error indicates
underestimated growth toxicity.

## Evaluate an external test set

Supporting Data S1 contains `figure_data/external_growth_predictions.csv`,
with observations and predictions for the same external records and fits.
To reproduce the chemical-weighted performance and paired MAE comparisons:

```bash
fish-growth-external data/figure_data/external_growth_predictions.csv --output-dir results/external
```

These are conditional comparisons of the supplied predictions. Reproducing the
model fits additionally requires the curated inputs and the saved split selections.
When the prediction table contains `growth_logp_calibrated`, it is included in
the external comparison automatically.

The external growth-only random forest and kernel ridge comparators can be
refitted using the prepared 20-record selections and 95-dimensional features:

```bash
fish-growth-baselines data/external_baselines/split_0 --output results/baselines_0.csv
```

The 64 fingerprint components are fitted on source-training structures without
external-test chemicals; the remaining inputs comprise 13 descriptors, log10
duration and 17 structural indicators.

To refit the mortality source model, cross-fitted growth calibration, growth-only
LightGBM, and the two joint LightGBM comparators for the external set:

```bash
fish-growth-source data/external_joint/split_0 --output-dir results/source_0
fish-growth-joint data/external_joint/split_0 --output results/joint_0.csv
```

The matched linear mortality model based on log P uses the prepared
`mortality_logp_inputs` in Supporting Data S1 and can be refitted for each split:

```bash
fish-growth-logp --data data/mortality_logp_inputs --split 0 --output results/logp_0.csv
```

The `external_joint` folders contain the prepared source, selected growth, and
query feature matrices, training labels and weights, chemical identifiers, and
the saved growth-parent fold assignments. The 20 selected growth records are
the same for all external comparators in each fit. The source model uses the
2048 fingerprint bits, 13 molecular descriptors, duration and endpoint fields;
the joint models also use the 17 structural indicators. For the mortality model,
each query uses the 96 h LC50 setting. These inputs do not contain all features
needed to refit every model for every original evaluation selection; the TabPFN
checkpoint is not distributed here.

The refits were checked with NumPy 2.3.1, pandas 2.3.2, SciPy 1.16.0,
scikit-learn 1.7.1, and LightGBM 4.6.0. The column-contiguous layout in
`prepared_mortality.py` matches the original scaler calculation; different
versions or numerical libraries may produce different fitted trees.

## Code layout

| Module | Purpose |
|---|---|
| `source.py` | Mortality LightGBM and chemical-level cross-fitting |
| `calibration.py` | Equal-chemical mean residual and calibrated prediction |
| `predict.py` | Calibration using released inputs and label selections |
| `metrics.py` | Original primary weighting, errors, and paired intervals |
| `external.py` | Chemical-weighted external metrics and paired intervals |
| `baselines.py` | External growth-only random forest and kernel ridge comparators |
| `prepared_mortality.py` | Mortality source refits and external growth calibration |
| `joint.py` | Growth-only and joint LightGBM comparators for the external set |
| `logp.py` | Matched log P mortality comparator and growth intercept |

This version targets growth NOEC. The earlier mixed-endpoint implementation
remains in Git history.
