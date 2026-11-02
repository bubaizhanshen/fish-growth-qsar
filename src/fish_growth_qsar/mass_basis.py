"""Compare external model errors for the existing mass-basis subsets."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fish_growth_qsar.external import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--mass-audit', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    audit = pd.read_csv(args.mass_audit)
    required = ['row_index', 'conn', 'standard_minus_raw_log10']
    if audit[required].isna().any().any() or audit.row_index.duplicated().any():
        raise ValueError('The mass audit must identify each record once')
    rows = predictions[['row_index', 'conn']].drop_duplicates()
    data = rows.merge(audit[required], on=['row_index', 'conn'],
                      how='left', validate='one_to_one')
    if data.standard_minus_raw_log10.isna().any():
        raise ValueError('Every prediction record needs a mass-basis audit')
    data['mass_basis_changed'] = data.standard_minus_raw_log10.abs() > 1e-6
    if (data.groupby('conn').mass_basis_changed.nunique() != 1).any():
        raise ValueError('A chemical has records in both mass-basis groups')

    groups = {
        'all': set(data.conn),
        'mass_basis_consistent': set(data.loc[~data.mass_basis_changed, 'conn']),
        'mass_basis_changed': set(data.loc[data.mass_basis_changed, 'conn']),
    }
    metrics, paired = [], []
    for group, chemicals in groups.items():
        subset = predictions[predictions.conn.isin(chemicals)]
        summary, _, comparisons = evaluate(subset)
        summary.insert(0, 'group', group)
        comparisons.insert(0, 'group', group)
        metrics.append(summary)
        paired.append(comparisons)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(metrics, ignore_index=True).to_csv(
        args.output_dir / 'mass_basis_subset_metrics.csv', index=False
    )
    pd.concat(paired, ignore_index=True).to_csv(
        args.output_dir / 'mass_basis_subset_paired.csv', index=False
    )


if __name__ == '__main__':
    main()
