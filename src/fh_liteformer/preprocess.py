"""Recompute the frozen normalization using eligible Training families only."""
import argparse
from pathlib import Path
import numpy as np
from .data import read_split
from .data_math import fit_scaler
from .io import config, configure_runtime, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/frozen_graph'))
    parser.add_argument('--output', type=Path, default=Path('runs/preprocessing.json'))
    args = parser.parse_args()
    configure_runtime('cpu')
    arrays, meta = read_split(args.data, 'train')
    actual = fit_scaler(arrays, meta)
    expected = config('preprocessing')
    if actual['training_families'] != expected['training_families']:
        raise ValueError('Normalization used a different Training-family population')
    for key in ('node_mean', 'node_scale', 'condition_mean', 'condition_scale', 'target_scale'):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-10, atol=1e-10)
    save_json(args.output, actual)
    print(f'Train-only normalization verified: {len(meta)} endpoints, {meta.family.nunique()} families')


if __name__ == '__main__':
    main()
