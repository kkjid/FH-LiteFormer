"""Hash-checked frozen observations; targets and metadata remain separate."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .io import config, load_json, sha256

KEYS = ('values', 'conditions', 'relative_time', 'padding_mask', 'feature_missing_mask', 'mounts')
AUDIT_ONLY = {'preflight.json', 'source_binding.json', 'formal_specification.json', 'generation.json'}


def dataset_identity(directory):
    directory = Path(directory)
    expected = config('dataset')
    for name, key in [('data_hashes.json', 'dataset_hash_manifest_sha256'),
                      ('family_allocation.json', 'family_allocation_sha256'),
                      ('graph_manifest.json', 'graph_manifest_sha256')]:
        if sha256(directory / name) != expected[key]:
            raise ValueError(f'Frozen dataset identity mismatch: {name}')
    return load_json(directory / 'data_hashes.json'), load_json(directory / 'family_allocation.json')


def read_npz(path, directory, hashes):
    relative = path.relative_to(directory).as_posix()
    if sha256(path) != hashes[relative]:
        raise ValueError(f'Frozen data hash mismatch: {relative}')
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def read_split(directory, split, *, targets=True):
    if split not in ('train', 'validation', 'test'):
        raise ValueError('split must be train, validation, or test')
    directory = Path(directory)
    hashes, allocation = dataset_identity(directory)
    result = {key: [] for key in KEYS}
    metadata, labels = [], []
    for family, group in sorted(allocation.items()):
        if group != split:
            continue
        home = directory / 'families' / f'f{family}'
        meta = read_npz(home / 'metadata.npz', directory, hashes)
        keep = np.ones(len(meta['family']), dtype=bool) if split == 'test' else meta['fit_eligible'].copy()
        if targets:
            target = read_npz(directory / 'targets' / split / f'f{family}.npz', directory, hashes)
            keep &= target['reference_observed']
            labels.append(target['reference_rul_h'][keep])
        visible = read_npz(home / 'visible.npz', directory, hashes)
        if set(visible) != set(KEYS):
            raise ValueError('Unexpected predictor field in visible data')
        for key in KEYS:
            result[key].append(visible[key][keep])
        metadata.append(pd.DataFrame({key: value[keep] for key, value in meta.items()}))
    arrays = {key: np.concatenate(value) for key, value in result.items()}
    if targets:
        arrays['target'] = np.concatenate(labels)
    meta = pd.concat(metadata, ignore_index=True)
    if targets:
        meta['reference_rul_h'] = arrays['target']
    meta['endpoint_id'] = [f'{f}:{v}:{t}' for f, v, t in zip(meta.family, meta.variant, meta.endpoint_time_h)]
    if not meta.endpoint_id.is_unique:
        raise ValueError('Duplicated endpoint identity')
    if split != 'test' and (set(meta.layout) != {0, 3, 5, 6} or not meta.fit_eligible.all()):
        raise ValueError('Held-out layout entered a fitting split')
    valid = ~arrays['padding_mask']
    lengths = valid.sum(1)
    if not np.array_equal(valid, np.arange(31)[None, :] < lengths[:, None]):
        raise ValueError('Expected right padding and contiguous valid prefixes')
    if not np.all(arrays['relative_time'][np.arange(len(lengths)), lengths - 1] == 0):
        raise ValueError('Last observation must be the current endpoint')
    if not np.all(arrays['relative_time'][valid] >= -24):
        raise ValueError('History exceeds the frozen 24-hour support')
    if not np.isfinite(arrays['values']).all() or not np.isfinite(arrays['conditions']).all():
        raise ValueError('Nonfinite predictor data')
    return arrays, meta


def released_files(hashes, supplement=False):
    for relative, expected in hashes.items():
        core = relative.startswith('targets/') or relative.endswith(('/visible.npz', '/metadata.npz'))
        if relative in ('family_allocation.json', 'graph_manifest.json') or core:
            yield relative, expected
        elif supplement and relative not in AUDIT_ONLY:
            yield relative, expected


def verify(directory, supplement=False):
    directory = Path(directory)
    hashes, allocation = dataset_identity(directory)
    count, size = 1, (directory / 'data_hashes.json').stat().st_size
    for relative, expected in released_files(hashes, supplement):
        path = directory / relative
        if sha256(path) != expected:
            raise ValueError(f'Frozen data hash mismatch: {relative}')
        count += 1
        size += path.stat().st_size
    return dict(status='PASS', identity=config('dataset')['freeze_identity'],
                scope='processed+supplement' if supplement else 'processed', files=count, bytes=size,
                split_families={split: sum(group == split for group in allocation.values())
                                for split in ('train', 'validation', 'test')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/frozen_graph'))
    parser.add_argument('--include-supplement', action='store_true')
    args = parser.parse_args()
    import json
    print(json.dumps(verify(args.data, args.include_supplement), indent=2))


if __name__ == '__main__':
    main()
