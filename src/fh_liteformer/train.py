"""Train the temporal parent, graph-correction parent and five-fold advantage gate."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from . import training_core as core
from .data import read_split
from .data_math import fit_scaler
from .io import config, configure_runtime, save_json, sha256


def source_hashes():
    home = Path(__file__).parent
    return {path.relative_to(home).as_posix(): sha256(path)
            for path in sorted(home.rglob('*')) if path.suffix in ('.py', '.json')}


def smoke_subset(arrays, meta, partition, training):
    families = []
    if training:
        for fold in range(5):
            families.extend(sorted(int(f) for f, value in partition.items() if value == fold)[:2])
    else:
        families = sorted(map(int, meta.family.unique()))[:2]
    indices = np.concatenate([np.flatnonzero(meta.family.to_numpy() == family)[:2] for family in families])
    return {key: value[indices] for key, value in arrays.items()}, meta.iloc[indices].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/frozen_graph'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 43, 44], default=[42, 43, 44])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--smoke', action='store_true', help='One-epoch, small Train/Validation diagnostic; not a reported result')
    args = parser.parse_args()
    device = configure_runtime(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    recipe = copy.deepcopy(config('training'))
    partition = config('crossfit_partition')
    training, train_meta = read_split(args.data, 'train')
    validation, val_meta = read_split(args.data, 'validation')
    if args.smoke:
        training, train_meta = smoke_subset(training, train_meta, partition, True)
        validation, val_meta = smoke_subset(validation, val_meta, partition, False)
        recipe.update(maximum_epochs=1, effective_batch_size=8, physical_batch_size=4,
                      checkpoint_selection='diagnostic earliest Validation minimum after one epoch')
    partition = {str(int(f)): partition[str(int(f))] for f in train_meta.family.unique()}
    scaler = fit_scaler(training, train_meta)
    save_json(args.output / 'preprocessing.json', scaler)
    save_json(args.output / 'crossfit_partition.json', partition)
    status = dict(status='RUNNING', mode='DIAGNOSTIC_SMOKE_ONLY' if args.smoke else 'FULL_TRAINING',
                  seeds=args.seeds, train_endpoints=len(train_meta), validation_endpoints=len(val_meta),
                  test_accessed=False, recipe=recipe)
    save_json(args.output / 'run.json', status)
    core.configure(args.output, device, recipe, source_hashes())
    for seed in args.seeds:
        core.train_standard('LightweightTransformer', seed, training, train_meta, validation, val_meta, recipe, scaler)
        core.train_s21(seed, training, train_meta, validation, val_meta, recipe, scaler)
        # Seal the same quantized values already used by the QAT forward pass.
        path = core.ckpt('S21A-Lightweight-Graph', seed)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        model = core.build('S21A-Lightweight-Graph', recipe)
        model.load_state_dict(payload['state'])
        model.base.head.seal()
        payload['state'] = core.state(model)
        core.atomic_torch(path, payload)
    status['status'] = 'PASS'
    status['stages_executed'] = ['temporal_parent', 'graph_correction_parent', 'five_training_family_crossfits', 'advantage_gate']
    save_json(args.output / 'run.json', status)
    print(status, flush=True)


if __name__ == '__main__':
    main()
