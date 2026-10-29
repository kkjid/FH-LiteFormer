"""Replay all three frozen C-r8 checkpoints on the common Reference-valid Test support."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import read_split
from .data_math import transform, batch
from .models import load_standalone
from .io import config, configure_runtime, save_json, sha256


def family_metrics(truth, prediction, family):
    frame = pd.DataFrame(dict(family=family, ae=np.abs(truth - prediction), se=(truth - prediction) ** 2))
    values = frame.groupby('family').agg(mae=('ae', 'mean'), mse=('se', 'mean'))
    return dict(mae_h=float(values.mae.mean()), rmse_h=float(np.sqrt(values.mse).mean()))


@torch.inference_mode()
def predict(model, tensors, target_scale, batch_size=128):
    return np.concatenate([model(batch(tensors, slice(start, start + batch_size))).cpu().numpy().astype(float)
                           for start in range(0, len(tensors['seq']), batch_size)]) * target_scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/frozen_graph'))
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--output', type=Path, default=Path('runs/frozen_evaluation.json'))
    args = parser.parse_args()
    device = configure_runtime(args.device)
    expected = config('results')
    artifact_hashes = config('artifacts')
    arrays, meta = read_split(args.data, 'test')
    if len(meta) != 23420 or meta.family.nunique() != 48:
        raise ValueError('Test support differs from the accepted experiment')
    tensors = transform(arrays, config('preprocessing'), device=device)
    del arrays
    rows = []
    for seed in (42, 43, 44):
        relative = f'checkpoints/C-r8__seed{seed}.pt'
        path = args.artifacts / relative
        if sha256(path) != artifact_hashes[relative]['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {relative}')
        model, payload = load_standalone(path, device)
        if payload['scaler'] != config('preprocessing'):
            raise ValueError('Checkpoint preprocessing differs from the frozen statistics')
        if sum(parameter.numel() for parameter in model.parameters()) != 130431:
            raise ValueError('Expected the complete 130431-parameter C-r8 system')
        prediction = predict(model, tensors, payload['scaler']['target_scale'])
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite model output')
        reference_relative = f'reference_predictions/C-r8__seed{seed}.parquet'
        reference_path = args.artifacts / reference_relative
        if sha256(reference_path) != artifact_hashes[reference_relative]['sha256']:
            raise ValueError('Reference-prediction hash mismatch')
        reference = pd.read_parquet(reference_path)
        if reference.endpoint_id.tolist() != meta.endpoint_id.tolist():
            raise ValueError('Reference endpoint order mismatch')
        np.testing.assert_array_equal(reference.reference_rul_h.to_numpy(), meta.reference_rul_h.to_numpy())
        max_error = float(np.max(np.abs(prediction - reference.prediction_h.to_numpy())))
        if max_error > expected['checkpoint_prediction_atol_h']:
            raise ValueError(f'Seed {seed}: checkpoint predictions differ by {max_error} h')
        result = family_metrics(meta.reference_rul_h.to_numpy(float), prediction, meta.family.to_numpy())
        original = family_metrics(reference.reference_rul_h.to_numpy(float), reference.prediction_h.to_numpy(float), reference.family.to_numpy())
        for key in ('mae_h', 'rmse_h'):
            if abs(original[key] - expected['seeds'][str(seed)][key]) > 1e-7:
                raise ValueError('Stored metric arithmetic differs from accepted records')
        rows.append(dict(seed=seed, **result, max_prediction_error_h=max_error))
        print(rows[-1], flush=True)
        del model
    mean = {key: float(np.mean([row[key] for row in rows])) for key in ('mae_h', 'rmse_h')}
    for key, value in mean.items():
        if abs(value - expected['mean'][key]) > expected['checkpoint_prediction_atol_h']:
            raise ValueError(f'Reconstructed mean {key} differs from the accepted result')
    result = dict(status='PASS', validation='frozen_checkpoint_replay', device=device,
                  torch_version=torch.__version__, endpoints=23420, families=48,
                  seed_results=rows, mean=mean, metric='equal-family MAE; mean of within-family RMSE; then equal seed mean')
    save_json(args.output, result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
