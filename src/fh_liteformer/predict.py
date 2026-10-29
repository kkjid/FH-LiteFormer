"""Target-free inference, including endpoints with censored Reference labels."""
import argparse
from pathlib import Path
import pandas as pd
from .data import read_split
from .data_math import transform
from .models import load_standalone
from .evaluate import predict
from .io import configure_runtime, config, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/frozen_graph'))
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='test')
    parser.add_argument('--seed', type=int, choices=[42, 43, 44], default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output', type=Path, default=Path('runs/predictions.csv'))
    args = parser.parse_args()
    device = configure_runtime(args.device)
    relative = f'checkpoints/C-r8__seed{args.seed}.pt'
    checkpoint = args.artifacts / relative
    if sha256(checkpoint) != config('artifacts')[relative]['sha256']:
        raise ValueError('Frozen checkpoint hash mismatch')
    model, payload = load_standalone(checkpoint, device)
    arrays, meta = read_split(args.data, args.split, targets=False)
    tensors = transform(arrays, payload['scaler'], device=device)
    prediction = predict(model, tensors, payload['scaler']['target_scale'])
    frame = pd.DataFrame(dict(endpoint_id=meta.endpoint_id, family=meta.family, seed=args.seed, prediction_h=prediction))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format='%.17g')
    print(f'Wrote {len(frame)} predictions without reading target files')


if __name__ == '__main__':
    main()
