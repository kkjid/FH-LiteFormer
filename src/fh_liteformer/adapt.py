"""Apply the accepted six-epoch rank-8 expert adaptation to a same-seed frozen parent."""
import argparse
import copy
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import read_split
from .data_math import transform, batch, family_weights
from .models.system import LightweightS21
from .models.compact import export
from .models.deployment import optimize
from .evaluate import predict, family_metrics
from .io import config, configure_runtime, save_json, sha256


def selected_rows(arrays, meta):
    indices = np.concatenate([np.flatnonzero(meta.family.to_numpy() == family)[:4]
                              for family in sorted(meta.family.unique())[:2]])
    return {key: value[indices] for key, value in arrays.items()}, meta.iloc[indices].reset_index(drop=True)


def state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fit(seed, parent_path, training, train_meta, validation, val_meta, recipe, output, device, smoke):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    payload = torch.load(parent_path, map_location='cpu', weights_only=True)
    if payload['seed'] != seed:
        raise ValueError('Parent and adaptation seeds differ')
    parent = LightweightS21(payload['config'])
    parent.load_state_dict(payload['state'], strict=True)
    model = export(parent, rank=8).to(device).eval()
    for parameter in model.base.parameters():
        parameter.requires_grad_(False)
    for parameter in model.gate.parameters():
        parameter.requires_grad_(False)
    inherited = {key: value.clone() for key, value in state(model).items()
                 if key.startswith(('base.', 'gate.'))}
    scaler = payload['scaler']
    tr, va = transform(training, scaler, device=device), transform(validation, scaler, device=device)
    # This cache is the unchanged frozen base prediction used by the accepted adaptation.
    with torch.inference_mode():
        pieces = [model.base(batch(tr, slice(start, start + recipe['physical_batch_size'])))
                  for start in range(0, len(train_meta), recipe['physical_batch_size'])]
        cached_base = torch.cat(pieces)
    cached_base = cached_base.clone()
    weights = family_weights(train_meta)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
    truth = val_meta.reference_rul_h.to_numpy(float)

    def validation_mae():
        prediction = predict(model, va, scaler['target_scale'], recipe['physical_batch_size'])
        return family_metrics(truth, prediction, val_meta.family.to_numpy())['mae_h']

    best = validation_mae()
    best_epoch = 0
    best_state = state(model)
    history = [dict(epoch=0, validation_mae_h=best, loss=None)]
    for epoch in range(1, recipe['epochs'] + 1):
        model.expert.train()
        model.base.eval()
        model.gate.eval()
        chosen = rng.choice(len(train_meta), len(train_meta), replace=True, p=weights)
        losses = []
        for first in range(0, len(train_meta), recipe['effective_batch_size']):
            indices = torch.tensor(chosen[first:first + recipe['effective_batch_size']], device=device)
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(indices), recipe['physical_batch_size']):
                selected = indices[start:start + recipe['physical_batch_size']]
                expert, diagnostics = model.expert(batch(tr, selected), cached_base[selected], True)
                loss = nn.functional.smooth_l1_loss(expert, tr['target'][selected], beta=recipe['smooth_l1_beta'])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite adaptation loss')
                (loss * len(selected) / len(indices)).backward()
                losses.append(float(loss.detach()))
            nn.utils.clip_grad_norm_(model.expert.parameters(), recipe['gradient_clip_norm'])
            optimizer.step()
        model.eval()
        score = validation_mae()
        if score < best:
            best, best_epoch, best_state = score, epoch, state(model)
        history.append(dict(epoch=epoch, validation_mae_h=score, loss=float(np.mean(losses))))
        print(f'C-r8 seed={seed} epoch={epoch} validation_mae_h={score:.9f} best_epoch={best_epoch}', flush=True)
    model.load_state_dict(best_state)
    for key, value in state(model).items():
        if key in inherited and not torch.equal(value, inherited[key]):
            raise RuntimeError('Adaptation changed the frozen temporal base or gate')
    deployed = optimize(model)
    path = output / f'C-r8__seed{seed}.pt'
    torch.save(dict(state=state(deployed), spec={'name': 'C-r8', 'route': 'C', 'rank': 8}, seed=seed,
                    config=payload['config'], scaler=scaler, optimized=True, diagnostic_smoke_only=smoke), path)
    record = dict(seed=seed, best_epoch=best_epoch, history=history, frozen_base_and_gate_unchanged=True,
                  parent_sha256=sha256(parent_path), checkpoint_sha256=sha256(path),
                  mode='DIAGNOSTIC_SMOKE_ONLY' if smoke else 'FULL_ADAPTATION', test_accessed=False)
    save_json(path.with_suffix('.json'), record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/frozen_graph'))
    parser.add_argument('--parents', type=Path, default=Path('artifacts/training'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', choices=[42, 43, 44], default=[42, 43, 44])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    device = configure_runtime(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    recipe = copy.deepcopy(config('adaptation'))
    training, train_meta = read_split(args.data, 'train')
    validation, val_meta = read_split(args.data, 'validation')
    if args.smoke:
        training, train_meta = selected_rows(training, train_meta)
        validation, val_meta = selected_rows(validation, val_meta)
        recipe.update(epochs=1, effective_batch_size=8, physical_batch_size=4)
    records = []
    for seed in args.seeds:
        parent = args.parents / 'checkpoints' / f'S21A-Lightweight-Graph__seed{seed}.pt'
        records.append(fit(seed, parent, training, train_meta, validation, val_meta, recipe,
                           args.output, device, args.smoke))
    save_json(args.output / 'run.json', dict(status='PASS', mode='DIAGNOSTIC_SMOKE_ONLY' if args.smoke else 'FULL_ADAPTATION',
                                           test_accessed=False, recipe=recipe, results=records))


if __name__ == '__main__':
    main()
