import itertools
from pathlib import Path
import numpy as np
import pytest
import torch
from fh_liteformer.data import read_split, KEYS
from fh_liteformer.data_math import transform, batch, fit_scaler
from fh_liteformer.evaluate import family_metrics
from fh_liteformer.fetch import safe_destination
from fh_liteformer.io import config, configure_runtime
from fh_liteformer.models import load_standalone
from fh_liteformer.models.system import LightweightS21
from fh_liteformer.models.compact import export, dependency_analysis
from fh_liteformer.topology import edges


@pytest.fixture(scope='module', autouse=True)
def runtime():
    configure_runtime('cpu')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_lossless_dependency_and_full_rank_svd(dtype):
    torch.manual_seed(17)
    model = LightweightS21(config('training')).to(dtype).eval()
    count, steps = 16, 31
    adjacency = torch.zeros(count, 6, 6, 6, dtype=torch.bool)
    for index, mounts in enumerate(itertools.product((0, 1), repeat=4)):
        source, destination, relation = edges(mounts)
        adjacency[index, relation, destination, source] = True
    tensors = dict(seq=torch.randn(count, steps, 91, dtype=dtype),
                   nodes=torch.randn(count, steps, 6, 7, dtype=dtype),
                   masks=torch.zeros(count, steps, 6, 7, dtype=dtype),
                   conditions=torch.randn(count, steps, 5, dtype=dtype),
                   time=torch.linspace(-1, 0, steps, dtype=dtype)[None].expand(count, -1),
                   padding=torch.zeros(count, steps, dtype=torch.bool),
                   length=torch.full((count,), steps, dtype=torch.long), adjacency=adjacency)
    compact = export(model).eval()
    full_rank = export(model, rank=64).eval()
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    with torch.inference_mode():
        expected = model(tensors, components=True)
        for candidate in (compact, full_rank):
            actual = candidate(tensors, components=True)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=tolerance, rtol=tolerance)


def test_conservative_dependency_union():
    plans = dependency_analysis()['deployment_union']
    assert [len(plan['destinations']) for plan in plans] == [6, 4, 1]
    assert [len(plan['relations']) for plan in plans] == [6, 5, 2]


def test_family_macro_is_not_endpoint_average():
    actual = family_metrics(np.array([0., 0., 0.]), np.array([0., 0., 12.]), np.array([1, 1, 2]))
    assert actual == {'mae_h': 6., 'rmse_h': 6.}


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'C:/drive', 'a\\escape'])
def test_archive_path_rejects_escape(tmp_path, name):
    with pytest.raises(ValueError):
        safe_destination(tmp_path, name)


def test_real_validation_target_separation_and_checkpoint():
    arrays, metadata = read_split(Path('data/frozen_graph'), 'validation')
    assert set(arrays) == set(KEYS) | {'target'}
    assert len(metadata) == 7846 and metadata.family.nunique() == 32
    assert set(metadata.layout) == {0, 3, 5, 6}
    arrays = {key: value[:8] for key, value in arrays.items()}
    tensors = transform(arrays, config('preprocessing'), device='cpu')
    model, payload = load_standalone('artifacts/checkpoints/C-r8__seed42.pt', 'cpu')
    assert sum(parameter.numel() for parameter in model.parameters()) == 130431
    assert 'target' not in batch(tensors, slice(None))
    with torch.inference_mode():
        first = model(batch(tensors, slice(None)))
        tensors['target'].fill_(float('nan'))
        second = model(batch(tensors, slice(None)))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.isfinite(second).all()
    assert payload['scaler'] == config('preprocessing')


def test_train_normalization_and_family_partition():
    arrays, metadata = read_split(Path('data/frozen_graph'), 'train')
    scaler = fit_scaler(arrays, metadata)
    expected = config('preprocessing')
    assert len(metadata) == 43130 and metadata.family.nunique() == 176
    assert scaler['training_families'] == expected['training_families']
    for name in ('node_mean', 'node_scale', 'condition_mean', 'condition_scale', 'target_scale'):
        np.testing.assert_allclose(scaler[name], expected[name], rtol=1e-10, atol=1e-10)
    partition = config('crossfit_partition')
    assert set(map(int, partition)) == set(scaler['training_families'])
    assert sorted(list(partition.values()).count(fold) for fold in range(5)) == [35, 35, 35, 35, 36]
