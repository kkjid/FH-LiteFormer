"""Verify that archived computational stages remain distinct and functional."""
import hashlib
import json
from pathlib import Path

import torch

from fh_liteformer.io import config, configure_runtime
from fh_liteformer.models.system import LightweightS21
from history.s21a_graph import S21
from history.lightweight import LightweightS21 as HistoricalLightweight
from history.compact import export


def test_historical_source_identity():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'history/manifest.json').read_text(encoding='utf8'))
    assert len(manifest['stages']) == 3
    for stage in manifest['stages']:
        for file in stage['files']:
            assert hashlib.sha256((root / file['destination']).read_bytes()).hexdigest() == file['snapshot_sha256']


def test_historical_systems_and_deployment_counts():
    configure_runtime('cpu')
    recipe = config('training')
    reference = S21(recipe)
    historical = HistoricalLightweight(recipe).eval()
    current = LightweightS21(recipe).eval()
    current.load_state_dict(historical.state_dict(), strict=True)
    assert sum(p.numel() for p in reference.parameters()) == 350523
    assert sum(p.numel() for p in historical.parameters()) == 333951
    assert sum(p.numel() for p in export(historical).parameters()) == 170367
    assert sum(p.numel() for p in export(historical, rank=8).parameters()) == 130431
    for key, value in historical.state_dict().items():
        torch.testing.assert_close(current.state_dict()[key], value, rtol=0, atol=0)
