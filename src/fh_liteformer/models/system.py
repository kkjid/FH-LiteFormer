"""Causal advantage gate over frozen temporal and graph-correction parents."""
import torch
from torch import nn
from .common import last
from .temporal import LightweightTemporal
from .graph import Correction

def gate_features(b,base,expert,diag):
    index=b['length']-1; rows=torch.arange(len(index),device=index.device)
    dt=(b['time'][rows,index]-b['time'][rows,(index-1).clamp_min(0)])[:,None]
    missing=last(b['masks'],b).mean((1,2))[:,None]
    sink=last(b['nodes'],b)[:,4:,:2].reshape(len(index),4)
    return torch.cat((base[:,None],(expert-base)[:,None],(expert-base).abs()[:,None],diag,dt,last(b['conditions'],b),missing,sink),-1).detach()


class Gate(nn.Module):
    def __init__(self,c):
        super().__init__(); self.network=nn.Sequential(nn.Linear(17,c['gate_width']),nn.SiLU(),nn.Linear(c['gate_width'],1))
        nn.init.zeros_(self.network[-1].weight); nn.init.zeros_(self.network[-1].bias)
        self.register_buffer('mean',torch.zeros(17)); self.register_buffer('scale',torch.ones(17))
    def logits(self,x): return self.network((x-self.mean)/self.scale).squeeze(-1)
    def forward(self,x): return self.logits(x).sigmoid()


class LightweightS21(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.base = LightweightTemporal(c)
        self.expert = Correction(c)
        self.gate = Gate(c)

    def forward(self, b, components=False):
        base = self.base(b)
        expert, diag = self.expert(b, base, True)
        weight = self.gate(gate_features(b, base, expert, diag))
        learned = base + weight * (expert - base)
        return dict(base=base, correction=expert, fixed_50_50_blend=(base + expert) / 2,
                    learned_gate=learned, gate_weight=weight) if components else learned


def build(name, config):
    classes = {"LightweightTransformer": LightweightTemporal, "Correction": Correction, "S21A-Lightweight-Graph": LightweightS21}
    try:
        return classes[name](config)
    except KeyError as exc:
        raise ValueError(f"Unknown training component: {name}") from exc
