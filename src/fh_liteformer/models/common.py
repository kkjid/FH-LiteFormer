"""Accepted typed encoders and bounded condition modulation."""
import torch
from torch import nn

class TypeEncoder(nn.Sequential):
    """Port of the graph_v2 biased Linear/ReLU/dropout per-type encoder."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__(nn.Linear(input_dim, hidden_dim, bias=True), nn.ReLU(), nn.Dropout(dropout))


class RelationParameters(nn.Module):
    """Port of independent W_src/W_dst/W_val/a per directed typed relation."""
    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        width = hidden_dim // heads
        self.W_src = nn.Parameter(torch.empty(heads, hidden_dim, width))
        self.W_dst = nn.Parameter(torch.empty(heads, hidden_dim, width))
        self.W_val = nn.Parameter(torch.empty(heads, hidden_dim, width))
        self.attention_vector = nn.Parameter(torch.empty(heads, 2 * width))
        for parameter in self.parameters():
            nn.init.xavier_uniform_(parameter, gain=1.0)


class ConditionFiLM(nn.Module):
    """Accepted (1 + 0.1*tanh(gamma))*z + beta for observable conditions only."""

    def __init__(self, condition_dim: int, width: int):
        super().__init__()
        self.condition_dim = condition_dim
        self.core = nn.Sequential(nn.Linear(condition_dim, width), nn.ReLU(),
                                  nn.Linear(width, 2 * width))

    def forward(self, encoded, condition):
        if condition.shape[:-1] != encoded.shape[:-1] or condition.shape[-1] != self.condition_dim:
            raise ValueError("FiLM condition shape differs from encoded sequence")
        gamma, beta = self.core(condition).chunk(2, dim=-1)
        return (1.0 + 0.1 * torch.tanh(gamma)) * encoded + beta


def last(x,b): return x[torch.arange(len(x),device=x.device),b['length']-1]
