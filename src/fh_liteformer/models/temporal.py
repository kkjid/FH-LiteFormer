"""Frozen lightweight temporal encoder and signed-power readout."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .common import last

class Temporal(nn.Module):
    def __init__(self,c,input_dim=91):
        super().__init__(); w=c['width']
        self.input=nn.Linear(input_dim,w)
        layer=nn.TransformerEncoderLayer(w,c['heads'],c['feedforward'],c['dropout'],
            activation='relu',batch_first=True,norm_first=False)
        self.core=nn.TransformerEncoder(layer,c['temporal_layers'],enable_nested_tensor=False)
        self.head=nn.Linear(w,1)
    def forward(self,b,snapshots=None):
        x=b['seq'] if snapshots is None else torch.cat((snapshots,b['time'][:,:,None]),-1)
        z=self.core(self.input(x),src_key_padding_mask=b['padding'])
        return F.softplus(self.head(last(z,b)).squeeze(-1))


class DyT(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.ones(width))
        self.beta = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class EdgeKAN(nn.Module):
    """phi_ji(x_i)=c_ji1*u_i+c_ji2*(2*u_i**2-1), u=tanh(x)."""
    def __init__(self, inputs, outputs):
        super().__init__()
        self.coefficients = nn.Parameter(torch.empty(outputs, 2 * inputs))
        self.bias = nn.Parameter(torch.zeros(outputs))
        nn.init.normal_(self.coefficients, std=1 / math.sqrt(2 * inputs))

    def forward(self, x):
        u = torch.tanh(x)
        basis = torch.cat((u, 2 * u.square() - 1), dim=-1)
        return F.linear(basis, self.coefficients, self.bias)


class KANFirst(nn.Module):
    """Inherited Transformer inserts ReLU here; bypass it without changing residuals."""
    def __init__(self, width, bottleneck):
        super().__init__()
        self.edge = EdgeKAN(width, bottleneck)

    def forward(self, x):
        return self.edge(x)


def identity_activation(x):
    return x


class ShiftAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        exponent = torch.round(torch.log2(weight.abs().clamp_min(2 ** -8))).clamp(-8, 0).to(torch.int32)
        negative = weight < 0
        q = torch.where(negative, -torch.ldexp(torch.ones_like(weight), exponent),
                        torch.ldexp(torch.ones_like(weight), exponent))
        ctx.save_for_backward(x, q)
        scaled = torch.ldexp(x, exponent)
        return torch.where(negative, -scaled, scaled).sum(-1, keepdim=True) + bias

    @staticmethod
    def backward(ctx, grad):
        x, q = ctx.saved_tensors
        # Identity STE for master weights; multiplication is allowed in training.
        return grad * q, (grad * x).reshape(-1, x.shape[-1]).sum(0), grad.sum().reshape(1)


class ShiftAddHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))
        self.bias = nn.Parameter(torch.empty(1))
        nn.init.uniform_(self.weight, -1 / math.sqrt(width), 1 / math.sqrt(width))
        nn.init.uniform_(self.bias, -1 / math.sqrt(width), 1 / math.sqrt(width))
        self.register_buffer('exponent', torch.zeros(width, dtype=torch.int32))
        self.register_buffer('negative', torch.zeros(width, dtype=torch.bool))
        self.register_buffer('sealed', torch.tensor(False))

    @torch.no_grad()
    def seal(self):
        self.exponent.copy_(torch.round(torch.log2(self.weight.abs().clamp_min(2 ** -8))).clamp(-8, 0).to(torch.int32))
        self.negative.copy_(self.weight < 0)
        self.sealed.fill_(True)

    def forward(self, x):
        if self.sealed:
            value = torch.ldexp(x, self.exponent)
            return torch.where(self.negative, -value, value).sum(-1, keepdim=True) + self.bias
        return ShiftAddFunction.apply(x, self.weight, self.bias)


class LightweightTemporal(Temporal):
    def __init__(self, c, *, dyt=True, kan=True, light_head=True):
        # Reuse attention, post-norm residuals, padding and last-valid readout.
        super().__init__(c)
        for layer in self.core.layers:
            if dyt:
                layer.norm1 = DyT(c['width'])
                layer.norm2 = DyT(c['width'])
                layer.activation_relu_or_gelu = 0
            if kan:
                layer.linear1 = KANFirst(c['width'], c['lightweight']['kan_bottleneck'])
                layer.linear2 = EdgeKAN(c['lightweight']['kan_bottleneck'], c['width'])
                layer.activation = identity_activation
                layer.activation_relu_or_gelu = 0  # Disable fused standard FFN fast path.
        if light_head:
            self.head = ShiftAddHead(c['width'])
