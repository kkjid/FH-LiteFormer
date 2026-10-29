"""Function-preserving inference export; original selected definitions."""
import copy
import torch
from torch import nn

class BatchedGraphEncoder(nn.Module):
    def __init__(self,graph):
        super().__init__();self.encoders=graph.encoders;self.roles=graph.roles
        self.layers=graph.layers;self.films=graph.films
        self.register_buffer('target_flags',graph.roles.weight.new_tensor([1,0,0,0,0,0]).reshape(1,6,1))
    def forward(self,b):
        if self.training:raise RuntimeError('Batched encoder export is inference-only')
        valid=~b['padding'];values=b['nodes'][valid];masks=b['masks'][valid];cond=b['conditions'][valid]
        parts=[]
        for kind,f in enumerate((7,5,5)):
            start=2*kind;end=start+2
            raw=torch.cat((values[:,start:end,:f],masks[:,start:end,:f],
                cond[:,None].expand(-1,2,-1),self.target_flags[:,start:end].expand(len(values),-1,-1)),-1)
            z=self.encoders[kind](raw)+self.roles.weight[start:end][None]
            if self.films is not None:
                gamma,beta=self.films[kind].core(cond).chunk(2,-1)
                z=(1+.1*torch.tanh(gamma))[:,None]*z+beta[:,None]
            parts.append(z)
        z=torch.cat(parts,1)
        adj=b['adjacency'][:,None].expand(-1,31,-1,-1,-1)[valid]
        for layer in self.layers:z=layer(z,adj)
        result=values.new_zeros((*b['padding'].shape,z.shape[-1]));result[valid]=z[:,0]
        return result


class StaticSealedHead(nn.Module):
    def __init__(self,head):
        super().__init__()
        # Retain serialized master state for full inventory transparency; never read it.
        self.weight=head.weight;self.bias=head.bias
        for key in ('exponent','negative','sealed'):self.register_buffer(key,getattr(head,key).detach().clone())
    def forward(self,x):
        scaled=torch.ldexp(x,self.exponent)
        return torch.where(self.negative,-scaled,scaled).sum(-1,keepdim=True)+self.bias


def optimize(model):
    result=copy.deepcopy(model).eval()
    graph=result.expert.graph
    if hasattr(graph,'inner'):graph.inner=BatchedGraphEncoder(graph.inner).eval()
    else:result.expert.graph=BatchedGraphEncoder(graph).eval()
    if hasattr(result.base.head,'sealed'):
        assert bool(result.base.head.sealed)
        result.base.head=StaticSealedHead(result.base.head).eval()
    result.expert.core.flatten_parameters()
    return result
