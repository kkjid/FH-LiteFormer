"""Typed relation attention and within-window GRU correction."""
import torch
from torch import nn
from torch.nn import functional as F
from .common import TypeEncoder, RelationParameters, ConditionFiLM, last

class RelationLayer(nn.Module):
    """E2 incoming-relation additive attention, new six-relation registry."""
    def __init__(self,c):
        super().__init__(); self.w=c['width']; self.h=c['heads']
        self.relations=nn.ModuleList([RelationParameters(self.w,self.h) for _ in range(6)])
        self.projection=nn.Linear(self.w,self.w); self.drop=nn.Dropout(c['dropout']); self.norm=nn.LayerNorm(self.w)
    def forward(self,nodes,adj):
        scores=[]; values=[]
        for p in self.relations:
            q=torch.einsum('bni,hid->bnhd',nodes,p.W_dst)
            k=torch.einsum('bni,hid->bnhd',nodes,p.W_src)
            v=torch.einsum('bni,hid->bnhd',nodes,p.W_val)
            ad,ass=p.attention_vector.chunk(2,-1)
            score=(q*ad).sum(-1)[:,:,None]+(k*ass).sum(-1)[:,None]
            scores.append(F.leaky_relu(score,.2)); values.append(v)
        score=torch.stack(scores,2).reshape(len(nodes),6,36,self.h)
        support=adj.permute(0,2,1,3).reshape(len(nodes),6,36)
        alpha=score.masked_fill(~support[:,:,:,None],-torch.inf).softmax(2).reshape(len(nodes),6,6,6,self.h)
        message=sum(torch.einsum('bnsh,bshd->bnhd',alpha[:,:,r],v) for r,v in enumerate(values)).reshape(len(nodes),6,self.w)
        return self.norm(nodes+self.drop(torch.relu(self.projection(message))))


class GraphEncoder(nn.Module):
    def __init__(self,c,film=False):
        super().__init__(); w=c['width']
        self.encoders=nn.ModuleList([TypeEncoder(2*f+6,w,c['dropout']) for f in (7,5,5)])
        self.roles=nn.Embedding(6,w)
        self.layers=nn.ModuleList([RelationLayer(c) for _ in range(c['graph_layers'])])
        self.films=nn.ModuleList([ConditionFiLM(5,w) for _ in range(3)]) if film else None
    def forward(self,b):
        valid=~b['padding']; values=b['nodes'][valid]; masks=b['masks'][valid]; cond=b['conditions'][valid]
        nodes=[]
        for j in range(6):
            kind=j//2; f=7 if j<2 else 5
            flag=values.new_full((len(values),1),float(j==0))
            raw=torch.cat((values[:,j,:f],masks[:,j,:f],cond,flag),-1)
            z=self.encoders[kind](raw)+self.roles.weight[j]
            if self.films is not None: z=self.films[kind](z,cond)
            nodes.append(z)
        z=torch.stack(nodes,1)
        adj=b['adjacency'][:,None].expand(-1,31,-1,-1,-1)[valid]
        for layer in self.layers: z=layer(z,adj)
        result=values.new_zeros((*b['padding'].shape,z.shape[-1])); result[valid]=z[:,0]
        return result


class Correction(nn.Module):
    """Graph-conditioned GRU reset at each 24h window; base-conditioned expert."""
    def __init__(self,c):
        super().__init__(); self.graph=GraphEncoder(c,film=True); w=c['expert_width']
        self.core=nn.GRU(c['width']+6,w,batch_first=True,num_layers=c['expert_layers'])
        self.head=nn.Sequential(nn.Linear(w+1,w),nn.ReLU(),nn.Dropout(c['dropout']),nn.Linear(w,1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
    def forward(self,b,base,diagnostics=False):
        message=self.graph(b)
        seq=torch.cat((message,b['conditions'],b['time'][:,:,None]),-1)
        state,_=self.core(seq); current=last(state,b)
        residual=self.head(torch.cat((current,base.detach()[:,None]),-1)).squeeze(-1)
        expert=F.softplus(base.detach()+residual)
        if not diagnostics: return expert
        idx=(b['length']-2).clamp_min(0)
        previous=state[torch.arange(len(state),device=state.device),idx]
        diag=torch.stack((current.norm(dim=-1)/4,(current-previous).norm(dim=-1)/4,last(message,b).norm(dim=-1)/4),-1)
        return expert,diag
