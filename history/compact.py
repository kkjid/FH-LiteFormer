"""A/B lossless deployment and rank-limited C branch; original selected definitions."""
import copy
import itertools
import torch
from torch import nn
from torch.nn import functional as F
from fh_liteformer.topology import edges

def dependency_analysis():
    plans = []
    for mounts in itertools.product((0, 1), repeat=4):
        src, dst, rel = edges(mounts)
        needed = {0}; stages = []
        for layer in (2, 1, 0):
            legal = [(int(s), int(d), int(r)) for s, d, r in zip(src, dst, rel) if int(d) in needed]
            previous = needed | {s for s, d, r in legal}  # destination score and residual
            stages.append(dict(layer=layer, destinations=sorted(needed), edges=legal,
                               inputs=sorted(previous), relations=sorted({r for s,d,r in legal})))
            needed = previous
        plans.append(dict(mounts=list(mounts), stages=list(reversed(stages))))
    union = []
    for layer in range(3):
        stages = [p['stages'][layer] for p in plans]
        union.append(dict(layer=layer,
            destinations=sorted(set().union(*(set(s['destinations']) for s in stages))),
            relations=sorted(set().union(*(set(s['relations']) for s in stages))),
            inputs=sorted(set().union(*(set(s['inputs']) for s in stages)))))
    return dict(per_topology=plans, deployment_union=union,
                scope='Conservative parameter union of all 16 legal physical topologies; partial destinations evaluated directly')


SOURCE_NODES = ((0,1),(0,1),(4,5),(2,3),(4,5),(4,5))


class CompactLayer(nn.Module):
    def __init__(self, parent, layer, compact=True, prune=True, rank=None, bases=None):
        super().__init__()
        self.w, self.h = parent.w, parent.h
        self.compact, self.prune, self.rank, self.bases = compact, prune, rank, bases
        plan = dependency_analysis()['deployment_union'][layer]
        destinations = plan['destinations'] if prune else list(range(6))
        relations = plan['relations'] if prune else list(range(6))
        self.register_buffer('destinations', torch.tensor(destinations, dtype=torch.long))
        self.register_buffer('relation_ids', torch.tensor(relations, dtype=torch.long))
        self.register_buffer('source_nodes', torch.tensor([SOURCE_NODES[r] for r in relations], dtype=torch.long))
        pars = [parent.relations[r] for r in relations]
        if compact:
            ad, ass = zip(*(p.attention_vector.detach().chunk(2,-1) for p in pars))
            self.u_dst = nn.Parameter(torch.stack([torch.einsum('hid,hd->hi',p.W_dst.detach(),a) for p,a in zip(pars,ad)]))
            self.u_src = nn.Parameter(torch.stack([torch.einsum('hid,hd->hi',p.W_src.detach(),a) for p,a in zip(pars,ass)]))
        else:
            self.W_dst = nn.Parameter(torch.stack([p.W_dst.detach().clone() for p in pars]))
            self.W_src = nn.Parameter(torch.stack([p.W_src.detach().clone() for p in pars]))
            self.attention = nn.Parameter(torch.stack([p.attention_vector.detach().clone() for p in pars]))
        matrix = torch.stack([p.W_val.detach().permute(1,0,2).reshape(self.w,self.w) for p in pars])
        if rank:
            u,s,vh = torch.linalg.svd(matrix,full_matrices=False)
            self.left = nn.Parameter(u[:,:,:rank]*s[:,:rank,None].transpose(1,2))
            self.right = nn.Parameter(vh[:,:rank])
        elif bases:
            nr = len(pars); actual = min(bases,nr)
            u,s,vh = torch.linalg.svd(matrix.reshape(nr,-1),full_matrices=False)
            self.coefficients = nn.Parameter(u[:,:actual]*s[:actual])
            self.basis = nn.Parameter(vh[:actual].reshape(actual,self.w,self.w))
        else:
            self.value = nn.Parameter(matrix.clone())
        self.projection = copy.deepcopy(parent.projection)
        self.norm = copy.deepcopy(parent.norm)
        self.drop = copy.deepcopy(parent.drop)

    def forward(self, nodes, adj):
        dest = nodes[:,self.destinations]
        source = nodes[:,self.source_nodes]  # N,R,2,I: only legal source roles
        if self.compact:
            q = torch.einsum('ndi,rhi->ndrh',dest,self.u_dst)
            k = torch.einsum('nrsi,rhi->nrsh',source,self.u_src)
        else:
            qv = torch.einsum('bni,rhid->bnrhd',dest,self.W_dst)
            kv = torch.einsum('brsi,rhid->brshd',source,self.W_src)
            ad,ass = self.attention.chunk(2,-1)
            q = (qv*ad[None,None]).sum(-1)
            k = (kv*ass[None,:,None]).sum(-1)
        score = F.leaky_relu(q[:,:,:,None]+k[:,None],.2)
        support = adj[:,self.relation_ids][:,:,self.destinations]
        support = support.gather(3,self.source_nodes[None,:,None].expand(len(nodes),-1,len(self.destinations),-1)).permute(0,2,1,3)
        shape = score.shape
        alpha = score.flatten(2,3).masked_fill(~support.flatten(2,3)[:,:,:,None],-torch.inf).softmax(2).reshape(shape)
        if self.rank:
            z = torch.einsum('nrsi,rik->nrsk',source,self.left)
            v = torch.einsum('nrsk,rko->nrso',z,self.right)
        elif self.bases:
            # Shared value basis is applied once per physical node, then mixed per relation.
            shared = torch.einsum('bni,kio->bnko',nodes,self.basis)
            selected = shared[:,self.source_nodes]
            v = torch.einsum('brsko,rk->brso',selected,self.coefficients)
        else:
            v = torch.einsum('nrsi,rio->nrso',source,self.value)
        v = v.reshape(len(nodes),len(self.relation_ids),2,self.h,self.w//self.h)
        message = torch.einsum('bdrsh,brsho->bdho',alpha,v).reshape(len(nodes),len(self.destinations),self.w)
        updated = self.norm(dest+self.drop(torch.relu(self.projection(message))))
        # Inactive nodes are outside the proven backward dependency cone.
        result = nodes.new_zeros(nodes.shape)
        result[:,self.destinations] = updated
        return result


def export(parent, compact=True, prune=True, rank=None, bases=None):
    model = copy.deepcopy(parent)
    model.expert.graph.layers = nn.ModuleList([
        CompactLayer(p,i,compact,prune,rank,bases).to(device=p.projection.weight.device,dtype=p.projection.weight.dtype)
        for i,p in enumerate(parent.expert.graph.layers)])
    return model
