"""Unchanged Train-only normalization and batch construction."""
import numpy as np
import torch
from .topology import edges

def family_weights(meta):
    _, inv, counts = np.unique(meta.family.to_numpy(), return_inverse=True, return_counts=True)
    w = 1/counts[inv]; return w/w.sum()


def fit_scaler(a, meta):
    # Equal family moments, shared across nodes of the same physical type.
    mask = (~a['padding_mask'])[:,:,None,None] & (~a['feature_missing_mask'][:,None])
    wf = family_weights(meta)
    per = mask.sum(1).clip(1)
    avg = (a['values']*mask).sum(1)/per
    second = (a['values'].astype(np.float64)**2*mask).sum(1)/per
    means = np.zeros((6,7)); scales = np.ones((6,7))
    for nodes in ((0,1),(2,3),(4,5)):
        mu = np.einsum('n,nkf->f',wf,avg[:,nodes])/len(nodes)
        sq = np.einsum('n,nkf->f',wf,second[:,nodes])/len(nodes)
        sd = np.sqrt(np.maximum(sq-mu**2,0)); sd[sd<1e-6] = 1
        means[list(nodes)] = mu; scales[list(nodes)] = sd
    valid = (~a['padding_mask']).astype(float); lens = valid.sum(1)
    cm = np.einsum('n,nf->f',wf,(a['conditions']*valid[:,:,None]).sum(1)/lens[:,None])
    csq = np.einsum('n,nf->f',wf,(a['conditions'].astype(float)**2*valid[:,:,None]).sum(1)/lens[:,None])
    cs = np.sqrt(np.maximum(csq-cm**2,0)); cs[cs<1e-6] = 1
    return dict(node_mean=means.tolist(),node_scale=scales.tolist(),condition_mean=cm.tolist(),
        condition_scale=cs.tolist(),target_scale=float(np.dot(wf,a['target'])),
        training_families=sorted(map(int,meta.family.unique())), policy='equal-family/type valid-timestep Train only')


def transform(a, scaler, device='cuda'):
    valid = ~a['padding_mask']; lens = valid.sum(1)
    mask = a['feature_missing_mask']
    nodes = (a['values']-np.asarray(scaler['node_mean']))/np.asarray(scaler['node_scale'])
    nodes = np.where(valid[:,:,None,None] & ~mask[:,None], nodes, 0).astype('float32')
    conditions = ((a['conditions']-np.asarray(scaler['condition_mean']))/np.asarray(scaler['condition_scale'])).astype('float32')
    conditions[~valid] = 0
    masks = np.broadcast_to(mask[:,None],nodes.shape).astype('float32').copy(); masks[~valid] = 0
    time = (a['relative_time']/24).astype('float32'); time[~valid] = 0
    seq = np.concatenate((nodes.reshape(len(nodes),31,42), masks.reshape(len(nodes),31,42),conditions,time[:,:,None],valid[:,:,None]),-1).astype('float32')
    adjacency = np.zeros((len(nodes),6,6,6), dtype=bool)
    for mounts in np.unique(a['mounts'],axis=0):
        ids = np.flatnonzero((a['mounts']==mounts).all(1)); s,d,r = edges(mounts)
        adjacency[ids[:,None],r[None],d[None],s[None]] = True
    result = dict(nodes=nodes,conditions=conditions,time=time,padding=a['padding_mask'],
                  masks=masks,seq=seq,adjacency=adjacency,length=lens.astype('int64'))
    if 'target' in a: result['target'] = (a['target']/scaler['target_scale']).astype('float32')
    return {k:torch.as_tensor(v,device=device) for k,v in result.items()}


def batch(t, ids):
    # Labels never enter the model dictionary, even as an unused extra key.
    return {k:v[ids] for k,v in t.items() if k != 'target'}


def subset(a, m, families):
    ids = np.flatnonzero(m.family.isin(families).to_numpy())
    return {k:v[ids] for k,v in a.items()}, m.iloc[ids].reset_index(drop=True), ids
