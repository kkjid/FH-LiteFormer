"""Ported parent, five-family-cross-fit and advantage-gate training kernels."""
import math
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from .data_math import transform, subset, batch, family_weights, fit_scaler
from .models.system import build, Gate, gate_features, LightweightS21 as S21
from .io import load_json as load, save_json as save, sha256 as sha

DEVICE = 'cpu'
OUT = None
EVIDENCE = None
_CONFIG = None
_SOURCE_HASHES = None


def configure(output, device, config, source_hashes):
    global OUT, EVIDENCE, DEVICE, _CONFIG, _SOURCE_HASHES
    OUT, EVIDENCE, DEVICE, _CONFIG, _SOURCE_HASHES = output, output, device, config, source_hashes


def config():
    return _CONFIG


def source_hashes():
    return _SOURCE_HASHES


def compatible(payload, name):
    if payload['source_sha256'] != source_hashes():
        raise ValueError('Training continuation requires identical public source/config files')
    if payload.get('config', config()) != config():
        raise ValueError('Training configuration mismatch')
    return True


def ckpt(name, seed):
    return OUT / 'checkpoints' / f'{name}__seed{seed}.pt'


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) if DEVICE.startswith('cuda') else None


def atomic_torch(path,z):
    path=type(OUT)(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.writing'); torch.save(z,temp); temp.replace(path)


def macro(y,p,meta):
    d=pd.DataFrame({'family':meta.family,'error':np.abs(y-p)})
    return float(d.groupby('family').error.mean().mean())


def state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


@torch.no_grad()
def predict(model,t,c,base=None,diag=False):
    model.eval(); out=[]; extras=[]
    for start in range(0,len(t['seq']),c['physical_batch_size']):
        ids=slice(start,start+c['physical_batch_size']); b=batch(t,ids)
        value=model(b) if base is None else model(b,base[ids],diagnostics=diag)
        if diag: p,d=value; out.append(p); extras.append(d)
        else: out.append(value)
    return (torch.cat(out),torch.cat(extras)) if diag else torch.cat(out)


def fit(model,tr,va,tm,vm,scaler,c,seed,cell,*,fixed_epochs=None,base_train=None,base_val=None):
    model=model.to(DEVICE); seed_all(seed); n=len(tm); w=family_weights(tm); rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    best=math.inf; best_epoch=0; best_state=None; history=[]; start_epoch=1; spent=0
    recovery=OUT/'training_state'/f'{cell}.pt'
    if recovery.exists():
        z=torch.load(recovery,map_location='cpu',weights_only=True)
        assert z['seed']==seed and compatible(z,cell.split('__')[0]) and z['training_families']==sorted(map(int,tm.family.unique()))
        model.load_state_dict(z['model']); opt.load_state_dict(z['optimizer']); rng.bit_generator.state=z['numpy_rng']
        torch.set_rng_state(z['torch_rng']); torch.cuda.set_rng_state_all(z['cuda_rng']) if DEVICE.startswith('cuda') else None
        history=z['history']; best=z['best']; best_epoch=z['best_epoch']; best_state=z['best_state']; start_epoch=len(history)+1; spent=z['seconds']
        print('RECOVER interrupted training',cell,'at epoch',start_epoch,flush=True)
    torch.cuda.reset_peak_memory_stats() if DEVICE.startswith('cuda') else None; started=time.perf_counter(); max_epochs=fixed_epochs or c['maximum_epochs']
    for epoch in range(start_epoch,max_epochs+1):
        model.train(); ids_np=rng.choice(n,n,replace=True,p=w); losses=[]
        for first in range(0,n,c['effective_batch_size']):
            chosen=torch.tensor(ids_np[first:first+c['effective_batch_size']],device=DEVICE)
            opt.zero_grad(set_to_none=True)
            for micro in range(0,len(chosen),c['physical_batch_size']):
                ids=chosen[micro:micro+c['physical_batch_size']]; b=batch(tr,ids)
                p=model(b) if base_train is None else model(b,base_train[ids])
                loss=nn.functional.smooth_l1_loss(p,tr['target'][ids],beta=c['smooth_l1_beta'])
                if not torch.isfinite(loss): raise RuntimeError(f'{cell}: nonfinite loss')
                (loss*len(ids)/len(chosen)).backward(); losses.append(loss.detach())
            nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip_norm']); opt.step()
        if fixed_epochs:
            score=None; best_epoch=epoch; best_state=state(model)
        else:
            p=predict(model,va,c,base_val).cpu().numpy().astype(float)*scaler['target_scale']
            score=macro(vm.reference_rul_h.to_numpy(dtype=float),p,vm)
            if score<best: best=score; best_epoch=epoch; best_state=state(model)
        record=dict(epoch=epoch,loss=float(torch.stack(losses).mean().cpu()),validation_family_macro_mae_h=score)
        history.append(record)
        spent_now=spent+time.perf_counter()-started
        print(f'EPOCH {cell} {epoch}/{max_epochs} loss={record["loss"]:.6f} val={score} best={best_epoch} seconds={spent_now:.1f}',flush=True)
        atomic_torch(recovery,dict(model=state(model),optimizer=opt.state_dict(),numpy_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=(torch.cuda.get_rng_state_all() if DEVICE.startswith('cuda') else []),history=history,best=best,best_epoch=best_epoch,
            best_state=best_state,seconds=spent_now,seed=seed,source_sha256=source_hashes(),training_families=sorted(map(int,tm.family.unique()))))
        save(OUT/'progress.json',dict(cell=cell,epoch=epoch,maximum_epochs=max_epochs,best_epoch=best_epoch,validation_mae_h=score,total_cell_seconds=spent_now))
    model.load_state_dict(best_state); model.eval()
    info=dict(best_epoch=best_epoch,epochs_completed=len(history),validation_mae_h=None if fixed_epochs else best,
        history=history,training_seconds=spent+time.perf_counter()-started,
        peak_training_gpu_allocated_bytes=(torch.cuda.max_memory_allocated() if DEVICE.startswith('cuda') else 0),training_families=sorted(map(int,tm.family.unique())),
        seed=seed,training_endpoints=n,selection='fixed preselected full-parent duration' if fixed_epochs else c['checkpoint_selection'])
    return model,info


def store(path,model,c,scaler,info,name,seed,**extra):
    atomic_torch(path,dict(state=state(model),config=c,scaler=scaler,info=info,model=name,seed=seed,source_sha256=source_hashes(),**extra))
    save(path.with_suffix('.json'),dict(info,model=name,seed=seed,checkpoint_sha256=sha(path)))


def load_model(path,name=None):
    z=torch.load(path,map_location='cpu',weights_only=True)
    model=build(name or z['model'],z['config']).to(DEVICE); model.load_state_dict(z['state']); model.eval()
    return model,z


def train_standard(name,seed,a,m,v,vm,c,scaler):
    path=ckpt(name,seed)
    if path.exists():
        z=torch.load(path,map_location='cpu',weights_only=True)
        assert compatible(z,name) and z['scaler']==scaler
        print('COMPLETED CHECKPOINT retained',name,seed,flush=True); return
    tr=transform(a,scaler,device=DEVICE); va=transform(v,scaler,device=DEVICE)
    seed_all(seed); model,info=fit(build(name,c),tr,va,m,vm,scaler,c,seed,f'{name}__seed{seed}')
    store(path,model,c,scaler,info,name,seed)
    print('FIT COMPLETE',name,seed,info['validation_mae_h'],flush=True)


def train_s21(seed,a,m,v,vm,c,scaler):
    final=ckpt('S21A-Lightweight-Graph',seed)
    if final.exists():
        z=torch.load(final,map_location='cpu',weights_only=True); assert z['source_sha256']==source_hashes()
        return
    base,bz=load_model(ckpt('LightweightTransformer',seed)); tr=transform(a,scaler,device=DEVICE); va=transform(v,scaler,device=DEVICE)
    base_train=predict(base,tr,c).detach(); base_val=predict(base,va,c).detach()
    expert_path=ckpt('Correction',seed)
    if expert_path.exists(): expert,ez=load_model(expert_path)
    else:
        seed_all(seed); expert,ei=fit(build('Correction',c),tr,va,m,vm,scaler,c,seed,f'Correction__seed{seed}',base_train=base_train,base_val=base_val)
        store(expert_path,expert,c,scaler,ei,'Correction',seed); ez={'info':ei}
    ep,vd=predict(expert,va,c,base_val,True)
    validation_features=gate_features(va,base_val,ep,vd)
    oof_base=np.full(len(m),np.nan); oof_expert=np.full(len(m),np.nan); oof_diag=np.full((len(m),3),np.nan)
    assignment=load(EVIDENCE/'crossfit_partition.json'); fold_records=[]
    # Free full parents and large tensors during the parent cross-fits.
    del base,expert,tr,va,base_train; torch.cuda.empty_cache() if DEVICE.startswith('cuda') else None
    for fold in range(5):
        held=sorted(int(k) for k,value in assignment.items() if value==fold)
        fitting=sorted(set(map(int,m.family.unique()))-set(held))
        aa,mm,_=subset(a,m,fitting); ha,hm,hold_ids=subset(a,m,held); fs=fit_scaler(aa,mm)
        ft=transform(aa,fs,device=DEVICE); ht=transform(ha,fs,device=DEVICE)
        record={}
        for parent,fixed in [('LightweightTransformer',bz['info']['best_epoch']),('Correction',ez['info']['best_epoch'])]:
            path=OUT/'crossfit'/f'seed{seed}_fold{fold}_{parent}.pt'
            if parent=='Correction':
                fbp=predict(fb,ft,c).detach(); hbp=predict(fb,ht,c).detach()
            if path.exists():
                model,z=load_model(path,parent); assert z['scaler']==fs and z['held_families']==held and z['source_sha256']==source_hashes()
                info=z['info']
            else:
                seed_all(seed); model,info=fit(build(parent,c),ft,None,mm,None,fs,c,seed,f'crossfit_seed{seed}_fold{fold}_{parent}',
                    fixed_epochs=fixed,base_train=fbp if parent=='Correction' else None)
                store(path,model,c,fs,info,parent,seed,held_families=held,fold=fold)
            record[parent]=dict(path=path.relative_to(OUT).as_posix(),sha256=sha(path),best_epoch=fixed,training_families=fitting,held_families=held)
            if parent=='LightweightTransformer': fb=model
            else: fe=model
        hp=predict(fb,ht,c); he,hd=predict(fe,ht,c,hp,True)
        oof_base[hold_ids]=hp.cpu().numpy().astype(float)*fs['target_scale']/scaler['target_scale']
        oof_expert[hold_ids]=he.cpu().numpy().astype(float)*fs['target_scale']/scaler['target_scale']
        oof_diag[hold_ids]=hd.cpu().numpy()
        fold_records.append(dict(fold=fold,parents=record))
        del fb,fe,model,ft,ht,fbp,hbp,hp,he,hd; torch.cuda.empty_cache() if DEVICE.startswith('cuda') else None
        print('CROSSFIT COMPLETE',seed,fold,flush=True)
    assert np.isfinite(oof_base).all() and np.isfinite(oof_expert).all() and np.isfinite(oof_diag).all()
    crossfit=OUT/'crossfit'/f'seed{seed}_oof.npz'
    np.savez(crossfit,base=oof_base,expert=oof_expert,diagnostics=oof_diag,family=m.family.to_numpy(),
             endpoint_id=m.endpoint_id.to_numpy(dtype=str),fold=np.array([assignment[str(int(f))] for f in m.family]))
    tr=transform(a,scaler,device=DEVICE); ob=torch.tensor(oof_base,dtype=torch.float32,device=DEVICE); oe=torch.tensor(oof_expert,dtype=torch.float32,device=DEVICE)
    od=torch.tensor(oof_diag,dtype=torch.float32,device=DEVICE); features=gate_features(tr,ob,oe,od)
    truth=tr['target']; advantage=(ob-truth).abs()-(oe-truth).abs(); label=(advantage>1e-12).float(); lossweight=.02+advantage.abs()
    seed_all(seed); gate=Gate(c).to(DEVICE); fw=torch.tensor(family_weights(m),dtype=torch.float32,device=DEVICE)
    with torch.no_grad():
        gate.mean.copy_((features*fw[:,None]).sum(0)); sd=((features-gate.mean)**2*fw[:,None]).sum(0).sqrt(); sd[sd<1e-6]=1; gate.scale.copy_(sd)
    opt=torch.optim.AdamW(gate.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    rng=np.random.default_rng(seed); best=math.inf; best_state=None; best_epoch=0; history=[]; started=time.perf_counter()
    for epoch in range(1,c['maximum_epochs']+1):
        gate.train(); ids_np=rng.choice(len(m),len(m),replace=True,p=family_weights(m)); losses=[]
        for first in range(0,len(m),c['effective_batch_size']):
            ids=torch.tensor(ids_np[first:first+c['effective_batch_size']],device=DEVICE); opt.zero_grad(set_to_none=True)
            loss=(nn.functional.binary_cross_entropy_with_logits(gate.logits(features[ids]),label[ids],reduction='none')*lossweight[ids]).mean()/lossweight[ids].mean()
            loss.backward(); nn.utils.clip_grad_norm_(gate.parameters(),c['gradient_clip_norm']); opt.step(); losses.append(loss.detach())
        gate.eval()
        with torch.no_grad(): p=(base_val+gate(validation_features)*(ep-base_val)).cpu().numpy().astype(float)*scaler['target_scale']
        score=macro(v['target'],p,vm); history.append(dict(epoch=epoch,loss=float(torch.stack(losses).mean().cpu()),validation_mae_h=score))
        if score<best: best=score; best_epoch=epoch; best_state=state(gate)
        if epoch==1 or epoch%5==0: print('GATE',seed,epoch,'validation',score,flush=True)
    gate.load_state_dict(best_state)
    system=S21(c).to(DEVICE); fullbase,_=load_model(ckpt('LightweightTransformer',seed)); fullexpert,_=load_model(expert_path)
    system.base.load_state_dict(fullbase.state_dict()); system.expert.load_state_dict(fullexpert.state_dict()); system.gate.load_state_dict(gate.state_dict())
    info=dict(best_epoch=best_epoch,epochs_completed=c['maximum_epochs'],validation_mae_h=best,history=history,training_seconds=time.perf_counter()-started,
        training_families=sorted(map(int,m.family.unique())),seed=seed,training_endpoints=len(m),crossfit_folds=5,
        base_checkpoint_sha256=sha(ckpt('LightweightTransformer',seed)),expert_checkpoint_sha256=sha(expert_path),selection=c['checkpoint_selection'])
    store(final,system,c,scaler,info,'S21A-Lightweight-Graph',seed,crossfit=fold_records,oof_path=crossfit.relative_to(OUT).as_posix(),oof_sha256=sha(crossfit))
    print('FIT COMPLETE S21A-Lightweight-Graph',seed,best,flush=True)
