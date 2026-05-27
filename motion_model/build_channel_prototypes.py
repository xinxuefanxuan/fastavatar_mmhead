#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from motion_model.channel_vae import ChannelTemporalVAE

CH_SLICE={"expr":(0,50),"head":(50,53),"jaw":(53,56)}

def read_jsonl(p:Path):
    out=[]
    with p.open('r',encoding='utf-8') as f:
        for l in f:
            l=l.strip()
            if l: out.append(json.loads(l))
    return out

def fit_len(x,t):
    if x.shape[0]>=t:return x[:t]
    return np.concatenate([x,np.repeat(x[-1:],t-x.shape[0],0)],0)

def load_motion_norm(npz_path,mean,std,target_len):
    a=np.load(npz_path,allow_pickle=False)
    if 'motion_norm' in a:m=np.asarray(a['motion_norm'],np.float32)
    else:
        raw=np.asarray(a['motion'] if 'motion' in a else a['motion_raw'],np.float32)
        m=(raw-mean[None,:])/np.maximum(std[None,:],1e-8)
    return fit_len(m[:,:56],target_len)

def load_model(ckpt,device):
    c=torch.load(ckpt,map_location=device)
    m=ChannelTemporalVAE(int(c['input_dim']),int(c['target_len']),int(c['latent_dim']),int(c['hidden_dim'])).to(device)
    m.load_state_dict(c['model']);m.eval()
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--label_jsonl',type=Path,default=Path('outputs/mmhead_debug/primitive_labels_v2/all_labeled_normalized.jsonl'))
    ap.add_argument('--train_manifest',type=Path,default=Path('outputs/mmhead_debug/motion_dataset_v1_ae_debug/train.jsonl'))
    ap.add_argument('--val_manifest',type=Path,default=Path('outputs/mmhead_debug/motion_dataset_v1_ae_debug/val.jsonl'))
    ap.add_argument('--expr_checkpoint',type=Path,default=Path('outputs/mmhead_debug/channel_vae_v1/expr/best.pt'))
    ap.add_argument('--head_checkpoint',type=Path,default=Path('outputs/mmhead_debug/channel_vae_v1/head/best.pt'))
    ap.add_argument('--jaw_checkpoint',type=Path,default=Path('outputs/mmhead_debug/channel_vae_v1/jaw/best.pt'))
    ap.add_argument('--norm_stats',type=Path,default=Path('outputs/mmhead_debug/motion_dataset_v1_ae_debug/norm_stats.json'))
    ap.add_argument('--output_path',type=Path,default=Path('outputs/mmhead_debug/channel_prototypes_v1/channel_prototypes.pt'))
    ap.add_argument('--primitive_classes',type=str,default='turn_left,turn_right,nod,smile,mouth_open,neutral')
    ap.add_argument('--max_per_label',type=int,default=450)
    ap.add_argument('--target_len',type=int,default=64)
    ap.add_argument('--device',type=str,default='cuda')
    args=ap.parse_args()
    dev=torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')

    labels=read_jsonl(args.label_jsonl)
    manifests=read_jsonl(args.train_manifest)+read_jsonl(args.val_manifest)
    m_map={str(r.get('sample_id','')):r for r in manifests}
    st=json.loads(args.norm_stats.read_text());mean=np.asarray(st['mean'],np.float32);std=np.asarray(st['std'],np.float32)
    classes=[x.strip() for x in args.primitive_classes.split(',') if x.strip()]

    models={'expr':load_model(args.expr_checkpoint,dev),'head':load_model(args.head_checkpoint,dev),'jaw':load_model(args.jaw_checkpoint,dev)}
    feats={c:defaultdict(list) for c in ['expr','head','jaw']}
    counts=defaultdict(int)
    motion_energy=[]

    for r in labels:
        sid=str(r.get('sample_id','')); lab=str(r.get('label',''))
        if lab not in classes: continue
        if counts[lab]>=args.max_per_label: continue
        mr=m_map.get(sid); 
        if mr is None: continue
        p=mr.get('npz_path') or mr.get('motion_path')
        if not p or not Path(p).exists(): continue
        mn=load_motion_norm(Path(p),mean,std,args.target_len)
        motion_energy.append((float(np.linalg.norm(mn,axis=1).mean()),mn))
        for ch,(s,e) in CH_SLICE.items():
            x=torch.from_numpy(mn[:,s:e][None].astype(np.float32)).to(dev)
            with torch.no_grad(): mu,_=models[ch].encode(x)
            feats[ch][lab].append(mu[0].cpu())
        counts[lab]+=1

    # neutral fallback by low-motion samples
    neutral_sparse = len(feats['expr'].get('neutral',[]))<5
    if neutral_sparse and motion_energy:
        motion_energy.sort(key=lambda x:x[0])
        low=[m for _,m in motion_energy[:min(50,len(motion_energy))]]
        for mn in low:
            for ch,(s,e) in CH_SLICE.items():
                x=torch.from_numpy(mn[:,s:e][None].astype(np.float32)).to(dev)
                with torch.no_grad(): mu,_=models[ch].encode(x)
                feats[ch]['neutral'].append(mu[0].cpu())

    prot={ch:{} for ch in ['expr','head','jaw']}
    for ch in prot:
        for lab in classes:
            arr=feats[ch].get(lab,[])
            if arr:
                z=torch.stack(arr,0).mean(0)
            else:
                z=torch.zeros(models[ch].latent_dim)
            prot[ch][lab]=z
            print(f"[{ch}] {lab}: count={len(arr)} norm={z.norm().item():.6f}")

    out={
        'primitive_classes':classes,
        'channels':['expr','head','jaw'],
        'prototypes':prot,
        'counts':dict(counts),
        'channel_vae_checkpoints':{'expr':str(args.expr_checkpoint),'head':str(args.head_checkpoint),'jaw':str(args.jaw_checkpoint)},
        'target_len':args.target_len,
    }
    args.output_path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(out,args.output_path)
    print(f'saved {args.output_path}')

if __name__=='__main__': main()
