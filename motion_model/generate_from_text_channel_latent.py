#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from motion_model.channel_vae import ChannelTemporalVAE
from motion_model.train_text_to_channel_latent import Net, DEFAULT_SCALES


def load_cvae(ckpt,device):
    c=torch.load(ckpt,map_location=device)
    m=ChannelTemporalVAE(int(c['input_dim']),int(c['target_len']),int(c['latent_dim']),int(c['hidden_dim'])).to(device)
    m.load_state_dict(c['model']);m.eval()
    return m,c

def stats(m):
    e=np.linalg.norm(m[:,:50],axis=1);h=np.linalg.norm(m[:,50:53],axis=1);j=np.linalg.norm(m[:,53:56],axis=1);y=m[:,51]
    return {'expr norm mean/max':(float(e.mean()),float(e.max())),'head norm mean/max':(float(h.mean()),float(h.max())),'jaw norm mean/max':(float(j.mean()),float(j.max())),'yaw min/max':(float(y.min()),float(y.max()))}

def compose_from_labels(labels,classes,protos,scales,device):
    out={}
    for ch in ['expr','head','jaw']:
        z0=protos[ch]['neutral'].to(device); z=z0.clone()
        for lab in labels:
            if lab in classes:
                z=z+float(scales[ch].get(lab,0.0))*(protos[ch][lab].to(device)-z0)
        out[ch]=z[None,:]
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prompt',type=str,required=True)
    ap.add_argument('--checkpoint',type=Path,default=None)
    ap.add_argument('--channel_prototypes',type=Path,required=True)
    ap.add_argument('--expr_checkpoint',type=Path,required=True)
    ap.add_argument('--head_checkpoint',type=Path,required=True)
    ap.add_argument('--jaw_checkpoint',type=Path,required=True)
    ap.add_argument('--norm_stats',type=Path,required=True)
    ap.add_argument('--output_npz',type=Path,required=True)
    ap.add_argument('--encoder_name',type=str,default='/home/yuanyuhao/models/all-MiniLM-L6-v2')
    ap.add_argument('--threshold',type=float,default=0.4)
    ap.add_argument('--topk',type=int,default=None)
    ap.add_argument('--temperature',type=float,default=1.0)
    ap.add_argument('--device',type=str,default='cuda')
    ap.add_argument('--manual_labels',type=str,default=None)
    ap.add_argument('--manual_channel_weights_json',type=Path,default=None)
    args=ap.parse_args()
    dev=torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')

    expr_m,expr_ck=load_cvae(args.expr_checkpoint,dev); head_m,head_ck=load_cvae(args.head_checkpoint,dev); jaw_m,jaw_ck=load_cvae(args.jaw_checkpoint,dev)
    cp=torch.load(args.channel_prototypes,map_location='cpu')
    classes=cp['primitive_classes']
    protos={ch:{k:v.float() for k,v in cp['prototypes'][ch].items()} for ch in ['expr','head','jaw']}
    scales={k:v.copy() for k,v in DEFAULT_SCALES.items()}
    if args.manual_channel_weights_json and args.manual_channel_weights_json.exists():
        cfg=json.loads(args.manual_channel_weights_json.read_text())
        for ch in scales:
            for k,v in cfg.get(ch,{}).items(): scales[ch][k]=float(v)

    if args.manual_labels:
        labels=[x.strip() for x in args.manual_labels.split(',') if x.strip()]
        z=compose_from_labels(labels,classes,protos,scales,dev)
        scores={c:(1.0 if c in labels else 0.0) for c in classes}
    else:
        if args.checkpoint is None: raise SystemExit('checkpoint required when manual_labels absent')
        ck=torch.load(args.checkpoint,map_location=dev)
        net=Net(int(ck['input_dim']),len(classes),int(ck['expr_latent_dim']),int(ck['head_latent_dim']),int(ck['jaw_latent_dim']),int(ck['args']['hidden_dim']),int(ck['args']['num_layers'])).to(dev)
        net.load_state_dict(ck['model']); net.eval()
        from sentence_transformers import SentenceTransformer
        enc=SentenceTransformer(args.encoder_name,device=args.device)
        emb=np.asarray(enc.encode([args.prompt],convert_to_numpy=True,normalize_embeddings=False),dtype=np.float32)
        with torch.no_grad(): ze,zh,zj,logits=net(torch.from_numpy(emb).to(dev))
        prob=torch.sigmoid(logits/max(args.temperature,1e-6))[0].cpu().numpy()
        if args.topk and args.topk>0 and args.topk<len(prob):
            ix=np.argsort(-prob)[:args.topk]; mask=np.zeros_like(prob); mask[ix]=1; prob=prob*mask
        prob=np.where(prob>=args.threshold,prob,0.0)
        scores={classes[i]:float(prob[i]) for i in range(len(classes))}
        z={'expr':ze,'head':zh,'jaw':zj}

    with torch.no_grad():
        expr=expr_m.decode(z['expr'])[0].cpu().numpy().astype(np.float32)
        head=head_m.decode(z['head'])[0].cpu().numpy().astype(np.float32)
        jaw=jaw_m.decode(z['jaw'])[0].cpu().numpy().astype(np.float32)
    mnorm=np.concatenate([expr,head,jaw],axis=1)
    st=json.loads(args.norm_stats.read_text()); mean=np.asarray(st['mean'],np.float32)[:56]; std=np.asarray(st['std'],np.float32)[:56]
    mraw=mnorm*std[None,:]+mean[None,:]

    args.output_npz.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output_npz,motion=mraw.astype(np.float32),motion_raw=mraw.astype(np.float32),motion_norm=mnorm.astype(np.float32),expr_delta=mraw[:,:50].astype(np.float32),head_delta=mraw[:,50:53].astype(np.float32),jaw_delta=mraw[:,53:56].astype(np.float32))
    print('prompt=',args.prompt)
    print('primitive_scores=',json.dumps(scores,ensure_ascii=False))
    print('selected=',[k for k,v in scores.items() if v>0])
    print('z_norms',float(z['expr'].norm().item()),float(z['head'].norm().item()),float(z['jaw'].norm().item()))
    for k,v in stats(mraw).items(): print(k,v)
    print('output=',args.output_npz)

if __name__=='__main__': main()
