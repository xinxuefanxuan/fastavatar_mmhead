#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from motion_model.channel_vae import ChannelTemporalVAE
from motion_model.train_text_to_channel_hybrid import HybridNet, DEFAULT_SCALES


def load_cvae(ckpt,dev):
    c=torch.load(ckpt,map_location=dev)
    m=ChannelTemporalVAE(int(c['input_dim']),int(c['target_len']),int(c['latent_dim']),int(c['hidden_dim'])).to(dev)
    m.load_state_dict(c['model']);m.eval(); return m,c

def compose(scores, classes, protos, scales, dev):
    B=scores.shape[0]; out={}
    for ch in ['expr','head','jaw']:
        z0=protos[ch]['neutral'].to(dev)
        z=z0[None,:].repeat(B,1)
        for i,c in enumerate(classes):
            z = z + scores[:,i:i+1]*float(scales[ch].get(c,0.0))*(protos[ch][c].to(dev)-z0)[None,:]
        out[ch]=z
    return out

def stats(m):
    e=np.linalg.norm(m[:,:50],axis=1);h=np.linalg.norm(m[:,50:53],axis=1);j=np.linalg.norm(m[:,53:56],axis=1);y=m[:,51]
    return {'expr norm mean/max':(float(e.mean()),float(e.max())),'head norm mean/max':(float(h.mean()),float(h.max())),'jaw norm mean/max':(float(j.mean()),float(j.max())),'yaw min/max':(float(y.min()),float(y.max()))}

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
    ap.add_argument('--generation_mode',choices=['proto_only','hybrid','direct_residual'],default='hybrid')
    ap.add_argument('--use_residual_with_manual_labels',action='store_true')
    args=ap.parse_args()
    dev=torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')

    expr_m,eck=load_cvae(args.expr_checkpoint,dev); head_m,hck=load_cvae(args.head_checkpoint,dev); jaw_m,jck=load_cvae(args.jaw_checkpoint,dev)
    cp=torch.load(args.channel_prototypes,map_location='cpu'); classes=cp['primitive_classes']
    protos={ch:{k:v.float() for k,v in cp['prototypes'][ch].items()} for ch in ['expr','head','jaw']}
    scales={k:v.copy() for k,v in DEFAULT_SCALES.items()}
    if args.manual_channel_weights_json and args.manual_channel_weights_json.exists():
        cfg=json.loads(args.manual_channel_weights_json.read_text())
        for ch in scales:
            for k,v in cfg.get(ch,{}).items(): scales[ch][k]=float(v)

    net=None; out=None
    if args.checkpoint is not None:
        ck=torch.load(args.checkpoint,map_location=dev)
        net=HybridNet(int(ck['input_dim']),len(classes),int(ck['expr_latent_dim']),int(ck['head_latent_dim']),int(ck['jaw_latent_dim']),int(ck['args']['hidden_dim']),int(ck['args']['num_layers']),float(ck['args'].get('dropout',0.1))).to(dev)
        net.load_state_dict(ck['model']); net.eval()

    if args.manual_labels:
        labs=[x.strip() for x in args.manual_labels.split(',') if x.strip()]
        s=np.zeros((1,len(classes)),np.float32)
        for i,c in enumerate(classes):
            if c in labs: s[0,i]=1.0
        scores=torch.from_numpy(s).to(dev)
        z_proto=compose(scores,classes,protos,scales,dev)
        if args.generation_mode=='hybrid' and args.use_residual_with_manual_labels and net is not None:
            enc=SentenceTransformer(args.encoder_name,device=args.device)
            emb=np.asarray(enc.encode([args.prompt],convert_to_numpy=True,normalize_embeddings=False),dtype=np.float32)
            with torch.no_grad(): out=net(torch.from_numpy(emb).to(dev))
        primitive_scores={classes[i]:float(scores[0,i].item()) for i in range(len(classes))}
    else:
        if net is None: raise SystemExit('checkpoint required when manual_labels absent')
        enc=SentenceTransformer(args.encoder_name,device=args.device)
        emb=np.asarray(enc.encode([args.prompt],convert_to_numpy=True,normalize_embeddings=False),dtype=np.float32)
        with torch.no_grad(): out=net(torch.from_numpy(emb).to(dev))
        logits=out['logits']; scores=torch.sigmoid(logits/max(args.temperature,1e-6))
        if args.topk and args.topk>0 and args.topk<scores.shape[1]:
            v,ix=torch.topk(scores,args.topk,dim=1); m=torch.zeros_like(scores); m.scatter_(1,ix,v); scores=m
        scores=torch.where(scores>=args.threshold,scores,torch.zeros_like(scores))
        z_proto=compose(scores,classes,protos,scales,dev)
        primitive_scores={classes[i]:float(scores[0,i].item()) for i in range(len(classes))}

    if out is None and net is not None and args.manual_labels:
        # obtain residual from prompt if needed in hybrid mode
        enc=SentenceTransformer(args.encoder_name,device=args.device)
        emb=np.asarray(enc.encode([args.prompt],convert_to_numpy=True,normalize_embeddings=False),dtype=np.float32)
        with torch.no_grad(): out=net(torch.from_numpy(emb).to(dev))

    if args.generation_mode=='proto_only' or out is None:
        zf=z_proto; residual_norms={'expr':0.0,'head':0.0,'jaw':0.0}
    elif args.generation_mode=='direct_residual':
        zf={'expr':out['r_expr'],'head':out['r_head'],'jaw':out['r_jaw']}
        residual_norms={k:float(zf[k].norm().item()) for k in zf}
    else:
        rs=float(torch.tensor( out['args']['residual_scale'] if isinstance(out,dict) and 'args' in out else 0.3)) if False else 0.3
        zf={
            'expr':z_proto['expr'] + rs*out['g_expr']*out['r_expr'],
            'head':z_proto['head'] + rs*out['g_head']*out['r_head'],
            'jaw': z_proto['jaw']  + rs*out['g_jaw'] *out['r_jaw'],
        }
        residual_norms={k:float((out['r_'+k] if k!='expr' else out['r_expr']).norm().item()) if k!='expr' else float(out['r_expr'].norm().item()) for k in ['expr','head','jaw']}

    with torch.no_grad():
        en=expr_m.decode(zf['expr'])[0].cpu().numpy().astype(np.float32)
        hn=head_m.decode(zf['head'])[0].cpu().numpy().astype(np.float32)
        jn=jaw_m.decode(zf['jaw'])[0].cpu().numpy().astype(np.float32)
    mnorm=np.concatenate([en,hn,jn],axis=1)
    st=json.loads(args.norm_stats.read_text()); mean=np.asarray(st['mean'],np.float32)[:56]; std=np.asarray(st['std'],np.float32)[:56]
    mraw=mnorm*std[None,:]+mean[None,:]
    args.output_npz.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output_npz,motion=mraw.astype(np.float32),motion_raw=mraw.astype(np.float32),motion_norm=mnorm.astype(np.float32),expr_delta=mraw[:,:50].astype(np.float32),head_delta=mraw[:,50:53].astype(np.float32),jaw_delta=mraw[:,53:56].astype(np.float32))

    print('prompt:',args.prompt)
    print('primitive scores:',json.dumps(primitive_scores,ensure_ascii=False))
    print('selected primitives:',[k for k,v in primitive_scores.items() if v>0])
    print('z_proto norms:',{k:float(v.norm().item()) for k,v in z_proto.items()})
    print('residual norms:',residual_norms)
    print('final z norms:',{k:float(v.norm().item()) for k,v in zf.items()})
    for k,v in stats(mraw).items(): print(k,v)
    print('output:',args.output_npz)

if __name__=='__main__': main()
