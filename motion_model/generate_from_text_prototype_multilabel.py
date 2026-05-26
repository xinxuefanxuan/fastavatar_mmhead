#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from motion_model.models import TemporalConvVAE

class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,hidden=512,layers=3,dropout=0.1):
        super().__init__(); seq=[]; d=in_dim
        for _ in range(max(1,layers-1)):
            seq += [nn.Linear(d,hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]; d=hidden
        seq += [nn.Linear(d,out_dim)]; self.net=nn.Sequential(*seq)
    def forward(self,x): return self.net(x)

def weighted_stats(m):
    ex=np.linalg.norm(m[:,:50],axis=1); hd=np.linalg.norm(m[:,50:53],axis=1); jw=np.linalg.norm(m[:,53:56],axis=1); yw=m[:,51]
    return {'expr norm mean/max':(ex.mean(),ex.max()),'head norm mean/max':(hd.mean(),hd.max()),'jaw norm mean/max':(jw.mean(),jw.max()),'head_yaw min/max':(yw.min(),yw.max())}

def apply_topk(scores,topk):
    if topk is None or topk<=0 or topk>=len(scores): return scores
    ix=np.argsort(-scores)[:topk]; out=np.zeros_like(scores); out[ix]=scores[ix]; return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prompt',type=str,required=True); ap.add_argument('--checkpoint',type=Path,default=None)
    ap.add_argument('--prototype_path',type=Path,required=True); ap.add_argument('--vae_checkpoint',type=Path,required=True); ap.add_argument('--norm_stats',type=Path,required=True)
    ap.add_argument('--prototype_scales_json',type=Path,default=Path('outputs/mmhead_debug/text_prototype_debug/prototype_scales_calibrated.json'))
    ap.add_argument('--output_npz',type=Path,required=True); ap.add_argument('--encoder_name',type=str,default='/home/yuanyuhao/models/all-MiniLM-L6-v2')
    ap.add_argument('--threshold',type=float,default=0.5); ap.add_argument('--topk',type=int,default=None); ap.add_argument('--temperature',type=float,default=1.0); ap.add_argument('--device',type=str,default='cuda')
    ap.add_argument('--manual_weights_json',type=Path,default=None)
    args=ap.parse_args()

    if args.manual_weights_json is None and args.checkpoint is None:
        raise ValueError('Either --checkpoint or --manual_weights_json must be provided.')

    classes=None
    c2i=None
    model=None
    if args.checkpoint is not None:
        ck=torch.load(args.checkpoint,map_location=args.device)
        classes=ck['primitive_classes']
        c2i=ck.get('class_to_idx',{c:i for i,c in enumerate(classes)})
        model=MLP(int(ck['embedding_dim']),len(classes),int(ck.get('args',{}).get('hidden_dim',512)),int(ck.get('args',{}).get('num_layers',3)),float(ck.get('args',{}).get('dropout',0.1))).to(args.device)
        model.load_state_dict(ck['model']); model.eval()

    prot=torch.load(args.prototype_path,map_location='cpu')
    if classes is None:
        classes=list(prot['label_to_mean_mu'].keys())
        c2i={c:i for i,c in enumerate(classes)}
    zbank=np.stack([np.asarray(prot['label_to_mean_mu'][c],dtype=np.float32) for c in classes],0)
    scales=np.ones((len(classes),),dtype=np.float32)
    if args.prototype_scales_json is not None and args.prototype_scales_json.exists():
        cfg=json.loads(args.prototype_scales_json.read_text(encoding='utf-8'))
        for i,c in enumerate(classes):
            if c in cfg: scales[i]=float(cfg[c])

    if args.manual_weights_json is not None:
        mw=json.loads(args.manual_weights_json.read_text(encoding='utf-8'))
        scores=np.zeros((len(classes),),dtype=np.float32)
        for k,v in mw.items():
            if k in c2i: scores[c2i[k]]=float(v)
        print('[Manual] raw amplitude weights used')
    else:
        from sentence_transformers import SentenceTransformer
        enc=SentenceTransformer(args.encoder_name,device=args.device)
        emb=np.asarray(enc.encode([args.prompt],convert_to_numpy=True,normalize_embeddings=False),dtype=np.float32)
        with torch.no_grad(): logits=model(torch.from_numpy(emb).to(args.device))[0]
        scores=torch.sigmoid(logits/max(args.temperature,1e-6)).detach().cpu().numpy()
        scores=apply_topk(scores,args.topk); scores=np.where(scores>=args.threshold,scores,0.0)

    final_weights=scores*scales
    ni=c2i['neutral']; z0=zbank[ni]; dirs=(zbank-z0[None,:]); z=z0 + final_weights @ dirs

    vck=torch.load(args.vae_checkpoint,map_location=args.device)
    vae=TemporalConvVAE(in_dim=56,latent_dim=int(vck.get('args',{}).get('latent_dim',64))).to(args.device); vae.load_state_dict(vck['model']); vae.eval()
    with torch.no_grad(): pred_norm=vae.decode(torch.from_numpy(z[None,:,None]).to(args.device).repeat(1,1,64))[0].cpu().numpy().astype(np.float32)

    st=json.loads(args.norm_stats.read_text()); mean=np.asarray(st['mean'],dtype=np.float32); std=np.asarray(st['std'],dtype=np.float32)
    motion=pred_norm*std[None,:]+mean[None,:]
    args.output_npz.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output_npz,motion=motion.astype(np.float32),motion_raw=motion.astype(np.float32),motion_norm=pred_norm.astype(np.float32),expr_delta=motion[:,0:50].astype(np.float32),head_delta=motion[:,50:53].astype(np.float32),jaw_delta=motion[:,53:56].astype(np.float32))

    print(f'prompt={args.prompt}')
    print('raw_scores='+json.dumps({classes[i]:float(scores[i]) for i in range(len(classes))},ensure_ascii=False))
    print('selected=' + json.dumps([classes[i] for i in range(len(classes)) if scores[i]>0],ensure_ascii=False))
    print('final_weights='+json.dumps({classes[i]:float(final_weights[i]) for i in range(len(classes))},ensure_ascii=False))
    s=weighted_stats(motion)
    for k,v in s.items(): print(f'{k}: {v[0]:.6f} / {v[1]:.6f}')
    print(f'output={args.output_npz}')

if __name__=='__main__': main()
