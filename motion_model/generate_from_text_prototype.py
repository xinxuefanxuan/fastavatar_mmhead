#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from motion_model.models import TemporalConvVAE

class TextToPrototypeMLP(nn.Module):
    def __init__(self,in_dim,latent_dim,num_classes,hidden_dim=512,num_layers=3,dropout=0.1,use_residual=False):
        super().__init__(); layers=[]; d=in_dim
        for _ in range(max(1,num_layers-1)):
            layers += [nn.Linear(d,hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]; d=hidden_dim
        self.backbone=nn.Sequential(*layers); self.logit_head=nn.Linear(d,num_classes); self.res_head=nn.Linear(d,latent_dim) if use_residual else None
    def forward(self,x):
        h=self.backbone(x); return self.logit_head(h),(self.res_head(h) if self.res_head is not None else None)

def parse_csv(s): return [x.strip() for x in s.split(',') if x.strip()]

def apply_topk_probs(probs, topk):
    if topk is None or topk<=0 or topk>=len(probs): return probs
    ix=np.argsort(-probs)[:topk]; out=np.zeros_like(probs); out[ix]=probs[ix]; s=out.sum(); return out/(s+1e-8)

def print_stats(m):
    ex,hd,jw=m[:,:50],m[:,50:53],m[:,53:56]
    for n,a in [('expr',ex),('head',hd),('jaw',jw)]:
        d=np.linalg.norm(a,axis=1); print(f'[{n}] norm mean={d.mean():.6f} max={d.max():.6f}')
    y=hd[:,1]; print(f'[head_yaw] min={y.min():.6f} max={y.max():.6f}')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prompt',type=str,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--prototype_path',type=Path,required=True)
    ap.add_argument('--vae_checkpoint',type=Path,required=True); ap.add_argument('--norm_stats',type=Path,required=True); ap.add_argument('--output_npz',type=Path,required=True)
    ap.add_argument('--encoder_name',type=str,default='/home/yuanyuhao/models/all-MiniLM-L6-v2'); ap.add_argument('--device',type=str,default='cuda')
    ap.add_argument('--temperature',type=float,default=1.0); ap.add_argument('--hard_top1',action='store_true'); ap.add_argument('--topk',type=int,default=None)
    ap.add_argument('--prototype_scales_json',type=Path,default=None); ap.add_argument('--latent_scale',type=float,default=1.0)
    ap.add_argument('--use_residual',action='store_true'); ap.add_argument('--residual_scale',type=float,default=0.1)
    ap.add_argument('--manual_weights_json',type=Path,default=None)
    args=ap.parse_args()

    from sentence_transformers import SentenceTransformer
    enc=SentenceTransformer(args.encoder_name,device=args.device)
    emb=np.asarray(enc.encode([args.prompt],convert_to_numpy=True,normalize_embeddings=False),dtype=np.float32)[0]

    ck=torch.load(args.checkpoint,map_location=args.device)
    primitive_classes=ck['primitive_classes']; c2i=ck.get('class_to_idx',{c:i for i,c in enumerate(primitive_classes)})
    use_res=args.use_residual or bool(ck.get('use_residual',False))
    model=TextToPrototypeMLP(int(ck['embedding_dim']),int(ck['latent_dim']),len(primitive_classes),int(ck.get('args',{}).get('hidden_dim',512)),int(ck.get('args',{}).get('num_layers',3)),float(ck.get('args',{}).get('dropout',0.1)),use_residual=use_res).to(args.device)
    model.load_state_dict(ck['model']); model.eval()

    prot=torch.load(args.prototype_path,map_location='cpu')
    zbank=np.stack([np.asarray(prot['label_to_mean_mu'][c],dtype=np.float32) for c in primitive_classes],0)
    scales=np.ones((len(primitive_classes),),dtype=np.float32)
    if args.prototype_scales_json is not None:
        cfg=json.loads(args.prototype_scales_json.read_text(encoding='utf-8'))
        for i,c in enumerate(primitive_classes):
            if c in cfg: scales[i]=float(cfg[c])

    with torch.no_grad():
        logits,res=model(torch.from_numpy(emb[None,...]).to(args.device))
    probs=torch.softmax(logits[0]/max(args.temperature,1e-6),dim=0).detach().cpu().numpy()
    probs=apply_topk_probs(probs,args.topk)
    if args.hard_top1:
        i=int(np.argmax(probs)); hp=np.zeros_like(probs); hp[i]=1.0; probs=hp

    if args.manual_weights_json is not None:
        mw=json.loads(args.manual_weights_json.read_text(encoding='utf-8'))
        probs=np.zeros_like(probs)
        for k,v in mw.items():
            if k in c2i: probs[c2i[k]]=float(v)
        s=probs.sum(); probs=probs/(s+1e-8)
        print('[ManualWeights] enabled')

    ni=primitive_classes.index('neutral'); z_neutral=zbank[ni]
    dirs=(zbank-z_neutral[None,:])*(scales[:,None])
    z_proto=z_neutral + probs @ dirs
    z_final=z_proto.copy()
    if res is not None and use_res:
        z_final = z_final + float(args.residual_scale)*res[0].detach().cpu().numpy()
    z_final = z_final * float(args.latent_scale)

    vck=torch.load(args.vae_checkpoint,map_location=args.device)
    vae=TemporalConvVAE(in_dim=56,latent_dim=int(vck.get('args',{}).get('latent_dim',64))).to(args.device)
    vae.load_state_dict(vck['model']); vae.eval()
    with torch.no_grad():
        pred_norm=vae.decode(torch.from_numpy(z_final[None,:,None]).to(args.device).repeat(1,1,64))[0].cpu().numpy().astype(np.float32)

    st=json.loads(args.norm_stats.read_text(encoding='utf-8')); mean=np.asarray(st['mean'],dtype=np.float32); std=np.asarray(st['std'],dtype=np.float32)
    motion_raw=pred_norm*std[None,:]+mean[None,:]

    args.output_npz.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output_npz,motion=motion_raw.astype(np.float32),motion_raw=motion_raw.astype(np.float32),motion_norm=pred_norm.astype(np.float32),expr_delta=motion_raw[:,0:50].astype(np.float32),head_delta=motion_raw[:,50:53].astype(np.float32),jaw_delta=motion_raw[:,53:56].astype(np.float32))

    top=np.argsort(-probs)[:3]
    print(f'[Prompt] {args.prompt}')
    print(f'[Config] hard_top1={args.hard_top1} topk={args.topk} temperature={args.temperature}')
    print('[Top3] ' + ', '.join([f"{primitive_classes[int(i)]}:{float(probs[int(i)]):.4f}" for i in top]))
    print('[Weights] ' + json.dumps({primitive_classes[i]:float(probs[i]) for i in range(len(primitive_classes))},ensure_ascii=False))
    print(f'[Latent] norm={float(np.linalg.norm(z_final)):.6f}')
    print_stats(motion_raw)
    print(f'[Output] {args.output_npz}')

if __name__=='__main__': main()
