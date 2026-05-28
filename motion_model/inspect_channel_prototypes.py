#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from motion_model.channel_vae import ChannelTemporalVAE


def stats(m):
    e=np.linalg.norm(m[:,:50],axis=1);h=np.linalg.norm(m[:,50:53],axis=1);j=np.linalg.norm(m[:,53:56],axis=1);y=m[:,51]
    return {'expr norm mean/max':(float(e.mean()),float(e.max())),'head norm mean/max':(float(h.mean()),float(h.max())),'jaw norm mean/max':(float(j.mean()),float(j.max())),'yaw min/max':(float(y.min()),float(y.max()))}

def load_cvae(ckpt,device):
    c=torch.load(ckpt,map_location=device)
    m=ChannelTemporalVAE(int(c['input_dim']),int(c['target_len']),int(c['latent_dim']),int(c['hidden_dim'])).to(device)
    m.load_state_dict(c['model']);m.eval()
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--channel_prototypes',type=Path,required=True)
    ap.add_argument('--norm_stats',type=Path,default=Path('outputs/mmhead_debug/motion_dataset_v1_ae_debug/norm_stats.json'))
    ap.add_argument('--decode',action='store_true')
    ap.add_argument('--expr_checkpoint',type=Path,default=None)
    ap.add_argument('--head_checkpoint',type=Path,default=None)
    ap.add_argument('--jaw_checkpoint',type=Path,default=None)
    ap.add_argument('--device',type=str,default='cuda')
    args=ap.parse_args()
    dev=torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')

    ck=torch.load(args.channel_prototypes,map_location='cpu')
    prot=ck['prototypes']; classes=ck['primitive_classes']
    print('primitive_classes:',classes)
    print('channels:',ck.get('channels'))
    print('counts:',ck.get('counts'))

    for ch in ['expr','head','jaw']:
        print('\n===',ch,'===')
        n=prot[ch]['neutral']
        for lab,z in prot[ch].items():
            print(lab,'z_norm=',float(z.norm()),'delta_to_neutral=',float((z-n).norm()))

    if args.decode:
        if not (args.expr_checkpoint and args.head_checkpoint and args.jaw_checkpoint):
            raise SystemExit('decode requires all three checkpoints')
        expr_m=load_cvae(args.expr_checkpoint,dev); head_m=load_cvae(args.head_checkpoint,dev); jaw_m=load_cvae(args.jaw_checkpoint,dev)
        st=json.loads(args.norm_stats.read_text()); mean=np.asarray(st['mean'],np.float32)[:56]; std=np.asarray(st['std'],np.float32)[:56]
        for lab in classes:
            with torch.no_grad():
                e=expr_m.decode(prot['expr'][lab][None].to(dev))[0].cpu().numpy().astype(np.float32)
                h=head_m.decode(prot['head'][lab][None].to(dev))[0].cpu().numpy().astype(np.float32)
                j=jaw_m.decode(prot['jaw'][lab][None].to(dev))[0].cpu().numpy().astype(np.float32)
            mn=np.concatenate([e,h,j],axis=1)
            mr=mn*std[None,:]+mean[None,:]
            print('\n[decode]',lab)
            for k,v in stats(mr).items(): print(k,v)

if __name__=='__main__': main()
