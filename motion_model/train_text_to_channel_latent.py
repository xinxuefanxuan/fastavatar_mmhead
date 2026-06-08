#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from motion_model.channel_vae import ChannelTemporalVAE

DEFAULT_SCALES={
 'expr':{'smile':1.0,'mouth_open':0.5,'nod':0.0,'turn_left':0.0,'turn_right':0.0,'neutral':0.0},
 'head':{'turn_left':1.0,'turn_right':1.0,'nod':1.0,'smile':0.0,'mouth_open':0.0,'neutral':0.0},
 'jaw': {'mouth_open':1.0,'smile':0.3,'nod':0.0,'turn_left':0.0,'turn_right':0.0,'neutral':0.0},
}

class DS(Dataset):
    def __init__(self,droot:Path,split:str,classes:list[str]):
        rows=[json.loads(l) for l in (droot/f'{split}.jsonl').read_text().splitlines() if l.strip()]
        emb=np.load(droot/f'{split}_embeddings.npy').astype(np.float32)
        ids=json.loads((droot/f'{split}_sample_ids.json').read_text())
        id2i={sid:i for i,sid in enumerate(ids)}
        self.x=[]; self.y=[]; self.labels=[]
        for r in rows:
            sid=r['sample_id']
            if sid not in id2i: continue
            y=np.zeros((len(classes),),np.float32)
            labs=r.get('labels',[])
            for i,c in enumerate(classes):
                if c in labs: y[i]=1.0
            self.x.append(emb[id2i[sid]]); self.y.append(y); self.labels.append(labs)
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return torch.from_numpy(self.x[i]), torch.from_numpy(self.y[i])

class Net(nn.Module):
    def __init__(self,in_dim,classes,expr_d,head_d,jaw_d,h=512,layers=3,drop=0.1):
        super().__init__(); seq=[]; d=in_dim
        for _ in range(max(1,layers-1)): seq += [nn.Linear(d,h),nn.ReLU(inplace=True),nn.Dropout(drop)]; d=h
        self.trunk=nn.Sequential(*seq)
        self.expr=nn.Linear(d,expr_d); self.head=nn.Linear(d,head_d); self.jaw=nn.Linear(d,jaw_d); self.cls=nn.Linear(d,classes)
    def forward(self,x):
        h=self.trunk(x)
        return self.expr(h),self.head(h),self.jaw(h),self.cls(h)

def load_cvae(ckpt,device):
    c=torch.load(ckpt,map_location=device)
    m=ChannelTemporalVAE(int(c['input_dim']),int(c['target_len']),int(c['latent_dim']),int(c['hidden_dim'])).to(device)
    m.load_state_dict(c['model']); m.eval()
    for p in m.parameters(): p.requires_grad=False
    return m,c

def compose_targets(y,classes,protos,scales,device):
    B=y.shape[0]
    outs={}
    for ch in ['expr','head','jaw']:
        z0=protos[ch]['neutral'].to(device)
        z=z0[None,:].repeat(B,1)
        for i,c in enumerate(classes):
            dz=(protos[ch][c].to(device)-z0)[None,:]
            z = z + y[:,i:i+1]*float(scales[ch].get(c,0.0))*dz
        outs[ch]=z
    return outs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset_dir',type=Path,default=Path('outputs/mmhead_debug/prompt_augmented_primitive_dataset_v2'))
    ap.add_argument('--channel_prototypes',type=Path,default=Path('outputs/mmhead_debug/channel_prototypes_v1/channel_prototypes.pt'))
    ap.add_argument('--expr_checkpoint',type=Path,default=Path('outputs/mmhead_debug/channel_vae_v1/expr/best.pt'))
    ap.add_argument('--head_checkpoint',type=Path,default=Path('outputs/mmhead_debug/channel_vae_v1/head/best.pt'))
    ap.add_argument('--jaw_checkpoint',type=Path,default=Path('outputs/mmhead_debug/channel_vae_v1/jaw/best.pt'))
    ap.add_argument('--norm_stats',type=Path,default=Path('outputs/mmhead_debug/motion_dataset_v1_ae_debug/norm_stats.json'))
    ap.add_argument('--output_dir',type=Path,default=Path('outputs/mmhead_debug/text_to_channel_latent_v1'))
    ap.add_argument('--primitive_classes',type=str,default='turn_left,turn_right,nod,smile,mouth_open,neutral')
    ap.add_argument('--hidden_dim',type=int,default=512); ap.add_argument('--num_layers',type=int,default=3)
    ap.add_argument('--batch_size',type=int,default=128); ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--device',type=str,default='cuda')
    ap.add_argument('--eval_every',type=int,default=10); ap.add_argument('--channel_scales_json',type=Path,default=None)
    ap.add_argument('--latent_loss_weight',type=float,default=1.0); ap.add_argument('--decoded_loss_weight',type=float,default=1.0); ap.add_argument('--bce_loss_weight',type=float,default=1.0)
    args=ap.parse_args()
    dev=torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    classes=[x.strip() for x in args.primitive_classes.split(',') if x.strip()]

    scales={k:v.copy() for k,v in DEFAULT_SCALES.items()}
    if args.channel_scales_json and args.channel_scales_json.exists():
        cfg=json.loads(args.channel_scales_json.read_text())
        for ch in scales:
            for k,v in cfg.get(ch,{}).items(): scales[ch][k]=float(v)

    trds=DS(args.dataset_dir,'train',classes); vads=DS(args.dataset_dir,'val',classes)
    tr=DataLoader(trds,batch_size=args.batch_size,shuffle=True); va=DataLoader(vads,batch_size=args.batch_size)

    expr_m,expr_ck=load_cvae(args.expr_checkpoint,dev); head_m,head_ck=load_cvae(args.head_checkpoint,dev); jaw_m,jaw_ck=load_cvae(args.jaw_checkpoint,dev)
    proto=torch.load(args.channel_prototypes,map_location='cpu')
    protos={ch:{k:v.float() for k,v in proto['prototypes'][ch].items()} for ch in ['expr','head','jaw']}

    in_dim=trds.x[0].shape[0]
    net=Net(in_dim,len(classes),expr_ck['latent_dim'],head_ck['latent_dim'],jaw_ck['latent_dim'],args.hidden_dim,args.num_layers).to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=args.lr)

    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'config.json').write_text(json.dumps({k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},indent=2))
    f=(args.output_dir/'train_log.jsonl').open('w')
    best=1e9
    for ep in range(1,args.epochs+1):
        for mode,ld in [('train',tr),('val',va)]:
            net.train(mode=='train')
            acc=[]; sums={'tot':0,'lat':0,'dec':0,'bce':0,'n':0}
            for x,y in ld:
                x,y=x.to(dev),y.to(dev)
                ze,zh,zj,logits=net(x)
                tgt=compose_targets(y,classes,protos,scales,dev)
                lat=F.mse_loss(ze,tgt['expr'])+F.mse_loss(zh,tgt['head'])+F.mse_loss(zj,tgt['jaw'])
                de=expr_m.decode(ze); dh=head_m.decode(zh); dj=jaw_m.decode(zj)
                te=expr_m.decode(tgt['expr']); th=head_m.decode(tgt['head']); tj=jaw_m.decode(tgt['jaw'])
                dec=F.mse_loss(de,te)+10*F.mse_loss(dh,th)+10*F.mse_loss(dj,tj)
                bce=F.binary_cross_entropy_with_logits(logits,y)
                loss=args.latent_loss_weight*lat+args.decoded_loss_weight*dec+args.bce_loss_weight*bce
                if mode=='train': opt.zero_grad(); loss.backward(); opt.step()
                pred=(torch.sigmoid(logits)>0.5).float(); acc.append((pred==y).all(dim=1).float().mean().item())
                bs=x.size(0); sums['n']+=bs
                sums['tot']+=loss.item()*bs; sums['lat']+=lat.item()*bs; sums['dec']+=dec.item()*bs; sums['bce']+=bce.item()*bs
            for k in ['tot','lat','dec','bce']: sums[k]/=max(1,sums['n'])
            sums['acc']=float(np.mean(acc)) if acc else 0.0
            if mode=='train': trm=sums
            else: vam=sums
        row={'epoch':ep,'total_loss':trm['tot'],'latent_loss':trm['lat'],'decoded_loss':trm['dec'],'bce_loss':trm['bce'],'exact_match_acc':trm['acc'],'val_total':vam['tot'],'val_latent':vam['lat'],'val_decoded':vam['dec'],'val_bce':vam['bce'],'val_exact_match_acc':vam['acc']}
        f.write(json.dumps(row)+'\n'); f.flush(); print(ep,row['val_total'])
        ck={'model':net.state_dict(),'args':vars(args),'primitive_classes':classes,'expr_latent_dim':expr_ck['latent_dim'],'head_latent_dim':head_ck['latent_dim'],'jaw_latent_dim':jaw_ck['latent_dim'],'input_dim':in_dim}
        torch.save(ck,args.output_dir/'last.pt')
        if vam['tot']<best: best=vam['tot']; torch.save(ck,args.output_dir/'best.pt')
    f.close()

if __name__=='__main__': main()
