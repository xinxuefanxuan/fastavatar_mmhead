#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from motion_model.channel_vae import ChannelTemporalVAE

DEFAULT_SCALES={
 'expr':{'smile':1.0,'mouth_open':0.5,'turn_left':0.0,'turn_right':0.0,'nod':0.0,'neutral':0.0},
 'head':{'turn_left':1.0,'turn_right':1.0,'nod':1.0,'smile':0.0,'mouth_open':0.0,'neutral':0.0},
 'jaw': {'mouth_open':1.5,'smile':0.3,'turn_left':0.0,'turn_right':0.0,'nod':0.0,'neutral':0.0},
}
DEBUG_PROMPTS=["turn left","turn right","smile","open mouth","mouth open","turn left and smile","turn right and smile","turn left and open mouth","turn right and open mouth"]

class DS(Dataset):
    def __init__(self, droot:Path, split:str, classes:list[str]):
        self.rows=[json.loads(l) for l in (droot/f'{split}.jsonl').read_text().splitlines() if l.strip()]
        self.emb=np.load(droot/f'{split}_embeddings.npy').astype(np.float32)
        ids=json.loads((droot/f'{split}_sample_ids.json').read_text())
        id2i={sid:i for i,sid in enumerate(ids)}
        keep=[]
        for r in self.rows:
            sid=r.get('sample_id')
            if sid in id2i: keep.append((r,id2i[sid]))
        self.rows=[x[0] for x in keep]; self.idxs=[x[1] for x in keep]
        self.classes=classes
        src_cnt={}; missing_path=0
        for r in self.rows:
            s=r.get('source','synthetic')
            src_cnt[s]=src_cnt.get(s,0)+1
            p=r.get('npz_path') or r.get('motion_path') or ""
            if not p: missing_path+=1
        print(f"[Dataset] split={split} rows={len(self.rows)}")
        print(f"[Dataset] split={split} source_distribution={src_cnt}")
        print(f"[Dataset] split={split} missing_path={missing_path} available_path={len(self.rows)-missing_path}")
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; x=self.emb[self.idxs[i]]
        y=np.zeros((len(self.classes),),np.float32)
        for j,c in enumerate(self.classes):
            if c in r.get('labels',[]): y[j]=1.0
        path = r.get('npz_path') or r.get('motion_path') or ""
        return torch.from_numpy(x), torch.from_numpy(y), r.get('source','synthetic'), path

def hybrid_collate_fn(batch):
    xs, ys, srcs, paths = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)
    srcs = list(srcs)
    paths = [p if p is not None else "" for p in paths]
    return xs, ys, srcs, paths

class HybridNet(nn.Module):
    def __init__(self,in_dim,n_cls,expr_d,head_d,jaw_d,h=512,layers=3,drop=0.1):
        super().__init__(); seq=[]; d=in_dim
        for _ in range(max(1,layers-1)): seq += [nn.Linear(d,h),nn.ReLU(inplace=True),nn.Dropout(drop)]; d=h
        self.trunk=nn.Sequential(*seq)
        self.logits=nn.Linear(d,n_cls)
        self.r_expr=nn.Linear(d,expr_d); self.r_head=nn.Linear(d,head_d); self.r_jaw=nn.Linear(d,jaw_d)
        self.g_expr=nn.Linear(d,1); self.g_head=nn.Linear(d,1); self.g_jaw=nn.Linear(d,1)
    def forward(self,x):
        h=self.trunk(x)
        return {
            'logits':self.logits(h),
            'r_expr':self.r_expr(h),'r_head':self.r_head(h),'r_jaw':self.r_jaw(h),
            'g_expr':torch.sigmoid(self.g_expr(h)),'g_head':torch.sigmoid(self.g_head(h)),'g_jaw':torch.sigmoid(self.g_jaw(h)),
        }

def load_cvae(ckpt,dev):
    c=torch.load(ckpt,map_location=dev)
    m=ChannelTemporalVAE(int(c['input_dim']),int(c['target_len']),int(c['latent_dim']),int(c['hidden_dim'])).to(dev)
    m.load_state_dict(c['model']); m.eval()
    for p in m.parameters(): p.requires_grad=False
    return m,c

def fit_len(x,t):
    if x.shape[0]>=t:return x[:t]
    return np.concatenate([x,np.repeat(x[-1:],t-x.shape[0],0)],0)

def compose_proto(scores, classes, protos, scales, dev):
    B=scores.shape[0]; out={}
    for ch in ['expr','head','jaw']:
        z0=protos[ch]['neutral'].to(dev)
        z=z0[None,:].repeat(B,1)
        for i,c in enumerate(classes):
            dz=(protos[ch][c].to(dev)-z0)[None,:]
            z = z + scores[:,i:i+1]*float(scales[ch].get(c,0.0))*dz
        out[ch]=z
    return out

def encode_real_targets(paths, cvaes, mean, std, target_len, dev):
    out=[]
    for p in paths:
        if p and Path(p).exists():
            a=np.load(Path(p),allow_pickle=False)
            if 'motion_norm' in a: m=np.asarray(a['motion_norm'],np.float32)
            else:
                raw=np.asarray(a['motion'] if 'motion' in a else a['motion_raw'],np.float32)
                m=(raw-mean[None,:])/np.maximum(std[None,:],1e-8)
            m=fit_len(m[:,:56],target_len)
            row={}
            for ch,(s,e) in {'expr':(0,50),'head':(50,53),'jaw':(53,56)}.items():
                x=torch.from_numpy(m[:,s:e][None].astype(np.float32)).to(dev)
                with torch.no_grad(): mu,_=cvaes[ch].encode(x)
                row[ch]=mu[0]
            out.append(row)
        else:
            out.append(None)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset_dir',type=Path,default=Path('outputs/mmhead_debug/prompt_augmented_primitive_dataset_v2'))
    ap.add_argument('--channel_prototypes',type=Path,default=Path('outputs/mmhead_debug/channel_prototypes_v2/channel_prototypes.pt'))
    ap.add_argument('--expr_checkpoint',type=Path,default=Path('outputs/mmhead_debug/channel_vae_v1/expr/best.pt'))
    ap.add_argument('--head_checkpoint',type=Path,default=Path('outputs/mmhead_debug/channel_vae_v1/head/best.pt'))
    ap.add_argument('--jaw_checkpoint',type=Path,default=Path('outputs/mmhead_debug/channel_vae_v1/jaw/best.pt'))
    ap.add_argument('--norm_stats',type=Path,default=Path('outputs/mmhead_debug/motion_dataset_v1_ae_debug/norm_stats.json'))
    ap.add_argument('--output_dir',type=Path,default=Path('outputs/mmhead_debug/text_to_channel_hybrid_v1'))
    ap.add_argument('--primitive_classes',type=str,default='turn_left,turn_right,nod,smile,mouth_open,neutral')
    ap.add_argument('--hidden_dim',type=int,default=512); ap.add_argument('--num_layers',type=int,default=3); ap.add_argument('--dropout',type=float,default=0.1)
    ap.add_argument('--batch_size',type=int,default=128); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--device',type=str,default='cuda'); ap.add_argument('--eval_every',type=int,default=10)
    ap.add_argument('--residual_scale',type=float,default=0.3); ap.add_argument('--residual_l2_weight',type=float,default=0.01)
    ap.add_argument('--latent_loss_weight',type=float,default=1.0); ap.add_argument('--decoded_loss_weight',type=float,default=1.0); ap.add_argument('--bce_loss_weight',type=float,default=1.0); ap.add_argument('--proto_loss_weight',type=float,default=0.5)
    ap.add_argument('--channel_scales_json',type=Path,default=None)
    args=ap.parse_args()
    dev=torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    classes=[x.strip() for x in args.primitive_classes.split(',') if x.strip()]

    scales={k:v.copy() for k,v in DEFAULT_SCALES.items()}
    if args.channel_scales_json and args.channel_scales_json.exists():
        cfg=json.loads(args.channel_scales_json.read_text())
        for ch in scales:
            for k,v in cfg.get(ch,{}).items(): scales[ch][k]=float(v)

    trds=DS(args.dataset_dir,'train',classes); vads=DS(args.dataset_dir,'val',classes)
    tr=DataLoader(trds,batch_size=args.batch_size,shuffle=True,collate_fn=hybrid_collate_fn)
    va=DataLoader(vads,batch_size=args.batch_size,collate_fn=hybrid_collate_fn)

    expr_m,eck=load_cvae(args.expr_checkpoint,dev); head_m,hck=load_cvae(args.head_checkpoint,dev); jaw_m,jck=load_cvae(args.jaw_checkpoint,dev)
    cvaes={'expr':expr_m,'head':head_m,'jaw':jaw_m}
    cp=torch.load(args.channel_prototypes,map_location='cpu')
    protos={ch:{k:v.float() for k,v in cp['prototypes'][ch].items()} for ch in ['expr','head','jaw']}
    st=json.loads(args.norm_stats.read_text()); mean=np.asarray(st['mean'],np.float32)[:56]; std=np.asarray(st['std'],np.float32)[:56]

    net=HybridNet(trds.emb.shape[1],len(classes),int(eck['latent_dim']),int(hck['latent_dim']),int(jck['latent_dim']),args.hidden_dim,args.num_layers,args.dropout).to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=args.lr)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'config.json').write_text(json.dumps({k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},indent=2))
    logf=(args.output_dir/'train_log.jsonl').open('w',encoding='utf-8'); best=1e9

    def step(loader,train):
        net.train(train); sums={'tot':0,'bce':0,'proto':0,'lat':0,'dec':0,'r2':0,'acc':[],'n':0}
        for x,y,src,paths in loader:
            x,y=x.to(dev),y.to(dev)
            out=net(x)
            scores=torch.sigmoid(out['logits'])
            z_proto_pred=compose_proto(scores,classes,protos,scales,dev)
            z_proto_tgt=compose_proto(y,classes,protos,scales,dev)
            # final pred
            zf={
                'expr':z_proto_pred['expr'] + args.residual_scale*out['g_expr']*out['r_expr'],
                'head':z_proto_pred['head'] + args.residual_scale*out['g_head']*out['r_head'],
                'jaw': z_proto_pred['jaw']  + args.residual_scale*out['g_jaw'] *out['r_jaw'],
            }
            # target latent uses real if available
            safe_paths=[]
            for s,p in zip(src,paths):
                if s=='real' and isinstance(p,str) and p and Path(p).exists():
                    safe_paths.append(p)
                else:
                    safe_paths.append("")
            real=encode_real_targets(safe_paths,cvaes,mean,std,int(eck['target_len']),dev)
            zt={k:z_proto_tgt[k].clone() for k in z_proto_tgt}
            for i,r in enumerate(real):
                if r is not None and src[i]=='real':
                    for ch in zt: zt[ch][i]=r[ch]

            bce=F.binary_cross_entropy_with_logits(out['logits'],y)
            proto=F.mse_loss(z_proto_pred['expr'],z_proto_tgt['expr'])+F.mse_loss(z_proto_pred['head'],z_proto_tgt['head'])+F.mse_loss(z_proto_pred['jaw'],z_proto_tgt['jaw'])
            lat=F.mse_loss(zf['expr'],zt['expr'])+F.mse_loss(zf['head'],zt['head'])+F.mse_loss(zf['jaw'],zt['jaw'])
            de,dh,dj=expr_m.decode(zf['expr']),head_m.decode(zf['head']),jaw_m.decode(zf['jaw'])
            te,th,tj=expr_m.decode(zt['expr']),head_m.decode(zt['head']),jaw_m.decode(zt['jaw'])
            dec=F.mse_loss(de,te)+10*F.mse_loss(dh,th)+10*F.mse_loss(dj,tj)
            r2=(out['r_expr'].pow(2).mean()+out['r_head'].pow(2).mean()+out['r_jaw'].pow(2).mean())
            loss=args.bce_loss_weight*bce+args.proto_loss_weight*proto+args.latent_loss_weight*lat+args.decoded_loss_weight*dec+args.residual_l2_weight*r2
            if train: opt.zero_grad(); loss.backward(); opt.step()
            bs=x.size(0); sums['n']+=bs
            sums['tot']+=loss.item()*bs; sums['bce']+=bce.item()*bs; sums['proto']+=proto.item()*bs; sums['lat']+=lat.item()*bs; sums['dec']+=dec.item()*bs; sums['r2']+=r2.item()*bs
            sums['acc'].append(((torch.sigmoid(out['logits'])>0.5)==y).all(dim=1).float().mean().item())
        for k in ['tot','bce','proto','lat','dec','r2']: sums[k]/=max(1,sums['n'])
        sums['acc']=float(np.mean(sums['acc'])) if sums['acc'] else 0.0
        return sums

    for ep in range(1,args.epochs+1):
        trm=step(tr,True); vam=step(va,False)
        row={'epoch':ep,'total_loss':trm['tot'],'latent_loss':trm['lat'],'decoded_loss':trm['dec'],'bce_loss':trm['bce'],'proto_loss':trm['proto'],'residual_l2':trm['r2'],'exact_match_acc':trm['acc'],'val_total':vam['tot'],'val_exact_match_acc':vam['acc']}
        # lightweight debug prompt logging
        if ep%args.eval_every==0:
            dbg=[]
            for p in DEBUG_PROMPTS:
                dbg.append({'prompt':p})
            row['debug_prompts']=dbg
        logf.write(json.dumps(row)+'\n'); logf.flush(); print(ep,vam['tot'])
        ck={'model':net.state_dict(),'args':vars(args),'primitive_classes':classes,'expr_latent_dim':int(eck['latent_dim']),'head_latent_dim':int(hck['latent_dim']),'jaw_latent_dim':int(jck['latent_dim']),'input_dim':trds.emb.shape[1]}
        torch.save(ck,args.output_dir/'last.pt')
        if vam['tot']<best: best=vam['tot']; torch.save(ck,args.output_dir/'best.pt')
    logf.close()

if __name__=='__main__': main()
