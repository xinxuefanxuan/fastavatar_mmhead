#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from motion_model.models import TemporalConvVAE

DEFAULT_PROMPTS=["turn left","turn right","smile","open mouth","turn left and smile","turn right and smile"]

def read_jsonl(p:Path):
    out=[]
    with p.open('r',encoding='utf-8') as f:
        for l in f:
            l=l.strip()
            if l: out.append(json.loads(l))
    return out

def weighted_mse(a,b):
    w=torch.ones((1,1,a.shape[-1]),device=a.device,dtype=a.dtype); w[...,50:53]=10.; w[...,53:56]=10.
    return ((a-b)**2*w).mean()

def parse_csv(s): return [x.strip() for x in s.split(',') if x.strip()]

def load_proto(proto_path, classes, scales_json):
    p=torch.load(proto_path,map_location='cpu')
    z=np.stack([np.asarray(p['label_to_mean_mu'][c],dtype=np.float32) for c in classes],0)
    scales=np.ones((len(classes),),dtype=np.float32)
    if scales_json is not None and Path(scales_json).exists():
        cfg=json.loads(Path(scales_json).read_text(encoding='utf-8'))
        for i,c in enumerate(classes):
            if c in cfg: scales[i]=float(cfg[c])
    return torch.from_numpy(z), torch.from_numpy(scales)

class DS(Dataset):
    def __init__(self, rows, emb, sid, c2i):
        sid2i={str(s):i for i,s in enumerate(sid)}
        self.items=[]
        for r in rows:
            s=str(r['sample_id'])
            if s not in sid2i: continue
            y=np.zeros((len(c2i),),dtype=np.float32)
            for lb in r.get('labels',[]):
                if lb in c2i: y[c2i[lb]]=1.0
            motion=None
            if 'npz_path' in r and r['npz_path']:
                p=Path(r['npz_path'])
                if p.exists():
                    npz=np.load(p,allow_pickle=True)
                    motion=np.asarray(npz['motion_norm'],dtype=np.float32)
            self.items.append({'emb':emb[sid2i[s]].astype(np.float32),'y':y,'motion':motion,'sample_id':s,'source':r.get('source','unknown')})
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]

def collate(batch):
    b=len(batch)
    em=torch.from_numpy(np.stack([x['emb'] for x in batch],0)).float()
    y=torch.from_numpy(np.stack([x['y'] for x in batch],0)).float()
    has_motion=torch.tensor([x['motion'] is not None for x in batch],dtype=torch.bool)
    motion=None
    if has_motion.any():
        t=batch[next(i for i,x in enumerate(batch) if x['motion'] is not None)]['motion'].shape[0]
        motion=np.zeros((b,t,56),dtype=np.float32)
        for i,x in enumerate(batch):
            if x['motion'] is not None: motion[i]=x['motion']
        motion=torch.from_numpy(motion).float()
    return {'emb':em,'y':y,'motion':motion,'has_motion':has_motion}

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512, layers=3, dropout=0.1):
        super().__init__(); seq=[]; d=in_dim
        for _ in range(max(1,layers-1)):
            seq += [nn.Linear(d,hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]; d=hidden
        seq += [nn.Linear(d,out_dim)]
        self.net=nn.Sequential(*seq)
    def forward(self,x): return self.net(x)

def apply_topk(scores, topk):
    if topk is None or topk<=0 or topk>=scores.shape[1]: return scores
    v,ix=torch.topk(scores,topk,dim=1); m=torch.zeros_like(scores).scatter(1,ix,1.0); return scores*m

def compose(scores,zbank,scales,classes):
    ni=classes.index('neutral'); z0=zbank[ni][None,:]
    dirs=(zbank-z0)*scales[:,None]
    return z0 + scores @ dirs

def metric(scores,y,th=0.5):
    pred=(scores>=th).float(); exact=(pred==y).all(dim=1).float().mean().item()
    pr,rc={},{}
    for i in range(y.shape[1]):
        tp=((pred[:,i]==1)&(y[:,i]==1)).sum().item(); fp=((pred[:,i]==1)&(y[:,i]==0)).sum().item(); fn=((pred[:,i]==0)&(y[:,i]==1)).sum().item()
        pr[i]=tp/max(1,tp+fp); rc[i]=tp/max(1,tp+fn)
    return exact,pr,rc

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset_dir',type=Path,default=Path('outputs/mmhead_debug/prompt_augmented_primitive_dataset_v1'))
    ap.add_argument('--prototype_path',type=Path,default=Path('outputs/mmhead_debug/primitive_prototypes_v1/prototypes.pt'))
    ap.add_argument('--vae_checkpoint',type=Path,default=Path('outputs/mmhead_debug/vae_debug_beta1e4/best.pt'))
    ap.add_argument('--norm_stats',type=Path,default=Path('outputs/mmhead_debug/motion_dataset_v1_ae_debug/norm_stats.json'))
    ap.add_argument('--prototype_scales_json',type=Path,default=Path('outputs/mmhead_debug/text_prototype_debug/prototype_scales_calibrated.json'))
    ap.add_argument('--output_dir',type=Path,default=Path('outputs/mmhead_debug/text_to_prototype_multilabel_v1'))
    ap.add_argument('--primitive_classes',type=str,default='turn_left,turn_right,nod,smile,mouth_open,neutral')
    ap.add_argument('--hidden_dim',type=int,default=512); ap.add_argument('--num_layers',type=int,default=3); ap.add_argument('--dropout',type=float,default=0.1)
    ap.add_argument('--batch_size',type=int,default=128); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--device',type=str,default='cuda'); ap.add_argument('--eval_every',type=int,default=10); ap.add_argument('--threshold',type=float,default=0.5); ap.add_argument('--topk',type=int,default=None)
    ap.add_argument('--bce_loss_weight',type=float,default=1.0); ap.add_argument('--latent_proto_loss_weight',type=float,default=1.0); ap.add_argument('--decoded_proto_loss_weight',type=float,default=1.0); ap.add_argument('--real_motion_loss_weight',type=float,default=0.0)
    args=ap.parse_args()

    classes=parse_csv(args.primitive_classes); c2i={c:i for i,c in enumerate(classes)}
    tr_rows=read_jsonl(args.dataset_dir/'train.jsonl'); va_rows=read_jsonl(args.dataset_dir/'val.jsonl')
    tr_emb=np.load(args.dataset_dir/'train_embeddings.npy').astype(np.float32); va_emb=np.load(args.dataset_dir/'val_embeddings.npy').astype(np.float32)
    tr_sid=json.loads((args.dataset_dir/'train_sample_ids.json').read_text()); va_sid=json.loads((args.dataset_dir/'val_sample_ids.json').read_text())

    ds_tr,ds_va=DS(tr_rows,tr_emb,tr_sid,c2i),DS(va_rows,va_emb,va_sid,c2i)
    dl_tr=DataLoader(ds_tr,batch_size=args.batch_size,shuffle=True,collate_fn=collate); dl_va=DataLoader(ds_va,batch_size=args.batch_size,shuffle=False,collate_fn=collate)

    model=MLP(tr_emb.shape[1],len(classes),args.hidden_dim,args.num_layers,args.dropout).to(args.device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    zbank,scales=load_proto(args.prototype_path,classes,args.prototype_scales_json); zbank=zbank.to(args.device); scales=scales.to(args.device)

    ck=torch.load(args.vae_checkpoint,map_location=args.device); vae=TemporalConvVAE(in_dim=56,latent_dim=int(ck.get('args',{}).get('latent_dim',64))).to(args.device); vae.load_state_dict(ck['model']); vae.eval()
    for p in vae.parameters(): p.requires_grad=False

    from sentence_transformers import SentenceTransformer
    dbg_enc=SentenceTransformer('/home/yuanyuhao/models/all-MiniLM-L6-v2',device=args.device)

    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'config.json').write_text(json.dumps(vars(args),indent=2,default=str),encoding='utf-8')
    best=1e18
    with (args.output_dir/'train_log.jsonl').open('w',encoding='utf-8') as f:
        for ep in range(1,args.epochs+1):
            logs={}
            for split,dl,train in [('train',dl_tr,True),('val',dl_va,False)]:
                if train: model.train()
                else: model.eval()
                acc=defaultdict(float); n=0; all_scores=[]; all_y=[]
                with torch.set_grad_enabled(train):
                    for b in dl:
                        e=b['emb'].to(args.device); y=b['y'].to(args.device); logits=model(e)
                        scores=torch.sigmoid(logits); scores=apply_topk(scores,args.topk)
                        z_pred=compose(scores,zbank,scales,classes)
                        z_tgt=compose(y,zbank,scales,classes)
                        dec_pred=vae.decode(z_pred[:,:,None].repeat(1,1,64)); dec_tgt=vae.decode(z_tgt[:,:,None].repeat(1,1,64))
                        bce=F.binary_cross_entropy_with_logits(logits,y)
                        lproto=((z_pred-z_tgt)**2).mean(); dproto=weighted_mse(dec_pred.transpose(1,2),dec_tgt.transpose(1,2))
                        rm=torch.tensor(0.,device=args.device)
                        if args.real_motion_loss_weight>0 and b['motion'] is not None and b['has_motion'].any():
                            m=b['motion'].to(args.device); mask=b['has_motion'].to(args.device)
                            rm=weighted_mse(dec_pred.transpose(1,2)[mask],m[mask])
                        loss=args.bce_loss_weight*bce+args.latent_proto_loss_weight*lproto+args.decoded_proto_loss_weight*dproto+args.real_motion_loss_weight*rm
                        if train:
                            opt.zero_grad(); loss.backward(); opt.step()
                        acc['total_loss']+=loss.item(); acc['bce_loss']+=bce.item(); acc['latent_proto_loss']+=lproto.item(); acc['decoded_proto_loss']+=dproto.item(); acc['real_motion_loss']+=rm.item(); n+=1
                        all_scores.append(scores.detach().cpu()); all_y.append(y.detach().cpu())
                scores=torch.cat(all_scores,0); yy=torch.cat(all_y,0)
                exact,pr,rc=metric(scores,yy,args.threshold)
                logs[split]={k:v/max(1,n) for k,v in acc.items()}
                logs[split]['exact_match_acc']=exact
                logs[split]['per_class_precision']={classes[i]:pr[i] for i in pr}
                logs[split]['per_class_recall']={classes[i]:rc[i] for i in rc}
                logs[split]['mean_pred_score']={classes[i]:float(scores[:,i].mean().item()) for i in range(len(classes))}

            rec={'epoch':ep,'train':logs['train'],'val':logs['val']}
            if args.eval_every>0 and (ep%args.eval_every==0 or ep==1):
                prompts=DEFAULT_PROMPTS; em=np.asarray(dbg_enc.encode(prompts,convert_to_numpy=True,normalize_embeddings=False),dtype=np.float32)
                dbg={}
                with torch.no_grad():
                    lg=model(torch.from_numpy(em).to(args.device)); sc=torch.sigmoid(lg/max(args.threshold,1e-6)); sc=apply_topk(sc,args.topk)
                    z=compose(sc,zbank,scales,classes); dec=vae.decode(z[:,:,None].repeat(1,1,64)).transpose(1,2).cpu().numpy()
                for i,p in enumerate(prompts):
                    d=dec[i]; ex=np.linalg.norm(d[:,:50],axis=1); hd=np.linalg.norm(d[:,50:53],axis=1); jw=np.linalg.norm(d[:,53:56],axis=1); yw=d[:,51]
                    dbg[p]={'scores':{classes[j]:float(sc[i,j].item()) for j in range(len(classes))},'final_weights':{classes[j]:float((sc[i,j]*scales[j]).item()) for j in range(len(classes))},'expr_norm_mean':float(ex.mean()),'expr_norm_max':float(ex.max()),'head_norm_mean':float(hd.mean()),'head_norm_max':float(hd.max()),'jaw_norm_mean':float(jw.mean()),'jaw_norm_max':float(jw.max()),'yaw_min':float(yw.min()),'yaw_max':float(yw.max())}
                rec['debug_eval_prompts']=dbg
            f.write(json.dumps(rec)+'\n'); f.flush()
            print(f"epoch={ep} train={logs['train']['total_loss']:.6f} val={logs['val']['total_loss']:.6f} exact={logs['val']['exact_match_acc']:.4f}")
            ckpt={'model':model.state_dict(),'embedding_dim':tr_emb.shape[1],'primitive_classes':classes,'class_to_idx':c2i,'args':vars(args)}
            torch.save(ckpt,args.output_dir/'last.pt')
            if logs['val']['total_loss']<best:
                best=logs['val']['total_loss']; torch.save(ckpt,args.output_dir/'best.pt')

if __name__=='__main__': main()
