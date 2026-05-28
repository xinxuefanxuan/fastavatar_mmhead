#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from motion_model.models import TemporalConvVAE

DEFAULT_DEBUG_PROMPTS=["turn left","turn right","smile","turn left and smile"]

def read_jsonl(path: Path)->list[dict]:
    out=[]
    with path.open('r',encoding='utf-8') as f:
        for l in f:
            l=l.strip()
            if l: out.append(json.loads(l))
    return out

def build_id_to_embedding(emb_dir: Path)->dict[str,np.ndarray]:
    emb=np.load(emb_dir/'embeddings.npy').astype(np.float32)
    ids=json.loads((emb_dir/'sample_ids.json').read_text(encoding='utf-8'))
    if len(ids)!=emb.shape[0]: raise SystemExit('embedding rows != sample_ids')
    return {str(s):emb[i] for i,s in enumerate(ids)}

def build_label_map(path: Path|None, label_field:str)->dict[str,str]:
    if path is None: return {}
    m={}
    for r in read_jsonl(path):
        sid=str(r.get('sample_id',''))
        lb=r.get(label_field)
        if sid and isinstance(lb,str) and lb: m[sid]=lb
    return m

def parse_list(s:str|None)->list[str]:
    if not s: return []
    return [x.strip() for x in s.split(',') if x.strip()]

def weighted_motion_mse(pred,gt):
    w=torch.ones((1,1,gt.shape[-1]),dtype=gt.dtype,device=gt.device); w[...,50:53]=10; w[...,53:56]=10
    return ((pred-gt)**2*w).mean()

class TextLatentDataset(Dataset):
    def __init__(self,items): self.items=items
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        it=self.items[i]
        return {
            'embedding':torch.from_numpy(np.asarray(it['embedding'],dtype=np.float32)),
            'motion_norm':torch.from_numpy(np.asarray(it['motion_norm'],dtype=np.float32)),
            'z_mu':torch.from_numpy(np.asarray(it['z_mu'],dtype=np.float32)),
            'sample_id':it['sample_id'],'label':it.get('label','unknown'),'label_idx':int(it.get('label_idx',-1))}

class TextToLatentMLP(nn.Module):
    def __init__(self,in_dim,latent_dim,hidden_dim=512,num_layers=3,num_classes=0):
        super().__init__(); layers=[]; d=in_dim
        for _ in range(max(1,num_layers-1)):
            layers += [nn.Linear(d,hidden_dim), nn.ReLU(inplace=True)]; d=hidden_dim
        self.backbone=nn.Sequential(*layers)
        self.z_head=nn.Linear(d,latent_dim)
        self.cls_head=nn.Linear(d,num_classes) if num_classes>0 else None
    def forward(self,x):
        h=self.backbone(x); z=self.z_head(h); logits=self.cls_head(h) if self.cls_head is not None else None
        return z,logits

def build_items(rows,id2emb,vae,device,label_map,include_labels,max_per_label,max_samples,class_to_idx):
    items=[]; c=defaultdict(int)
    with torch.no_grad():
        for r in rows:
            sid=str(r.get('sample_id',''))
            if sid not in id2emb: continue
            label=label_map.get(sid,'unknown')
            if include_labels is not None and label not in include_labels: continue
            if max_per_label is not None and c[label]>=max_per_label: continue
            npz=np.load(Path(r['npz_path']),allow_pickle=True)
            m=np.asarray(npz['motion_norm'],dtype=np.float32)
            mu,_=vae.encode(torch.from_numpy(m[None,...]).to(device))
            z_mu=mu.mean(dim=2)[0].detach().cpu().numpy().astype(np.float32)
            items.append({'embedding':id2emb[sid],'motion_norm':m,'z_mu':z_mu,'sample_id':sid,'label':label,'label_idx':class_to_idx.get(label,-1)})
            c[label]+=1
            if max_samples is not None and len(items)>=max_samples: break
    return items

def make_balanced_sampler(items):
    labels=[it.get('label','unknown') for it in items]; c=Counter(labels)
    w=np.asarray([1.0/max(1,c[l]) for l in labels],dtype=np.float64)
    return WeightedRandomSampler(torch.from_numpy(w),num_samples=len(w),replacement=True)

def tensor_stats(t):
    return {'mean':float(t.mean().item()),'std':float(t.std(unbiased=False).item()),'norm_mean':float(t.norm(dim=1).mean().item())}

def summarize(items): return dict(sorted(Counter([it.get('label','unknown') for it in items]).items()))

def run_epoch(model,vae,loader,device,train,opt,use_label_loss,label_loss_weight):
    model.train() if train else model.eval()
    acc=defaultdict(float); n=0; zps=[]; zms=[]
    decs=defaultdict(float)
    with torch.set_grad_enabled(train):
        for b in loader:
            emb=b['embedding'].to(device); motion=b['motion_norm'].to(device); z_mu=b['z_mu'].to(device)
            z_pred,logits=model(emb)
            dec=vae.decode(z_pred[:,:,None].repeat(1,1,motion.shape[1]))
            latent=((z_pred-z_mu)**2).mean(); m_loss=weighted_motion_mse(dec,motion)
            cls=torch.tensor(0.0,device=device); cls_acc=torch.tensor(0.0,device=device)
            if use_label_loss and logits is not None:
                y=b['label_idx'].to(device); mask=(y>=0)
                if mask.any():
                    cls=F.cross_entropy(logits[mask],y[mask]); pred=logits.argmax(dim=1); cls_acc=(pred[mask]==y[mask]).float().mean()
            loss=latent+m_loss+(label_loss_weight*cls if use_label_loss else 0.0)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            acc['latent_mse']+=float(latent.item()); acc['decoded_motion_loss']+=float(m_loss.item()); acc['cls_loss']+=float(cls.item()); acc['cls_acc']+=float(cls_acc.item()); acc['total_loss']+=float(loss.item())
            expr=dec[:,:,:50].norm(dim=2); head=dec[:,:,50:53].norm(dim=2); jaw=dec[:,:,53:56].norm(dim=2); yaw=dec[:,:,51]
            decs['expr_norm_mean']+=float(expr.mean().item()); decs['expr_norm_max']+=float(expr.max().item()); decs['head_norm_mean']+=float(head.mean().item()); decs['head_norm_max']+=float(head.max().item()); decs['jaw_norm_mean']+=float(jaw.mean().item()); decs['jaw_norm_max']+=float(jaw.max().item()); decs['yaw_min']+=float(yaw.min().item()); decs['yaw_max']+=float(yaw.max().item())
            zps.append(z_pred.detach().cpu()); zms.append(z_mu.detach().cpu()); n+=1
    metrics={k:v/max(1,n) for k,v in acc.items()}; decm={k:v/max(1,n) for k,v in decs.items()}
    zp=torch.cat(zps,dim=0); zm=torch.cat(zms,dim=0)
    zstats={'z_mu':tensor_stats(zm),'z_pred':tensor_stats(zp),'z_pred_var_mean':float(zp.var(dim=0,unbiased=False).mean().item()),'cosine_similarity':float(F.cosine_similarity(zp,zm,dim=1).mean().item())}
    return metrics,zstats,decm

def eval_prompts(model,vae,prompts,device,encoder,target_len=64):
    if not prompts: return {}
    embs=encoder.encode(prompts,convert_to_numpy=True,batch_size=min(64,len(prompts)),normalize_embeddings=False)
    out={}
    with torch.no_grad():
        for p,e in zip(prompts,embs):
            z,_=model(torch.from_numpy(np.asarray(e,dtype=np.float32)).to(device).unsqueeze(0))
            dec=vae.decode(z[:,:,None].repeat(1,1,target_len))[0]
            expr=dec[:,:50].norm(dim=1); head=dec[:,50:53].norm(dim=1); jaw=dec[:,53:56].norm(dim=1); yaw=dec[:,51]
            out[p]={'expr_norm_mean':float(expr.mean().item()),'expr_norm_max':float(expr.max().item()),'head_norm_mean':float(head.mean().item()),'head_norm_max':float(head.max().item()),'jaw_norm_mean':float(jaw.mean().item()),'jaw_norm_max':float(jaw.max().item()),'yaw_min':float(yaw.min().item()),'yaw_max':float(yaw.max().item())}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train_manifest',type=Path,required=True); ap.add_argument('--val_manifest',type=Path,required=True); ap.add_argument('--text_embeddings_dir',type=Path,required=True)
    ap.add_argument('--vae_checkpoint',type=Path,required=True); ap.add_argument('--norm_stats',type=Path,required=True); ap.add_argument('--output_dir',type=Path,required=True)
    ap.add_argument('--latent_dim',type=int,default=64); ap.add_argument('--hidden_dim',type=int,default=512); ap.add_argument('--num_layers',type=int,default=3)
    ap.add_argument('--batch_size',type=int,default=128); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--device',type=str,default='cuda')
    ap.add_argument('--eval_every',type=int,default=10); ap.add_argument('--debug_eval_prompts',type=str,default=','.join(DEFAULT_DEBUG_PROMPTS))
    ap.add_argument('--max_train_samples',type=int,default=None); ap.add_argument('--max_val_samples',type=int,default=None)
    ap.add_argument('--label_jsonl',type=Path,default=None); ap.add_argument('--include_labels',type=str,default=None); ap.add_argument('--max_per_label',type=int,default=None); ap.add_argument('--balanced_sampler',action='store_true')
    ap.add_argument('--use_label_loss',action='store_true'); ap.add_argument('--label_field',type=str,default='label'); ap.add_argument('--label_loss_weight',type=float,default=1.0); ap.add_argument('--label_classes',type=str,default='turn_left,turn_right,nod,smile,mouth_open,neutral,other')
    args=ap.parse_args()

    from sentence_transformers import SentenceTransformer
    tr_rows=read_jsonl(args.train_manifest); va_rows=read_jsonl(args.val_manifest)
    tr_emb=build_id_to_embedding(args.text_embeddings_dir/'train'); va_emb=build_id_to_embedding(args.text_embeddings_dir/'val')
    label_map=build_label_map(args.label_jsonl,args.label_field)
    include_labels=set(parse_list(args.include_labels)) if args.include_labels else None
    label_classes=parse_list(args.label_classes); class_to_idx={c:i for i,c in enumerate(label_classes)}
    print(f"[Debug] max_train_samples={args.max_train_samples}"); print(f"[Debug] max_val_samples={args.max_val_samples}")
    print(f"[Labels] label_classes={label_classes}"); print(f"[Labels] class_to_idx={class_to_idx}")

    ckpt=torch.load(args.vae_checkpoint,map_location=args.device); latent_ckpt=int(ckpt.get('args',{}).get('latent_dim',args.latent_dim)); latent_dim=int(args.latent_dim or latent_ckpt)
    vae=TemporalConvVAE(in_dim=56,latent_dim=latent_ckpt).to(args.device); vae.load_state_dict(ckpt['model']); vae.eval()
    for p in vae.parameters(): p.requires_grad=False

    tr_items=build_items(tr_rows,tr_emb,vae,args.device,label_map,include_labels,args.max_per_label,args.max_train_samples,class_to_idx)
    va_items=build_items(va_rows,va_emb,vae,args.device,label_map,include_labels,args.max_per_label,args.max_val_samples,class_to_idx)
    if len(tr_items)==0 or len(va_items)==0: raise SystemExit('empty dataset after sample_id matching/filtering')
    print('[LabelDist][train]',summarize(tr_items)); print('[LabelDist][val]',summarize(va_items))
    print(f"[Labels] overlap_train={sum(1 for x in tr_items if x['sample_id'] in label_map)} overlap_val={sum(1 for x in va_items if x['sample_id'] in label_map)}")

    ds_tr=TextLatentDataset(tr_items); ds_va=TextLatentDataset(va_items)
    emb_dim=ds_tr[0]['embedding'].numel(); num_classes=len(label_classes) if (args.label_jsonl is not None and args.use_label_loss) else 0
    model=TextToLatentMLP(emb_dim,latent_dim,args.hidden_dim,args.num_layers,num_classes=num_classes).to(args.device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    tr_sampler=make_balanced_sampler(tr_items) if args.balanced_sampler else None
    dl_tr=DataLoader(ds_tr,batch_size=args.batch_size,shuffle=(tr_sampler is None),sampler=tr_sampler); dl_va=DataLoader(ds_va,batch_size=args.batch_size,shuffle=False)

    debug_prompts=parse_list(args.debug_eval_prompts); debug_encoder=SentenceTransformer('/home/yuanyuhao/models/all-MiniLM-L6-v2',device=args.device)

    args.output_dir.mkdir(parents=True,exist_ok=True); (args.output_dir/'config.json').write_text(json.dumps(vars(args),indent=2,default=str),encoding='utf-8')
    best=1e18
    with (args.output_dir/'train_log.jsonl').open('w',encoding='utf-8') as lf:
        for ep in range(1,args.epochs+1):
            tr_metrics,tr_z,_=run_epoch(model,vae,dl_tr,args.device,True,opt,args.use_label_loss and num_classes>0,args.label_loss_weight)
            va_metrics,va_z,va_dec=run_epoch(model,vae,dl_va,args.device,False,opt,args.use_label_loss and num_classes>0,args.label_loss_weight)
            rec={'epoch':ep,'train':tr_metrics,'val':va_metrics,'train_z_stats':tr_z,'val_z_stats':va_z,'val_decoded_motion_stats':va_dec}
            if args.eval_every>0 and (ep%args.eval_every==0 or ep==1): rec['debug_eval_prompts']=eval_prompts(model,vae,debug_prompts,args.device,debug_encoder)
            lf.write(json.dumps(rec)+'\n'); lf.flush()
            print(f"epoch={ep} train_total={tr_metrics['total_loss']:.6f} val_total={va_metrics['total_loss']:.6f} train_cls={tr_metrics['cls_loss']:.6f}/{tr_metrics['cls_acc']:.4f} val_cls={va_metrics['cls_loss']:.6f}/{va_metrics['cls_acc']:.4f}")
            out={'model':model.state_dict(),'args':vars(args),'embedding_dim':emb_dim,'latent_dim':latent_dim,'vae_checkpoint':str(args.vae_checkpoint),'has_classifier':num_classes>0,'label_classes':label_classes,'class_to_idx':class_to_idx}
            torch.save(out,args.output_dir/'last.pt')
            if va_metrics['total_loss']<best: best=va_metrics['total_loss']; torch.save(out,args.output_dir/'best.pt')

if __name__=='__main__': main()
