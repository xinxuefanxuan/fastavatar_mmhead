#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from motion_model.models import TemporalConvVAE

DEFAULT_DEBUG_PROMPTS=["turn left","turn right","smile","open mouth","turn left and smile"]

def read_jsonl(p:Path)->list[dict]:
    out=[]
    with p.open('r',encoding='utf-8') as f:
        for l in f:
            l=l.strip()
            if l: out.append(json.loads(l))
    return out

def parse_csv(s:str|None)->list[str]:
    if not s: return []
    return [x.strip() for x in s.split(',') if x.strip()]

def weighted_motion_mse(pred,gt):
    w=torch.ones((1,1,gt.shape[-1]),dtype=gt.dtype,device=gt.device); w[...,50:53]=10.; w[...,53:56]=10.
    return ((pred-gt)**2*w).mean()

def build_id2emb(dirp:Path)->dict[str,np.ndarray]:
    emb=np.load(dirp/'embeddings.npy').astype(np.float32)
    ids=json.loads((dirp/'sample_ids.json').read_text(encoding='utf-8'))
    return {str(s):emb[i] for i,s in enumerate(ids)}

def load_label_map(path:Path,label_field:str)->dict[str,str]:
    m={}
    for r in read_jsonl(path):
        sid=str(r.get('sample_id','')); lb=r.get(label_field)
        if sid and isinstance(lb,str) and lb: m[sid]=lb
    return m

def load_proto(path:Path, primitive_classes:list[str], scales_json:Path|None):
    p=torch.load(path,map_location='cpu')
    mean_map=p['label_to_mean_mu']
    for c in primitive_classes:
        if c not in mean_map: raise RuntimeError(f'prototype missing class: {c}')
    if 'neutral' not in primitive_classes: raise RuntimeError('primitive_classes must include neutral')
    z=torch.from_numpy(np.stack([np.asarray(mean_map[c],dtype=np.float32) for c in primitive_classes],axis=0))
    scale=np.ones((len(primitive_classes),),dtype=np.float32)
    if scales_json is not None:
        cfg=json.loads(scales_json.read_text(encoding='utf-8'))
        for i,c in enumerate(primitive_classes):
            if c in cfg: scale[i]=float(cfg[c])
    return z, torch.from_numpy(scale)

class DS(Dataset):
    def __init__(self, items): self.items=items
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        it=self.items[i]
        return {k:it[k] for k in it}

class TextToPrototypeMLP(nn.Module):
    def __init__(self,in_dim,latent_dim,num_classes,hidden_dim=512,num_layers=3,dropout=0.1,use_residual=False):
        super().__init__(); layers=[]; d=in_dim
        for _ in range(max(1,num_layers-1)):
            layers += [nn.Linear(d,hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d=hidden_dim
        self.backbone=nn.Sequential(*layers)
        self.logit_head=nn.Linear(d,num_classes)
        self.use_residual=use_residual
        self.res_head=nn.Linear(d,latent_dim) if use_residual else None
    def forward(self,x):
        h=self.backbone(x); lg=self.logit_head(h); rz=self.res_head(h) if self.res_head is not None else None
        return lg,rz

def compose_z(logits,residual,z_proto_bank,proto_scales,primitive_classes,temperature=1.0,hard_top1=False,topk=None,residual_scale=0.1):
    probs=torch.softmax(logits/max(temperature,1e-6),dim=1)
    if topk is not None and topk>0 and topk<probs.shape[1]:
        v,ix=torch.topk(probs,k=topk,dim=1); mask=torch.zeros_like(probs).scatter(1,ix,1.0); probs=probs*mask; probs=probs/(probs.sum(dim=1,keepdim=True)+1e-8)
    if hard_top1:
        i=torch.argmax(probs,dim=1); probs=torch.zeros_like(probs).scatter(1,i[:,None],1.0)
    ni=primitive_classes.index('neutral')
    z_neutral=z_proto_bank[ni][None,:]
    dirs=(z_proto_bank-z_neutral)*proto_scales[:,None]
    z_proto=z_neutral + probs @ dirs
    z_final=z_proto + float(residual_scale)*residual if residual is not None else z_proto
    return probs,z_proto,z_final

def build_items(rows,id2emb,label_map,class_to_idx,max_per_label,max_samples):
    out=[]; cnt=defaultdict(int)
    for r in rows:
        sid=str(r.get('sample_id',''))
        if sid not in id2emb or sid not in label_map: continue
        lb=label_map[sid]
        if lb not in class_to_idx: continue
        if max_per_label is not None and cnt[lb]>=max_per_label: continue
        npz=np.load(Path(r['npz_path']),allow_pickle=True); mn=np.asarray(npz['motion_norm'],dtype=np.float32)
        out.append({'embedding':torch.from_numpy(id2emb[sid]),'motion_norm':torch.from_numpy(mn),'label_idx':torch.tensor(class_to_idx[lb],dtype=torch.long),'label':lb,'sample_id':sid})
        cnt[lb]+=1
        if max_samples is not None and len(out)>=max_samples: break
    return out

def collate(batch):
    return {
        'embedding':torch.stack([x['embedding'] for x in batch],0).float(),
        'motion_norm':torch.stack([x['motion_norm'] for x in batch],0).float(),
        'label_idx':torch.stack([x['label_idx'] for x in batch],0).long(),
        'label':[x['label'] for x in batch],
        'sample_id':[x['sample_id'] for x in batch],
    }

def summarize(items): return dict(sorted(Counter([x['label'] for x in items]).items()))

def sampler(items):
    c=Counter([x['label'] for x in items]); w=np.asarray([1.0/c[x['label']] for x in items],dtype=np.float64)
    return WeightedRandomSampler(torch.from_numpy(w),num_samples=len(w),replacement=True)

def run_epoch(model,vae,loader,opt,train,args,z_proto_bank,proto_scales,primitive_classes,class_proto):
    model.train() if train else model.eval(); n=0
    acc=defaultdict(float); zpn=[]; zfn=[]; ent=[]; mp=[]; decs=defaultdict(float)
    with torch.set_grad_enabled(train):
        for b in loader:
            emb=b['embedding'].to(args.device); mn=b['motion_norm'].to(args.device); y=b['label_idx'].to(args.device)
            logits,res=model(emb)
            if res is not None: res=res.to(args.device)
            probs,zp,zf=compose_z(logits,res,z_proto_bank,proto_scales,primitive_classes,args.temperature,args.hard_top1,args.topk,args.residual_scale)
            dec=vae.decode(zf[:,:,None].repeat(1,1,mn.shape[1]))
            cls=F.cross_entropy(logits,y)
            lproto=((zf-class_proto[y])**2).mean()
            dloss=weighted_motion_mse(dec,mn)
            rl2=(res.pow(2).mean() if res is not None else torch.tensor(0.,device=args.device))
            loss=args.cls_loss_weight*cls + args.latent_proto_loss_weight*lproto + args.decoded_motion_loss_weight*dloss + args.residual_l2_weight*rl2
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            pred=torch.argmax(logits,1); cacc=(pred==y).float().mean()
            acc['total_loss']+=float(loss.item()); acc['cls_loss']+=float(cls.item()); acc['latent_proto_loss']+=float(lproto.item()); acc['decoded_motion_loss']+=float(dloss.item()); acc['residual_l2']+=float(rl2.item()); acc['cls_acc']+=float(cacc.item())
            zpn.append(zp.detach().cpu()); zfn.append(zf.detach().cpu());
            e=-(probs*(probs+1e-8).log()).sum(dim=1); ent.append(e.detach().cpu()); mp.append(torch.max(probs,dim=1).values.detach().cpu())
            ex=dec[:,:,:50].norm(dim=2); hd=dec[:,:,50:53].norm(dim=2); jw=dec[:,:,53:56].norm(dim=2); yw=dec[:,:,51]
            decs['expr_norm_mean']+=float(ex.mean().item()); decs['expr_norm_max']+=float(ex.max().item()); decs['head_norm_mean']+=float(hd.mean().item()); decs['head_norm_max']+=float(hd.max().item()); decs['jaw_norm_mean']+=float(jw.mean().item()); decs['jaw_norm_max']+=float(jw.max().item()); decs['yaw_min']+=float(yw.min().item()); decs['yaw_max']+=float(yw.max().item())
            n+=1
    m={k:v/max(1,n) for k,v in acc.items()}; dm={k:v/max(1,n) for k,v in decs.items()}
    zp=torch.cat(zpn,0); zf=torch.cat(zfn,0); ent=torch.cat(ent,0); mp=torch.cat(mp,0)
    zs={'z_proto_norm_mean':float(zp.norm(dim=1).mean().item()),'z_proto_norm_std':float(zp.norm(dim=1).std(unbiased=False).item()),'z_final_norm_mean':float(zf.norm(dim=1).mean().item()),'z_final_norm_std':float(zf.norm(dim=1).std(unbiased=False).item()),'primitive_probs_entropy_mean':float(ent.mean().item()),'max_primitive_prob_mean':float(mp.mean().item())}
    return m,zs,dm

def debug_eval(model,vae,encoder,args,z_proto_bank,proto_scales,primitive_classes,prompts):
    out={}
    embs=encoder.encode(prompts,convert_to_numpy=True,normalize_embeddings=False)
    with torch.no_grad():
        for p,e in zip(prompts,embs):
            logits,res=model(torch.from_numpy(np.asarray(e,dtype=np.float32)).to(args.device).unsqueeze(0))
            probs,zp,zf=compose_z(logits,res,z_proto_bank,proto_scales,primitive_classes,args.temperature,args.hard_top1,args.topk,args.residual_scale)
            dec=vae.decode(zf[:,:,None].repeat(1,1,64))[0]
            pr=probs[0].detach().cpu().numpy(); idx=np.argsort(-pr)[:3]
            ex=dec[:,:50].norm(dim=1); hd=dec[:,50:53].norm(dim=1); jw=dec[:,53:56].norm(dim=1); yw=dec[:,51]
            out[p]={'top3':[{'class':primitive_classes[int(i)],'prob':float(pr[int(i)])} for i in idx],'expr_norm_mean':float(ex.mean().item()),'expr_norm_max':float(ex.max().item()),'head_norm_mean':float(hd.mean().item()),'head_norm_max':float(hd.max().item()),'jaw_norm_mean':float(jw.mean().item()),'jaw_norm_max':float(jw.max().item()),'yaw_min':float(yw.min().item()),'yaw_max':float(yw.max().item())}
            print(f"[DebugPrompt] {p} top3={out[p]['top3']}")
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train_manifest',type=Path,required=True); ap.add_argument('--val_manifest',type=Path,required=True); ap.add_argument('--text_embeddings_dir',type=Path,required=True)
    ap.add_argument('--label_jsonl',type=Path,required=True); ap.add_argument('--label_field',type=str,default='label')
    ap.add_argument('--prototype_path',type=Path,required=True); ap.add_argument('--vae_checkpoint',type=Path,required=True); ap.add_argument('--norm_stats',type=Path,required=True)
    ap.add_argument('--output_dir',type=Path,required=True); ap.add_argument('--primitive_classes',type=str,default='neutral,turn_left,turn_right,nod,smile,mouth_open')
    ap.add_argument('--latent_dim',type=int,default=64); ap.add_argument('--hidden_dim',type=int,default=512); ap.add_argument('--num_layers',type=int,default=3); ap.add_argument('--dropout',type=float,default=0.1)
    ap.add_argument('--batch_size',type=int,default=128); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--device',type=str,default='cuda')
    ap.add_argument('--temperature',type=float,default=1.0); ap.add_argument('--hard_top1',action='store_true'); ap.add_argument('--topk',type=int,default=None)
    ap.add_argument('--use_residual',action='store_true'); ap.add_argument('--residual_scale',type=float,default=0.1); ap.add_argument('--prototype_scales_json',type=Path,default=None)
    ap.add_argument('--cls_loss_weight',type=float,default=1.0); ap.add_argument('--latent_proto_loss_weight',type=float,default=1.0); ap.add_argument('--decoded_motion_loss_weight',type=float,default=1.0); ap.add_argument('--residual_l2_weight',type=float,default=0.01)
    ap.add_argument('--max_train_samples',type=int,default=None); ap.add_argument('--max_val_samples',type=int,default=None); ap.add_argument('--max_per_label',type=int,default=None); ap.add_argument('--balanced_sampler',action='store_true')
    ap.add_argument('--eval_every',type=int,default=10); ap.add_argument('--debug_eval_prompts',type=str,default=','.join(DEFAULT_DEBUG_PROMPTS))
    args=ap.parse_args()

    primitive_classes=parse_csv(args.primitive_classes); class_to_idx={c:i for i,c in enumerate(primitive_classes)}
    print(f'[Classes] {primitive_classes}'); print(f'[ClassesMap] {class_to_idx}')
    tr_rows=read_jsonl(args.train_manifest); va_rows=read_jsonl(args.val_manifest)
    tr_emb=build_id2emb(args.text_embeddings_dir/'train'); va_emb=build_id2emb(args.text_embeddings_dir/'val')
    label_map=load_label_map(args.label_jsonl,args.label_field)
    print(f'[Labels] loaded={len(label_map)}')

    tr_items=build_items(tr_rows,tr_emb,label_map,class_to_idx,args.max_per_label,args.max_train_samples)
    va_items=build_items(va_rows,va_emb,label_map,class_to_idx,args.max_per_label,args.max_val_samples)
    print(f'[Overlap] train={sum(1 for r in tr_rows if str(r.get("sample_id","")) in label_map)} val={sum(1 for r in va_rows if str(r.get("sample_id","")) in label_map)}')
    print(f'[FinalSize] train={len(tr_items)} val={len(va_items)}')
    print('[LabelDist][train]',summarize(tr_items)); print('[LabelDist][val]',summarize(va_items))
    if len(tr_items)==0 or len(va_items)==0: raise SystemExit('empty train/val set')

    zbank, pscale = load_proto(args.prototype_path, primitive_classes, args.prototype_scales_json)
    zbank=zbank.to(args.device); pscale=pscale.to(args.device)
    class_proto=zbank.clone()

    vck=torch.load(args.vae_checkpoint,map_location=args.device); vlatent=int(vck.get('args',{}).get('latent_dim',args.latent_dim))
    vae=TemporalConvVAE(in_dim=56,latent_dim=vlatent).to(args.device); vae.load_state_dict(vck['model']); vae.eval()
    for p in vae.parameters(): p.requires_grad=False

    ds_tr,ds_va=DS(tr_items),DS(va_items)
    emb_dim=int(ds_tr[0]['embedding'].numel())
    model=TextToPrototypeMLP(emb_dim,args.latent_dim,len(primitive_classes),args.hidden_dim,args.num_layers,args.dropout,args.use_residual).to(args.device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    dl_tr=DataLoader(ds_tr,batch_size=args.batch_size,shuffle=not args.balanced_sampler,sampler=(sampler(tr_items) if args.balanced_sampler else None),collate_fn=collate)
    dl_va=DataLoader(ds_va,batch_size=args.batch_size,shuffle=False,collate_fn=collate)

    from sentence_transformers import SentenceTransformer
    dbg_prompts=parse_csv(args.debug_eval_prompts); dbg_encoder=SentenceTransformer('/home/yuanyuhao/models/all-MiniLM-L6-v2',device=args.device)

    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'config.json').write_text(json.dumps(vars(args),indent=2,default=str),encoding='utf-8')
    best=1e18
    with (args.output_dir/'train_log.jsonl').open('w',encoding='utf-8') as f:
        for ep in range(1,args.epochs+1):
            trm,trz,_=run_epoch(model,vae,dl_tr,opt,True,args,zbank,pscale,primitive_classes,class_proto)
            vam,vaz,vds=run_epoch(model,vae,dl_va,opt,False,args,zbank,pscale,primitive_classes,class_proto)
            rec={'epoch':ep,'train':trm,'val':vam,'z_stats':{'train':trz,'val':vaz},'val_decoded_motion_stats':vds}
            if args.eval_every>0 and (ep%args.eval_every==0 or ep==1): rec['debug_eval_prompts']=debug_eval(model,vae,dbg_encoder,args,zbank,pscale,primitive_classes,dbg_prompts)
            f.write(json.dumps(rec)+'\n'); f.flush()
            print(f"epoch={ep} train_total={trm['total_loss']:.6f} val_total={vam['total_loss']:.6f} train_cls={trm['cls_acc']:.4f} val_cls={vam['cls_acc']:.4f}")
            ck={'model':model.state_dict(),'args':vars(args),'embedding_dim':emb_dim,'latent_dim':args.latent_dim,'primitive_classes':primitive_classes,'class_to_idx':class_to_idx,'prototype_path':str(args.prototype_path),'use_residual':args.use_residual}
            torch.save(ck,args.output_dir/'last.pt')
            if vam['total_loss']<best:
                best=vam['total_loss']; torch.save(ck,args.output_dir/'best.pt')

if __name__=='__main__': main()
