"""From-scratch paired RGB-to-NIR on the hash-bound local HSI capture-day split."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.models.generator import UNetGenerator
from src.training.train_meal_official import seed_epoch,should_flip
from scripts.capsicum_job import atomic_json,check_space,check_stop,file_digest,job_lock

OUTPUT = ROOT/'results/hsi_unet_v2'
WEIGHTS = ROOT/'checkpoints/hsi_unet_v2'
MANIFEST = ROOT/'data/hsi_prepared_v2/manifest.json'


class Pairs(Dataset):
    def __init__(self,manifest,split,epoch=0):
        self.rows = [r for r in manifest['rows'] if r['split']==split]
        self.epoch,self.split = epoch,split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row = self.rows[index]
        array = np.load(ROOT/row['prepared'],allow_pickle=False)
        if array.shape!=(4,256,256) or not np.isfinite(array).all() or array.min()<0 or array.max()>1:
            raise ValueError('Invalid prepared pair: '+row['id'])
        if self.split=='train' and should_flip(row['id'],self.epoch):
            array = array[:,:,::-1].copy()
        tensor = torch.from_numpy(array.copy())*2-1
        return tensor[:3],tensor[3:],row['id']


def pass_epoch(model,manifest,split,epoch,batch,report,optimizer=None,limit=None):
    train = optimizer is not None
    if train!=(split=='train'):
        raise ValueError('Optimizer may only see training pairs')
    model.train(train)
    loader = DataLoader(Pairs(manifest,split,epoch),batch_size=batch,shuffle=train,
        num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(42+epoch))
    total,n,rows = 0.,0,[]
    for i,(rgb,nir,ids) in enumerate(loader,1):
        check_stop(OUTPUT)
        if limit and i>limit:
            break
        rgb,nir = rgb.cuda(),nir.cuda()
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred = model(rgb)
            loss = nn.functional.l1_loss(pred.float(),nir)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite generator loss')
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
        n += len(ids)
        total += loss.item()*len(ids)
        if not train:
            p,t = ((pred.float()+1)/2).detach().cpu().numpy(),((nir+1)/2).cpu().numpy()
            for key,a,b in zip(ids,p[:,0],t[:,0]):
                mse = float(np.mean((a-b)**2,dtype=np.float64))
                rows.append({'id':key,'l1_01':float(np.mean(np.abs(a-b))),
                    'psnr_db':float(-10*np.log10(max(mse,1e-12))),
                    'ssim':float(structural_similarity(b,a,data_range=1.))})
        if i==1 or i%10==0:
            report(stage='training' if train else split,epoch=epoch,batch=i,batches=len(loader),loss=total/n)
    metrics = {'l1_minus1_plus1':total/n,'n':n}
    if rows:
        metrics.update({key:float(np.mean([r[key] for r in rows])) for key in ('l1_01','psnr_db','ssim')})
    return metrics,rows


def save(path,value):
    check_space(ROOT)
    temp = path.with_suffix('.tmp')
    torch.save(value,temp)
    temp.replace(path)


def run(args):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    WEIGHTS.mkdir(parents=True,exist_ok=True)
    state = {'pid':os.getpid(),'protocol':'hsi_unet_v2'}
    def report(**values):
        state.update(values,updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(OUTPUT/'status.json',state)
        print(json.dumps(values),flush=True)
    with job_lock(OUTPUT/'job.lock'):
        try:
            check_space(ROOT)
            check_stop(OUTPUT)
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError('CUDA BF16 required')
            manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
            digest = file_digest(MANIFEST)
            alignment = json.loads((OUTPUT/'aligned_audit.json').read_text(encoding='utf-8'))
            if not alignment['passed'] or alignment['manifest_sha256']!=digest:
                raise ValueError('Missing passed alignment audit for this manifest')
            if manifest['counts']!={'train':93,'val':18,'test':33}:
                raise ValueError('Unexpected HSI splits')
            for row in manifest['rows']:
                if file_digest(ROOT/row['prepared'])!=row['prepared_sha256']:
                    raise ValueError('Prepared pair hash changed')
            config = {'protocol':'hsi_unet_v2','architecture':'UNetGenerator4','base_filters':64,
                'initialization':'scratch','epochs':args.epochs,'batch_size':8,'seed':42,'lr':2e-4,
                'optimizer':'Adam beta(.5,.999)','scheduler':'ReduceLROnPlateau factor .5 patience 5',
                'loss':'L1 on [-1,1]','amp':'BF16, float32 loss, finite gradient clip 1',
                'selection':'validation L1','patience':20,'manifest_sha256':digest,
                'preprocessing':{'input':manifest['input'],'target':manifest['target'],'nir_scale':manifest['nir_scale']},
                'torch':str(torch.__version__),'code_hashes':{str(p.relative_to(ROOT)):file_digest(p) for p in
                    [Path(__file__),ROOT/'src/models/generator.py',ROOT/'src/training/train_meal_official.py']},
                'metrics':'Mean per-image PSNR/SSIM on fixed [0,1] range, MSE floor 1e-12',
                'limitations':manifest['limitations']}
            protocol = OUTPUT/'protocol.json'
            if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8'))!=config:
                raise ValueError('Frozen generator protocol changed')
            atomic_json(protocol,config)
            if (OUTPUT/'test_metrics.json').exists():
                report(stage='complete',already_complete=True)
                return
            torch.set_num_threads(4)
            def initialize():
                seed_epoch(0)
                model = UNetGenerator(base_filters=64).cuda()
                optimizer = torch.optim.Adam(model.parameters(),lr=2e-4,betas=(.5,.999))
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=5)
                return model,optimizer,scheduler
            model,optimizer,scheduler = initialize()
            start,best,best_epoch,history,finished = 1,float('inf'),0,[],False
            last_path,best_path = WEIGHTS/'last.pt',WEIGHTS/'best.pt'
            if last_path.exists():
                last = torch.load(last_path,map_location='cpu',weights_only=True)
                if last['config']!=config:
                    raise ValueError('Checkpoint config mismatch')
                model.load_state_dict(last['G_state_dict'],strict=True)
                optimizer.load_state_dict(last['optimizer'])
                scheduler.load_state_dict(last['scheduler'])
                start,best,best_epoch = last['epoch']+1,last['best'],last['best_epoch']
                history,finished = last['history'],last['training_complete']
                del last
            else:
                train,_ = pass_epoch(model,manifest,'train',0,8,report,optimizer,limit=2)
                val,_ = pass_epoch(model,manifest,'val',0,8,report,limit=2)
                atomic_json(OUTPUT/'smoke.json',{'train':train,'val':val,'discarded_updates':True})
                del model,optimizer,scheduler
                torch.cuda.empty_cache()
                model,optimizer,scheduler = initialize()
            started = time.monotonic()
            budget_seconds = float(args.budget_hours)*3600.0
            if not finished:
                for epoch in range(start,args.epochs+1):
                    if budget_seconds>0 and time.monotonic()-started>budget_seconds:
                        report(stage='paused_budget',budget_hours=args.budget_hours)
                        return
                    check_space(ROOT)
                    seed_epoch(epoch)
                    train,_ = pass_epoch(model,manifest,'train',epoch,8,report,optimizer)
                    val,_ = pass_epoch(model,manifest,'val',epoch,8,report)
                    scheduler.step(val['l1_minus1_plus1'])
                    improved = val['l1_minus1_plus1']<best
                    if improved:
                        best,best_epoch = val['l1_minus1_plus1'],epoch
                    history.append({'epoch':epoch,'train':train,'val':val,'lr':optimizer.param_groups[0]['lr']})
                    finished = epoch==args.epochs or epoch-best_epoch>=20
                    value = {'epoch':epoch,'G_state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),'config':config,'best':best,'best_epoch':best_epoch,
                        'history':history,'training_complete':finished}
                    if improved:
                        save(best_path,value)
                    save(last_path,value)
                    atomic_json(OUTPUT/'epochs.json',history)
                    report(stage='epoch_complete',epoch=epoch,best_epoch=best_epoch,validation=val)
                    if finished:
                        break
            if not finished:
                raise RuntimeError('No valid training completion condition')
            best_state = torch.load(best_path,map_location='cpu',weights_only=True)
            model.load_state_dict(best_state['G_state_dict'],strict=True)
            test,rows = pass_epoch(model,manifest,'test',best_epoch,8,report)
            atomic_json(OUTPUT/'test_predictions.json',rows)
            atomic_json(OUTPUT/'test_metrics.json',{'best_epoch':best_epoch,'completed_epochs':history[-1]['epoch'],
                'test':test,'checkpoint_sha256':file_digest(best_path),'manifest_sha256':digest,'limitations':manifest['limitations']})
            report(stage='complete',best_epoch=best_epoch,test=test)
        except BaseException as exc:
            report(stage='paused' if isinstance(exc,InterruptedError) else 'failed',error=f'{type(exc).__name__}: {exc}')
            raise


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--budget-hours',type=float,default=8.0,help='Wall-clock training budget in hours (default 8.0; set 0 or negative to disable)')
    args = parser.parse_args()
    if not 1<=args.epochs<=200:
        parser.error('epochs must be between 1 and 200')
    run(args)
