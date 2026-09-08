"""Audited-overhead RGB multitask baseline, independent of historical checkpoints.

An internal control, NOT a reproduction of CalorieCLIP/Oatsty. Test is read only
after validation-selected training completes. Checkpoints never replace Demo.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.capsicum_job import atomic_json, check_space, check_stop, file_digest, job_lock
from src.models.resnet_multitask import ResNetMultiTask

MANIFEST = ROOT/'results/meal_official_v1/manifest.json'
MANIFEST_SHA = 'f1e293b826c98e517c3d1179871ec0a10d8ee471652b5d2e6f89f9ed1c755daa'
OUTPUT = ROOT/'results/meal_rgb_official_v1'
WEIGHTS = ROOT/'checkpoints/meal_rgb_official_v1'
TARGET_NAMES = ['calories','mass']
MEAN = [0.485,0.456,0.406]
STD = [0.229,0.224,0.225]


BASE_SEED = 42

def set_base_seed(s):
    global BASE_SEED
    BASE_SEED = s

def seed_epoch(epoch):
    seed = BASE_SEED + epoch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def should_flip(dish_id,epoch):
    return int(hashlib.sha256(f'{BASE_SEED}:{epoch}:{dish_id}'.encode()).hexdigest()[:8],16) < 2**31


class MealDataset(Dataset):
    def __init__(self,manifest,split,epoch=0):
        self.rows = [r for r in manifest['rows'] if r['split']==split]
        self.split,self.epoch = split,epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row = self.rows[index]
        with Image.open(ROOT/row['image']) as source:
            image = source.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
        if self.split=='train' and should_flip(row['dish_id'],self.epoch):
            image = ImageOps.mirror(image)
        rgb = TF.normalize(TF.to_tensor(image),MEAN,STD)
        return rgb,torch.tensor(row['targets'],dtype=torch.float32),row['category_idx'],row['dish_id']


class MealNet(nn.Module):
    def __init__(self,manifest,pretrained=True):
        super().__init__()
        self.network = ResNetMultiTask(num_classes=len(manifest['category_to_idx']),
            input_channels=3,pretrained=pretrained,num_regression_targets=2)
        self.register_buffer('target_center',torch.tensor(manifest['target_stats']['mean'],dtype=torch.float32))
        self.register_buffer('target_scale',torch.tensor(manifest['target_stats']['std'],dtype=torch.float32))

    def forward(self,images):
        output = self.network(images)
        # Output contract is always physical kcal/g, not hidden z-scores.
        return output['logits'],output['nutrition'].float()*self.target_scale+self.target_center


def losses(logits,prediction,target,classes,scale):
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Non-finite regression tensors')
    reg = ((prediction-target).abs()/scale).mean()
    cls = nn.functional.cross_entropy(logits.float(),classes)
    total = reg+0.2*cls
    if not torch.isfinite(total):
        raise FloatingPointError('Non-finite loss')
    return total,reg


def regression_metrics(target,prediction):
    target,prediction = np.asarray(target,dtype=np.float64),np.asarray(prediction,dtype=np.float64)
    if target.ndim!=2 or target.shape!=prediction.shape or target.shape[1]!=2:
        raise ValueError('Expected N x 2 targets/predictions')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Non-finite metric inputs')
    result = {}
    for i,name in enumerate(TARGET_NAMES):
        y,p = target[:,i],prediction[:,i]
        err = p-y
        nonzero = y>0
        variance = float(((y-y.mean())**2).sum())
        mae = float(np.abs(err).mean())
        result[name] = {'n':len(y),'mae':mae,'rmse':float(np.sqrt((err**2).mean())),
            'r2':float(1-(err**2).sum()/variance) if variance>0 else None,
            'mape_nonzero_percent':float(np.abs(err[nonzero]/y[nonzero]).mean()*100) if nonzero.any() else None,
            'mape_n':int(nonzero.sum()),'mae_over_mean_target_percent':mae/float(y.mean())*100 if y.mean()>0 else None,
            'negative_predictions':int((p<0).sum())}
    return result


def epoch_pass(model,manifest,split,epoch,batch_size,report,optimizer=None,max_batches=None):
    training = optimizer is not None
    if training != (split=='train'):
        raise ValueError('Optimization is allowed only on training rows')
    model.train(training)
    loader = DataLoader(MealDataset(manifest,split,epoch),batch_size=batch_size,
        shuffle=training,num_workers=0,pin_memory=True,drop_last=False,
        generator=torch.Generator().manual_seed(BASE_SEED+epoch))
    total,reg_total,n = 0.,0.,0
    truths,predictions,pred_classes,true_classes,dish_ids = [],[],[],[],[]
    for batch,(images,target,classes,ids) in enumerate(loader,1):
        check_stop(OUTPUT)
        if max_batches and batch>max_batches:
            break
        images,target,classes = images.cuda(),target.cuda(),classes.cuda()
        if not torch.isfinite(images).all():
            raise FloatingPointError('Non-finite images')
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,prediction = model(images)
            loss,reg = losses(logits,prediction,target,classes,model.target_scale)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
        size = len(ids)
        n += size
        total += loss.item()*size
        reg_total += reg.item()*size
        truths.extend(target.detach().cpu().tolist())
        predictions.extend(prediction.detach().cpu().tolist())
        pred_classes.extend(logits.argmax(1).detach().cpu().tolist())
        true_classes.extend(classes.cpu().tolist())
        dish_ids.extend(ids)
        if batch==1 or batch%25==0:
            report(stage='training' if training else 'validation' if split=='val' else 'test',
                   epoch=epoch,batch=batch,batches=len(loader),loss=total/n,
                   allocated_gpu_gib=round(torch.cuda.max_memory_allocated()/1024**3,3))
    if not n:
        raise ValueError('Empty epoch')
    return {'loss':total/n,'reg_normalized_l1':reg_total/n,'n':n,
        'metrics':regression_metrics(truths,predictions),
        'coarse_category_accuracy':sum(a==b for a,b in zip(true_classes,pred_classes))/n}, \
        {'ids':dish_ids,'truths':truths,'predictions':predictions,'classes':true_classes,'pred_classes':pred_classes}


def save_checkpoint(path,value):
    check_space(ROOT)
    temporary = path.with_suffix('.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def run(args):
    set_base_seed(getattr(args, 'seed', 42))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    WEIGHTS.mkdir(parents=True,exist_ok=True)
    state = {'pid':os.getpid(),'protocol':'meal_rgb_official_v1','target_names':TARGET_NAMES}
    def report(**values):
        state.update(values,updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(OUTPUT/'status.json',state)
        print(json.dumps(values,ensure_ascii=False),flush=True)
    with job_lock(OUTPUT/'job.lock'):
        try:
            check_stop(OUTPUT)
            check_space(ROOT)
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError('This protocol requires CUDA BF16; no implicit CPU/FP16 fallback')
            if file_digest(MANIFEST)!=MANIFEST_SHA:
                raise ValueError('Frozen manifest hash mismatch')
            manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
            if manifest['target_names']!=TARGET_NAMES:
                raise ValueError('Target names/order mismatch')
            pretrained = Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth'
            if not pretrained.exists():
                raise ValueError('ImageNet V2 cache missing; no unexpected weight download')
            if not file_digest(pretrained).startswith('11ad3fa6'):
                raise ValueError('ImageNet V2 cache hash mismatch')
            config = {'protocol':'meal_rgb_official_v1','manifest_sha256':MANIFEST_SHA,
                'imagenet_sha256':file_digest(pretrained),'seed':42,'epochs':args.epochs,
                'patience':8,'batch_size':args.batch_size,'size':[256,256],
                'optimizer':'AdamW','lr_backbone':1e-5,'lr_heads':1e-4,'weight_decay':1e-4,
                'loss':'mean(L1/ training target std) + 0.2 * coarse-category CE',
                'selection':'minimum validation normalized regression L1 only',
                'amp':'BF16; FP32 loss; strict finite gradients; clip norm 1',
                'augmentation':'dish/epoch hash horizontal flip; bilinear resize 256; ImageNet normalization',
                'target_names':TARGET_NAMES,'category_to_idx':manifest['category_to_idx'],
                'target_stats':manifest['target_stats'],'torch':str(torch.__version__),
                'torchvision':str(torchvision.__version__),'gpu':torch.cuda.get_device_name(),
                'code_hashes':{str(p.relative_to(ROOT)):file_digest(p) for p in
                    [Path(__file__),ROOT/'src/models/resnet_multitask.py']},
                'limitations':manifest['limitations']+['Internal RGB control, not external method reproduction.',
                    '11 derived coarse categories, not official food recognition classes.',
                    'Single-seed pilot; predictions are not clamped to improve reported error.']}
            protocol_path = OUTPUT/'protocol.json'
            if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=config:
                raise ValueError('Protocol changed; choose a new version, never silently resume')
            atomic_json(protocol_path,config)
            report(stage='verifying_images',counts=manifest['counts'])
            for i,row in enumerate(manifest['rows'],1):
                check_stop(OUTPUT)
                if file_digest(ROOT/row['image'])!=row['image_sha256']:
                    raise ValueError('Image changed: '+row['dish_id'])
                if i%600==0:
                    report(images_verified=i)
            torch.set_num_threads(4)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            seed_epoch(0)
            model = MealNet(manifest).cuda()
            optimizer = torch.optim.AdamW([
                {'params':model.network.features.parameters(),'lr':1e-5},
                {'params':list(model.network.classifier.parameters())+list(model.network.regressor.parameters()),'lr':1e-4}],weight_decay=1e-4)
            last_path,best_path = WEIGHTS/'last.pt',WEIGHTS/'best.pt'
            start,best,best_epoch,history,finished = 1,float('inf'),0,[],False
            if last_path.exists():
                checkpoint = torch.load(last_path,map_location='cpu',weights_only=True)
                if checkpoint['config']!=config:
                    raise ValueError('Checkpoint protocol mismatch')
                model.load_state_dict(checkpoint['model'],strict=True)
                optimizer.load_state_dict(checkpoint['optimizer'])
                start,best,best_epoch = checkpoint['epoch']+1,checkpoint['best'],checkpoint['best_epoch']
                history,finished = checkpoint['history'],checkpoint['training_complete']
                report(stage='resumed',epoch=start-1)
            else:
                if best_path.exists():
                    raise ValueError('Best checkpoint exists without resumable last checkpoint')
                # Real data/updates, followed by a full model/optimizer reset.
                smoke_train,_ = epoch_pass(model,manifest,'train',0,args.batch_size,report,optimizer,max_batches=2)
                smoke_val,_ = epoch_pass(model,manifest,'val',0,args.batch_size,report,max_batches=2)
                atomic_json(OUTPUT/'smoke.json',{'train':smoke_train,'val':smoke_val,'discarded_updates':True})
                del optimizer,model
                torch.cuda.empty_cache()
                seed_epoch(0)
                model = MealNet(manifest).cuda()
                optimizer = torch.optim.AdamW([
                    {'params':model.network.features.parameters(),'lr':1e-5},
                    {'params':list(model.network.classifier.parameters())+list(model.network.regressor.parameters()),'lr':1e-4}],weight_decay=1e-4)
            begun = time.monotonic()
            if not finished:
                for epoch in range(start,args.epochs+1):
                    if time.monotonic()-begun > args.max_hours*3600:
                        report(stage='paused_time_budget')
                        return
                    check_space(ROOT)
                    seed_epoch(epoch)
                    train,_ = epoch_pass(model,manifest,'train',epoch,args.batch_size,report,optimizer)
                    val,_ = epoch_pass(model,manifest,'val',epoch,args.batch_size,report)
                    improved = val['reg_normalized_l1']<best
                    if improved:
                        best,best_epoch = val['reg_normalized_l1'],epoch
                    history.append({'epoch':epoch,'train':train,'val':val})
                    finished = epoch==args.epochs or epoch-best_epoch>=8
                    checkpoint = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'epoch':epoch,'best':best,'best_epoch':best_epoch,'config':config,
                        'history':history,'training_complete':finished,'target_names':TARGET_NAMES}
                    if improved:
                        save_checkpoint(best_path,checkpoint)
                    save_checkpoint(last_path,checkpoint)
                    atomic_json(OUTPUT/'epochs.json',history)
                    report(stage='epoch_complete',epoch=epoch,best_epoch=best_epoch,
                           validation_calorie_mae=val['metrics']['calories']['mae'])
                    if finished:
                        break
            if not finished:
                raise RuntimeError('Training did not reach a valid stopping condition')
            test_path = OUTPUT/'test_metrics.json'
            if not test_path.exists():
                checkpoint = torch.load(best_path,map_location='cpu',weights_only=True)
                model.load_state_dict(checkpoint['model'],strict=True)
                metrics,details = epoch_pass(model,manifest,'test',best_epoch,args.batch_size,report)
                with (OUTPUT/'test_predictions.csv').open('w',encoding='utf-8',newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['dish_id','true_calories','true_mass','pred_calories','pred_mass','true_coarse_class','pred_coarse_class'])
                    for i,dish in enumerate(details['ids']):
                        writer.writerow([dish,*details['truths'][i],*details['predictions'][i],details['classes'][i],details['pred_classes'][i]])
                atomic_json(test_path,{'best_epoch':best_epoch,'checkpoint_sha256':file_digest(best_path),
                    'manifest_sha256':MANIFEST_SHA,'metrics':metrics,'limitations':config['limitations']})
            report(stage='complete',best_epoch=best_epoch,result=str(test_path))
        except BaseException as exc:
            report(stage='paused' if isinstance(exc,InterruptedError) else 'failed',
                   error=f'{type(exc).__name__}: {exc}')
            raise


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--max-hours',type=float,default=8.)
    parser.add_argument('--seed',type=int,default=42,help='Random seed for training (default 42)')
    args = parser.parse_args()
    if args.epochs<1 or args.batch_size<2 or not 0<args.max_hours<=12:
        parser.error('Invalid bounded training budget')
    run(args)
