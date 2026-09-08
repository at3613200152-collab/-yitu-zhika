"""Matched RGB+predicted-NIR control using a newly audited HSI generator."""
import argparse
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.training.train_meal_official import MealNet,epoch_pass,seed_epoch,MEAN,STD,MANIFEST,MANIFEST_SHA
from src.models.generator import UNetGenerator
from scripts.capsicum_job import atomic_json,check_space,check_stop,file_digest,job_lock

OUTPUT=ROOT/'results/meal_nir_official_v1'
WEIGHTS=ROOT/'checkpoints/meal_nir_official_v1'
GENERATOR=ROOT/'checkpoints/hsi_unet_v2/best.pt'
GENERATOR_RESULT=ROOT/'results/hsi_unet_v2/test_metrics.json'


class NirMealNet(MealNet):
    def __init__(self,manifest,generator_state,pretrained=True):
        # The same 3-channel construction preserves shared initial head weights.
        super().__init__(manifest,pretrained)
        original=self.network.features[0]
        expanded=nn.Conv2d(4,64,7,stride=2,padding=3,bias=False)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:,:3].copy_(original.weight)
        self.network.features[0]=expanded
        self.network.input_channels=4
        self.generator=UNetGenerator(base_filters=64)
        self.generator.load_state_dict(generator_state,strict=True)
        self.generator.requires_grad_(False).eval()
        self.register_buffer('rgb_mean',torch.tensor(MEAN).view(1,3,1,1))
        self.register_buffer('rgb_std',torch.tensor(STD).view(1,3,1,1))

    def train(self,mode=True):
        super().train(mode)
        self.generator.eval()
        return self

    def forward(self,images):
        with torch.no_grad():
            rgb01=images*self.rgb_std+self.rgb_mean
            nir=self.generator(rgb01*2-1).float()
            if not torch.isfinite(nir).all() or nir.shape!=(len(images),1,256,256):
                raise FloatingPointError('Invalid frozen generator output')
            nir_normalized=((nir+1)/2-.485)/.229
        return super().forward(torch.cat([images,nir_normalized],dim=1))


def save(path,payload):
    check_space(ROOT)
    tmp=path.with_suffix('.tmp')
    torch.save(payload,tmp)
    tmp.replace(path)


def run(wait_for_generator=False):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    WEIGHTS.mkdir(parents=True,exist_ok=True)
    state={'pid':os.getpid(),'protocol':'meal_nir_official_v1'}
    def report(**values):
        check_stop(OUTPUT)
        state.update(values,updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(OUTPUT/'status.json',state)
        print(json.dumps(values),flush=True)
    with job_lock(OUTPUT/'job.lock'):
        try:
            wait_start=time.monotonic()
            while not GENERATOR_RESULT.exists():
                if not wait_for_generator:
                    raise ValueError('Audited HSI generator must finish first')
                source_status=ROOT/'results/hsi_unet_v2/status.json'
                if source_status.exists():
                    source=json.loads(source_status.read_text(encoding='utf-8'))
                    if source.get('stage') in ('failed','paused','paused_budget'):
                        raise RuntimeError('Generator paused or failed; downstream training stopped')
                if time.monotonic()-wait_start>8*3600:
                    raise TimeoutError('Generator wait budget exceeded')
                report(stage='waiting_for_generator')
                time.sleep(10)
            check_space(ROOT)
            if file_digest(MANIFEST)!=MANIFEST_SHA:
                raise ValueError('Frozen meal manifest mismatch')
            manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
            source_result=json.loads(GENERATOR_RESULT.read_text(encoding='utf-8'))
            generator_sha=file_digest(GENERATOR)
            if generator_sha!=source_result['checkpoint_sha256']:
                raise ValueError('Generator result/checkpoint hash mismatch')
            generator_checkpoint=torch.load(GENERATOR,map_location='cpu',weights_only=True)
            if generator_checkpoint['config']['manifest_sha256']!=source_result['manifest_sha256']:
                raise ValueError('Generator data provenance mismatch')
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError('CUDA BF16 required')
            baseline=json.loads((ROOT/'results/meal_rgb_official_v1/protocol.json').read_text(encoding='utf-8'))
            imagenet=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth'
            if file_digest(imagenet)!=baseline['imagenet_sha256']:
                raise ValueError('ImageNet initialization differs from RGB control')
            config={**baseline,'protocol':'meal_nir_official_v1','input_channels':4,
                'generator_sha256':generator_sha,'generator_manifest_sha256':source_result['manifest_sha256'],
                'generator_preprocessing':generator_checkpoint['config']['preprocessing'],
                'generator_policy':'Frozen eval mode; predicts after the same paired RGB horizontal flip',
                'nir_normalization':'([generator output+1]/2 - .485)/.229',
                'extra_channel_initialization':'zero; all shared RGB weights/heads match RGB construction',
                'torch':str(torch.__version__),
                'code_hashes':{str(p.relative_to(ROOT)):file_digest(p) for p in
                    [Path(__file__),ROOT/'src/training/train_meal_official.py',ROOT/'src/models/resnet_multitask.py',ROOT/'src/models/generator.py']},
                'limitations':baseline['limitations']+['Uses local 144-scan HSI-derived NIR; domain transfer to meal photos remains.']}
            if config['epochs']!=30 or config['batch_size']!=8:
                raise ValueError('Unexpected RGB control budget')
            protocol=OUTPUT/'protocol.json'
            if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8'))!=config:
                raise ValueError('Frozen paired protocol changed')
            atomic_json(protocol,config)
            if (OUTPUT/'test_metrics.json').exists():
                report(stage='complete',already_complete=True)
                return
            for row in manifest['rows']:
                if file_digest(ROOT/row['image'])!=row['image_sha256']:
                    raise ValueError('Meal image changed')
            torch.set_num_threads(4)
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            def initialize():
                seed_epoch(0)
                model=NirMealNet(manifest,generator_checkpoint['G_state_dict']).cuda()
                optimizer=torch.optim.AdamW([
                    {'params':model.network.features.parameters(),'lr':1e-5},
                    {'params':list(model.network.classifier.parameters())+list(model.network.regressor.parameters()),'lr':1e-4}],weight_decay=1e-4)
                return model,optimizer
            model,optimizer=initialize()
            start,best,best_epoch,history,finished=1,float('inf'),0,[],False
            best_path,last_path=WEIGHTS/'best.pt',WEIGHTS/'last.pt'
            if last_path.exists():
                last=torch.load(last_path,map_location='cpu',weights_only=True)
                if last['config']!=config:
                    raise ValueError('Checkpoint config mismatch')
                model.load_state_dict(last['model'],strict=True)
                optimizer.load_state_dict(last['optimizer'])
                start,best,best_epoch=last['epoch']+1,last['best'],last['best_epoch']
                history,finished=last['history'],last['training_complete']
                del last
            else:
                train,_=epoch_pass(model,manifest,'train',0,8,report,optimizer,max_batches=2)
                val,_=epoch_pass(model,manifest,'val',0,8,report,max_batches=2)
                atomic_json(OUTPUT/'smoke.json',{'train':train,'val':val,'discarded_updates':True})
                del model,optimizer
                torch.cuda.empty_cache()
                model,optimizer=initialize()
            begun=time.monotonic()
            if not finished:
                for epoch in range(start,31):
                    check_stop(OUTPUT)
                    if time.monotonic()-begun>8*3600:
                        report(stage='paused_budget')
                        return
                    seed_epoch(epoch)
                    train,_=epoch_pass(model,manifest,'train',epoch,8,report,optimizer)
                    val,_=epoch_pass(model,manifest,'val',epoch,8,report)
                    improved=val['reg_normalized_l1']<best
                    if improved:
                        best,best_epoch=val['reg_normalized_l1'],epoch
                    history.append({'epoch':epoch,'train':train,'val':val})
                    finished=epoch==30 or epoch-best_epoch>=8
                    checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                        'best':best,'best_epoch':best_epoch,'config':config,'history':history,
                        'training_complete':finished,'target_names':['calories','mass']}
                    if improved:
                        save(best_path,checkpoint)
                    save(last_path,checkpoint)
                    atomic_json(OUTPUT/'epochs.json',history)
                    report(stage='epoch_complete',epoch=epoch,best_epoch=best_epoch,validation_calorie_mae=val['metrics']['calories']['mae'])
                    if finished:
                        break
            if not finished:
                raise RuntimeError('Training did not complete')
            best_state=torch.load(best_path,map_location='cpu',weights_only=True)
            model.load_state_dict(best_state['model'],strict=True)
            metrics,details=epoch_pass(model,manifest,'test',best_epoch,8,report)
            with (OUTPUT/'test_predictions.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['dish_id','true_calories','true_mass','pred_calories','pred_mass','true_coarse_class','pred_coarse_class'])
                for i,dish in enumerate(details['ids']):
                    writer.writerow([dish,*details['truths'][i],*details['predictions'][i],details['classes'][i],details['pred_classes'][i]])
            atomic_json(OUTPUT/'test_metrics.json',{'best_epoch':best_epoch,'checkpoint_sha256':file_digest(best_path),
                'manifest_sha256':MANIFEST_SHA,'generator_sha256':generator_sha,'metrics':metrics,'limitations':config['limitations']})
            report(stage='complete',best_epoch=best_epoch)
        except BaseException as exc:
            state.update(stage='paused' if isinstance(exc,InterruptedError) else 'failed',
                         error=f'{type(exc).__name__}: {exc}',updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(OUTPUT/'status.json',state)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-for-generator',action='store_true')
    run(parser.parse_args().wait_for_generator)
