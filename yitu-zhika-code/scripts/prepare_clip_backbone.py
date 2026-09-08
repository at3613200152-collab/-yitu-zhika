"""Download hash-verified generic OpenAI CLIP from OpenCLIP's official HF mapping."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.capsicum_job import atomic_json,check_space,file_digest,job_lock,resume_download

DEST=Path('D:/yitu-data/pretrained/clip')
REVISION='a6f597a30f7b82c51704746581f9a4e41421e878'
MODEL_SHA='e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31'
AUTHOR_REVISION='26b9bd644f65323fcddb68a86a2c0c5505c4bc81'


def run():
    DEST.mkdir(parents=True,exist_ok=True)
    with job_lock(DEST/'job.lock'):
        check_space(DEST)
        state={'pid':os.getpid(),'scope':'generic OpenAI CLIP only, no Nutrition5k-trained checkpoint'}
        def report(**values):
            state.update(values,updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(DEST/'status.json',state)
            print(json.dumps(values),flush=True)
        try:
            report(stage='source_configs')
            sources=[]
            for repository,revision,name,output_name in [
                ('timm/vit_base_patch32_clip_224.openai',REVISION,'open_clip_config.json','open_clip_config.json'),
                ('jc-builds/CalorieCLIP',AUTHOR_REVISION,'config.json','calorieclip_config.json'),
                ('jc-builds/CalorieCLIP',AUTHOR_REVISION,'calorie_clip.py','calorieclip_author_reference.txt')]:
                url=f'https://huggingface.co/{repository}/resolve/{revision}/{name}'
                path=DEST/output_name
                response=requests.get(url,timeout=(10,30))
                response.raise_for_status()
                if len(response.content)>100000:
                    raise ValueError('Unexpected config size')
                if path.exists() and path.read_bytes()!=response.content:
                    raise ValueError('Pinned source file differs')
                if not path.exists():
                    path.write_bytes(response.content)
                sources.append({'url':url,'file':str(path),'sha256':file_digest(path)})
            url=f'https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/{REVISION}/open_clip_model.safetensors'
            model=DEST/'open_clip_model.safetensors'
            resume_download(url,model,605143284,report,max_seconds=3600,max_no_progress=5)
            report(stage='checksum')
            if file_digest(model)!=MODEL_SHA:
                raise ValueError('Generic CLIP safetensors hash mismatch')
            atomic_json(DEST/'provenance.json',{'backbone':'ViT-B-32 OpenAI','revision':REVISION,
                'sha256':MODEL_SHA,'bytes':605143284,'url':url,'source_configs':sources,
                'mapping':'open_clip.get_pretrained_cfg(ViT-B-32,openai).hf_hub',
                'calorieclip_author_revision':AUTHOR_REVISION,'food_finetuned_weights_used':False})
            report(stage='complete',sha256=MODEL_SHA)
        except BaseException as exc:
            report(stage='failed',error_type=type(exc).__name__)
            raise


if __name__=='__main__':
    run()
