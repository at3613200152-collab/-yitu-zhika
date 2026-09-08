"""Frozen generic CLIP category similarities; no pseudo-labels or nutrition guesses."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import open_clip
from safetensors.torch import load_file
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.capsicum_job import atomic_json, file_digest, check_space, check_stop, job_lock
from scripts.prepare_clip_backbone import DEST, MODEL_SHA

MANIFEST = ROOT/'results/meal_expanded_v2/manifest.json'
MANIFEST_SHA = '32f4b53665c950fb86c54f842d3d134aaabfcaa5302e15ad7ed82e6898395afb'
OUT = ROOT/'results/clip_semantics_v2'
PROMPTS = {
    'dairy': 'a photo of dairy food, cheese, milk or yogurt',
    'dessert': 'a photo of a dessert or sweet food',
    'egg': 'a photo of eggs or an egg dish',
    'grain': 'a photo of rice, pasta, bread or grains',
    'meat': 'a photo of meat or poultry',
    'mixed': 'a photo of a mixed meal with several kinds of food',
    'other': 'a photo of an unidentified food',
    'sauce_condiment': 'a photo of sauce or condiments',
    'seafood': 'a photo of fish or seafood',
    'soup_stew': 'a photo of soup or stew',
    'vegetable': 'a photo of vegetables or fruit',
}


def read_cache():
    audit = json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    if not audit['passed'] or audit['manifest_sha256'] != MANIFEST_SHA or audit['model_sha256'] != MODEL_SHA:
        raise ValueError('Semantic cache provenance mismatch')
    if file_digest(OUT/'scores.pt') != audit['cache_sha256']:
        raise ValueError('Semantic cache changed')
    payload = torch.load(OUT/'scores.pt', map_location='cpu', weights_only=True)
    if payload['prompts'] != PROMPTS or payload['manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('Semantic prompt/manifest mismatch')
    scores = payload['scores']
    if any(t.shape != (11,) or not torch.isfinite(t).all() for t in scores.values()):
        raise ValueError('Invalid semantic score vectors')
    return scores, audit


class Images(Dataset):
    def __init__(self, rows):
        views = {}
        for row in rows:
            for view in row.get('views', [dict(path=row['image'], file_sha256=row['image_sha256'])]):
                views[view['file_sha256']] = view['path']
        self.views = sorted(views.items())
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711])])

    def __len__(self):
        return len(self.views)

    def __getitem__(self, index):
        sha, path = self.views[index]
        if file_digest(ROOT/path) != sha:
            raise ValueError('CLIP input image changed')
        with Image.open(ROOT/path) as source:
            image = self.transform(source.convert('RGB'))
        return sha, image


def run(verify_only=False):
    if verify_only:
        _, audit = read_cache()
        print(json.dumps(audit), flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with job_lock(OUT/'job.lock'):
        if (OUT/'audit.json').exists():
            return run(True)
        check_space(ROOT)
        if file_digest(MANIFEST) != MANIFEST_SHA or file_digest(DEST/'open_clip_model.safetensors') != MODEL_SHA:
            raise ValueError('Frozen manifest/CLIP model mismatch')
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        classes = sorted(manifest['category_to_idx'], key=manifest['category_to_idx'].get)
        if set(classes) != PROMPTS.keys():
            raise ValueError('Prompt categories mismatch')
        torch.set_num_threads(4)
        model = open_clip.create_model('ViT-B-32', pretrained=None, force_quick_gelu=True)
        model.load_state_dict(load_file(str(DEST/'open_clip_model.safetensors')), strict=True)
        model.requires_grad_(False).eval().cuda()
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        with torch.inference_mode():
            text = model.encode_text(tokenizer([PROMPTS[c] for c in classes]).cuda(), normalize=True).float()
        dataset = Images(manifest['rows'])
        scores = {}
        for shas, images in DataLoader(dataset, batch_size=16, num_workers=0):
            check_stop(OUT)
            with torch.inference_mode():
                features = model.encode_image(images.cuda(), normalize=True).float()
                values = (features @ text.T).cpu()
            for sha, value in zip(shas, values):
                scores[sha] = value.clone()
            state = dict(stage='caching_frozen_semantics', images=len(scores), total=len(dataset),
                         updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(OUT/'status.json', state)
            if len(scores) % 256 == 0:
                print(json.dumps(state), flush=True)
        payload = dict(scores=scores, prompts=PROMPTS, classes=classes, manifest_sha256=MANIFEST_SHA)
        temporary = OUT/'scores.tmp'
        torch.save(payload, temporary)
        temporary.replace(OUT/'scores.pt')
        audit = dict(passed=True, images=len(scores), model_sha256=MODEL_SHA,
                     manifest_sha256=MANIFEST_SHA, cache_sha256=file_digest(OUT/'scores.pt'),
                     source_code_sha256=file_digest(Path(__file__)), prompts=PROMPTS,
                     meaning='Frozen generic CLIP cosine similarities; not calibrated probabilities or labels',
                     transform='224 bicubic short-edge resize, center crop, CLIP normalization; original view',
                     used_nutrition_labels=False, fitting_performed=False,
                     limitation='No proven retention of ingredient details after center crop; no cooking/weight truth')
        atomic_json(OUT/'audit.json', audit)
        atomic_json(OUT/'status.json', dict(stage='complete', images=len(scores)))
        run(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    run(parser.parse_args().verify_only)
