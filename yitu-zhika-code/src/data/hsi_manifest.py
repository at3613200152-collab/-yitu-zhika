"""Capture-day split and audited RGB/859 nm preparation for the local HSI subset."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.capsicum_job import atomic_json,check_space,check_stop,file_digest,job_lock

DATA = ROOT/'data/HSIFoodIngr-64'
OUTPUT = ROOT/'data/hsi_prepared_v1'


def parse_header(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    if not text.lstrip().startswith('ENVI'):
        raise ValueError('Expected ENVI header')
    fields = {}
    pattern = r'^([A-Za-z][A-Za-z0-9 _-]*)[ \t]*=[ \t]*(\{[^}]*\}|[^\r\n]*)'
    for match in re.finditer(pattern,text,re.MULTILINE):
        key,value = match.group(1).strip().lower(),match.group(2).strip()
        if key in fields:
            raise ValueError('Duplicate ENVI field: '+key)
        fields[key] = value.strip('{}').strip()
    required = ['samples','lines','bands','data type','interleave','header offset','byte order','wavelength','acquisition date']
    if any(k not in fields for k in required):
        raise ValueError('Missing required ENVI metadata')
    for key in ['samples','lines','bands','data type','header offset','byte order']:
        fields[key] = int(fields[key])
    fields['interleave'] = fields['interleave'].lower()
    fields['wavelength'] = [float(x.strip()) for x in fields['wavelength'].split(',') if x.strip()]
    fields['capture_day'] = datetime.strptime(fields['acquisition date'],'%d-%m-%Y').date().isoformat()
    if (fields['data type']!=4 or fields['byte order'] not in (0,1)
        or fields['interleave'] not in ('bil','bip','bsq') or fields['header offset']<0):
        raise ValueError('Unsupported ENVI format; no guessed fallback')
    wl = np.asarray(fields['wavelength'])
    if len(wl)!=fields['bands'] or not np.isfinite(wl).all() or not (np.diff(wl)>0).all():
        raise ValueError('Invalid wavelength metadata')
    if min(fields[k] for k in ('samples','lines','bands'))<1:
        raise ValueError('Invalid cube dimensions')
    return fields


def read_nir(raw,header,target_nm=860.):
    h,w,b = header['lines'],header['samples'],header['bands']
    expected = h*w*b*4+header['header offset']
    if Path(raw).stat().st_size!=expected:
        raise ValueError('Raw size does not exactly match ENVI dimensions/offset')
    index = int(np.argmin(np.abs(np.array(header['wavelength'])-target_nm)))
    wavelength = header['wavelength'][index]
    if abs(wavelength-target_nm)>2.:
        raise ValueError('NIR wavelength not present within 2 nm')
    cube = np.memmap(raw,mode='r',dtype='<f4' if header['byte order']==0 else '>f4',offset=header['header offset'])
    if header['interleave']=='bil':
        band = cube.reshape(h,b,w)[:,index,:]
    elif header['interleave']=='bip':
        band = cube.reshape(h,w,b)[:,:,index]
    else:
        band = cube.reshape(b,h,w)[index]
    result = np.array(band,dtype=np.float32,copy=True)
    if not np.isfinite(result).all() or float(result.std())<1e-8:
        raise ValueError('Non-finite or constant NIR band')
    return result,index,wavelength


def split_days(days):
    ranked = sorted(set(days),key=lambda d:hashlib.sha256(('42:'+d).encode()).hexdigest())
    if len(ranked)<5:
        raise ValueError('Insufficient acquisition groups')
    return {d:'test' if i==0 else 'val' if i==1 else 'train' for i,d in enumerate(ranked)}


def training_scale(records):
    samples = [r['nir'][::4,::4].reshape(-1) for r in records if r['split']=='train']
    if not samples:
        raise ValueError('No training pixels')
    values = np.concatenate(samples)
    low,high = np.percentile(values,[.5,99.5]).tolist()
    if not np.isfinite([low,high]).all() or high-low<1e-6:
        raise ValueError('Invalid training-derived NIR range')
    return {'low':low,'high':high,'percentiles':[.5,99.5],
            'source':'training NIR only; every fourth row/column at native resolution',
            'sampled_pixels':len(values)}


def build():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with job_lock(OUTPUT/'job.lock'):
        check_space(ROOT)
        manifest_path = OUTPUT/'manifest.json'
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding='utf-8'))
            for row in previous['rows']:
                check_stop(OUTPUT)
                for field in ('hdr','raw','rgb_source','prepared'):
                    if file_digest(ROOT/row[field])!=row[field+'_sha256']:
                        raise ValueError('Frozen HSI file changed: '+row[field])
            print(json.dumps({'already_prepared':True,'counts':previous['counts'],'sha256':file_digest(manifest_path)}),flush=True)
            return
        headers = sorted(DATA.rglob('*.hdr'))
        parsed = [(path,parse_header(path)) for path in headers]
        days = split_days([h['capture_day'] for _,h in parsed])
        records,seen = [],{}
        for i,(path,header) in enumerate(parsed,1):
            check_stop(OUTPUT)
            check_space(ROOT)
            raw,rgb_path = path.with_suffix('.dat'),path.with_suffix('.png')
            if not raw.exists() or not rgb_path.exists():
                raise ValueError('Missing matched raw/PNG for '+str(path))
            nir,index,nm = read_nir(raw,header)
            with Image.open(rgb_path) as source:
                if source.size!=(header['samples'],header['lines']):
                    raise ValueError('RGB/HSI geometry mismatch')
                if 'A' in source.getbands() and source.getchannel('A').getextrema()!=(255,255):
                    raise ValueError('Nonopaque PNG needs explicit compositing policy')
                rgb = source.convert('RGB')
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                rgb = np.asarray(rgb.resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)/255.
            if digest in seen:
                raise ValueError('Duplicate RGB capture: '+str(path))
            seen[digest] = str(path)
            record = {'id':path.stem,'capture_day':header['capture_day'],'split':days[header['capture_day']],
                'hdr':str(path.relative_to(ROOT)),'raw':str(raw.relative_to(ROOT)),
                'rgb_source':str(rgb_path.relative_to(ROOT)),
                'hdr_sha256':file_digest(path),'raw_sha256':file_digest(raw),'rgb_source_sha256':file_digest(rgb_path),
                'rgb_pixel_sha256':digest,'nir_band_index':index,'nir_wavelength_nm':nm,
                'raw_shape':[header['lines'],header['samples'],header['bands']],
                'interleave':header['interleave'],'nir':nir,'rgb':rgb}
            records.append(record)
            if i%12==0:
                print(json.dumps({'stage':'verified','cubes':i,'total':len(parsed)}),flush=True)
        if len(records)!=144 or len({r['id'] for r in records})!=len(records):
            raise ValueError('Local HSI subset changed; version the protocol')
        scale = training_scale(records)
        rows = []
        for record in records:
            check_stop(OUTPUT)
            nir = record.pop('nir')
            rgb = record.pop('rgb')
            record['nir_below_range_fraction'] = float((nir<scale['low']).mean())
            record['nir_above_range_fraction'] = float((nir>scale['high']).mean())
            nir = np.clip((nir-scale['low'])/(scale['high']-scale['low']),0,1)
            nir = np.asarray(Image.fromarray(nir).resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)
            tensor = np.concatenate([rgb,nir[None]],axis=0).astype(np.float32)
            target = OUTPUT/(record['id']+'.npy')
            if target.exists():
                existing = np.load(target,allow_pickle=False)
                if not np.array_equal(existing,tensor):
                    raise ValueError('Prepared file differs; refusing overwrite')
            else:
                with target.open('wb') as f:
                    np.save(f,tensor,allow_pickle=False)
            rows.append({**record,'prepared':str(target.relative_to(ROOT)),'prepared_sha256':file_digest(target)})
        manifest = {'protocol':'hsi_rgb859_capture_day_v1','source':'https://doi.org/10.7910/DVN/E7WDNQ',
            'source_paper':'https://doi.org/10.1109/ACCESS.2023.3243243',
            'input':'Dataset-provided aligned RGB PNG; bilinear resize to 256x256; [0,1]',
            'target':'Nearest measured 860 nm band; fixed training-only scaling; clipped [0,1]; bilinear 256x256',
            'nir_scale':scale,'day_to_split':days,'counts':dict(Counter(r['split'] for r in rows)),
            'rows':rows,'builder_sha256':file_digest(Path(__file__)),
            'limitations':['Local 144-scan subset, not the complete 3389-pair dataset.',
                'Capture-day grouped split is our protocol, not an official RGB-to-NIR split.',
                'Capture date groups may contain different ingredient mixtures; validation/test group count is small.',
                'Scaled NIR is a relative intensity image, not absolute spectral reflectance.',
                'RGB/NIR spatial correspondence follows the dataset publication; optical registration errors may remain.',
                'HSI-to-Nutrition5k is a domain transfer; only downstream paired experiments measure calorie benefit.']}
        atomic_json(manifest_path,manifest)
        print(json.dumps({'manifest':str(manifest_path),'sha256':file_digest(manifest_path),'counts':manifest['counts'],
            'days':days,'nir_scale':scale},indent=2),flush=True)


if __name__=='__main__':
    build()
