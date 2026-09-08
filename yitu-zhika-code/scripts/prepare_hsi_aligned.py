"""Apply the training-verified uniform cube-to-PNG rotation to a new data version."""
import copy
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.capsicum_job import atomic_json,check_space,file_digest,job_lock


def align_nir(array):
    if array.shape!=(4,256,256):
        raise ValueError('Expected prepared RGB/NIR tensor')
    result=array.copy()
    result[3]=np.rot90(array[3],3)
    return result


def run():
    source=ROOT/'data/hsi_prepared_v1/manifest.json'
    evidence=ROOT/'results/hsi_unet_v2/alignment_audit.json'
    manifest=json.loads(source.read_text(encoding='utf-8'))
    audit=json.loads(evidence.read_text(encoding='utf-8'))
    if not audit['passed'] or audit['training_scans_checked']!=93 or audit['nir_rot90_k']!=3 or audit['manifest_sha256']!=file_digest(source):
        raise ValueError('All-training orientation verification is required')
    destination=ROOT/'data/hsi_prepared_v2'
    destination.mkdir(parents=True,exist_ok=True)
    with job_lock(destination/'job.lock'):
        check_space(ROOT)
        result=copy.deepcopy(manifest)
        result.update(protocol='hsi_rgb859_capture_day_aligned_v2',
            source_manifest_sha256=file_digest(source),builder_sha256=file_digest(Path(__file__)),
            orientation={'nir_rot90_k':3,'source':'All 93 training scans; gradient agreement of visible HSI band and PNG',
                         'evidence_sha256':file_digest(evidence)},
            target=manifest['target']+'; native HSI rotated 90 degrees clockwise to match dataset PNG')
        for row in result['rows']:
            path=ROOT/row['prepared']
            if file_digest(path)!=row['prepared_sha256']:
                raise ValueError('Source prepared file changed')
            tensor=align_nir(np.load(path,allow_pickle=False))
            output=destination/(row['id']+'.npy')
            if output.exists():
                if not np.array_equal(np.load(output,allow_pickle=False),tensor):
                    raise ValueError('Aligned artifact differs')
            else:
                with output.open('wb') as stream:
                    np.save(stream,tensor,allow_pickle=False)
            row['unaligned_prepared_sha256']=row['prepared_sha256']
            row['prepared']=str(output.relative_to(ROOT))
            row['prepared_sha256']=file_digest(output)
        output=destination/'manifest.json'
        if output.exists() and json.loads(output.read_text(encoding='utf-8'))!=result:
            raise ValueError('Frozen aligned manifest differs')
        atomic_json(output,result)
        aligned_audit={**audit,'source_manifest_sha256':file_digest(source),'manifest_sha256':file_digest(output)}
        atomic_json(ROOT/'results/hsi_unet_v2/aligned_audit.json',aligned_audit)
        print(json.dumps({'manifest':str(output),'sha256':file_digest(output),'counts':result['counts']}),flush=True)


if __name__=='__main__':
    run()
