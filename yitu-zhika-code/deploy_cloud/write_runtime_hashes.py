import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = sorted((root / 'checkpoints').rglob('*.pt'))
hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(root / 'runtime_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
