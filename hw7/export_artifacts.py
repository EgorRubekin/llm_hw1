"""Export artifacts from a local experiment directory."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED, ZIP_DEFLATED
def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def export(folder, destination):
    root = folder / 'artifacts_compression_hw'
    audit = json.loads((root/'resume_audit.json').read_text())
    for rel, key in [('prepared/dev.jsonl','dev_sha256'), ('teacher_answers/answers.jsonl','answers_sha256')]:
        if sha256(root/rel) != audit[key]:
            raise ValueError(f'Checksum mismatch: {rel}')
    manifest = json.loads((root/'submission_manifest.json').read_text())
    paths = {p for pattern in ['eval/*','prepared/*','teacher_answers/answers.jsonl','training/*/*.json','environments/*.json','quantized/*.config.json','*.json','results.csv'] for p in root.glob(pattern) if p.is_file()}
    assert (root/'teacher_answers/answers.jsonl') in paths
    for entry in manifest['files']:
        path = root / entry['path']
        if sha256(path) != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')
        paths.add(path)
    with ZipFile(destination,'x',allowZip64=True) as archive:
        for path in sorted(paths):
            archive.write(path,'hw7/artifacts_compression_hw/'+str(path.relative_to(root)),compress_type=ZIP_STORED if path.suffix=='.safetensors' else ZIP_DEFLATED)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,default=Path('hw7_artifacts.zip'))
    args=parser.parse_args()
    export(args.folder,args.output)
