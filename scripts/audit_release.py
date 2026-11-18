"""Check tracked release files and hashes without reading user images or outputs."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    if not manifest['files']:
        raise SystemExit('Release manifest is empty')
    errors = []
    for item in manifest['files']:
        path = ROOT/item['path']
        if not path.is_file():
            errors.append('Missing: '+item['path'])
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            errors.append('Changed since release: '+item['path'])
        if path.suffix in ('.py','.json','.csv','.md','.txt','.ps1','.html'):
            text = data.decode('utf-8-sig')
            if re.search(r'[A-Za-z]:[\\/](?:Users|python项目|文件)[\\/]', text):
                errors.append('Private absolute path: '+item['path'])
            if re.search(r'(?<![\w-])(?:ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,})', text):
                errors.append('Possible credential: '+item['path'])
        if path.suffix == '.py':
            compile(data, str(path), 'exec')
        if path.suffix == '.docx':
            with ZipFile(path) as archive:
                xml = archive.read('word/document.xml').decode('utf-8')
                if re.search(r'[A-Za-z]:[\\/](?:Users|python项目|文件)[\\/]', xml):
                    errors.append('Private path in Word: '+item['path'])
    for name, digest in json.loads((ROOT/'docs/skill_reuse/source_manifest.json').read_text(encoding='utf-8'))['sha256'].items():
        actual = hashlib.sha256((ROOT/'docs/skill_reuse/upstream'/name).read_bytes()).hexdigest()
        if actual != digest:
            errors.append('Upstream provenance changed: '+name)
    # User-created local assets are not release files; gitignore keeps them private.
    forbidden = [p['path'] for p in manifest['files'] if
                 p['path'].endswith(('.onnx','.pt','.pth','.cbz','.cbr','.jpg','.jpeg')) or
                 p['path'].startswith(('.venv/','outputs/','evaluation/annotations/'))]
    errors.extend('Forbidden release asset: '+p for p in forbidden)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS {len(manifest["files"])} release files: hashes, paths, syntax, assets and upstream provenance')
    print('Local outputs are excluded from this audit; inspect git status before each public push.')


if __name__ == '__main__':
    main()
