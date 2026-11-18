"""Download the pinned model; verify against the recorded SHA256."""
import hashlib
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'models/manifest.json').read_text(encoding='utf-8'))
target = root/'models'/manifest['file']
url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{manifest['file']}"
temporary = target.with_suffix('.download')
try:
    urllib.request.urlretrieve(url, temporary)
    actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if actual != manifest['sha256']:
        raise ValueError('SHA256 mismatch: ' + actual)
    temporary.replace(target)
finally:
    if temporary.exists():
        temporary.unlink()
print(target)
