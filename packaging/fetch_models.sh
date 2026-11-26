#!/bin/sh
# The SHiVA models from the manifest's pinned Hugging Face revision, checksum-verified.
#   packaging/fetch_models.sh DIR
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
dir=${1:?usage: fetch_models.sh DIR}
mkdir -p "$dir"
python3 - "$root/models.json" "$dir" <<'PY'
import hashlib, json, sys, urllib.request
from pathlib import Path
m = json.load(open(sys.argv[1])); out = Path(sys.argv[2])
for a in m['assets']:
    p = out / a['filename']
    if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == a['sha256']:
        continue
    data = urllib.request.urlopen(m['base_url'] + a['filename']).read()
    if hashlib.sha256(data).hexdigest() != a['sha256'] or len(data) != a['bytes']:
        sys.exit(f"{a['filename']} does not match its checksum")
    p.write_bytes(data)
    print('fetched', a['filename'])
PY
