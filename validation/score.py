"""Dice of the shivai outputs (command line, or the web app's Save all) against tools/shivai_reference.py.

    python score.py <output dir> <reference dir>
"""
import sys
from pathlib import Path
import nibabel as nib, numpy as np

app, ref = (Path(p) for p in sys.argv[1:3])
pairs = {'brainmask': '14_brainmask_cropped', 'pvs_clusters': '17_pvs_labelled', 'wmh_clusters': '21_wmh_labelled', 'lac_clusters': '23_lac_labelled'}
for name, r in pairs.items():
    a = next(app.glob(f'*_{name}.nii.gz'), None)
    if a is None or not (ref / f'{r}.nii.gz').exists():
        continue
    x, y = nib.load(a), nib.load(ref / f'{r}.nii.gz')
    assert np.allclose(x.affine, y.affine, atol=1e-3), (name, x.affine, y.affine)
    p, q = x.get_fdata() > 0, y.get_fdata() > 0
    print(f'{name:14s} app {int(x.get_fdata().max()):4d} clusters {p.sum():6d} vox | SHiVAi {int(y.get_fdata().max()):4d} clusters {q.sum():6d} vox | Dice {2 * (p & q).sum() / max(p.sum() + q.sum(), 1):.4f}')
