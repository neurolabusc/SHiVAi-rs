"""Tiny .shiva graph with every op variant and its reference.py output: the backend conformance fixture.

    python make_fixture.py ../fixtures
"""
import json, sys
from pathlib import Path
import numpy as np
from lower import write, read
from reference import run

rng = np.random.default_rng(7)
ops, blobs = [], []
def weights(*shape):
    a = (rng.standard_normal(shape) * 0.15).astype(np.float16)
    blobs.append(a.ravel())
    return [sum(b.size for b in blobs[:-1]), a.size]
def conv(x, out, k, stride, pad, cin, cout, act, add=None):
    op = {'op': 'conv', 'out': out, 'x': x, 'k': k, 'pad': pad, 'cin': cin, 'cout': cout, 'act': act, 'stride': stride,
          'w': weights(cout, cin, k, k, k), 'b': weights(cout)}
    if add: op['add'] = add
    ops.append(op)
def convT(x, out, k, pad, cin, cout, act):
    ops.append({'op': 'convT', 'out': out, 'x': x, 'k': k, 'pad': pad, 'cin': cin, 'cout': cout, 'act': act,
                'w': weights(cin, cout, k, k, k), 'b': weights(cout)})
# input 2 x 7 x 6 x 5 (odd and even dims)
conv('input', 'a', 3, 1, 'same', 2, 6, 'swish')                 # 7 6 5
conv('a', 'b', 3, 2, 'same', 6, 9, 'swish')                     # 4 3 3
conv('b', 'c', 1, 1, 'same', 9, 9, 'none')
conv('c', 'd', 3, 1, 'same', 9, 9, 'swish', add='b')            # residual
convT('d', 'e', 3, 'same', 9, 5, 'swish')                       # 8 6 6
convT('d', 'f', 1, 'same', 9, 5, 'none')
conv('e', 'g', 3, 1, 'same', 5, 5, 'swish', add='f')
ops.append({'op': 'concat', 'out': 'h', 'x': ['g', 'a']})       # crop 1 0 1
conv('h', 'i', 3, 2, 'valid', 11, 4, 'swish')                   # 3 2 2
convT('i', 'j', 3, 'valid', 4, 3, 'none')                       # 7 5 5
ops.append({'op': 'concat', 'out': 'k', 'x': ['j', 'a']})       # crop 0 -1 0 (pads)
conv('k', 'output', 1, 1, 'same', 9, 1, 'sigmoid')
graph = {'format': 'shiva-unet/1', 'input': {'channels': 2, 'dims': [7, 6, 5]}, 'output': 'output', 'ops': ops}
out = Path(sys.argv[1])
write(out / 'conformance.shiva', graph, np.concatenate(blobs))
g, w = read(out / 'conformance.shiva')
x = rng.random((2, 7, 6, 5)).astype(np.float32)
y = run(g, w, x)
json.dump({'dims': [7, 6, 5], 'input': x.ravel().tolist(), 'output': y.ravel().tolist()}, open(out / 'conformance.json', 'w'))
print(y.shape, float(y.min()), float(y.max()))
