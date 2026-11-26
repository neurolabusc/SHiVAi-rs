"""Float32 numpy executor of a .shiva graph: the semantics every backend must match.

    python reference.py <model.shiva> <input NDHWC .npy> [<tf output .npy>]
"""
import sys, time
import numpy as np
from lower import read, same_pad_begin


def sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


ACT = {'none': lambda x: x, 'sigmoid': sigmoid, 'swish': lambda x: x * sigmoid(x)}


def conv(x, w, b, k, s, pad):
    """x [C, D, H, W]; w [O, C, k, k, k]."""
    c, *dims = x.shape
    if pad == 'same':
        out = [-(-n // s) for n in dims]
        lo = [same_pad_begin(n, k, s) for n in dims]
    else:
        out = [(n - k) // s + 1 for n in dims]
        lo = [0, 0, 0]
    pad = [(l, (o - 1) * s + k - n - l) for l, o, n in zip(lo, out, dims)]
    xp = np.pad(x, [(0, 0)] + [(a, max(e, 0)) for a, e in pad])
    y = np.zeros((w.shape[0], *out), np.float32)
    for t in np.ndindex(k, k, k):
        sl = tuple(slice(t[a], t[a] + (out[a] - 1) * s + 1, s) for a in range(3))
        y += np.einsum('oc,cdhw->odhw', w[:, :, t[0], t[1], t[2]], xp[(slice(None),) + sl], optimize=True)
    return y + b[:, None, None, None]


def conv_transpose(x, w, b, k, pad):
    """x [C, D, H, W]; w [C, O, k, k, k]; stride 2, o = 2i + t; 'same' keeps 2n, 'valid' 2n + k - 2."""
    c, *dims = x.shape
    y = np.zeros((w.shape[1], *[2 * n + k - 1 for n in dims]), np.float32)
    for t in np.ndindex(k, k, k):
        sl = tuple(slice(t[a], t[a] + 2 * dims[a], 2) for a in range(3))
        y[(slice(None),) + sl] += np.einsum('co,cdhw->odhw', w[:, :, t[0], t[1], t[2]], x, optimize=True)
    n = [2 * d + (k - 2 if pad == 'valid' else 0) for d in dims]
    return y[:, :n[0], :n[1], :n[2]] + b[:, None, None, None]


def run(graph, weights, x):
    """x [C, D, H, W] float32 -> output [C', D, H, W]."""
    W = lambda r: weights[r[0]:r[0] + r[1]].astype(np.float32)
    env = {'input': x}
    for op in graph['ops']:
        if op['op'] == 'concat':
            a, b = (env[t] for t in op['x'])
            c = [m - n for m, n in zip(a.shape[1:], b.shape[1:])]
            a = np.pad(a, [(0, 0)] + [(max(-v, 0), 0) for v in c])[:, max(c[0], 0):, max(c[1], 0):, max(c[2], 0):]
            env[op['out']] = np.concatenate([a, b])
            continue
        k = op['k']
        if op['op'] == 'conv':
            y = conv(env[op['x']], W(op['w']).reshape(op['cout'], op['cin'], k, k, k), W(op['b']), k, op['stride'], op['pad'])
        else:
            y = conv_transpose(env[op['x']], W(op['w']).reshape(op['cin'], op['cout'], k, k, k), W(op['b']), k, op['pad'])
        if 'add' in op:
            y += env[op['add']]
        env[op['out']] = ACT[op['act']](y)
    return env[graph['output']]


if __name__ == '__main__':
    graph, weights = read(sys.argv[1])
    x = np.load(sys.argv[2])[0].transpose(3, 0, 1, 2).astype(np.float32)
    t = time.time()
    y = run(graph, weights, x)[0]
    print(f'reference {time.time() - t:.1f}s')
    np.save(sys.argv[2].replace('.npy', '_ref.npy'), y)
    if len(sys.argv) > 3:
        r = np.load(sys.argv[3])
        d = np.abs(y - r)
        a, b = r > .5, y > .5
        print(f'max|d| {d.max():.4g} mean|d| {d.mean():.3g} dice {2 * (a & b).sum() / max(a.sum() + b.sum(), 1):.5f} vox {a.sum()} {b.sum()}')
