"""Lower one SHiVA ResUnet3D SavedModel fold to a backend-neutral .shiva file.

    python lower.py <fold.tf_inference> <out.shiva>

The file is: b'SHIVA\\x00\\x01\\x00', uint32 JSON length, JSON graph, zero padding to 16 bytes, float16 weights.
Ops (activations NCDHW, weights OIDHW / IODHW as ONNX, every shape derived from the input at run time):
  conv   {x, out, k, stride, pad, cin, cout, w, b, add?, act}  out = act(conv(x) + b [+ add])
  convT  {x, out, k, pad, cin, cout, w, b, act}                 stride 2, output o = 2i + tap
  concat {x: [a, b], out}         [a, b] on b's grid: a[i + crop], crop = dims(a) - dims(b) per axis
                                  (crop < 0 pads zeros at the start, as TF's Cropping3D/ZeroPadding3D)
pad is TF's 'same' (conv: ceil(n/s), convT: 2n) or 'valid' (conv: (n-k)//s + 1, convT: 2n + k - 2).
act is 'none', 'swish' (x * sigmoid(x)) or 'sigmoid'. w/b are [offset, count] in float16 elements.
"""
import json, struct, subprocess, sys, tempfile
from pathlib import Path

import numpy as np, onnx
from onnx import TensorProto as T, helper, numpy_helper as nh

MAGIC = b'SHIVA\x00\x01\x00'


def attr(node, name):
    return next(helper.get_attribute_value(a) for a in node.attribute if a.name == name)


def to_float32_math(g):
    """Mixed-precision Casts become Identity: weights stored float16 are exact in float32."""
    for n in g.node:
        if n.op_type == 'Cast':
            n.op_type = 'Identity'
            del n.attribute[:]
    g.input[0].type.tensor_type.elem_type = T.FLOAT


def to_ncdhw(g):
    inp = g.input[0]
    # NDHWC -> NCDHW: a Transpose, or a Reshape when C == 1
    tr = next(n for n in g.node if n.op_type in ('Transpose', 'Reshape') and n.input[0] in
              {inp.name} | {c.output[0] for c in g.node if c.input and c.input[0] == inp.name})
    if tr.op_type == 'Transpose':
        assert attr(tr, 'perm') == [0, 4, 1, 2, 3]
    src = tr.input[0]
    for n in g.node:
        n.input[:] = [src if x == tr.output[0] else x for x in n.input]
    g.node.remove(tr)
    d = inp.type.tensor_type.shape.dim
    shape = [x.dim_value for x in d]  # N D H W C
    for k, v in enumerate([1, shape[4], shape[1], shape[2], shape[3]]):
        d[k].ClearField('dim_param')
        d[k].dim_value = v
    out = g.output[0]
    rs = next(n for n in g.node if n.output[0] == out.name)
    assert rs.op_type == 'Reshape'  # NCDHW -> NDHWC with C == 1: same memory
    rs.op_type = 'Identity'
    del rs.input[1:]
    out.type.tensor_type.ClearField('shape')


def same_pad_begin(n, k, s):
    out = -(-n // s)
    return max((out - 1) * s + k - n, 0) // 2


def lower(model):
    g = model.graph
    to_float32_math(g)
    to_ncdhw(g)
    inp = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    dims = {g.input[0].name: inp[2:]}  # spatial dims by the rules the runtimes use
    const = {i.name: nh.to_array(i).astype(np.float32) for i in g.initializer}
    alias, unit = {}, []
    nodes = []
    for n in g.node:
        n.input[:] = [alias.get(x, x) for x in n.input]
        if n.op_type == 'Identity':
            alias[n.output[0]] = n.input[0]
        elif n.op_type == 'Reshape' and list(const.get(n.input[1], [0])[2:]) == [1, 1, 1]:
            # NDHWC <-> NCDHW at a 1x1x1 bottleneck: a no-op
            alias[n.output[0]] = n.input[0]
            unit.append(n.input[0])
        else:
            nodes.append(n)
    out_name = alias.get(g.output[0].name, g.output[0].name)
    users = {}
    for n in nodes:
        for x in n.input:
            users.setdefault(x, []).append(n)
    ops, producer = [], {}  # producer: tensor -> op dict whose 'out' it is

    def fusable(t):
        op = producer.get(t)
        return op if op and op['op'] in ('conv', 'convT') and op['act'] == 'none' and len(users.get(t, [])) == 1 else None

    slices = {}
    for n in nodes:
        t, ins = n.output[0], list(n.input)
        if n.op_type in ('Conv', 'ConvTranspose'):
            w = const[ins[1]]
            k = attr(n, 'kernel_shape')[0]
            assert attr(n, 'kernel_shape') == [k] * 3 and attr(n, 'dilations') == [1, 1, 1]
            s = attr(n, 'strides')[0]
            op = {'op': 'conv' if n.op_type == 'Conv' else 'convT', 'x': ins[0], 'out': t, 'k': k,
                  'w': w, 'b': const[ins[2]] if len(ins) > 2 and ins[2] else np.zeros(w.shape[1 if n.op_type == 'ConvTranspose' else 0], np.float32),
                  'act': 'none'}
            d_in = dims[ins[0]]
            pads = attr(n, 'pads') if any(a.name == 'pads' for a in n.attribute) else [0] * 6
            if n.op_type == 'Conv':
                op.update(stride=s, cout=w.shape[0], cin=w.shape[1])
                same = [-(-d // s) for d in d_in]
                begin = [same_pad_begin(d, k, s) for d in d_in]
                end = [max((o - 1) * s + k - d - b, 0) for o, d, b in zip(same, d_in, begin)]
                if pads == begin + end:  # k == 1 is 'same' and 'valid' alike
                    op['pad'], dims[t] = 'same', same
                else:
                    assert not any(pads), (n.name, pads, d_in)
                    op['pad'], dims[t] = 'valid', [(d - k) // s + 1 for d in d_in]
            else:
                assert s == 2 and pads[:3] == [0, 0, 0], n.name
                op.update(cin=w.shape[0], cout=w.shape[1])
                if pads[3:] == [k - 2] * 3:
                    op['pad'] = 'same'
                    dims[t] = [2 * d for d in d_in]
                else:
                    assert not any(pads), n.name
                    op['pad'] = 'valid'
                    dims[t] = [2 * d + k - 2 for d in d_in]
            ops.append(op)
            producer[t] = op
        elif n.op_type in ('Mul', 'Add') and (ins[1] in const or ins[0] in const):
            x, c = (ins[0], const[ins[1]]) if ins[1] in const else (ins[1], const[ins[0]])
            op = fusable(x)
            assert op is not None and 'add' not in op, n.name
            c = c.reshape(-1)
            if n.op_type == 'Mul':
                op['w'] = op['w'] * (c[:, None, None, None, None] if op['op'] == 'conv' else c[None, :, None, None, None])
                op['b'] = op['b'] * c
            else:
                op['b'] = op['b'] + c
            op['out'] = t
            producer[t] = op
            dims[t] = dims[x]
        elif n.op_type == 'Add':
            a, b = ins
            op = fusable(a)
            if op is None or 'add' in op:
                a, b = b, a
                op = fusable(a)
            assert op is not None and 'add' not in op, n.name
            assert dims[a] == dims[b], n.name
            op['add'] = b
            ops.remove(op)  # the residual may come from a later op (a 1x1 shortcut)
            ops.append(op)
            op['out'] = t
            producer[t] = op
            dims[t] = dims[a]
        elif n.op_type == 'Sigmoid':
            x = ins[0]
            uses = users[x]
            op = producer.get(x)
            assert op and op['act'] == 'none', n.name
            if len(uses) == 2 and any(u.op_type == 'Mul' and set(u.input) == {x, t} for u in uses):
                mul = next(u for u in uses if u.op_type == 'Mul')
                op['act'] = 'swish'
                op['out'] = mul.output[0]
                producer[mul.output[0]] = op
                dims[mul.output[0]] = dims[x]
                mul.op_type = 'Fused'
            else:
                assert len(uses) == 1, n.name
                op['act'] = 'sigmoid'
                op['out'] = t
                producer[t] = op
                dims[t] = dims[x]
        elif n.op_type == 'Fused':
            continue
        elif n.op_type == 'Slice':
            starts, ends, axes = (const[x].astype(np.int64) for x in ins[1:4])
            crop = [0, 0, 0]
            for st, en, ax in zip(starts, ends, axes):
                assert en >= 2 ** 31 - 1 and (ax in (0, 1) and st == 0 or ax >= 2), n.name
                if ax >= 2:
                    crop[ax - 2] = int(st)
            dims[t] = [d - c for d, c in zip(dims[ins[0]], crop)]
            slices[t] = (ins[0], crop)
        elif n.op_type == 'Pad':
            p = const[ins[1]].astype(np.int64).tolist()
            assert len(ins) < 3 or not ins[2] or not const[ins[2]].any(), n.name
            assert p[:2] == [0, 0] and p[5:] == [0] * 5 and min(p) >= 0, (n.name, p)
            crop = [-int(v) for v in p[2:5]]
            dims[t] = [d - c for d, c in zip(dims[ins[0]], crop)]
            slices[t] = (ins[0], crop)
        elif n.op_type == 'Concat':
            assert attr(n, 'axis') == 1 and len(ins) == 2
            a, crop = slices.pop(ins[0], (ins[0], [0, 0, 0]))
            assert ins[1] not in slices and dims[ins[0]] == dims[ins[1]], (n.name, dims[ins[0]], dims[ins[1]])
            # the runtimes derive crop as (upsampled - skip) dims
            assert crop == [u - k for u, k in zip(dims[a], dims[ins[1]])], n.name
            dims[t] = dims[ins[1]]
            ops.append({'op': 'concat', 'x': [a, ins[1]], 'out': t})
            producer[t] = ops[-1]
        else:
            raise ValueError(f'unsupported {n.op_type} {n.name}')
    assert not slices
    assert all(dims[x] == [1, 1, 1] for x in unit)
    assert producer[out_name]['out'] == out_name
    # short, stable tensor names
    names = {g.input[0].name: 'input'}
    for op in ops:
        names[op['out']] = f't{len(names)}'
    names[out_name] = 'output'
    weights, blobs = [], 0
    graph = []
    for op in ops:
        o = {'op': op['op'], 'out': names[op['out']]}
        if op['op'] == 'concat':
            o['x'] = [names[x] for x in op['x']]
        else:
            o.update(x=names[op['x']], k=op['k'], pad=op['pad'], cin=op['cin'], cout=op['cout'], act=op['act'])
            if op['op'] == 'conv':
                o['stride'] = op['stride']
            if 'add' in op:
                o['add'] = names[op['add']]
            for key in ('w', 'b'):
                a = op[key].astype(np.float16)
                assert np.isfinite(a).all(), (o['out'], key)
                o[key] = [blobs, a.size]
                weights.append(a.ravel())
                blobs += a.size
        graph.append(o)
    return {'format': 'shiva-unet/1', 'input': {'channels': inp[1], 'dims': inp[2:]}, 'output': 'output', 'ops': graph}, np.concatenate(weights)


def write(path, graph, weights):
    j = json.dumps(graph, separators=(',', ':')).encode()
    head = MAGIC + struct.pack('<I', len(j)) + j
    head += b'\0' * (-len(head) % 16)
    Path(path).write_bytes(head + weights.astype('<f2').tobytes())


def read(path):
    b = Path(path).read_bytes()
    assert b[:8] == MAGIC
    (n,) = struct.unpack('<I', b[8:12])
    graph = json.loads(b[12:12 + n])
    start = 12 + n + (-(12 + n) % 16)
    return graph, np.frombuffer(b[start:], '<f2')


def main():
    src, dst = sys.argv[1:3]
    if src.endswith('.keras'):  # the brain mask model: a Keras file and its models.py
        import importlib.util, keras
        spec = importlib.util.spec_from_file_location('kmodel', Path(src).parent / 'models.py')
        sys.modules['kmodel'] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules['kmodel'])
        exported = tempfile.mkdtemp()
        model = keras.saving.load_model(src, compile=False)
        model(np.zeros((1, *model.image_shape), np.float32))  # export needs one call
        model.export(exported)
        src = exported
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / 'raw.onnx'
        subprocess.run([sys.executable, '-m', 'tf2onnx.convert', '--saved-model', src, '--signature_def', 'serving_default',
                        '--opset', '17', '--output', str(raw)], check=True, capture_output=True)
        graph, weights = lower(onnx.load(raw))
    write(dst, graph, weights)
    kinds = {}
    for op in graph['ops']:
        kinds[op['op']] = kinds.get(op['op'], 0) + 1
    print(dst, Path(dst).stat().st_size, kinds)


if __name__ == '__main__':
    main()
