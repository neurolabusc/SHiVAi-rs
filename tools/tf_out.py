"""usage: tf_out.py savedmodel input.npy channels(0,1|0) out.npy"""
import sys, numpy as np, tensorflow as tf
sm, inp, ch, out = sys.argv[1:5]
x = np.load(inp)[..., [int(c) for c in ch.split(',')]]
f = tf.saved_model.load(sm).signatures['serving_default']
name, spec = next(iter(f.structured_input_signature[1].items()))
np.save(out, list(f(**{name: tf.constant(x, spec.dtype)}).values())[0].numpy().squeeze())
np.save(out.replace('.npy', '_in.npy'), x)
