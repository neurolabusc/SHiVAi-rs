# Model conversion and references (development only)

The app needs none of this; it reads `.shiva` files. Python 3.11 with `tensorflow==2.18.*`, `tf2onnx`, `onnx`, `nibabel`, `scipy`.

- `lower.py <fold.tf_inference | model.keras> <out.shiva>`: SHiVA SavedModel or Keras model to a `.shiva` graph (BN folded, float16 weights).
- `reference.py <model.shiva> <input.npy> [tf.npy]`: float32 numpy executor; the semantics every backend matches.
- `tf_out.py`: TensorFlow output of a SavedModel, for checking `lower.py`.
- `make_fixture.py ../fixtures`: the conformance graph and its reference output.
- `shivai_reference.py <T1> <models> <out> [FLAIR]`: every preprocessing stage with SHiVAi's own interfaces and TF models (needs `shivai`, `nipype`, `quickshear`, ANTs for the FLAIR).

Fold 0 of every model matched TensorFlow (Dice ≥ 0.999 at 0.5 on a WMH-challenge subject); the ensembles reproduce SHiVAi end to end (`validation/`).
