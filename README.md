# SHiVAi-rs

Markers of cerebral small vessel disease from brain MRI, without Python, TensorFlow or CUDA:
**perivascular spaces** (PVS) from a T1-weighted image, and **white matter hyperintensities** (WMH)
and **lacunes** from a T1 with a FLAIR of the same session. It is a Rust port of the default
[SHiVAi](https://github.com/pboutinaud/SHiVAi) pipeline and runs the published SHiVA models

- in the **browser** (WebAssembly + WebGPU): no image leaves your computer, and
- as a **command line** tool on macOS (Metal; a CPU backend runs anywhere).

**Not for clinical use.**

**Limitations:** SWI and T2*-GRE images are not supported, so cerebral microbleeds are not segmented.
Only PVS (T1), WMH and lacunes (T1 + FLAIR).

## Live demo

<https://neurolabusc.github.io/SHiVAi-rs/>

To run it locally you need [Rust](https://rustup.rs) with the WebAssembly target
(`rustup target add wasm32-unknown-unknown`), [wasm-pack](https://rustwasm.github.io/wasm-pack/) and
[Bun](https://bun.sh) (or Node.js with npm):

```sh
bun install
bun run dev        # builds the WebAssembly, then serves http://localhost:8092
```

`npm install && npm run dev` works the same way. Use a browser with WebGPU and a GPU: recent desktop
Chrome or Edge, or Safari on macOS 26. On the first visit the page reloads once to install a service
worker (`public/coi-sw.js`) that makes it cross-origin isolated, so niimath can use threads; GitHub Pages
cannot send those headers itself.

### Using the page

1. **Load images.** Choose an example from **Load example…** (a T1 + FLAIR, or a T1 alone, from
   [OpenNeuro ds007522](https://doi.org/10.18112/openneuro.ds007522.v2.0.1), CC0), or drop a NIfTI
   image or a DICOM folder on the **T1-weighted** or **FLAIR** panel. Dropped DICOM is converted by
   dcm2niix in the browser. Dropping two images anywhere puts the one named "FLAIR" in the FLAIR panel.
   The FLAIR is optional; × removes it.
2. **Choose markers.** PVS needs only the T1; WMH and lacunes need the FLAIR (the lacune model is a
   244 MB download).
3. **Run.** The first run downloads the models (about 80 MB per marker, cached by the browser).
4. **Inspect.** The **Result** panel shows the cropped, conformed T1 with an overlay. **All markers:
   probability** (the default) shows every marker's probability map at once, PVS in *winter*, WMH in *warm*
   and lacunes in *violet*; **All markers: clusters** shows the thresholded clusters. Single markers, the brain mask, the
   T1 and the **FLAIR in T1 space** are also listed. The table gives the cluster count and volume of
   each marker.
5. **Save.** **Save** downloads the shown overlay(s) as NIfTI; **Save all** downloads every output, the
   summary table and the provenance as one `.zip` (see [Outputs](#outputs)).

## Command line (macOS)

Download `shivai-rs-VERSION-macos-arm64.pkg` from the [releases](https://github.com/neurolabusc/SHiVAi-rs/releases) and open it (Apple
silicon, macOS 13 or later; signed and notarized). It installs `/usr/local/bin/shivai-rs`,
`/usr/local/bin/niimath`, and the SHiVA models with the licences in `/usr/local/share/shivai-rs`
(package `io.github.neurolabusc.shivai-rs`). The command is `shivai-rs` because Python SHiVAi installs `shivai`.

```sh
shivai-rs --t1 T1w.nii.gz --pvs --out results                                   # PVS only
shivai-rs --t1 T1w.nii.gz --flair FLAIR.nii.gz --pvs --wmh --lac --out results  # all markers
```

The inputs are NIfTI; convert DICOM first with [dcm2niix](https://github.com/rordenlab/dcm2niix).
Without marker flags, PVS is segmented, plus WMH when a FLAIR is given. `--device cpu` runs without the GPU; `--help` lists every option. Models are read from `--models DIR`,
`$SHIVAI_MODELS` or `/usr/local/share/shivai-rs/models`, else downloaded once into `~/.cache/shivai-rs/<revision>` (`$XDG_CACHE_HOME` if set). A FLAIR
needs `niimath`: the one beside `shivai-rs` is preferred, then the `PATH` (or `--niimath`). Without the
package, `cargo build --release -p shivai-rs` builds `target/release/shivai-rs`.

On an M4 Pro, T1 + FLAIR with all three markers takes about 23 s on Metal and 33 s on the CPU (SME on
M4 and later, NEON on earlier Apple silicon); about 40 s in the browser.

## Outputs

The command line and the web app's **Save all** (one `<name>_shivai.zip`) write the same files,
`<name>_<suffix>`, where `<name>` is the T1 file name without `.nii.gz`. Images are on the cropped 1 mm T1 grid that SHiVAi segments on.

| Suffix | Contents |
| --- | --- |
| `t1.nii.gz` | T1: conformed, defaced, cropped, normalised |
| `brainmask.nii.gz` | Brain mask (uint8) |
| `flair.nii.gz` | FLAIR registered to the T1 and normalised (with a FLAIR) |
| `<marker>_probability.nii.gz` | Mean probability of the model's folds (float32); marker `pvs`, `wmh` or `lac` |
| `<marker>_clusters.nii.gz` | Clusters above threshold, numbered from 1 (int16) |
| `summary.tsv` | `marker`, `clusters`, `volume_mm3` per marker |
| `shivai.json` | Provenance: `schema`, `version`, `software`, `backend`, `t1`, `flair` (input file names), `seconds`, `registration` (niimath command, with a FLAIR), `models` {`license`, `license_owner`, `revision`, `sha256` per file}, `markers` [{`marker`, `label`, `clusters`, `volume_mm3`, `threshold`, `min_size`, `cluster_voxels`}], `warnings` |

**Stability:** this is version 0.1. From 1.0 these names, the summary columns, the provenance keys
(versioned by `schema`) and the command-line flags change only with a major version. Outputs in other
spaces, if added, will get new suffixes.

## How it works

The pipeline is SHiVAi's default configuration, step for step:

1. **Brain mask**: the SHiVA brain-mask network in two passes (whole field of view, then a crop on the
   first mask), each followed by thresholding, morphological opening and the largest cluster.
2. **Preprocessing**: conforming to 1 mm (nibabel's grid and scipy's spline interpolation),
   quickshear defacing, a 160 × 214 × 176 crop centred on the brain, and normalisation to the 99th
   percentile inside it.
3. **FLAIR**: registered rigidly to the T1 crop with niimath (`-robustfov -allineate -warp shr -cost nmi`) where
   SHiVAi uses ANTs, then defaced and normalised the same way.
4. **Markers**: the mean of each model's folds (5 for PVS and WMH, 3 for lacunes), thresholded (PVS 0.5,
   WMH and lacunes 0.2) and split into 26-connected clusters with SHiVAi's minimum sizes.

The repository holds one implementation of all of this:

| Path | Contents |
| --- | --- |
| `crates/core` | Preprocessing, the network execution plan, GPU kernel generation (WGSL and Metal), post-processing, NIfTI |
| `crates/cli` | The `shivai-rs` executable: Metal and CPU execution, niimath, model download |
| `crates/wasm` | The same pipeline for the browser (wasm-bindgen) |
| `src/`, `index.html` | The web app: NiiVue display, dcm2niix import, a worker running the pipeline with WebGPU and niimath |
| `models.json` | The model files: checksums, licence, owner, provenance and the pinned download revision |
| `tools/` | Model conversion (`lower.py`) and SHiVAi reference outputs (Python; development only) |
| `packaging/` | The macOS installer: niimath build, model fetch, signing, notarization |

The SHiVA TensorFlow models were converted once, with `tools/lower.py`, to `.shiva` files: a small
graph (convolution, transposed convolution, concatenation) with batch normalisation folded in and
float16 weights. They are hosted on the
[neurodeskorg/webapps](https://huggingface.co/datasets/neurodeskorg/webapps/tree/main/shivai/models)
Hugging Face dataset at the revision `models.json` pins, and every download is checksum-verified.

## Validation

Against SHiVAi itself (TensorFlow, ANTs; `tools/shivai_reference.py`) on the example subject
(ds007522 sub-HML0541), scored with `validation/score.py`:

| | Command line (Metal) | Browser (WebGPU) |
| --- | --- | --- |
| Brain mask | Dice 1.000 | 1.000 |
| Perivascular spaces (131 clusters in SHiVAi) | 0.999 (131) | 0.999 (131) |
| White matter hyperintensities (23.2 ml) | 0.974 (23.8 ml) | 0.965 (23.9 ml) |
| Lacunes (1) | 1.000 (1) | 1.000 (1) |

The WMH difference comes from niimath's registration in place of ANTs. Every preprocessing step was
also compared with SHiVAi's own output, including for a T1 stored in another orientation, and fold 0
of every converted model matched TensorFlow (Dice ≥ 0.999).

## Development

```sh
make test        # Rust unit tests, conformance graph on CPU and Metal, resampling vs scipy
make lint        # cargo fmt --check, clippy -D warnings
make smoke       # web app in headless Chromium (SwiftShader): boot + WebGPU conformance graph
make smoke-gpu   # the same on this GPU, plus the T1 + FLAIR example checked against SHiVAi
```

Per-model inference benchmarks: `cargo run --release -p shivai-rs --example inference -- cpu|metal MODEL
OUT.f32 REPEATS T1 [FLAIR]` and `node test/inference.mjs MODEL INPUT.f32 OUT.f32 REPEATS f16|f32`
(browser input: channel-first, each channel x, y, z in C order). In the browser, a Rust panic is reported with
its own message (`panicMessage`), not WebAssembly's bare "unreachable".

With SHiVAi's reference outputs (`tools/shivai_reference.py`), `SHIVAI_REFERENCE_DIR=… SHIVAI_REFERENCE_T1=…
cargo test --release -p shivai-core stages` compares every preprocessing step. `fixtures/conformance.*` (from
`tools/make_fixture.py`) is the test every backend must pass; a new backend (for example CUDA) implements
the three kernels in `crates/core/src/kernels.rs` and runs that plan.

### Publishing the web app

`.github/workflows/pages.yml` builds the site and deploys it to GitHub Pages on every push to `main`
(repository Settings → Pages → Source: GitHub Actions). For a custom domain, add
`public/CNAME` and build with `SHIVAI_BASE=/`.

### Releasing the macOS installer

```sh
make pkg          # builds shivai-rs and a self-contained niimath, fetches the models, signs, packages
make notarize     # notarizes and staples dist/shivai-rs-VERSION-macos-arm64.pkg (NOTARY_PROFILE)
make release      # both
```

`make pkg-adhoc` builds an unsigned package for local testing. Attach the `.pkg` and its `.sha256` to
the GitHub release.

## Licence and citation

The code is **AGPL-3.0-or-later**, derived from SHiVAi (© Fealinx); see `NOTICE` for the other
sources. The SHiVA segmentation models are © Fealinx (Philippe Boutinaud et al.) under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): attribution, non-commercial
use only, share alike. The installer also contains niimath (BSD-2-Clause).

If you use this software, cite the models:

- Boutinaud P, et al. 3D segmentation of perivascular spaces on T1-weighted 3 Tesla MR images with a
  convolutional autoencoder and a U-shaped neural network. *Front Neuroinform.* 2021;15:641600.
- Tsuchida A, et al. Early detection of white matter hyperintensities using SHIVA-WMH detector.
  *Hum Brain Mapp.* 2024;45(1):e26548.
