Guidance for AI agents working in this repository. Machine-specific state is in `CLAUDE.local.md` when
present (git-ignored).

## What this is

SHiVAi's default pipeline (brain mask, preprocessing, PVS/WMH/lacune networks, cluster post-processing)
in Rust, for the browser (WebAssembly + WebGPU) and the command line (Metal, CPU). See README.md.

## Rules

- One implementation: pipeline logic lives in `crates/core` only. `crates/cli` (native) and `crates/wasm` +
  `src/worker.ts` (browser) are thin drivers around its stages (`T1Prep` → `Prepared`), its plan and its
  generated kernels (`kernels.rs`, WGSL and MSL from one description). `src/gpu.ts` is a generic dispatcher.
- Every backend must pass `fixtures/conformance.*` (`cargo test`, `make smoke`). Preprocessing changes must
  keep `SHIVAI_REFERENCE_DIR=… SHIVAI_REFERENCE_T1=… cargo test --release -p shivai-core stages` passing against SHiVAi's own outputs
  (`tools/shivai_reference.py`); end-to-end runs are scored with `validation/score.py`.
- Model files are pinned by `models.json` (checksums, licence CC BY-NC-SA 4.0, owner Fealinx, provenance,
  Hugging Face revision). Never change a model without regenerating it with `tools/lower.py` and re-pinning.
- CPU: one packed-convolution path (`cpu.rs`). FP32 SME (`sme.c`, runtime-detected, M4+, layers with ≥ 16
  output channels) else the portable register-tiled kernel (NEON + FMA on aarch64). Feature `sme` builds
  `sme.c` only on macOS arm64; one notarized binary serves all Apple silicon.
- GPU: `src/gpu.ts` and `crates/cli/src/metal.rs` wait every 8 layers. Registration is
  `niimath -robustfov -allineate <t1 crop> -warp shr -cost nmi` in both CLI and browser
  (`pipeline::REGISTRATION`).
- Outputs are a public contract owned by `crates/core/src/report.rs` (names, `summary.tsv`, provenance with
  `schema`); both drivers only call it. A breaking change bumps `SCHEMA` and is listed in the README (`Outputs` matches `suffixes`/`add_marker`/`finish`).
- Browser errors: the wasm panic hook keeps the message; `src/worker.ts` reports `panicMessage()` in place of "unreachable".
- The page must stay cross-origin isolated (COOP/COEP from `vite.config.ts` locally, `public/coi-sw.js`
  on Pages) for threaded niimath: cross-origin fetches must be CORS (Hugging Face is).
- AGPL-3.0-or-later; keep `NOTICE` current when adding third-party code.

## Commands

`bun run dev` (http://localhost:8092), `make test lint`, `make smoke` (SwiftShader), `make smoke-gpu`
(real GPU, full example vs SHiVAi), `make pkg` / `make notarize` / `make release` (macOS installer in
`dist/`). The web build goes to `site/` (not `dist/`, which holds the installer); `SHIVAI_BASE` sets its base
path. `.github/workflows/pages.yml` deploys the site on every push to `main`. The Metal test skips only
without a Metal device; setup failures fail. `SHIVAI_TEST_PORT` overrides the smoke port (4174). Benchmarks:
`crates/cli/examples/inference.rs`, `test/inference.mjs` (port 4198).

## Packaging

- Binary and package are `shivai-rs` (Python SHiVAi owns `shivai`). `dist/shivai-rs-VERSION-macos-arm64.pkg`,
  identifier `io.github.neurolabusc.shivai-rs`, installs `/usr/local/bin/{shivai-rs,niimath}` and
  `/usr/local/share/shivai-rs/{models,LICENSE,NOTICE,LICENSE-niimath,models.json}`; `package_macos.sh` rejects
  non-system dylibs and `minos` ≠ 13.0.
- `packaging/build_niimath.sh`: niimath pinned commit, static libomp built from LLVM sources for macOS 13, no
  zstd. It configures niimath's `src/` directly: the superbuild does not forward OpenMP hints or
  `CMAKE_OSX_DEPLOYMENT_TARGET`.
- CLI lookup order: models `--models`, `$SHIVAI_MODELS`, `/usr/local/share/shivai-rs/models`, user cache
  `~/.cache/shivai-rs/<revision>` (`$XDG_CACHE_HOME`, downloaded); niimath `--niimath`/`$SHIVAI_NIIMATH`, else beside the executable, else `PATH`.

## Next steps

- Cerebral microbleeds (CMB): no code yet. SHiVAi needs only SWI (or T2*-GRE), T1 optional: the brain mask
  comes from the T1, the SWI is registered rigidly to the T1 crop, the mask brought back by the inverse
  transform, and segmentation runs in SWI space. Models: SHiVA CMB v5 (5 folds, ResUnet3D-8.7.2-1.5-SWAN),
  to convert with `tools/lower.py` and verify against TensorFlow on an image with microbleeds. Test data:
  OpenNeuro ds007522 (CC0; cite Barnes et al., doi:10.18112/openneuro.ds007522.v2.0.1), T1w + 3D
  multi-echo GRE (5 echoes, TE 6.7–22.3 ms). Use echo 5 (or RMS of echoes 3–5); screen the oldest
  participants with the CLI, rank by cluster count, confirm visually.
- Registration dominates run time (~14 s native, ~20 s in the browser with 8 threads). Option: a rigid
  schedule for niimath's fast engine (affine `-cost fast` takes 0.9 s but cut WMH Dice to 0.88).
- No CUDA backend yet: implement the three kernels of `kernels.rs` and pass `fixtures/conformance.*`.
- Deferred clean-ups: `kernels.rs` emission dedup and dialect simplification (gate on byte-identical kernels); the
  direct (≤ 8 channel) GPU path still declares workgroup arrays (benchmark before removing);
  `affine_transform`/`morph`/`transpose` hot loops (benchmark first); a synthetic CI test of
  `T1Prep → segment` (CI never sets `SHIVAI_REFERENCE_DIR`); SHA-pinned Actions.
