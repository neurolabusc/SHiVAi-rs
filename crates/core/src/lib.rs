// SHiVAi-rs core: the SHiVAi pipeline (preprocessing, network execution plan, GPU kernel generation,
// post-processing) shared by the native executable and the WebAssembly build.
pub mod cpu;
pub mod kernels;
pub mod nifti;
mod par;
pub mod pipeline;
pub mod plan;
pub mod preprocess;
pub mod report;
pub mod shiva;
#[cfg(all(feature = "sme", target_os = "macos", target_arch = "aarch64"))]
mod sme;

/// Model files with their checksums, licence, provenance and pinned download location.
pub const MANIFEST: &str = include_str!("../../../models.json");
