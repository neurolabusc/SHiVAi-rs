// shivai-rs: SHiVAi's small vessel disease segmentation as a native executable (Metal or CPU), with
// native niimath for the rigid FLAIR-to-T1 registration.
#[cfg(target_os = "macos")]
mod metal;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use shivai_core::nifti::{self, Data};
use shivai_core::pipeline::{self, to_network, Ensemble, Prepared, T1Prep, GRID, REGISTRATION};
use shivai_core::plan::plan;
use shivai_core::preprocess::Volume;
use shivai_core::report::{self, Report};
use shivai_core::shiva::{parse, Model};
use shivai_core::{cpu, MANIFEST};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Where the macOS installer puts the models and niimath.
const INSTALLED_MODELS: &str = "/usr/local/share/shivai-rs/models";

const HELP: &str =
    "shivai-rs VERSION: SHiVAi small vessel disease segmentation (not for clinical use)

Usage: shivai-rs --t1 T1.nii.gz [--flair FLAIR.nii.gz] --out DIR [options]

  --t1 FILE        T1-weighted NIfTI (required)
  --flair FILE     FLAIR of the same session (needed for --wmh and --lac)
  --out DIR        Output folder (created)
  --pvs --wmh --lac
                   Markers to segment (default: pvs, plus wmh with a FLAIR)
  --models DIR     Folder of the .shiva model files (default: $SHIVAI_MODELS, then
                   /usr/local/share/shivai-rs/models, else downloaded once into the user cache)
  --device DEV     metal (macOS default) or cpu
  --niimath PATH   niimath for the FLAIR registration (default: $SHIVAI_NIIMATH, else the one
                   beside shivai-rs, else PATH)
  --force          Replace existing outputs
  --quiet          No progress messages
  --help, -h       This help
  --version, -v    Version

Outputs, on the cropped 1 mm T1 grid, as <name>_<suffix>: t1, brainmask, flair,
<marker>_probability and <marker>_clusters (.nii.gz), summary.tsv and shivai.json (provenance).
The SHiVA models are CC BY-NC-SA 4.0 (Fealinx); see the provenance file.
";

#[derive(Clone, Copy, Debug, PartialEq)]
enum Device {
    Cpu,
    #[cfg(target_os = "macos")]
    Metal,
}

struct Args {
    t1: PathBuf,
    flair: Option<PathBuf>,
    out: PathBuf,
    tasks: Vec<&'static str>,
    models: PathBuf,
    download: bool,
    device: Device,
    niimath: PathBuf,
    force: bool,
    quiet: bool,
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut a = std::env::args().skip(1);
    let (mut t1, mut flair, mut out, mut models) = (
        None,
        None,
        None,
        std::env::var_os("SHIVAI_MODELS").map(PathBuf::from),
    );
    let mut tasks = Vec::new();
    #[cfg(target_os = "macos")]
    let mut device = Device::Metal;
    #[cfg(not(target_os = "macos"))]
    let mut device = Device::Cpu;
    let mut niimath = std::env::var_os("SHIVAI_NIIMATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| "niimath".into());
    let (mut force, mut quiet) = (false, false);
    while let Some(arg) = a.next() {
        let mut value = || a.next().ok_or(format!("{arg} needs a value."));
        match arg.as_str() {
            "--t1" => t1 = Some(PathBuf::from(value()?)),
            "--flair" => flair = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--models" => models = Some(PathBuf::from(value()?)),
            "--niimath" => niimath = PathBuf::from(value()?),
            "--device" => {
                device = match value()?.as_str() {
                    "cpu" => Device::Cpu,
                    #[cfg(target_os = "macos")]
                    "metal" => Device::Metal,
                    d => return Err(format!("Unknown device {d}.")),
                }
            }
            "--pvs" => tasks.push("pvs"),
            "--wmh" => tasks.push("wmh"),
            "--lac" => tasks.push("lac"),
            "--force" => force = true,
            "--quiet" => quiet = true,
            "--help" | "-h" => {
                print!("{}", HELP.replace("VERSION", VERSION));
                return Ok(None);
            }
            "--version" | "-v" => {
                println!("shivai-rs {VERSION}");
                return Ok(None);
            }
            other => return Err(format!("Unknown option {other}; see --help.")),
        }
    }
    let t1 = t1.ok_or("--t1 is required; see --help.")?;
    let out = out.ok_or("--out is required; see --help.")?;
    if tasks.is_empty() {
        tasks.push("pvs");
        if flair.is_some() {
            tasks.push("wmh");
        }
    }
    tasks.sort_by_key(|t| ["pvs", "wmh", "lac"].iter().position(|x| x == t));
    tasks.dedup();
    if flair.is_none() && tasks.iter().any(|t| *t != "pvs") {
        return Err("--wmh and --lac need --flair.".into());
    }
    // A niimath installed beside shivai-rs (the macOS package) wins over PATH.
    if niimath == Path::new("niimath") {
        if let Some(sibling) = std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(|d| d.join("niimath")))
            .filter(|p| p.exists())
        {
            niimath = sibling;
        }
    }
    let (models, download) = match models {
        Some(dir) => (dir, false),
        None if Path::new(INSTALLED_MODELS).exists() => (PathBuf::from(INSTALLED_MODELS), false),
        None => (cache_dir()?, true),
    };
    Ok(Some(Args {
        t1,
        flair,
        out,
        tasks,
        models,
        download,
        device,
        niimath,
        force,
        quiet,
    }))
}

fn manifest() -> Value {
    serde_json::from_str(MANIFEST).unwrap()
}

fn cache_dir() -> Result<PathBuf, String> {
    let m = manifest();
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .ok_or("No cache folder; pass --models DIR.")?;
    Ok(base
        .join("shivai-rs")
        .join(m["revision"].as_str().unwrap_or("local")))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The manifest's files for one model, checksum-verified; with `download`, a missing or stale file is
/// fetched from the manifest's pinned Hugging Face revision first.
fn load(dir: &Path, model: &str, download: bool, say: &dyn Fn(&str)) -> Result<Vec<Model>, String> {
    let m = manifest();
    let files: Vec<&Value> = m["assets"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|a| a["model"] == model)
        .collect();
    let mut out = Vec::new();
    for f in files {
        let name = f["filename"].as_str().unwrap();
        let want = f["sha256"].as_str().unwrap();
        let path = dir.join(name);
        let mut bytes = std::fs::read(&path).ok();
        if download && bytes.as_deref().is_none_or(|b| sha256_hex(b) != want) {
            let url = format!("{}{name}", m["base_url"].as_str().unwrap());
            say(&format!(
                "Downloading {name} ({} MB)",
                f["bytes"].as_u64().unwrap() >> 20
            ));
            let agent: ureq::Agent = ureq::Agent::config_builder()
                .timeout_connect(Some(std::time::Duration::from_secs(30)))
                .timeout_recv_body(Some(std::time::Duration::from_secs(1800)))
                .build()
                .into();
            let data = agent
                .get(&url)
                .call()
                .map_err(|e| format!("Cannot download {url}: {e}"))?
                .into_body()
                .with_config()
                .limit(1 << 30)
                .read_to_vec()
                .map_err(|e| format!("Cannot download {url}: {e}"))?;
            if sha256_hex(&data) != want {
                return Err(format!("{name} from {url} does not match its checksum."));
            }
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("Cannot create {}: {e}", dir.display()))?;
            // Per process: parallel first runs share the cache.
            let partial = path.with_extension(format!("partial-{}", std::process::id()));
            std::fs::write(&partial, &data).map_err(|e| e.to_string())?;
            std::fs::rename(&partial, &path).map_err(|e| e.to_string())?;
            bytes = Some(data);
        }
        let bytes = bytes.ok_or(format!("Cannot read {}.", path.display()))?;
        if sha256_hex(&bytes) != want {
            return Err(format!("{name} does not match its checksum."));
        }
        out.push(parse(&bytes)?);
    }
    Ok(out)
}

enum Session {
    Cpu(cpu::Session),
    #[cfg(target_os = "macos")]
    Metal(metal::Session),
}

/// Runs ensembles; one session (one graph) is resident at a time.
struct Predictor<'a> {
    device: Device,
    current: Option<(String, Session)>,
    say: &'a dyn Fn(&str),
}

impl Predictor<'_> {
    fn run(&mut self, folds: &[Model], channels: &[&[f32]]) -> Result<Vec<f32>, String> {
        if self.current.as_ref().map(|(k, _)| k) != Some(&folds[0].signature) {
            self.current = None;
            let p = plan(&folds[0], GRID)?;
            let session = match self.device {
                Device::Cpu => Session::Cpu(cpu::Session::new(p)),
                #[cfg(target_os = "macos")]
                Device::Metal => {
                    let s = metal::Session::new(p)?;
                    (self.say)(&format!("Metal session {} MiB", s.bytes() >> 20));
                    Session::Metal(s)
                }
            };
            self.current = Some((folds[0].signature.clone(), session));
        }
        let input = to_network(channels);
        let mut ensemble = Ensemble::new();
        for fold in folds {
            let out = match &mut self.current.as_mut().unwrap().1 {
                Session::Cpu(s) => {
                    s.load(fold);
                    s.run(&input)
                }
                #[cfg(target_os = "macos")]
                Session::Metal(s) => {
                    s.load(fold);
                    s.run(&input)?
                }
            };
            ensemble.add(&out);
        }
        Ok(ensemble.mean())
    }
}

/// dual_preprocessing.py: the FLAIR registered onto the T1 crop by native niimath.
fn register(flair: &Path, prepared: &Prepared, niimath: &Path) -> Result<Volume, String> {
    // A fresh directory (create_dir fails if it exists), so nothing pre-placed is written or read.
    let work = (0..16)
        .find_map(|i| {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos());
            let p = std::env::temp_dir().join(format!("shivai-{}-{nanos}-{i}", std::process::id()));
            std::fs::create_dir(&p).ok().map(|_| p)
        })
        .ok_or("Cannot create a temporary folder.")?;
    let (fixed, out) = (
        work.join("t1_cropped.nii.gz"),
        work.join("flair_on_t1.nii.gz"),
    );
    let t = &prepared.t1_cropped;
    let result = (|| {
        std::fs::write(
            &fixed,
            nifti::write(Data::F32(&t.data), t.dims, &t.affine, "SHiVAi T1 crop"),
        )
        .map_err(|e| e.to_string())?;
        let run = Command::new(niimath)
            .arg(flair)
            .args(["-robustfov", "-allineate"])
            .arg(&fixed)
            .args(REGISTRATION)
            .arg(&out)
            .args(["-odt", "float"])
            .output()
            .map_err(|e| {
                format!(
                    "Cannot run niimath ({}): {e}. Install niimath or pass --niimath.",
                    niimath.display()
                )
            })?;
        if !run.status.success() {
            return Err(format!(
                "niimath registration failed (needs niimath v1.0.20261115 or later): {}",
                String::from_utf8_lossy(&run.stderr).trim()
            ));
        }
        nifti::read(&std::fs::read(&out).map_err(|e| e.to_string())?)
    })();
    let _ = std::fs::remove_dir_all(&work);
    result
}

fn run(a: Args) -> Result<(), String> {
    let started = Instant::now();
    let quiet = a.quiet;
    let say = move |m: &str| {
        if !quiet {
            eprintln!("[{:6.1}s] {m}", started.elapsed().as_secs_f32());
        }
    };
    let name = report::stem(
        &a.t1
            .file_name()
            .map_or("t1".into(), |n| n.to_string_lossy()),
    );
    let path = |suffix: &str| a.out.join(format!("{name}_{suffix}"));
    // Every file this run writes, checked before any work so nothing is replaced without --force.
    let planned: Vec<PathBuf> = report::suffixes(a.flair.is_some(), &a.tasks)
        .iter()
        .map(|s| path(s))
        .collect();
    // An input is never an output, even with --force (e.g. FLAIR named <stem>_flair.nii.gz in --out).
    let inputs: Vec<PathBuf> = [Some(&a.t1), a.flair.as_ref()]
        .into_iter()
        .flatten()
        .filter_map(|p| p.canonicalize().ok())
        .collect();
    for p in planned.iter().filter(|p| p.exists()) {
        if p.canonicalize().is_ok_and(|c| inputs.contains(&c)) {
            return Err(format!(
                "{} is an input; choose another --out.",
                p.display()
            ));
        }
        if !a.force {
            return Err(format!(
                "{} exists; pass --force to replace it.",
                p.display()
            ));
        }
    }
    let t1 = nifti::read(
        &std::fs::read(&a.t1).map_err(|e| format!("Cannot read {}: {e}", a.t1.display()))?,
    )?;
    let brainmask = load(&a.models, "brainmask", a.download, &say)?;
    let mut predictor = Predictor {
        device: a.device,
        current: None,
        say: &say,
    };
    say(&format!(
        "Device {:?}; T1 {:?} voxels; models {}",
        a.device,
        t1.dims,
        a.models.display()
    ));
    let (mut prep, input) = T1Prep::new(&t1);
    say("Brain mask, first pass");
    let input = prep.after_premask(predictor.run(&brainmask, &[&input])?)?;
    say("Brain mask, second pass");
    let prepared = prep.after_mask(predictor.run(&brainmask, &[&input])?)?;
    for w in &prepared.warnings {
        say(&format!("Warning: {w}"));
    }
    let mut report = Report::new(&prepared);
    let flair = match &a.flair {
        Some(f) => {
            say(&format!(
                "Registering the FLAIR to the T1 ({})",
                a.niimath.display()
            ));
            let v = prepared.secondary(&register(f, &prepared, &a.niimath)?)?;
            report.add_flair(&v);
            Some(v)
        }
        None => None,
    };
    for id in &a.tasks {
        let task = pipeline::task(id).unwrap();
        let folds = load(&a.models, id, a.download, &say)?;
        let mut channels: Vec<&[f32]> = vec![&prepared.t1.data];
        if task.flair {
            channels.push(&flair.as_ref().unwrap().data);
        }
        say(&format!("{} ({} folds)", task.label, folds.len()));
        let s = prepared.segment(task, predictor.run(&folds, &channels)?);
        say(&format!(
            "{}: {} clusters, {:.2} ml",
            task.label,
            s.sizes.len(),
            s.volume_mm3 / 1000.0
        ));
        report.add_marker(task, &s);
    }
    let file_name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().into_owned());
    let outputs = report.finish(json!({
        "software": "SHiVAi-rs command line",
        "backend": format!("{:?}", a.device),
        "t1": file_name(&a.t1),
        "flair": a.flair.as_deref().and_then(file_name),
        "seconds": (started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
    }));
    std::fs::create_dir_all(&a.out)
        .map_err(|e| format!("Cannot create {}: {e}", a.out.display()))?;
    for (suffix, bytes) in &outputs {
        let p = &path(suffix);
        let partial = p.with_extension("partial");
        std::fs::write(&partial, bytes)
            .map_err(|e| format!("Cannot write {}: {e}", p.display()))?;
        std::fs::rename(&partial, p).map_err(|e| format!("Cannot write {}: {e}", p.display()))?;
    }
    say(&format!("Done in {:.1} s", started.elapsed().as_secs_f32()));
    Ok(())
}

fn main() {
    if let Err(e) = parse_args().and_then(|a| a.map(run).transpose()) {
        eprintln!("shivai-rs: {e}");
        std::process::exit(1);
    }
}
