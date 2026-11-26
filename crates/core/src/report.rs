// Everything a run writes, named and formatted once for the CLI and the web app. Files are
// `<stem>_<suffix>`; the suffixes, the summary columns and the provenance keys are a public contract
// (bump SCHEMA on a breaking change). Outputs are on the cropped 1 mm T1 grid; other spaces, if
// added, get new suffixes.
use crate::nifti::{self, Data};
use crate::pipeline::{Prepared, Segmentation, Task, GRID, REGISTRATION};
use crate::preprocess::Volume;
use serde_json::{json, Value};

pub const SCHEMA: u32 = 1;

/// `<stem>` of the output names: the T1 file name without `.nii` / `.nii.gz` (any case).
pub fn stem(t1_name: &str) -> String {
    let lower = t1_name.to_ascii_lowercase();
    let cut = [".nii.gz", ".nii"]
        .iter()
        .find(|e| lower.ends_with(*e))
        .map_or(0, |e| e.len());
    t1_name[..t1_name.len() - cut].to_string()
}

/// Every suffix a run with these inputs writes (the CLI checks them before any work).
pub fn suffixes(flair: bool, markers: &[&str]) -> Vec<String> {
    let mut s = vec!["t1.nii.gz".to_string(), "brainmask.nii.gz".into()];
    if flair {
        s.push("flair.nii.gz".into());
    }
    for m in markers {
        s.push(format!("{m}_probability.nii.gz"));
        s.push(format!("{m}_clusters.nii.gz"));
    }
    s.extend(["summary.tsv".into(), "shivai.json".into()]);
    s
}

fn registration_command() -> String {
    format!("niimath -robustfov -allineate {}", REGISTRATION.join(" "))
}

pub struct Report {
    files: Vec<(String, Vec<u8>)>,
    markers: Vec<Value>,
    warnings: Vec<String>,
    flair: bool,
    affine: crate::preprocess::Affine,
}

impl Report {
    pub fn new(p: &Prepared) -> Self {
        let a = p.t1.affine;
        Self {
            files: vec![
                (
                    "t1.nii.gz".into(),
                    nifti::write(
                        Data::F32(&p.t1.data),
                        GRID,
                        &a,
                        "SHiVAi T1: conformed, defaced, cropped, normalised",
                    ),
                ),
                (
                    "brainmask.nii.gz".into(),
                    nifti::write(Data::U8(&p.brain), GRID, &a, "SHiVAi brain mask"),
                ),
            ],
            markers: vec![],
            warnings: p.warnings.clone(),
            flair: false,
            affine: a,
        }
    }

    /// The FLAIR from `Prepared::secondary`.
    pub fn add_flair(&mut self, flair: &Volume) {
        self.flair = true;
        self.files.push((
            "flair.nii.gz".into(),
            nifti::write(
                Data::F32(&flair.data),
                GRID,
                &self.affine,
                "SHiVAi FLAIR on the T1 crop, normalised",
            ),
        ));
    }

    /// Adds a marker's images; returns its summary (as recorded in provenance).
    pub fn add_marker(&mut self, task: &Task, s: &Segmentation) -> Value {
        let id = task.id;
        self.files.push((
            format!("{id}_probability.nii.gz"),
            nifti::write(
                Data::F32(&s.probability),
                GRID,
                &self.affine,
                &format!("SHiVAi {id} probability"),
            ),
        ));
        self.files.push((
            format!("{id}_clusters.nii.gz"),
            nifti::write(
                Data::I16(&s.labels),
                GRID,
                &self.affine,
                &format!("SHiVAi {id} clusters"),
            ),
        ));
        let m = json!({ "marker": id, "label": task.label, "clusters": s.sizes.len(),
            "volume_mm3": (s.volume_mm3 * 100.0).round() / 100.0, "threshold": task.threshold, "min_size": task.min_size,
            "cluster_voxels": s.sizes });
        self.markers.push(m.clone());
        m
    }

    /// The files, `(suffix, bytes)` in `suffixes` order, with the summary table and provenance. `run`
    /// holds the driver's facts: `software`, `backend`, `t1`, `flair` (input file names), `seconds`.
    pub fn finish(mut self, run: Value) -> Vec<(String, Vec<u8>)> {
        let mut tsv = String::from("marker\tclusters\tvolume_mm3\n");
        for m in &self.markers {
            tsv += &format!(
                "{}\t{}\t{}\n",
                m["marker"].as_str().unwrap(),
                m["clusters"],
                m["volume_mm3"]
            );
        }
        let manifest: Value = serde_json::from_str(crate::MANIFEST).unwrap();
        let sha256: serde_json::Map<String, Value> = manifest["assets"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|a| {
                a["model"] == "brainmask" || self.markers.iter().any(|m| m["marker"] == a["model"])
            })
            .map(|a| {
                (
                    a["filename"].as_str().unwrap().to_string(),
                    a["sha256"].clone(),
                )
            })
            .collect();
        let mut provenance = json!({
            "schema": SCHEMA,
            "version": env!("CARGO_PKG_VERSION"),
            "registration": self.flair.then(registration_command),
            "models": { "license": manifest["license"], "license_owner": manifest["license_owner"],
                        "revision": manifest["revision"], "sha256": sha256 },
            "markers": self.markers,
            "warnings": self.warnings,
        });
        if let (Value::Object(p), Value::Object(r)) = (&mut provenance, run) {
            p.extend(r);
        }
        self.files.push(("summary.tsv".into(), tsv.into_bytes()));
        self.files.push((
            "shivai.json".into(),
            serde_json::to_vec_pretty(&provenance).unwrap(),
        ));
        self.files
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{task, Prepared, Segmentation};

    // The CLI refuses to overwrite only what `suffixes` lists, so it must match what is written.
    #[test]
    fn suffixes_match_written_files() {
        let v = Volume {
            data: vec![0.0],
            dims: [1; 3],
            affine: [[0.0; 4]; 4],
            pixdim: None,
        };
        let p = Prepared {
            t1: v.clone(),
            t1_cropped: v.clone(),
            brain: vec![0],
            warnings: vec![],
        };
        let s = Segmentation {
            probability: vec![0.0],
            labels: vec![0],
            sizes: vec![],
            volume_mm3: 0.0,
        };
        let mut r = Report::new(&p);
        r.add_flair(&v);
        for id in ["pvs", "wmh"] {
            r.add_marker(task(id).unwrap(), &s);
        }
        let written: Vec<_> = r.finish(json!({})).into_iter().map(|(n, _)| n).collect();
        assert_eq!(written, suffixes(true, &["pvs", "wmh"]));
    }
}
