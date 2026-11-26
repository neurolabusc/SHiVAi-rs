// SHiVAi's segmentation pipeline (default configuration, SHiVA brain mask) as resumable stages, so a
// synchronous native driver and an asynchronous browser driver run identical steps around their own
// network execution and registration:
//
//   let (mut prep, input) = T1Prep::new(&t1);          // input: brain-mask pass 1 (GRID, x fastest)
//   let input = prep.after_premask(network(input))?;   // input: pass 2
//   let prepared = prep.after_mask(network(input))?;
//   let flair = prepared.secondary(&registered_flair);  // FLAIR resliced onto prepared.t1_cropped
//   let seg = prepared.segment(task, network(channels));
//
// Networks see NCDHW in C order over (x, y, z); `to_network`/`from_network` convert. An ensemble is
// the mean of its folds with each fold's values below 0.001 set to 0 (scripts/predict_multi.py).
use crate::preprocess::*;

pub const GRID: [usize; 3] = [160, 214, 176];
/// niimath -allineate options for a second contrast: SHiVA's rigid ANTs step (dual_preprocessing.py)
/// as AFNI-style shift+rotate with mutual information. The FLAIR is first cropped with `-robustfov`
/// (neck removed): ~8% faster, WMH unchanged on HML0541, and robust to a wide field of view.
pub const REGISTRATION: [&str; 4] = ["-warp", "shr", "-cost", "nmi"];
const CONFORMED: [usize; 3] = [256, 256, 256];

pub struct Task {
    pub id: &'static str,
    pub label: &'static str,
    pub flair: bool,
    /// Compared in f32, as numpy compares SHiVAi's float32 maps; f64 so provenance records 0.2, not 0.2000000029.
    pub threshold: f64,
    pub min_size: usize,
}

// SHiVAi's defaults (utils/parsing.py).
pub const TASKS: [Task; 3] = [
    Task {
        id: "pvs",
        label: "Perivascular spaces",
        flair: false,
        threshold: 0.5,
        min_size: 5,
    },
    Task {
        id: "wmh",
        label: "White matter hyperintensities",
        flair: true,
        threshold: 0.2,
        min_size: 1,
    },
    Task {
        id: "lac",
        label: "Lacunes",
        flair: true,
        threshold: 0.2,
        min_size: 3,
    },
];

pub fn task(id: &str) -> Option<&'static Task> {
    TASKS.iter().find(|t| t.id == id)
}

fn transpose(data: &[f32], [nx, ny, nz]: [usize; 3], to_network: bool, out: &mut [f32]) {
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let (nifti, network) = (x + nx * (y + ny * z), (x * ny + y) * nz + z);
                if to_network {
                    out[network] = data[nifti];
                } else {
                    out[nifti] = data[network];
                }
            }
        }
    }
}

/// Channels (each a GRID volume, x fastest) to one NCDHW network input.
pub fn to_network(channels: &[&[f32]]) -> Vec<f32> {
    let v = channels[0].len();
    let mut out = vec![0f32; v * channels.len()];
    for (i, c) in channels.iter().enumerate() {
        transpose(c, GRID, true, &mut out[i * v..(i + 1) * v]);
    }
    out
}

/// Accumulates fold outputs (network order) into an ensemble mean on the NIfTI grid.
pub struct Ensemble {
    sum: Vec<f32>,
    folds: usize,
}

impl Default for Ensemble {
    fn default() -> Self {
        Self::new()
    }
}

impl Ensemble {
    pub fn new() -> Self {
        Self {
            sum: vec![0.0; GRID.iter().product()],
            folds: 0,
        }
    }
    pub fn add(&mut self, fold: &[f32]) {
        for (s, &v) in self.sum.iter_mut().zip(fold) {
            if v >= 0.001 {
                *s += v;
            }
        }
        self.folds += 1;
    }
    pub fn mean(self) -> Vec<f32> {
        let n = self.folds.max(1) as f32;
        let mean: Vec<f32> = self.sum.iter().map(|s| s / n).collect();
        let mut out = vec![0f32; mean.len()];
        transpose(&mean, GRID, false, &mut out);
        out
    }
}

/// shiva_mask_wf.py and preprocessing.py, between the network passes.
pub struct T1Prep {
    conformed: Volume,
    pre_affine: Affine,
    first_affine: Affine,
    warnings: Vec<String>,
}

impl T1Prep {
    /// Conforms the T1 and returns the first brain-mask input (the whole field of view on GRID).
    pub fn new(t1: &Volume) -> (Self, Vec<f32>) {
        let conformed = conform(t1, CONFORMED, Some([1.0; 3]), 3);
        let pre = conform(t1, GRID, None, 3);
        let input = normalize(&pre.data, None);
        (
            Self {
                conformed,
                pre_affine: pre.affine,
                first_affine: [[0.0; 4]; 4],
                warnings: vec![],
            },
            input,
        )
    }

    /// The pre-mask (network output on GRID) to the second brain-mask input, cropped on it.
    pub fn after_premask(&mut self, premask: Vec<f32>) -> Result<Vec<f32>, String> {
        let premask = conform(
            &Volume {
                data: premask,
                dims: GRID,
                affine: self.pre_affine,
                pixdim: None,
            },
            CONFORMED,
            Some([1.0; 3]),
            0,
        );
        let bin = threshold_mask(
            &premask.data,
            CONFORMED,
            voxel_sizes(&premask.affine),
            0.5,
            3,
            600000.0,
            &mut self.warnings,
        );
        let b = crop_box(&bin, CONFORMED, GRID)?; // first: an empty mask is an error, not a panic
        let normalised = normalize(&self.conformed.data, Some(&bin));
        let (data, affine) = crop(&normalised, CONFORMED, &self.conformed.affine, &b);
        self.first_affine = affine;
        Ok(data)
    }

    /// The brain mask (network output on the second crop) to the prepared T1.
    pub fn after_mask(self, mask: Vec<f32>) -> Result<Prepared, String> {
        let mut warnings = self.warnings;
        let c = &self.conformed;
        let mask_conformed = resample_to(&mask, GRID, &self.first_affine, CONFORMED, &c.affine, 0);
        let brain = threshold_mask(
            &mask_conformed,
            CONFORMED,
            voxel_sizes(&c.affine),
            0.5,
            3,
            30000.0,
            &mut warnings,
        );
        let b = crop_box(&brain, CONFORMED, GRID)?;
        let (cropped, affine) = crop(
            &quickshear(&c.data, CONFORMED, &brain),
            CONFORMED,
            &c.affine,
            &b,
        );
        let (brain_cropped, _) = crop(&brain, CONFORMED, &c.affine, &b);
        Ok(Prepared {
            t1: Volume {
                data: normalize(&cropped, Some(&brain_cropped)),
                dims: GRID,
                affine,
                pixdim: None,
            },
            t1_cropped: Volume {
                data: cropped,
                dims: GRID,
                affine,
                pixdim: None,
            },
            brain: brain_cropped,
            warnings,
        })
    }
}

pub struct Prepared {
    /// Conformed, defaced, cropped on the brain, normalised: the network's T1 channel.
    pub t1: Volume,
    /// The same before normalisation: the registration target.
    pub t1_cropped: Volume,
    /// The brain mask on the same grid.
    pub brain: Vec<u8>,
    pub warnings: Vec<String>,
}

pub struct Segmentation {
    pub probability: Vec<f32>,
    pub labels: Vec<i16>,
    pub sizes: Vec<usize>,
    pub volume_mm3: f64,
}

impl Prepared {
    /// dual_preprocessing.py: a second contrast already resliced onto `t1_cropped` (rigid
    /// registration), defaced and normalised with the crop's brain mask.
    pub fn secondary(&self, registered: &Volume) -> Result<Volume, String> {
        if registered.dims != GRID {
            return Err("The registered image is not on the T1 crop grid.".into());
        }
        let defaced = quickshear(&registered.data, GRID, &self.brain);
        Ok(Volume {
            data: normalize(&defaced, Some(&self.brain)),
            dims: GRID,
            affine: self.t1.affine,
            pixdim: None,
        })
    }

    pub fn voxel_mm3(&self) -> f64 {
        let m = &self.t1.affine;
        (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
            .abs()
    }

    /// postprocessing/clusters.py on an ensemble mean.
    pub fn segment(&self, task: &Task, probability: Vec<f32>) -> Segmentation {
        let (labels, sizes) = label_clusters(
            &probability,
            GRID,
            &self.brain,
            task.threshold as f32,
            task.min_size,
        );
        let volume_mm3 = sizes.iter().sum::<usize>() as f64 * self.voxel_mm3();
        Segmentation {
            probability,
            labels,
            sizes,
            volume_mm3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A pre-mask with nothing above threshold (e.g. a non-T1 image in the T1 slot) is an error.
    #[test]
    fn empty_premask_is_an_error_not_a_panic() {
        let mut affine = [[0.0; 4]; 4];
        (0..4).for_each(|i| affine[i][i] = 1.0);
        let t1 = Volume {
            data: vec![1.0; 32 * 32 * 32],
            dims: [32; 3],
            affine,
            pixdim: Some([1.0; 3]),
        };
        let (mut prep, _) = T1Prep::new(&t1);
        let err = prep
            .after_premask(vec![0.0; GRID.iter().product()])
            .unwrap_err();
        assert!(err.contains("empty"), "{err}");
    }
}
