// The SHiVAi pipeline for the browser. The stages, plan, kernels and post-processing are shivai-core;
// JavaScript runs the networks on WebGPU (a generic dispatcher over GpuPlan) and the FLAIR registration
// (niimath WebAssembly), and calls back into Job between them:
//
//   job = new Job(t1Nifti); input = job.first_input()
//   input = job.after_premask(await ensemble('brainmask', input))
//   job.after_mask(await ensemble('brainmask', input))
//   job.set_flair(await niimath(flair, job.registration_target()))
//   job.add_marker(task, await ensemble(task, job.task_input(task)))
//   job.complete(runJson); job.output_names() / job.output(name)
use serde_json::json;
use shivai_core::kernels::{kernel, Binding, Lang};
use shivai_core::nifti::{self, Data};
use shivai_core::pipeline::{self, to_network, Prepared, T1Prep, GRID};
use shivai_core::plan::{pack_bias, pack_weights, plan, to_channels_last, Plan};
use shivai_core::preprocess::Volume;
use shivai_core::report::Report;
use shivai_core::shiva::{half_to_f32, parse, to_half, Op};
use wasm_bindgen::prelude::*;

fn js<E: std::fmt::Display>(e: E) -> JsError {
    JsError::new(&e.to_string())
}

static PANIC: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// A panic aborts with WebAssembly's bare "unreachable"; keep its message for `panicMessage`.
#[wasm_bindgen(start)]
fn start() {
    std::panic::set_hook(Box::new(|info| {
        if let Ok(mut m) = PANIC.lock() {
            *m = Some(info.to_string());
        }
    }));
}

/// The message of the last Rust panic, if any.
#[wasm_bindgen(js_name = panicMessage)]
pub fn panic_message() -> Option<String> {
    PANIC.lock().ok()?.clone()
}

/// niimath -allineate options for the FLAIR (SHiVA's rigid ANTs step).
#[wasm_bindgen]
pub fn registration() -> Vec<String> {
    pipeline::REGISTRATION
        .iter()
        .map(|s| s.to_string())
        .collect()
}

#[wasm_bindgen]
pub fn manifest() -> String {
    shivai_core::MANIFEST.to_string()
}

#[wasm_bindgen]
pub struct Job {
    prep: Option<T1Prep>,
    first: Vec<f32>,
    prepared: Option<Prepared>,
    flair: Option<Volume>,
    report: Option<Report>,
    outputs: Vec<(String, Vec<u8>)>,
}

#[wasm_bindgen]
impl Job {
    #[wasm_bindgen(constructor)]
    pub fn new(t1_nifti: &[u8]) -> Result<Job, JsError> {
        let t1 = nifti::read(t1_nifti).map_err(js)?;
        let (prep, first) = T1Prep::new(&t1);
        Ok(Job {
            prep: Some(prep),
            first,
            prepared: None,
            flair: None,
            report: None,
            outputs: vec![],
        })
    }

    /// The first brain-mask input (network order).
    pub fn first_input(&self) -> Vec<f32> {
        to_network(&[&self.first])
    }

    /// The pre-mask (ensemble mean, NIfTI order) to the second brain-mask input (network order).
    pub fn after_premask(&mut self, premask: Vec<f32>) -> Result<Vec<f32>, JsError> {
        let input = self
            .prep
            .as_mut()
            .ok_or_else(|| js("out of order"))?
            .after_premask(premask)
            .map_err(js)?;
        Ok(to_network(&[&input]))
    }

    pub fn after_mask(&mut self, mask: Vec<f32>) -> Result<(), JsError> {
        let prepared = self
            .prep
            .take()
            .ok_or_else(|| js("out of order"))?
            .after_mask(mask)
            .map_err(js)?;
        self.report = Some(Report::new(&prepared));
        self.prepared = Some(prepared);
        Ok(())
    }

    pub fn warnings(&self) -> Vec<String> {
        self.prepared
            .as_ref()
            .map(|p| p.warnings.clone())
            .unwrap_or_default()
    }

    /// The cropped T1 (gzipped NIfTI): the target the FLAIR is registered to.
    pub fn registration_target(&self) -> Result<Vec<u8>, JsError> {
        let t = &self
            .prepared
            .as_ref()
            .ok_or_else(|| js("out of order"))?
            .t1_cropped;
        Ok(nifti::write(
            Data::F32(&t.data),
            t.dims,
            &t.affine,
            "SHiVAi T1 crop",
        ))
    }

    /// The FLAIR resliced onto the registration target (NIfTI bytes).
    pub fn set_flair(&mut self, registered: &[u8]) -> Result<(), JsError> {
        let p = self.prepared.as_ref().ok_or_else(|| js("out of order"))?;
        let v = p
            .secondary(&nifti::read(registered).map_err(js)?)
            .map_err(js)?;
        self.report.as_mut().unwrap().add_flair(&v);
        self.flair = Some(v);
        Ok(())
    }

    /// The network input for a marker (network order).
    pub fn task_input(&self, task: &str) -> Result<Vec<f32>, JsError> {
        let p = self.prepared.as_ref().ok_or_else(|| js("out of order"))?;
        let t = pipeline::task(task).ok_or_else(|| js(format!("unknown marker {task}")))?;
        if t.flair {
            let f = self
                .flair
                .as_ref()
                .ok_or_else(|| js(format!("{} needs a FLAIR", t.label)))?;
            Ok(to_network(&[&p.t1.data, &f.data]))
        } else {
            Ok(to_network(&[&p.t1.data]))
        }
    }

    /// The ensemble mean (NIfTI order) to clusters; returns the marker summary as JSON.
    pub fn add_marker(&mut self, task: &str, mean: Vec<f32>) -> Result<String, JsError> {
        let p = self.prepared.as_ref().ok_or_else(|| js("out of order"))?;
        let t = pipeline::task(task).ok_or_else(|| js(format!("unknown marker {task}")))?;
        let s = p.segment(t, mean);
        Ok(self.report.as_mut().unwrap().add_marker(t, &s).to_string())
    }

    /// Adds the summary table and provenance; `run` is JSON with software, backend, t1, flair, seconds.
    pub fn complete(&mut self, run: &str) -> Result<(), JsError> {
        let report = self.report.take().ok_or_else(|| js("out of order"))?;
        self.outputs = report.finish(serde_json::from_str(run).map_err(js)?);
        Ok(())
    }

    /// Output suffixes (file names are `<stem>_<suffix>`).
    pub fn output_names(&self) -> Vec<String> {
        self.outputs.iter().map(|(n, _)| n.clone()).collect()
    }

    pub fn output(&self, name: &str) -> Option<Vec<u8>> {
        self.outputs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, b)| b.clone())
    }
}

/// Fold mean on the NIfTI grid; each fold's values below 0.001 are 0.
#[wasm_bindgen]
pub struct Ensemble(pipeline::Ensemble);

#[wasm_bindgen]
impl Ensemble {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Ensemble {
        Ensemble(pipeline::Ensemble::new())
    }
    /// One fold's output (network order).
    pub fn add(&mut self, fold: &[f32]) {
        self.0.add(fold);
    }
    pub fn mean(self) -> Vec<f32> {
        self.0.mean()
    }
}

impl Default for Ensemble {
    fn default() -> Self {
        Self::new()
    }
}

/// One network's WebGPU program at the pipeline grid: WGSL per step, buffers, packed weights.
#[wasm_bindgen]
pub struct GpuPlan {
    plan: Plan,
    half: bool,
    signature: String,
}

#[wasm_bindgen]
impl GpuPlan {
    /// `half`: store activations as f16 (needs shader-f16); arithmetic stays f32.
    #[wasm_bindgen(constructor)]
    pub fn new(model: &[u8], half: bool) -> Result<GpuPlan, JsError> {
        Self::for_dims(model, half, GRID[0], GRID[1], GRID[2])
    }

    /// Another grid (the conformance test runs a small one).
    #[wasm_bindgen(js_name = forDims)]
    pub fn for_dims(
        model: &[u8],
        half: bool,
        d: usize,
        h: usize,
        w: usize,
    ) -> Result<GpuPlan, JsError> {
        let m = parse(model).map_err(js)?;
        let plan = plan(&m, [d, h, w]).map_err(js)?;
        Ok(GpuPlan {
            plan,
            half,
            signature: m.signature,
        })
    }

    /// Steps (WGSL, dispatch, bindings, weight lengths), slot sizes in bytes, input and output layout.
    pub fn program(&self) -> String {
        let bytes = if self.half { 2 } else { 4 };
        let steps: Vec<_> = self
            .plan
            .steps
            .iter()
            .map(|s| {
                let k = kernel(s, Lang::Wgsl { half: self.half });
                let bindings: Vec<_> = k
                    .bindings
                    .iter()
                    .map(|b| match b {
                        Binding::Slot(i) => json!({ "slot": i }),
                        Binding::Weights => json!("weights"),
                        Binding::Biases => json!("biases"),
                    })
                    .collect();
                let (w, b) = if s.node.op == Op::Concat { (0, 0) } else { (s.weight_len(), s.shape.padded) };
                json!({ "name": s.node.out, "code": k.code, "groups": k.groups, "bindings": bindings, "weights": w, "biases": b })
            })
            .collect();
        json!({
            "slots": self.plan.slots.iter().map(|n| n * bytes).collect::<Vec<_>>(),
            "steps": steps,
            "inputSlot": 0,
            "outputSlot": self.plan.output_slot,
            "outputBytes": self.plan.output.voxels() * self.plan.output.padded * bytes,
        })
        .to_string()
    }

    /// Graphs with equal signatures share a program; only their weights differ.
    pub fn signature(&self) -> String {
        self.signature.clone()
    }

    /// Every step's packed weights then biases, concatenated in step order (concat steps add nothing).
    pub fn weights(&self, model: &[u8]) -> Result<Vec<f32>, JsError> {
        let m = parse(model).map_err(js)?;
        if m.signature != self.signature {
            return Err(js("these weights belong to another graph"));
        }
        let mut out = Vec::new();
        for s in self.plan.steps.iter().filter(|s| s.node.op != Op::Concat) {
            out.extend(pack_weights(s, &m.weights));
            out.extend(pack_bias(s, &m.weights));
        }
        Ok(out)
    }

    /// Network input (NCDHW) to the input slot's bytes.
    pub fn pack_input(&self, input: &[f32]) -> Vec<u8> {
        let p = &self.plan;
        let v = to_channels_last(input, p.input.channels, p.input.padded);
        if self.half {
            v.iter().flat_map(|&x| to_half(x).to_le_bytes()).collect()
        } else {
            v.iter().flat_map(|x| x.to_le_bytes()).collect()
        }
    }

    /// The output slot's bytes to the first output channel (network order).
    pub fn unpack_output(&self, raw: &[u8]) -> Vec<f32> {
        let padded = self.plan.output.padded;
        let n = self.plan.output.voxels();
        if self.half {
            (0..n)
                .map(|v| {
                    half_to_f32(u16::from_le_bytes([
                        raw[2 * v * padded],
                        raw[2 * v * padded + 1],
                    ]))
                })
                .collect()
        } else {
            (0..n)
                .map(|v| {
                    f32::from_le_bytes(raw[4 * v * padded..4 * v * padded + 4].try_into().unwrap())
                })
                .collect()
        }
    }
}
