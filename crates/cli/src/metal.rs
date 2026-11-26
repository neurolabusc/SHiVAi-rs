// Metal executor: runs the core's MSL kernels over the shared plan (IEEE FP32, no fast math).
use metal::*;
use shivai_core::kernels::{kernel, Binding, Lang};
use shivai_core::plan::{pack_bias, pack_weights, to_channels_last, Plan};
use shivai_core::shiva::{Model, Op};

struct Step {
    pipeline: ComputePipelineState,
    groups: MTLSize,
    threads: MTLSize,
    bindings: Vec<Binding>,
    weights: Option<(Buffer, Buffer)>,
}

pub struct Session {
    plan: Plan,
    queue: CommandQueue,
    slots: Vec<Buffer>,
    steps: Vec<Step>,
}

fn write(buffer: &Buffer, data: &[f32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.contents().cast::<f32>(), data.len())
    }
}

impl Session {
    pub fn new(plan: Plan) -> Result<Self, String> {
        let device =
            Device::system_default().ok_or("No Metal device is available; use --device cpu.")?;
        let largest = plan.slots.iter().max().copied().unwrap_or(0) as u64 * 4;
        if largest > device.max_buffer_length() {
            return Err(format!(
                "This model needs a {} MiB buffer; this GPU allows {} MiB.",
                largest >> 20,
                device.max_buffer_length() >> 20
            ));
        }
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let shared = MTLResourceOptions::StorageModeShared;
        let slots = plan
            .slots
            .iter()
            .map(|&n| device.new_buffer((n as u64 * 4).max(16), shared))
            .collect();
        let mut steps = Vec::new();
        for step in &plan.steps {
            let k = kernel(step, Lang::Msl);
            let library = device
                .new_library_with_source(&k.code, &opts)
                .map_err(|e| format!("{}: {e}", step.node.out))?;
            let function = library.get_function("main0", None)?;
            let pipeline = device.new_compute_pipeline_state_with_function(&function)?;
            let weights = (step.node.op != Op::Concat).then(|| {
                let n = (step.weight_len() * 4) as u64;
                (
                    device.new_buffer(n, shared),
                    device.new_buffer(step.shape.padded as u64 * 4, shared),
                )
            });
            let size = |v: [u32; 3]| MTLSize::new(v[0] as u64, v[1] as u64, v[2] as u64);
            steps.push(Step {
                pipeline,
                groups: size(k.groups),
                threads: size(k.threads),
                bindings: k.bindings,
                weights,
            });
        }
        Ok(Self {
            plan,
            queue: device.new_command_queue(),
            slots,
            steps,
        })
    }

    /// Bytes of activation and weight buffers.
    pub fn bytes(&self) -> u64 {
        self.slots.iter().map(|b| b.length()).sum::<u64>()
            + self
                .steps
                .iter()
                .filter_map(|s| s.weights.as_ref())
                .map(|(w, b)| w.length() + b.length())
                .sum::<u64>()
    }

    pub fn load(&mut self, model: &Model) {
        for (step, s) in self.plan.steps.iter().zip(&self.steps) {
            if let Some((w, b)) = &s.weights {
                write(w, &pack_weights(step, &model.weights));
                write(b, &pack_bias(step, &model.weights));
            }
        }
    }

    /// NCDHW input (z fastest) -> the first output channel, same order.
    pub fn run(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let p = &self.plan;
        write(
            &self.slots[0],
            &to_channels_last(input, p.input.channels, p.input.padded),
        );
        // One command buffer per layer (a whole volume can hold the GPU for seconds); wait every 8.
        let mut pending = Vec::with_capacity(8);
        for (i, (step, s)) in p.steps.iter().zip(&self.steps).enumerate() {
            let command = self.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&s.pipeline);
            for (index, b) in s.bindings.iter().enumerate() {
                let buffer = match b {
                    Binding::Slot(slot) => &self.slots[*slot],
                    Binding::Weights => &s.weights.as_ref().unwrap().0,
                    Binding::Biases => &s.weights.as_ref().unwrap().1,
                };
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            encoder.dispatch_thread_groups(s.groups, s.threads);
            encoder.end_encoding();
            command.commit();
            pending.push((&step.node.out, command.to_owned()));
            if pending.len() == 8 || i + 1 == self.steps.len() {
                let mut failed = None;
                for (name, command) in pending.drain(..) {
                    command.wait_until_completed();
                    if command.status() == MTLCommandBufferStatus::Error {
                        failed.get_or_insert(name);
                    }
                }
                if let Some(name) = failed {
                    return Err(format!("Metal failed at {name}."));
                }
            }
        }
        let out = self.slots[p.output_slot].contents().cast::<f32>();
        Ok((0..p.output.voxels())
            .map(|v| unsafe { *out.add(v * p.output.padded) })
            .collect())
    }
}
