// Execution plan of a .shiva graph for one input size: shapes, the
// channel padding the vector kernels use, and activation buffers shared by lifetime. Activations
// are NDHWC with channels padded; padding channels hold zeros (zero weights and biases).
use crate::shiva::{Model, Node, Op};
use std::collections::HashMap;

pub fn padded_channels(c: usize) -> usize {
    if c <= 4 {
        4
    } else {
        c.div_ceil(8) * 8
    }
}

pub fn same_pad_begin(n: usize, k: usize, s: usize) -> usize {
    let out = n.div_ceil(s);
    ((out - 1) * s + k).saturating_sub(n) / 2
}

pub fn output_dims(node: &Node, dims: [usize; 3]) -> Option<[usize; 3]> {
    let mut out = [0; 3];
    for a in 0..3 {
        let n = dims[a];
        out[a] = match node.op {
            Op::Conv { stride } if node.same => n.div_ceil(stride),
            Op::Conv { stride } => (n.checked_sub(node.k)?) / stride + 1,
            Op::ConvT => 2 * n + if node.same { 0 } else { node.k - 2 },
            Op::Concat => n,
        };
    }
    Some(out)
}

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub dims: [usize; 3],
    pub channels: usize,
    pub padded: usize,
}

impl Step {
    /// Packed weight elements of a convolution step (pack_weights).
    pub fn weight_len(&self) -> usize {
        self.node.k.pow(3) * self.inputs[0].padded * self.shape.padded
    }
}

impl Shape {
    pub fn voxels(&self) -> usize {
        self.dims.iter().product()
    }
}

pub struct Step {
    pub node: Node,
    pub inputs: Vec<Shape>,
    pub shape: Shape,
    pub pad_begin: [usize; 3],
    pub crop: [i64; 3],
    pub in_place: bool,
    pub slot: usize,
    pub input_slots: Vec<usize>, // x (one or two), then the residual
}

pub struct Plan {
    pub steps: Vec<Step>,
    pub slots: Vec<usize>, // elements per slot
    pub input: Shape,
    pub output: Shape,
    pub output_slot: usize,
}

pub fn plan(model: &Model, dims: [usize; 3]) -> Result<Plan, String> {
    let input = Shape {
        dims,
        channels: model.input_channels,
        padded: padded_channels(model.input_channels),
    };
    let mut tensors: HashMap<String, Shape> = HashMap::from([("input".to_string(), input)]);
    let get = |t: &HashMap<String, Shape>, n: &str| {
        t.get(n).copied().ok_or(format!("Unknown tensor {n}."))
    };
    let mut steps = Vec::new();
    for node in &model.nodes {
        let inputs: Vec<Shape> = node
            .x
            .iter()
            .map(|n| get(&tensors, n))
            .collect::<Result<_, _>>()?;
        let (shape_dims, channels, crop, pad_begin) = if node.op == Op::Concat {
            let (a, b) = (inputs[0], inputs[1]);
            let crop = std::array::from_fn(|i| a.dims[i] as i64 - b.dims[i] as i64);
            (b.dims, a.channels + b.channels, crop, [0; 3])
        } else {
            if inputs[0].channels != node.cin {
                return Err(format!(
                    "{}: expected {} channels, got {}.",
                    node.out, node.cin, inputs[0].channels
                ));
            }
            let out = output_dims(node, inputs[0].dims)
                .filter(|d| d.iter().all(|&n| n > 0))
                .ok_or(format!("{}: input too small.", node.out))?;
            let pad = match node.op {
                Op::Conv { stride } if node.same => {
                    std::array::from_fn(|i| same_pad_begin(inputs[0].dims[i], node.k, stride))
                }
                _ => [0; 3],
            };
            if let Some(add) = &node.add {
                let r = get(&tensors, add)?;
                if r.channels != node.cout || r.dims != out {
                    return Err(format!("{}: residual shape mismatch.", node.out));
                }
            }
            (out, node.cout, [0; 3], pad)
        };
        let shape = Shape {
            dims: shape_dims,
            channels,
            padded: padded_channels(channels),
        };
        tensors.insert(node.out.clone(), shape);
        steps.push(Step {
            node: node.clone(),
            inputs,
            shape,
            pad_begin,
            crop,
            in_place: false,
            slot: 0,
            input_slots: vec![],
        });
    }
    let output = get(&tensors, &model.output)?;
    if output.dims != dims {
        return Err("The network output grid differs from its input.".into());
    }
    // Buffers by lifetime: the smallest free slot that holds a tensor, else the largest free slot
    // grows, else a new slot. A residual add whose operand dies there accumulates in place.
    let elements = |s: &Shape| s.voxels() * s.padded;
    let reads =
        |s: &Step| -> Vec<String> { s.node.x.iter().cloned().chain(s.node.add.clone()).collect() };
    let mut remaining: HashMap<String, usize> = HashMap::from([(model.output.clone(), 1)]);
    for s in &steps {
        for n in reads(s) {
            *remaining.entry(n).or_default() += 1;
        }
    }
    let mut slots = vec![elements(&input)];
    let mut slot_of: HashMap<String, usize> = HashMap::from([("input".to_string(), 0)]);
    let mut free: Vec<usize> = Vec::new();
    for s in &mut steps {
        let need = elements(&s.shape);
        let add = s.node.add.clone();
        s.in_place = add
            .as_ref()
            .is_some_and(|a| remaining[a] == 1 && *a != s.node.x[0] && a != "input");
        let slot = if s.in_place {
            slot_of[add.as_ref().unwrap()]
        } else {
            free.sort_by_key(|&i| slots[i]);
            let pick = free
                .iter()
                .position(|&i| slots[i] >= need)
                .or(if free.is_empty() {
                    None
                } else {
                    Some(free.len() - 1)
                });
            match pick {
                Some(p) => {
                    let i = free.remove(p);
                    slots[i] = slots[i].max(need);
                    i
                }
                None => {
                    slots.push(need);
                    slots.len() - 1
                }
            }
        };
        slot_of.insert(s.node.out.clone(), slot);
        s.slot = slot;
        s.input_slots = reads(s).iter().map(|n| slot_of[n]).collect();
        for n in reads(s) {
            let left = remaining[&n] - 1;
            remaining.insert(n.clone(), left);
            if left == 0 && !(s.in_place && Some(&n) == add.as_ref()) {
                free.push(slot_of[&n]);
            }
        }
    }
    Ok(Plan {
        output_slot: slot_of[&model.output],
        steps,
        slots,
        input,
        output,
    })
}

/// Weights as [tap][input channel (padded)][output channel (padded)], tap = (kd * k + kh) * k + kw.
pub fn pack_weights(step: &Step, weights: &[f32]) -> Vec<f32> {
    let n = &step.node;
    let taps = n.k.pow(3);
    let (pin, pout) = (step.inputs[0].padded, step.shape.padded);
    let w = &weights[n.w.0..n.w.0 + n.w.1];
    let mut out = vec![0f32; taps * pin * pout];
    for o in 0..n.cout {
        for i in 0..n.cin {
            let base = if n.op == Op::ConvT {
                (i * n.cout + o) * taps
            } else {
                (o * n.cin + i) * taps
            };
            for t in 0..taps {
                out[(t * pin + i) * pout + o] = w[base + t];
            }
        }
    }
    out
}

pub fn pack_bias(step: &Step, weights: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; step.shape.padded];
    out[..step.node.b.1].copy_from_slice(&weights[step.node.b.0..step.node.b.0 + step.node.b.1]);
    out
}

/// NCDHW (C channels, z fastest) to NDHWC with `padded` channels.
pub fn to_channels_last(data: &[f32], channels: usize, padded: usize) -> Vec<f32> {
    let voxels = data.len() / channels;
    let mut out = vec![0f32; voxels * padded];
    for c in 0..channels {
        for v in 0..voxels {
            out[v * padded + c] = data[c * voxels + v];
        }
    }
    out
}
