// CPU backend: executes the shared plan (NDHWC, padded channels, lifetime-shared slots) with rayon.
// The oracle for the GPU backends and the fallback without one.
use crate::par::chunks_mut_enumerate;
use crate::plan::{pack_bias, pack_weights, to_channels_last, Plan, Step};
use crate::shiva::{Act, Model, Op};

pub struct Session {
    plan: Plan,
    slots: Vec<Vec<f32>>,
    weights: Vec<(Vec<f32>, Vec<f32>)>,
}

fn act(a: Act, v: f32) -> f32 {
    match a {
        Act::None => v,
        Act::Swish => v / (1.0 + (-v).exp()),
        Act::Sigmoid => 1.0 / (1.0 + (-v).exp()),
    }
}

/// `out[r][c] = Σ_k a[k][r] · b[k][c]` for one block: `a` is depth × ROWS, `b` depth × cols.
fn matmul(a: &[f32], b: &[f32], out: &mut [f32], rows: usize, cols: usize, depth: usize) {
    #[cfg(all(feature = "sme", target_os = "macos", target_arch = "aarch64"))]
    // SME's 16-wide tiles waste most lanes on narrow layers, where NEON tiles are faster.
    if cols >= 16 && crate::sme::available() {
        crate::sme::matmul(a, b, out, rows, cols, depth);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if avx2_fma() {
        // SAFETY: AVX2 and FMA were detected at run time.
        unsafe { tiles_avx2(a, b, out, rows, cols, depth) };
        return;
    }
    // Padded channel counts are multiples of 4, and of 8 above 4.
    if cols.is_multiple_of(8) {
        tiles::<8>(a, b, out, rows, cols, depth)
    } else {
        tiles::<4>(a, b, out, rows, cols, depth)
    }
}

/// Haswell (2013) and later: the x86-64 baseline is SSE2 without FMA, so the same tiles are also
/// compiled for AVX2 + FMA and chosen at run time; one binary serves every x86-64.
#[cfg(target_arch = "x86_64")]
fn avx2_fma() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn tiles_avx2(a: &[f32], b: &[f32], out: &mut [f32], rows: usize, cols: usize, depth: usize) {
    if cols.is_multiple_of(8) {
        tiles_impl::<8, true>(a, b, out, rows, cols, depth)
    } else {
        tiles_impl::<4, true>(a, b, out, rows, cols, depth)
    }
}

/// Register tiles of 8 rows × C columns, which the compiler keeps in vector registers (NEON, SSE/AVX).
fn tiles<const C: usize>(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    depth: usize,
) {
    tiles_impl::<C, false>(a, b, out, rows, cols, depth)
}

/// `FUSED`: the caller is compiled with FMA, so `mul_add` is one instruction.
#[inline(always)]
fn tiles_impl<const C: usize, const FUSED: bool>(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    depth: usize,
) {
    const R: usize = 8;
    for r0 in (0..rows).step_by(R) {
        for c0 in (0..cols).step_by(C) {
            let mut acc = [[0f32; C]; R];
            for k in 0..depth {
                let x: &[f32; R] = a[k * ROWS + r0..][..R].try_into().unwrap();
                let w: &[f32; C] = b[k * cols + c0..][..C].try_into().unwrap();
                for (acc, &x) in acc.iter_mut().zip(x) {
                    for (acc, &w) in acc.iter_mut().zip(w) {
                        *acc = madd::<FUSED>(x, w, *acc);
                    }
                }
            }
            for (r, acc) in acc.iter().enumerate().take(rows - r0) {
                out[(r0 + r) * cols + c0..][..C].copy_from_slice(acc);
            }
        }
    }
}

/// Fused on aarch64, where FMA is baseline, and in FMA-enabled callers; elsewhere `mul_add` may be
/// a slow library call.
#[inline(always)]
fn madd<const FUSED: bool>(x: f32, w: f32, acc: f32) -> f32 {
    if FUSED || cfg!(target_arch = "aarch64") {
        x.mul_add(w, acc)
    } else {
        acc + x * w
    }
}

/// Positions per block: each block is packed (im2col) once and multiplied by the weights.
const ROWS: usize = 64;

/// `residual`: a separate residual buffer; for an in-place step `out` already holds it.
fn conv(step: &Step, x: &[f32], w: &[f32], b: &[f32], residual: Option<&[f32]>, out: &mut [f32]) {
    if step.node.op == Op::ConvT {
        conv_impl::<true>(step, x, w, b, residual, out);
    } else {
        conv_impl::<false>(step, x, w, b, residual, out);
    }
}

fn conv_impl<const TRANSPOSED: bool>(
    step: &Step,
    x: &[f32],
    w: &[f32],
    b: &[f32],
    residual: Option<&[f32]>,
    out: &mut [f32],
) {
    let n = &step.node;
    let [id, ih, iw] = step.inputs[0].dims;
    let [_, h, wd] = step.shape.dims;
    let (pin, pout, k) = (step.inputs[0].padded, step.shape.padded, n.k);
    let stride = match n.op {
        Op::Conv { stride } => stride,
        _ => 2,
    };
    let depth = k * k * k * n.cin;
    let run = |scratch: &mut (Vec<f32>, Vec<f32>), (block, out): (usize, &mut [f32])| {
        let (packed, sums) = scratch;
        if TRANSPOSED {
            packed.fill(0.0);
        }
        let rows = out.len() / pout;
        let origins: [[i64; 3]; ROWS] = std::array::from_fn(|row| {
            let pos = block * ROWS + row;
            let dst = [pos / (h * wd), pos / wd % h, pos % wd];
            std::array::from_fn(|a| {
                if TRANSPOSED {
                    dst[a] as i64
                } else {
                    (dst[a] * stride) as i64 - step.pad_begin[a] as i64
                }
            })
        });
        for (t, tap_out) in packed.chunks_exact_mut(n.cin * ROWS).enumerate() {
            let tap = [t / (k * k), t / k % k, t % k];
            for (row, dst) in origins[..rows].iter().enumerate() {
                let src: [i64; 3] = if TRANSPOSED {
                    if (0..3).any(|a| dst[a] < tap[a] as i64 || (dst[a] - tap[a] as i64) & 1 != 0) {
                        continue;
                    }
                    std::array::from_fn(|a| (dst[a] - tap[a] as i64) / 2)
                } else {
                    std::array::from_fn(|a| dst[a] + tap[a] as i64)
                };
                if src[0] < 0
                    || src[1] < 0
                    || src[2] < 0
                    || src[0] >= id as i64
                    || src[1] >= ih as i64
                    || src[2] >= iw as i64
                {
                    if !TRANSPOSED {
                        for channel in tap_out.as_chunks_mut::<ROWS>().0.iter_mut() {
                            channel[row] = 0.0;
                        }
                    }
                    continue;
                }
                let base = ((src[0] as usize * ih + src[1] as usize) * iw + src[2] as usize) * pin;
                for (channel, &value) in tap_out
                    .as_chunks_mut::<ROWS>()
                    .0
                    .iter_mut()
                    .zip(&x[base..base + n.cin])
                {
                    channel[row] = value;
                }
            }
        }
        matmul(packed, w, sums, rows, pout, depth);
        if step.in_place {
            for (acc, sum) in out.chunks_mut(pout).zip(sums.chunks(pout)) {
                for ((a, &v), &bias) in acc.iter_mut().zip(sum).zip(b) {
                    *a += v + bias;
                }
            }
        } else {
            for (acc, sum) in out.chunks_mut(pout).zip(sums.chunks(pout)) {
                for ((a, &v), &bias) in acc.iter_mut().zip(sum).zip(b) {
                    *a = v + bias;
                }
            }
        }
        if let Some(r) = residual {
            for (a, &v) in out.iter_mut().zip(&r[block * ROWS * pout..]) {
                *a += v;
            }
        }
        match n.act {
            Act::None => {}
            Act::Swish => out.iter_mut().for_each(|a| *a = act(Act::Swish, *a)),
            Act::Sigmoid => out.iter_mut().for_each(|a| *a = act(Act::Sigmoid, *a)),
        }
    };
    let scratch = || (vec![0.0; depth * ROWS], vec![0.0; ROWS * pout]);
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(ROWS * pout)
            .enumerate()
            .with_min_len(16)
            .for_each_init(scratch, run);
    }
    #[cfg(not(feature = "parallel"))]
    {
        let mut scratch = scratch();
        for block in out.chunks_mut(ROWS * pout).enumerate() {
            run(&mut scratch, block);
        }
    }
}

fn concat(step: &Step, a: &[f32], b: &[f32], out: &mut [f32]) {
    let (sa, sb) = (step.inputs[0], step.inputs[1]);
    let [_, h, w] = step.shape.dims;
    let pout = step.shape.padded;
    let c = step.crop;
    chunks_mut_enumerate(out, pout, |(pos, o)| {
        o.fill(0.0);
        let (x, y, z) = (
            (pos / (h * w)) as i64 + c[0],
            (pos / w % h) as i64 + c[1],
            (pos % w) as i64 + c[2],
        );
        if x >= 0
            && y >= 0
            && z >= 0
            && x < sa.dims[0] as i64
            && y < sa.dims[1] as i64
            && z < sa.dims[2] as i64
        {
            let src =
                ((x as usize * sa.dims[1] + y as usize) * sa.dims[2] + z as usize) * sa.padded;
            o[..sa.channels].copy_from_slice(&a[src..src + sa.channels]);
        }
        o[sa.channels..sa.channels + sb.channels]
            .copy_from_slice(&b[pos * sb.padded..pos * sb.padded + sb.channels]);
    });
}

impl Session {
    pub fn new(plan: Plan) -> Self {
        let slots = plan.slots.iter().map(|&n| vec![0f32; n]).collect();
        Self {
            plan,
            slots,
            weights: vec![],
        }
    }

    pub fn load(&mut self, model: &Model) {
        self.weights = self
            .plan
            .steps
            .iter()
            .map(|s| {
                if s.node.op == Op::Concat {
                    (vec![], vec![])
                } else {
                    (
                        pack_weights(s, &model.weights)
                            .chunks(s.inputs[0].padded * s.shape.padded)
                            .flat_map(|tap| tap[..s.node.cin * s.shape.padded].iter().copied())
                            .collect(),
                        pack_bias(s, &model.weights),
                    )
                }
            })
            .collect();
    }

    /// NCDHW input (z fastest) -> the first output channel, same order.
    pub fn run(&mut self, input: &[f32]) -> Vec<f32> {
        let p = &self.plan;
        let packed = to_channels_last(input, p.input.channels, p.input.padded);
        self.slots[0][..packed.len()].copy_from_slice(&packed);
        for (i, step) in p.steps.iter().enumerate() {
            let len = step.shape.voxels() * step.shape.padded;
            let mut out = std::mem::take(&mut self.slots[step.slot]);
            {
                let slice = |s: usize, n: usize| &self.slots[s][..n];
                let size = |sh: &crate::plan::Shape| sh.voxels() * sh.padded;
                if step.node.op == Op::Concat {
                    concat(
                        step,
                        slice(step.input_slots[0], size(&step.inputs[0])),
                        slice(step.input_slots[1], size(&step.inputs[1])),
                        &mut out[..len],
                    );
                } else {
                    let x = slice(step.input_slots[0], size(&step.inputs[0]));
                    let (w, b) = &self.weights[i];
                    let residual = (step.node.add.is_some() && !step.in_place)
                        .then(|| slice(step.input_slots[1], len));
                    conv(step, x, w, b, residual, &mut out[..len]);
                }
            }
            self.slots[step.slot] = out;
        }
        let out = &self.slots[p.output_slot];
        (0..p.output.voxels())
            .map(|v| out[v * p.output.padded])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    type Matmul = fn(&[f32], &[f32], &mut [f32], usize, usize, usize);

    // A partial block (63 rows) and column tiles of 8 and 4 (24 and 12 columns): integer data, so
    // every kernel must be exact, and nothing past rows × cols is written.
    #[test]
    fn matmul_partial_tiles() {
        let (rows, depth) = (63, 17);
        for cols in [24usize, 12] {
            let a: Vec<_> = (0..depth * 64).map(|i| (i % 7) as f32 - 3.0).collect();
            let b: Vec<_> = (0..depth * cols).map(|i| (i % 5) as f32 - 2.0).collect();
            // matmul is SME (24 columns) or AVX2 + FMA where available, so also check the portable kernels.
            let mut kernels: Vec<Matmul> = vec![super::matmul];
            if cols == 24 {
                kernels.push(super::tiles::<8>);
            } else {
                kernels.push(super::tiles::<4>);
            }
            for kernel in kernels {
                let mut out = vec![1234.0; rows * cols + 1];
                kernel(&a, &b, &mut out, rows, cols, depth);
                for r in 0..rows {
                    for c in 0..cols {
                        let expected: f32 =
                            (0..depth).map(|k| a[k * 64 + r] * b[k * cols + c]).sum();
                        assert_eq!(out[r * cols + c], expected);
                    }
                }
                assert_eq!(out[rows * cols], 1234.0);
            }
        }
    }
}
