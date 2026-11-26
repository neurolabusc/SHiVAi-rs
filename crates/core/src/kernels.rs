// GPU kernels for every plan step, generated as WGSL (WebGPU), MSL (Metal) or CUDA C++ from one
// description.
// Convolutions are an FP32 implicit matrix multiplication: a workgroup stages a reduction chunk of
// activations and weights in workgroup memory and every thread accumulates a register block of
// positions by output channels; layers with ≤ 8 padded output channels accumulate directly. Activations may be stored as f16 (WGSL with shader-f16); arithmetic
// is always f32. Bindings, in order: conv = input, weights, biases, result[, residual];
// concat = first, second, result.
use crate::plan::Step;
use crate::shiva::{Act, Op};
use std::fmt::Write;

const POSITIONS: usize = 2; // spatial positions per thread
const CHUNK: usize = 16; // reduction elements staged per barrier
const ROWS: usize = 64; // thread rows per workgroup
const INVOCATIONS: usize = 256; // WebGPU baseline maxComputeInvocationsPerWorkgroup
const STORAGE: usize = 16384; // WebGPU baseline maxComputeWorkgroupStorageSize
const MAX_GROUPS: usize = 65535; // WebGPU's maxComputeWorkgroupsPerDimension

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Lang {
    Wgsl {
        half: bool,
    },
    Msl,
    /// CUDA C++ for NVRTC; the entry point is `main0`, like MSL.
    Cuda,
}

/// Operators MSL has built in, for CUDA's 16-byte aligned `vec4`.
const CUDA_PRELUDE: &str = "typedef unsigned int uint;
struct __align__(16) vec4 {
  float x, y, z, w;
  __device__ vec4() {}
  __device__ explicit vec4(float s) : x(s), y(s), z(s), w(s) {}
  __device__ vec4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
};
__device__ __forceinline__ vec4 operator+(vec4 a, vec4 b) { return vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ vec4 operator/(vec4 a, vec4 b) { return vec4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
__device__ __forceinline__ vec4 operator-(vec4 a) { return vec4(-a.x, -a.y, -a.z, -a.w); }
__device__ __forceinline__ vec4 operator*(float s, vec4 a) { return vec4(s * a.x, s * a.y, s * a.z, s * a.w); }
__device__ __forceinline__ void operator+=(vec4& a, vec4 b) { a = a + b; }
__device__ __forceinline__ vec4 exp(vec4 a) { return vec4(expf(a.x), expf(a.y), expf(a.z), expf(a.w)); }
";

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Binding {
    Slot(usize),
    Weights,
    Biases,
}

#[derive(Clone, Debug)]
pub struct Kernel {
    pub code: String,
    pub groups: [u32; 3],
    pub threads: [u32; 3],
    pub bindings: Vec<Binding>,
}

fn channel_block(out: usize) -> (usize, usize) {
    for channels in [8, 4] {
        let mut tile = out.min(32);
        while tile >= channels {
            if out.is_multiple_of(tile)
                && tile.is_multiple_of(channels)
                && (tile / channels) * ROWS <= INVOCATIONS
                && 4 * CHUNK * (POSITIONS * ROWS + tile) <= STORAGE
            {
                return (channels, tile);
            }
            tile -= channels;
        }
    }
    unreachable!("padded channels are multiples of 4")
}

fn grid(groups: usize) -> (usize, usize) {
    let x = groups.min(MAX_GROUPS);
    (x, groups.div_ceil(x))
}

/// Spelling of the few constructs that differ between WGSL and the C++ dialects (MSL, CUDA).
struct Dialect {
    msl: bool,           // C++ syntax: MSL or CUDA
    cuda: bool,          // CUDA, where MSL's float4 is the prelude's vec4
    store: &'static str, // activation storage scalar
}

impl Dialect {
    fn vec4(&self) -> &'static str {
        if self.cuda {
            "vec4"
        } else if self.msl {
            "float4"
        } else {
            "vec4<f32>"
        }
    }
    fn store4(&self) -> String {
        if self.msl {
            self.vec4().into()
        } else {
            format!("vec4<{}>", self.store)
        }
    }
    fn uint(&self, v: usize) -> String {
        format!("{v}u")
    }
    fn local(&self, name: &str, ty: &str, value: &str) -> String {
        if self.msl {
            format!("{ty} {name} = {value};")
        } else {
            format!("var {name} = {value};")
        }
    }
    fn let_(&self, name: &str, ty: &str, value: &str) -> String {
        if self.msl {
            format!("{ty} {name} = {value};")
        } else {
            format!("let {name} = {value};")
        }
    }
    fn barrier(&self) -> &'static str {
        if self.cuda {
            "__syncthreads();"
        } else if self.msl {
            "threadgroup_barrier(mem_flags::mem_threadgroup);"
        } else {
            "workgroupBarrier();"
        }
    }
    fn to_f32(&self, e: &str) -> String {
        if self.msl {
            e.to_string()
        } else {
            format!("f32({e})")
        }
    }
    fn to_store4(&self, e: &str) -> String {
        if self.msl {
            e.to_string()
        } else {
            format!("vec4<{}>({e})", self.store)
        }
    }
}

const CUDA: Dialect = Dialect {
    msl: true,
    cuda: true,
    store: "f32",
};

fn activation(a: Act, d: &Dialect) -> String {
    let one = format!("{}(1.0)", d.vec4());
    match a {
        Act::None => String::new(),
        Act::Swish => format!("value = value / ({one} + exp(-value));"),
        Act::Sigmoid => format!("value = {one} / ({one} + exp(-value));"),
    }
}

pub fn kernel(step: &Step, lang: Lang) -> Kernel {
    if step.node.op == Op::Concat {
        concat(step, lang)
    } else if let Some(k) = (lang == Lang::Cuda)
        .then(|| direct(step).or_else(|| split(step)))
        .flatten()
    {
        k
    } else {
        conv(step, lang)
    }
}

/// CUDA direct convolution for stride-1 and transposed layers on large grids, where the implicit
/// matrix multiplication spends more on im2col indexing than on arithmetic. A block stages an input
/// tile with its halo (4 channels at a time, as float4) and those channels' weights in shared
/// memory; each thread accumulates P positions along w (8 apart, so shared reads do not conflict)
/// by CT output channels. Threads are 8 (w) × 8 (h) × 4 (d); grid z splits the output channels.
/// A transposed (stride-2) layer is 8 stride-1 convolutions, one per output parity class: a block
/// covers one class on the input lattice, so it visits only the taps that reach it (1 or 2 of the
/// 3 per axis) and its threads never diverge.
fn direct(step: &Step) -> Option<Kernel> {
    let n = &step.node;
    let [dd, h, w] = step.shape.dims;
    let [id, ih, iw] = step.inputs[0].dims;
    let (pin, pout, k) = (step.inputs[0].padded, step.shape.padded, n.k);
    let transposed = n.op == Op::ConvT;
    if !transposed && n.op != (Op::Conv { stride: 1 }) {
        return None;
    }
    // Output positions per class: half the output grid (rounded up) for a transposed layer.
    let [qd, qh, qw] = if transposed {
        [dd, h, w].map(|v| v.div_ceil(2))
    } else {
        [dd, h, w]
    };
    if qw < 32 || qh < 8 || qd < 4 {
        return None;
    }
    let ct = (4..=32)
        .rev()
        .step_by(4)
        .find(|c| pout.is_multiple_of(*c))?;
    let p = if ct <= 16 { 4 } else { 2 };
    let (tw, th, td) = (8 * p, 8, 4);
    // A transposed layer reads input q - m for m ≤ (k - 1) / 2; a convolution reads out + tap - pad.
    let reach = if transposed { (k - 1) / 2 } else { k - 1 };
    let (hw, hh, hd) = (tw + reach, th + reach, td + reach);
    let halo = hw * hh * hd;
    let taps = k * k * k;
    let w_len = taps * ct; // vec4 per (tap, 4 input channels, 4 output channels)
    if 16 * (halo + w_len) > 48 * 1024 {
        return None;
    }
    let (gw, gh, gd) = (qw.div_ceil(tw), qh.div_ceil(th), qd.div_ceil(td));
    let classes = if transposed { 8 } else { 1 };
    let [p0, p1, p2] = if transposed {
        [reach; 3]
    } else {
        step.pad_begin
    };
    let residual_binding = n.add.is_some() && !step.in_place;
    let ct4 = ct / 4;
    let residual = match (&n.add, step.in_place) {
        (None, _) => "",
        (Some(_), true) => " value += result[at];",
        (Some(_), false) => " value += residual[at];",
    };
    let act = format!(" {}", activation(n.act, &CUDA));
    let (taps_open, taps_close, row) = if transposed {
        (
            format!(
                "for (int kd = pd; kd < {k}; kd += 2)
    for (int kh = ph; kh < {k}; kh += 2)
    #pragma unroll 1
    for (int kw = pw; kw < {k}; kw += 2) {{
      const int t = (kd * {k} + kh) * {k} + kw;"
            ),
            "}",
            format!(
                "tile + ((lz + {reach} - (kd - pd) / 2) * {hh} + ly + {reach} - (kh - ph) / 2) * {hw} + lx + {reach} - (kw - pw) / 2"
            ),
        )
    } else {
        (
            format!(
                "#pragma unroll 1
    for (int t = 0; t < {taps}; t++) {{
      const int kd = t / {kk}, kh = t / {k} % {k}, kw = t % {k};",
                kk = k * k
            ),
            "}",
            format!("tile + ((lz + kd) * {hh} + ly + kh) * {hw} + lx + kw"),
        )
    };
    // Output coordinates of lattice point (q) in this class.
    let out = |q: &str, par: &str| {
        if transposed {
            format!("2 * ({q}) + {par}")
        } else {
            q.to_string()
        }
    };
    let code = format!(
        "{CUDA_PRELUDE}extern \"C\" __global__ void __launch_bounds__(256) main0(const vec4* __restrict__ input,
  const vec4* __restrict__ weights, const vec4* __restrict__ biases, vec4* result{res}) {{
  __shared__ vec4 tile[{halo}];
  __shared__ vec4 wt[{w_len}];
  const int lx = threadIdx.x, ly = threadIdx.y, lz = threadIdx.z;
  const int tid = (lz * 8 + ly) * 8 + lx;
  const int bw = blockIdx.x % {gw} * {tw}, bh = blockIdx.x / {gw} * {th}, bd = blockIdx.y * {td};
  const int c0 = blockIdx.z / {classes} * {ct};
  const int pc = blockIdx.z % {classes}, pd = pc >> 2, ph = (pc >> 1) & 1, pw = pc & 1;
  vec4 acc[{p}][{ct4}];
  #pragma unroll
  for (int i = 0; i < {p}; i++) {{
    #pragma unroll
    for (int j = 0; j < {ct4}; j++) {{ acc[i][j] = vec4(0.0f); }}
  }}
  for (int ci = 0; ci < {pin4}; ci++) {{
    for (int i = tid; i < {halo}; i += 256) {{
      const int x = bd + i / {hhw} - {p0}, y = bh + i / {hw} % {hh} - {p1}, z = bw + i % {hw} - {p2};
      tile[i] = (x >= 0 && y >= 0 && z >= 0 && x < {id} && y < {ih} && z < {iw})
        ? input[((x * {ih} + y) * {iw} + z) * {pin4} + ci] : vec4(0.0f);
    }}
    for (int i = tid; i < {w_len}; i += 256) {{
      const int j = i % {ct4}, c = i / {ct4} % 4, t = i / {ct4} / 4;
      wt[i] = weights[(t * {pin} + ci * 4 + c) * {pout4} + c0 / 4 + j];
    }}
    __syncthreads();
    {taps_open}
      const vec4* row = {row};
      vec4 x[{p}];
      #pragma unroll
      for (int i = 0; i < {p}; i++) {{ x[i] = row[i * 8]; }}
      const vec4* wr = wt + t * {w4};
      #pragma unroll
      for (int j = 0; j < {ct4}; j++) {{
        const vec4 w0 = wr[j], w1 = wr[{ct4} + j], w2 = wr[{ct4x2} + j], w3 = wr[{ct4x3} + j];
        #pragma unroll
        for (int i = 0; i < {p}; i++) {{
          acc[i][j] += x[i].x * w0; acc[i][j] += x[i].y * w1; acc[i][j] += x[i].z * w2; acc[i][j] += x[i].w * w3;
        }}
      }}
    {taps_close}
    __syncthreads();
  }}
  const int od = {od}, oh = {oh};
  if (od >= {dd} || oh >= {h}) {{ return; }}
  #pragma unroll
  for (int i = 0; i < {p}; i++) {{
    const int ow = {ow};
    if (ow < {w}) {{
      #pragma unroll
      for (int j = 0; j < {ct4}; j++) {{
        const unsigned int at = ((od * {h} + oh) * {w} + ow) * {pout4} + c0 / 4 + j;
        vec4 value = acc[i][j] + biases[c0 / 4 + j];{residual}{act}
        result[at] = value;
      }}
    }}
  }}
}}
",
        res = if residual_binding { ", const vec4* __restrict__ residual" } else { "" },
        pin4 = pin / 4,
        pout4 = pout / 4,
        hhw = hh * hw,
        w4 = 4 * ct4,
        ct4x2 = 2 * ct4,
        ct4x3 = 3 * ct4,
        od = out("bd + lz", "pd"),
        oh = out("bh + ly", "ph"),
        ow = out("bw + lx + i * 8", "pw"),
    );
    let mut bindings = vec![
        Binding::Slot(step.input_slots[0]),
        Binding::Weights,
        Binding::Biases,
        Binding::Slot(step.slot),
    ];
    if residual_binding {
        bindings.push(Binding::Slot(step.input_slots[1]));
    }
    Some(Kernel {
        code,
        groups: [(gw * gh) as u32, gd as u32, (classes * pout / ct) as u32],
        threads: [8, 8, 4],
        bindings,
    })
}

/// The input coordinates (x, y, z) of output position (od, oh, ow) at tap (td, th, tw); a
/// transposed convolution returns 0 from taps that miss the stride-2 lattice.
fn tap_source(step: &Step, d: &Dialect) -> String {
    let n = &step.node;
    let u = |v: usize| d.uint(v);
    let int = if d.msl { "int" } else { "i32" };
    match n.op {
        Op::Conv { stride } => {
            let p = step.pad_begin;
            ["x", "y", "z"]
                .iter()
                .zip(["od", "oh", "ow"].iter().zip(["td", "th", "tw"]))
                .enumerate()
                .map(|(a, (v, (o, t)))| {
                    d.let_(
                        v,
                        int,
                        &format!("{int}({o} * {} + {t}) - {}", u(stride), p[a]),
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
        _ => {
            let mut s = String::new();
            for (v, (o, t)) in ["x", "y", "z"]
                .iter()
                .zip(["od", "oh", "ow"].iter().zip(["td", "th", "tw"]))
            {
                let _ = write!(
                    s,
                    "{} if ({v}2 < 0 || ({v}2 & 1) != 0) {{ return 0.0; }} {} ",
                    d.let_(&format!("{v}2"), int, &format!("{int}({o}) - {int}({t})")),
                    d.let_(v, int, &format!("{v}2 / 2"))
                );
            }
            s
        }
    }
}

/// The im2col read shared by the C++ dialects (WGSL spells its own): reduction element k of output
/// position `position`.
fn cpp_gather(step: &Step, source: &str) -> String {
    let n = &step.node;
    let [id, ih, iw] = step.inputs[0].dims;
    let [dd, h, w] = step.shape.dims;
    let (pin, k, cin) = (step.inputs[0].padded, n.k, n.cin);
    let (spatial, reduction) = (dd * h * w, k * k * k * cin);
    format!(
        "  if (position >= {spatial}u || k >= {reduction}u) {{ return 0.0; }}
  uint tap = k / {cin}u;
  uint tw = tap % {k}u; uint th = tap / {k}u % {k}u; uint td = tap / {kk}u;
  uint ow = position % {w}u; uint oh = position / {w}u % {h}u; uint od = position / {wh}u;
  {source}
  if (x < 0 || y < 0 || z < 0 || x >= {id} || y >= {ih} || z >= {iw}) {{ return 0.0; }}
  return input[((uint(x) * {ih}u + uint(y)) * {iw}u + uint(z)) * {pin}u + k % {cin}u];
",
        kk = k * k,
        wh = w * h
    )
}

/// Positions per block of the CUDA split-reduction kernel.
const SPLIT_POSITIONS: usize = 4;

/// CUDA kernel for layers on small grids (deep in the U-Net), which the tiled kernels would run on
/// a few blocks: a block takes SPLIT_POSITIONS positions by CT output channels, its threads split
/// the reduction, and the partial sums are added in shared memory.
fn split(step: &Step) -> Option<Kernel> {
    let n = &step.node;
    let spatial = step.shape.voxels();
    if spatial > 16384 {
        return None;
    }
    let (pin, pout, cin) = (step.inputs[0].padded, step.shape.padded, n.cin);
    let reduction = n.k.pow(3) * cin;
    let ct = (4..=32)
        .rev()
        .step_by(4)
        .find(|c| pout.is_multiple_of(*c))?;
    let ct4 = ct / 4;
    let slices = 256 / ct4;
    let threads = slices * ct4;
    let pb = SPLIT_POSITIONS;
    let d = CUDA;
    let gather = cpp_gather(step, &tap_source(step, &d));
    let residual_binding = n.add.is_some() && !step.in_place;
    let residual = match (&n.add, step.in_place) {
        (None, _) => "",
        (Some(_), true) => " value += result[at];",
        (Some(_), false) => " value += residual[at];",
    };
    let (gx, gy) = grid(spatial.div_ceil(pb));
    let code = format!(
        "{CUDA_PRELUDE}static __device__ __forceinline__ float gather(const float* __restrict__ input, uint position, uint k) {{
{gather}}}
extern \"C\" __global__ void __launch_bounds__({threads}) main0(const float* __restrict__ input,
  const vec4* __restrict__ weights, const vec4* __restrict__ biases, vec4* result{res}) {{
  __shared__ vec4 part[{slices}][{pb}][{ct4}];
  const uint j = threadIdx.x % {ct4}u, r = threadIdx.x / {ct4}u;
  const uint base = (blockIdx.x + blockIdx.y * {gx}u) * {pb}u;
  const uint c4 = blockIdx.z * {ct4}u;
  vec4 acc[{pb}];
  #pragma unroll
  for (int i = 0; i < {pb}; i++) {{ acc[i] = vec4(0.0f); }}
  for (uint k = r; k < {reduction}u; k += {slices}u) {{
    const vec4 w = weights[(k / {cin}u * {pin}u + k % {cin}u) * {pout4}u + c4 + j];
    #pragma unroll
    for (int i = 0; i < {pb}; i++) {{ acc[i] += gather(input, base + i, k) * w; }}
  }}
  #pragma unroll
  for (int i = 0; i < {pb}; i++) {{ part[r][i][j] = acc[i]; }}
  __syncthreads();
  if (threadIdx.x >= {outs}u) {{ return; }}
  const uint i = threadIdx.x / {ct4}u, jj = threadIdx.x % {ct4}u;
  if (base + i >= {spatial}u) {{ return; }}
  vec4 value = biases[c4 + jj];
  for (uint q = 0; q < {slices}u; q++) {{ value += part[q][i][jj]; }}
  const uint at = (base + i) * {pout4}u + c4 + jj;{residual} {act}
  result[at] = value;
}}
",
        res = if residual_binding { ", const vec4* __restrict__ residual" } else { "" },
        pout4 = pout / 4,
        outs = pb * ct4,
        act = activation(n.act, &d),
    );
    let mut bindings = vec![
        Binding::Slot(step.input_slots[0]),
        Binding::Weights,
        Binding::Biases,
        Binding::Slot(step.slot),
    ];
    if residual_binding {
        bindings.push(Binding::Slot(step.input_slots[1]));
    }
    Some(Kernel {
        code,
        groups: [gx as u32, gy as u32, (pout / ct) as u32],
        threads: [threads as u32, 1, 1],
        bindings,
    })
}

fn conv(step: &Step, lang: Lang) -> Kernel {
    let d = Dialect {
        msl: lang != Lang::Wgsl { half: false } && lang != Lang::Wgsl { half: true },
        cuda: lang == Lang::Cuda,
        store: if lang == (Lang::Wgsl { half: true }) {
            "f16"
        } else {
            "f32"
        },
    };
    let n = &step.node;
    let [id, ih, iw] = step.inputs[0].dims;
    let [dd, h, w] = step.shape.dims;
    let (pin, pout, k) = (step.inputs[0].padded, step.shape.padded, n.k);
    let spatial = dd * h * w;
    let cin = n.cin;
    let reduction = k * k * k * cin;
    let (channels, tile) = channel_block(pout);
    let (columns, positions, lanes, out_vectors) =
        (tile / channels, POSITIONS * ROWS, channels / 4, pout / 4);
    let (gx, gy) = grid(spatial.div_ceil(positions));
    let source = tap_source(step, &d);
    let residual_binding = n.add.is_some() && !step.in_place;
    let st4 = d.store4();
    let mut s = String::new();
    let gather = cpp_gather(step, &source);
    // Declarations and the gather function.
    let residual_buffer = |space: &str| {
        if residual_binding {
            format!(", {space}const {st4}* residual")
        } else {
            String::new()
        }
    };
    if d.cuda {
        let _ = write!(
            s,
            "{CUDA_PRELUDE}static __device__ __forceinline__ float gather(const float* __restrict__ input, uint position, uint k) {{
{gather}}}
extern \"C\" __global__ void __launch_bounds__({threads}) main0(const float* __restrict__ input, const vec4* __restrict__ weights,
  const vec4* __restrict__ biases, vec4* result{res}) {{
  const uint3 group = blockIdx; const uint3 local_id = threadIdx;
  __shared__ float a[{a_len}];
  __shared__ vec4 b[{b_len}];
",
            threads = columns * ROWS,
            a_len = CHUNK * positions,
            b_len = CHUNK * tile / 4,
            res = residual_buffer("")
        );
    } else if d.msl {
        let _ = write!(s, "#include <metal_stdlib>\nusing namespace metal;\n");
        let _ = write!(
            s,
            "static inline float gather(device const float* input, uint position, uint k) {{
{gather}}}
kernel void main0(device const float* input [[buffer(0)]], device const float4* weights [[buffer(1)]],
  device const float4* biases [[buffer(2)]], device float4* result [[buffer(3)]]{res},
  uint3 group [[threadgroup_position_in_grid]], uint3 local_id [[thread_position_in_threadgroup]]) {{
  threadgroup float a[{a_len}];
  threadgroup float4 b[{b_len}];
",
            a_len = CHUNK * positions,
            b_len = CHUNK * tile / 4,
            res = if residual_binding { format!("{} [[buffer(4)]]", residual_buffer("device ")) } else { String::new() }
        );
    } else {
        let t = d.store;
        let _ = write!(
            s,
            "{enable}
@group(0) @binding(0) var<storage, read> input: array<{t}>;
@group(0) @binding(1) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> biases: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> result: array<{st4}>;
{res}
var<workgroup> a: array<f32, {a_len}>;
var<workgroup> b: array<vec4<f32>, {b_len}>;
fn gather(position: u32, k: u32) -> f32 {{
  if (position >= {spatial}u || k >= {reduction}u) {{ return 0.0; }}
  let tap = k / {cin}u;
  let tw = tap % {k}u; let th = tap / {k}u % {k}u; let td = tap / {kk}u;
  let ow = position % {w}u; let oh = position / {w}u % {h}u; let od = position / {wh}u;
  {source}
  if (x < 0 || y < 0 || z < 0 || x >= {id} || y >= {ih} || z >= {iw}) {{ return 0.0; }}
  return {ret};
}}
@compute @workgroup_size({columns}, {ROWS}, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {{
",
            enable = if t == "f16" { "enable f16;" } else { "" },
            res = if residual_binding { format!("@group(0) @binding(4) var<storage, read> residual: array<{st4}>;") } else { String::new() },
            kk = k * k,
            wh = w * h,
            a_len = CHUNK * positions,
            b_len = CHUNK * tile / 4,
            ret = d.to_f32(&format!("input[((u32(x) * {ih}u + u32(y)) * {iw}u + u32(z)) * {pin}u + k % {cin}u]"))
        );
    }
    let ui = if d.msl { "uint" } else { "u32" };
    let v4 = d.vec4();
    let _ = writeln!(
        s,
        "  {}",
        d.let_(
            "base",
            ui,
            &format!("(group.x + group.y * {gx}u) * {positions}u")
        )
    );
    let _ = writeln!(
        s,
        "  {}",
        d.let_(
            "channel",
            ui,
            &format!("group.z * {tile}u + local_id.x * {channels}u")
        )
    );
    let _ = writeln!(
        s,
        "  {}",
        d.let_("tid", ui, &format!("local_id.y * {columns}u + local_id.x"))
    );
    for p in 0..POSITIONS {
        for j in 0..lanes {
            let _ = writeln!(
                s,
                "  {}",
                d.local(&format!("sum{p}_{j}"), v4, &format!("{v4}(0.0)"))
            );
        }
    }
    let decl = |name: &str| {
        if d.msl {
            format!("uint {name}")
        } else {
            format!("var {name}")
        }
    };
    // Small channel counts cost more in staging/barriers than they save in shared loads.
    if pout <= 8 {
        let _ = writeln!(s, "  for ({} = 0u; k < {reduction}u; k++) {{", decl("k"));
        for j in 0..lanes {
            let _ = writeln!(s, "    {}", d.let_(&format!("weight{j}"), v4,
                &format!("weights[(k / {cin}u * {pin}u + k % {cin}u) * {out_vectors}u + channel / 4u + {j}u]")));
        }
        for p in 0..POSITIONS {
            let _ = writeln!(
                s,
                "    {}",
                d.let_(
                    &format!("activation{p}"),
                    if d.msl { "float" } else { "f32" },
                    &format!(
                        "gather({}base + local_id.y + {}u, k)",
                        if d.msl { "input, " } else { "" },
                        p * ROWS
                    )
                )
            );
            for j in 0..lanes {
                let _ = writeln!(s, "    sum{p}_{j} += activation{p} * weight{j};");
            }
        }
        s.push_str("  }\n");
    } else {
        let _ = write!(
        s,
        "  for ({start} = 0u; start < {reduction}u; start += {CHUNK}u) {{
    for ({i} = tid; i < {a_len}u; i += {stride}u) {{ a[i] = gather({inp}base + i % {positions}u, start + i / {positions}u); }}
    for ({i} = tid; i < {b_len}u; i += {stride}u) {{
      {kdecl}
      {vdecl}
      if (k < {reduction}u) {{ value = weights[(k / {cin}u * {pin}u + k % {cin}u) * {out_vectors}u + (group.z * {tile}u) / 4u + i % {tile4}u]; }}
      b[i] = value;
    }}
    {barrier}
    for ({kk} = 0u; k < {CHUNK}u; k++) {{
",
        start = decl("start"),
        i = decl("i"),
        kk = decl("k"),
        a_len = CHUNK * positions,
        b_len = CHUNK * tile / 4,
        stride = columns * ROWS,
        tile4 = tile / 4,
        inp = if d.msl { "input, " } else { "" },
        kdecl = d.let_("k", ui, &format!("start + i / {}u", tile / 4)),
        vdecl = d.local("value", v4, &format!("{v4}(0.0)")),
        barrier = d.barrier()
    );
        for j in 0..lanes {
            let _ = writeln!(
                s,
                "      {}",
                d.let_(
                    &format!("weight{j}"),
                    v4,
                    &format!("b[k * {}u + local_id.x * {lanes}u + {j}u]", tile / 4)
                )
            );
        }
        for p in 0..POSITIONS {
            let _ = writeln!(
                s,
                "      {}",
                d.let_(
                    &format!("activation{p}"),
                    if d.msl { "float" } else { "f32" },
                    &format!("a[k * {positions}u + local_id.y + {}u]", p * ROWS)
                )
            );
            for j in 0..lanes {
                let _ = writeln!(s, "      sum{p}_{j} += activation{p} * weight{j};");
            }
        }
        let _ = write!(s, "    }}\n    {}\n  }}\n", d.barrier());
    }
    for p in 0..POSITIONS {
        let _ = writeln!(
            s,
            "  {}",
            d.let_(
                &format!("position{p}"),
                ui,
                &format!("base + local_id.y + {}u", p * ROWS)
            )
        );
        let _ = writeln!(s, "  if (position{p} < {spatial}u) {{");
        for j in 0..lanes {
            let at = format!("position{p} * {out_vectors}u + channel / 4u + {j}u");
            let res = match (&n.add, step.in_place) {
                (None, _) => String::new(),
                (Some(_), true) => format!(
                    " value += {};",
                    if d.msl {
                        format!("result[{at}]")
                    } else {
                        format!("vec4<f32>(result[{at}])")
                    }
                ),
                (Some(_), false) => format!(
                    " value += {};",
                    if d.msl {
                        format!("residual[{at}]")
                    } else {
                        format!("vec4<f32>(residual[{at}])")
                    }
                ),
            };
            let _ = writeln!(
                s,
                "    {{ {}{res} {} result[{at}] = {}; }}",
                d.local(
                    "value",
                    v4,
                    &format!("sum{p}_{j} + biases[channel / 4u + {j}u]")
                ),
                activation(n.act, &d),
                d.to_store4("value")
            );
        }
        s.push_str("  }\n");
    }
    s.push_str("}\n");
    let mut bindings = vec![
        Binding::Slot(step.input_slots[0]),
        Binding::Weights,
        Binding::Biases,
        Binding::Slot(step.slot),
    ];
    if residual_binding {
        bindings.push(Binding::Slot(step.input_slots[1]));
    }
    Kernel {
        code: s,
        groups: [gx as u32, gy as u32, (pout / tile) as u32],
        threads: [columns as u32, ROWS as u32, 1],
        bindings,
    }
}

fn concat(step: &Step, lang: Lang) -> Kernel {
    let (a, b) = (step.inputs[0], step.inputs[1]);
    let [dd, h, w] = step.shape.dims;
    let pout = step.shape.padded;
    let size = dd * h * w * pout;
    let (gx, gy) = grid(size.div_ceil(256));
    let c = step.crop;
    let (ac, abc, ap, bp) = (a.channels, a.channels + b.channels, a.padded, b.padded);
    let [ad, ah, aw] = a.dims;
    let wh = w * h;
    let code = match lang {
        Lang::Cuda => format!(
            "extern \"C\" __global__ void __launch_bounds__(256) main0(const float* __restrict__ first,
  const float* __restrict__ second, float* __restrict__ result) {{
  unsigned int i = (blockIdx.x + blockIdx.y * {gx}u) * 256u + threadIdx.x;
  if (i >= {size}u) {{ return; }}
  unsigned int c = i % {pout}u; unsigned int p = i / {pout}u;
  float value = 0.0f;
  if (c < {ac}u) {{
    int x = int(p / {wh}u) + {c0}; int y = int(p / {w}u % {h}u) + {c1}; int z = int(p % {w}u) + {c2};
    if (x >= 0 && y >= 0 && z >= 0 && x < {ad} && y < {ah} && z < {aw}) {{
      value = first[((unsigned(x) * {ah}u + unsigned(y)) * {aw}u + unsigned(z)) * {ap}u + c];
    }}
  }} else if (c < {abc}u) {{
    value = second[p * {bp}u + c - {ac}u];
  }}
  result[i] = value;
}}
",
            c0 = c[0],
            c1 = c[1],
            c2 = c[2]
        ),
        Lang::Msl => format!(
            "#include <metal_stdlib>\nusing namespace metal;
kernel void main0(device const float* first [[buffer(0)]], device const float* second [[buffer(1)]],
  device float* result [[buffer(2)]], uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {{
  uint i = (group.x + group.y * {gx}u) * 256u + tid;
  if (i >= {size}u) {{ return; }}
  uint c = i % {pout}u; uint p = i / {pout}u;
  float value = 0.0f;
  if (c < {ac}u) {{
    int x = int(p / {wh}u) + {c0}; int y = int(p / {w}u % {h}u) + {c1}; int z = int(p % {w}u) + {c2};
    if (x >= 0 && y >= 0 && z >= 0 && x < {ad} && y < {ah} && z < {aw}) {{
      value = first[((uint(x) * {ah}u + uint(y)) * {aw}u + uint(z)) * {ap}u + c];
    }}
  }} else if (c < {abc}u) {{
    value = second[p * {bp}u + c - {ac}u];
  }}
  result[i] = value;
}}
",
            c0 = c[0],
            c1 = c[1],
            c2 = c[2]
        ),
        Lang::Wgsl { half } => {
            let t = if half { "f16" } else { "f32" };
            format!(
                "{enable}
@group(0) @binding(0) var<storage, read> first: array<{t}>;
@group(0) @binding(1) var<storage, read> second: array<{t}>;
@group(0) @binding(2) var<storage, read_write> result: array<{t}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) tid: u32) {{
  let i = (group.x + group.y * {gx}u) * 256u + tid;
  if (i >= {size}u) {{ return; }}
  let c = i % {pout}u;
  let p = i / {pout}u;
  var value = {t}(0.0);
  if (c < {ac}u) {{
    let x = i32(p / {wh}u) + {c0};
    let y = i32(p / {w}u % {h}u) + {c1};
    let z = i32(p % {w}u) + {c2};
    if (x >= 0 && y >= 0 && z >= 0 && x < {ad} && y < {ah} && z < {aw}) {{
      value = first[((u32(x) * {ah}u + u32(y)) * {aw}u + u32(z)) * {ap}u + c];
    }}
  }} else if (c < {abc}u) {{
    value = second[p * {bp}u + c - {ac}u];
  }}
  result[i] = value;
}}
",
                enable = if half { "enable f16;" } else { "" },
                c0 = c[0],
                c1 = c[1],
                c2 = c[2]
            )
        }
    };
    Kernel {
        code,
        groups: [gx as u32, gy as u32, 1],
        threads: [256, 1, 1],
        bindings: vec![
            Binding::Slot(step.input_slots[0]),
            Binding::Slot(step.input_slots[1]),
            Binding::Slot(step.slot),
        ],
    }
}

/// CUDA layout kernels around a plan, so only compact NCDHW volumes cross PCIe: `input` writes the
/// network input as `plan::to_channels_last` does (arguments: the NCDHW upload, slot 0) and `output`
/// gathers the first channel of the result (arguments: the output slot, the download buffer).
pub fn cuda_layout(plan: &crate::plan::Plan) -> (Kernel, Kernel) {
    let launch = |n: usize, code: String| {
        let (gx, gy) = grid(n.div_ceil(256));
        Kernel {
            code: code.replace("GX", &gx.to_string()),
            groups: [gx as u32, gy as u32, 1],
            threads: [256, 1, 1],
            bindings: vec![],
        }
    };
    let (i, o) = (plan.input, plan.output);
    let (voxels, padded) = (i.voxels(), i.padded);
    let input = launch(
        voxels * padded,
        format!(
            "extern \"C\" __global__ void __launch_bounds__(256) main0(const float* __restrict__ data,
  float* __restrict__ result) {{
  unsigned int i = (blockIdx.x + blockIdx.y * GXu) * 256u + threadIdx.x;
  if (i >= {n}u) {{ return; }}
  unsigned int c = i % {padded}u;
  result[i] = c < {channels}u ? data[c * {voxels}u + i / {padded}u] : 0.0f;
}}
",
            n = voxels * padded,
            channels = i.channels
        ),
    );
    let output = launch(
        o.voxels(),
        format!(
            "extern \"C\" __global__ void __launch_bounds__(256) main0(const float* __restrict__ data,
  float* __restrict__ result) {{
  unsigned int i = (blockIdx.x + blockIdx.y * GXu) * 256u + threadIdx.x;
  if (i < {n}u) {{ result[i] = data[i * {padded}u]; }}
}}
",
            n = o.voxels(),
            padded = o.padded
        ),
    );
    (input, output)
}
