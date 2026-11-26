// GPU kernels for every plan step, generated as WGSL (WebGPU) or MSL (Metal) from one description.
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
    Wgsl { half: bool },
    Msl,
}

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

/// Spelling of the few constructs that differ between WGSL and MSL.
struct Dialect {
    msl: bool,
    store: &'static str, // activation storage scalar
}

impl Dialect {
    fn vec4(&self) -> &'static str {
        if self.msl {
            "float4"
        } else {
            "vec4<f32>"
        }
    }
    fn store4(&self) -> String {
        if self.msl {
            "float4".into()
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
        if self.msl {
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
    } else {
        conv(step, lang)
    }
}

fn conv(step: &Step, lang: Lang) -> Kernel {
    let d = Dialect {
        msl: lang == Lang::Msl,
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
    let u = |v: usize| d.uint(v);
    let int = if d.msl { "int" } else { "i32" };
    let source = match n.op {
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
    };
    let residual_binding = n.add.is_some() && !step.in_place;
    let st4 = d.store4();
    let mut s = String::new();
    // Declarations and the gather function.
    if d.msl {
        let _ = write!(s, "#include <metal_stdlib>\nusing namespace metal;\n");
        let _ = write!(
            s,
            "static inline float gather(device const float* input, uint position, uint k) {{
  if (position >= {spatial}u || k >= {reduction}u) {{ return 0.0; }}
  uint tap = k / {cin}u;
  uint tw = tap % {k}u; uint th = tap / {k}u % {k}u; uint td = tap / {kk}u;
  uint ow = position % {w}u; uint oh = position / {w}u % {h}u; uint od = position / {wh}u;
  {source}
  if (x < 0 || y < 0 || z < 0 || x >= {id} || y >= {ih} || z >= {iw}) {{ return 0.0; }}
  return input[((uint(x) * {ih}u + uint(y)) * {iw}u + uint(z)) * {pin}u + k % {cin}u];
}}
kernel void main0(device const float* input [[buffer(0)]], device const float4* weights [[buffer(1)]],
  device const float4* biases [[buffer(2)]], device float4* result [[buffer(3)]]{res},
  uint3 group [[threadgroup_position_in_grid]], uint3 local_id [[thread_position_in_threadgroup]]) {{
  threadgroup float a[{a_len}];
  threadgroup float4 b[{b_len}];
",
            kk = k * k,
            wh = w * h,
            a_len = CHUNK * positions,
            b_len = CHUNK * tile / 4,
            res = if residual_binding { ", device const float4* residual [[buffer(4)]]" } else { "" }
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
