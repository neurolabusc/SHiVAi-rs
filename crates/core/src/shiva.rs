// .shiva model files (tools/lower.py): "SHIVA\0" v1, u32 JSON length, the graph JSON, zero padding
// to 16 bytes, little-endian float16 weights.
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    Conv { stride: usize },
    ConvT,
    Concat,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub out: String,
    pub x: Vec<String>, // one input, or the two concatenated
    pub k: usize,
    pub same: bool,
    pub cin: usize,
    pub cout: usize,
    pub act: Act,
    pub add: Option<String>,
    pub w: (usize, usize),
    pub b: (usize, usize),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    None,
    Swish,
    Sigmoid,
}

pub struct Model {
    pub input_channels: usize,
    pub output: String,
    pub nodes: Vec<Node>,
    pub weights: Vec<f32>,
    /// The graph JSON's ops, so folds of one ensemble can share a session.
    pub signature: String,
}

pub fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((h >> 10) & 0x1f) as i32;
    let mantissa = (h & 0x3ff) as f32;
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 if mantissa != 0.0 => f32::NAN,
        31 => sign * f32::INFINITY,
        e => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    }
}

/// float32 to IEEE half, round to nearest even (subnormals included).
pub fn to_half(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let e = exponent - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mantissa | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut half = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        if rest > halfway || (rest == halfway && half & 1 == 1) {
            half += 1;
        }
        return sign | half as u16;
    }
    let mut half = ((e as u32) << 10) | (mantissa >> 13);
    let rest = mantissa & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && half & 1 == 1) {
        half += 1; // a carry into the exponent is correct
    }
    sign | half as u16
}

pub fn parse(bytes: &[u8]) -> Result<Model, String> {
    if bytes.len() < 12 || &bytes[..8] != b"SHIVA\0\x01\0" {
        return Err("Not a SHiVA model file.".into());
    }
    let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let graph: Value = serde_json::from_slice(bytes.get(12..12 + len).ok_or("Truncated model.")?)
        .map_err(|e| format!("Bad model graph: {e}"))?;
    if graph["format"] != "shiva-unet/1" {
        return Err(format!(
            "Unsupported SHiVA graph format {}.",
            graph["format"]
        ));
    }
    let start = (12 + len).div_ceil(16) * 16;
    let weights = bytes[start..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&c| half_to_f32(u16::from_le_bytes(c)))
        .collect();
    let str_of = |v: &Value| {
        v.as_str()
            .map(String::from)
            .ok_or("Bad model graph.".to_string())
    };
    let usize_of = |v: &Value| {
        v.as_u64()
            .map(|n| n as usize)
            .ok_or("Bad model graph.".to_string())
    };
    let range =
        |v: &Value| -> Result<(usize, usize), String> { Ok((usize_of(&v[0])?, usize_of(&v[1])?)) };
    let mut nodes = Vec::new();
    for o in graph["ops"].as_array().ok_or("Bad model graph.")? {
        let op = match o["op"].as_str() {
            Some("conv") => Op::Conv {
                stride: usize_of(&o["stride"])?,
            },
            Some("convT") => Op::ConvT,
            Some("concat") => Op::Concat,
            other => return Err(format!("Unsupported op {other:?}.")),
        };
        if op == Op::Concat {
            let x = o["x"]
                .as_array()
                .ok_or("Bad concat.")?
                .iter()
                .map(str_of)
                .collect::<Result<_, _>>()?;
            nodes.push(Node {
                op,
                out: str_of(&o["out"])?,
                x,
                k: 0,
                same: true,
                cin: 0,
                cout: 0,
                act: Act::None,
                add: None,
                w: (0, 0),
                b: (0, 0),
            });
            continue;
        }
        let act = match o["act"].as_str() {
            Some("none") => Act::None,
            Some("swish") => Act::Swish,
            Some("sigmoid") => Act::Sigmoid,
            other => return Err(format!("Unsupported activation {other:?}.")),
        };
        nodes.push(Node {
            op,
            out: str_of(&o["out"])?,
            x: vec![str_of(&o["x"])?],
            k: usize_of(&o["k"])?,
            same: o["pad"] == "same",
            cin: usize_of(&o["cin"])?,
            cout: usize_of(&o["cout"])?,
            act,
            add: o.get("add").map(str_of).transpose()?,
            w: range(&o["w"])?,
            b: range(&o["b"])?,
        });
    }
    Ok(Model {
        input_channels: usize_of(&graph["input"]["channels"])?,
        output: str_of(&graph["output"])?,
        nodes,
        weights,
        signature: graph["ops"].to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::{half_to_f32, to_half};

    #[test]
    fn half_conversion_round_trips_and_rounds_to_nearest_even() {
        for h in (0..=u16::MAX).filter(|h| h & 0x7c00 != 0x7c00) {
            assert_eq!(to_half(half_to_f32(h)), h, "{h:#06x}");
        }
        // halfway between 1 and 1 + 2^-10 rounds to even; above it rounds up; 70000 overflows
        assert_eq!(to_half(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(to_half(1.0 + 2f32.powi(-11) + 2f32.powi(-20)), 0x3c01);
        assert_eq!(to_half(70000.0), 0x7c00);
    }
}
