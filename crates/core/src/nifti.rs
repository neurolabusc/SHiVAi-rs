// ponytail: header fields are indexed as in the NIfTI layout.
#![allow(clippy::needless_range_loop)]
// NIfTI-1 scalar volumes with nibabel's geometry rule (the sform when its code is set, else the
// qform, else nibabel's centred base affine). Data is x fastest.
use crate::preprocess::{Affine, Volume};
use std::io::{Read, Write};

/// Voxels at most (128 Mi, as SynthSR's reader): bounds memory and decompression of hostile files.
const MAX_VOXELS: usize = 128 << 20;

pub fn read(bytes: &[u8]) -> Result<Volume, String> {
    let owned;
    let buf = if bytes.starts_with(&[0x1f, 0x8b]) {
        // Decompress the header first, then only as much as it says the image holds.
        let mut gz = flate2::read::MultiGzDecoder::new(bytes);
        let mut v = vec![0u8; 352];
        let bad = |e: std::io::Error| format!("Cannot decompress NIfTI: {e}");
        gz.read_exact(&mut v).map_err(bad)?;
        let need = Header::decode(&v[..348]).byte_len().unwrap_or(0);
        gz.take(need.saturating_sub(352) as u64)
            .read_to_end(&mut v)
            .map_err(bad)?;
        owned = v;
        &owned[..]
    } else {
        bytes
    };
    if buf.len() < 352 || &buf[344..347] != b"n+1" {
        return Err("Choose a NIfTI-1 image (.nii or .nii.gz).".into());
    }
    let h = Header::decode(&buf[..348]);
    let need = h
        .byte_len()
        .ok_or("Unsupported NIfTI header (dimensions, datatype or voxel offset).")?;
    let dim = h.dim;
    if (4..=(dim[0] as usize).min(7)).any(|a| dim[a] > 1) {
        return Err("Choose a 3D image, not a 4D series.".into());
    }
    if (1..4).any(|a| dim[a] < 1) {
        return Err("Unsupported image dimensions.".into());
    }
    let dims = [dim[1] as usize, dim[2] as usize, dim[3] as usize];
    let (width, decode) = h.scalar_decoder()?;
    let raw = buf
        .get(h.vox_offset as usize..need)
        .ok_or("The NIfTI voxel data is truncated.")?;
    let (slope, inter) = if h.scl_slope != 0.0 && h.scl_slope.is_finite() {
        (h.scl_slope as f64, h.scl_inter as f64)
    } else {
        (1.0, 0.0)
    };
    let data: Vec<f32> = raw
        .chunks_exact(width)
        .map(|c| (decode(c) * slope + inter) as f32)
        .collect();
    if !data.iter().any(|v| v.is_finite()) {
        return Err("The image has no finite intensities.".into());
    }
    let pixdim = [h.pixdim[1], h.pixdim[2], h.pixdim[3]];
    if pixdim.iter().any(|&p| !(p.is_finite() && p > 0.0)) {
        return Err("The image has invalid voxel sizes (pixdim).".into());
    }
    let mut affine: Affine = [[0.0; 4]; 4];
    affine[3][3] = 1.0;
    if h.sform_code > 0 {
        for r in 0..3 {
            for c in 0..4 {
                affine[r][c] = h.srow[r][c] as f64;
            }
        }
    } else if h.qform_code > 0 {
        let [b, c, d] = h.quaternion.map(|v| v as f64);
        let a = (1.0 - (b * b + c * c + d * d)).max(0.0).sqrt();
        let qfac = if h.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
        let rot = [
            [
                a * a + b * b - c * c - d * d,
                2.0 * (b * c - a * d),
                2.0 * (b * d + a * c),
            ],
            [
                2.0 * (b * c + a * d),
                a * a + c * c - b * b - d * d,
                2.0 * (c * d - a * b),
            ],
            [
                2.0 * (b * d - a * c),
                2.0 * (c * d + a * b),
                a * a + d * d - c * c - b * b,
            ],
        ];
        for r in 0..3 {
            for k in 0..3 {
                affine[r][k] = rot[r][k] * pixdim[k] as f64 * if k == 2 { qfac } else { 1.0 };
            }
            affine[r][3] = h.qoffset[r] as f64;
        }
    } else {
        // nibabel's base affine: centred on the volume, x flipped (radiological voxel order).
        for a in 0..3 {
            let flip = if a == 0 { -1.0 } else { 1.0 };
            affine[a][a] = flip * pixdim[a] as f64;
            affine[a][3] = -flip * (dims[a] as f64 - 1.0) / 2.0 * pixdim[a] as f64;
        }
    }
    let m = &affine;
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if affine.iter().flatten().any(|v| !v.is_finite()) || det == 0.0 {
        return Err("Invalid NIfTI affine.".into());
    }
    Ok(Volume {
        data,
        dims,
        affine,
        pixdim: Some(pixdim),
    })
}

type Decoder = Box<dyn Fn(&[u8]) -> f64>;

struct Header {
    dim: [i16; 8],
    pixdim: [f32; 8],
    vox_offset: f32,
    scl_slope: f32,
    scl_inter: f32,
    qform_code: i16,
    sform_code: i16,
    quaternion: [f32; 3],
    qoffset: [f32; 3],
    srow: [[f32; 4]; 3],
    datatype: i16,
    le: bool,
}

impl Header {
    fn decode(b: &[u8]) -> Self {
        let le = i32::from_le_bytes(b[0..4].try_into().unwrap()) == 348;
        let i16at = |o: usize| {
            let x = [b[o], b[o + 1]];
            if le {
                i16::from_le_bytes(x)
            } else {
                i16::from_be_bytes(x)
            }
        };
        let f32at = |o: usize| {
            let x: [u8; 4] = b[o..o + 4].try_into().unwrap();
            if le {
                f32::from_le_bytes(x)
            } else {
                f32::from_be_bytes(x)
            }
        };
        Self {
            dim: std::array::from_fn(|a| i16at(40 + 2 * a)),
            pixdim: std::array::from_fn(|a| f32at(76 + 4 * a)),
            vox_offset: f32at(108),
            scl_slope: f32at(112),
            scl_inter: f32at(116),
            qform_code: i16at(252),
            sform_code: i16at(254),
            quaternion: std::array::from_fn(|a| f32at(256 + 4 * a)),
            qoffset: std::array::from_fn(|a| f32at(268 + 4 * a)),
            srow: std::array::from_fn(|r| std::array::from_fn(|k| f32at(280 + (r * 4 + k) * 4))),
            datatype: i16at(70),
            le,
        }
    }

    /// End of the voxel data in bytes, when dimensions, datatype and offset are sane.
    fn byte_len(&self) -> Option<usize> {
        let width = match self.datatype {
            2 | 256 => 1,
            4 | 512 => 2,
            8 | 768 | 16 => 4,
            64 => 8,
            _ => return None,
        };
        let d = self.dim;
        if (1..4).any(|a| d[a] < 1) || !(self.vox_offset.is_finite() && self.vox_offset >= 352.0) {
            return None;
        }
        let n = (d[1] as usize)
            .checked_mul(d[2] as usize)?
            .checked_mul(d[3] as usize)?;
        if n > MAX_VOXELS {
            return None;
        }
        (self.vox_offset as usize).checked_add(n * width)
    }

    /// Bytes per voxel and a scalar decoder.
    fn scalar_decoder(&self) -> Result<(usize, Decoder), String> {
        let le = self.le;
        macro_rules! dec {
            ($t:ty) => {
                Box::new(move |c: &[u8]| {
                    let a = c.try_into().unwrap();
                    (if le {
                        <$t>::from_le_bytes(a)
                    } else {
                        <$t>::from_be_bytes(a)
                    }) as f64
                })
            };
        }
        Ok(match self.datatype {
            2 => (1, Box::new(|c: &[u8]| c[0] as f64)),
            256 => (1, Box::new(|c: &[u8]| c[0] as i8 as f64)),
            4 => (2, dec!(i16)),
            512 => (2, dec!(u16)),
            8 => (4, dec!(i32)),
            768 => (4, dec!(u32)),
            16 => (4, dec!(f32)),
            64 => (8, dec!(f64)),
            t => {
                return Err(format!(
                    "Unsupported NIfTI datatype {t}; use a scalar image."
                ))
            }
        })
    }
}

pub enum Data<'a> {
    F32(&'a [f32]),
    U8(&'a [u8]),
    I16(&'a [i16]),
}

/// Gzipped NIfTI-1 with the affine as sform (code 2), qform unset.
pub fn write(data: Data, dims: [usize; 3], affine: &Affine, description: &str) -> Vec<u8> {
    let (code, bits, body): (i16, i16, Vec<u8>) = match data {
        Data::F32(d) => (16, 32, d.iter().flat_map(|v| v.to_le_bytes()).collect()),
        Data::U8(d) => (2, 8, d.to_vec()),
        Data::I16(d) => (4, 16, d.iter().flat_map(|v| v.to_le_bytes()).collect()),
    };
    let mut b = vec![0u8; 352];
    let put16 = |b: &mut [u8], o: usize, x: i16| b[o..o + 2].copy_from_slice(&x.to_le_bytes());
    let put32 = |b: &mut [u8], o: usize, x: f32| b[o..o + 4].copy_from_slice(&x.to_le_bytes());
    b[0..4].copy_from_slice(&348i32.to_le_bytes());
    put16(&mut b, 40, 3);
    for a in 0..7 {
        put16(&mut b, 42 + a * 2, if a < 3 { dims[a] as i16 } else { 1 });
    }
    put16(&mut b, 70, code);
    put16(&mut b, 72, bits);
    put32(&mut b, 76, 1.0);
    for a in 0..3 {
        put32(
            &mut b,
            80 + a * 4,
            (0..3)
                .map(|r| affine[r][a] * affine[r][a])
                .sum::<f64>()
                .sqrt() as f32,
        );
    }
    put32(&mut b, 108, 352.0);
    put32(&mut b, 112, 1.0);
    b[123] = 10;
    let desc = description.as_bytes();
    b[148..148 + desc.len().min(79)].copy_from_slice(&desc[..desc.len().min(79)]);
    put16(&mut b, 254, 2);
    for r in 0..3 {
        for c in 0..4 {
            put32(&mut b, 280 + 16 * r + 4 * c, affine[r][c] as f32);
        }
    }
    b[344..348].copy_from_slice(b"n+1\0");
    b.extend(body);
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(6));
    gz.write_all(&b).unwrap();
    gz.finish().unwrap()
}

#[cfg(test)]
mod tests {
    // Without qform or sform, nibabel 5.5 gives shape (3, 4, 5), zooms (1, 2, 3) the diagonal
    // (-1, 2, 3) and translation (1, -3, -6).
    #[test]
    fn untransformed_header_uses_nibabels_base_affine() {
        let mut b = vec![0u8; 352 + 60];
        b[0..4].copy_from_slice(&348i32.to_le_bytes());
        for (i, d) in [3i16, 3, 4, 5, 1, 1, 1, 1].iter().enumerate() {
            b[40 + 2 * i..42 + 2 * i].copy_from_slice(&d.to_le_bytes());
        }
        b[70..72].copy_from_slice(&2i16.to_le_bytes()); // uint8
        for (i, z) in [1f32, 1.0, 2.0, 3.0].iter().enumerate() {
            b[76 + 4 * i..80 + 4 * i].copy_from_slice(&z.to_le_bytes());
        }
        b[108..112].copy_from_slice(&352f32.to_le_bytes());
        b[344..348].copy_from_slice(b"n+1\0");
        let a = super::read(&b).unwrap().affine;
        assert_eq!([a[0][0], a[1][1], a[2][2]], [-1.0, 2.0, 3.0]);
        assert_eq!([a[0][3], a[1][3], a[2][3]], [1.0, -3.0, -6.0]);
    }
}
