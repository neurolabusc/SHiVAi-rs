// SHiVAi's preprocessing and cluster post-processing (shivai/utils/preprocessing.py,
// postprocessing/clusters.py), with nibabel's conform, scipy.ndimage.affine_transform and quickshear
// ported to match. Volumes are x fastest.
#![allow(clippy::needless_range_loop)]

pub type Affine = [[f64; 4]; 4];

#[derive(Clone)]
pub struct Volume {
    pub data: Vec<f32>,
    pub dims: [usize; 3],
    pub affine: Affine,
    pub pixdim: Option<[f32; 3]>,
}

pub fn invert4(m: &Affine) -> Affine {
    let [[a, b, c, x], [d, e, f, y], [g, h, i, z], _] = *m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let r = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ];
    let mut out = [[0.0; 4]; 4];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = r[row][col] / det;
        }
        out[row][3] = -(out[row][0] * x + out[row][1] * y + out[row][2] * z);
    }
    out[3][3] = 1.0;
    out
}

pub fn multiply4(a: &Affine, b: &Affine) -> Affine {
    let mut out = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            out[r][c] = (0..4).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    out
}

/// float32, as NIfTI's pixdim holds them.
pub fn voxel_sizes(a: &Affine) -> [f32; 3] {
    std::array::from_fn(|c| {
        (a[0][c] * a[0][c] + a[1][c] * a[1][c] + a[2][c] * a[2][c]).sqrt() as f32
    })
}

// ---- scipy.ndimage.affine_transform(mode='constant'), orders 0, 1, 3 -------------------------

const POLE: f64 = -0.267_949_192_431_122_7; // sqrt(3) - 2

fn spline_filter(data: &[f32], dims: [usize; 3]) -> Vec<f64> {
    let mut out: Vec<f64> = data.iter().map(|&v| v as f64).collect();
    let strides = [1, dims[0], dims[0] * dims[1]];
    let mut line = vec![0f64; *dims.iter().max().unwrap()];
    for axis in 0..3 {
        let n = dims[axis];
        if n < 2 {
            continue;
        }
        let s = strides[axis];
        let others: Vec<usize> = (0..3).filter(|&a| a != axis).collect();
        let zn = POLE.powi(n as i32 - 1);
        for j in 0..dims[others[1]] {
            for i in 0..dims[others[0]] {
                let base = i * strides[others[0]] + j * strides[others[1]];
                for k in 0..n {
                    line[k] = out[base + k * s] * 6.0;
                }
                let mut c0 = line[0] + zn * line[n - 1];
                let mut z = POLE;
                for k in 1..n - 1 {
                    c0 += z * (line[k] + zn * line[n - 1 - k]);
                    z *= POLE;
                }
                line[0] = c0 / (1.0 - zn * zn);
                for k in 1..n {
                    line[k] += POLE * line[k - 1];
                }
                line[n - 1] = (POLE * line[n - 2] + line[n - 1]) * POLE / (POLE * POLE - 1.0);
                for k in (0..n - 1).rev() {
                    line[k] = POLE * (line[k + 1] - line[k]);
                }
                for k in 0..n {
                    out[base + k * s] = line[k];
                }
            }
        }
    }
    out
}

fn mirror(i: i64, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * n as i64 - 2;
    let i = i.abs() % period;
    (if i >= n as i64 { period - i } else { i }) as usize
}

fn cubic(x: f64) -> [f64; 4] {
    let t = x - x.floor();
    [
        (1.0 - t).powi(3) / 6.0,
        (4.0 - 6.0 * t * t + 3.0 * t.powi(3)) / 6.0,
        (1.0 + 3.0 * t + 3.0 * t * t - 3.0 * t.powi(3)) / 6.0,
        t.powi(3) / 6.0,
    ]
}

/// `matrix` maps output voxel to input voxel coordinates.
pub fn affine_transform(
    data: &[f32],
    dims: [usize; 3],
    matrix: &Affine,
    out_dims: [usize; 3],
    order: u8,
) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let coef = if order == 3 {
        spline_filter(data, dims)
    } else {
        vec![]
    };
    let mut out = vec![0f32; out_dims.iter().product()];
    let plane = out_dims[0] * out_dims[1];
    crate::par::chunks_mut_enumerate(&mut out, plane, |(k, slab)| {
        for j in 0..out_dims[1] {
            for i in 0..out_dims[0] {
                let m = matrix;
                let (fi, fj, fk) = (i as f64, j as f64, k as f64);
                let x = m[0][0] * fi + m[0][1] * fj + m[0][2] * fk + m[0][3];
                let y = m[1][0] * fi + m[1][1] * fj + m[1][2] * fk + m[1][3];
                let z = m[2][0] * fi + m[2][1] * fj + m[2][2] * fk + m[2][3];
                if x < 0.0
                    || y < 0.0
                    || z < 0.0
                    || x > (nx - 1) as f64
                    || y > (ny - 1) as f64
                    || z > (nz - 1) as f64
                {
                    continue;
                }
                let o = &mut slab[i + out_dims[0] * j];
                match order {
                    0 => {
                        *o = data[x.round() as usize
                            + nx * (y.round() as usize + ny * z.round() as usize)]
                    }
                    1 => {
                        let (fx, fy, fz) = (x.floor(), y.floor(), z.floor());
                        let (tx, ty, tz) = (x - fx, y - fy, z - fz);
                        let mut v = 0.0;
                        for c in 0..8 {
                            let (a, b, d) = (c & 1, (c >> 1) & 1, c >> 2);
                            let w = (if a == 1 { tx } else { 1.0 - tx })
                                * (if b == 1 { ty } else { 1.0 - ty })
                                * (if d == 1 { tz } else { 1.0 - tz });
                            if w != 0.0 {
                                let xi = (fx as usize + a).min(nx - 1);
                                let yi = (fy as usize + b).min(ny - 1);
                                let zi = (fz as usize + d).min(nz - 1);
                                v += w * data[xi + nx * (yi + ny * zi)] as f64;
                            }
                        }
                        *o = v as f32;
                    }
                    _ => {
                        let (wx, wy, wz) = (cubic(x), cubic(y), cubic(z));
                        let (bx, by, bz) = (
                            x.floor() as i64 - 1,
                            y.floor() as i64 - 1,
                            z.floor() as i64 - 1,
                        );
                        let ix: [usize; 4] = std::array::from_fn(|t| mirror(bx + t as i64, nx));
                        let iy: [usize; 4] =
                            std::array::from_fn(|t| mirror(by + t as i64, ny) * nx);
                        let iz: [usize; 4] =
                            std::array::from_fn(|t| mirror(bz + t as i64, nz) * nx * ny);
                        let mut v = 0.0;
                        for c in 0..4 {
                            for b in 0..4 {
                                let row = iz[c] + iy[b];
                                v += wz[c]
                                    * wy[b]
                                    * (wx[0] * coef[row + ix[0]]
                                        + wx[1] * coef[row + ix[1]]
                                        + wx[2] * coef[row + ix[2]]
                                        + wx[3] * coef[row + ix[3]]);
                            }
                        }
                        *o = v as f32;
                    }
                }
            }
        }
    });
    out
}

// ---- nibabel conform ---------------------------------------------------------------------------

/// nibabel io_orientation for a near-orthogonal affine: per voxel axis, its RAS axis and sign.
pub fn orientation(a: &Affine) -> [(usize, f64); 3] {
    let z = voxel_sizes(a);
    let mut r: [[f64; 3]; 3] = std::array::from_fn(|row| {
        std::array::from_fn(|c| a[row][c] / if z[c] == 0.0 { 1.0 } else { z[c] as f64 })
    });
    std::array::from_fn(|c| {
        let mut best = 0;
        for row in 1..3 {
            if r[row][c].abs() > r[best][c].abs() {
                best = row;
            }
        }
        let sign = r[best][c].signum();
        r[best] = [0.0; 3];
        (best, sign)
    })
}

/// Affine and shape after nibabel's as_reoriented(to RAS).
pub fn ras_grid(dims: [usize; 3], affine: &Affine) -> ([usize; 3], Affine) {
    let mut shape = [0; 3];
    let mut t = [[0.0; 4]; 4];
    t[3][3] = 1.0;
    for (input, (axis, sign)) in orientation(affine).into_iter().enumerate() {
        shape[axis] = dims[input];
        t[input][axis] = sign;
        if sign < 0.0 {
            t[input][3] = (dims[input] - 1) as f64;
        }
    }
    (shape, multiply4(affine, &t))
}

pub fn rescale_affine(
    a: &Affine,
    shape: [usize; 3],
    zooms: [f64; 3],
    new_shape: [usize; 3],
) -> Affine {
    let s = voxel_sizes(a);
    let centre = shape.map(|n| ((n - 1) / 2) as f64);
    let middle = new_shape.map(|n| ((n - 1) / 2) as f64);
    let mut out = [[0.0; 4]; 4];
    out[3][3] = 1.0;
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][c] * zooms[c] / s[c] as f64;
        }
        let centroid = a[r][0] * centre[0] + a[r][1] * centre[1] + a[r][2] * centre[2] + a[r][3];
        out[r][3] =
            centroid - (out[r][0] * middle[0] + out[r][1] * middle[1] + out[r][2] * middle[2]);
    }
    out
}

pub fn resample_to(
    data: &[f32],
    dims: [usize; 3],
    affine: &Affine,
    grid_dims: [usize; 3],
    grid_affine: &Affine,
    order: u8,
) -> Vec<f32> {
    affine_transform(
        data,
        dims,
        &multiply4(&invert4(affine), grid_affine),
        grid_dims,
        order,
    )
}

/// shivai Conform: `voxel_size` None keeps the field of view; a size equal to the input's is
/// nearest neighbour. Clipped to the input's range.
pub fn conform(
    v: &Volume,
    out_dims: [usize; 3],
    voxel_size: Option<[f64; 3]>,
    mut order: u8,
) -> Volume {
    let pixdim = v.pixdim.unwrap_or_else(|| voxel_sizes(&v.affine));
    let zooms = voxel_size.unwrap_or_else(|| {
        std::array::from_fn(|a| v.dims[a] as f64 * pixdim[a] as f64 / out_dims[a] as f64)
    });
    if (0..3).all(|a| zooms[a] == pixdim[a] as f64) {
        order = 0;
    }
    let (shape, ras) = ras_grid(v.dims, &v.affine);
    let affine = rescale_affine(&ras, shape, zooms, out_dims);
    let mut data = resample_to(&v.data, v.dims, &v.affine, out_dims, &affine, order);
    let (min, max) = v
        .data
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    for x in data.iter_mut() {
        *x = x.clamp(min, max);
    }
    Volume {
        data,
        dims: out_dims,
        affine,
        pixdim: None,
    }
}

// ---- intensities, masks, crop ------------------------------------------------------------------

/// numpy.percentile, linear.
pub fn percentile(values: &mut [f32], q: f64) -> f64 {
    let at = (values.len() - 1) as f64 * q / 100.0;
    let lo = at.floor() as usize;
    let (_, &mut low, above) = values.select_nth_unstable_by(lo, |a, b| a.total_cmp(b));
    let high = above
        .iter()
        .copied()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(low);
    low as f64 + (high as f64 - low as f64) * (at - lo as f64)
}

/// shivai normalization: / p99 (inside the mask when given), clip at 1.3, / 1.3.
pub fn normalize(data: &[f32], mask: Option<&[u8]>) -> Vec<f32> {
    let clean: Vec<f32> = data
        .iter()
        .map(|&v| if v.is_finite() && v > 0.0 { v } else { 0.0 })
        .collect();
    if !clean.iter().any(|&v| v > 0.0) {
        return clean;
    }
    let mut inside: Vec<f32> = match mask {
        Some(m) => clean
            .iter()
            .zip(m)
            .filter(|(_, &k)| k != 0)
            .map(|(&v, _)| v)
            .collect(),
        None => clean.clone(),
    };
    // An empty mask, or a 99th percentile of 0 (SHiVAi would divide by zero): leave unscaled.
    if inside.is_empty() {
        return clean;
    }
    let p = percentile(&mut inside, 99.0);
    if p <= 0.0 {
        return clean;
    }
    clean
        .iter()
        .map(|&v| ((v as f64 / p).min(1.3) / 1.3) as f32)
        .collect()
}

fn morph(mask: Vec<u8>, dims: [usize; 3], radius: [usize; 3], erode: bool) -> Vec<u8> {
    let mut current = mask;
    let strides = [1, dims[0], dims[0] * dims[1]];
    for axis in 0..3 {
        let (s, n) = (strides[axis], dims[axis]);
        for _ in 0..radius[axis] {
            let next: Vec<u8> = (0..current.len())
                .map(|i| {
                    let c = i / s % n;
                    let a = if c > 0 { current[i - s] } else { erode as u8 };
                    let b = if c < n - 1 {
                        current[i + s]
                    } else {
                        erode as u8
                    };
                    if erode {
                        current[i] & a & b
                    } else {
                        current[i] | a | b
                    }
                })
                .collect();
            current = next;
        }
    }
    current
}

pub struct Components {
    pub labels: Vec<i32>,
    pub sizes: Vec<usize>, // index 0 unused
    pub tops: Vec<usize>,
}

/// 26-connected components, labels 1.. in scan order (x fastest).
pub fn components(mask: &[u8], dims: [usize; 3]) -> Components {
    let [nx, ny, nz] = dims;
    let mut labels = vec![0i32; mask.len()];
    let mut queue = Vec::new(); // bounded by the largest component
    let (mut sizes, mut tops) = (vec![0], vec![0]);
    for seed in 0..mask.len() {
        if mask[seed] == 0 || labels[seed] != 0 {
            continue;
        }
        let id = sizes.len() as i32;
        let (mut head, mut top) = (0, 0);
        labels[seed] = id;
        queue.clear();
        queue.push(seed);
        while head < queue.len() {
            let i = queue[head];
            head += 1;
            let (x, y, z) = (i % nx, i / nx % ny, i / (nx * ny));
            top = top.max(z);
            for dz in -1i64..=1 {
                let zz = z as i64 + dz;
                if zz < 0 || zz >= nz as i64 {
                    continue;
                }
                for dy in -1i64..=1 {
                    let yy = y as i64 + dy;
                    if yy < 0 || yy >= ny as i64 {
                        continue;
                    }
                    for dx in -1i64..=1 {
                        let xx = x as i64 + dx;
                        if xx < 0 || xx >= nx as i64 {
                            continue;
                        }
                        let j = xx as usize + nx * (yy as usize + ny * zz as usize);
                        if mask[j] != 0 && labels[j] == 0 {
                            labels[j] = id;
                            queue.push(j);
                        }
                    }
                }
            }
        }
        sizes.push(queue.len());
        tops.push(top);
    }
    Components {
        labels,
        sizes,
        tops,
    }
}

/// shivai threshold(binarize=True): > thr, opening by `open` mm, then when several clusters
/// remain those above `min_vol` mm³, the biggest kept and dilated by 5 mm.
pub fn threshold_mask(
    data: &[f32],
    dims: [usize; 3],
    zooms: [f32; 3],
    thr: f32,
    open: usize,
    min_vol: f64,
    warnings: &mut Vec<String>,
) -> Vec<u8> {
    let mut mask: Vec<u8> = data.iter().map(|&v| (v > thr) as u8).collect();
    if open > 0 {
        let smallest = zooms.iter().copied().fold(f32::INFINITY, f32::min);
        let cap = dims.iter().max().unwrap() / 4;
        let radius = zooms.map(|z| {
            ((open as f32 * smallest / z).trunc() as usize)
                .min(cap)
                .max(1)
        });
        mask = morph(morph(mask, dims, radius, true), dims, radius, false);
    }
    let c = components(&mask, dims);
    if c.sizes.len() > 2 {
        let minimum = min_vol / (zooms[0] as f64 * zooms[1] as f64 * zooms[2] as f64);
        let mut order: Vec<usize> = (1..c.sizes.len()).collect();
        order.sort_by(|&a, &b| c.sizes[b].cmp(&c.sizes[a]));
        let kept: Vec<usize> = order
            .into_iter()
            .filter(|&id| c.sizes[id] as f64 > minimum)
            .collect();
        if let Some(&biggest) = kept.first() {
            let top = kept.iter().copied().fold(biggest, |best, id| {
                if c.tops[id] > c.tops[best] {
                    id
                } else {
                    best
                }
            });
            if top != biggest {
                warnings.push(
                    "The biggest brain-mask cluster is not the highest one; check the brain mask."
                        .into(),
                );
            }
            let cluster: Vec<u8> = c
                .labels
                .iter()
                .map(|&l| (l == biggest as i32) as u8)
                .collect();
            let cluster = morph(
                cluster,
                dims,
                zooms.map(|z| ((5.0 / z).trunc() as usize).max(1)),
                false,
            );
            for (m, k) in mask.iter_mut().zip(cluster) {
                *m &= k;
            }
        } else {
            warnings.push(format!(
                "No brain-mask cluster is larger than {min_vol} mm³."
            ));
            mask.fill(0);
        }
    }
    mask
}

pub struct CropBox {
    pub lo: [i64; 3],
    pub lo_clamped: [usize; 3],
    pub offset: [usize; 3],
    pub span: [usize; 3],
    pub size: [usize; 3],
}

/// shivai crop: a `size` box centred on the mask's centre of mass (ceil), moved up when the
/// mask reaches above it. The grid must already be RAS (conformed).
pub fn crop_box(mask: &[u8], dims: [usize; 3], size: [usize; 3]) -> Result<CropBox, String> {
    let [nx, ny, _] = dims;
    let (mut sum, mut n, mut top) = ([0f64; 3], 0usize, 0usize);
    for (i, &m) in mask.iter().enumerate() {
        if m == 0 {
            continue;
        }
        let (x, y, z) = (i % nx, i / nx % ny, i / (nx * ny));
        sum[0] += x as f64;
        sum[1] += y as f64;
        sum[2] += z as f64;
        top = top.max(z);
        n += 1;
    }
    if n == 0 {
        return Err("The brain mask is empty.".into());
    }
    let centre = sum.map(|s| (s / n as f64).ceil() as i64);
    let half = size.map(|s| (s / 2) as i64);
    let mut lo: [i64; 3] = std::array::from_fn(|a| centre[a] - half[a]);
    let hi: [i64; 3] = std::array::from_fn(|a| centre[a] + half[a]);
    let offset = lo.map(|v| if v < 0 { (-v) as usize } else { 0 });
    let mut lo_clamped = lo.map(|v| v.max(0) as usize);
    let hi_clamped: [usize; 3] = std::array::from_fn(|a| (hi[a].max(0) as usize).min(dims[a]));
    let span: [usize; 3] = std::array::from_fn(|a| hi_clamped[a].saturating_sub(lo_clamped[a]));
    if hi_clamped[2] <= top {
        let delta = top - hi_clamped[2] + 5;
        lo_clamped[2] += delta;
        lo[2] += delta as i64;
    }
    Ok(CropBox {
        lo,
        lo_clamped,
        offset,
        span,
        size,
    })
}

pub fn crop<T: Copy + Default>(
    data: &[T],
    dims: [usize; 3],
    affine: &Affine,
    b: &CropBox,
) -> (Vec<T>, Affine) {
    let [nx, ny, _] = dims;
    let [sx, sy, sz] = b.size;
    let mut out = vec![T::default(); sx * sy * sz];
    let span: [usize; 3] =
        std::array::from_fn(|a| b.span[a].min(dims[a].saturating_sub(b.lo_clamped[a])));
    let [lx, ly, lz] = b.lo_clamped;
    let [ox, oy, oz] = b.offset;
    for z in 0..span[2] {
        for y in 0..span[1] {
            let src = lx + nx * (ly + y + ny * (lz + z));
            let dst = ox + sx * (oy + y + sy * (oz + z));
            out[dst..dst + span[0]].copy_from_slice(&data[src..src + span[0]]);
        }
    }
    let mut a = *affine;
    for r in 0..3 {
        a[r][3] = affine[r][3]
            + affine[r][0] * b.lo[0] as f64
            + affine[r][1] * b.lo[1] as f64
            + affine[r][2] * b.lo[2] as f64;
    }
    (out, a)
}

/// quickshear (buff 10 voxels) with a brain mask on the same RAS grid.
pub fn quickshear(data: &[f32], dims: [usize; 3], mask: &[u8]) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let mut brain = vec![0u8; ny * nz]; // [p][s], RAS -> RPS flips y
    for z in 0..nz {
        for y in 0..ny {
            if (0..nx).any(|x| mask[x + nx * (y + ny * z)] != 0) {
                brain[(ny - 1 - y) * nz + z] = 1;
            }
        }
    }
    let at = |p: i64, s: i64| {
        brain
            [(((p + ny as i64) % ny as i64) as usize) * nz + ((s + nz as i64) % nz as i64) as usize]
            as i64
    };
    let cross = |o: (i64, i64), a: (i64, i64), b: (i64, i64)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut lower: Vec<(i64, i64)> = Vec::new();
    for p in 0..ny as i64 {
        for s in 0..nz as i64 {
            if 4 * at(p, s) - at(p - 1, s) - at(p + 1, s) - at(p, s - 1) - at(p, s + 1) == 0 {
                continue;
            }
            while lower.len() >= 2
                && cross(lower[lower.len() - 2], lower[lower.len() - 1], (p, s)) <= 0
            {
                lower.pop();
            }
            lower.push((p, s));
        }
    }
    if lower.len() < 2 {
        return data.to_vec();
    }
    let slope = (lower[1].1 - lower[0].1) as f64 / (lower[1].0 - lower[0].0) as f64;
    let intercept = lower[0].1 as f64 - lower[0].0 as f64 * slope - 10.0;
    let ys: Vec<f64> = (0..nz).map(|i| i as f64 * slope + intercept).collect();
    let mut keep = vec![1u8; ny * nz];
    // quickshear zips the indices where ys > 0 with ys from the start, as here
    for (k, p) in ys
        .iter()
        .enumerate()
        .filter(|(_, &y)| y > 0.0)
        .map(|(i, _)| i)
        .enumerate()
    {
        if p >= ny {
            continue;
        }
        let y = ys[k].trunc() as i64;
        let end = if y < 0 {
            (nz as i64 + y).max(0) as usize
        } else {
            (y as usize).min(nz)
        };
        for s in 0..end {
            keep[p * nz + s] = 0;
        }
    }
    let mut out = data.to_vec();
    for z in 0..nz {
        for y in 0..ny {
            if keep[(ny - 1 - y) * nz + z] == 0 {
                out[nx * (y + ny * z)..nx * (y + ny * z) + nx].fill(0.0);
            }
        }
    }
    out
}

/// postprocessing/clusters.py label_clusters with a binary brain mask: > threshold,
/// 26-connected, touching the brain, at least `min_size` voxels, at most 25 % outside.
/// Labels largest first; ties keep numpy's scan order (x slowest).
pub fn label_clusters(
    p: &[f32],
    dims: [usize; 3],
    brain: &[u8],
    threshold: f32,
    min_size: usize,
) -> (Vec<i16>, Vec<usize>) {
    let mask: Vec<u8> = p.iter().map(|&v| (v > threshold) as u8).collect();
    let c = components(&mask, dims);
    let [nx, ny, nz] = dims;
    let mut inside = vec![0usize; c.sizes.len()];
    let mut first = vec![usize::MAX; c.sizes.len()];
    for (i, &l) in c.labels.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let l = l as usize;
        if brain[i] != 0 {
            inside[l] += 1;
        }
        let order = (i % nx) * ny * nz + (i / nx % ny) * nz + i / (nx * ny);
        first[l] = first[l].min(order);
    }
    let mut kept: Vec<usize> = (1..c.sizes.len())
        .filter(|&id| {
            inside[id] > 0
                && c.sizes[id] >= min_size
                && (c.sizes[id] - inside[id]) as f64 <= 0.25 * c.sizes[id] as f64
        })
        .collect();
    kept.sort_by(|&a, &b| c.sizes[b].cmp(&c.sizes[a]).then(first[a].cmp(&first[b])));
    let mut renumber = vec![0i16; c.sizes.len()];
    for (k, &id) in kept.iter().enumerate() {
        renumber[id] = (k + 1).min(i16::MAX as usize) as i16;
    }
    (
        c.labels.iter().map(|&l| renumber[l as usize]).collect(),
        kept.iter().map(|&id| c.sizes[id]).collect(),
    )
}
