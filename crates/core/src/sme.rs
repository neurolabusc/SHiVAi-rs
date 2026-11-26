// FP32 SME outer products (sme.c); M4 and later. Callers fall back to cpu::tiles without it.
unsafe extern "C" {
    fn shivai_sme_available() -> i32;
    fn shivai_sme_matmul(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        rows: usize,
        cols: usize,
        depth: usize,
    );
}

pub fn available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| unsafe { shivai_sme_available() != 0 })
}

/// As `cpu::matmul`; `a` holds depth × 64 values, `b` depth × cols, `out` rows × cols (rows ≤ 64).
pub fn matmul(a: &[f32], b: &[f32], out: &mut [f32], rows: usize, cols: usize, depth: usize) {
    assert!(
        rows <= 64 && a.len() >= depth * 64 && b.len() >= depth * cols && out.len() >= rows * cols
    );
    // SAFETY: lengths checked above (once per 64-position block); SME is runtime-gated by `available`.
    unsafe { shivai_sme_matmul(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), rows, cols, depth) }
}
