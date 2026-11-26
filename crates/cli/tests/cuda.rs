// The CUDA executor on the conformance graph (fixtures/, from tools/make_fixture.py).
#![cfg(target_os = "linux")]
#![allow(dead_code)] // the executor module is shared with the binary
#[path = "../src/cuda.rs"]
mod cuda;
use serde_json::Value;
use shivai_core::{plan::plan, shiva::parse};

/// The largest difference, NaN if any is (`f32::max` would drop it and pass the test).
fn worst(max: f32, d: f32) -> f32 {
    if max.is_nan() || d.is_nan() {
        f32::NAN
    } else {
        max.max(d)
    }
}

#[test]
fn cuda_matches_reference() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/");
    let model = parse(&std::fs::read(format!("{dir}conformance.shiva")).unwrap()).unwrap();
    let f: Value =
        serde_json::from_slice(&std::fs::read(format!("{dir}conformance.json")).unwrap()).unwrap();
    let v = |k: &str| {
        f[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect::<Vec<_>>()
    };
    let dims = [0, 1, 2].map(|i| f["dims"][i].as_u64().unwrap() as usize);
    // Skip only without an NVIDIA device; a missing NVRTC or a kernel failure must fail the test.
    if !cuda::device_present() {
        eprintln!("no CUDA device; skipped");
        return;
    }
    let mut s = cuda::Session::new(plan(&model, dims).unwrap()).unwrap();
    eprintln!("{}", s.description);
    s.load(&model).unwrap();
    let out = s.run(&v("input")).unwrap();
    let max = out
        .iter()
        .zip(v("output"))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, worst);
    assert!(max < 1e-5, "max difference {max}");
}

// The conformance graph on larger grids against the CPU executor (which the fixture checks), so
// every op variant also runs through the direct (w ≥ 32) and generic (w < 32) CUDA kernels; the
// fixture's own grid takes the split-reduction kernel.
#[test]
fn cuda_matches_cpu_on_larger_grids() {
    use shivai_core::kernels::{kernel, Lang};
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/");
    let model = parse(&std::fs::read(format!("{dir}conformance.shiva")).unwrap()).unwrap();
    if !cuda::device_present() {
        eprintln!("no CUDA device; skipped");
        return;
    }
    let mut kinds = std::collections::BTreeSet::new();
    for dims in [[12, 18, 70], [40, 40, 24]] {
        let p = plan(&model, dims).unwrap();
        for s in p
            .steps
            .iter()
            .filter(|s| s.node.op != shivai_core::shiva::Op::Concat)
        {
            let code = kernel(s, Lang::Cuda).code;
            kinds.insert(if code.contains("vec4 tile[") {
                "direct"
            } else if code.contains("vec4 part[") {
                "split"
            } else {
                "generic"
            });
        }
        let n = dims.iter().product::<usize>() * model.input_channels;
        // Deterministic, roughly unit-scale input.
        let input: Vec<f32> = (0..n)
            .map(|i| ((i as u64 * 2654435761) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let mut c = shivai_core::cpu::Session::new(plan(&model, dims).unwrap());
        c.load(&model);
        let expected = c.run(&input);
        let mut s = cuda::Session::new(p).unwrap();
        s.load(&model).unwrap();
        let out = s.run(&input).unwrap();
        let max = out
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, worst);
        assert!(max < 1e-5, "{dims:?}: max difference {max}");
    }
    assert_eq!(kinds.len(), 3, "kernels exercised: {kinds:?}");
}
