// The Metal executor on the conformance graph (fixtures/, from tools/make_fixture.py).
#![cfg(target_os = "macos")]
#![allow(dead_code)] // the executor module is shared with the binary
#[path = "../src/metal.rs"]
mod metal;
use serde_json::Value;
use shivai_core::{plan::plan, shiva::parse};

#[test]
fn metal_matches_reference() {
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
    // Skip only without a device; a shader or pipeline failure must fail the test.
    if ::metal::Device::system_default().is_none() {
        eprintln!("no Metal device; skipped");
        return;
    }
    let mut s = metal::Session::new(plan(&model, dims).unwrap()).unwrap();
    s.load(&model);
    let out = s.run(&v("input")).unwrap();
    let max = out
        .iter()
        .zip(v("output"))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(max < 1e-5, "max difference {max}");
}
