// fixtures/conformance.* (tools/make_fixture.py): every op variant against tools/reference.py.
use serde_json::Value;
use shivai_core::{cpu, plan::plan, shiva::parse};

fn fixture() -> (shivai_core::shiva::Model, [usize; 3], Vec<f32>, Vec<f32>) {
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
    let d = f["dims"].as_array().unwrap();
    (
        model,
        [0, 1, 2].map(|i| d[i].as_u64().unwrap() as usize),
        v("input"),
        v("output"),
    )
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn cpu_matches_reference() {
    let (model, dims, input, output) = fixture();
    let mut s = cpu::Session::new(plan(&model, dims).unwrap());
    s.load(&model);
    assert!(max_diff(&s.run(&input), &output) < 1e-5);
}
