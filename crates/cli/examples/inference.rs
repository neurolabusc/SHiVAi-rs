use shivai_core::{
    cpu, nifti,
    pipeline::{to_network, GRID},
    plan::plan,
    shiva::parse,
};
use std::{path::Path, time::Instant};
#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "../src/metal.rs"]
mod metal;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() >= 6,
        "inference cpu|metal MODEL OUTPUT REPEATS T1 [FLAIR]"
    );
    let model = parse(&std::fs::read(&args[2]).unwrap()).unwrap();
    let channels: Vec<_> = args[5..]
        .iter()
        .map(|f| nifti::read(&std::fs::read(f).unwrap()).unwrap())
        .collect();
    assert!(channels.iter().all(|c| c.dims == GRID));
    assert_eq!(channels.len(), model.input_channels);
    let input = to_network(
        &channels
            .iter()
            .map(|c| c.data.as_slice())
            .collect::<Vec<_>>(),
    );
    let setup = Instant::now();
    let p = plan(&model, GRID).unwrap();
    let mut run: Box<dyn FnMut() -> Vec<f32>> = match args[1].as_str() {
        "cpu" => {
            let mut s = cpu::Session::new(p);
            s.load(&model);
            Box::new(move || s.run(&input))
        }
        #[cfg(target_os = "macos")]
        "metal" => {
            let mut s = metal::Session::new(p).unwrap();
            s.load(&model);
            Box::new(move || s.run(&input).unwrap())
        }
        _ => panic!("unknown device"),
    };
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.;
    let mut times = Vec::new();
    let mut out = vec![];
    for _ in 0..args[4].parse().unwrap() {
        let start = Instant::now();
        out = run();
        times.push(start.elapsed().as_secs_f64() * 1000.);
        assert!(out.iter().all(|x| x.is_finite()));
    }
    std::fs::write(
        &args[3],
        out.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
    .unwrap();
    println!(
        "{}",
        serde_json::json!({"device": args[1], "model": Path::new(&args[2]).file_name().unwrap().to_str().unwrap(), "setup_ms": setup_ms, "run_ms": times, "sum": out.iter().map(|x| *x as f64).sum::<f64>()})
    );
}
