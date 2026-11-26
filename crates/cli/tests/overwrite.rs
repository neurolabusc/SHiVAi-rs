// An existing image output is refused without --force, before the T1 or the models are read.
#[test]
fn existing_image_output_is_refused() {
    let out = std::env::temp_dir().join(format!("shivai-overwrite-{}", std::process::id()));
    std::fs::create_dir_all(&out).unwrap();
    let image = out.join("T1w_pvs_clusters.nii.gz");
    std::fs::write(&image, b"keep").unwrap();
    let run = std::process::Command::new(env!("CARGO_BIN_EXE_shivai-rs"))
        .args([
            "--t1",
            "missing/T1w.nii.gz",
            "--pvs",
            "--models",
            "missing",
            "--out",
        ])
        .arg(&out)
        .output()
        .unwrap();
    let message = String::from_utf8_lossy(&run.stderr);
    assert!(
        !run.status.success() && message.contains("T1w_pvs_clusters.nii.gz exists"),
        "{message}"
    );
    assert_eq!(std::fs::read(&image).unwrap(), b"keep");

    // An input with an output's name is refused even with --force.
    let flair = out.join("T1w_flair.nii.gz");
    std::fs::write(&flair, b"input").unwrap();
    let run = std::process::Command::new(env!("CARGO_BIN_EXE_shivai-rs"))
        .args([
            "--t1",
            "missing/T1w.nii.gz",
            "--wmh",
            "--force",
            "--models",
            "missing",
            "--flair",
        ])
        .arg(&flair)
        .arg("--out")
        .arg(&out)
        .output()
        .unwrap();
    let message = String::from_utf8_lossy(&run.stderr);
    assert!(
        !run.status.success() && message.contains("is an input"),
        "{message}"
    );
    assert_eq!(std::fs::read(&flair).unwrap(), b"input");
    std::fs::remove_dir_all(&out).unwrap();
}
