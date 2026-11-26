// Preprocessing against scipy (fixture) and, with SHIVAI_REFERENCE_DIR and SHIVAI_REFERENCE_T1, against
// every stage of SHiVAi itself (tools/shivai_reference.py), each stage fed SHiVAi's input.
use serde_json::Value;
use shivai_core::nifti::read;
use shivai_core::preprocess::*;

#[test]
fn resampling_matches_scipy() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/affine_transform.json"
    );
    let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let floats = |k: &str| {
        f[k].as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect::<Vec<_>>()
    };
    let dims3 =
        |k: &str| -> [usize; 3] { std::array::from_fn(|i| f[k][i].as_u64().unwrap() as usize) };
    let m: Affine =
        std::array::from_fn(|r| std::array::from_fn(|c| f["matrix"][r][c].as_f64().unwrap()));
    for order in [0u8, 1, 3] {
        let out = affine_transform(&floats("input"), dims3("dims"), &m, dims3("outDims"), order);
        let want = floats(&format!("order{order}"));
        assert!(
            out.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-5),
            "order {order}"
        );
    }
}

fn mismatches(a: &[f32], b: &[f32], tol: f32) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| (*x - *y).abs() > tol)
        .count()
}

#[test]
fn stages_match_shivai() {
    let (Some(dir), Some(t1)) = (
        std::env::var_os("SHIVAI_REFERENCE_DIR"),
        std::env::var_os("SHIVAI_REFERENCE_T1"),
    ) else {
        eprintln!("set SHIVAI_REFERENCE_DIR and SHIVAI_REFERENCE_T1; skipped");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let r = |n: &str| read(&std::fs::read(dir.join(format!("{n}.nii.gz"))).unwrap()).unwrap();
    let grid = [160, 214, 176];
    let conformed_dims = [256, 256, 256];
    let t1 = read(&std::fs::read(t1).unwrap()).unwrap();
    // Nearest-neighbour conforming of an oblique T1 differs at rounding ties only (about 0.06 % of voxels).
    let c = conform(&t1, conformed_dims, Some([1.0; 3]), 3);
    assert!(mismatches(&c.data, &r("01_conformed").data, 1e-3) < c.data.len() / 500);
    let pre = conform(&t1, grid, None, 3);
    assert!(mismatches(&pre.data, &r("02_preconform").data, 1e-2) == 0);
    let r02 = r("02_preconform");
    assert_eq!(
        mismatches(
            &normalize(&r02.data, None),
            &r("03_preconform_norm").data,
            1e-4
        ),
        0
    );
    let r05 = r("05_premask_conformed");
    let mut warnings = vec![];
    let m06 = threshold_mask(
        &r05.data,
        conformed_dims,
        voxel_sizes(&r05.affine),
        0.5,
        3,
        600000.0,
        &mut warnings,
    );
    assert_eq!(
        mismatches(
            &m06.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            &r("06_premask_bin").data,
            0.5
        ),
        0
    );
    let r01 = r("01_conformed");
    let r10 = r("10_mask_conformed");
    let brain = threshold_mask(
        &r10.data,
        conformed_dims,
        voxel_sizes(&r10.affine),
        0.5,
        3,
        30000.0,
        &mut warnings,
    );
    let r11 = r("11_brainmask");
    assert_eq!(
        mismatches(
            &brain.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            &r11.data,
            0.5
        ),
        0
    );
    let mask11: Vec<u8> = r11.data.iter().map(|&v| (v > 0.5) as u8).collect();
    let defaced = quickshear(&r01.data, conformed_dims, &mask11);
    assert_eq!(mismatches(&defaced, &r("12_defaced").data, 1e-4), 0);
    let b = crop_box(&mask11, conformed_dims, grid).unwrap();
    let (cropped, affine) = crop(&r("12_defaced").data, conformed_dims, &r01.affine, &b);
    let r13 = r("13_t1_cropped");
    assert_eq!(mismatches(&cropped, &r13.data, 1e-4), 0);
    assert!((0..3).all(|i| (affine[i][3] - r13.affine[i][3]).abs() < 1e-3));
    let r14 = r("14_brainmask_cropped");
    let mask14: Vec<u8> = r14.data.iter().map(|&v| (v > 0.5) as u8).collect();
    assert_eq!(
        mismatches(
            &normalize(&r13.data, Some(&mask14)),
            &r("15_t1_final").data,
            1e-4
        ),
        0
    );
    let (labels, sizes) = label_clusters(&r("16_pvs").data, grid, &mask14, 0.5, 5);
    let r17 = r("17_pvs_labelled");
    assert_eq!(
        mismatches(
            &labels.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            &r17.data,
            0.5
        ),
        0
    );
    assert_eq!(
        sizes.len(),
        r17.data.iter().fold(0f32, |a, &b| a.max(b)) as usize
    );
}
