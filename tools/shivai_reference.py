"""Stage-by-stage reference of SHiVAi's T1 (+FLAIR) path with its own nipype interfaces and TF models.

    python shivai_reference.py <T1.nii.gz> <models dir> <out dir> [<FLAIR>]

Models dir holds brainmask/ (the .keras + models.py), T1-PVS/, T1.FLAIR-WMH/ and optionally T1.FLAIR-LAC/ as downloaded by shiva_download.
Each stage is written to <out dir>/<NN>_<name>.nii.gz; the JavaScript pipeline is tested against them.
Mirrors workflows/preprocessing.py, preprocessing_shiva_masking.py and shiva_mask_wf.py with the
default configuration (99th percentile, threshold 0.5, 1 mm, 160x214x176, RAS).
"""
import os, shutil, subprocess, sys
from pathlib import Path

import nibabel as nib, numpy as np

from shivai.interfaces.image import Conform, Crop, Normalization, Resample_from_to, Threshold
from shivai.postprocessing.clusters import label_clusters

DIMS = (160, 214, 176)


def run(interface, **inputs):
    return interface(**inputs).run().outputs


def predict(models, image_files, out):
    """predict_multi.py: per-fold <0.001 -> 0, then the mean over folds."""
    import tensorflow as tf
    x = np.stack([nib.load(f).get_fdata(dtype=np.float32) for f in image_files], -1)[None]
    folds = []
    for m in models:
        if str(m).endswith('.keras'):
            import importlib.util, keras
            spec = importlib.util.spec_from_file_location('kmodel', Path(m).parent / 'models.py')
            sys.modules['kmodel'] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sys.modules['kmodel'])
            p = keras.saving.load_model(m, compile=False).predict(x, batch_size=1, verbose=0)
        else:
            f = tf.saved_model.load(str(m)).signatures['serving_default']
            name, spec = next(iter(f.structured_input_signature[1].items()))
            p = list(f(**{name: tf.constant(x, spec.dtype)}).values())[0].numpy()
        p = p[0].squeeze()
        p[p < 0.001] = 0
        folds.append(p)
    nib.save(nib.Nifti1Image(np.mean(folds, 0).astype(np.float32), nib.load(image_files[0]).affine), out)
    return out


def main():
    t1, models, out = (Path(p).resolve() for p in sys.argv[1:4])
    flair = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else None
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    brainmask = sorted((models / 'brainmask').glob('*.keras'))
    stage = lambda n, name, src: shutil.copy(src, out / f'{n:02d}_{name}.nii.gz') and str(out / f'{n:02d}_{name}.nii.gz')

    conformed = stage(1, 'conformed', run(Conform, img=str(t1), dimensions=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0),
                                          voxels_tolerance=(0.0, 0.0, 0.0), orientation='RAS').resampled)
    # brain_mask_creation: whole field of view squeezed into the network grid
    pre = stage(2, 'preconform', run(Conform, img=str(t1), dimensions=DIMS, orientation='RAS').resampled)
    pre_norm = stage(3, 'preconform_norm', run(Normalization, input_image=pre, percentile=99.0).intensity_normalized)
    premask = predict(brainmask, [pre_norm], out / '04_premask.nii.gz')
    premask_conf = stage(5, 'premask_conformed', run(Conform, img=str(premask), dimensions=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0),
                                                     voxels_tolerance=(0.0, 0.0, 0.0), orientation='RAS', order=0, ignore_bad_affine=True).resampled)
    premask_bin = stage(6, 'premask_bin', run(Threshold, img=premask_conf, threshold=0.5, binarize=True, open=3, minVol=600000, clusterCheck='size').thresholded)
    norm_pre = stage(7, 'norm_with_premask', run(Normalization, input_image=conformed, brain_mask=premask_bin, percentile=99.0).intensity_normalized)
    crop_pre = stage(8, 'crop_with_premask', run(Crop, apply_to=norm_pre, roi_mask=premask_bin, final_dimensions=DIMS).cropped)
    mask_raw = predict(brainmask, [crop_pre], out / '09_mask.nii.gz')
    # preprocessing.py with the shiva mask
    mask_conf = stage(10, 'mask_conformed', run(Resample_from_to, moving_image=str(mask_raw), fixed_image=conformed, spline_order=0).resampled_image)
    mask = stage(11, 'brainmask', run(Threshold, img=mask_conf, threshold=0.5, binarize=True, open=3, minVol=30000, clusterCheck='size').thresholded)
    subprocess.run(['quickshear', conformed, mask, str(out / '12_defaced.nii.gz')], check=True)
    crop = run(Crop, apply_to=str(out / '12_defaced.nii.gz'), roi_mask=mask, final_dimensions=DIMS)
    cropped = stage(13, 't1_cropped', crop.cropped)
    mask_crop = stage(14, 'brainmask_cropped', run(Resample_from_to, moving_image=mask, fixed_image=cropped, spline_order=0).resampled_image)
    t1_final = stage(15, 't1_final', run(Normalization, input_image=cropped, brain_mask=mask_crop, percentile=99.0).intensity_normalized)
    pvs = predict(sorted((models / 'T1-PVS').glob('*.tf_inference')), [t1_final], out / '16_pvs.nii.gz')
    seg = label_clusters(nib.load(pvs).get_fdata(), 0.5, 5 - 1, nib.load(mask_crop).get_fdata())
    nib.save(nib.Nifti1Image(seg.astype(np.int16), nib.load(pvs).affine), out / '17_pvs_labelled.nii.gz')
    print('PVS clusters', int(seg.max()), 'voxels', int((seg > 0).sum()))
    if flair:
        # dual_preprocessing.py: ANTs rigid onto the T1 crop with the default ANTS settings (utils/parsing.py),
        # then defaced and normalised with the crop's mask
        from nipype.interfaces import ants
        reg = ants.Registration(moving_image=str(flair), fixed_image=cropped, fixed_image_masks=[mask_crop], float=True,
                                output_transform_prefix='flair_to_t1_', transforms=['Rigid'], transform_parameters=[(0.1,)],
                                metric=['MI'], metric_weight=[1], radius_or_number_of_bins=[32], interpolation='WelchWindowedSinc',
                                shrink_factors=[[4, 2, 1, 1]], smoothing_sigmas=[[4, 2, 1, 0]], sigma_units=['mm'],
                                number_of_iterations=[[1000, 500, 250, 125]], sampling_strategy=['Regular'], sampling_percentage=[0.25],
                                winsorize_lower_quantile=0.005, winsorize_upper_quantile=0.995, use_histogram_matching=False,
                                output_warped_image=True, num_threads=8)
        fl_crop = stage(18, 'flair_registered', reg.run().outputs.warped_image)
        subprocess.run(['quickshear', fl_crop, mask_crop, str(out / '18_flair_defaced.nii.gz')], check=True)
        fl_final = stage(19, 'flair_final', run(Normalization, input_image=str(out / '18_flair_defaced.nii.gz'), brain_mask=mask_crop, percentile=99.0).intensity_normalized)
        wmh = predict(sorted((models / 'T1.FLAIR-WMH').glob('*.tf_inference')), [t1_final, fl_final], out / '20_wmh.nii.gz')
        seg = label_clusters(nib.load(wmh).get_fdata(), 0.2, 1 - 1, nib.load(mask_crop).get_fdata())
        nib.save(nib.Nifti1Image(seg.astype(np.int16), nib.load(wmh).affine), out / '21_wmh_labelled.nii.gz')
        print('WMH clusters', int(seg.max()), 'voxels', int((seg > 0).sum()))
        if (models / 'T1.FLAIR-LAC').exists():
            lac = predict(sorted((models / 'T1.FLAIR-LAC').glob('*.tf_inference')), [t1_final, fl_final], out / '22_lac.nii.gz')
            seg = label_clusters(nib.load(lac).get_fdata(), 0.2, 3 - 1, nib.load(mask_crop).get_fdata())
            nib.save(nib.Nifti1Image(seg.astype(np.int16), nib.load(lac).affine), out / '23_lac_labelled.nii.gz')
            print('LAC clusters', int(seg.max()), 'voxels', int((seg > 0).sum()))


if __name__ == '__main__':
    main()
