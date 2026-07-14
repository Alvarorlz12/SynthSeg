"""

inject-known-resolution gate for the resolution-qc head (the sim-to-real go/no-go).

takes raw native IXI T1 volumes, runs the trained head on them, and asks two things:

  Test A (metadata-GT, no injection): run the head on the clean native scan; it should predict ~native
      (fine) resolution. validates the fine end on real anatomy/contrast/noise (no degradation injected).
  Test B (inject): degrade each scan to a known per-axis spacing `s` with the same training operator
      (DynamicGaussianBlur then MimicAcquisition, nearest-down/linear-up) and check the head recovers `s`.

operator parity is why we reuse MimicAcquisition instead of a numpy reimpl: the head trained on images
degraded by `labels_to_image_model`'s in-graph block (l2i_model.py:244-253). here we rebuild the same two
layers but feed a constant chosen resolution instead of SampleResolution's random output, so the kernel is
byte-identical to training and anatomy/contrast transfer isn't confounded with operator mismatch.

frame (default = native, P2): real IXI is ~0.94x0.94x1.2 mm but the head was trained in a 1mm-atlas frame.
we treat the native grid as the 1mm atlas (atlas_res=1) and inject `s` in voxel units (~mm, since the IXI
voxel ~1mm). no resampling, so no extra interpolation confound. `--resample-1mm` switches to the P1 frame
(resample to true 1mm isotropic first, the deployment-realistic prep). reported spacing is in the chosen
frame's units.

normalisation parity: training normalises the image to [0,1] (IntensityAugmentation clip then min-max)
before the degradation, and the head reads the un-renormalised degraded image. we mirror that: robust
min-max of the real image to [0,1] (percentile clip, not the synthetic clip=300), then degrade, then head.

Usage (GPU node, synthqc env):
    python scripts/experiments/inject_resolution_gate.py path/to/checkpoint.h5 \
        --bids-root /path/to/datasets/ixi/bids --n-subjects 20

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors
Licensed under the Apache License, Version 2.0; see http://www.apache.org/licenses/LICENSE-2.0
"""

import os
import sys
import argparse

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import nibabel as nib
from keras import models
import keras.layers as KL
import keras.backend as K

from SynthSeg.training_resolution_qc import build_directional_features
from ext.lab2im import layers as l2i_layers
from ext.lab2im import edit_tensors as l2i_et


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', help='trained checkpoint, e.g. path/to/checkpoint.h5')
    p.add_argument('--bids-root', default=None)
    p.add_argument('--n-subjects', type=int, default=20)
    p.add_argument('--inject-iso', default='2,3,4',
                   help='ISOTROPIC spacings to inject (all axes). Keep <= max_res_iso (4): training only drew '
                        'all-axes-equal spacing from U(1,4), so a (6,6,6) config is joint-pattern OOD and would '
                        'probe extrapolation, not pure sim->real transfer.')
    p.add_argument('--inject-aniso', default='4,6,8',
                   help='single-axis spacings to inject (other axes native); in-distribution up to max_res_aniso (8).')
    p.add_argument('--resample-1mm', action='store_true',
                   help='P1 frame: resample each scan to true 1mm isotropic first (else P2 native frame).')
    p.add_argument('--seed', type=int, default=0)
    # must match the trained head
    p.add_argument('--max-res-iso', type=float, default=4.0)
    p.add_argument('--max-res-aniso', type=float, default=8.0)
    p.add_argument('--context', action='store_true', help='set only if the checkpoint was trained WITH context.')
    p.add_argument('--spectral', action='store_true',
                   help='set if the checkpoint was trained WITH the per-axis spectral features (7-feat head).')
    p.add_argument('--rolloff', action='store_true',
                   help='set if the checkpoint was trained WITH the per-axis cumulative-energy roll-off feature (5-feat head).')
    p.add_argument('--drop-abs', action='store_true',
                   help='set if the checkpoint was trained WITHOUT the absolute log-raw energies lg1/lg2 (g-norm+roll head).')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--activation', type=str, default='elu')
    return p.parse_args()


ATLAS = 1.0   # the head's frame: treat the image grid as 1mm voxels (the label normalisation uses min_res=atlas)


# models
def build_head(shape, no_context, hidden, activation, spectral=False, rolloff=False, drop_abs=False, n_dims=3):
    """image to per-axis predicted normalized log-spacing. same layer names as training (resqc_h/resqc_out)
    so load_weights(by_name=True) loads the trained head; the dir-feat Lambdas carry no weights."""
    assert no_context, 'this gate only supports the no_context head; --context not implemented.'
    img = KL.Input(shape=list(shape) + [1], name='gate_image_in')
    dir_feat = build_directional_features(img, n_dims, spectral=spectral, rolloff=rolloff, drop_abs=drop_abs)
    h = KL.Dense(hidden, activation=activation, name='resqc_h')(dir_feat)
    out = KL.Dense(1, activation='sigmoid', name='resqc_out')(h)
    pred = KL.Lambda(lambda t: t[..., 0], name='resqc_pred')(out)         # [B, nd]
    return models.Model(img, pred)


def build_degrade(shape, max_res, n_dims=3):
    """replicates labels_to_image_model:244-253 (DynamicGaussianBlur then MimicAcquisition) but with the
    resolution + thickness fed as constant inputs, so a chosen per-axis spacing is injected deterministically."""
    atlas = [ATLAS] * n_dims
    img = KL.Input(shape=list(shape) + [1], name='deg_image_in')
    res = KL.Input(shape=[n_dims], name='deg_res_in')                     # chosen spacing (voxel units = ~mm)
    thick = KL.Input(shape=[n_dims], name='deg_thick_in')                 # chosen slice thickness (<= res)
    sigma = l2i_et.blurring_sigma_for_downsampling(atlas, res, thickness=thick)
    max_sig = 0.75 * np.array([max_res] * n_dims) / np.array(atlas)
    blurred = l2i_layers.DynamicGaussianBlur(max_sig, 1.03)([img, sigma])
    degraded = l2i_layers.MimicAcquisition(atlas, atlas, list(shape), False,
                                           randomize_kernel=False, randomize_up_method=False)([blurred, res])
    return models.Model([img, res, thick], degraded)


# data
def find_t1(bids_root, n):
    """Raw native IXI T1 (BIDS), no derivatives."""
    DERIVED = ('space-', 'res-', 'desc-', 't1-linear', 'bfc-n4', 'art-con', 'derivatives/', '_old')
    out = []
    for dp, _, fns in os.walk(bids_root, followlinks=True):
        low_dp = dp.lower().replace('\\', '/')
        if any(t in low_dp for t in DERIVED):
            continue
        for fn in fns:
            low = fn.lower()
            if low.endswith(('.nii', '.nii.gz')) and 't1' in low and not any(t in low for t in DERIVED):
                out.append(os.path.join(dp, fn))
    return sorted(out)[:n]


def robust_minmax(vol, lo=0.5, hi=99.5):
    """mirror training's clip then min-max to [0,1], but with percentile clip (the real intensity scale differs
    wildly from the synthetic clip=300)."""
    vol = vol.astype(np.float32)
    a, b = np.percentile(vol, lo), np.percentile(vol, hi)
    vol = np.clip(vol, a, b)
    return (vol - a) / (b - a + 1e-8)


def resample_iso1mm(vol, zooms):
    """P1 frame: resample to 1mm isotropic with linear interpolation (deployment-realistic prep)."""
    from scipy.ndimage import zoom as ndzoom
    factors = [z / 1.0 for z in zooms[:3]]                 # new_n = old_n * (zoom/1mm)
    return ndzoom(vol.astype(np.float32), factors, order=1)


def main():
    a = parse_args()
    rng = np.random.RandomState(a.seed)
    max_res = float(max(a.max_res_iso, a.max_res_aniso))
    log_min, log_span = float(np.log(ATLAS)), float(np.log(max_res) - np.log(ATLAS))
    inject_iso = [float(x) for x in a.inject_iso.split(',') if x]
    inject_aniso = [float(x) for x in a.inject_aniso.split(',') if x]

    def denorm(pred):
        return np.exp(np.asarray(pred) * log_span + log_min)   # normalized log-spacing to spacing (frame units)

    paths = find_t1(a.bids_root, a.n_subjects)
    assert paths, 'no raw native IXI T1 found under %s' % a.bids_root
    print('subjects:', len(paths), '| frame:', 'iso-1mm' if a.resample_1mm else 'native', '| ckpt:', a.checkpoint)

    heads, degrades = {}, {}        # cached by shape (IXI T1 are typically one shape, one build)

    def get_models(shape):
        if shape not in heads:
            h = build_head(shape, not a.context, a.hidden, a.activation,
                           spectral=a.spectral, rolloff=a.rolloff, drop_abs=a.drop_abs)
            h.load_weights(a.checkpoint, by_name=True)
            heads[shape] = h
            degrades[shape] = build_degrade(shape, max_res)
        return heads[shape], degrades[shape]

    testA = []                      # per-axis predicted spacing on the clean scan
    testB = {}                      # (label, axis-stratum) to list of (pred, gt)

    def record_B(key, preds, gts):
        testB.setdefault(key, []).append((preds, gts))

    for k, path in enumerate(paths):
        # reorient to canonical (~RAS) so the array axes match training's convention (aff=eye gives array==RAS) and
        # axis-0 means the same anatomical direction across subjects/modalities (IXI T1 is PSR, T2 is LAS). the
        # weight-shared head is axis-permutation consistent, so this only fixes the per-axis labelling, not the values.
        img = nib.as_closest_canonical(nib.load(path))
        vol = np.asarray(img.dataobj, dtype=np.float32)
        # per-axis native voxel size (mm) from the canonical affine: the columns of affine[:3,:3] are the directions
        # of the reoriented array axes, so their norms are the voxel sizes aligned to the array axes the head predicts
        # on. affine-derived (not header.get_zooms()) so the order tracks the reoriented data; warn
        # once if the header disagrees (would expose a latent axis-misalignment in the resample path).
        aff_zooms = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0)).astype(float)
        hdr_zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
        if k == 0 and not np.allclose(sorted(aff_zooms), sorted(hdr_zooms), atol=1e-3):
            print('  [warn] affine vs header zooms disagree: aff=%s hdr=%s'
                  % (np.round(aff_zooms, 3), np.round(hdr_zooms, 3)))
        if a.resample_1mm:
            vol = resample_iso1mm(vol, aff_zooms)
        vol = robust_minmax(vol)
        shape = vol.shape
        head, degrade = get_models(shape)
        x = vol[np.newaxis, ..., np.newaxis]

        # Test A: clean scan, the head should read ~ the native effective resolution per array axis
        sA = denorm(head.predict(x)[0])
        testA.append((sA, aff_zooms))

        # Test B: inject known spacings with the exact training operator
        n_dims = 3
        for s in inject_iso:
            res = np.array([s] * n_dims, dtype='float32')
            thick = np.array([rng.uniform(ATLAS, r) for r in res], dtype='float32')   # thickness ~ U(1,r) per axis
            deg = degrade.predict([x, res[np.newaxis], thick[np.newaxis]])
            pB = denorm(head.predict(deg)[0])                                          # [3]
            record_B(('iso', s), pB, res)
        for s in inject_aniso:
            for ax in range(n_dims):
                res = np.array([ATLAS] * n_dims, dtype='float32'); res[ax] = s
                thick = np.array([rng.uniform(ATLAS, r) for r in res], dtype='float32')
                deg = degrade.predict([x, res[np.newaxis], thick[np.newaxis]])
                pB = denorm(head.predict(deg)[0])
                record_B(('aniso', s), pB, res)
        print('  [%2d/%2d] %s  shape=%s  TestA pred=%s  native_mm=%s'
              % (k + 1, len(paths), os.path.basename(path)[:34], shape, np.round(sA, 2), np.round(aff_zooms, 2)))

    # report
    preds_A = np.array([p for p, _ in testA])                # [N, 3] predicted spacing per array axis
    nat_A = np.array([z for _, z in testA])                  # [N, 3] native voxel size (mm) per canonical array axis
    # GT for a clean scan is frame-dependent:
    #   P1 (--resample-1mm): grid is now 1mm iso but detail-loss persists, head should read ~ the native mm zoom.
    #   P2 (native grid):    head treats the grid as 1mm-atlas units, so a full-res native scan should read ~1.0 on
    #                        every axis (it cannot perceive the native mm anisotropy without a mm reference).
    gt_A = nat_A if a.resample_1mm else np.ones_like(nat_A)
    gt_name = 'NATIVE mm zoom (P1)' if a.resample_1mm else '1.0 grid-unit (P2)'
    print('\n=== TEST A (clean scan; GT = %s) ===' % gt_name)
    print('  per-axis predicted spacing  mean=%s  std=%s' % (np.round(preds_A.mean(0), 2), np.round(preds_A.std(0), 2)))
    print('  per-axis NATIVE mm zoom     mean=%s  std=%s' % (np.round(nat_A.mean(0), 2), np.round(nat_A.std(0), 2)))
    print('  per-axis MAE vs GT: %s   (pooled %.3f)'
          % (np.round(np.abs(preds_A - gt_A).mean(0), 3), float(np.abs(preds_A - gt_A).mean())))
    print('  per-axis MAE vs 1.0 (old yardstick): %s' % np.round(np.abs(preds_A - 1.0).mean(0), 3))
    # decides "bias vs correct detection": does the head put its coarsest reading on the truly-coarsest native axis?
    # if pred-argmax tracks native-argmax, a coarse clean reading is correct anisotropy detection, not a fixed-axis bias.
    pred_coarse, nat_coarse = preds_A.argmax(1), nat_A.argmax(1)
    print('  coarsest-axis: pred-argmax hist=%s  native-argmax hist=%s  match=%.2f'
          % (np.bincount(pred_coarse, minlength=3), np.bincount(nat_coarse, minlength=3),
             float((pred_coarse == nat_coarse).mean())))
    print('  fraction predicted < 1.3 per axis: %s' % np.round((preds_A < 1.3).mean(0), 2))

    print('\n=== TEST B (injected spacing recovery; operator-matched) ===')
    rows = []
    for (kind, s), items in sorted(testB.items()):
        preds = np.concatenate([p for p, _ in items])        # all axes, all subjects
        gts = np.concatenate([g for _, g in items])
        inj = gts > ATLAS + 1e-6                              # the degraded axis/axes
        nat = ~inj
        mae_inj = float(np.abs(preds[inj] - gts[inj]).mean()) if inj.any() else float('nan')
        mae_nat = float(np.abs(preds[nat] - gts[nat]).mean()) if nat.any() else float('nan')
        meanpred_inj = float(preds[inj].mean()) if inj.any() else float('nan')
        rows.append((kind, s, meanpred_inj, mae_inj, mae_nat))
        print('  %-6s s=%.1f  injected-axis: mean pred=%.2f  MAE=%.3f   |  native-axis MAE=%.3f'
              % (kind, s, meanpred_inj, mae_inj, mae_nat))

    # pooled r over all injected-axis (pred, gt) pairs
    allp = np.concatenate([np.concatenate([p for p, _ in items]) for items in testB.values()])
    allg = np.concatenate([np.concatenate([g for _, g in items]) for items in testB.values()])
    inj = allg > ATLAS + 1e-6
    r = float(np.corrcoef(allp[inj], allg[inj])[0, 1]) if inj.sum() > 2 else float('nan')
    print('\n  pooled r (injected axes, pred vs true) = %.3f   over %d (subject x config x axis) points'
          % (r, int(inj.sum())))
    # aniso-only r: the iso config repeats the same spacing on all 3 axes (near-duplicate, correlated points that
    # inflate effective N); the single-axis aniso points are independent, so this is the more honest correlation.
    aniso_items = [items for k, items in testB.items() if k[0] == 'aniso']
    if aniso_items:
        ap = np.concatenate([np.concatenate([p for p, _ in items]) for items in aniso_items])
        ag = np.concatenate([np.concatenate([g for _, g in items]) for items in aniso_items])
        ai = ag > ATLAS + 1e-6
        ra = float(np.corrcoef(ap[ai], ag[ai])[0, 1]) if ai.sum() > 2 else float('nan')
        print('  aniso-only r (independent single-axis points) = %.3f   over %d points' % (ra, int(ai.sum())))
    print('\nNOTE: Test A GT is now the NATIVE per-axis mm zoom under --resample-1mm (P1): detail-loss persists after '
          'upsampling to 1mm, so the head SHOULD read the original mm spacing per axis. Under P2 (native frame) the GT '
          'is ~1.0 grid-units. The coarsest-axis match decides whether a coarse clean reading is a CONFOUND (lands on '
          'the wrong axis) or CORRECT native-anisotropy detection (lands on the true ~1.2mm axis).')


if __name__ == '__main__':
    main()
