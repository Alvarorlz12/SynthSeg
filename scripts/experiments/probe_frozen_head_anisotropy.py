"""

frozen-head anisotropy sensitivity probe (no training).

why: the head's per-axis read is confounded on real scans by a training-distribution
gap. SampleConditionalGMM (labels_to_image_model.py:203) draws each voxel i.i.d., so the
synthetic clean content is spectrally roughly white/isotropic, and every axis's clean baseline bandwidth
is identical in training. that leaves the head free to read "this axis is smoother" as "this axis is
lower-res", which inverts on real brains (A-P is genuinely smoother anatomy). the proposed fix is a
domain-randomization augmentation that injects a randomized per-axis intrinsic (content-only,
pre-degradation) low-pass, so the clean baseline becomes a per-sample nuisance decorrelated from the
resolution label. the plan hinges on one untested assumption: that this content low-pass can be
calibrated into a band that does not collide with the resolution-degradation signature (a Gaussian
content low-pass and the degradation blur overlap by construction, and on the ~62% native axes the
content low-pass would be the only band reduction, which could poison the iso/absolute anchor).

what this probe decides (no epochs; reuses the gate's frozen-head infra):
  (1) mechanism (confirm/falsify the linchpin): take a clean synthetic volume (true resolution = native on
      all axes), apply a known per-axis Gaussian content low-pass (not MimicAcquisition), and read the frozen
      head's per-axis prediction. if the head reads the smoothed axis as coarser (higher apparent spacing) and
      its coarsest-axis argmax tracks the smoothed axis, the mechanism is demonstrated in a controlled
      synthetic setting, reproducing the IXI inversion. if it does not move, the distribution-gap story is
      wrong and the plan needs rethinking.
  (2) calibration band (de-risk the load-bearing detail): sweep the content sigma and, on the same clean
      volume, sweep a real degradation (build_degrade, the exact training operator) to a range of spacings.
      compare (a) how far a content-blurred but truly native axis departs from ~1.0 (the label noise the
      augmentation would inject), and (b) at matched apparent spacing, whether the per-axis feature vectors
      of content-blur vs real degradation are distinguishable (the grid imprint gives real degradation a
      finite-diff / roll-off signature that a pure Gaussian content blur lacks). separable features mean a
      retrained head can tell the augmentation from a true resolution step, so a safe band exists; indistinct
      features mean spectral collision, and the augmentation will inject genuine label noise.

synthetic-only, weight-free diagnostic: the directional features are a fixed transform and the head is the
frozen trained checkpoint (loaded by_name exactly as the inject gate does). runs on CPU.

Usage (synthqc env; CPU is fine):
    python scripts/experiments/probe_frozen_head_anisotropy.py path/to/checkpoint.h5 --rolloff
    # 4-feature head:  python scripts/experiments/probe_frozen_head_anisotropy.py path/to/checkpoint.h5
    # quick local smoke at a smaller shape:  ... --output-shape 96 --n-levels 5 --n-volumes 3

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
from scipy.ndimage import gaussian_filter1d
from keras import models

from SynthSeg.labels_to_image_model import labels_to_image_model
from SynthSeg.model_inputs import build_model_inputs
from ext.lab2im import utils
from ext.lab2im import layers as l2i_layers

# reuse the exact gate infra (frozen head loaded by_name + the byte-identical training degradation operator).
# import the sibling module flatly (robust to package layout): add this file's dir to sys.path first.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inject_resolution_gate import build_head, build_degrade, ATLAS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', help='trained checkpoint, e.g. path/to/checkpoint.h5')
    p.add_argument('--spectral', action='store_true', help='checkpoint trained with the spectral features (7-feat head).')
    p.add_argument('--rolloff', action='store_true', help='checkpoint trained with the roll-off feature (5-feat head).')
    p.add_argument('--drop-abs', action='store_true', help='checkpoint trained without lg1/lg2 (g-norm+roll head).')
    p.add_argument('--context', action='store_true', help='unsupported here; the no_context head only.')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--activation', type=str, default='elu')
    # generation (match training defaults so the clean content is the training distribution)
    p.add_argument('--output-shape', type=int, default=160, help='160 = training/gate; smaller = quick local smoke.')
    p.add_argument('--n-levels', type=int, default=5, help='sets output_div_by_n = 2**n_levels (160/32=5).')
    p.add_argument('--n-neutral-labels', type=int, default=18)
    p.add_argument('--max-res-iso', type=float, default=4.0)
    p.add_argument('--max-res-aniso', type=float, default=8.0)
    p.add_argument('--n-volumes', type=int, default=6,
                   help='clean synthetic volumes to average over. Use ~20 for the EXP3 gate (bootstrap CI).')
    p.add_argument('--content-sigmas', default='0.5,0.75,1.0,1.5,2.0,3.0',
                   help='EXP1/EXP2 per-axis Gaussian CONTENT low-pass sigmas to sweep (voxels).')
    p.add_argument('--inject-spacings', default='1.5,2,3,4',
                   help='EXP1/EXP2 REAL degradation spacings to sweep (exact training operator).')
    # EXP3: the transferable-only mild-band gate
    p.add_argument('--gate-only', action='store_true',
                   help='skip EXP1/EXP2 and run only the EXP0 sanity check + the EXP3 gate.')
    p.add_argument('--mild-content-sigmas', default='0.4,0.5,0.6,0.75,0.9,1.1',
                   help='EXP3 mild-band content sigmas (span apparent ~1.0-2.0 so the interpolation has support).')
    p.add_argument('--mild-spacings', default='1.5,2.0',
                   help='EXP3 mild REAL degradation spacings (the band that matches real anatomical anisotropy).')
    p.add_argument('--n-bootstrap', type=int, default=2000, help='EXP3 bootstrap resamples over volumes for the CI.')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def make_clean_generator(a, gen_labels, labels_shape, atlas_res, output_div):
    """the training generator (clean scope, randomise_res on so the graph is identical), but we tap the
    pre-degradation clean image, not the degraded output. randomise_res must stay True for return_resolution,
    but we never read the degraded output[0], so the sampled resolution is irrelevant to the tapped clean image
    (the clean GMM content is always native, regardless of the resolution injected downstream)."""
    return labels_to_image_model(labels_shape=labels_shape, n_channels=1,
                                 generation_labels=gen_labels, output_labels=gen_labels,
                                 n_neutral_labels=a.n_neutral_labels, atlas_res=atlas_res,
                                 target_res=None, output_shape=a.output_shape, output_div_by_n=output_div,
                                 flipping=False, aff=np.eye(4),
                                 scaling_bounds=False, rotation_bounds=False, shearing_bounds=False,
                                 translation_bounds=False, nonlin_std=0,
                                 randomise_res=True, max_res_iso=a.max_res_iso, max_res_aniso=a.max_res_aniso,
                                 grid_ablation='none', bias_field_std=0, return_resolution=True)


def tap_clean_image(generator):
    """the clean synthetic image = the output of IntensityAugmentation (l2i:222), before the resolution
    degradation block. grab it by layer type (exactly one such layer), giving a Model(inputs to clean image)."""
    ia = [l for l in generator.layers if isinstance(l, l2i_layers.IntensityAugmentation)]
    assert len(ia) == 1, 'expected exactly one IntensityAugmentation layer, found %d' % len(ia)
    return models.Model(generator.inputs, ia[0].output)


def blur_axis(vol, ax, sigma):
    """pure per-axis Gaussian content low-pass (the augmentation candidate) on a [X,Y,Z] volume. truncated
    Gaussian (scipy default truncate=4.0); reflect mode (no edge wrap). not renormalized afterwards: intrinsic
    smoothing genuinely lowers contrast, the absolute-energy drop the head's lg features would see."""
    if sigma <= 0:
        return vol
    return gaussian_filter1d(vol.astype(np.float32), sigma=float(sigma), axis=ax, mode='reflect')


def exp3_gate(a, AX, n_dims, cleans, pred_axis, feat_axes, degrade, rng, feat_names):
    """EXP3. EXP2's raw feature-L2 mixes in the absolute lg1/lg2 energies, which do not transfer (real absolute
    energy != synthetic), so this gate re-scores content-vs-real distinguishability on the transferable features
    only (g1n, g2n, roll, the cues that deploy): (a) in the mild band the augmentation will actually use,
    (b) interpolating the content feature vector at the exact real apparent spacing per volume (removes the
    nearest-match confound), and (c) with a bootstrap CI over volumes. pass means a transferable mild-band
    separation survives, i.e. the augmentation can teach a cue that transfers."""
    mild_sigmas = sorted(float(x) for x in a.mild_content_sigmas.split(',') if x)
    mild_spac = [float(x) for x in a.mild_spacings.split(',') if x]
    base_n = 2 if a.drop_abs else 4                                      # feat order: g1n,g2n,[lg1,lg2],[spec3],[roll]
    roll_idx = (base_n + (3 if a.spectral else 0)) if a.rolloff else None
    transf_idx = [0, 1] + ([roll_idx] if roll_idx is not None else [])   # transferable = relative g-norm (+ roll)
    F = len(feat_names)
    print('\n--- EXP 3  TRANSFERABLE-only mild-band distinguishability GATE (the pre-GPU decision) ---')
    print('  transferable feats = %s | mild content sigmas=%s | mild real spacings=%s | n_vol=%d n_boot=%d'
          % ([feat_names[i] for i in transf_idx], mild_sigmas, mild_spac, len(cleans), a.n_bootstrap))

    nC, nS = len(mild_sigmas), len(mild_spac)
    c_ap = np.zeros((len(cleans), nC)); c_fv = np.zeros((len(cleans), nC, F))     # content: apparent + axis-AX feats
    for vi, v in enumerate(cleans):
        for si, sg in enumerate(mild_sigmas):
            vb = blur_axis(v, AX, sg)
            c_ap[vi, si] = pred_axis(vb)[AX]; c_fv[vi, si] = feat_axes(vb)[AX]
    r_ap = np.zeros((len(cleans), nS)); r_fv = np.zeros((len(cleans), nS, F))     # real degradation: apparent + feats
    for vi, v in enumerate(cleans):
        for si, s in enumerate(mild_spac):
            res = np.array([ATLAS] * n_dims, 'float32'); res[AX] = s
            thick = np.array([rng.uniform(ATLAS, r) for r in res], 'float32')      # thickness ~ U(1,s) as in training
            deg = degrade.predict([v[np.newaxis, ..., np.newaxis], res[np.newaxis], thick[np.newaxis]])[0, ..., 0]
            r_ap[vi, si] = pred_axis(deg)[AX]; r_fv[vi, si] = feat_axes(deg)[AX]

    # per (volume, mild s): interpolate the content feat vector at the real's exact apparent spacing, then the
    # transferable-only L2 gap (and full L2 for reference, and the roll delta).
    tL2 = np.zeros((len(cleans), nS)); fL2 = np.zeros((len(cleans), nS)); rdl = np.zeros((len(cleans), nS))
    for vi in range(len(cleans)):
        order = np.argsort(c_ap[vi]); xp = c_ap[vi][order]
        for si in range(nS):
            ci = np.array([np.interp(r_ap[vi, si], xp, c_fv[vi][order, f]) for f in range(F)])
            d = r_fv[vi, si] - ci
            tL2[vi, si] = np.linalg.norm(d[transf_idx]); fL2[vi, si] = np.linalg.norm(d)
            if roll_idx is not None:
                rdl[vi, si] = d[roll_idx]

    def boot_ci(per_vol, seed=1):
        rs = np.random.RandomState(seed); n = len(per_vol)
        stats = [per_vol[rs.randint(0, n, n)].mean() for _ in range(a.n_bootstrap)]
        return float(np.mean(per_vol)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

    for si, s in enumerate(mild_spac):
        print('    s=%.2f (apparent %.2f) | transf-L2=%.3f  full-L2=%.3f  roll-delta=%+.3f'
              % (s, r_ap[:, si].mean(), tL2[:, si].mean(), fL2[:, si].mean(), rdl[:, si].mean()))
    t_m, t_lo, t_hi = boot_ci(tL2.mean(1)); f_m, f_lo, f_hi = boot_ci(fL2.mean(1)); r_m, r_lo, r_hi = boot_ci(rdl.mean(1))
    print('  pooled mild band: transferable-L2 = %.3f  [95%% CI %.3f, %.3f]' % (t_m, t_lo, t_hi))
    print('                    full-L2         = %.3f  [95%% CI %.3f, %.3f]   (reference; lg-inflated)' % (f_m, f_lo, f_hi))
    if roll_idx is not None:
        print('                    roll-delta      = %+.3f [95%% CI %+.3f, %+.3f]' % (r_m, r_lo, r_hi))

    # gate: (a) the transferable separation must be meaningfully above the ~0.1 noise floor (lower-CI > 0.10), and
    # (b) the roll-off cue (the only amplitude/phase-invariant channel) must move beyond ~2 quantization bins with a
    # CI that excludes 0, i.e. roll genuinely separates content-blur from real degradation, not just lg.
    roll_step = 2.0 / a.output_shape                                   # roll-off normalized-freq bin = 1/(N//2)
    gate_l2 = t_lo > 0.10
    gate_roll = (roll_idx is None) or (abs(r_m) > max(0.025, 2 * roll_step) and r_lo * r_hi > 0)
    print('\n  === EXP3 GATE (transferable-only, mild band) ===')
    print('  (a) transferable-L2 lower-CI %.3f > 0.10 ?  %s' % (t_lo, 'PASS' if gate_l2 else 'FAIL'))
    if roll_idx is not None:
        print('  (b) |roll-delta| %.3f > max(.025, 2*bin=%.3f) AND CI excludes 0 ?  %s'
              % (abs(r_m), 2 * roll_step, 'PASS' if gate_roll else 'FAIL'))
    go = gate_l2 and gate_roll
    print('  >>> %s' % (
        'GO -> a TRANSFERABLE mild-band separation survives; the augmentation can teach a cue that DEPLOYS. '
        'Proceed to the GPU run: mild randomized per-axis sigma ~U(0,0.8) post-IntensityAugmentation, DROP lg1/lg2, '
        'grid_ablation=kernel_phase. Watch Test-B: does it still DETECT real degradation after ignoring smoothness?'
        if go else
        'NO-GO as-is -> the mild-band transferable separation is too thin / lg-driven. Do NOT retrain blindly: first '
        'STRENGTHEN the transferable bandwidth cue (a Nyquist / noise-floor CUTOFF feature, or a 2nd roll fraction), '
        'or switch to a STRUCTURED low-freq anisotropy aug form; then re-run this gate.'))


def main():
    a = parse_args()
    assert not a.context, 'this probe only supports the no_context head (matches the gate).'
    rng = np.random.RandomState(a.seed)
    n_dims = 3
    output_div = 2 ** a.n_levels
    content_sigmas = [float(x) for x in a.content_sigmas.split(',') if x]
    inject_spacings = [float(x) for x in a.inject_spacings.split(',') if x]
    max_res = float(max(a.max_res_iso, a.max_res_aniso))
    log_min, log_span = float(np.log(ATLAS)), float(np.log(max_res) - np.log(ATLAS))

    def denorm(pred):
        """frozen head's normalized log-spacing to apparent spacing (mm/voxel units), exact gate inverse."""
        return np.exp(np.asarray(pred) * log_span + log_min)

    # data / generator (training distribution)
    DATA = os.path.join(ROOT, 'data')
    PRIORS = os.path.join(DATA, 'labels_classes_priors')
    LABELS_DIR = os.path.join(DATA, 'training_label_maps')
    gen_labels = utils.load_array_if_path(os.path.join(PRIORS, 'generation_labels.npy'))
    gen_classes = utils.load_array_if_path(os.path.join(PRIORS, 'generation_classes.npy'))
    labels_paths = utils.list_images_in_folder(LABELS_DIR)
    labels_shape, _, _, _, _, atlas_res = utils.get_volume_info(labels_paths[0], aff_ref=np.eye(4))

    generator = make_clean_generator(a, gen_labels, labels_shape, atlas_res, output_div)
    clean_model = tap_clean_image(generator)
    src = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels), batchsize=1,
                             n_channels=1, generation_classes=gen_classes, prior_distributions='uniform')

    shape = tuple([a.output_shape] * n_dims)
    head = build_head(shape, True, a.hidden, a.activation, spectral=a.spectral, rolloff=a.rolloff, drop_abs=a.drop_abs)
    head.load_weights(a.checkpoint, by_name=True)
    degrade = build_degrade(shape, max_res)
    # feature model: reuse the head's own directional-feature layer (no duplicate-name clash), gives [B, nd, F]
    feat_name = 'dir_feat_cat' if (a.spectral or a.rolloff) else 'dir_feat'
    feat_model = models.Model(head.input, head.get_layer(feat_name).output)
    feat_names = (['g1n', 'g2n']
                  + ([] if a.drop_abs else ['lg1', 'lg2'])
                  + (['cen', 'ent', 'band'] if a.spectral else [])
                  + (['roll'] if a.rolloff else []))

    print('=' * 96)
    print('FROZEN-HEAD ANISOTROPY PROBE  | ckpt=%s | shape=%s | feats=%s | n_vol=%d'
          % (a.checkpoint, shape, feat_names, a.n_volumes))
    print('=' * 96)

    # generate the clean synthetic volumes once (reused across all sweeps)
    cleans = []
    for _ in range(a.n_volumes):
        x = clean_model.predict(next(src))[0, ..., 0]               # [X,Y,Z], min-max normalized, native content
        cleans.append(x.astype(np.float32))
    print('clean volumes: intensity range [%.3f, %.3f], mean=%.3f'
          % (np.min(cleans), np.max(cleans), float(np.mean(cleans))))

    def pred_axis(vol):
        """frozen head per-axis apparent spacing on a [X,Y,Z] volume."""
        return denorm(head.predict(vol[np.newaxis, ..., np.newaxis])[0])     # [3] mm

    def feat_axes(vol):
        """frozen head's per-axis feature vectors on a [X,Y,Z] volume, gives [3, F]."""
        return np.asarray(feat_model.predict(vol[np.newaxis, ..., np.newaxis])[0])

    # exp 0 sanity: clean (undegraded) synthetic should read ~native (~1.0) on all axes
    base = np.array([pred_axis(v) for v in cleans])                 # [N, 3]
    print('\n--- EXP 0  clean (sigma=0) sanity ---')
    print('  per-axis apparent spacing  mean=%s  std=%s' % (np.round(base.mean(0), 3), np.round(base.std(0), 3)))
    print('  (expect ~1.0 on all axes if the clean synthetic reads native)')

    AX = 0   # the axis EXP2/EXP3 manipulate (the head is permutation-consistent, so any axis is representative)

    # --gate-only: the mechanism (EXP1) is already confirmed; run only the decisive transferable gate (EXP3).
    if a.gate_only:
        exp3_gate(a, AX, n_dims, cleans, pred_axis, feat_axes, degrade, rng, feat_names)
        return

    # exp 1 mechanism: does a pure content low-pass on one axis make the head read that axis coarse?
    #   for each axis k and each content sigma, blur only axis k, then read per-axis apparent spacing.
    #   mechanism confirmed if apparent[k] rises with sigma and coarsest-axis argmax == k.
    print('\n--- EXP 1  mechanism: content low-pass on a single axis (apparent spacing of the BLURRED axis) ---')
    print('  blur-axis |  ' + '  '.join('s=%.2f' % s for s in content_sigmas) + '   | argmax-match')
    mech_rise, mech_match = [], []
    for k in range(n_dims):
        row, matches = [], []
        for sg in content_sigmas:
            preds = np.array([pred_axis(blur_axis(v, k, sg)) for v in cleans])   # [N,3]
            row.append(preds[:, k].mean())
            matches.append(float((preds.argmax(1) == k).mean()))
        # rise = apparent[k] at the largest sigma minus the clean baseline on axis k
        mech_rise.append(row[-1] - base[:, k].mean())
        mech_match.append(matches[-1])
        print('   axis-%d   |  ' % k + '  '.join('%5.2f' % v for v in row)
              + '   | ' + '  '.join('%.2f' % m for m in matches))
    print('  => mean apparent-spacing RISE (largest sigma vs clean): %s' % np.round(mech_rise, 3))
    print('  => coarsest-axis match at largest sigma (per blurred axis): %s' % np.round(mech_match, 2))

    # exp 2 calibration band: content low-pass vs the exact real degradation, on axis 0, matched by
    #   apparent spacing. compares (a) departure-from-native and (b) feature-vector distinguishability.
    AX = 0
    print('\n--- EXP 2  calibration: CONTENT low-pass vs REAL degradation on axis-%d ---' % AX)

    # content sweep on axis AX: apparent spacing + mean feature vector of that axis
    print('  [content low-pass]  sigma -> apparent spacing | axis-%d feats %s' % (AX, feat_names))
    content_rows = []
    for sg in content_sigmas:
        aps, fvs = [], []
        for v in cleans:
            vb = blur_axis(v, AX, sg)
            aps.append(pred_axis(vb)[AX]); fvs.append(feat_axes(vb)[AX])
        ap, fv = float(np.mean(aps)), np.mean(fvs, 0)
        content_rows.append((sg, ap, fv))
        print('    sigma=%4.2f -> %5.2f mm | %s' % (sg, ap, np.array2string(fv, precision=3, suppress_small=True)))

    # real degradation sweep on axis AX (exact training operator): apparent spacing + mean feature vector
    print('  [real degradation]  spacing -> apparent spacing | axis-%d feats %s' % (AX, feat_names))
    degrade_rows = []
    for s in inject_spacings:
        res = np.array([ATLAS] * n_dims, dtype='float32'); res[AX] = s
        aps, fvs = [], []
        for v in cleans:
            thick = np.array([rng.uniform(ATLAS, r) for r in res], dtype='float32')   # thickness ~ U(1,s), as in training
            deg = degrade.predict([v[np.newaxis, ..., np.newaxis], res[np.newaxis], thick[np.newaxis]])[0, ..., 0]
            aps.append(pred_axis(deg)[AX]); fvs.append(feat_axes(deg)[AX])
        ap, fv = float(np.mean(aps)), np.mean(fvs, 0)
        degrade_rows.append((s, ap, fv))
        print('    s=%4.2f     -> %5.2f mm | %s' % (s, ap, np.array2string(fv, precision=3, suppress_small=True)))

    # distinguishability: for each real degradation s, find the content sigma with the nearest apparent spacing,
    # then measure the feature-vector L2 gap (and the largest per-feature delta). separable means a safe band exists.
    print('\n  [distinguishability]  at MATCHED apparent spacing, content-blur vs real-degradation feature gap:')
    print('    real-s (apparent) ~= content-sigma (apparent) | feat-L2 | largest per-feat delta')
    gaps = []
    for s, ap_s, fv_s in degrade_rows:
        j = int(np.argmin([abs(ap_c - ap_s) for _, ap_c, _ in content_rows]))
        sg, ap_c, fv_c = content_rows[j]
        l2 = float(np.linalg.norm(fv_s - fv_c))
        di = int(np.argmax(np.abs(fv_s - fv_c)))
        gaps.append(l2)
        print('    s=%.2f (%.2f mm) ~= sigma=%.2f (%.2f mm) | L2=%.3f | %s delta=%+.3f'
              % (s, ap_s, sg, ap_c, l2, feat_names[di], float(fv_s[di] - fv_c[di])))

    # verdict (heuristic; read the tables above for the full picture)
    print('\n=== VERDICT (heuristic) ===')
    mech_ok = (np.mean(mech_rise) > 0.3) and (np.mean(mech_match) > 0.5)
    print('  MECHANISM (linchpin): %s' % (
        'CONFIRMED -- content smoothness is read as lower resolution (mean rise %.2f mm, match %.2f). '
        'Reproduces the IXI inversion in controlled synthetic; the distribution-gap story holds.'
        % (float(np.mean(mech_rise)), float(np.mean(mech_match))) if mech_ok else
        'NOT CONFIRMED -- the head did NOT systematically read content smoothness as resolution '
        '(mean rise %.2f mm, match %.2f). The distribution-gap story may be wrong -> RETHINK the plan.'
        % (float(np.mean(mech_rise)), float(np.mean(mech_match)))))
    # a crude separability read: median matched-apparent feature L2 gap vs the clean-to-degraded feature scale
    band_hint = float(np.median(gaps)) if gaps else float('nan')
    print('  CALIBRATION BAND: median matched-apparent feature-L2 gap (content vs degradation) = %.3f.' % band_hint)
    print('    Larger gap => content-blur and real-degradation are FEATURE-separable (the grid imprint shows) =>')
    print('    a retrained head can distinguish the augmentation from a true resolution step => a safe band EXISTS.')
    print('    Near-zero gap => spectral COLLISION => the augmentation injects genuine label noise. Read the')
    print('    per-feature deltas above (esp. roll-off / g-norm) to see WHICH cue separates them, if any.')
    print('\n  NOTE: this EXP2 L2 is INFLATED by the non-transferable lg1/lg2 absolute energies; the EXP3 gate below')
    print('        re-scores on transferable features only and is the decision-maker.')

    # EXP3 always runs (the decisive transferable-only mild-band gate)
    exp3_gate(a, AX, n_dims, cleans, pred_axis, feat_axes, degrade, rng, feat_names)


if __name__ == '__main__':
    main()
