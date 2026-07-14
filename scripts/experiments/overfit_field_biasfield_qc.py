"""

FIELD-ESTIMATION bias-QC diagnostic (DeepN4-style), on a frozen anatomy.

A 3D U-Net predicts the log-bias field b_hat(x) from the image, supervised PER-VOXEL (masked L2) against
the in-graph ground-truth log_bias. That is ~1e6 supervision signals per sample instead of the 1 scalar a
severity regressor gets, which is what makes it trainable at bs=1: a scalar objective at bs=1 gravitates to
the mean. The QC severity score is a fixed in-graph reduction of
b_hat over the brain mask: masked std (== physical std_log) by default, or masked RMS with --rms-score (so
std-vs-RMS is just a reduction choice, not a learning-target change).

Precedent: Kanakaraj 2024 "DeepN4", a SynthSeg-family 3D U-Net estimating the log-bias field, masked L2
loss, log-space. Here we have the exact ground-truth field in-graph, so supervision is direct.

Requires labels_to_image_model to expose the field + mask as named layers 'bias_field_log' and
'bias_mask_labels' (added alongside 'bias_field_std').

Frozen-anatomy protocol (same as overfit_biasfield_qc.py): freeze --k anatomy tuples, only the in-graph
bias sigma varies; PHASE 0 probe at init, PHASE 1 overfit, PHASE 2 probe post-training.

Run on a GPU node in the `synthqc` env (a full 160^3 U-Net is heavy; if OOM on a 32GB V100, use
--output-shape 128 and/or --unet-feat-count 16):
    srun --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=01:30:00 --pty bash -lc \
      'module load miniforge; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate synthqc; \
       cd ~/SynthQC; python scripts/experiments/overfit_field_biasfield_qc.py --steps 300'

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
from keras import models
from keras.optimizers import Adam
import keras.layers as KL
import keras.backend as K

from SynthSeg.labels_to_image_model import labels_to_image_model, _masked_std
from SynthSeg.model_inputs import build_model_inputs
from SynthSeg import metrics_model as metrics
from ext.lab2im import utils
from ext.lab2im.layers import GaussianBlur
from ext.neuron import models as nrn_models
from ext.neuron import layers as nrn_layers


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch', type=int, default=1, help='effective batch (dense supervision works at 1; >1 is heavy)')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--clipnorm', type=float, default=0.0)
    p.add_argument('--k', type=int, default=1, help='number of frozen anatomy tuples')
    p.add_argument('--fresh', action='store_true', help='RANDOMISED-CONTRAST regime: draw a fresh anatomy+contrast+'
                   'bias every step and every probe draw (the real task) instead of freezing the anatomy')
    p.add_argument('--fixed-contrast', action='store_true', help='DIAGNOSTIC (requires --fresh): keep the GMM '
                   'means/stds frozen while anatomy + in-graph bias stay fresh -> isolates whether random CONTRAST '
                   '(not gradient noise) is what degrades the fresh regime. Expect r up / shrinkage down if so.')
    p.add_argument('--log-input', action='store_true', help='feed log(image) to the U-Net instead of image: the '
                   'multiplicative bias becomes additive in log-space -> separable from the multiplicative contrast, '
                   'and the per-sample min-max gain becomes a ~global constant the masked-std score removes (exact when '
                   'background~0, true here; the L2 loss stays offset-sensitive so the net also learns the -log(M) offset).')
    p.add_argument('--smooth-sigma', type=float, default=0.0, help='SMOOTHNESS PRIOR (DeepN4, Opt A): masked Gaussian '
                   'blur (this sigma, in voxels; ~4-5 keeps the ~40-vox control-point scale while killing sub-15-vox '
                   'noise; 8 over-smooths the genuine field to ~45%) applied in-graph to the '
                   'predicted field before both the loss and the score. 0 = off.')
    p.add_argument('--probe-n', type=int, default=96)
    p.add_argument('--log-step', type=int, default=20)
    p.add_argument('--rms-score', action='store_true', help='use masked RMS (not std) as the QC severity reduction')
    p.add_argument('--save-fields', type=int, default=0, help='after training, save this many [image | GT | pred | diff] '
                   'field-slice PNGs (GT & pred share the colour scale) to diagnose shape vs amplitude/shrinkage of b_hat')
    p.add_argument('--plot-dir', type=str, default='field_plots', help='dir (under repo root) for --save-fields PNGs')
    # architecture
    p.add_argument('--arch', choices=['unet', 'cp'], default='unet', help="'unet' = full encoder-decoder predicting a "
                   "dense 160^3 field (validated default); 'cp' = CONTROL-POINT head: encoder-only -> 1x1x1 conv to one "
                   "coefficient per R^3 bottleneck cell (R=output_shape/2^(n_levels-1)) -> trilinear Resize to full res, "
                   "band-limited by construction. Use --n-levels 6 for R=5 (match generator), --n-levels 5 for R=10.")
    p.add_argument('--skip-n-concatenations', type=int, default=0, help='[unet arch] DROP the top-N high-res skip '
                   'connections (keep the deep/coarse ones): removes the fine-detail path where multiplicative GMM '
                   'contrast aliases into the band-limited field, while keeping skip-based localisation. Sweep 1,2,3.')
    p.add_argument('--demean-loss', action='store_true', help='subtract the masked-mean from both b_hat and the GT '
                   'field before the L2 (train the field shape only): the absolute DC offset is unrecoverable from a '
                   'min-max image and penalising it makes the net shrink the whole field; the masked-std score is DC-invariant.')
    # diagnostics
    p.add_argument('--brain-mask', action='store_true', help='DIAGNOSTIC: restrict the score, loss AND std target to '
                   'the %d CEREBRAL labels only (exclude extracerebral fat/skull/eyes/CSF/vessels), instead of the '
                   'default whole-head labels!=0. Matches DeepN4 (loss within a brain mask) and removes the visible '
                   'extracerebral confound from both the target and the measurement.' % len(BRAIN_LABELS))
    p.add_argument('--null-test', type=int, default=0, help='DIAGNOSTIC [unet arch]: after training, probe this many '
                   'BIAS-FREE draws (a separate generator with bias_field_std=0; random contrast+anatomy) through a '
                   'weight-copied U-Net, and report the score distribution (an honest bias score must read ~0) + '
                   'corr(score, image brightness/contrast) -> tests whether the score reads intensity, not bias.')
    p.add_argument('--realistic-contrast', action='store_true', help='draw the GMM contrast from the '
                   'realistic T1w priors (generation_classes_contrast_specific + prior_means_t1/prior_stds_t1, NORMAL '
                   'dist) instead of fully-random uniform. still randomised per sample (N(realistic_mean, prior_std)) '
                   'but preserves the tissue-intensity ordering (WM>GM>CSF) that makes bias identifiable. Tests whether '
                   'the contrast wall is self-inflicted by SynthSeg contrast-agnostic defaults.')
    p.add_argument('--contrast-jitter', type=float, default=1.0, help='[with --realistic-contrast] scale the prior '
                   'STDs (per-sample spread around the realistic means/stds) by this factor: 1.0 = T1 default spread, '
                   '>1 = broader randomisation toward multi-contrast robustness while keeping the tissue ordering -> '
                   'dials randomisation UP without collapsing identifiability (so the final model still trains randomised).')
    p.add_argument('--floor-subtract', action='store_true', help='[needs --null-test] also report score metrics after '
                   'quadrature floor-subtraction sqrt(max(score^2 - floor^2, 0)) using the null-test floor: masked-std '
                   'is a positively-biased statistic, so this removes the estimator-variance floor (fixes the clean-end '
                   'false-positive that a single linear calibration cannot).')
    p.add_argument('--output-shape', type=int, default=160)
    p.add_argument('--n-levels', type=int, default=5)
    p.add_argument('--nb-conv-per-level', type=int, default=2)
    p.add_argument('--conv-size', type=int, default=3)
    p.add_argument('--unet-feat-count', type=int, default=24)
    p.add_argument('--feat-multiplier', type=int, default=2)
    p.add_argument('--activation', type=str, default='elu')
    p.add_argument('--batch-norm', type=int, default=-1, help='conv batch_norm axis (-1 = instance-norm@bs1, DeepN4-style; '
                   'pass 999 for None)')
    # bias / label
    p.add_argument('--bias-field-std', type=float, default=0.5)
    p.add_argument('--bias-scale', type=float, default=0.025)
    p.add_argument('--n-neutral-labels', type=int, default=18)
    return p.parse_args()


def masked_rms(args):
    """RMS of the field over non-background voxels: sqrt(mean(field^2 | mask)). Returns [B, C]."""
    field, labels = args
    mask = K.cast(K.not_equal(labels, 0), field.dtype)
    n = K.sum(mask, axis=[1, 2, 3]) + 1e-8
    ms = K.sum(K.square(field) * mask, axis=[1, 2, 3]) / n
    return K.sqrt(K.maximum(ms, 0.))


def masked_l2_persample(args):
    """Per-voxel L2 between predicted and GT field over the brain mask. Returns [B, 1]."""
    pred, gt, labels = args
    mask = K.cast(K.not_equal(labels, 0), pred.dtype)
    axes = [1, 2, 3, 4]
    n = K.sum(mask, axis=axes) + 1e-8
    se = K.sum(K.square(pred - gt) * mask, axis=axes) / n
    return K.expand_dims(se, -1)


def masked_l2_demeaned(args):
    """Like masked_l2_persample but subtracts each tensor's masked mean first, so only the field shape is penalised
    (the absolute DC offset, unrecoverable from a min-max image, is dropped). The masked-std score is DC-invariant,
    so this trains the score-relevant component. Returns [B, 1]."""
    pred, gt, labels = args
    mask = K.cast(K.not_equal(labels, 0), pred.dtype)
    axes = [1, 2, 3, 4]
    n_kd = K.sum(mask, axis=axes, keepdims=True) + 1e-8          # [B,1,1,1,1]
    pred_dm = pred - K.sum(pred * mask, axis=axes, keepdims=True) / n_kd
    gt_dm = gt - K.sum(gt * mask, axis=axes, keepdims=True) / n_kd
    n = K.sum(mask, axis=axes) + 1e-8                            # [B]
    se = K.sum(K.square(pred_dm - gt_dm) * mask, axis=axes) / n  # [B]
    return K.expand_dims(se, -1)


def _masked_mean(args):
    """Mean of a field over non-background voxels. Returns [B, 1]."""
    field, labels = args
    mask = K.cast(K.not_equal(labels, 0), field.dtype)
    axes = [1, 2, 3, 4]
    n = K.sum(mask, axis=axes) + 1e-8
    return K.expand_dims(K.sum(field * mask, axis=axes) / n, -1)


# Cerebral generation labels = the generation labels SynthSeg actually segments as brain structures
# (generation_labels & synthseg_segmentation_labels, minus background). Everything else in generation_labels is
# extracerebral: the 500-series soft tissue/skull/eyes, general CSF (24), vessels (30/62), optic chiasm (85),
# 5th ventricle (72), 136/137/163/164. Derived from data/labels_classes_priors.
BRAIN_LABELS = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28,
                41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]


def _brain_mask_from_labels(labels):
    """Binary {0,1} mask, 1 where the (integer) label is a cerebral structure. The label map is nearest-neighbour
    resampled so it stays integer-valued, so equality is exact; we cast to float first so the test works whether the
    labels tensor is int32 or float."""
    lab = K.cast(labels, K.floatx())
    acc = None
    for c in BRAIN_LABELS:
        eq = K.cast(K.equal(lab, float(c)), K.floatx())
        acc = eq if acc is None else acc + eq
    return K.clip(acc, 0., 1.)


def make_generator(a, gen_labels, labels_shape, atlas_res, output_div, bias_field_std):
    """Build the labels_to_image_model generator. Factored so the (bias-free) null-test generator is identical to
    the training one except for bias_field_std."""
    return labels_to_image_model(labels_shape=labels_shape, n_channels=1,
                                 generation_labels=gen_labels, output_labels=gen_labels,
                                 n_neutral_labels=a.n_neutral_labels, atlas_res=atlas_res,
                                 target_res=None, output_shape=a.output_shape, output_div_by_n=output_div,
                                 flipping=False, aff=np.eye(4),
                                 scaling_bounds=False, rotation_bounds=False, shearing_bounds=False,
                                 translation_bounds=False, nonlin_std=0, randomise_res=False,
                                 bias_field_std=bias_field_std, bias_scale=a.bias_scale,
                                 intensity_gamma_std=0., return_bias_std=True)


def maybe_log_input(a, generator, name):
    """Optionally feed log(image) to the U-Net (bias additive in log-space). Distinct `name` per call avoids keras
    layer-name collisions when built more than once (training + null-test)."""
    if a.log_input:
        log_img = KL.Lambda(lambda x: K.log(K.maximum(x, 1e-6)), name=name)(generator.outputs[0])
        return models.Model(generator.inputs, log_img)
    return generator


def build_unet_field(a, image_shape, input_model, bn, name):
    """The dense field U-Net (encoder+decoder+skips, linear output). Factored so the null-test U-Net is built with
    the identical architecture, so trained weights transfer by position."""
    return nrn_models.unet(nb_features=a.unet_feat_count, input_shape=image_shape,
                           nb_levels=a.n_levels, conv_size=a.conv_size, nb_labels=1,
                           feat_mult=a.feat_multiplier, nb_conv_per_level=a.nb_conv_per_level,
                           activation=a.activation, batch_norm=bn, use_residuals=True,
                           skip_n_concatenations=a.skip_n_concatenations,
                           final_pred_activation='linear', input_model=input_model, name=name)


def _floor_bins(scores, trues):
    """Print the pred score binned by true std_log, exposes the floor (does a near-zero-bias image still score
    high?) and whether the score saturates. The lowest bin is the in-distribution clean end."""
    edges = [0.0, 0.02, 0.05, 0.10, 0.15, 0.21, np.inf]
    print('    floor check - pred score by TRUE std_log bin:')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (trues >= lo) & (trues < hi)
        if m.sum() == 0:
            continue
        hs = '%.2f' % hi if np.isfinite(hi) else ' inf'
        print('      true[%.2f,%s)  n=%2d  true_mean=%.3f  pred mean/min/max = %.3f / %.3f / %.3f'
              % (lo, hs, int(m.sum()), trues[m].mean(), scores[m].mean(), scores[m].min(), scores[m].max()))


def run_null_test(a, gen_labels, gen_classes, labels_paths, labels_shape, atlas_res, output_div,
                  image_shape, field_model, reduce_fn, bn, contrast_kw):
    """Bias-free probe: a separate generator with bias_field_std=0 (random contrast+anatomy, zero bias) feeding a
    copy of the trained U-Net (weights transferred by position). An honest bias score must read ~0 on these images;
    a non-zero score that correlates with image brightness/contrast means the score reads intensity structure, not bias."""
    # bias_field_std must be >0 (labels_to_image_model gates the bias_* named layers on it, l2i_model.py:198), so use
    # a negligible 1e-6 (true std_log ~1e-6, ~5e4x below the ~0.05 score floor, so effectively bias-free).
    print('\n=== NULL TEST: %d ~BIAS-FREE draws (bias_field_std=1e-6; random contrast+anatomy) ===' % a.null_test)
    null_gen = make_generator(a, gen_labels, labels_shape, atlas_res, output_div, 1e-6)
    null_input = maybe_log_input(a, null_gen, 'log_image_null')
    null_model = build_unet_field(a, image_shape, null_input, bn, 'biasfield_null')
    src_w, dst_w = field_model.get_weights(), null_model.get_weights()
    assert len(src_w) == len(dst_w), 'null-test weight-count mismatch (%d vs %d)' % (len(src_w), len(dst_w))
    null_model.set_weights(src_w)

    null_labels = null_gen.get_layer('bias_mask_labels').output
    null_mask = KL.Lambda(_brain_mask_from_labels, name='brain_mask_null')(null_labels) if a.brain_mask else null_labels
    null_pred = null_model.outputs[0]
    null_score = KL.Lambda(reduce_fn, name='null_score')([null_pred, null_mask])
    img = null_gen.outputs[0]
    img_mean = KL.Lambda(_masked_mean, name='null_img_mean')([img, null_mask])
    img_std = KL.Lambda(reduce_fn, name='null_img_std')([img, null_mask])
    null_probe = models.Model(null_gen.inputs, [null_score, img_mean, img_std])

    src = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels), batchsize=1,
                             n_channels=1, generation_classes=gen_classes, **contrast_kw)
    ns, ims, isd = [], [], []
    for _ in range(a.null_test):
        s, m, sd = null_probe.predict(next(src))
        ns.append(float(s)); ims.append(float(m)); isd.append(float(sd))
    ns, ims, isd = np.array(ns), np.array(ims), np.array(isd)

    def _r(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else float('nan')

    print('    TRUE bias = 0 by construction -> an honest score must read ~0.')
    print('    null pred score  min/mean/max = %.4f / %.4f / %.4f  (std=%.4f)'
          % (ns.min(), ns.mean(), ns.max(), ns.std()))
    print('    corr(null score, masked image MEAN)         = %.3f' % _r(ns, ims))
    print('    corr(null score, masked image STD/contrast) = %.3f' % _r(ns, isd))
    print('    -> non-zero mean and/or high corr with image contrast = the score reads INTENSITY, not bias.')
    floor = float(np.sqrt(np.mean(ns ** 2)))   # RMS of the null score = sqrt(E[Var_error]); used by --floor-subtract
    print('    => floor (RMS of null score) = %.4f  [used by --floor-subtract]' % floor)
    return floor


def report_floor_subtracted(res, floor):
    """Quadrature floor-subtraction. masked-std(b_hat) is a positively-biased estimator: score^2 ~ Var(b_true) +
    Var(error), and the null test measures Var(error)=floor^2 directly (b_true=0 there). So sqrt(max(score^2-floor^2,0))
    is the (approximately) unbiased amplitude estimate; it mainly fixes the clean-end floor/false-positive, which
    the single linear calibration used elsewhere in this script cannot represent (that map is nonlinear)."""
    scores, trues = res['scores'], res['trues']
    corr = np.sqrt(np.maximum(scores ** 2 - floor ** 2, 0.0))
    r = float(np.corrcoef(corr, trues)[0, 1]) if corr.std() > 1e-9 else float('nan')
    mae = float(np.abs(corr - trues).mean())
    h = len(corr) // 2
    mae_oos = float('nan')
    if h >= 3 and corr[:h].std() > 1e-9:
        b, a0 = np.polyfit(corr[:h], trues[:h], 1)
        mae_oos = float(np.abs((b * corr[h:] + a0) - trues[h:]).mean())
    print('\n=== FLOOR-SUBTRACTED SCORE (floor=%.4f; sqrt(max(score^2 - floor^2, 0))) ===' % floor)
    print('    corrected score min/mean/max = %.4f / %.4f / %.4f  (std=%.4f)'
          % (corr.min(), corr.mean(), corr.max(), corr.std()))
    print('    r(corrected,true) = %.3f   MAE = %.4f   (OUT-of-sample lin-calib MAE = %.4f)' % (r, mae, mae_oos))
    _floor_bins(corr, trues)


def probe_report(tag, probe, draw_fn, n_draws):
    """probe outputs [pred_score (physical std/rms of b_hat), true_std_log, field_l2 (raw), field_l2_dm (demeaned/shape)]."""
    scores, trues, l2s, l2dms = [], [], [], []
    for i in range(n_draws):
        s, t, l2, l2dm = probe.predict(draw_fn(i))
        scores.append(float(s)); trues.append(float(t)); l2s.append(float(l2)); l2dms.append(float(l2dm))
    scores, trues, l2s, l2dms = np.array(scores), np.array(trues), np.array(l2s), np.array(l2dms)
    r = float(np.corrcoef(scores, trues)[0, 1]) if scores.std() > 1e-9 else float('nan')
    mae = float(np.abs(scores - trues).mean())
    baseline_l2 = float((trues ** 2).mean())  # predict-zero-field masked L2 ~ RMS^2 ~ std^2 (mean small)
    # linear calibration true ~ b*score + a0, fit on the first half and evaluated on the held-out second half. Report
    # MAE raw, in-sample (optimistic, sees the fit) and out-of-sample. What mae_cal_oos means: the per-draw bias
    # field/std is sampled i.i.d. in-graph every draw, so it is always held out at the bias level; but only under
    # --fresh is it also held out in anatomy+contrast (a genuine new-subject QC-accuracy number). Under frozen the same
    # anatomies (which the encoder also trained on) recur in both halves, so there it is a calibration diagnostic, not QC.
    mae_cal = float('nan')
    mae_cal_oos = float('nan')
    h = n_draws // 2
    if scores.std() > 1e-9 and h >= 3 and scores[:h].std() > 1e-9:
        b, a0 = np.polyfit(scores[:h], trues[:h], 1)
        mae_cal = float(np.abs((b * scores[:h] + a0) - trues[:h]).mean())
        mae_cal_oos = float(np.abs((b * scores[h:] + a0) - trues[h:]).mean())

    print('  [%s] over %d draws:' % (tag, n_draws))
    print('    true  std_log        min/mean/max = %.3f / %.3f / %.3f  (std=%.4f)'
          % (trues.min(), trues.mean(), trues.max(), trues.std()))
    print('    pred  score (of b_hat)   min/mean/max = %.3f / %.3f / %.3f  (std=%.4f)'
          % (scores.min(), scores.mean(), scores.max(), scores.std()))
    print('    score MAE vs true     = %.4f   (lin-calib: in-sample %.4f, OUT-of-sample %.4f)'
          % (mae, mae_cal, mae_cal_oos))
    print('    Pearson r(score,true) = %.3f' % r)
    print('    field masked-L2  mean = %.5f raw / %.5f demeaned(shape)   (predict-zero baseline ~ %.5f)'
          % (l2s.mean(), l2dms.mean(), baseline_l2))
    _floor_bins(scores, trues)
    return {'r': r, 'score_std': float(scores.std()), 'mae': mae, 'mae_cal': mae_cal, 'mae_cal_oos': mae_cal_oos,
            'l2': float(l2s.mean()), 'l2_dm': float(l2dms.mean()), 'baseline_l2': baseline_l2,
            'scores': scores, 'trues': trues}


def save_field_plots(generator, gt_field, pred_field, mask_labels, draw_fn, a):
    """Save [image | GT log-bias | pred log-bias | pred-GT] central-slice PNGs to inspect the shape vs the
    amplitude/shrinkage of the predicted field. GT and pred share the colour scale, so a flatter (paler) pred is
    shrinkage, a structurally-different pattern is a localisation/offset problem."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plotdir = os.path.join(ROOT, a.plot_dir)
    utils.mkdir(plotdir)
    fviz = models.Model(generator.inputs, [generator.outputs[0], gt_field, pred_field, mask_labels])
    for k in range(a.save_fields):
        im, gt, pr, ms = fviz.predict(draw_fn(k))
        im, gt, pr = im[0, ..., 0], gt[0, ..., 0], pr[0, ..., 0]
        ms = ms[0, ..., 0] != 0
        z = im.shape[2] // 2
        m2 = ms[:, :, z]
        gt2 = np.where(m2, gt[:, :, z], np.nan)
        pr2 = np.where(m2, pr[:, :, z], np.nan)
        df2 = np.where(m2, pr[:, :, z] - gt[:, :, z], np.nan)
        vmax = float(np.nanmax(np.abs(gt2))) if np.any(m2) else 1.0
        vmax = max(vmax, 1e-6)
        ts = float(np.std(gt[ms])) if np.any(ms) else 0.0
        ps = float(np.std(pr[ms])) if np.any(ms) else 0.0
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(im[:, :, z].T, cmap='gray', origin='lower'); ax[0].set_title('image')
        ax[1].imshow(gt2.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower'); ax[1].set_title('GT log-bias  std=%.3f' % ts)
        ax[2].imshow(pr2.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower'); ax[2].set_title('pred log-bias  std=%.3f' % ps)
        cb = ax[3].imshow(df2.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower'); ax[3].set_title('pred - GT')
        for a_ in ax:
            a_.axis('off')
        fig.colorbar(cb, ax=list(ax), fraction=0.025)
        fig.savefig(os.path.join(plotdir, 'field_%02d.png' % k), dpi=90, bbox_inches='tight')
        plt.close(fig)
    print('>>> saved %d field plots to %s (GT & pred share the colour scale -> a paler pred = shrinkage)'
          % (a.save_fields, plotdir))


def main():
    a = parse_args()
    print('config:', vars(a))
    if a.fixed_contrast and not a.fresh:
        raise SystemExit('--fixed-contrast requires --fresh (it freezes the GMM contrast while anatomy+bias stay fresh)')
    bn = None if a.batch_norm == 999 else a.batch_norm
    reduce_fn = masked_rms if a.rms_score else _masked_std
    score_name = 'RMS' if a.rms_score else 'std'
    # conv_enc pools nb_levels-1 times; the cp head's Resize is free, so it only needs the encoder's pools to
    # divide cleanly (2^(n_levels-1)), so it keeps 160^3 at n_levels=6, unlike the unet's 2^n_levels convention.
    output_div = 2 ** (a.n_levels - 1) if a.arch == 'cp' else 2 ** a.n_levels

    DATA = os.path.join(ROOT, 'data')
    PRIORS = os.path.join(DATA, 'labels_classes_priors')
    LABELS_DIR = os.path.join(DATA, 'training_label_maps')

    gen_labels = utils.load_array_if_path(os.path.join(PRIORS, 'generation_labels.npy'))
    gen_classes = utils.load_array_if_path(os.path.join(PRIORS, 'generation_classes.npy'))
    labels_paths = utils.list_images_in_folder(LABELS_DIR)
    labels_shape, _, n_dims, _, _, atlas_res = utils.get_volume_info(labels_paths[0], aff_ref=np.eye(4))

    # contrast regime. Default = fully-random uniform GMM (SynthSeg's contrast-agnostic default; the hardest case for
    # bias estimation, no tissue-intensity anchor). --realistic-contrast = randomised-but-structured T1w contrast
    # (normal dist around real tissue means via the contrast-specific classes), which preserves the WM>GM>CSF ordering
    # that makes bias identifiable while still varying per sample (the final model still trains on randomised images,
    # just structured). --contrast-jitter scales the per-sample spread to dial randomisation up without losing ordering.
    if a.realistic_contrast:
        gen_classes = utils.load_array_if_path(os.path.join(PRIORS, 'generation_classes_contrast_specific.npy'))
        pmeans = np.array(utils.load_array_if_path(os.path.join(PRIORS, 'prior_means_t1.npy')), dtype='float64')
        pstds = np.array(utils.load_array_if_path(os.path.join(PRIORS, 'prior_stds_t1.npy')), dtype='float64')
        if a.contrast_jitter != 1.0:
            pmeans[1] *= a.contrast_jitter   # broaden the per-sample spread of the tissue means
            pstds[1] *= a.contrast_jitter    # and of the tissue stds
        contrast_kw = dict(prior_distributions='normal', prior_means=pmeans, prior_stds=pstds)
        print('>>> --realistic-contrast ON: T1w priors, %d contrast-specific classes, normal dist, jitter x%.2f '
              '(still randomised per sample, but structured).' % (len(np.unique(gen_classes)), a.contrast_jitter))
    else:
        contrast_kw = dict(prior_distributions='uniform')

    generator = make_generator(a, gen_labels, labels_shape, atlas_res, output_div, a.bias_field_std)

    image_shape = generator.outputs[0].get_shape().as_list()[1:]   # [160,160,160,1]

    # optionally feed log(image): bias becomes additive in log-space (log preserves the shape, so image_shape stands)
    unet_input_model = maybe_log_input(a, generator, 'log_image')
    if a.log_input:
        print('>>> --log-input ON: the U-Net consumes log(image)')

    # predict the log-bias field b_hat(x), linear output. Two heads selectable via --arch:
    if a.arch == 'cp':
        # guard the cp identities: the image must not have been cropped (else the
        # bottleneck = output_shape/2^(n_levels-1) identity breaks), and the prediction grid R must not be coarser
        # than the generator's GT control grid (ceil(output_shape*bias_scale)), else the head structurally cannot
        # represent the GT field (e.g. n_levels=7 gives R=2 < 4).
        cp_R = a.output_shape // 2 ** (a.n_levels - 1)
        cp_gt_grid = int(np.ceil(a.output_shape * a.bias_scale))
        assert image_shape[:3] == [a.output_shape] * 3, \
            'cp: image is %s, not %d^3 (a crop/resample broke the bottleneck identity)' % (image_shape[:3], a.output_shape)
        assert cp_R >= cp_gt_grid, \
            'cp: prediction grid R=%d is coarser than the GT control grid %d (bias_scale=%g) -> cannot represent the ' \
            'GT field; use fewer levels' % (cp_R, cp_gt_grid, a.bias_scale)
        # control-point head: encoder-only, then 1x1x1 conv = one coefficient per R^3 bottleneck cell
        # (R = output_shape / 2^(n_levels-1)), then trilinear Resize to full res. Same upsample operator the
        # generator uses to build the GT field, so b_hat is band-limited to the generator's smooth subspace
        # by construction (no decoder, no blur, no unstable division).
        enc = nrn_models.conv_enc(nb_features=a.unet_feat_count, input_shape=image_shape,
                                  nb_levels=a.n_levels, conv_size=a.conv_size,
                                  feat_mult=a.feat_multiplier, nb_conv_per_level=a.nb_conv_per_level,
                                  activation=a.activation, batch_norm=bn, use_residuals=True,
                                  input_model=unet_input_model, name='biasfield')
        coeffs = KL.Conv3D(1, 1, padding='same', activation='linear', name='cp_coeffs')(enc.outputs[0])
        pred_field = nrn_layers.Resize(size=image_shape[:3], interp_method='linear', name='cp_upsample')(coeffs)
        print('>>> --arch cp: control-point head, bottleneck R=%d^3 -> trilinear Resize to %d^3'
              % (round(a.output_shape / 2 ** (a.n_levels - 1)), a.output_shape))
    else:
        # full 3D U-Net (encoder+decoder, skip connections) predicting a dense field.
        assert a.skip_n_concatenations < a.n_levels - 1, \
            ('--skip-n-concatenations %d drops ALL skips at n_levels=%d (degenerates to a no-skip encoder-decoder); '
             'use a value < %d' % (a.skip_n_concatenations, a.n_levels, a.n_levels - 1))
        field_model = build_unet_field(a, image_shape, unet_input_model, bn, 'biasfield')
        pred_field = field_model.outputs[0]                        # [B,160,160,160,1]

    gt_field = generator.get_layer('bias_field_log').output
    mask_labels = generator.get_layer('bias_mask_labels').output
    # active mask = whole-head (labels!=0, default) or cerebral-only (--brain-mask). Under --brain-mask the std
    # target is recomputed over the brain too, so target+loss+score all live on the same region (a coherent A/B).
    if a.brain_mask:
        active_mask = KL.Lambda(_brain_mask_from_labels, name='brain_mask')(mask_labels)
        true_std = KL.Lambda(reduce_fn, name='true_std_masked')([gt_field, active_mask])   # GT std/rms over brain [B,1]
        print('>>> --brain-mask ON: score/loss/std-target over %d CEREBRAL labels only (excludes extracerebral '
              'fat/skull/eyes/CSF/vessels).' % len(BRAIN_LABELS))
    else:
        active_mask = mask_labels
        true_std = generator.get_layer('bias_field_std').output    # physical std_log over whole head [B,1]

    # optional smoothness prior: masked Gaussian blur of the predicted field before loss and score. GaussianBlur has
    # no compute_output_shape, so its use_mask=True (2-input) path breaks the keras-2.3 graph build like
    # BiasFieldCorruption did; we instead build a masked normalized convolution from two 1-input blurs + Lambdas
    # (blur(field*mask)/blur(mask), remasked). Not bit-identical to the layer's per-axis-interleaved use_mask path
    # (this is a cleaner single-pass normalized convolution), and it touches no shared layer.
    if a.arch == 'cp' and a.smooth_sigma > 0:
        print('>>> NOTE: --smooth-sigma ignored with --arch cp (the control-point head is already band-limited)')
    if a.smooth_sigma > 0 and a.arch != 'cp':
        sig = [float(a.smooth_sigma)] * 3
        mask_f = KL.Lambda(lambda t: K.cast(K.not_equal(t, 0), K.floatx()), name='smooth_mask')(active_mask)
        field_masked = KL.Multiply(name='smooth_premask')([pred_field, mask_f])
        blur_field = GaussianBlur(sig, name='smooth_blur_field')(field_masked)
        blur_mask = GaussianBlur(sig, name='smooth_blur_mask')(mask_f)
        pred_field = KL.Lambda(lambda x: x[0] / (x[1] + K.epsilon()) * x[2],
                               name='smooth_field')([blur_field, blur_mask, mask_f])
        print('>>> --smooth-sigma %.1f ON: predicted field masked-Gaussian-blurred before loss + score' % a.smooth_sigma)

    # in-graph loss: masked per-voxel L2 (mean over batch); shape-only (demeaned) if --demean-loss
    l2_loss_fn = masked_l2_demeaned if a.demean_loss else masked_l2_persample
    loss = KL.Lambda(lambda x: K.mean(l2_loss_fn(x)), name='qc_loss')([pred_field, gt_field, active_mask])
    loss._keras_shape = tuple(loss.get_shape().as_list())
    qc_model = models.Model(generator.inputs, loss)

    # probe: predicted severity score (reduction of b_hat), the true std_log, and the per-sample field L2
    pred_score = KL.Lambda(reduce_fn, name='biasfield_score')([pred_field, active_mask])
    field_l2 = KL.Lambda(masked_l2_persample, name='biasfield_l2')([pred_field, gt_field, active_mask])
    field_l2_dm = KL.Lambda(masked_l2_demeaned, name='biasfield_l2_dm')([pred_field, gt_field, active_mask])
    probe = models.Model(generator.inputs, [pred_score, true_std, field_l2, field_l2_dm])

    base_gen = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels), batchsize=1,
                                  n_channels=1, generation_classes=gen_classes, **contrast_kw)
    if a.fresh:
        # randomised-contrast regime: a fresh batch of distinct anatomies/contrasts every training step,
        # and a fresh single draw for every probe sample (the real task, not a frozen-anatomy proof).
        train_src = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels), batchsize=a.batch,
                                       n_channels=1, generation_classes=gen_classes, **contrast_kw)
        get_train = lambda step: next(train_src)
        probe_draw = lambda i: next(base_gen)
        print('FRESH: new anatomy+contrast+bias every step; score = masked %s of the predicted field.' % score_name)
        if a.fixed_contrast:
            # draw one GMM contrast (means/stds, shape [1, n_labels, 1]) and reuse it for every draw; the anatomy
            # (label map) and the in-graph bias sigma still vary, so only the contrast is held fixed.
            _, means0, stds0 = next(base_gen)
            def _fix(inp):
                bsz = inp[0].shape[0]
                return [inp[0], np.repeat(means0, bsz, axis=0), np.repeat(stds0, bsz, axis=0)]
            _gt_fn, _pd_fn = get_train, probe_draw
            get_train = lambda step: _fix(_gt_fn(step))
            probe_draw = lambda i: _fix(_pd_fn(i))
            print('>>> --fixed-contrast ON: GMM means/stds frozen; anatomy + in-graph bias still fresh.')
    else:
        frozen_inputs = [next(base_gen) for _ in range(a.k)]
        train_inputs = [[np.repeat(x, a.batch, axis=0) for x in f] for f in frozen_inputs]
        get_train = lambda step: train_inputs[step % len(train_inputs)]
        probe_draw = lambda i: frozen_inputs[i % len(frozen_inputs)]
        print('frozen %d anatomy tuple(s); score = masked %s of the predicted field.' % (a.k, score_name))

    print('\n=== PHASE 0: probe at init (no training) ===')
    probe_report('init', probe, probe_draw, a.probe_n)

    opt = Adam(lr=a.lr, clipnorm=a.clipnorm) if a.clipnorm > 0 else Adam(lr=a.lr)
    qc_model.compile(optimizer=opt, loss=metrics.IdentityLoss().loss)
    dummy = np.zeros((a.batch, 1))
    print('\n=== PHASE 1: overfit the field (%d steps, batch=%d, lr=%.1e, bn=%s, feat=%d, fresh=%s) ==='
          % (a.steps, a.batch, a.lr, bn, a.unet_feat_count, a.fresh))
    run = None
    for step in range(1, a.steps + 1):
        l = float(np.mean(qc_model.train_on_batch(get_train(step), dummy)))
        run = l if run is None else 0.97 * run + 0.03 * l
        if step % a.log_step == 0 or step == 1:
            print('    step %4d   field-L2=%.5f   running=%.5f' % (step, l, run))

    print('\n=== PHASE 2: probe post-training ===')
    res = probe_report('post', probe, probe_draw, a.probe_n)

    print('\n=== VERDICT ===')
    res_l2 = res['l2_dm'] if a.demean_loss else res['l2']   # judge reconstruction on the shape L2 when training demeaned
    if res['r'] > 0.7 and res_l2 < 0.5 * res['baseline_l2']:
        print('FIELD RECOVERED and the %s-score TRACKS the true severity (r=%.2f, L2=%.5f << baseline %.5f).'
              % (score_name, res['r'], res_l2, res['baseline_l2']))
        print('-> the field-estimation head WORKS.')
    elif res['r'] > 0.4:
        print('PROMISING: score tracks partially (r=%.2f), field L2=%.5f vs baseline %.5f. More steps / tune.'
              % (res['r'], res_l2, res['baseline_l2']))
    else:
        print('NOT YET: r=%.2f, field L2=%.5f vs baseline %.5f. Inspect (norm, lr, smoothing, capacity).'
              % (res['r'], res_l2, res['baseline_l2']))

    null_floor = None
    if a.null_test > 0:
        if a.arch != 'unet':
            print('\n>>> --null-test skipped (only implemented for --arch unet)')
        else:
            null_floor = run_null_test(a, gen_labels, gen_classes, labels_paths, labels_shape, atlas_res, output_div,
                                       image_shape, field_model, reduce_fn, bn, contrast_kw)

    if a.floor_subtract:
        if null_floor is None:
            print('\n>>> --floor-subtract needs --null-test (no floor measured); skipped.')
        else:
            report_floor_subtracted(res, null_floor)

    if a.save_fields > 0:
        save_field_plots(generator, gt_field, pred_field, active_mask, probe_draw, a)


if __name__ == '__main__':
    main()
