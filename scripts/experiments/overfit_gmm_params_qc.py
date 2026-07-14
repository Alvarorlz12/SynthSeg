"""
GMM-parameter regression (the contrast axis) -- simple version.

Question: can we recover, from the generated image, the per-label GMM generation parameters
(the mean and std that SynthSeg draws to paint each anatomical label)?

Two arms:

  blind    (--arm blind):    image -> conv encoder -> dense layers -> predicted (mean, std) for each
                             target label. The network sees only the image; it has to figure out on its
                             own where each tissue is and read its intensity.

  anchored (--arm anchored): image + the true label map -> average the image intensity inside each
                             label's mask -> (mean, std) for each target label. No learned localisation:
                             the label map says where every tissue is, so the parameters are computed
                             directly. Analytic, no training.

The target labels (TARGET_LABELS below) are the structures SynthSeg segments (data/labels_table.txt),
except background; their names are in that file. Edit the list to change what is estimated.

Metrics (both reported, for the means and the stds):
  r  : median per-image Pearson correlation between the estimated and drawn parameters across the target
       labels -- measures whether the estimated profile has the right shape / ranking. Scale-free.
  R2 : variance explained vs a 'predict-prior' baseline (always guess the average), computed on the
       per-image z-scored profile. Unlike r, it penalises getting the values wrong, not just their order,
       so R2 <= 0 means "no better than ignoring the image and predicting the mean". The per-image z-score
       (centre + scale each image's profile) is the one bit of gauge-fixing R2 needs to be comparable
       across images under the min-max normalisation.

Reading the result: the anchored arm is the reference of what is recoverable when the tissue locations
are known; the blind arm is the same question without that knowledge. The gap between them is what the
segmentation buys.

Local CPU smoke (synthqc env, from repo root SynthQC), use >=96^3 so the crop contains a brain:
    python scripts/experiments/overfit_gmm_params_qc.py --arm anchored --output-shape 96 --probe-n 24
    python scripts/experiments/overfit_gmm_params_qc.py --arm blind --output-shape 96 --steps 200 --probe-n 24
Real training is GPU/cluster (larger --output-shape and --steps).

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
import tensorflow as tf
from keras import models
from keras.optimizers import Adam
import keras.layers as KL

from SynthSeg.labels_to_image_model import labels_to_image_model
from SynthSeg.model_inputs import build_model_inputs
from ext.lab2im import utils
from ext.neuron import models as nrn_models


# The structures SynthSeg segments (data/labels_table.txt), except background. Names are in that file.
TARGET_LABELS = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 24, 28,
                 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]
MIN_VOX = 10   # a target label must have at least this many voxels in the cropped image to be measured


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['blind', 'anchored'], default='blind',
                   help='blind = image only (encoder + dense); anchored = image + true label map (analytic).')
    p.add_argument('--regime', choices=['clean', 'full'], default='clean',
                   help='clean = no artefacts (pure GMM contrast); full = bias field + gamma + random resolution.')
    p.add_argument('--predict', choices=['means', 'both'], default='both',
                   help='estimate the means only, or the means and the stds.')
    # training (blind arm only)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--probe-n', type=int, default=48)
    p.add_argument('--log-step', type=int, default=20)
    # encoder / head
    p.add_argument('--hidden', type=int, default=128, help='units in the dense head.')
    p.add_argument('--output-shape', type=int, default=160)
    p.add_argument('--n-levels', type=int, default=5)
    p.add_argument('--nb-conv-per-level', type=int, default=2)
    p.add_argument('--conv-size', type=int, default=3)
    p.add_argument('--feat-count', type=int, default=24, help='encoder feature count at the first level.')
    p.add_argument('--feat-multiplier', type=int, default=2)
    p.add_argument('--activation', type=str, default='elu')
    p.add_argument('--n-neutral-labels', type=int, default=18)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def make_generator(a, gen_labels, labels_shape, atlas_res, output_div):
    """The SynthSeg generator: paints the label map with a random GMM and (optionally) corrupts it. Outputs
    [normalised image, label map]. flipping is off (it would swap the left/right target labels)."""
    full = a.regime == 'full'
    return labels_to_image_model(
        labels_shape=labels_shape, n_channels=1,
        generation_labels=gen_labels, output_labels=gen_labels, # keep label values on outputs[1] for the masks
        n_neutral_labels=a.n_neutral_labels, atlas_res=atlas_res,
        target_res=None, output_shape=a.output_shape, output_div_by_n=output_div,
        flipping=False, aff=np.eye(4),
        scaling_bounds=False, rotation_bounds=False, shearing_bounds=False, translation_bounds=False,
        nonlin_std=0.0, nonlin_scale=0.0625,
        randomise_res=full, max_res_iso=4.0, max_res_aniso=8.0,
        bias_field_std=(0.5 if full else 0.0),
        intensity_gamma_std=(0.5 if full else 0.0),
        return_bias_std=False, return_resolution=False)


def build_blind(a, generator, n_out, image_shape):
    """The blind arm: a convolutional encoder on the image, then dense layers, then n_out linear outputs
    (the estimated means, then the estimated stds). It only reads generator.outputs[0] = the image."""
    enc = nrn_models.conv_enc(input_model=generator, input_shape=image_shape, nb_levels=a.n_levels,
                              conv_size=a.conv_size, nb_features=a.feat_count, feat_mult=a.feat_multiplier,
                              nb_conv_per_level=a.nb_conv_per_level, activation=a.activation,
                              batch_norm=None, name='gmm_enc')
    feat = enc.outputs[0]
    x = KL.GlobalAveragePooling3D(name='gmm_gap')(feat)
    x = KL.Dense(a.hidden, activation=a.activation, name='gmm_dense1')(x)
    x = KL.Dense(a.hidden, activation=a.activation, name='gmm_dense2')(x)
    out = KL.Dense(n_out, activation=None, name='gmm_out')(x)   # [B, n_out] = [means... | stds...]
    return models.Model(generator.inputs, out)


def drawn_params(batch, idxs):
    """Ground truth: the per-label means and stds that were fed to the generator for this batch."""
    means = np.asarray(batch[1])[0, idxs, 0]    # batch[1] = means, shape [B, n_labels, 1]
    stds = np.asarray(batch[2])[0, idxs, 0]     # batch[2] = stds
    return means, stds


def measure_anchored(generator, batch, target_labels):
    """Anchored arm: run the generator, then for each target label average the image intensity inside the
    label's mask in the output label map (aligned with the output image). Absent labels -> NaN."""
    image, labelmap = generator.predict_on_batch(batch)
    image = np.asarray(image)[0, ..., 0]
    labelmap = np.asarray(labelmap)[0, ..., 0]
    means, stds = [], []
    for lab in target_labels:
        mask = labelmap == lab
        if int(mask.sum()) >= MIN_VOX:
            vals = image[mask]
            means.append(float(vals.mean()))
            stds.append(float(vals.std()))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    return np.asarray(means), np.asarray(stds)


def pearson(x, y):
    """Pearson r over the finite pairs; NaN if fewer than 3 usable points or no variance."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def zscore_rows(matrix):
    """z-score each row over its finite entries (per-image gauge-fixing): subtract the row mean and divide
    by the row std, so profiles from different images become comparable despite the min-max scale. Rows
    with fewer than 3 finite entries or no spread are left as NaN."""
    z = np.full(matrix.shape, np.nan, dtype=float)
    for i in range(matrix.shape[0]):
        ok = np.isfinite(matrix[i])
        if int(ok.sum()) >= 3:
            v = matrix[i, ok]
            sd = v.std()
            if sd > 1e-9:
                z[i, ok] = (v - v.mean()) / sd
    return z


def r2_vs_prior(y_true, y_pred):
    """R2 = 1 - SS_res / SS_tot, with SS_tot the 'predict-prior' baseline (always guess the mean), on the
    flattened finite pairs. R2 > 0 beats that baseline; R2 = 0 ties it; R2 < 0 is worse than it."""
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[ok], y_pred[ok]
    if len(yt) < 3:
        return float('nan')
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def mse(y_true, y_pred):
    """Mean squared distance over the finite pairs (NaN if none). On the z-scored profile it is the gauge-
    fixed distance between the estimated and true solutions; on an already-comparable target (e.g. the
    effective normalised tissue means in [0, 1]) call it directly on the raw values."""
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if not ok.any():
        return float('nan')
    return float(np.mean((y_true[ok] - y_pred[ok]) ** 2))


def probe(tag, a, generator, blind_model, src, idxs, n, predict_both):
    """Collect the estimated and drawn parameters over n images, then report, per parameter type:
      r   = median per-image Pearson correlation across the target labels (shape / ranking);
      R2  = variance explained vs the predict-prior baseline, on the per-image z-scored profile (so it
            also penalises wrong values, not just wrong order);
      MSE = mean squared distance on the same z-scored profile (the distance between the
            estimated and true solutions; ~ 1 - R2)."""
    n_lab = len(idxs)
    est_m, gt_m, est_s, gt_s = [], [], [], []
    for _ in range(n):
        batch = next(src)
        gtm, gts = drawn_params(batch, idxs)
        if a.arm == 'blind':
            pred = np.asarray(blind_model.predict_on_batch(batch))[0]
            estm = pred[:n_lab]
            ests = pred[n_lab:2 * n_lab] if predict_both else np.full(n_lab, np.nan)
        else:
            estm, ests = measure_anchored(generator, batch, TARGET_LABELS)
        est_m.append(estm); gt_m.append(gtm); est_s.append(ests); gt_s.append(gts)
    est_m = np.asarray(est_m, float); gt_m = np.asarray(gt_m, float)
    est_s = np.asarray(est_s, float); gt_s = np.asarray(gt_s, float)

    zt_m, ze_m = zscore_rows(gt_m).flatten(), zscore_rows(est_m).flatten()
    r_means = np.array([pearson(est_m[i], gt_m[i]) for i in range(n)])
    r2_means = r2_vs_prior(zt_m, ze_m)
    mse_means = mse(zt_m, ze_m)
    print('  [%s] over %d images (%d target labels):' % (tag, n, n_lab))
    print('    MEANS  r(median)=%.3f   R2(vs prior)=%.3f   MSE(z)=%.3f' % (np.nanmedian(r_means), r2_means, mse_means))
    if predict_both:
        zt_s, ze_s = zscore_rows(gt_s).flatten(), zscore_rows(est_s).flatten()
        r_stds = np.array([pearson(est_s[i], gt_s[i]) for i in range(n)])
        r2_stds = r2_vs_prior(zt_s, ze_s)
        mse_stds = mse(zt_s, ze_s)
        print('    STDS   r(median)=%.3f   R2(vs prior)=%.3f   MSE(z)=%.3f' % (np.nanmedian(r_stds), r2_stds, mse_stds))
        return float(np.nanmedian(r_means)), r2_means, mse_means, float(np.nanmedian(r_stds)), r2_stds, mse_stds
    return float(np.nanmedian(r_means)), r2_means, mse_means, float('nan'), float('nan'), float('nan')


def main():
    a = parse_args()
    np.random.seed(a.seed)
    tf.random.set_seed(a.seed)
    print('config:', vars(a))
    predict_both = a.predict == 'both'
    output_div = 2 ** a.n_levels

    DATA = os.path.join(ROOT, 'data')
    PRIORS = os.path.join(DATA, 'labels_classes_priors')
    LABELS_DIR = os.path.join(DATA, 'training_label_maps')

    gen_labels = np.asarray(utils.load_array_if_path(os.path.join(PRIORS, 'generation_labels.npy'))).astype('int32')
    gen_classes = np.asarray(utils.load_array_if_path(os.path.join(PRIORS, 'generation_classes.npy'))).astype('int32')
    labels_paths = utils.list_images_in_folder(LABELS_DIR)
    labels_shape, _, n_dims, _, _, atlas_res = utils.get_volume_info(labels_paths[0], aff_ref=np.eye(4))

    # index of each target label inside the generation-label vector (so we can read its drawn mean/std)
    label_to_idx = {int(l): i for i, l in enumerate(gen_labels)}
    missing = [l for l in TARGET_LABELS if l not in label_to_idx]
    if missing:
        raise ValueError('TARGET_LABELS not in generation_labels: %s' % missing)
    idxs = [label_to_idx[l] for l in TARGET_LABELS]
    n_lab = len(idxs)
    n_out = 2 * n_lab if predict_both else n_lab
    print('  target labels: %d  (%s)' % (n_lab, TARGET_LABELS))

    generator = make_generator(a, gen_labels, labels_shape, atlas_res, output_div)
    image_shape = generator.outputs[0].get_shape().as_list()[1:]

    src = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels), batchsize=a.batch,
                             n_channels=1, generation_classes=gen_classes, prior_distributions='uniform')

    blind_model = None
    if a.arm == 'blind':
        blind_model = build_blind(a, generator, n_out, image_shape)
        n_trainable = int(np.sum([np.prod(w.shape.as_list()) for w in blind_model.trainable_weights]))
        print('  blind head trainable params: %d' % n_trainable)

        print('\n=== probe at init (no training) ===')
        probe('init', a, generator, blind_model, src, idxs, a.probe_n, predict_both)

        blind_model.compile(optimizer=Adam(lr=a.lr), loss='mse')
        print('\n=== train (%d steps, regime=%s, predict=%s) ===' % (a.steps, a.regime, a.predict))
        run = None
        for step in range(1, a.steps + 1):
            batch = next(src)
            gt_means, gt_stds = drawn_params(batch, idxs)
            target = np.concatenate([gt_means, gt_stds]) if predict_both else gt_means
            loss = float(blind_model.train_on_batch(batch, target[np.newaxis, :]))
            run = loss if run is None else 0.97 * run + 0.03 * loss
            if step % a.log_step == 0 or step == 1:
                print('    step %4d   loss=%.4f   running=%.4f' % (step, loss, run))

        print('\n=== probe after training ===')
        rm, r2m, msem, rs, r2s, mses = probe('post', a, generator, blind_model, src, idxs, a.probe_n, predict_both)
    else:
        print('\n=== anchored (analytic, no training) ===')
        rm, r2m, msem, rs, r2s, mses = probe('anchored', a, generator, None, src, idxs, a.probe_n, predict_both)

    print('\n=== SUMMARY (arm=%s, regime=%s) ===' % (a.arm, a.regime))
    print('  means:  r(median)=%.3f   R2(vs prior)=%.3f   MSE(z)=%.3f' % (rm, r2m, msem))
    if predict_both:
        print('  stds:   r(median)=%.3f   R2(vs prior)=%.3f   MSE(z)=%.3f' % (rs, r2s, mses))


if __name__ == '__main__':
    main()
