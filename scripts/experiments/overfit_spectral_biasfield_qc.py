"""

Spectral / band-limited bias-QC head: estimate the smooth log-bias field as a low-dimensional
parametrisation over the whole volume, then compute a severity metric (masked-std) on it.

The head predicts a spatial control grid with a weight-shared 1x1x1 conv on the encoder bottleneck, then
a band-limited trilinear upsample to full res. Spatial correspondence is the point: a global readout
(Flatten, Dense, one output per DCT coefficient) makes each coefficient an independent global regression
with no spatial correspondence, which produces decorrelated, input-independent blobs. The loss is masked
(brain) + demeaned (drop the DC offset, unrecoverable under min-max). The field is still output over the
whole volume incl. background (the grid spans the FOV, the upsample fills everything), so supervision
(masked) and output (whole-volume) are decoupled.

This is the band-limited control-grid approach: the field lives in a small smooth subspace. It is the
frequency-basis cousin of, and near-equivalent to, overfit_field_biasfield_qc.py --arch cp (control
grid + trilinear). The literal cosine/DCT basis (InverseDCT3D, kept below, unused) is a future option,
only worth it to plug in the Hadamard papers' trainable frequency scaling/thresholding + sparsity prior
on the control grid; it needs this same spatial readout, not the global one.

Snapshots at init, 1/4, 1/2, 3/4 and final: [image | GT field | pred field | pred-GT] central-slice
PNGs for a fixed cached set of samples (the head is a standalone image-to-field Model run on a cached
image, so the GT stays fixed and only the prediction sharpens). The field panels span the whole slice
(not masked) to show the field in the background too.

Local CPU smoke (synthqc env):
    python scripts/experiments/overfit_spectral_biasfield_qc.py --output-shape 48 --n-levels 4 --steps 40 \
        --n-coeffs 5 --probe-n 8 --n-viz 2
Real run on a GPU (synthqc env, from the repo root):
    python scripts/experiments/overfit_spectral_biasfield_qc.py --output-shape 160 --n-levels 6 \
        --n-coeffs 5 --steps 400 --fresh

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
import keras.backend as K

from SynthSeg.labels_to_image_model import labels_to_image_model
from SynthSeg.model_inputs import build_model_inputs


def _masked_std(args):
    """std of B(x) over non-background voxels. Returns [batch, 1].

    Was labels_to_image_model._masked_std until the generator was cut back to the
    single _whole_std target and this went with it. It is kept here, local to the per-voxel field
    probes, because their score IS the masked std of the reconstructed field."""
    field, labels = args
    mask = tf.cast(tf.not_equal(labels, 0), field.dtype)
    axes = [1, 2, 3]
    n = tf.reduce_sum(mask, axis=axes) + 1e-8
    mean = tf.reduce_sum(field * mask, axis=axes) / n
    mean2 = tf.reduce_sum(tf.square(field) * mask, axis=axes) / n
    return tf.sqrt(tf.maximum(mean2 - mean * mean, 0.0))
from SynthSeg import metrics_model as metrics
from ext.lab2im import utils
from ext.neuron import models as nrn_models
from ext.neuron import layers as nrn_layers


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4, help='1e-4 matches the working U-Net/cp heads; 3e-4 diverged '
                   '(the band-limited field amplitude blew up).')
    p.add_argument('--clipnorm', type=float, default=1.0, help='gradient-norm clip (0 = off) -- stabilises the bs=1 '
                   'field-amplitude blow-up seen at higher lr.')
    p.add_argument('--fresh', action='store_true', help='RANDOMISED-CONTRAST regime: fresh anatomy+contrast+bias every '
                   'step (the real task) instead of freezing --k anatomy tuples for a quick overfit sanity.')
    p.add_argument('--k', type=int, default=1, help='number of frozen anatomy tuples (ignored under --fresh)')
    p.add_argument('--realistic-contrast', action='store_true', help='draw the GMM contrast from realistic T1w priors '
                   '(structured, WM>GM>CSF ordering preserved) instead of fully-random uniform.')
    # band-limited head
    p.add_argument('--n-coeffs', type=int, default=5, help='control-grid size PER AXIS (the field\'s low-dimensional '
                   'parametrisation). The generator\'s true field grid is ceil(N*bias_scale) (e.g. 4 at 160^3), so a '
                   'value a touch above that gives headroom; too large lets the field absorb anatomy. The encoder '
                   'bottleneck res is R=N/2^(n_levels-1); if n_coeffs != R the predicted grid is resized to n_coeffs.')
    # evolution snapshots
    p.add_argument('--plot-dir', type=str, default='spectral_field_plots', help='dir (under repo root) for PNGs')
    p.add_argument('--n-viz', type=int, default=3, help='fixed samples snapshotted at each evolution checkpoint')
    p.add_argument('--probe-n', type=int, default=64)
    p.add_argument('--log-step', type=int, default=20)
    # encoder architecture
    p.add_argument('--output-shape', type=int, default=96)
    p.add_argument('--n-levels', type=int, default=5, help='encoder pools n_levels-1 times -> bottleneck R = '
                   'output_shape / 2^(n_levels-1); output_shape must be divisible by that.')
    p.add_argument('--nb-conv-per-level', type=int, default=2)
    p.add_argument('--conv-size', type=int, default=3)
    p.add_argument('--feat-count', type=int, default=16)
    p.add_argument('--feat-multiplier', type=int, default=2)
    p.add_argument('--activation', type=str, default='elu')
    p.add_argument('--batch-norm', type=int, default=-1, help='conv batch_norm axis (-1 = instance-norm@bs1; 999 = None)')
    # bias / labels
    p.add_argument('--bias-field-std', type=float, default=0.5)
    p.add_argument('--bias-scale', type=float, default=0.025)
    p.add_argument('--n-neutral-labels', type=int, default=18)
    return p.parse_args()


class InverseDCT3D(KL.Layer):
    """Fixed (non-trainable) inverse 3D DCT-III. Kept for a future cosine-basis upsample (predict a control grid,
    forward-DCT, apply the Hadamard papers' trainable frequency scaling/thresholding + sparsity, then inverse-DCT).
    Not on the current path (the head now uses a trilinear upsample of the control grid, which is proven and matches
    the generator's own field basis). Input [B,m,m,m] gives output [B,N,N,N,1] via the orthonormal DCT-III basis."""

    def __init__(self, out_size, n_coeffs, **kwargs):
        super(InverseDCT3D, self).__init__(**kwargs)
        self.out_size = int(out_size)
        self.n_coeffs = int(n_coeffs)

    def build(self, input_shape):
        N, m = self.out_size, self.n_coeffs
        n = np.arange(N)[:, None]
        k = np.arange(m)[None, :]
        s = np.where(k == 0, np.sqrt(1.0 / N), np.sqrt(2.0 / N))
        basis = (s * np.cos(np.pi * (2 * n + 1) * k / (2.0 * N))).astype('float32')
        self.basis = K.constant(basis)
        super(InverseDCT3D, self).build(input_shape)

    def call(self, c):
        if len(c.shape) == 5:
            c = c[..., 0]
        b = self.basis
        t = tf.einsum('bijk,xi->bxjk', c, b)
        t = tf.einsum('bxjk,yj->bxyk', t, b)
        t = tf.einsum('bxyk,zk->bxyz', t, b)
        return tf.expand_dims(t, -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_size, self.out_size, self.out_size, 1)

    def get_config(self):
        cfg = super(InverseDCT3D, self).get_config()
        cfg.update({'out_size': self.out_size, 'n_coeffs': self.n_coeffs})
        return cfg


def make_generator(a, gen_labels, labels_shape, atlas_res, output_div):
    """Build the labels_to_image_model generator (return_bias_std=True exposes 'bias_field_log'
    and 'bias_field_std' as named layers)."""
    return labels_to_image_model(labels_shape=labels_shape, n_channels=1,
                                 generation_labels=gen_labels, output_labels=gen_labels,
                                 n_neutral_labels=a.n_neutral_labels, atlas_res=atlas_res,
                                 target_res=None, output_shape=a.output_shape, output_div_by_n=output_div,
                                 flipping=False, aff=np.eye(4),
                                 scaling_bounds=False, rotation_bounds=False, shearing_bounds=False,
                                 translation_bounds=False, nonlin_std=0, randomise_res=False,
                                 bias_field_std=a.bias_field_std, bias_scale=a.bias_scale,
                                 intensity_gamma_std=0., return_bias_std=True)


def build_spectral_head(a, image_shape, bn):
    """Standalone head model: image to spatial control grid (weight-shared 1x1x1 conv on the encoder bottleneck),
    then a band-limited trilinear upsample to the full N^3 log-bias field. Returned as its own Model (own image Input)
    so it can be composed onto the generator for training/probing and run alone on a cached image for the evolution
    snapshots (so the GT stays fixed while only the prediction evolves). Spatial correspondence is preserved (a
    bottleneck cell maps to a field location), which a global-coefficient readout does not have;
    the field is band-limited by construction and spans the whole FOV incl. background."""
    enc = nrn_models.conv_enc(nb_features=a.feat_count, input_shape=image_shape, nb_levels=a.n_levels,
                              conv_size=a.conv_size, feat_mult=a.feat_multiplier, nb_conv_per_level=a.nb_conv_per_level,
                              activation=a.activation, batch_norm=bn, use_residuals=True, name='spectral_enc')
    # weight-shared 1x1x1 conv on the bottleneck, one control value per grid cell (spatial correspondence)
    g = KL.Conv3D(1, 1, padding='same', activation='linear', name='ctrl_grid')(enc.outputs[0])   # [B, R, R, R, 1]
    R = int(g.shape[1])
    if a.n_coeffs != R:
        g = nrn_layers.Resize(size=[a.n_coeffs] * 3, interp_method='linear', name='regrid')(g)   # [B, m, m, m, 1]
    field = nrn_layers.Resize(size=image_shape[:3], interp_method='linear', name='grid_upsample')(g)  # [B,N,N,N,1]
    return models.Model(enc.inputs, field, name='spectral_head')


def masked_l2_persample(args):
    """Per-voxel L2 between predicted and GT field over the brain mask (labels!=0). Returns [B,1]."""
    pred, gt, labels = args
    mask = K.cast(K.not_equal(labels, 0), pred.dtype)
    axes = [1, 2, 3, 4]
    n = K.sum(mask, axis=axes) + 1e-8
    se = K.sum(K.square(pred - gt) * mask, axis=axes) / n
    return K.expand_dims(se, -1)


def masked_l2_demeaned(args):
    """Masked L2 after subtracting each field's masked mean (train the field shape only; the DC offset is
    unrecoverable from a min-max image and the masked-std score is DC-invariant). Returns [B,1]."""
    pred, gt, labels = args
    mask = K.cast(K.not_equal(labels, 0), pred.dtype)
    axes = [1, 2, 3, 4]
    n_kd = K.sum(mask, axis=axes, keepdims=True) + 1e-8
    pred_dm = pred - K.sum(pred * mask, axis=axes, keepdims=True) / n_kd
    gt_dm = gt - K.sum(gt * mask, axis=axes, keepdims=True) / n_kd
    n = K.sum(mask, axis=axes) + 1e-8
    se = K.sum(K.square(pred_dm - gt_dm) * mask, axis=axes) / n
    return K.expand_dims(se, -1)


def probe_report(tag, probe, draw_fn, n):
    """probe outputs [pred_score = masked-std of the reconstructed field, true std_log, masked-demeaned field L2]."""
    sc, tr, l2 = [], [], []
    for i in range(n):
        s, t, m = probe.predict(draw_fn(i))
        sc.append(float(s)); tr.append(float(t)); l2.append(float(m))
    sc, tr, l2 = np.array(sc), np.array(tr), np.array(l2)
    r = float(np.corrcoef(sc, tr)[0, 1]) if sc.std() > 1e-9 and tr.std() > 1e-9 else float('nan')
    mae = float(np.abs(sc - tr).mean())
    base = float((tr ** 2).mean())  # predict-zero masked-demeaned L2 = variance of GT over mask = mean(true_std^2)
    # out-of-sample linear-calibration MAE (fit on first half, eval on held-out second half)
    mae_oos = float('nan')
    h = n // 2
    if sc.std() > 1e-9 and h >= 3 and sc[:h].std() > 1e-9:
        b, a0 = np.polyfit(sc[:h], tr[:h], 1)
        mae_oos = float(np.abs((b * sc[h:] + a0) - tr[h:]).mean())
    print('  [%s] over %d draws:' % (tag, n))
    print('    true std_log  min/mean/max = %.3f / %.3f / %.3f  (std=%.4f)'
          % (tr.min(), tr.mean(), tr.max(), tr.std()))
    print('    pred score    min/mean/max = %.3f / %.3f / %.3f  (std=%.4f)'
          % (sc.min(), sc.mean(), sc.max(), sc.std()))
    print('    Pearson r(score,true) = %.3f   score MAE = %.4f   (OUT-of-sample lin-calib MAE = %.4f)'
          % (r, mae, mae_oos))
    print('    field masked-L2 (shape) mean = %.5f   (predict-zero baseline ~ %.5f)' % (l2.mean(), base))
    return {'r': r, 'mae': mae, 'l2': float(l2.mean()), 'baseline': base, 'score_std': float(sc.std())}


def save_snapshot(head_model, viz_samples, tag, plotdir):
    """Save [image | GT field | pred field | pred-GT] central-slice PNGs for the fixed cached viz samples (each a
    tuple (image, GT field, mask) captured once). Only the head prediction changes across snapshots. Field panels span
    the whole slice (not masked) so you can see the field extend into the background; GT & pred share the colour scale,
    so a paler pred = shrinkage and a different pattern = a shape/localisation problem."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for i, (img_np, gt_np, ms_np) in enumerate(viz_samples):
        pr = head_model.predict(img_np)
        im, gt, pr = img_np[0, ..., 0], gt_np[0, ..., 0], pr[0, ..., 0]
        msk = ms_np[0, ..., 0] != 0
        z = im.shape[2] // 2
        vmax = max(float(np.max(np.abs(gt[:, :, z]))), 1e-6)
        ts = float(np.std(gt[msk])) if np.any(msk) else 0.0   # masked std (== the bias_field_std target)
        ps = float(np.std(pr[msk])) if np.any(msk) else 0.0
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(im[:, :, z].T, cmap='gray', origin='lower'); ax[0].set_title('image')
        ax[1].imshow(gt[:, :, z].T, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        ax[1].set_title('GT field  std=%.3f' % ts)
        ax[2].imshow(pr[:, :, z].T, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        ax[2].set_title('pred field  std=%.3f' % ps)
        cb = ax[3].imshow((pr - gt)[:, :, z].T, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        ax[3].set_title('pred - GT')
        for a_ in ax:
            a_.axis('off')
        fig.colorbar(cb, ax=list(ax), fraction=0.025)
        fig.suptitle('%s   sample %d' % (tag, i), y=1.02)
        fig.savefig(os.path.join(plotdir, 'snap_%s_i%d.png' % (tag, i)), dpi=90, bbox_inches='tight')
        plt.close(fig)


def main():
    a = parse_args()
    print('config:', vars(a))
    bn = None if a.batch_norm == 999 else a.batch_norm
    output_div = 2 ** (a.n_levels - 1)   # conv_enc pools n_levels-1 times
    assert a.output_shape % output_div == 0, \
        '--output-shape %d must be divisible by 2^(n_levels-1)=%d' % (a.output_shape, output_div)
    gt_grid = int(np.ceil(a.output_shape * a.bias_scale))
    if a.n_coeffs < gt_grid:
        print('>>> WARNING: --n-coeffs %d < generator GT grid %d (ceil(%d*%.3f)) -> the control grid may be too coarse '
              'to represent the field; consider raising it.' % (a.n_coeffs, gt_grid, a.output_shape, a.bias_scale))

    DATA = os.path.join(ROOT, 'data')
    PRIORS = os.path.join(DATA, 'labels_classes_priors')
    LABELS_DIR = os.path.join(DATA, 'training_label_maps')
    gen_labels = utils.load_array_if_path(os.path.join(PRIORS, 'generation_labels.npy'))
    gen_classes = utils.load_array_if_path(os.path.join(PRIORS, 'generation_classes.npy'))
    labels_paths = utils.list_images_in_folder(LABELS_DIR)
    labels_shape, _, n_dims, _, _, atlas_res = utils.get_volume_info(labels_paths[0], aff_ref=np.eye(4))

    if a.realistic_contrast:
        gen_classes = utils.load_array_if_path(os.path.join(PRIORS, 'generation_classes_contrast_specific.npy'))
        pmeans = np.array(utils.load_array_if_path(os.path.join(PRIORS, 'prior_means_t1.npy')), dtype='float64')
        pstds = np.array(utils.load_array_if_path(os.path.join(PRIORS, 'prior_stds_t1.npy')), dtype='float64')
        contrast_kw = dict(prior_distributions='normal', prior_means=pmeans, prior_stds=pstds)
        print('>>> --realistic-contrast ON: structured T1w priors (WM>GM>CSF ordering), still randomised per sample.')
    else:
        contrast_kw = dict(prior_distributions='uniform')

    generator = make_generator(a, gen_labels, labels_shape, atlas_res, output_div)
    image_shape = generator.outputs[0].get_shape().as_list()[1:]   # [N, N, N, 1]
    print('>>> image shape %s ; encoder bottleneck R=%d^3 ; control grid %d^3 ; GT field grid ~%d^3'
          % (image_shape[:3], a.output_shape // output_div, a.n_coeffs, gt_grid))

    head_model = build_spectral_head(a, image_shape, bn)
    pred_field = head_model(generator.outputs[0])                  # compose on the generator image (shares weights)
    gt_field = generator.get_layer('bias_field_log').output
    # outputs[1] are the labels the generator emits; with output_labels == generation_labels the
    # ConvertLabels map is the identity, so labels != 0 is the same mask the removed
    # 'bias_mask_labels' layer exposed, at the same shape (crop_shape == output_shape here).
    mask_labels = generator.outputs[1]
    true_std = generator.get_layer('bias_field_std').output        # _whole_std of the GT log-field [B,1]

    # in-graph loss: masked (brain) + demeaned (shape only) L2, the recipe the working U-Net uses. The field is still
    # output over the whole volume (grid upsample fills the FOV); only the supervision is masked.
    loss = KL.Lambda(lambda x: K.mean(masked_l2_demeaned(x)), name='qc_loss')([pred_field, gt_field, mask_labels])
    loss._keras_shape = tuple(loss.get_shape().as_list())
    qc_model = models.Model(generator.inputs, loss)

    # probe: masked-std score of the reconstructed field, the true std_log, and the masked-demeaned field L2 (shape)
    pred_score = KL.Lambda(_masked_std, name='spectral_score')([pred_field, mask_labels])
    field_l2 = KL.Lambda(masked_l2_demeaned, name='field_l2_dm')([pred_field, gt_field, mask_labels])
    probe = models.Model(generator.inputs, [pred_score, true_std, field_l2])
    # capture (image, GT field, mask) once per viz sample so the GT stays fixed across evolution snapshots
    capture = models.Model(generator.inputs, [generator.outputs[0], gt_field, mask_labels])

    # data source(s)
    base_gen = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels), batchsize=1,
                                  n_channels=1, generation_classes=gen_classes, **contrast_kw)
    if a.fresh:
        train_src = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels), batchsize=a.batch,
                                       n_channels=1, generation_classes=gen_classes, **contrast_kw)
        get_train = lambda step: next(train_src)
        probe_draw = lambda i: next(base_gen)
        print('FRESH: new anatomy+contrast+bias every step.')
    else:
        frozen_inputs = [next(base_gen) for _ in range(a.k)]
        train_inputs = [[np.repeat(x, a.batch, axis=0) for x in f] for f in frozen_inputs]
        get_train = lambda step: train_inputs[step % len(train_inputs)]
        probe_draw = lambda i: frozen_inputs[i % len(frozen_inputs)]
        print('FROZEN %d anatomy tuple(s).' % a.k)

    # cache fixed viz samples once (image + GT field + mask) so every snapshot shows the same field, only the
    # prediction evolving
    viz_samples = [capture.predict(probe_draw(i)) for i in range(a.n_viz)]
    plotdir = os.path.join(ROOT, a.plot_dir)
    utils.mkdir(plotdir)

    print('\n=== PHASE 0: probe at init (no training) ===')
    probe_report('init', probe, probe_draw, a.probe_n)
    save_snapshot(head_model, viz_samples, 'step0000_init', plotdir)

    opt = Adam(lr=a.lr, clipnorm=a.clipnorm) if a.clipnorm > 0 else Adam(lr=a.lr)
    qc_model.compile(optimizer=opt, loss=metrics.IdentityLoss().loss)
    dummy = np.zeros((a.batch, 1))
    snap_at = sorted(set(int(round(f * a.steps)) for f in (0.25, 0.5, 0.75, 1.0)) - {0})
    print('\n=== PHASE 1: train (%d steps, batch=%d, lr=%.1e, fresh=%s) ; snapshots at steps %s ==='
          % (a.steps, a.batch, a.lr, a.fresh, snap_at))
    run = None
    for step in range(1, a.steps + 1):
        l = float(np.mean(qc_model.train_on_batch(get_train(step), dummy)))
        run = l if run is None else 0.97 * run + 0.03 * l
        if step % a.log_step == 0 or step == 1:
            print('    step %4d   field-L2(shape)=%.5f   running=%.5f' % (step, l, run))
        if step in snap_at:
            save_snapshot(head_model, viz_samples, 'step%04d' % step, plotdir)

    print('\n=== PHASE 2: probe post-training ===')
    res = probe_report('post', probe, probe_draw, a.probe_n)
    print('>>> saved evolution snapshots (init + %s) to %s' % (snap_at, plotdir))

    print('\n=== VERDICT ===')
    if res['r'] > 0.7 and res['l2'] < 0.5 * res['baseline']:
        print('FIELD RECOVERED and the score TRACKS severity (r=%.2f, shape-L2=%.5f << baseline %.5f). The band-limited '
              'control-grid head works.' % (res['r'], res['l2'], res['baseline']))
    elif res['r'] > 0.4:
        print('PROMISING: score tracks partially (r=%.2f, shape-L2=%.5f vs baseline %.5f). More steps / tune.'
              % (res['r'], res['l2'], res['baseline']))
    else:
        print('NOT YET: r=%.2f, shape-L2=%.5f vs baseline %.5f. Inspect (lr, n_coeffs, capacity, more steps).'
              % (res['r'], res['l2'], res['baseline']))


if __name__ == '__main__':
    main()
