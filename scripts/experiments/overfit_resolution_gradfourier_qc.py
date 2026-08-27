"""
Resolution QC, blind arm: regress per-axis effective resolution (mm) from the image plus hand-computed
feature channels (derivative-of-Gaussian gradients and directional Fourier high-pass), fed to a 3D CNN.
This is the blind control for the segmentation-anchored arm: image only, no anatomical anchor. It is run
with the honest metric below (per-axis log-MAE by stratum plus the signed coarsest-axis match), because a
blind spectral head can score well on synthetic and still invert on real.

Why it's expected to fail on real: a blur along an axis is spectrally indistinguishable from
low-bandwidth anatomy along that axis. Synthetic i.i.d. GMM content is spectrally white/isotropic, so
the CNN can learn "smoother axis = lower res" and score well there. Real brains have
genuine anatomical anisotropy, so that rule inverts.
Fourier features are the most exposed to this (they are the marginal spectrum); we include them to
give the blind route its best shot.

The honest metric: report per-axis log-MAE and mm-MAE by stratum, plus the signed coarsest-axis match
(which axis is lowest-res, does it get it right or invert?), restricted to anisotropic draws with a
unique coarsest axis. Pooled Pearson r overstates (bimodal label ~62% at 1mm). Plain MSE is sign-blind
and would hide the inversion; the argmax-match is the diagnostic that exposed it.

Local CPU build-smoke (synthqc env, from repo root SynthQC):
    python scripts/experiments/overfit_resolution_gradfourier_qc.py --output-shape 32 --n-levels 2 --steps 4 --probe-n 24
Real training is GPU/cluster (full 160^3):
    srun --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=02:00:00 --pty bash -lc \
      'module load miniforge; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate synthqc; \
       cd ~/SynthQC; python -u scripts/experiments/overfit_resolution_gradfourier_qc.py --steps 2000 | tee gradfft_run.log'
Real eval = ADNI (native ~0.94x0.94x1.25 + MimicAcquisition-injected known degradation + FLAIR) via the existing gate.

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
from SynthSeg import metrics_model as metrics
from ext.lab2im import utils
from ext.neuron import models as nrn_models

EPS = 1e-8


def parse_args():
    p = argparse.ArgumentParser()
    # training
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--clipnorm', type=float, default=1.0)
    p.add_argument('--probe-n', type=int, default=96)
    p.add_argument('--log-step', type=int, default=20)
    p.add_argument('--huber-delta', type=float, default=0.1, help='Huber transition on the normalized log-spacing.')
    p.add_argument('--degraded-weight', type=float, default=4.0,
                   help='up-weight degraded (high-spacing) axes by (1+this*y) to fight the ~62%% 1mm point-mass shrinkage.')
    # resolution sampling
    p.add_argument('--res-mode', choices=['default', 'all-axes'], default='default',
                   help="default = SynthSeg's single-axis-aniso + ~62%% 1mm point mass (matches deployment/ADNI). "
                        "all-axes = sample ALL THREE axes INDEPENDENTLY in U(atlas, --res-max), "
                        "killing the 1mm point mass (a diagnostic: does it move the coarsest-axis match above chance?).")
    p.add_argument('--res-max', type=float, default=9.0,
                   help='upper bound (mm) of the per-axis draw in --res-mode all-axes. Floor is the atlas res (1mm); '
                        'sub-1mm is unreachable (would invent detail the atlas lacks).')
    p.add_argument('--max-res-iso', type=float, default=4.0, help='(default mode) isotropic-branch upper bound.')
    p.add_argument('--max-res-aniso', type=float, default=8.0, help='(default mode) single-axis-branch upper bound.')
    p.add_argument('--grid-ablation', choices=['none', 'blur_only', 'kernel_random', 'kernel_phase'], default='none',
                   help='grid-cheat ablation; kernel_phase strips the SynthSeg linear-up watermark (sim->real check).')
    # feature channels
    p.add_argument('--use-gradient', type=int, default=1, help='append derivative-of-Gaussian gradient channels.')
    p.add_argument('--use-fourier', type=int, default=1, help='append directional Fourier high-pass channels.')
    p.add_argument('--dog-sigmas', type=str, default='1,2,4', help='comma-separated derivative-of-Gaussian scales (vox).')
    p.add_argument('--fft-cutoff', type=float, default=0.5, help='fraction of Nyquist above which a directional band is kept.')
    p.add_argument('--use-image', type=int, default=1, help='also feed the raw image channel to the CNN.')
    # encoder / head
    p.add_argument('--head', choices=['cnn', 'per-axis'], default='per-axis',
                   help="cnn = the literal 'channels -> conv encoder -> GAP -> Dense(3)' form (the global pool is "
                        "axis-invariant and COLLAPSES to the mode -- kept as a control). per-axis = collapse-resistant: "
                        "reduce the DoG/Fourier channels to per-axis SCALAR energies (raw-log + cross-axis-normalized) "
                        "-> a WEIGHT-SHARED per-axis head (the same recipe as --no-context, with the advanced features).")
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--output-shape', type=int, default=160)
    p.add_argument('--n-levels', type=int, default=5)
    p.add_argument('--nb-conv-per-level', type=int, default=2)
    p.add_argument('--conv-size', type=int, default=3)
    p.add_argument('--unet-feat-count', type=int, default=24)
    p.add_argument('--feat-multiplier', type=int, default=2)
    p.add_argument('--activation', type=str, default='elu')
    p.add_argument('--n-neutral-labels', type=int, default=18)
    return p.parse_args()


# gradient channels: derivative-of-Gaussian (scale-aware, not raw finite differences)
def _dog_kernel_1d(sigma):
    """1-D derivative-of-Gaussian, zero-sum, L1-normalized; a scale-aware directional derivative."""
    r = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype='float32')
    g = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    dg = -(x / sigma ** 2) * g
    dg = dg - dg.mean()
    dg = dg / (np.abs(dg).sum() + 1e-8)
    return dg.astype('float32')


def build_gradient_channels(image, sigmas, n_dims=3):
    """Per-axis, per-scale derivative-of-Gaussian responses of the image, as separate channels [B,X,Y,Z,1] each.
    A low-res axis is smoother along that direction, so weaker DoG response, at every scale."""
    chans = []
    for sigma in sigmas:
        k1d = _dog_kernel_1d(sigma)
        L = len(k1d)
        for a in range(n_dims):
            fshape = [1, 1, 1, 1, 1]
            fshape[a] = L
            filt = tf.constant(k1d.reshape(fshape))                      # conv3d filter [D,H,W,in=1,out=1]
            fn = (lambda ft: (lambda t: tf.nn.conv3d(t, ft, strides=[1, 1, 1, 1, 1], padding='SAME')))(filt)
            chans.append(KL.Lambda(fn, name='dog_s%g_a%d' % (sigma, a))(image))
    return chans


# fourier channels: directional high-pass via a single 3D FFT (spectral, not the v4 roll-off scalar)
def build_fourier_channels(image, N, n_dims=3, cutoff=0.5):
    """For each axis a, keep only spectral energy with |k_a| > cutoff*Nyquist, inverse FFT, magnitude map =
    how much fine detail exists along axis a. A low-res axis has none. This is the marginal spectrum, the
    feature most exposed to the resolution-vs-anatomy identifiability wall on real data."""
    freqs = np.fft.fftfreq(N).astype('float32')                          # matches tf.signal.fft3d ordering
    masks = []
    for a in range(n_dims):
        m1d = (np.abs(freqs) > cutoff * 0.5).astype('float32')
        shp = [1, 1, 1]
        shp[a] = N
        masks.append(np.ones((N, N, N), 'float32') * m1d.reshape(shp))
    masks_t = tf.constant(np.stack(masks, 0))                            # [n_dims, N, N, N]

    def fn(t):
        x = tf.cast(t[..., 0], tf.complex64)                            # [B,N,N,N]
        X = tf.signal.fft3d(x)
        outs = []
        for a in range(n_dims):
            xf = tf.signal.ifft3d(X * tf.cast(masks_t[a], tf.complex64))
            outs.append(tf.abs(xf))
        return tf.stack(outs, axis=-1)                                  # [B,N,N,N,n_dims]
    return KL.Lambda(fn, name='fft_highpass')(image)


# model
def build_feature_model(a, generator, n_dims):
    """Concatenate [image | DoG gradient channels | Fourier high-pass channels] into a multi-channel feature volume."""
    image = generator.outputs[0]                                        # [B,X,Y,Z,1]
    N = image.get_shape().as_list()[1]
    sigmas = [float(s) for s in a.dog_sigmas.split(',') if s.strip()]
    chans = []
    if a.use_image:
        chans.append(image)
    if a.use_gradient:
        chans += build_gradient_channels(image, sigmas, n_dims)
    if a.use_fourier:
        chans.append(build_fourier_channels(image, N, n_dims, a.fft_cutoff))
    assert chans, 'no input channels selected (need at least one of --use-image/--use-gradient/--use-fourier).'
    feat = KL.Concatenate(axis=-1, name='feat_concat')(chans) if len(chans) > 1 else chans[0]
    return models.Model(generator.inputs, feat)


# collapse-resistant per-axis head: reduce features to per-axis scalar energies, weight-shared head
def build_per_axis_features(a, image, sigmas, n_dims, N):
    """Per-axis scalar energies of the features, gives [B, n_dims, F]. Preserves axis identity (unlike the
    conv-encoder+GAP path, which mixes and globally-pools it away and collapses to the mode)."""
    feats = []                                                          # each entry: [B, n_dims]
    if a.use_gradient:
        for sigma in sigmas:
            k1d = _dog_kernel_1d(sigma)
            L = len(k1d)
            per_axis = []
            for ax in range(n_dims):
                fshape = [1, 1, 1, 1, 1]
                fshape[ax] = L
                filt = tf.constant(k1d.reshape(fshape))
                fn = (lambda ft: (lambda t: K.mean(K.square(tf.nn.conv3d(t, ft, [1, 1, 1, 1, 1], 'SAME')),
                                                   axis=[1, 2, 3, 4])))(filt)
                per_axis.append(KL.Lambda(fn, name='paE_dog_s%g_a%d' % (sigma, ax))(image))     # [B]
            feats.append(KL.Lambda(lambda xs: K.stack(xs, axis=1), name='paE_dog_s%g' % sigma)(per_axis))  # [B,nd]
    if a.use_fourier:
        freqs = np.fft.fftfreq(N).astype('float32')
        masks = []
        for ax in range(n_dims):
            m1d = (np.abs(freqs) > a.fft_cutoff * 0.5).astype('float32')
            shp = [1, 1, 1]
            shp[ax] = N
            masks.append(np.ones((N, N, N), 'float32') * m1d.reshape(shp))
        masks_t = tf.constant(np.stack(masks, 0))

        def fft_energy(t):
            x = tf.cast(t[..., 0], tf.complex64)
            X = tf.signal.fft3d(x)
            es = []
            for ax in range(n_dims):
                xf = tf.abs(tf.signal.ifft3d(X * tf.cast(masks_t[ax], tf.complex64)))
                es.append(K.mean(K.square(xf), axis=[1, 2, 3]))         # [B]
            return K.stack(es, axis=1)                                  # [B, nd]
        feats.append(KL.Lambda(fft_energy, name='paE_fft')(image))
    assert feats, 'per-axis head needs at least one of --use-gradient/--use-fourier.'
    if len(feats) == 1:
        return KL.Lambda(lambda t: K.expand_dims(t, -1), name='paE_stack')(feats[0])
    return KL.Lambda(lambda xs: K.stack(xs, axis=-1), name='paE_stack')(feats)  # [B, nd, F]


def build_per_axis_head(a, generator, n_dims):
    """DoG/Fourier per-axis energies, then [raw-log | cross-axis-normalized], then a weight-shared Dense head that
    maps each axis's descriptor to its log-spacing. The shared head forces the model to read each axis
    (collapse-resistant). The cross-axis normalization is the feature that carries the synthetic signal and also the
    one exposed to the resolution-vs-anatomy ambiguity on real data; it is kept so the blind arm runs at its best."""
    image = generator.outputs[0]
    N = image.get_shape().as_list()[1]
    sigmas = [float(s) for s in a.dog_sigmas.split(',') if s.strip()]
    raw = build_per_axis_features(a, image, sigmas, n_dims, N)          # [B, nd, F]
    logr = KL.Lambda(lambda t: K.log(t + EPS), name='pa_logr')(raw)                          # absolute sharpness
    crossn = KL.Lambda(lambda t: t / (K.mean(t, axis=1, keepdims=True) + EPS), name='pa_crossn')(raw)  # relative
    feat = KL.Concatenate(axis=-1, name='pa_feat')([logr, crossn])     # [B, nd, 2F]
    h = KL.Dense(a.hidden, activation=a.activation, name='pa_h')(feat)  # weight-shared over the nd axis
    out = KL.Dense(1, activation='sigmoid', name='pa_out')(h)          # [B, nd, 1]
    pred = KL.Lambda(lambda t: t[..., 0], name='pa_pred')(out)         # [B, nd]
    n_feat = raw.get_shape().as_list()[-1]
    print('  per-axis head: %d features/axis (raw-log + cross-axis-norm) -> weight-shared Dense' % (2 * n_feat))
    return models.Model(generator.inputs, pred)


def build_regressor(a, generator, n_dims):
    if a.head == 'per-axis':
        return build_per_axis_head(a, generator, n_dims)
    # cnn head: the literal 'channels, conv encoder, GAP, Dense(3)' form. The global pool is axis-invariant
    # so collapse-prone (predicts the mode). Kept as a control. batch_norm=None (BN at bs=1 erases the
    # spatial-amplitude statistic the resolution cue lives in).
    feat_model = build_feature_model(a, generator, n_dims)
    feat_shape = feat_model.outputs[0].get_shape().as_list()[1:]
    n_ch = feat_shape[-1]
    enc = nrn_models.conv_enc(input_model=feat_model, input_shape=feat_shape,
                              nb_levels=a.n_levels, conv_size=a.conv_size,
                              nb_features=a.unet_feat_count, feat_mult=a.feat_multiplier,
                              nb_conv_per_level=a.nb_conv_per_level, activation=a.activation,
                              batch_norm=None, use_residuals=True, name='resgf_enc')
    f = enc.outputs[0]
    gmean = KL.GlobalAveragePooling3D(name='resgf_gmean')(f)
    gstd = KL.Lambda(lambda t: K.std(t, axis=[1, 2, 3]), name='resgf_gstd')(f)
    ctx = KL.Concatenate(name='resgf_ctx')([gmean, gstd])
    h = KL.Dense(a.hidden, activation=a.activation, name='resgf_h')(ctx)
    out = KL.Dense(n_dims, activation='sigmoid', name='resgf_out')(h)   # [B, n_dims] normalized log-spacing in (0,1)
    model = models.Model(generator.inputs, out)
    print('  input channels fed to the CNN: %d' % n_ch)
    return model


def huber_elt(y_true, y_pred, delta):
    e = K.abs(y_true - y_pred)
    quad = K.minimum(e, delta)
    return 0.5 * K.square(quad) + delta * (e - quad)


def build_loss(generator, regression_model, log_min, log_span, huber_delta, degraded_weight):
    s = generator.get_layer('resolution').output                       # [B, n_dims] mm
    y = KL.Lambda(lambda x: K.clip((K.log(x) - log_min) / log_span, 0., 1.), name='resgf_label')(s)
    pred = regression_model.outputs[0]

    def _wloss(args):
        yt, yp = args
        elt = huber_elt(yt, yp, huber_delta)
        w = 1.0 + degraded_weight * yt
        return K.sum(w * elt) / (K.sum(w) + K.epsilon())
    loss = KL.Lambda(_wloss, name='qc_loss')([y, pred])
    loss._keras_shape = tuple(loss.get_shape().as_list())
    return models.Model(generator.inputs, loss)


# probe / honest metrics
def _pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else float('nan')


def probe_report(tag, probe, draw_fn, n, log_min, log_span, max_res_iso, min_res=1.0):
    preds, trues = [], []
    for i in range(n):
        p, s = probe.predict(draw_fn(i))
        preds.append(np.asarray(p)[0]); trues.append(np.asarray(s)[0])
    preds = np.array(preds); trues = np.array(trues)                    # [N, nd]
    nd = trues.shape[1]
    s_pred = np.exp(preds * log_span + log_min)                        # [N, nd] mm

    mae_mm = float(np.abs(s_pred - trues).mean())
    mae_log = float(np.abs(np.log(s_pred) - np.log(trues)).mean())
    r_axes = [_pearson(s_pred[:, a], trues[:, a]) for a in range(nd)]
    r_pool = _pearson(s_pred.flatten(), trues.flatten())
    mae_base_min = float(np.abs(min_res - trues).mean())
    mae_base_mean = float(np.abs(trues.mean() - trues).mean())

    # per-axis mm-MAE + per-axis log-MAE
    mae_ax_mm = [float(np.abs(s_pred[:, a] - trues[:, a]).mean()) for a in range(nd)]

    # strata
    mn, mx = trues.min(axis=1), trues.max(axis=1)
    native = mx <= min_res + 1e-3
    aniso = (~native) & ((mx - mn) > 1e-3)
    ax_native = trues <= min_res + 1e-3
    ax_degr = ~ax_native
    def _mae(mask):
        return float(np.abs(s_pred[mask] - trues[mask]).mean()) if mask.any() else float('nan')
    def _logmae(mask):
        return float(np.abs(np.log(s_pred[mask]) - np.log(trues[mask])).mean()) if mask.any() else float('nan')

    # the honest metric: signed coarsest-axis match on aniso draws with a unique coarsest axis.
    coarse_ok = fine_ok = 0
    coarse_tot = 0
    inv = 0
    for i in range(len(trues)):
        if not aniso[i]:
            continue
        t = trues[i]
        # unique coarsest?
        order = np.argsort(t)
        if t[order[-1]] - t[order[-2]] < 1e-3:
            continue
        coarse_tot += 1
        tc, pc = int(np.argmax(t)), int(np.argmax(s_pred[i]))
        tf_, pf_ = int(np.argmin(t)), int(np.argmin(s_pred[i]))
        coarse_ok += int(tc == pc)
        fine_ok += int(tf_ == pf_)
        inv += int(pc == tf_)                                           # predicts the finest as coarsest = inversion
    match_c = (coarse_ok / coarse_tot) if coarse_tot else float('nan')
    match_f = (fine_ok / coarse_tot) if coarse_tot else float('nan')
    inv_rate = (inv / coarse_tot) if coarse_tot else float('nan')

    print('  [%s] over %d draws (%d axes):' % (tag, n, nd))
    print('    true spacing(mm) min/mean/max = %.2f/%.2f/%.2f | pred = %.2f/%.2f/%.2f'
          % (trues.min(), trues.mean(), trues.max(), s_pred.min(), s_pred.mean(), s_pred.max()))
    print('    MAE = %.4f mm | log-MAE = %.4f   (baselines: predict-%gmm %.4f mm, predict-mean %.4f mm)'
          % (mae_mm, mae_log, min_res, mae_base_min, mae_base_mean))
    print('    per-axis mm-MAE = [%s] | pooled r = %.3f | per-axis r = [%s]'
          % (', '.join('%.3f' % v for v in mae_ax_mm), r_pool, ', '.join('%.3f' % v for v in r_axes)))
    print('    log-MAE by stratum: native-axis n=%d %.4f | degraded-axis n=%d %.4f'
          % (int(ax_native.sum()), _logmae(ax_native), int(ax_degr.sum()), _logmae(ax_degr)))
    print('    >>> COARSEST-AXIS MATCH (aniso, unique coarsest, n=%d) = %.3f  [finest-match=%.3f | INVERSION-rate=%.3f | chance~0.33]'
          % (coarse_tot, match_c, match_f, inv_rate))
    return {'r_pool': r_pool, 'mae_mm': mae_mm, 'mae_log': mae_log, 'match_coarse': match_c,
            'inv_rate': inv_rate, 'mae_base_min': mae_base_min}


def res_bounds(a, atlas_res):
    """(max_res_iso, max_res_aniso, eff_max_mm) for the chosen sampling mode.
    all-axes: set max_res_aniso == atlas so SampleResolution nullifies the single-axis branch (layers.py build:
    np.array_equal(min_res, max_res_aniso) returns None) and always takes the isotropic branch, which draws all
    n_dims axes independently in U(atlas, max_res_iso), with no 1mm point mass (a ~5% prob_min
    pristine remains as the clean-end calibration)."""
    amin = float(np.min(atlas_res))
    if a.res_mode == 'all-axes':
        return a.res_max, amin, a.res_max
    return a.max_res_iso, a.max_res_aniso, float(max(a.max_res_iso, a.max_res_aniso))


def make_generator(a, gen_labels, labels_shape, atlas_res, output_div):
    max_iso, max_aniso, _ = res_bounds(a, atlas_res)
    return labels_to_image_model(labels_shape=labels_shape, n_channels=1,
                                 generation_labels=gen_labels, output_labels=gen_labels,
                                 n_neutral_labels=a.n_neutral_labels, atlas_res=atlas_res,
                                 target_res=None, output_shape=a.output_shape, output_div_by_n=output_div,
                                 flipping=False, aff=np.eye(4),
                                 scaling_bounds=False, rotation_bounds=False, shearing_bounds=False,
                                 translation_bounds=False, nonlin_std=0,
                                 randomise_res=True, max_res_iso=max_iso, max_res_aniso=max_aniso,
                                 bias_field_std=0, return_resolution=True)


def main():
    a = parse_args()
    print('config:', vars(a))
    output_div = 2 ** a.n_levels

    DATA = os.path.join(ROOT, 'data')
    PRIORS = os.path.join(DATA, 'labels_classes_priors')
    LABELS_DIR = os.path.join(DATA, 'training_label_maps')

    gen_labels = utils.load_array_if_path(os.path.join(PRIORS, 'generation_labels.npy'))
    gen_classes = utils.load_array_if_path(os.path.join(PRIORS, 'generation_classes.npy'))
    labels_paths = utils.list_images_in_folder(LABELS_DIR)
    labels_shape, _, n_dims, _, _, atlas_res = utils.get_volume_info(labels_paths[0], aff_ref=np.eye(4))

    min_res = float(np.min(atlas_res))
    eff_max_iso, _, max_res = res_bounds(a, atlas_res)
    log_min = float(np.log(min_res))
    log_span = float(np.log(max_res) - log_min)
    print('  res-mode=%s -> per-axis spacing in [%.2f, %.2f]mm%s'
          % (a.res_mode, min_res, max_res, ' (all axes independent, no 1mm point mass)' if a.res_mode == 'all-axes' else ''))

    generator = make_generator(a, gen_labels, labels_shape, atlas_res, output_div)
    regression_model = build_regressor(a, generator, n_dims)
    qc_model = build_loss(generator, regression_model, log_min, log_span, a.huber_delta, a.degraded_weight)
    n_trainable = int(np.sum([K.count_params(w) for w in regression_model.trainable_weights]))
    print('  trainable params: %d' % n_trainable)

    s_true = generator.get_layer('resolution').output
    probe = models.Model(generator.inputs, [regression_model.outputs[0], s_true])

    train_src = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels), batchsize=a.batch,
                                   n_channels=1, generation_classes=gen_classes, prior_distributions='uniform')
    probe_src = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels), batchsize=1,
                                   n_channels=1, generation_classes=gen_classes, prior_distributions='uniform')

    print('\n=== PHASE 0: probe at init ===')
    probe_report('init', probe, lambda i: next(probe_src), a.probe_n, log_min, log_span, eff_max_iso, min_res)

    opt = Adam(lr=a.lr, clipnorm=a.clipnorm) if a.clipnorm > 0 else Adam(lr=a.lr)
    qc_model.compile(optimizer=opt, loss=metrics.IdentityLoss().loss)
    dummy = np.zeros((a.batch, 1))
    print('\n=== PHASE 1: train (%d steps, gradient=%d, fourier=%d, grid=%s) ==='
          % (a.steps, a.use_gradient, a.use_fourier))
    run = None
    for step in range(1, a.steps + 1):
        l = float(np.mean(qc_model.train_on_batch(next(train_src), dummy)))
        run = l if run is None else 0.97 * run + 0.03 * l
        if step % a.log_step == 0 or step == 1:
            print('    step %4d   loss=%.5f   running=%.5f' % (step, l, run))

    print('\n=== PHASE 2: probe post-training (SYNTHETIC -- necessary, NOT sufficient) ===')
    res = probe_report('post', probe, lambda i: next(probe_src), a.probe_n, log_min, log_span, eff_max_iso, min_res)

    print('\n=== VERDICT (Brazo A, blind grad+fourier) ===')
    print('  synthetic log-MAE=%.4f | coarsest-axis match=%.3f | inversion-rate=%.3f'
          % (res['mae_log'], res['match_coarse'], res['inv_rate']))
    print('  REMINDER: synthetic success is EXPECTED and NOT sufficient. The decisive test is ADNI (native + '
          'MimicAcquisition-injected + FLAIR): does the coarsest-axis match hold, or invert? Gate any "train more" '
          'decision on the REAL result, not this synthetic number.')


if __name__ == '__main__':
    main()
