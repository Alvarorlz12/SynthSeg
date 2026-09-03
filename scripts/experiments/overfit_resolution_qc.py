"""

Per-axis effective-resolution QC diagnostic (voxel spacing), the resolution analogue of
overfit_field_biasfield_qc.py.

Goal: recover the per-axis effective resolution (voxel spacing, in mm) of a synthetic image directly
from the image, end-to-end in one TF/keras graph (SynthSeg philosophy). SynthSeg's randomise_res path
degrades the GMM image to a random per-axis resolution `s` (blur by a latent slice thickness t<=s, then
downsample to the s-grid with nearest and resample back to the HR grid with linear) and stores it at HR.
So the in-graph `resolution` label is the effective resolution behind a fine (e.g. 1mm) grid, i.e. the
deployment case "acquired low-res, resampled to 1mm". The label is `resolution` (spacing); the slice
`thickness` is a nuisance latent, not a target.

Identifiability (and the risk): `s` is encoded ~deterministically in the resampling-grid imprint (the
linear-up ramp period), far more than in the blur (which only carries min(s,t)~t). That deterministic
cue is also a SynthSeg-specific artifact, so the main risk is the sim-to-real gap; --grid-ablation is
what measures how much of the score rides on it (switch the resampling cue off with blur_only, or
perturb it with kernel_random / kernel_phase).

Head (shared directional): a shared 3D conv encoder gives a global context vector; per axis we also
compute explicit directional roughness features (mean-squared 1st/2nd finite differences along that
axis, raw-log and cross-axis-normalized, a learned generalization of an AFNI-FWHM cue). One weight-shared
Dense head maps each axis's [directional | context] descriptor to a normalized log-spacing prediction
(sigmoid in [0,1]). Sharing the head across axes gives axis-permutation consistency; a vanilla conv
encoder is not axis-equivariant, so permutation/flip augmentation and I/O canonicalization handle
real-world orientation later.

Loss = log-Huber on the per-axis normalized log-spacing (blur/spacing scale multiplicatively).

Run a CPU build-smoke locally (synthqc env), e.g.:
    python scripts/experiments/overfit_resolution_qc.py --output-shape 32 --n-levels 2 --steps 4 --probe-n 8
Real training needs a GPU (full 160^3), from the repo root:
    python -u scripts/experiments/overfit_resolution_qc.py --steps 800 | tee res_run.log

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

from SynthSeg.labels_to_image_model import labels_to_image_model
from SynthSeg.model_inputs import build_model_inputs
from SynthSeg import metrics_model as metrics
from ext.lab2im import utils
from ext.neuron import models as nrn_models


def parse_args():
    p = argparse.ArgumentParser()
    # training
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--clipnorm', type=float, default=0.0)
    p.add_argument('--probe-n', type=int, default=96)
    p.add_argument('--log-step', type=int, default=20)
    p.add_argument('--huber-delta', type=float, default=0.1,
                   help='Huber transition on the normalized log-spacing label (label units in [0,1]).')
    p.add_argument('--degraded-weight', type=float, default=0.0,
                   help='up-weight the per-axis loss of degraded (high-spacing) axes by (1 + this*normalized_label) '
                        'to counter shrinkage from the ~62%% min_res point mass. 0 = uniform (plain mean).')
    # resolution sampling (the in-graph label)
    p.add_argument('--max-res-iso', type=float, default=4.0, help='upper bound of the isotropic LR draw U(min,this).')
    p.add_argument('--max-res-aniso', type=float, default=8.0, help='upper bound of the single-axis aniso LR draw.')
    p.add_argument('--grid-ablation', choices=['none', 'blur_only', 'kernel_random', 'kernel_phase'], default='none',
                   help='ABLATION of the resampling-grid cheat (the #1 sim->real risk). blur_only: bypass the '
                        'nearest-down/linear-up resampling so ONLY the Gaussian blur (effective-cutoff) cue remains. '
                        'kernel_random: randomize the resample interp method (nearest/linear) + sub-voxel grid phase '
                        '(HARSH/unrealistic -- nearest-up block edges poison finite-diff features). kernel_phase: '
                        'REALISTIC variant -- keep the linear reconstruction, jitter only the sub-voxel phase (per-axis, '
                        'scaled by degradation so native axes stay bit-exact identity). none = full pipeline.')
    # head architecture
    p.add_argument('--hidden', type=int, default=64, help='hidden units of the shared per-axis head.')
    p.add_argument('--no-context', action='store_true',
                   help='ABLATION: drop the conv-encoder global context -> head = pure directional roughness->Dense '
                        '(a learned per-axis FWHM). Cheaper; tests whether the directional cue alone suffices.')
    p.add_argument('--ctx-dim', type=int, default=16,
                   help='down-project the conv-encoder global context to this many features before broadcasting to '
                        'the per-axis head, so the (axis-invariant) context does NOT drown the 4 directional features '
                        '-- the raw 2C(~768)-d context caused a collapse-to-mode. Ignored under --no-context.')
    # encoder
    p.add_argument('--output-shape', type=int, default=160)
    p.add_argument('--n-levels', type=int, default=5)
    p.add_argument('--nb-conv-per-level', type=int, default=2)
    p.add_argument('--conv-size', type=int, default=3)
    p.add_argument('--unet-feat-count', type=int, default=24)
    p.add_argument('--feat-multiplier', type=int, default=2)
    p.add_argument('--activation', type=str, default='elu')
    p.add_argument('--n-neutral-labels', type=int, default=18)
    return p.parse_args()


# loss
def huber_elt(y_true, y_pred, delta=0.1):
    """per-element huber on the normalized log-spacing label, not reduced, gives [B, n_dims]."""
    e = K.abs(y_true - y_pred)
    quad = K.minimum(e, delta)        # quadratic part, capped at delta (avoids tf.where)
    lin = e - quad                    # linear excess beyond delta
    return 0.5 * K.square(quad) + delta * lin


# directional roughness features (explicit, per axis, from the image)
def _axis_diff(ax, order):
    """Return a fn computing the {1st|2nd} finite difference along spatial axis `ax` (1,2,3) of a 5D tensor."""
    def fn(t):
        def sl(start, end):
            idx = [slice(None)] * 5
            idx[ax] = slice(start, end)
            return t[tuple(idx)]
        if order == 1:
            return sl(1, None) - sl(0, -1)
        return sl(2, None) - 2.0 * sl(1, -1) + sl(0, -2)
    return fn


def _grad_energy(image, order, n_dims=3):
    """mean-squared finite difference (order 1 or 2) along each spatial axis, gives [B, n_dims]. a low-res axis is
    smoother along that direction so it has lower high-frequency energy, which carries the per-axis cutoff."""
    es = []
    for a in range(n_dims):
        ax = a + 1
        d = KL.Lambda(_axis_diff(ax, order), name='diff_o%d_ax%d' % (order, a))(image)
        e = KL.Lambda(lambda t: K.mean(K.square(t), axis=[1, 2, 3, 4]), name='ge_o%d_ax%d' % (order, a))(d)  # [B]
        es.append(e)
    return KL.Lambda(lambda xs: K.stack(xs, axis=-1), name='gestack_o%d' % order)(es)  # [B, n_dims]


def build_directional_features(image, n_dims=3):
    """per-axis directional descriptor [B, n_dims, F_dir]. combines cross-axis-normalized energies (scale/contrast-
    invariant; the relative roughness across axes is the cleanest directional cue) with log raw energies (absolute
    sharpness, lets the head tell an isotropically-low-res image from a native one)."""
    g1 = _grad_energy(image, 1, n_dims)   # [B, nd]
    g2 = _grad_energy(image, 2, n_dims)   # [B, nd]
    norm = lambda t: t / (K.mean(t, axis=1, keepdims=True) + K.epsilon())
    g1n = KL.Lambda(lambda t: norm(t), name='g1_norm')(g1)
    g2n = KL.Lambda(lambda t: norm(t), name='g2_norm')(g2)
    lg1 = KL.Lambda(lambda t: K.log(t + K.epsilon()), name='g1_log')(g1)
    lg2 = KL.Lambda(lambda t: K.log(t + K.epsilon()), name='g2_log')(g2)
    # stack the 4 per-axis features along a new last axis, gives [B, nd, 4]
    return KL.Lambda(lambda xs: K.stack(xs, axis=-1), name='dir_feat')([g1n, g2n, lg1, lg2])


# model: shared directional per-axis head
def build_resqc_model(a, generator, n_dims):
    image = generator.outputs[0]                                  # [B, X, Y, Z, 1] (min-max normalized before the
    #                                                               resolution degradation; ~[0,1] after linear-up,
    #                                                               not re-normalized, the dir features handle scale)
    image_shape = image.get_shape().as_list()[1:]

    dir_feat = build_directional_features(image, n_dims)          # [B, nd, 4]

    if a.no_context:
        per_axis = dir_feat
    else:
        # shared conv encoder gives global context (no decoder; the cue is a global+directional frequency property).
        # batch_norm=None on purpose (same lesson as the bias head: BN @ bs=1 normalizes away the spatial-amplitude
        # statistic we depend on).
        enc = nrn_models.conv_enc(input_model=generator, input_shape=image_shape,
                                  nb_levels=a.n_levels, conv_size=a.conv_size,
                                  nb_features=a.unet_feat_count, feat_mult=a.feat_multiplier,
                                  nb_conv_per_level=a.nb_conv_per_level, activation=a.activation,
                                  batch_norm=None, use_residuals=True, name='resqc_enc')
        feat = enc.outputs[0]                                     # [B, w, w, w, C]
        gmean = KL.GlobalAveragePooling3D(name='resqc_gmean')(feat)               # [B, C]
        gstd = KL.Lambda(lambda t: K.std(t, axis=[1, 2, 3]), name='resqc_gstd')(feat)   # [B, C]
        ctx = KL.Concatenate(name='resqc_ctx')([gmean, gstd])                     # [B, 2C]
        # down-project the (high-dim, axis-invariant) context so it cannot drown the 4 directional features: the raw
        # 2C(~768) context vs 4 directional let the head fit a per-sample constant, collapsing to the mode (predict 1mm).
        ctx = KL.Dense(a.ctx_dim, activation=a.activation, name='resqc_ctx_proj')(ctx)   # [B, ctx_dim]
        ctx_t = KL.Lambda(lambda x: K.tile(K.expand_dims(x, 1), [1, n_dims, 1]),
                          name='resqc_ctx_tile')(ctx)                             # [B, nd, ctx_dim]
        per_axis = KL.Concatenate(axis=-1, name='resqc_per_axis')([dir_feat, ctx_t])   # [B, nd, 4+ctx_dim]

    # one weight-shared head: Dense applies over the last axis and broadcasts over the nd axis, so same weights for
    # every axis (axis-permutation consistent by construction).
    h = KL.Dense(a.hidden, activation=a.activation, name='resqc_h')(per_axis)     # [B, nd, hidden]
    out = KL.Dense(1, activation='sigmoid', name='resqc_out')(h)                  # [B, nd, 1]
    pred = KL.Lambda(lambda t: t[..., 0], name='resqc_pred')(out)                 # [B, nd] in (0, 1)

    return models.Model(generator.inputs, pred)


def build_resqc_loss(generator, regression_model, log_min, log_span, huber_delta, degraded_weight=0.0):
    # read the per-axis spacing target from the generator's named layer (in mm), gives normalized log-spacing in [0,1].
    # normalize as (log s - log min_res) / (log max_res - log min_res) so the [0,1] range tracks the atlas resolution
    # (with the current 1mm atlas, log_min=0 and this reduces to log(s)/log(max_res)).
    s = generator.get_layer('resolution').output                                 # [B, nd] mm
    y = KL.Lambda(lambda x: K.clip((K.log(x) - log_min) / log_span, 0., 1.), name='resqc_label')(s)
    pred = regression_model.outputs[0]                                           # [B, nd]

    # weighted log-Huber: up-weight degraded (high-spacing) axes by (1 + degraded_weight * y) to counter the
    # regress-to-the-mode shrinkage caused by the ~62% min_res point mass (degraded_weight=0 gives plain mean).
    def _wloss(args):
        yt, yp = args
        elt = huber_elt(yt, yp, huber_delta)                                     # [B, nd]
        w = 1.0 + degraded_weight * yt
        return K.sum(w * elt) / (K.sum(w) + K.epsilon())
    loss = KL.Lambda(_wloss, name='qc_loss')([y, pred])
    loss._keras_shape = tuple(loss.get_shape().as_list())                        # keras 2.3 quirk (cf. training_qc.py)
    return models.Model(generator.inputs, loss)


# probe / metrics
def _pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else float('nan')


def probe_report(tag, probe, draw_fn, n_draws, log_min, log_span, max_res_iso, min_res=1.0):
    """probe outputs [pred (normalized log-spacing) [B,nd], s_true (mm) [B,nd]]."""
    preds, trues = [], []
    for i in range(n_draws):
        p, s = probe.predict(draw_fn(i))
        preds.append(np.asarray(p)[0]); trues.append(np.asarray(s)[0])           # each [nd]
    preds = np.array(preds)                       # [N, nd] normalized
    trues = np.array(trues)                       # [N, nd] mm
    nd = trues.shape[1]
    s_pred = np.exp(preds * log_span + log_min)   # [N, nd] mm (exact inverse of the label normalization)

    mae_mm = float(np.abs(s_pred - trues).mean())
    mae_log = float(np.abs(np.log(s_pred) - np.log(trues)).mean())
    r_axes = [_pearson(s_pred[:, a], trues[:, a]) for a in range(nd)]
    r_pool = _pearson(s_pred.flatten(), trues.flatten())

    # baselines on the same draws: predict min_res everywhere (strong, ~62% of the label is the min_res spike),
    # and predict the per-set mean spacing.
    mae_base_min = float(np.abs(min_res - trues).mean())
    mae_base_mean = float(np.abs(trues.mean() - trues).mean())

    # per-draw regime: native (all axes ~min_res), iso (all equal, > min_res), aniso (one axis differs)
    mn, mx = trues.min(axis=1), trues.max(axis=1)
    native = mx <= min_res + 1e-3
    iso = (~native) & ((mx - mn) <= 1e-3)
    aniso = (~native) & ((mx - mn) > 1e-3)

    # per-axis strata: native axis (==min) vs degraded; and the sub-bucket above the iso ceiling (only the aniso
    # path reaches (max_res_iso, max_res_aniso]; these are still in-distribution, not extrapolation).
    ax_native = trues <= min_res + 1e-3
    ax_degr = ~ax_native
    ax_above = ax_degr & (trues > max_res_iso + 1e-3)

    def _mae(mask):
        return float(np.abs(s_pred[mask] - trues[mask]).mean()) if mask.any() else float('nan')

    print('  [%s] over %d draws (%d axes):' % (tag, n_draws, nd))
    print('    true  spacing(mm)  min/mean/max = %.2f / %.2f / %.2f' % (trues.min(), trues.mean(), trues.max()))
    print('    pred  spacing(mm)  min/mean/max = %.2f / %.2f / %.2f' % (s_pred.min(), s_pred.mean(), s_pred.max()))
    print('    MAE = %.4f mm   |   log-MAE = %.4f' % (mae_mm, mae_log))
    print('      baselines: predict-%gmm MAE = %.4f mm   predict-mean MAE = %.4f mm' % (min_res, mae_base_min, mae_base_mean))
    print('    Pearson r: pooled = %.3f   per-axis = [%s]' % (r_pool, ', '.join('%.3f' % v for v in r_axes)))
    print('    MAE by draw regime:  native n=%d MAE=%.4f | iso n=%d MAE=%.4f | aniso n=%d MAE=%.4f'
          % (int(native.sum()), _mae(np.repeat(native[:, None], nd, 1)),
             int(iso.sum()), _mae(np.repeat(iso[:, None], nd, 1)),
             int(aniso.sum()), _mae(np.repeat(aniso[:, None], nd, 1))))
    print('    MAE by axis stratum: native-axis n=%d MAE=%.4f | degraded n=%d MAE=%.4f | above-iso(>%.0f) n=%d MAE=%.4f'
          % (int(ax_native.sum()), _mae(ax_native), int(ax_degr.sum()), _mae(ax_degr),
             max_res_iso, int(ax_above.sum()), _mae(ax_above)))
    return {'r_pool': r_pool, 'r_axes': r_axes, 'mae_mm': mae_mm, 'mae_log': mae_log,
            'mae_base_min': mae_base_min, 's_pred': s_pred, 'trues': trues}


def make_generator(a, gen_labels, labels_shape, atlas_res, output_div):
    """labels_to_image_model with randomise_res on and the per-axis spacing exposed as 'resolution'. clean scope:
    deformation / bias / flipping off so only the resolution degradation drives the image (gamma intensity-aug stays
    on, it does not interfere with the geometric resolution cue and adds realism)."""
    return labels_to_image_model(labels_shape=labels_shape, n_channels=1,
                                 generation_labels=gen_labels, output_labels=gen_labels,
                                 n_neutral_labels=a.n_neutral_labels, atlas_res=atlas_res,
                                 target_res=None, output_shape=a.output_shape, output_div_by_n=output_div,
                                 flipping=False, aff=np.eye(4),
                                 scaling_bounds=False, rotation_bounds=False, shearing_bounds=False,
                                 translation_bounds=False, nonlin_std=0,
                                 randomise_res=True, max_res_iso=a.max_res_iso, max_res_aniso=a.max_res_aniso,
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

    min_res = float(np.min(atlas_res))                       # SampleResolution's min_resolution = the atlas resolution
    max_res = float(max(a.max_res_iso, a.max_res_aniso))
    log_min = float(np.log(min_res))
    log_span = float(np.log(max_res) - log_min)              # normalize log-spacing over [min_res, max_res] to [0,1]

    generator = make_generator(a, gen_labels, labels_shape, atlas_res, output_div)
    regression_model = build_resqc_model(a, generator, n_dims)
    qc_model = build_resqc_loss(generator, regression_model, log_min, log_span, a.huber_delta, a.degraded_weight)

    # probe: predicted normalized log-spacing + the true spacing (mm)
    s_true = generator.get_layer('resolution').output
    probe = models.Model(generator.inputs, [regression_model.outputs[0], s_true])

    # fresh regime: new anatomy+contrast+resolution every step / probe draw (the real task)
    train_src = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels), batchsize=a.batch,
                                   n_channels=1, generation_classes=gen_classes, prior_distributions='uniform')
    probe_src = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels), batchsize=1,
                                   n_channels=1, generation_classes=gen_classes, prior_distributions='uniform')
    get_train = lambda step: next(train_src)
    probe_draw = lambda i: next(probe_src)

    print('\n=== PHASE 0: probe at init (no training) ===')
    probe_report('init', probe, probe_draw, a.probe_n, log_min, log_span, a.max_res_iso, min_res)

    opt = Adam(lr=a.lr, clipnorm=a.clipnorm) if a.clipnorm > 0 else Adam(lr=a.lr)
    qc_model.compile(optimizer=opt, loss=metrics.IdentityLoss().loss)
    dummy = np.zeros((a.batch, 1))
    print('\n=== PHASE 1: train (%d steps, batch=%d, lr=%.1e, feat=%d, ctx=%s) ==='
          % (a.steps, a.batch, a.lr, a.unet_feat_count, not a.no_context))
    run = None
    for step in range(1, a.steps + 1):
        l = float(np.mean(qc_model.train_on_batch(get_train(step), dummy)))
        run = l if run is None else 0.97 * run + 0.03 * l
        if step % a.log_step == 0 or step == 1:
            print('    step %4d   loss=%.5f   running=%.5f' % (step, l, run))

    print('\n=== PHASE 2: probe post-training ===')
    res = probe_report('post', probe, probe_draw, a.probe_n, log_min, log_span, a.max_res_iso, min_res)

    print('\n=== VERDICT ===')
    beats_base = res['mae_mm'] < 0.5 * res['mae_base_min']
    if res['r_pool'] > 0.7 and beats_base:
        print('RESOLUTION RECOVERED: r=%.2f, MAE=%.3f mm << predict-%gmm baseline %.3f mm.'
              % (res['r_pool'], res['mae_mm'], min_res, res['mae_base_min']))
        print('-> the per-axis head learns the in-graph signal.')
    elif res['r_pool'] > 0.4:
        print('PROMISING: r=%.2f, MAE=%.3f mm vs baseline %.3f mm. More steps / tune.'
              % (res['r_pool'], res['mae_mm'], res['mae_base_min']))
    else:
        print('NOT YET: r=%.2f, MAE=%.3f mm vs baseline %.3f mm. Inspect (features, lr, head capacity).'
              % (res['r_pool'], res['mae_mm'], res['mae_base_min']))


if __name__ == '__main__':
    main()
