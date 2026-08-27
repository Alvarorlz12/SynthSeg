"""

Trains a regressor to recover the per-axis effective resolution (voxel spacing, in mm) of a synthetic
image directly from the image, end-to-end in one TF/keras graph (SynthSeg philosophy).

Trains through the reusable train_model (from training_qc.py), which saves a checkpoint every epoch, and
carries no in-process probe: evaluation is a separate step on held-out / real data.

Pipeline: labels_to_image_model(randomise_res=True, return_resolution=True) degrades the GMM image to a
random per-axis resolution `s` and exposes it as the named layer 'resolution'; per axis we compute
directional roughness features (mean-squared 1st/2nd finite differences along that axis); one weight-shared
Dense head maps each axis's descriptor to a normalized log-spacing prediction; the weighted log-Huber loss
is computed in-graph and returned as the model output, so it is compiled with metrics.IdentityLoss().

Defaults: no_context=True (the conv-encoder global context drowns the 4 directional features and collapses
the head to the min_res mode), degraded_weight=4.0 (soft hurdle against the ~62% min_res point mass),
lr=3e-4.

Clone of training_biasfield_qc.py with the regression target swapped (bias_std to per-axis
resolution) and the head replaced by the shared directional regressor.

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


# python imports
import numpy as np
import tensorflow as tf
from keras import models
import keras.layers as KL
import keras.backend as K

# project imports
from SynthSeg.training_qc import train_model          # reused verbatim (compiles with IdentityLoss + Adam, saves per epoch)
from SynthSeg.model_inputs import build_model_inputs
from SynthSeg.labels_to_image_model import labels_to_image_model

# third-party imports
from ext.lab2im import utils
from ext.neuron import models as nrn_models


def training(labels_dir,
             model_dir,
             generation_labels,
             output_labels=None,
             generation_classes=None,
             n_neutral_labels=None,
             subjects_prob=None,
             batchsize=1,
             n_channels=1,
             output_shape=160,
             prior_distributions='uniform',
             prior_means=None,
             prior_stds=None,
             # resolution sampling (the in-graph label)
             max_res_iso=4.0,
             max_res_aniso=8.0,
             # orientation augmentation: randomly reorient the anatomy via 90-degree rotations + flips only
             # (exact, no interpolation) so anatomical direction is decorrelated from the array/degradation axes,
             # so the head can't confuse anatomical spectral anisotropy with resolution anisotropy. Deliberately no
             # oblique rotation / scaling / nonlinear here: those resample the labels (nearest) and cause staircase
             # aliasing (spurious high-freq that would confound the very high-freq cue the resolution head reads).
             reorient=False,
             # loss
             huber_delta=0.1,
             degraded_weight=4.0,
             # head architecture
             no_context=True,
             hidden=64,
             ctx_dim=16,
             spectral=False,
             rolloff=False,
             # drop the absolute log-raw energies lg1/lg2 (non-transferable + confound-feeding) to force the head
             # onto the transferable g-norm/roll cues.
             drop_abs=False,
             n_levels=5,
             nb_conv_per_level=2,
             conv_size=3,
             unet_feat_count=24,
             feat_multiplier=2,
             activation='elu',
             # training
             lr=3e-4,
             epochs=20,
             steps_per_epoch=200,
             checkpoint=None):

    """
    Train a regressor to recover the per-axis effective resolution (voxel spacing, mm) from a synthetic
    image, end-to-end in one graph. A checkpoint is saved every epoch in model_dir.

    # NB: each time we provide a parameter with separate values for each axis (e.g. with a numpy array
    # or a sequence), these values refer to the RAS axes.

    :param labels_dir: path of a folder with all the input label maps, or path to a single label map.
    :param model_dir: path of a directory where the models will be saved during training.
    :param generation_labels: list of all the label values in the input label maps. Can be a sequence, a
    1d numpy array, or the path to such an array.

    # generation parameters
    :param output_labels: (optional) does not affect the resolution target (the label is the in-graph
    sampled spacing). Defaults to generation_labels.
    :param generation_classes: (optional) indices regrouping generation labels into GMM-intensity classes.
    :param n_neutral_labels: (optional) number of non-sided generation labels. Defaults to all of them
    (flipping is off on this path).
    :param subjects_prob: (optional) relative sampling importance of each label map.
    :param batchsize: (optional) number of images generated per mini-batch. Default 1.
    :param n_channels: (optional) number of channels to synthesise. Default 1.
    :param output_shape: (optional) shape of the cropped synthetic image. Default 160.
    :param prior_distributions: (optional) 'uniform' or 'normal' for the GMM priors. Default 'uniform'.
    :param prior_means: (optional) hyper-parameters of the prior over the GMM means.
    :param prior_stds: (optional) hyper-parameters of the prior over the GMM standard deviations.

    # resolution / label parameters
    :param max_res_iso: (optional) upper bound of the isotropic LR draw U(min_res, this). Default 4.0.
    :param max_res_aniso: (optional) upper bound of the single-axis anisotropic LR draw. Default 8.0.
    :param huber_delta: (optional) Huber transition on the normalized log-spacing label (label units in
    [0,1]; 0.1 ~= 23% of spacing). Default 0.1.
    :param degraded_weight: (optional) up-weight the per-axis loss of degraded (high-spacing) axes by
    (1 + this * normalized_label) to counter shrinkage from the ~62% min_res point mass. Default 4.0.

    # head architecture
    :param no_context: (optional) if True (default) the head is the pure directional
    roughness into a shared Dense (a learned per-axis FWHM). If False, a down-projected conv-encoder global
    context is concatenated to the directional features.
    :param hidden: (optional) hidden units of the shared per-axis head. Default 64.
    :param ctx_dim: (optional) down-projection of the conv-encoder context (ignored if no_context). Default 16.
    :param n_levels: (optional) levels of the conv encoder; also sets output_div_by_n = 2**n_levels. Default 5.
    :param nb_conv_per_level: (optional) convolutions per encoder level. Default 2.
    :param conv_size: (optional) convolution kernel size. Default 3.
    :param unet_feat_count: (optional) features at the first encoder level. Default 24.
    :param feat_multiplier: (optional) feature multiplier between levels. Default 2.
    :param activation: (optional) activation function ('elu' or 'relu'). Default 'elu'.

    # training parameters
    :param lr: (optional) learning rate. Default 3e-4.
    :param epochs: (optional) number of epochs (also the checkpoint-saving frequency). Default 20.
    :param steps_per_epoch: (optional) steps per epoch. Default 200. Total steps = epochs * steps_per_epoch.
    :param checkpoint: (optional) path of a saved model to load before starting training.
    """

    # prepare data files
    labels_paths = utils.list_images_in_folder(labels_dir)
    generation_labels = utils.load_array_if_path(generation_labels)
    output_labels = generation_labels if output_labels is None else utils.load_array_if_path(output_labels)
    generation_classes = utils.load_array_if_path(generation_classes)
    if n_neutral_labels is None:
        n_neutral_labels = generation_labels.shape[0]

    labels_shape, _, n_dims, _, _, atlas_res = utils.get_volume_info(labels_paths[0], aff_ref=np.eye(4))

    # normalized log-spacing range: (log s - log min_res) / (log max_res - log min_res), mapped to [0, 1]
    min_res = float(np.min(atlas_res))                       # SampleResolution's min_resolution = atlas res
    max_res = float(max(max_res_iso, max_res_aniso))
    log_min = float(np.log(min_res))
    log_span = float(np.log(max_res) - log_min)

    # 1) generator graph: synthesis + in-graph per-axis spacing exposed as the named layer 'resolution'.
    #    reorient=True randomly reorients the anatomy via 90-degree
    #    rotations + flips only (exact axis permutations/reflections, no interpolation, no FOV clip), applied to the
    #    label map before the array-aligned degradation. This decorrelates anatomical direction from the array/
    #    degradation axes (the resolution label stays per-array-axis, and is unaffected since the degradation is the
    #    last, array-aligned step). We deliberately do not add oblique rotation / scaling / nonlinear here: those
    #    resample the labels (nearest) and cause staircase aliasing that would confound the high-freq resolution cue.
    reorient_kw = (dict(flipping=True, enable_90_rotations=True)
                   if reorient else
                   dict(flipping=False, enable_90_rotations=False))
    generator = labels_to_image_model(labels_shape=labels_shape,
                                      n_channels=n_channels,
                                      generation_labels=generation_labels,
                                      output_labels=output_labels,
                                      n_neutral_labels=n_neutral_labels,
                                      atlas_res=atlas_res,
                                      target_res=None,
                                      output_shape=output_shape,
                                      output_div_by_n=2 ** n_levels,
                                      aff=np.eye(4), scaling_bounds=False, rotation_bounds=False,
                                      shearing_bounds=False, translation_bounds=False, nonlin_std=0, **reorient_kw,
                                      randomise_res=True, max_res_iso=max_res_iso, max_res_aniso=max_res_aniso,
                                      bias_field_std=0, return_resolution=True)

    # 2) shared directional per-axis regression head on the synthetic image (generator.outputs[0])
    regression_model = build_resqc_model(generator, n_dims,
                                         no_context=no_context, hidden=hidden, ctx_dim=ctx_dim, spectral=spectral,
                                         rolloff=rolloff, drop_abs=drop_abs,
                                         n_levels=n_levels, nb_conv_per_level=nb_conv_per_level,
                                         conv_size=conv_size, unet_feat_count=unet_feat_count,
                                         feat_multiplier=feat_multiplier, activation=activation)

    # report the regressor's trainable size (with no_context this is a tiny ~hundreds-of-params shared head
    # on top of fixed finite-difference features; the generator layers carry no trainable weights).
    n_trainable = int(np.sum([K.count_params(w) for w in regression_model.trainable_weights]))
    n_total = int(regression_model.count_params())
    print('regressor trainable parameters: %d  (total incl. non-trainable generator layers: %d)'
          % (n_trainable, n_total))

    # 3) in-graph weighted log-Huber loss against the normalized log-spacing label
    qc_model = build_resqc_loss(generator, regression_model, log_min, log_span, huber_delta, degraded_weight)

    # input generator: GMM means/stds + dummy zero target (IdentityLoss ignores y_true)
    model_inputs = build_model_inputs(path_label_maps=labels_paths,
                                      n_labels=len(generation_labels),
                                      batchsize=batchsize,
                                      n_channels=n_channels,
                                      subjects_prob=subjects_prob,
                                      generation_classes=generation_classes,
                                      prior_means=prior_means,
                                      prior_stds=prior_stds,
                                      prior_distributions=prior_distributions)
    input_generator = utils.build_training_generator(model_inputs, batchsize)

    train_model(qc_model, input_generator, lr, epochs, steps_per_epoch, model_dir, 'qc', checkpoint)


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
    """Mean-squared finite difference (order 1 or 2) along each spatial axis, [B, n_dims]. A low-res axis is
    smoother along that direction (lower high-frequency energy), so this directly carries the per-axis cutoff."""
    es = []
    for a in range(n_dims):
        ax = a + 1
        d = KL.Lambda(_axis_diff(ax, order), name='diff_o%d_ax%d' % (order, a))(image)
        e = KL.Lambda(lambda t: K.mean(K.square(t), axis=[1, 2, 3, 4]), name='ge_o%d_ax%d' % (order, a))(d)  # [B]
        es.append(e)
    return KL.Lambda(lambda xs: K.stack(xs, axis=-1), name='gestack_o%d' % order)(es)  # [B, n_dims]


def build_directional_features(image, n_dims=3, spectral=False, rolloff=False, drop_abs=False):
    """Per-axis directional descriptor [B, n_dims, F_dir]. Combines cross-axis-normalized energies (scale/contrast-
    invariant; the relative roughness across axes is the cleanest directional cue) with log raw energies (absolute
    sharpness, lets the head tell an isotropically-low-res image from a native one).
    If spectral=True, also appends per-axis 1D-spectrum shape features (centroid / entropy / hi-lo band ratio in
    normalized frequency), a contrast- and sampling-density-invariant absolute cutoff cue that (unlike the raw
    log energies) transfers sim-to-real and, being per-axis-absolute, survives the cross-axis normalization that
    cancels the isotropic case. The descriptor is 4 features with spectral=False, 7 with spectral=True.
    drop_abs=True drops the absolute log-raw energies lg1/lg2, which do not transfer (real absolute energy !=
    synthetic) and feed the anatomical-anisotropy confound, so dropping them forces the head onto the
    transferable g-norm/roll cues."""
    g1 = _grad_energy(image, 1, n_dims)   # [B, nd]
    g2 = _grad_energy(image, 2, n_dims)   # [B, nd]
    norm = lambda t: t / (K.mean(t, axis=1, keepdims=True) + K.epsilon())
    g1n = KL.Lambda(lambda t: norm(t), name='g1_norm')(g1)
    g2n = KL.Lambda(lambda t: norm(t), name='g2_norm')(g2)
    base = [g1n, g2n]                                                              # relative (transferable) backbone
    if not drop_abs:
        base.append(KL.Lambda(lambda t: K.log(t + K.epsilon()), name='g1_log')(g1))   # absolute log-raw energy
        base.append(KL.Lambda(lambda t: K.log(t + K.epsilon()), name='g2_log')(g2))
    # stack the finite-difference features along a new last axis, [B, nd, 2 (drop_abs) or 4]
    fd = KL.Lambda(lambda xs: K.stack(xs, axis=-1), name='dir_feat')(base)
    extra = []
    if spectral:
        extra.append(build_spectral_features(image, n_dims))                       # [B, nd, 3] (spectrum shape feats)
    if rolloff:
        extra.append(build_rolloff_features(image, n_dims))                        # [B, nd, 1] (bandwidth cue)
    if not extra:
        return fd                                                                  # 4 feats, or 2 with drop_abs
    return KL.Concatenate(axis=-1, name='dir_feat_cat')([fd] + extra)              # 4+1=5, 4+3=7, or 2+1=3


# per-axis 1D spectral shape features (the transferable absolute-resolution cue)
def _axis_power(image, ax, tag='spec'):
    """Mean 1D power spectrum along spatial axis `ax` (1,2,3) of [B,X,Y,Z,1], averaged over the other two
    spatial axes, giving [B, F] with F = N_ax//2 + 1 (real FFT)."""
    def fn(t):
        x = t[..., 0]                                              # [B, X, Y, Z]
        others = [d for d in (1, 2, 3) if d != ax]
        xt = tf.transpose(x, [0] + others + [ax])                  # [B, o1, o2, N_ax]
        p = tf.square(tf.abs(tf.signal.rfft(xt)))                  # [B, o1, o2, F]
        return tf.reduce_mean(p, axis=[1, 2])                      # [B, F]
    return KL.Lambda(fn, name='%s_pow_ax%d' % (tag, ax - 1))(image)


def _spectral_summaries(p):
    """[B, F] power spectrum to [B, 3] shape features in normalized frequency (DC dropped): spectral centroid,
    normalized spectral entropy, and log(high-band / low-band) energy ratio (split at half-Nyquist). All are
    ratios/shapes of the spectrum (not absolute magnitude), so contrast-invariant and comparable across volume shapes;
    a coarser axis concentrates power at low frequency (lower centroid, lower entropy, lower band ratio)."""
    def fn(p):
        eps = K.epsilon()
        F = tf.shape(p)[-1]
        p = p[:, 1:]                                               # drop DC (the mean), [B, F-1]
        freqs = tf.cast(tf.range(1, F), 'float32') / tf.cast(F, 'float32')   # normalized freq in (0, 1), [F-1]
        pn = p / (tf.reduce_sum(p, -1, keepdims=True) + eps)       # power distribution
        centroid = tf.reduce_sum(freqs * pn, -1)                                          # [B]
        entropy = -tf.reduce_sum(pn * tf.math.log(pn + eps), -1) / (tf.math.log(tf.cast(F - 1, 'float32')) + eps)
        hi = tf.cast(freqs > 0.5, 'float32')                       # above half-Nyquist
        e_hi = tf.reduce_sum(p * hi, -1)
        e_lo = tf.reduce_sum(p * (1. - hi), -1)
        band = tf.math.log((e_hi + eps) / (e_lo + eps))                                   # [B]
        return tf.stack([centroid, entropy, band], axis=-1)                               # [B, 3]
    return KL.Lambda(fn)(p)


def build_spectral_features(image, n_dims=3):
    """Per-axis 1D-spectrum shape features, [B, n_dims, 3]."""
    feats = [_spectral_summaries(_axis_power(image, a + 1)) for a in range(n_dims)]        # n_dims x [B, 3]
    return KL.Lambda(lambda xs: K.stack(xs, axis=1), name='spec_feat')(feats)              # [B, nd, 3]


# per-axis cumulative-energy spectral roll-off (the amplitude-invariant bandwidth cue)
def _axis_rolloff(image, ax, frac=0.95):
    """Cumulative-energy spectral roll-off along spatial axis `ax` (1,2,3), giving [B, 1]: the normalized frequency
    below which `frac` (default 95%) of that axis's spectral energy lies (DC dropped). It is a percentile of the
    cumulative power spectrum, so invariant to overall amplitude by construction; it reads the per-axis detail
    bandwidth (where the signal ends), not its magnitude: a low-amplitude-but-full-bandwidth anatomical axis stays
    ~1.0 while a band-limited (resolution-degraded) axis drops. This decouples the amplitude-vs-bandwidth confusion
that makes the plain energy & spectral-centroid features read genuinely-smooth anatomy (e.g. A-P) as 'low
    resolution'. The threshold crossing is a hard count
    (no argmax), fine since the directional features are a fixed transform of the image (no upstream weights)."""
    p = _axis_power(image, ax, tag='roll')                         # [B, F]
    def fn(p):
        eps = K.epsilon()
        p = p[:, 1:]                                               # drop DC, [B, M], M = N_ax // 2
        M = tf.cast(tf.shape(p)[-1], 'float32')
        cum = tf.cumsum(p, axis=-1)                                # [B, M]
        fracc = cum / (cum[:, -1:] + eps)                          # [B, M], monotone increasing to 1
        idx = tf.reduce_sum(tf.cast(fracc < frac, 'float32'), -1)  # [B] = count of bins below frac = crossing index
        return tf.expand_dims(idx / (M + eps), -1)                 # [B, 1] normalized roll-off freq in [0, 1]
    return KL.Lambda(fn, name='rolloff_ax%d' % (ax - 1))(p)


def build_rolloff_features(image, n_dims=3, frac=0.95):
    """Per-axis cumulative-energy spectral roll-off, [B, n_dims, 1]."""
    feats = [_axis_rolloff(image, a + 1, frac=frac) for a in range(n_dims)]                # n_dims x [B, 1]
    return KL.Lambda(lambda xs: K.stack(xs, axis=1), name='roll_feat')(feats)              # [B, nd, 1]


# model: shared directional per-axis head
def build_resqc_model(generator, n_dims,
                      no_context=True, hidden=64, ctx_dim=16, spectral=False, rolloff=False, drop_abs=False,
                      n_levels=5, nb_conv_per_level=2, conv_size=3, unet_feat_count=24,
                      feat_multiplier=2, activation='elu'):
    image = generator.outputs[0]                                  # [B, X, Y, Z, 1] (min-max normalized before the
    #                                                               resolution degradation; ~[0,1] after linear-up,
    #                                                               not re-normalized; the dir features handle scale)
    image_shape = image.get_shape().as_list()[1:]

    dir_feat = build_directional_features(image, n_dims, spectral=spectral, rolloff=rolloff, drop_abs=drop_abs)

    if no_context:
        per_axis = dir_feat
    else:
        # shared conv encoder giving global context (no decoder; the cue is a global+directional frequency property).
        # batch_norm=None on purpose (same lesson as the bias head: BN @ bs=1 normalizes away the spatial-amplitude
        # statistic we depend on).
        enc = nrn_models.conv_enc(input_model=generator, input_shape=image_shape,
                                  nb_levels=n_levels, conv_size=conv_size,
                                  nb_features=unet_feat_count, feat_mult=feat_multiplier,
                                  nb_conv_per_level=nb_conv_per_level, activation=activation,
                                  batch_norm=None, use_residuals=True, name='resqc_enc')
        feat = enc.outputs[0]                                     # [B, w, w, w, C]
        gmean = KL.GlobalAveragePooling3D(name='resqc_gmean')(feat)               # [B, C]
        gstd = KL.Lambda(lambda t: K.std(t, axis=[1, 2, 3]), name='resqc_gstd')(feat)   # [B, C]
        ctx = KL.Concatenate(name='resqc_ctx')([gmean, gstd])                     # [B, 2C]
        # down-project the (high-dim, axis-invariant) context so it cannot drown the 4 directional features.
        ctx = KL.Dense(ctx_dim, activation=activation, name='resqc_ctx_proj')(ctx)   # [B, ctx_dim]
        ctx_t = KL.Lambda(lambda x: K.tile(K.expand_dims(x, 1), [1, n_dims, 1]),
                          name='resqc_ctx_tile')(ctx)                             # [B, nd, ctx_dim]
        per_axis = KL.Concatenate(axis=-1, name='resqc_per_axis')([dir_feat, ctx_t])   # [B, nd, 4+ctx_dim]

    # one weight-shared head: Dense applies over the last axis and broadcasts over the nd axis, so same weights for
    # every axis (axis-permutation consistent by construction).
    h = KL.Dense(hidden, activation=activation, name='resqc_h')(per_axis)         # [B, nd, hidden]
    out = KL.Dense(1, activation='sigmoid', name='resqc_out')(h)                  # [B, nd, 1]
    pred = KL.Lambda(lambda t: t[..., 0], name='resqc_pred')(out)                 # [B, nd] in (0, 1)

    return models.Model(generator.inputs, pred)


def build_resqc_loss(generator, regression_model, log_min, log_span, huber_delta, degraded_weight=0.0):
    # read the per-axis spacing target from the generator's named layer (mm), normalized log-spacing in [0,1].
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


def huber_elt(y_true, y_pred, delta=0.1):
    """Per-element Huber on the (log-spacing, normalized) label, not reduced, [B, n_dims]."""
    e = K.abs(y_true - y_pred)
    quad = K.minimum(e, delta)        # quadratic part, capped at delta (avoids tf.where)
    lin = e - quad                    # linear excess beyond delta
    return 0.5 * K.square(quad) + delta * lin
