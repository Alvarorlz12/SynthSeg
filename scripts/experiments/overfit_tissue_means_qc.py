"""
per-tissue mean regressor (csf / gm / wm).

target = the mean intensity of each tissue, measured on the final normalised image the head sees
(outputs[0], in [0, 1]), computed in-graph over the true tissue masks. this is well defined because
the generation ties each tissue to a single gaussian (grouped generation_classes), so the mean of a
tissue is one intensity, not an average of different draws.

this is a blind regressor: the model reads only the image and outputs 3 numbers. it does not segment.
if we ever want to go through a segmentation we use synthseg directly on the real image and compute the
means analytically from it (that is the anchored computation with a synthseg mask), not a segmentation
head inside this model.

arms:
  encoder   image -> conv encoder -> dense -> one mean per tissue (the regressor; reads only the image)
  anchored  true masks -> masked pool -> one mean per tissue (no training; sanity check, loss ~0)

--tissues picks which means to regress. regressing a single one (--tissues GM) asks whether one mean is
estimable at all, which is the cleanest form of the question. the probe scores every prediction against a
blind baseline that reads the intensity levels off the histogram without localising anything, and reports how
selective the prediction is (does it track the tissue it was asked for, or just the overall brightness).

local cpu smoke (synthqc env, from repo root SynthQC), use >=96 so the crop holds a brain:
    python scripts/experiments/overfit_tissue_means_qc.py --arm anchored --steps 0 --output-shape 96 --n-levels 2
    python scripts/experiments/overfit_tissue_means_qc.py --arm encoder --steps 60 --output-shape 96 --n-levels 2
    python scripts/experiments/overfit_tissue_means_qc.py --arm encoder --tissues GM --steps 60 --output-shape 96 --n-levels 2
real training needs a gpu (larger --output-shape and --steps).

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

EPS = 1e-6

# fixed order: 0=csf, 1=gm, 2=wm. these groups must match the tied classes in the generation array below
# (each tissue must fall inside a single generation class, else its mean mixes intensities). --tissues selects
# which of them are regressed; the other two are still measured, to score the selectivity of the prediction.
ALL_TISSUES = ['CSF', 'GM', 'WM']
TISSUE_GROUPS = {
    'CSF': [4, 5, 43, 44, 14, 15, 24, 72],   # lateral + inf-lateral ventricles l/r, 3rd/4th/5th, extra-cerebral csf
    'GM':  [3, 42, 8, 47],                    # cerebral + cerebellar cortex l/r
    'WM':  [2, 41, 7, 46],                    # cerebral + cerebellar white matter l/r
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['encoder', 'anchored'], default='encoder',
                   help='encoder = blind regressor (image -> the means); anchored = true masks -> means (analytic sanity).')
    p.add_argument('--tissues', type=str, default='CSF,GM,WM',
                   help='tissue means to regress, comma separated (CSF, GM, WM). one at a time tests whether a '
                        'single mean is estimable at all. gm and wm are the decisive ones: they have similar '
                        'volume and interchangeable intensity, so only shape separates them.')
    p.add_argument('--gen-classes', type=str, default='generation_classes_3tissues_grouped.npy',
                   help='file in data/labels_classes_priors used as generation_classes (the tissue grouping).')
    p.add_argument('--regime', choices=['clean', 'gamma', 'bias', 'res', 'full'], default='clean',
                   help='clean = gmm contrast only; the rest add gamma / bias field / random resolution.')
    p.add_argument('--clip-off', action='store_true',
                   help='disable intensity clipping (exact min-max normalisation), recommended in the clean regime.')
    p.add_argument('--deform', action='store_true', help='add spatial deformation (off by default).')
    p.add_argument('--holdout', type=int, default=4,
                   help='label maps kept out of training, probed separately. anatomy is rigid and there are only '
                        'a handful of maps, so a net could localise a tissue by memorising these brains rather '
                        'than by reading their shape; the gap between the two probes measures that.')
    p.add_argument('--blind-fit-n', type=int, default=256,
                   help='images used to fit the blind baseline (0 to skip it).')
    p.add_argument('--min-vox', type=int, default=8, help='a tissue with fewer voxels in the crop is not scored.')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--clipnorm', type=float, default=1.0)
    p.add_argument('--huber-delta', type=float, default=0.1)
    p.add_argument('--loss', choices=['huber', 'mse'], default='huber',
                   help='mse rewards matching the extremes (higher pred variance); huber is l1-like for large errors.')
    p.add_argument('--probe-n', type=int, default=64)
    p.add_argument('--diag-n', type=int, default=8, help='images used for the one-off alignment check.')
    p.add_argument('--log-step', type=int, default=20)
    p.add_argument('--bias-field-std', type=float, default=0.5)
    p.add_argument('--max-res-iso', type=float, default=4.0)
    p.add_argument('--max-res-aniso', type=float, default=8.0)
    p.add_argument('--head', choices=['gap', 'conv'], default='gap',
                   help='gap = global-average-pool + dense; conv = convolve to k channels then average over space '
                        '(the synthseg qc dice head; keeps spatial info until the output).')
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--output-shape', type=int, default=160)
    p.add_argument('--n-levels', type=int, default=5)
    p.add_argument('--nb-conv-per-level', type=int, default=2)
    p.add_argument('--conv-size', type=int, default=3)
    p.add_argument('--feat-count', type=int, default=16, help='encoder feature count at the first level.')
    p.add_argument('--feat-multiplier', type=int, default=2)
    p.add_argument('--activation', type=str, default='elu')
    p.add_argument('--n-neutral-labels', type=int, default=18)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def build_tissue_lut(gen_labels, names):
    """lut[label] = index of the tissue in names, or len(names) for any label outside them (one-hot all zero)."""
    k = len(names)
    valid = set(int(x) for x in gen_labels)
    lut = np.full(int(gen_labels.max()) + 1, k, dtype='int32')
    for idx, name in enumerate(names):
        for lab in TISSUE_GROUPS[name]:
            if lab not in valid:
                print('  [warn] label %d (%s) is not in generation_labels; skipped' % (lab, name))
                continue
            lut[lab] = idx
    return lut, k


def check_alignment(gen_labels, gen_classes):
    """each tissue must sit inside a single generation class, otherwise its mean mixes several draws."""
    lab2gen = {int(l): int(c) for l, c in zip(gen_labels, gen_classes)}
    ok = True
    for name in ALL_TISSUES:
        classes = sorted(set(lab2gen[l] for l in TISSUE_GROUPS[name] if l in lab2gen))
        print('  %-3s -> generation class(es) %s' % (name, classes))
        if len(classes) > 1:
            ok = False
            print('    [misaligned] %s spans %d classes; its mean would mix different intensities' % (name, len(classes)))
    if not ok:
        raise ValueError('tissue groups are not aligned with the generation classes (see above)')


def make_generator(a, gen_labels, labels_shape, atlas_res, output_div):
    clip = 0 if a.clip_off else 300
    return labels_to_image_model(
        labels_shape=labels_shape, n_channels=1,
        generation_labels=gen_labels, output_labels=gen_labels,   # keep label values on outputs[1] for the masks
        n_neutral_labels=a.n_neutral_labels, atlas_res=atlas_res,
        target_res=None, output_shape=a.output_shape, output_div_by_n=output_div,
        flipping=False, aff=np.eye(4),                            # flipping off, it would swap the l/r masks
        scaling_bounds=(0.2 if a.deform else False),
        rotation_bounds=(15 if a.deform else False),
        shearing_bounds=(0.012 if a.deform else False),
        translation_bounds=False,
        nonlin_std=(3.0 if a.deform else 0.0), nonlin_scale=0.0625,
        randomise_res=(a.regime in ('res', 'full')),
        max_res_iso=a.max_res_iso, max_res_aniso=a.max_res_aniso,
        bias_field_std=(a.bias_field_std if a.regime in ('bias', 'full') else 0.0),
        intensity_gamma_std=(0.5 if a.regime in ('gamma', 'full') else 0.0),
        intensity_clip=clip,
        return_bias_std=False, return_resolution=False)


def labels_to_onehot(labels, lut_np, k, name):
    lut = tf.constant(lut_np, dtype='int32')

    def fn(lab):
        idx = tf.gather(lut, tf.cast(lab[..., 0], 'int32'))       # [b,x,y,z]
        return tf.one_hot(idx, depth=k, dtype='float32')          # [b,x,y,z,k]
    return KL.Lambda(fn, name=name)(labels)


def masked_moment_pool(p, image, name):
    """per-class mean of the image weighted by p: mu_k = sum(p_k * i) / sum(p_k). scale-free in the volume."""
    def fn(t):
        pp, img = t
        w = K.sum(pp, axis=[1, 2, 3]) + EPS                       # [b,k]
        m1 = K.sum(pp * img, axis=[1, 2, 3]) / w                  # [b,k]
        return m1
    return KL.Lambda(fn, name=name)([p, image])                   # [b,k]


def build_target(generator, lut_np, k, suffix=''):
    onehot = labels_to_onehot(generator.outputs[1], lut_np, k, 'tm_onehot_true' + suffix)
    mu_true = masked_moment_pool(onehot, generator.outputs[0], 'tm_pool_true' + suffix)
    present = KL.Lambda(lambda oh: K.sum(oh, axis=[1, 2, 3]), name='tm_present' + suffix)(onehot)  # [b,k] counts
    return mu_true, present


def build_head_encoder(a, generator, k):
    enc = nrn_models.conv_enc(input_model=generator, input_shape=a.image_shape, nb_levels=a.n_levels,
                              conv_size=a.conv_size, nb_features=a.feat_count, feat_mult=a.feat_multiplier,
                              nb_conv_per_level=a.nb_conv_per_level, activation=a.activation,
                              batch_norm=None, use_residuals=False, name='tm_enc')  # batch norm off: amplitude is the signal
    feat = enc.outputs[0]
    gmean = KL.GlobalAveragePooling3D(name='tm_gmean')(feat)
    gstd = KL.Lambda(lambda t: K.std(t, axis=[1, 2, 3]), name='tm_gstd')(feat)
    ctx = KL.Concatenate(name='tm_ctx')([gmean, gstd])
    h = KL.Dense(a.hidden, activation=a.activation, name='tm_h')(ctx)
    return KL.Dense(k, activation=None, name='tm_pred')(h)                          # [b,k] linear (sigmoid saturates)


def build_head_conv(a, generator, k):
    # the synthseg qc (dice) head: convolve down to the k output channels and average over space, instead of
    # global-average-pooling the features first and then a dense layer. keeps spatial location until the output,
    # so the convs can look for each tissue locally and read its intensity, then aggregate.
    enc = nrn_models.conv_enc(input_model=generator, input_shape=a.image_shape, nb_levels=a.n_levels,
                              conv_size=a.conv_size, nb_features=a.feat_count, feat_mult=a.feat_multiplier,
                              nb_conv_per_level=a.nb_conv_per_level, activation=a.activation,
                              batch_norm=None, use_residuals=False, name='tm_enc')
    # the inner width is fixed, not k: tying it to the number of targets would make the head narrower whenever we
    # regress fewer tissues, and a flat result on one tissue could then be blamed on the head instead of the task.
    # only the last conv emits k, and it is linear, so the output is an affine read of the averaged maps.
    last = enc.outputs[0]
    last = KL.MaxPool3D(pool_size=(2, 2, 2), padding='same', name='tm_conv_pool')(last)
    last = KL.Conv3D(max(16, k), kernel_size=5, padding='same', activation=a.activation, name='tm_conv0')(last)
    last = KL.Conv3D(k, kernel_size=5, padding='same', activation=None, name='tm_conv1')(last)
    return KL.Lambda(lambda t: tf.reduce_mean(t, axis=[1, 2, 3]), name='tm_conv_pred')(last)     # [b,k]


def build_head_anchored(generator, lut_np, k):
    onehot = labels_to_onehot(generator.outputs[1], lut_np, k, 'tm_onehot_anch')
    return masked_moment_pool(onehot, generator.outputs[0], 'tm_pool_anch')         # equals the target by construction


def build_loss(mu_true, mu_pred, present, min_vox, huber_delta, use_mse=False):
    def fn(t):
        yt, yp, pres = t
        w = K.cast(pres >= float(min_vox), 'float32')             # [b,k] 1 where the tissue is present
        e = K.abs(yt - yp)
        if use_mse:
            elt = K.square(e)                                     # mse, elementwise [b,k]
        else:
            quad = K.minimum(e, huber_delta)
            elt = 0.5 * K.square(quad) + huber_delta * (e - quad)  # huber, elementwise [b,k]
        return K.expand_dims(K.sum(w * elt, axis=1) / (K.sum(w, axis=1) + EPS), -1)  # [b,1]
    loss = KL.Lambda(fn, name='tm_loss')([mu_true, mu_pred, present])
    loss._keras_shape = tuple(loss.get_shape().as_list())
    return loss


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def r2_vs_prior(y, yhat):
    if len(y) < 3:
        return float('nan')
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def score(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return np.abs(y - yhat).mean(), np.mean((y - yhat) ** 2), pearson(y, yhat), r2_vs_prior(y, yhat)


def kmeans1d(x, k, iters=30):
    """k intensity levels of a 1-d histogram, with the mass fraction of each. the levels start spread over the
    range rather than over the quantiles, or a sparse level is never claimed by any voxel."""
    lo, hi = float(x.min()), float(x.max())
    c = lo + (np.arange(k) + 0.5) / k * (hi - lo)
    for _ in range(iters):
        lab = np.abs(x[:, None] - c[None, :]).argmin(1)
        for j in range(k):
            m = lab == j
            if m.sum() > 0:
                c[j] = x[m].mean()
    lab = np.abs(x[:, None] - c[None, :]).argmin(1)
    return c, np.array([float((lab == j).mean()) for j in range(k)])


def blind_features(img, rng, n_sub=20000, n_levels=8):
    """everything an image says to a predictor that is not allowed to localise anything. the histogram is
    summarised as n_levels intensity levels with the mass fraction of each, ordered by mass, plus the level at
    the image border (the background) and the overall level and spread. ordering the levels by mass is still a
    function of the histogram alone, so none of this carries spatial information, but it does hand over the
    volume cue (guess a tissue from how much of the image it takes up) for free, which is exactly the cue we
    must not mistake for localisation. eight levels, not four: the image is painted from 28 generation classes,
    so its histogram has many more modes than the three tissues, and the largest non-background one is not even
    a tissue (the extra-cerebral classes take up more of a crop than any of csf, gm or wm)."""
    b = np.concatenate([img[0].ravel(), img[-1].ravel(), img[:, 0].ravel(),
                        img[:, -1].ravel(), img[:, :, 0].ravel(), img[:, :, -1].ravel()])
    x = img.ravel()
    x = x[rng.choice(x.size, min(n_sub, x.size), replace=False)]
    c, w = kmeans1d(x, n_levels)
    o = np.argsort(-w)
    return np.concatenate([c[o], w[o], [float(np.median(b)), float(x.mean()), float(x.std())]])


def fit_blind(probe, src, n, rng, ridge=1e-2):
    """fit a linear predictor from the blind features to the true tissue means. it is the strongest predictor
    that does not localise: it is fitted, per tissue, on the same targets as the net, and it may use the volume
    cue, but it only ever sees the histogram. it is fitted on its own images and scored later on the probe
    images, so it is out of sample exactly like the net. this is the bar the net has to clear before any
    correlation can be read as localisation."""
    x, y = [], []
    for _ in range(n):
        out = probe.predict(next(src))
        x.append(blind_features(np.asarray(out[5])[0, ..., 0], rng))
        y.append(np.asarray(out[3])[0])                            # the true means of all three tissues
    x, y = np.array(x), np.array(y)
    mu, sd = x.mean(0), x.std(0) + EPS
    z = np.concatenate([(x - mu) / sd, np.ones((len(x), 1))], axis=1)
    w = np.linalg.solve(z.T.dot(z) + ridge * np.eye(z.shape[1]), z.T.dot(y))   # [f+1, 3]

    def predict(image):
        f = np.concatenate([(blind_features(image, rng) - mu) / sd, [1.0]])
        return f.dot(w)                                                        # [3], one mean per tissue
    return predict


def probe_report(tag, probe, src, n, min_vox, names, blind=None):
    """per-tissue mae / mse / pearson r / r2 of the predicted mean against the true effective mean, next to the
    two references it has to be read against.

    blind   the fitted predictor that only sees the histogram. anything up to here needs no localisation.
    levels  the mean of the three true tissue means, the same number for every tissue. this is what a predictor
            that somehow knew the three levels exactly, but not which level belongs to which tissue, would say.
            it is the ceiling of level knowledge without assignment, and the assignment is the part that needs
            localisation. beating it is sufficient evidence of localisation, not necessary.

    selectivity is the same prediction scored against the true means of all three tissues. the three true means
    are nearly uncorrelated with each other, so a prediction that tracks the other two just as well is reading
    the overall brightness of the image, not the tissue it was asked for."""
    pr, gt, cnt, gt_all, cnt_all, bl = [], [], [], [], [], []
    for _ in range(n):
        pp, tt, mm, ta, ma, im = probe.predict(next(src))
        pr.append(np.asarray(pp)[0]); gt.append(np.asarray(tt)[0]); cnt.append(np.asarray(mm)[0])
        gt_all.append(np.asarray(ta)[0]); cnt_all.append(np.asarray(ma)[0])
        if blind is not None:
            bl.append(blind(np.asarray(im)[0, ..., 0]))
    pr, gt, cnt = np.array(pr), np.array(gt), np.array(cnt)
    gt_all, cnt_all = np.array(gt_all), np.array(cnt_all)
    bl = np.array(bl) if blind is not None else None
    levels = gt_all.mean(axis=1)
    print('  [%s] per-tissue effective mean (n=%d):' % (tag, n))
    for idx, name in enumerate(names):
        keep = cnt[:, idx] >= min_vox
        if keep.sum() < 3:
            print('    %-3s  (too few present)' % name)
            continue
        j = ALL_TISSUES.index(name)
        yt, yp = gt[keep, idx], pr[keep, idx]
        # pred std near 0 = collapsed to a constant (then r is nan by definition)
        print('    %-3s  net          mae=%.4f  mse=%.5f  r=%.3f  r2=%.3f  (target mean=%.3f std=%.3f | pred std=%.4f)'
              % ((name,) + score(yt, yp) + (yt.mean(), yt.std(), yp.std())))
        if bl is not None:
            print('    %-3s  blind        mae=%.4f  mse=%.5f  r=%.3f  r2=%.3f' % ((name,) + score(yt, bl[keep, j])))
        print('    %-3s  levels       mae=%.4f  mse=%.5f  r=%.3f  r2=%.3f' % ((name,) + score(yt, levels[keep])))
        sel = []
        for i, other in enumerate(ALL_TISSUES):
            ok = keep & (cnt_all[:, i] >= min_vox)
            sel.append('%s=%.3f' % (other, pearson(gt_all[ok, i], pr[ok, idx])))
        print('    %-3s  selectivity  r vs true %s' % (name, '  '.join(sel)))


def alignment_diagnostic(generator, tissue_groups, src, n, min_vox):
    """for a few images, spread of the per-structure means inside each tissue. near 0 (up to partial volume)
    when the grouping worked, since every structure in a tissue shares one drawn intensity. this catches a
    broken grouping that the anchored r~1 check does not."""
    acc = {t: [] for t in ALL_TISSUES}
    for _ in range(n):
        img, lab = generator.predict_on_batch(next(src))
        img = np.asarray(img)[0, ..., 0]
        lab = np.round(np.asarray(lab)[0, ..., 0]).astype(int)
        for t in ALL_TISSUES:
            ms = [img[lab == l].mean() for l in tissue_groups[t] if int((lab == l).sum()) >= min_vox]
            if len(ms) > 1:
                acc[t].append(float(np.std(ms)))
    print('  spread of member means per tissue (near 0 = one intensity, clean regime):')
    for t in ALL_TISSUES:
        v = acc[t]
        print('    %-3s  spread=%s' % (t, ('%.4f' % np.mean(v)) if v else 'n/a'))


def main():
    a = parse_args()
    np.random.seed(a.seed)
    tf.random.set_seed(a.seed)
    print('config:', vars(a))
    output_div = 2 ** a.n_levels

    DATA = os.path.join(ROOT, 'data')
    PRIORS = os.path.join(DATA, 'labels_classes_priors')
    LABELS_DIR = os.path.join(DATA, 'training_label_maps')
    gen_labels = np.asarray(utils.load_array_if_path(os.path.join(PRIORS, 'generation_labels.npy'))).astype('int32')
    gen_classes = np.asarray(utils.load_array_if_path(os.path.join(PRIORS, a.gen_classes))).astype('int32')
    labels_paths = utils.list_images_in_folder(LABELS_DIR)
    labels_shape, _, n_dims, _, _, atlas_res = utils.get_volume_info(labels_paths[0], aff_ref=np.eye(4))

    names = [t.strip().upper() for t in a.tissues.split(',') if t.strip()]
    unknown = [t for t in names if t not in ALL_TISSUES]
    if unknown or not names:
        raise ValueError('--tissues %s: pick from %s' % (a.tissues, ALL_TISSUES))

    print('  generation classes: %s (%d classes)' % (a.gen_classes, len(np.unique(gen_classes))))
    print('  regressing: %s' % ', '.join(names))
    lut_np, k = build_tissue_lut(gen_labels, names)
    check_alignment(gen_labels, gen_classes)

    generator = make_generator(a, gen_labels, labels_shape, atlas_res, output_div)
    a.image_shape = generator.outputs[0].get_shape().as_list()[1:]
    mu_true, present = build_target(generator, lut_np, k)

    # the true means of all three tissues, for the selectivity of a prediction that only covers some of them
    if names == ALL_TISSUES:
        mu_all, present_all = mu_true, present
    else:
        lut_all, k_all = build_tissue_lut(gen_labels, ALL_TISSUES)
        mu_all, present_all = build_target(generator, lut_all, k_all, '_all')

    if a.arm == 'anchored':
        mu_pred = build_head_anchored(generator, lut_np, k)
    elif a.head == 'conv':
        mu_pred = build_head_conv(a, generator, k)
    else:
        mu_pred = build_head_encoder(a, generator, k)

    probe = models.Model(generator.inputs,
                         [mu_pred, mu_true, present, mu_all, present_all, generator.outputs[0]])
    n_trainable = int(np.sum([K.count_params(w) for w in probe.trainable_weights]))
    print('  trainable params: %d' % n_trainable)

    loss = build_loss(mu_true, mu_pred, present, a.min_vox, a.huber_delta, a.loss == 'mse')
    loss_model = models.Model(generator.inputs, loss)

    # anatomy is rigid without --deform and there are only a handful of label maps, so a net can localise a
    # tissue by memorising these brains instead of reading their shape. hold some maps out and probe them apart:
    # memorised localisation shows up as a gap between the two probes, real localisation transfers.
    n_hold = min(max(a.holdout, 0), len(labels_paths) - 1)
    train_paths = labels_paths[:len(labels_paths) - n_hold] if n_hold else labels_paths
    hold_paths = labels_paths[len(labels_paths) - n_hold:] if n_hold else []
    print('  label maps: %d for training, %d held out' % (len(train_paths), len(hold_paths)))

    def make_src(paths, batchsize):
        return build_model_inputs(path_label_maps=paths, n_labels=len(gen_labels), batchsize=batchsize,
                                  n_channels=1, generation_classes=gen_classes, prior_distributions='uniform')

    src = make_src(train_paths, a.batch)
    probe_src = make_src(train_paths, 1)
    hold_src = make_src(hold_paths, 1) if hold_paths else None
    fit_src = make_src(train_paths, 1)

    print('\nalignment check:')
    alignment_diagnostic(generator, TISSUE_GROUPS, probe_src, a.diag_n, a.min_vox)

    blind = None
    if a.blind_fit_n > 0:
        print('\nfitting the blind baseline (%d images):' % a.blind_fit_n)
        blind = fit_blind(probe, fit_src, a.blind_fit_n, np.random.RandomState(a.seed))
        print('  done')

    print('\nprobe before training:')
    probe_report('init', probe, probe_src, a.probe_n, a.min_vox, names, blind)

    if a.steps > 0 and n_trainable > 0:
        opt = Adam(lr=a.lr, clipnorm=a.clipnorm) if a.clipnorm > 0 else Adam(lr=a.lr)
        loss_model.compile(optimizer=opt, loss=metrics.IdentityLoss().loss)
        dummy = np.zeros((a.batch, 1))
        print('\ntraining (%d steps, arm=%s, regime=%s):' % (a.steps, a.arm, a.regime))
        run = None
        for step in range(1, a.steps + 1):
            l = float(loss_model.train_on_batch(next(src), dummy))
            run = l if run is None else 0.97 * run + 0.03 * l
            if step % a.log_step == 0 or step == 1:
                print('    step %4d   loss=%.5f   running=%.5f' % (step, l, run))
        print('\nprobe after training, anatomy seen during training:')
        probe_report('post', probe, probe_src, a.probe_n, a.min_vox, names, blind)
        if hold_src is not None:
            print('\nprobe after training, anatomy never seen (the one that counts):')
            probe_report('post-held-out', probe, hold_src, a.probe_n, a.min_vox, names, blind)
    else:
        print('\nno training (anchored arm or --steps 0):')
        probe_report('analytic', probe, probe_src, a.probe_n, a.min_vox, names, blind)


if __name__ == '__main__':
    main()
