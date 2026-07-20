"""
Offline evaluation for a per-tissue mean regressor checkpoint (see SynthSeg/training_tissue_means.py).

loads a saved checkpoint, rebuilds the probe, and for a set of fresh images reports, per tissue, the mae /
mse / pearson r / r2 of the predicted mean against the true effective mean, next to the two references it
has to be read against:

  blind   a linear predictor fitted on the histogram of the image (intensity levels and their mass
          fractions, ordered by mass, plus the border level and the overall level and spread). it never
          localises anything, only reads the histogram, so it is the bar the net must clear before any
          correlation can be read as localisation. fitted on its own images, scored on the probe images.
  levels  the mean of the three true tissue means, the same number for every tissue: what a predictor that
          knew the three levels but not which belongs to which tissue would say. beating it needs the
          assignment, which needs localisation.

selectivity scores the same prediction against the true means of all three tissues: a prediction that
tracks the other two just as well is reading the overall brightness, not the tissue it was asked for. the
architecture args must match the ones the checkpoint was trained with.

    python scripts/experiments/eval_tissue_means.py <labels_dir> --checkpoint models/.../tm_050.h5

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors
Licensed under the Apache License, Version 2.0; see http://www.apache.org/licenses/LICENSE-2.0
"""


import os
import sys
from argparse import ArgumentParser

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from keras import models
import keras.backend as K
from ext.lab2im import utils
from SynthSeg.model_inputs import build_model_inputs
from SynthSeg import training_tissue_means as tm

eps = 1e-6


def make_predict(probe, frozen_bn):
    # read the graph with the learning phase pinned instead of going through predict, which pins it to 0.
    # phase 1 keeps the batch norm on the image's own statistics, which is the function training fits at
    # batchsize 1; phase 0 applies the moving averages, as predict does. no updates are passed, so neither
    # call moves them. nothing else in the generator is conditioned on the phase, so the image is the same
    # either way.
    fn = K.function(probe.inputs + [K.learning_phase()], probe.outputs)
    phase = 0 if frozen_bn else 1
    return lambda inputs: fn(inputs + [phase])


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def parse_args():
    p = ArgumentParser()
    p.add_argument('labels_dir', type=str)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--generation_labels', type=str, default='data/labels_classes_priors/generation_labels.npy')
    p.add_argument('--generation_classes', type=str,
                   default='data/labels_classes_priors/generation_classes_3tissues_grouped.npy')
    p.add_argument('--tissues', type=str, default='CSF,GM,WM')
    # must match the training run: the split is deterministic on the sorted paths, so a different value here
    # scores the net on anatomy it was trained on while calling it held out.
    p.add_argument('--holdout', type=int, required=True)
    p.add_argument('--neutral_labels', type=int, default=18)
    p.add_argument('--output_shape', type=int, default=160)
    p.add_argument('--no_deform', action='store_true')
    # intensity regime: must match the flags the checkpoint was trained with, else the probe scores the
    # net on a distribution it never saw (silent). the defaults are the clean regime, like training.
    p.add_argument('--randomise_res', action='store_true')
    p.add_argument('--bias_std', type=float, default=0.)
    p.add_argument('--gamma_std', type=float, default=0.)
    # fraction of images each corruption fires on; must match training to score the regime the net saw.
    p.add_argument('--bias_prob', type=float, default=.95)
    p.add_argument('--gamma_prob', type=float, default=1.)
    p.add_argument('--clip', type=int, default=0)
    p.add_argument('--max_res_iso', type=float, default=4.)
    p.add_argument('--max_res_aniso', type=float, default=8.)
    # architecture: these must match the checkpoint, because load_weights(by_name) silently skips a layer
    # it cannot find in the file, so a mismatched arch here scores a different net without saying so.
    p.add_argument('--n_levels', type=int, default=5)
    p.add_argument('--conv_per_level', type=int, default=3)
    p.add_argument('--conv_size', type=int, default=5)
    p.add_argument('--unet_feat', type=int, default=24)
    p.add_argument('--feat_mult', type=int, default=2)
    p.add_argument('--activation', type=str, default='relu')
    # must match the checkpoint. 'instance' = the per-image norm used in both train and inference (no frozen
    # read to worry about); 'batch' = the faithful BatchNormalization on --batch_norm's axis; 'none' = off.
    p.add_argument('--norm', type=str, default='batch', choices=['batch', 'instance', 'none'])
    p.add_argument('--batch_norm', type=str, default='-1')
    p.add_argument('--no_residuals', action='store_true')
    # a batch norm at batchsize 1 normalises each image by its own statistics, so training fits an
    # instance-norm net and that is the function to score. keras's predict instead applies the moving
    # averages, one fixed normalisation for every image, which under randomised contrast is a different
    # function of the same weights. default reads the net that was trained; --frozen_bn reads it the way
    # predict does. with --batch_norm none the two are identical and the flag does nothing.
    p.add_argument('--frozen_bn', action='store_true')
    p.add_argument('--blind_fit_n', type=int, default=256)
    p.add_argument('--probe_n', type=int, default=256)
    p.add_argument('--min_vox', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


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
    b = np.concatenate([img[0].ravel(), img[-1].ravel(), img[:, 0].ravel(),
                        img[:, -1].ravel(), img[:, :, 0].ravel(), img[:, :, -1].ravel()])
    x = img.ravel()
    x = x[rng.choice(x.size, min(n_sub, x.size), replace=False)]
    c, w = kmeans1d(x, n_levels)
    o = np.argsort(-w)
    return np.concatenate([c[o], w[o], [float(np.median(b)), float(x.mean()), float(x.std())]])


def fit_blind(predict, src, n, rng, ridge=1e-2):
    x, y = [], []
    for _ in range(n):
        out = predict(next(src))
        x.append(blind_features(np.asarray(out[5])[0, ..., 0], rng))
        y.append(np.asarray(out[3])[0])
    x, y = np.array(x), np.array(y)
    mu, sd = x.mean(0), x.std(0) + eps
    z = np.concatenate([(x - mu) / sd, np.ones((len(x), 1))], axis=1)
    w = np.linalg.solve(z.T.dot(z) + ridge * np.eye(z.shape[1]), z.T.dot(y))

    def predict(image):
        f = np.concatenate([(blind_features(image, rng) - mu) / sd, [1.0]])
        return f.dot(w)
    return predict


def probe_report(tag, predict, src, n, min_vox, names, blind=None):
    pr, gt, cnt, gt_all, cnt_all, bl = [], [], [], [], [], []
    for _ in range(n):
        pp, tt, mm, ta, ma, im = predict(next(src))
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
        j = tm.all_tissues.index(name)
        yt, yp = gt[keep, idx], pr[keep, idx]
        print('    %-3s  net          mae=%.4f  mse=%.5f  r=%.3f  r2=%.3f  (target mean=%.3f std=%.3f | pred std=%.4f)'
              % ((name,) + score(yt, yp) + (yt.mean(), yt.std(), yp.std())))
        if bl is not None:
            print('    %-3s  blind        mae=%.4f  mse=%.5f  r=%.3f  r2=%.3f' % ((name,) + score(yt, bl[keep, j])))
        print('    %-3s  levels       mae=%.4f  mse=%.5f  r=%.3f  r2=%.3f' % ((name,) + score(yt, levels[keep])))
        sel = []
        for i, other in enumerate(tm.all_tissues):
            ok = keep & (cnt_all[:, i] >= min_vox)
            sel.append('%s=%.3f' % (other, pearson(gt_all[ok, i], pr[ok, idx])))
        print('    %-3s  selectivity  r vs true %s' % (name, '  '.join(sel)))


def main():
    a = parse_args()
    np.random.seed(a.seed)
    print('config:', vars(a))
    gen_labels = np.asarray(utils.load_array_if_path(resolve(a.generation_labels))).astype('int32')
    gen_classes = np.asarray(utils.load_array_if_path(resolve(a.generation_classes))).astype('int32')
    names = [t.strip().upper() for t in a.tissues.split(',') if t.strip()]

    # same deterministic split as training
    labels_paths = sorted(utils.list_images_in_folder(resolve(a.labels_dir)))
    n_hold = min(max(a.holdout, 0), len(labels_paths) - 1)
    train_paths = labels_paths[:len(labels_paths) - n_hold] if n_hold else labels_paths
    hold_paths = labels_paths[len(labels_paths) - n_hold:] if n_hold else []
    print('  label maps: %d seen, %d held out' % (len(train_paths), len(hold_paths)))

    labels_shape, _, _, _, _, atlas_res = utils.get_volume_info(labels_paths[0], aff_ref=np.eye(4))
    scaling = False if a.no_deform else .2
    rotation = False if a.no_deform else 15
    shearing = False if a.no_deform else .012
    nonlin = 0. if a.no_deform else 4.
    generator = tm.build_generator(labels_shape, atlas_res, gen_labels, a.output_shape, 2 ** a.n_levels,
                                   a.neutral_labels, scaling, rotation, shearing, False, nonlin, .04,
                                   a.randomise_res, a.max_res_iso, a.max_res_aniso, a.bias_std,
                                   a.gamma_std, a.clip, not a.no_deform,
                                   bias_prob=a.bias_prob, gamma_prob=a.gamma_prob)
    image_shape = generator.outputs[0].get_shape().as_list()[1:]

    lut, k = tm.build_tissue_lut(gen_labels, names)
    tm.check_alignment(gen_labels, gen_classes, names)
    instance_norm = (a.norm == 'instance')
    batch_norm = (None if a.norm != 'batch' or a.batch_norm.strip().lower() in ('none', 'off')
                  else int(a.batch_norm))
    mu_pred = tm.build_regression_model(generator, image_shape, k, a.n_levels, a.conv_per_level,
                                        a.conv_size, a.unet_feat, a.feat_mult, a.activation,
                                        batch_norm, not a.no_residuals, instance_norm)
    mu_true, present = tm.build_target(generator, lut, k)
    if names == tm.all_tissues:
        mu_all, present_all = mu_true, present
    else:
        lut_all, k_all = tm.build_tissue_lut(gen_labels, tm.all_tissues)
        mu_all, present_all = tm.build_target(generator, lut_all, k_all, '_all')

    probe = models.Model(generator.inputs, [mu_pred, mu_true, present, mu_all, present_all,
                                            generator.outputs[0]])
    tm.load_weights_checked(probe, resolve(a.checkpoint))
    predict = make_predict(probe, a.frozen_bn)
    print('  loaded %s' % a.checkpoint)
    print('  batch norm read %s' % ('frozen, as predict does (moving averages)' if a.frozen_bn else
                                    "on each image's own statistics, i.e. the net that was trained"))

    def make_src(paths):
        return build_model_inputs(path_label_maps=paths, n_labels=len(gen_labels), batchsize=1,
                                  n_channels=1, generation_classes=gen_classes, prior_distributions='uniform')

    blind = None
    if a.blind_fit_n > 0:
        print('\nfitting the blind baseline (%d images):' % a.blind_fit_n)
        blind = fit_blind(predict, make_src(train_paths), a.blind_fit_n, np.random.RandomState(a.seed))
        print('  done')

    print('\nprobe, anatomy seen during training:')
    probe_report('seen', predict, make_src(train_paths), a.probe_n, a.min_vox, names, blind)
    if hold_paths:
        print('\nprobe, anatomy never seen (the one that counts):')
        probe_report('held-out', predict, make_src(hold_paths), a.probe_n, a.min_vox, names, blind)


if __name__ == '__main__':
    main()
