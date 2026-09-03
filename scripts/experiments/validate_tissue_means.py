"""Offline validation curve for the tissue-means regressor, the analogue of validate_qc.py.

validate_qc.py scores every saved checkpoint on a fixed set of real images sitting on disk, and that is
what makes its curve readable: every point differs from the next only by the weights. We cannot do the
same thing directly, because our validation images do not exist until the generator makes them, and it
makes a new one on every call. So the fixed set is built once here and then reused, which is the same
idea with one extra step. Scoring each checkpoint on its own fresh draw would put the sampling noise of
100 images on every point of the curve, and that noise is about the size of the effect we are looking
for, so the curve would say nothing.

The other thing this script exists for. A batch norm at batchsize 1 normalises each image by its own
statistics, so what training fits is an instance-norm net; keras's predict runs the other branch, the
one with the moving averages, which applies one fixed normalisation to every image. Under randomised
contrast the per-image statistics are exactly what varies, so those are two different functions of the
same weights, and keras's predict-time read is the second one: it cannot cross
var(target) no matter what the network does, because its floor is the constant offset the mismatch
produces. This script reads the checkpoints the first way, the one the training fitted, so the train and
validation curves are the same function and the gap between them means what it usually means.

Run it with the same architecture and regime flags as the training run. load_weights(by_name) is silent
about layers it cannot match, so load_weights_checked refuses a mismatch rather than scoring a different
net without saying so.

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib
"""

import json
import os
import re
import sys
import glob
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import keras.backend as K
import keras.layers as KL
import keras.models as KM

from ext.lab2im import utils
from experiments import synth_dataset as ds
from experiments import training_tissue_means as tm
from SynthSeg.model_inputs import build_model_inputs

eps = 1e-6


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def epoch_of(path):
    return int(re.search(r'(\d+)', os.path.basename(path)).group(1))


def masked_mse(pred, true, present, min_vox):
    # the same quantity build_loss puts in the graph: mean of the squared error over the tissues that are
    # actually in the crop. reproduced here rather than read off a graph tensor because the scorer below
    # takes an image, not a label map, so the target is not in its graph.
    w = (present >= float(min_vox)).astype('float32')
    return (w * (true - pred) ** 2).sum(axis=1) / (w.sum(axis=1) + eps)


# what decides the contents of a set for this head. The three tissue means and the presence gate
# are what its target is, so `tissues` is in here and the bias regime is not.
FINGERPRINT = ('labels_dir', 'holdout', 'n_images', 'seed', 'output_shape', 'neutral_labels',
               'tissues', 'generation_labels', 'generation_classes', 'no_deform', 'randomise_res',
               'bias_std', 'bias_prob', 'gamma_std', 'gamma_prob', 'clip', 'max_res_iso',
               'max_res_aniso')
ARRAYS = ('images', 'mu_true', 'present')


def build_validation_set(a, gen_labels, gen_classes, names, hold_paths, set_path):
    """Draw n_images once from the held-out label maps and keep them. This is the fixed set.

    The file it writes is not a cache, it is the dataset: every number that ends up in a table is scored
    against it, so it has to outlive the run that made it and it has to be able to say what it is. It
    carries its own fingerprint (the arguments that decide its contents) and refuses to be reused under
    a different one, because the failure it would otherwise cause is silent -- an existing file gets
    reused whatever you ask for, and you score the new question against the old images.

    Layout that goes with this, outside the repo and next to the real data:
        <data>/qc-data/synth/<head>/<split>/<name>.npz
    e.g. .../synth/contrast/val/val_1000_clean.npz. The name is for humans; the fingerprint inside is
    what decides whether two files are comparable, so a renamed file cannot lie about itself.
    """

    if set_path is not None and os.path.exists(set_path):
        return ds.load_checked(set_path, ARRAYS, ds.fingerprint(a, FINGERPRINT, len(hold_paths), spatial(a)))

    labels_shape, _, _, _, _, atlas_res = utils.get_volume_info(hold_paths[0], aff_ref=np.eye(4))
    scaling = False if a.no_deform else .2
    rotation = False if a.no_deform else 15
    shearing = False if a.no_deform else .012
    nonlin = 0. if a.no_deform else 4.
    generator = tm.build_generator(labels_shape, atlas_res, gen_labels, a.output_shape, 2 ** a.n_levels,
                                   a.neutral_labels, scaling, rotation, shearing, False, nonlin, .04,
                                   a.randomise_res, a.max_res_iso, a.max_res_aniso, a.bias_std,
                                   a.gamma_std, a.clip, not a.no_deform,
                                   bias_prob=a.bias_prob, gamma_prob=a.gamma_prob)
    lut, k = tm.build_tissue_lut(gen_labels, names)
    mu_true, present = tm.build_target(generator, lut, k)
    probe = KM.Model(generator.inputs, [generator.outputs[0], mu_true, present])

    src = build_model_inputs(path_label_maps=hold_paths, n_labels=len(gen_labels), batchsize=1,
                             n_channels=1, generation_classes=gen_classes, prior_distributions='uniform')
    np.random.seed(a.seed)
    arrays = None
    info = utils.LoopInfo(a.n_images, 10, 'drawing', True)
    for i in range(a.n_images):
        info.update(i)
        img, mu, p = probe.predict(next(src))
        if arrays is None:   # the shapes are only known once the generator has produced one
            arrays = ds.open_set(set_path, a.n_images,
                                 [('images', img.shape[1:], 'float32'),
                                  ('mu_true', mu.shape[1:], 'float32'),
                                  ('present', p.shape[1:], 'float32')])
        arrays[0][i], arrays[1][i], arrays[2][i] = img[0], mu[0], p[0]
    images, mus, pres = arrays

    ds.close_set(set_path, ARRAYS, arrays, ds.fingerprint(a, FINGERPRINT, len(hold_paths), spatial(a)))
    return images, mus, pres


def build_scorer(a, image_shape, k):
    """image in, the three means out, with exactly the architecture training built.

    build_regression_model chains the encoder onto whatever model it is handed, so a one-layer model
    standing in for the generator gets an image-input net out of it without a second definition of the
    architecture. That matters more than it looks: a copy of the head written out again here is a copy
    that can drift from the one that was trained, and nothing would say so.
    """
    img_in = KL.Input(shape=image_shape, name='val_image_input')
    stand_in = KM.Model(img_in, img_in)
    instance_norm = (a.norm == 'instance')
    batch_norm = (None if a.norm != 'batch' or a.batch_norm.strip().lower() in ('none', 'off')
                  else int(a.batch_norm))
    mu_pred = tm.build_regression_model(stand_in, image_shape, k, a.n_levels, a.conv_per_level,
                                        a.conv_size, a.unet_feat, a.feat_mult, a.activation,
                                        batch_norm, not a.no_residuals, instance_norm)
    return KM.Model(stand_in.inputs, [mu_pred])


def spatial(a):
    """The deformation block. Not command-line arguments -- both the training script and this one
    hardcode the same numbers (.2 / 15 / .012 / False / 4. / .04, verified) -- but they decide the
    images, so they go in the fingerprint. `output_div_by_n` is 2**n_levels and rounds the output shape,
    which is why the architecture's depth is in here too."""
    off = a.no_deform
    return dict(scaling=False if off else .2, rotation=False if off else 15,
                shearing=False if off else .012, translation=False,
                nonlin_std=0. if off else 4., nonlin_scale=.04,
                flipping=not off, output_div_by_n=2 ** a.n_levels)


def select_checkpoints(ckpts, a):
    """--epochs picks the ones you actually need; --step_eval thins the rest.

    This exists because scoring is not cheap and is paid per checkpoint: at ~0.5 s an image on a
    500-image set, one checkpoint is 4 minutes, so a 100-epoch curve is 7 hours while the single row a
    table needs is 4 minutes. --step_eval cannot stand in for it, because it strides from the FIRST
    checkpoint, so [::5] gives epochs 1, 6, 11 ... and never the 10 you asked for.

    An epoch that does not exist is an error, not an empty selection: asking for 10 in a run that died
    at 9 has to say so, or the scoring loop finds nothing to do and exits looking like a success."""
    if not a.epochs:
        return ckpts[::a.step_eval]
    want = {int(e) for e in str(a.epochs).replace(' ', '').split(',') if e}
    keep = [c for c in ckpts if epoch_of(c) in want]
    missing = sorted(want - {epoch_of(c) for c in keep})
    if missing:
        raise SystemExit('no checkpoint for epoch(s) %s in this model dir (it has %d, up to %d)'
                         % (', '.join(map(str, missing)), len(ckpts),
                            epoch_of(ckpts[-1]) if ckpts else 0))
    return keep


def validate_training(a, model_dir, validation_dir, images, mu_true, present, k):
    """Score every checkpoint on the fixed set and write one npz per epoch."""

    utils.mkdir(validation_dir)
    net = build_scorer(a, list(images.shape[1:]), k)
    # read the graph with the learning phase pinned instead of going through predict, which pins it to 0.
    # phase 1 keeps the batch norm on the image's own statistics, which is the function training fits at
    # batchsize 1. no updates are passed, so neither call moves the moving averages.
    fn = K.function(net.inputs + [K.learning_phase()], net.outputs)
    phase = 0 if a.frozen_bn else 1

    ckpts = sorted(glob.glob(os.path.join(model_dir, 'tm_*.h5')), key=epoch_of)
    ckpts = select_checkpoints(ckpts, a)
    # only the batch arm has anything to say about the learning phase: with instance norm or none the
    # net holds no BatchNormalization layer, so pinning the phase changes nothing and saying 'batch norm
    # read ...' there is a claim about a layer that is not in the graph.
    how = ('norm=%s, the learning phase does not enter this net' % a.norm if a.norm != 'batch' else
           'batch norm read ' + ('frozen, as predict does' if a.frozen_bn
                                 else "on each image's own statistics, i.e. the net that was trained"))
    print('  %d checkpoints, %d validation images, %s' % (len(ckpts), len(images), how))

    info = utils.LoopInfo(len(ckpts), 1, 'validating', True)
    for i, ckpt in enumerate(ckpts):
        out = os.path.join(validation_dir, os.path.basename(ckpt).replace('.h5', '.npz'))
        if os.path.isfile(out) and not a.recompute:
            continue
        info.update(i)
        tm.load_weights_checked(net, ckpt)
        pred = np.concatenate([fn([images[j:j + 1], phase])[0] for j in range(len(images))])
        loss = masked_mse(pred, mu_true, present, a.min_vox)
        np.savez(out, pred=pred, true=mu_true, present=present, loss=loss,
                 epoch=epoch_of(ckpt), tissues=np.array(a.tissues.split(',')))


def read_validation_dir(validation_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(validation_dir, 'tm_*.npz')), key=epoch_of):
        d = np.load(f)
        pred, true = d['pred'].astype('float64'), d['true'].astype('float64')
        rows.append(dict(epoch=int(d['epoch']), loss=float(d['loss'].mean()),
                         var=float(true.var(axis=0).mean()),
                         pred_std=pred.std(axis=0),
                         r=np.array([np.corrcoef(pred[:, j], true[:, j])[0, 1]
                                     if len(np.unique(pred[:, j])) > 1 else np.nan
                                     for j in range(pred.shape[1])]),
                         tissues=[str(t) for t in d['tissues']]))
    return rows


def plot_validation_curves(validation_dirs, architecture_names, path_tensorboard_files=None,
                           figsize=(11, 5), fontsize=13, y_lim=None, log=False):
    """Validation curve per training, against var(target), with the best epoch marked.

    var(target) is the score of a network that has given up and predicts the mean, so it is the line
    every curve is read against. The train curve is optional and comes from the tensorboard file the
    run wrote, the same tag draw_learning_curve reads.
    """
    validation_dirs = utils.reformat_to_list(validation_dirs)
    architecture_names = utils.reformat_to_list(architecture_names)
    plt.figure(figsize=figsize)

    for validation_dir, name in zip(validation_dirs, architecture_names):
        rows = read_validation_dir(validation_dir)
        if not rows:
            continue
        ep = np.array([r['epoch'] for r in rows])
        val = np.array([r['loss'] for r in rows])
        var = np.array([r['var'] for r in rows])
        line, = plt.plot(ep, val, linewidth=2, label='%s validation' % name)
        best = int(np.argmin(val))
        plt.scatter(ep[best], val[best], s=100, color=line.get_color(), zorder=3)
        plt.axhline(var.mean(), color='grey', linewidth=1.2, linestyle='--')
        plt.annotate('var(target) = %.4f, the score of predicting the mean' % var.mean(),
                     (ep.max(), var.mean()), xytext=(0, 6), textcoords='offset points',
                     color='grey', fontsize=fontsize - 4, ha='right')
        print('%s: best epoch %d, validation mse %.5f, var(target) %.5f, ratio %.3f'
              % (name, ep[best], val[best], var.mean(), val[best] / var.mean()))

    if path_tensorboard_files is not None:
        from tensorflow.python.summary.summary_iterator import summary_iterator
        import logging
        logging.getLogger('tensorflow').disabled = True
        for path, name in zip(utils.reformat_to_list(path_tensorboard_files), architecture_names):
            steps, losses = [], []
            for e in summary_iterator(path):
                for v in e.summary.value:
                    if v.tag in ('loss', 'epoch_loss'):
                        steps.append(e.step + 1); losses.append(v.simple_value)
            plt.plot(np.array(steps), np.array(losses), linewidth=2, linestyle=':',
                     label='%s train' % name)

    plt.grid()
    plt.legend(fontsize=fontsize)
    plt.xlabel('Epochs', fontsize=fontsize)
    plt.ylabel('MSE', fontsize=fontsize)
    if log:
        plt.yscale('log')
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.title('Validation curves', fontsize=fontsize)
    plt.tight_layout(pad=1)
    plt.show()


def parse_args():
    p = ArgumentParser()
    p.add_argument('labels_dir', type=str)
    p.add_argument('--model_dir', type=str, default=None,
                   help='not needed with --build_only')
    p.add_argument('--validation_dir', type=str, default=None,
                   help='defaults to model_dir/validation')
    p.add_argument('--dataset', '--cache', type=str, default=None, dest='dataset',
                   help='where the fixed set lives. NOT a cache: it is the dataset, it carries its '
                        'own fingerprint and is refused if the settings do not match. A path without a '
                        '.npz extension is a DIRECTORY of images.npy + mu_true.npy + present.npy + '
                        'meta.json, which is memory-mappable and the form to use; .npz still works and '
                        'is read whole into ram. Keep it outside the repo, e.g. '
                        '<data>/qc-data/synth/contrast/val/val_1000_clean')
    p.add_argument('--n_images', type=int, default=100,
                   help='images in the fixed set. 100 at 160^3 is about 1.6 GB in ram and on disk')
    p.add_argument('--step_eval', type=int, default=1)
    p.add_argument('--epochs', type=str, default=None,
                   help="comma-separated epochs to score, e.g. '10' or '10,50,100'. Overrides "
                        '--step_eval, which strides from the first checkpoint and so cannot land on a '
                        'given epoch. Use it when you need a table row rather than a curve')
    p.add_argument('--recompute', action='store_true')
    p.add_argument('--plot_only', action='store_true')
    p.add_argument('--generation_labels', type=str, default='data/labels_classes_priors/generation_labels.npy')
    p.add_argument('--generation_classes', type=str,
                   default='data/labels_classes_priors/generation_classes_3tissues_grouped.npy')
    p.add_argument('--tissues', type=str, default='CSF,GM,WM')
    # must match the training run: the split is deterministic on the sorted paths, so a different value
    # here validates on anatomy the net was trained on while calling it held out.
    p.add_argument('--holdout', type=int, required=True)
    p.add_argument('--neutral_labels', type=int, default=18)
    p.add_argument('--output_shape', type=int, default=160)
    p.add_argument('--no_deform', action='store_true')
    # intensity regime: must match the flags the checkpoints were trained with, else this scores the net
    # on a distribution it never saw, silently.
    p.add_argument('--randomise_res', action='store_true')
    p.add_argument('--bias_std', type=float, default=0.)
    p.add_argument('--gamma_std', type=float, default=0.)
    # in the generator's signature, not on the training script's command line by accident: they
    # decide how often the corruption fires, so a run that changes them needs its own set.
    p.add_argument('--bias_prob', type=float, default=.95)
    p.add_argument('--gamma_prob', type=float, default=1.)
    p.add_argument('--clip', type=int, default=0)
    p.add_argument('--max_res_iso', type=float, default=4.)
    p.add_argument('--max_res_aniso', type=float, default=8.)
    # architecture: must match the checkpoints. load_weights_checked raises rather than load a mismatch.
    p.add_argument('--n_levels', type=int, default=5)
    p.add_argument('--conv_per_level', type=int, default=3)
    p.add_argument('--conv_size', type=int, default=5)
    p.add_argument('--unet_feat', type=int, default=24)
    p.add_argument('--feat_mult', type=int, default=2)
    p.add_argument('--activation', type=str, default='relu')
    # must match the checkpoint. 'instance' = per-image norm in train and inference (train == val, no frozen
    # read); 'batch' = BatchNormalization on --batch_norm's axis; 'none' = off.
    p.add_argument('--norm', type=str, default='batch', choices=['batch', 'instance', 'none'])
    p.add_argument('--batch_norm', type=str, default='-1')
    p.add_argument('--no_residuals', action='store_true')
    p.add_argument('--frozen_bn', action='store_true',
                   help='read the checkpoints the way predict does, with the moving averages, instead of '
                        "the way training fitted them. this is predict's read.")
    p.add_argument('--build_only', action='store_true',
                   help='build the dataset and stop: no checkpoint is scored and --model_dir is '
                        'not needed. Building is minutes of GPU and scoring is hours, so they are '
                        'worth separating -- and a set that exists is what every later run reuses')
    p.add_argument('--min_vox', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    print('config:', vars(a))
    assert a.model_dir or a.build_only, '--model_dir is required unless --build_only'
    validation_dir = (a.validation_dir if a.build_only else
                      a.validation_dir or os.path.join(resolve(a.model_dir), 'validation'))

    if not a.plot_only:
        gen_labels = np.asarray(utils.load_array_if_path(resolve(a.generation_labels))).astype('int32')
        gen_classes = np.asarray(utils.load_array_if_path(resolve(a.generation_classes))).astype('int32')
        names = [t.strip().upper() for t in a.tissues.split(',') if t.strip()]
        tm.check_alignment(gen_labels, gen_classes, names)

        labels_paths = sorted(utils.list_images_in_folder(resolve(a.labels_dir)))
        n_hold = min(max(a.holdout, 0), len(labels_paths) - 1)
        hold_paths = labels_paths[len(labels_paths) - n_hold:] if n_hold else labels_paths
        print('  %d label maps, validating on the %d held out: %s'
              % (len(labels_paths), len(hold_paths), [os.path.basename(p) for p in hold_paths]))

        print('\nbuilding the fixed validation set:')
        images, mu_true, present = build_validation_set(a, gen_labels, gen_classes, names, hold_paths,
                                                        resolve(a.dataset) if a.dataset else None)
        if a.build_only:
            print('--build_only: the dataset is written, nothing scored.')
            return
        if a.dataset:
            # the scoring loop skips checkpoints already scored, so a directory of scores has to know
            # which dataset produced them or a rebuilt set leaves stale numbers in a curve.
            ds.stamp_validation_dir(validation_dir, resolve(a.dataset),
                                    ds.fingerprint(a, FINGERPRINT, len(hold_paths), spatial(a)))
        _, k = tm.build_tissue_lut(gen_labels, names)

        print('\nscoring the checkpoints:')
        validate_training(a, resolve(a.model_dir), validation_dir, images, mu_true, present, k)

    print('\ncurve:')
    tb = glob.glob(os.path.join(resolve(a.model_dir), 'logs', 'train', 'events*'))
    plot_validation_curves(validation_dir, os.path.basename(resolve(a.model_dir)),
                           path_tensorboard_files=tb[0] if tb else None)


if __name__ == '__main__':
    main()
