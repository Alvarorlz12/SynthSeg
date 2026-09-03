"""

Validate the bias-field severity regressor across the epochs of one training, on real images.

Same shape as validate_rs.py, which is validate.py's: loop over the checkpoints of a folder, run
predict_bf on a fixed validation set, and drop one bf_results.csv per epoch in its own subfolder, so
an interrupted run resumes by skipping what is already on disk.

THE VALIDATION SET IS MANUFACTURED. No real dataset carries bias severity as a truth, so
make_bf_validation.py builds one: each image is read the way deployment reads it, a field is applied
the way the generator applies it, and std(log B) is written down. The set is frozen on disk because a
fresh draw per checkpoint would put the sampling noise of the set on every point of the curve.

TWO THINGS ABOUT READING THE CURVE. The images are already prepared, so predict_bf is called with
preprocessed=True and nothing is normalised twice. And every anatomy appears once per severity of the
ladder, sharing one field shape, so a real image's own unknown residual bias is a constant within its
ladder rather than noise across the set: what the curve measures is the response to severity, and the
intercept is what the image already had.

The aggregate MAE is not the only number. slope and corr, from read_scores, say whether the head
follows the severity at all, which a MAE dominated by an offset can hide.

Usage, from a script or a notebook:

    from QC import validate_bf
    validate_bf.validate_training(image_dir='<data>/qc-data/validation/img_bf',
                                  gt_csv='<data>/qc-data/validation/gt/bf/targets.csv',
                                  models_dir='models/biasfield/bf_after_gamma',
                                  validation_main_dir='<data>/qc-data/validation/scores/bias/bf_a')
    validate_bf.plot_validation_curves(['.../scores/bias/bf_a'])

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
"""


# python imports
import os
import re
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt

# project imports
from QC.predict_bf import predict_bf

# third-party imports
from ext.lab2im import utils


def validate_training(image_dir,
                      gt_csv,
                      models_dir,
                      validation_main_dir,
                      step_eval=1,
                      preprocessed=True,
                      cropping=160,
                      target_res=1.,
                      n_levels=5,
                      nb_conv_per_level=3,
                      conv_size=5,
                      unet_feat_count=24,
                      feat_multiplier=2,
                      activation='relu',
                      norm='instance',
                      recompute=False):
    """Validate the models saved at different epochs of one training, all in the same folder.
    :param image_dir: folder of corrupted images, as make_bf_validation.py writes them.
    :param gt_csv: the targets.csv that came with them, joined by stem.
    :param models_dir: folder of the bf_*.h5 to validate.
    :param validation_main_dir: folder the per-epoch subfolders are written under.
    :param step_eval: (optional) validate one model every step_eval. The cost of this curve is in the
    checkpoints and not in the images, and neighbouring checkpoints are 1000 steps apart at a fixed
    learning rate, so a stride loses little. Default is 1.
    :param preprocessed: (optional) the images are already resampled, cropped and normalised, which
    they are when they come from make_bf_validation.py. Default is True.
    :param cropping: (optional) the window, when preprocessed is False. Default is 160.
    :param target_res: (optional) resolution images are resampled to, when preprocessed is False.
    :param norm: (optional) the normalisation the checkpoints were trained with. It is an
    architecture argument: a mismatch is refused rather than silently loaded. Default is 'instance'.
    :param recompute: (optional) score a checkpoint again even when a result file for it already
    exists, under either layout. Default is False, i.e. an interrupted curve resumes."""

    utils.mkdir(validation_main_dir)

    list_models = utils.list_files(models_dir, expr=['bf', '.h5'], cond_type='and')[::step_eval]
    assert list_models, 'no bf_*.h5 in %s' % models_dir
    loop_info = utils.LoopInfo(len(list_models), 1, 'validating', True)
    for model_idx, path_model in enumerate(list_models):

        stem = os.path.basename(path_model).replace('.h5', '')
        score_path = os.path.join(validation_main_dir, stem, 'bf_results.csv')
        # a batch scoring run writes the same checkpoint flat, as predbf_<stem>.csv, in this same
        # folder. Both count as done: read_scores and epoch_files already accept either, so a curve
        # started one way is continued rather than recomputed from the first checkpoint.
        flat_path = os.path.join(validation_main_dir, 'predbf_%s.csv' % stem)

        if recompute or not (os.path.isfile(score_path) or os.path.isfile(flat_path)):
            utils.mkdir(os.path.dirname(score_path))
            loop_info.update(model_idx)
            predict_bf(path_images=image_dir,
                       path_out=score_path,
                       path_model=path_model,
                       gt_csv=gt_csv,
                       cropping=cropping,
                       target_res=target_res,
                       preprocessed=preprocessed,
                       n_levels=n_levels,
                       nb_conv_per_level=nb_conv_per_level,
                       conv_size=conv_size,
                       unet_feat_count=unet_feat_count,
                       feat_multiplier=feat_multiplier,
                       activation=activation,
                       norm=norm,
                       recompute=True,
                       verbose=False)


def read_scores(path_csv, column='abs_err', subjects=None):
    """One column of a predict_bf csv, as a float array.

    The file holds std_log_bias, true_std_log_bias and abs_err. Four derived columns on top:

      signed_err   predicted minus true, so a systematic under-read shows as a negative mean
      corr         correlation between predicted and true over the whole file, as a length-1 array
      slope        least-squares slope of predicted on true, same
      intercept    its intercept, which on this set is what the images already carried

    :param subjects: (optional) keep only these rows, a list of stems or a callable on the stem. The
    stems end in _bf0 .. _bfN, one per rung of the severity ladder, so a single severity is cut out
    with a callable on that suffix."""
    with open(path_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return np.array([])

    if subjects is not None:
        keep = subjects if callable(subjects) else (lambda s: s in set(utils.reformat_to_list(subjects)))
        rows = [r for r in rows if keep(r.get('subject', ''))]
        if not rows:
            return np.array([])

    if column in rows[0]:
        return np.array([float(r[column]) for r in rows if r[column] not in (None, '')])

    blank = lambda r: any(r.get(k) in (None, '') for k in ('std_log_bias', 'true_std_log_bias'))
    rows = [r for r in rows if not blank(r)]
    if not rows:
        return np.array([])
    pred = np.array([float(r['std_log_bias']) for r in rows])
    true = np.array([float(r['true_std_log_bias']) for r in rows])

    if column == 'signed_err':
        return pred - true
    if column in ('corr', 'slope', 'intercept'):
        if true.size < 2 or np.all(true == true[0]):
            return np.array([np.nan])
        if column == 'corr':
            return np.array([np.corrcoef(pred, true)[0, 1]])
        a, b = np.polyfit(true, pred, 1)
        return np.array([a if column == 'slope' else b])
    return np.array([])


def epoch_files(validation_dir):
    """(epoch, csv) for every validated checkpoint, under either layout: the per-epoch subfolders
    validate_training writes, or the flat predbf_bf_XXX.csv a batch scoring run writes."""
    out = []
    for d in sorted(glob.glob(os.path.join(validation_dir, '*', ''))):
        for p in glob.glob(os.path.join(d, 'bf_results.csv')):
            out.append((int(re.sub('[^0-9]', '', os.path.basename(os.path.dirname(d)))), p))
    for p in sorted(glob.glob(os.path.join(validation_dir, 'predbf_*.csv'))):
        out.append((int(re.sub('[^0-9]', '', os.path.basename(p))), p))
    return sorted(set(out))


def plot_validation_curves(list_validation_dirs, architecture_names=None, column='abs_err',
                           size_max_circle=100, figsize=(11, 6), y_lim=None, fontsize=18,
                           list_linestyles=None, list_colours=None, plot_legend=False,
                           subjects=None, title='Validation curves', ylabel=None, best='min'):
    """Plot the validation curves of several trainings, from the folders validate_training filled.
    :param column: (optional) which column to average, including the derived ones of read_scores.
    Default is 'abs_err'.
    :param subjects: (optional) restrict every curve to these stems, which is how one rung of the
    severity ladder is plotted on its own.
    :param best: (optional) 'min' or 'max', which end of the curve gets the marker. corr and slope
    are better when larger. Default is 'min'."""

    n_curves = len(list_validation_dirs)

    if architecture_names is None:
        architecture_names = [os.path.basename(os.path.normpath(d)) for d in list_validation_dirs]
    else:
        architecture_names = utils.reformat_to_list(architecture_names, len(list_validation_dirs))

    if plot_legend is False:
        list_legend_labels = ['_nolegend_'] * n_curves
    else:
        list_legend_labels = architecture_names
        if plot_legend is not True:
            list_legend_labels = ['_nolegend_' if i >= plot_legend else list_legend_labels[i] for i in range(n_curves)]

    list_linestyles = utils.reformat_to_list(list_linestyles) if list_linestyles is not None else [None] * n_curves
    list_colours = utils.reformat_to_list(list_colours) if list_colours is not None else [None] * n_curves

    plt.figure(figsize=figsize)
    for net_val_dir, net_name, linestyle, colour, legend_label in zip(list_validation_dirs, architecture_names,
                                                                     list_linestyles, list_colours,
                                                                     list_legend_labels):
        list_net_scores, list_epochs = [], []
        for epoch, path in epoch_files(net_val_dir):
            scores = read_scores(path, column, subjects=subjects)
            if scores.size and np.isfinite(scores).any():
                list_net_scores.append(float(np.nanmean(scores)))
                list_epochs.append(epoch)

        if list_net_scores:
            list_net_scores = np.array(list_net_scores)
            list_epochs = np.array(list_epochs)
            list_epochs, idx = np.unique(list_epochs, return_index=True)
            list_net_scores = list_net_scores[idx]
            pick = np.argmax(list_net_scores) if best == 'max' else np.argmin(list_net_scores)
            print('\n' + net_name)
            print('epochs validated: %d' % len(list_epochs))
            print('epoch best score: %d' % list_epochs[pick])
            print('best score: %0.4f' % list_net_scores[pick])
            plt.plot(list_epochs, list_net_scores, label=legend_label, linestyle=linestyle, color=colour)
            plt.scatter(list_epochs[pick], list_net_scores[pick], s=size_max_circle, color=colour)

    plt.grid()
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.ylabel(column if ylabel is None else ylabel, fontsize=fontsize)
    plt.xlabel('Epochs', fontsize=fontsize)
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1] + 0.01)
    plt.title(title, fontsize=fontsize)
    if plot_legend:
        plt.legend(fontsize=fontsize)
    plt.tight_layout(pad=1)
    plt.show()
