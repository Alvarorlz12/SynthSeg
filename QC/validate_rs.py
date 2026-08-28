"""

Validate the per-axis resolution regressor across the epochs of one training, on real images.

Same shape as validate_tm.py, which is validate.py's: loop over the checkpoints of a folder, run
predict_rs on a fixed validation set, and drop one rs_results.csv per epoch in its own subfolder, so an
interrupted run resumes by skipping what is already on disk. There is no gt_dir: the truth is the voxel
spacing in each image's own header, which predict_rs writes into the true_* columns, so validation
takes one folder and there is no pairing by sorting order to get wrong.

The validation set has to be images nobody has resampled. FreeSurfer --conform'd volumes are 1 mm on
every axis whatever they were acquired at, so every true deficit is 0 and a relu head scores perfectly
by answering 0: point this at the raw BIDS, and at a set that spans the range.

Do not score this in aggregate. The training loss is a plain mse on the deficit in millimetres, so a
relative error is weighted by s^2 and the coarse end carries most of it, while the QC call is made in
(1, 2] mm. The csv keeps true_R/A/S per image so the bands can be cut afterwards from the same files:
use the subjects argument of read_scores, or group on the true columns. The curve is paired: the same
images against their own headers at every epoch, so it ranks epochs without measuring performance.

Usage, from a script or a notebook:

    from QC import validate_rs
    validate_rs.validate_training(image_dir='$WORK/qc-data/validation/img',
                                  models_dir='$WORK/SynthQC/models/resolution/rs_extra_uniform_mm_a',
                                  validation_main_dir='$WORK/qc-data/validation/scores/resolution/rs_a')
    validate_rs.plot_validation_curves(['.../scores/resolution/rs_a'])

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
import numpy as np
import matplotlib.pyplot as plt

# project imports
from QC.predict_rs import predict_rs, AXES

# third-party imports
from ext.lab2im import utils


def validate_training(image_dir,
                      models_dir,
                      validation_main_dir,
                      step_eval=1,
                      cropping=160,
                      target_res=1.,
                      minmax_norm=False,
                      n_levels=5,
                      nb_conv_per_level=3,
                      conv_size=5,
                      unet_feat_count=24,
                      feat_multiplier=2,
                      activation='relu',
                      norm='instance',
                      recompute=False):
    """This function validates models saved at different epochs of the same training.
    All models are assumed to be in the same folder.
    The results of each model are saved in a subfolder in validation_main_dir.
    :param image_dir: path of the folder with validation images. There is no ground truth folder: the
    truth is each image's own header.
    :param models_dir: path of the folder with the models to validate.
    :param validation_main_dir: path of the folder where all the models validation subfolders will be saved.
    :param step_eval: (optional) If step_eval > 1 skips models when validating, by validating on models step_eval apart.
    :param cropping: (optional) the window the network sees, cropped around the centre of the volume and padded up to
    the same size when the head is smaller. Default is 160, the shape training cropped to. It is the only window knob.
    :param target_res: (optional) resolution the images are resampled to before anything else, and the grid the
    deficit is read against. Default is 1mm, the grid training left its degraded volumes on. None hands the
    network a domain it never saw.
    :param minmax_norm: (optional) normalise with an exact min-max instead of predict.py's p0.5-p99.5. Default is
    False, i.e. the percentile predict_tm and SynthSeg deploy with.
    :param n_levels: (optional) number of levels of the encoder. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutional layers per level. Default is 3.
    :param conv_size: (optional) size of the convolution kernels. Default is 5.
    :param unet_feat_count: (optional) number of feature maps for the first level. Default is 24.
    :param feat_multiplier: (optional) multiply the number of feature by this number at each new level. Default is 2.
    :param activation: (optional) activation function. Default is 'relu'.
    :param norm: (optional) the normalisation the checkpoints were trained with, among 'instance', 'batch' and
    'none'. It is an architecture argument: a mismatch is refused by load_weights_checked rather than silently
    loaded. Default is 'instance'.
    :param recompute: (optional) whether to recompute result files even if they already exist."""

    # create result folder
    utils.mkdir(validation_main_dir)

    # loop over models
    list_models = utils.list_files(models_dir, expr=['rs', '.h5'], cond_type='and')[::step_eval]
    loop_info = utils.LoopInfo(len(list_models), 1, 'validating', True)
    for model_idx, path_model in enumerate(list_models):

        # build names and create folders
        model_val_dir = os.path.join(validation_main_dir, os.path.basename(path_model).replace('.h5', ''))
        score_path = os.path.join(model_val_dir, 'rs_results.csv')
        utils.mkdir(model_val_dir)

        if (not os.path.isfile(score_path)) | recompute:
            loop_info.update(model_idx)
            predict_rs(path_images=image_dir,
                       path_out=score_path,
                       path_model=path_model,
                       cropping=cropping,
                       target_res=target_res,
                       minmax_norm=minmax_norm,
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
    """One column of a predict_rs csv, as a float array.

    Every column the file holds is available by name: s_R/A/S the predicted spacings, true_R/A/S the
    header's, err_R/A/S the per-axis absolute errors, and abs_err their mean. ras_axes is the
    permutation predict_rs applied to that image's header; it is not numeric and asking for it raises.
    Two derived columns are understood on top:

      err_max    the largest of the three per-axis errors, i.e. the worst axis of that image
      err_worst  the error on the axis whose true spacing is coarsest, i.e. the axis a QC call is
                 usually about. Not the same as err_max, and the gap between the two is informative.

    :param subjects: (optional) keep only these rows. Either a list of subject stems or a callable on
    the stem. This is how a resolution band is cut out of a set whose coarse images dominate the mean."""
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

    # derived columns
    err_cols = ['err_%s' % a for a in AXES]
    true_cols = ['true_%s' % a for a in AXES]
    blank = lambda r, *ks: any(r.get(k) in (None, '') for k in ks)
    if column == 'err_max':
        return np.array([max(float(r[k]) for k in err_cols) for r in rows if not blank(r, *err_cols)])
    if column == 'err_worst':
        out = []
        for r in rows:
            if blank(r, *(err_cols + true_cols)):
                continue
            j = int(np.argmax([float(r[k]) for k in true_cols]))
            out.append(float(r[err_cols[j]]))
        return np.array(out)
    return np.array([])


def plot_validation_curves(list_validation_dirs, architecture_names=None, column='abs_err',
                           size_max_circle=100, figsize=(11, 6), y_lim=None, fontsize=18,
                           list_linestyles=None, list_colours=None, plot_legend=False,
                           subjects=None, title='Validation curves', ylabel=None):
    """This function plots the validation curves of several trainings, based on the results of validate_training().
    It takes as input a list of validation folders (one for each training), each containing subfolders with the
    per-image scores of the corresponding validated epoch.
    :param list_validation_dirs: list of all the validation folders of the trainings to plot.
    :param architecture_names: (optional) list of the names of the trainings. Default is the parent folder name.
    :param column: (optional) which column of the predict_rs csv to average. Default is 'abs_err', the mean
    absolute error in millimetres over the three axes. See read_scores for the derived ones.
    :param size_max_circle: (optional) size of the marker for epochs achieving the best validation scores.
    :param figsize: (optional) size of the figure to draw.
    :param fontsize: (optional) fontsize used for the graph.
    :param subjects: (optional) restrict every curve to these subject stems, see read_scores. This is how a
    resolution band is plotted on its own.
    :param title: (optional) title of the figure.
    :param ylabel: (optional) label of the y axis. Default is the column name."""

    n_curves = len(list_validation_dirs)

    # reformat model names
    if architecture_names is None:
        architecture_names = [os.path.basename(os.path.dirname(d)) for d in list_validation_dirs]
    else:
        architecture_names = utils.reformat_to_list(architecture_names, len(list_validation_dirs))

    # prepare legend labels
    if plot_legend is False:
        list_legend_labels = ['_nolegend_'] * n_curves
    else:
        list_legend_labels = architecture_names
        if plot_legend is not True:
            list_legend_labels = ['_nolegend_' if i >= plot_legend else list_legend_labels[i] for i in range(n_curves)]

    # prepare linestyles and colours
    list_linestyles = utils.reformat_to_list(list_linestyles) if list_linestyles is not None else [None] * n_curves
    list_colours = utils.reformat_to_list(list_colours) if list_colours is not None else [None] * n_curves

    # loop over architectures
    plt.figure(figsize=figsize)
    for net_val_dir, net_name, linestyle, colour, legend_label in zip(list_validation_dirs, architecture_names,
                                                                     list_linestyles, list_colours,
                                                                     list_legend_labels):

        list_epochs_dir = utils.list_subfolders(net_val_dir, whole_path=False)

        # loop over epochs
        list_net_scores = list()
        list_epochs = list()
        for epoch_dir in list_epochs_dir:
            path_epoch_scores = utils.list_files(os.path.join(net_val_dir, epoch_dir), expr='rs_results')
            if len(path_epoch_scores) == 1:
                scores = read_scores(path_epoch_scores[0], column, subjects=subjects)
                if scores.size:
                    list_net_scores.append(np.mean(np.abs(scores)))
                    list_epochs.append(int(re.sub('[^0-9]', '', epoch_dir)))

        # plot validation scores for current architecture
        if list_net_scores:  # check that archi has been validated for at least 1 epoch
            list_net_scores = np.array(list_net_scores)
            list_epochs = np.array(list_epochs)
            list_epochs, idx = np.unique(list_epochs, return_index=True)
            list_net_scores = list_net_scores[idx]
            min_score = np.min(list_net_scores)
            epoch_min_score = list_epochs[np.argmin(list_net_scores)]
            print('\n' + net_name)
            print('epochs validated: %d' % len(list_epochs))
            print('epoch min score: %d' % epoch_min_score)
            print('min score: %0.4f' % min_score)
            plt.plot(list_epochs, list_net_scores, label=legend_label, linestyle=linestyle, color=colour)
            plt.scatter(epoch_min_score, min_score, s=size_max_circle, color=colour)

    # finalise plot
    plt.grid()
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.ylabel(column if ylabel is None else ylabel, fontsize=fontsize)
    plt.xlabel('Epochs', fontsize=fontsize)
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1] + 0.01)  # set right/left limits of plot
    plt.title(title, fontsize=fontsize)
    if plot_legend:
        plt.legend(fontsize=fontsize)
    plt.tight_layout(pad=1)
    plt.show()
