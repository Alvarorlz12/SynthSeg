"""

Validate the tissue-means regressor across the epochs of one training, on real images.

Same shape as validate.py and validate_qc.py: loop over the checkpoints of a folder, run the matching
predict_* on a fixed validation set, and drop one result file per epoch in its own subfolder, so a run
that is interrupted resumes by skipping what is already on disk.

What differs, and only because the underlying prediction differs: predict_tm writes a csv rather than a
.npy, so the score file here is tm_results.csv and the curve is read back through csv.DictReader. Named
columns are also what lets every other score -- per-tissue MAE, the three pooled, the deliverable -- come
out of these same files afterwards without re-running anything.

A tissue that fell below min_vox in the crop is written as an empty cell rather than a zero, and a reader
that parsed the file as a float matrix would turn that into a real 0.0 and pull the mean down. That gate
does not fire on a whole brain: the smallest of the three groups ever measured was 87210 voxels, so an
empty cell means the crop missed the head, which is a finding rather than a number to average over.

The score plotted is the mean absolute error on the deliverable, the GM-WM contrast, which is the column
predict_tm calls abs_err. That is what makes the FLAIR rows usable: the sequence nulls CSF by design, so
a FLAIR CSF mean is a real measurement of a different physical quantity than a T1w CSF mean, while GM and
WM carry over and the deliverable is built from those two.

The FLAIR CSF cell is NOT blank. The mask is anatomical and full of voxels; only the intensity means
something else. So a per-CSF curve has to drop those rows deliberately, on the stem, rather than expect
them to be missing.

WHAT THIS CURVE IS FOR. Ten images is far below this project's error bar for an absolute number. The
curve is PAIRED -- the same ten images, against the same anchor, at every epoch -- so the between-image
variance and the anchor's own bias are common to every point and cancel in the comparison. It ranks
epochs. It does not measure performance, and no headline should come out of it.

Usage, from a script or a notebook:

    from QC import validate_tm
    validate_tm.validate_training(image_dir='<data>/qc-data/validation/img',
                                  gt_dir='<data>/qc-data/validation/gt/ss/segs',
                                  models_dir='models/contrast/tm_e2_instance_artefacts',
                                  validation_main_dir='<data>/qc-data/validation/scores/tm_e2')
    validate_tm.plot_validation_curves(['.../scores/tm_e1b', '.../scores/tm_e2', '.../scores/tm_e3'])

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
from QC.predict_tm import predict_tm

# third-party imports
from ext.lab2im import utils


def validate_training(image_dir,
                      gt_dir,
                      models_dir,
                      validation_main_dir,
                      tissues=None,
                      step_eval=1,
                      cropping=160,
                      target_res=1.,
                      n_levels=5,
                      nb_conv_per_level=3,
                      conv_size=5,
                      unet_feat_count=24,
                      feat_multiplier=2,
                      activation='relu',
                      norm='instance',
                      min_vox=8,
                      recompute=False):
    """This function validates models saved at different epochs of the same training.
    All models are assumed to be in the same folder.
    The results of each model are saved in a subfolder in validation_main_dir.
    :param image_dir: path of the folder with validation images.
    :param gt_dir: path of the folder with ground truth label maps.
    These are matched to the validation images by sorting order.
    :param models_dir: path of the folder with the models to validate.
    :param validation_main_dir: path of the folder where all the models validation subfolders will be saved.
    :param tissues: (optional) comma separated groups the checkpoints were trained with, since that fixes
    the width of the output. Default is None: every key of QC.training_tm.tissue_groups, in its order.
    :param step_eval: (optional) If step_eval > 1 skips models when validating, by validating on models step_eval apart.
    :param cropping: (optional) the window the network sees, cropped around the centre of the volume and padded up to
    the same size when the head is smaller. Default is 160, the shape training cropped to. It is the only window knob.
    :param target_res: (optional) resolution the images are resampled to before anything else. This must match the
    resolution of the training label maps. Set to None to disable the resampling. Default is 1mm.
    :param n_levels: (optional) number of levels of the encoder. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutional layers per level. Default is 3.
    :param conv_size: (optional) size of the convolution kernels. Default is 5.
    :param unet_feat_count: (optional) number of feature maps for the first level. Default is 24.
    :param feat_multiplier: (optional) multiply the number of feature by this number at each new level. Default is 2.
    :param activation: (optional) activation function. Default is 'relu'.
    :param norm: (optional) the normalisation the checkpoints were TRAINED with, among 'instance', 'batch' and 'none'.
    It is an architecture argument, not a detail: the wrong one is refused by load_weights_checked rather than
    silently loaded. Default is 'instance'.
    :param min_vox: (optional) a tissue with fewer voxels than this in the crop is left blank in the ground truth
    columns rather than averaged over nothing. Default is 8, training's own gate.
    :param recompute: (optional) whether to recompute result files even if they already exist."""

    # create result folder
    utils.mkdir(validation_main_dir)

    # loop over models
    list_models = utils.list_files(models_dir, expr=['tm', '.h5'], cond_type='and')[::step_eval]
    loop_info = utils.LoopInfo(len(list_models), 1, 'validating', True)
    for model_idx, path_model in enumerate(list_models):

        # build names and create folders
        model_val_dir = os.path.join(validation_main_dir, os.path.basename(path_model).replace('.h5', ''))
        score_path = os.path.join(model_val_dir, 'tm_results.csv')
        utils.mkdir(model_val_dir)

        if (not os.path.isfile(score_path)) | recompute:
            loop_info.update(model_idx)
            predict_tm(path_images=image_dir,
                       path_out=score_path,
                       path_model=path_model,
                       tissues=tissues,
                       gt_folder=gt_dir,
                       cropping=cropping,
                       target_res=target_res,
                       n_levels=n_levels,
                       nb_conv_per_level=nb_conv_per_level,
                       conv_size=conv_size,
                       unet_feat_count=unet_feat_count,
                       feat_multiplier=feat_multiplier,
                       activation=activation,
                       norm=norm,
                       min_vox=min_vox,
                       recompute=True,
                       verbose=False)


def read_scores(path_csv, column='abs_err', subjects=None):
    """One column of a predict_tm csv, as a float array, with the empty cells dropped.

    Empty is not zero. predict_tm leaves a cell blank when a tissue fell below min_vox in the crop, and
    parsing the file as a float matrix would turn that into a real 0.0 and drag the mean down. On a whole
    brain that gate does not fire, so a blank here is the crop having missed the head -- better seen as a
    short array than as a depressed mean.

    Besides the columns the csv holds, four DERIVED error columns are understood, because predict_tm
    writes predictions and truths and not their difference:

      err_<tissue>  |<tissue> - true_<tissue>|, the absolute error on one tissue mean
      err_mean      the mean of those over the tissues present in the file

    'abs_err' stays what predict_tm wrote: the error on the normalised difference |GM-WM|/(GM+WM).
    That one is NOT err_GM and err_WM combined, and it is not an average of them either: the ratio is
    invariant to a scale error that both tissue columns would be charged for.

    :param subjects: (optional) keep only these rows. Either a list of subject stems or a callable on
    the stem. The set mixes T1w and FLAIR, and on FLAIR the CSF mean and the GM-WM difference are not
    the same physical quantity as on T1w, so averaging the two is a real error and this is how a
    figure is restricted to one modality."""
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
    blank = lambda r, *ks: any(r.get(k) in (None, '') for k in ks)
    if column.startswith('err_') and column != 'err_mean':
        t = column[4:]
        if t not in rows[0] or ('true_%s' % t) not in rows[0]:
            return np.array([])
        return np.array([abs(float(r[t]) - float(r['true_%s' % t]))
                         for r in rows if not blank(r, t, 'true_%s' % t)])
    if column == 'err_mean':
        tissues = [k for k in rows[0] if ('true_%s' % k) in rows[0] and k != 'contrast']
        if not tissues:
            return np.array([])
        cols = tissues + ['true_%s' % t for t in tissues]
        return np.array([np.mean([abs(float(r[t]) - float(r['true_%s' % t])) for t in tissues])
                         for r in rows if not blank(r, *cols)])
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
    :param column: (optional) which column of the predict_tm csv to average. Default is 'abs_err', the absolute error
    on the deliverable. 'contrast' plots the prediction itself, which is a sanity check and not a score.
    :param size_max_circle: (optional) size of the marker for epochs achieving the best validation scores.
    :param figsize: (optional) size of the figure to draw.
    :param fontsize: (optional) fontsize used for the graph.
    :param subjects: (optional) restrict every curve to these subject stems, see read_scores.
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
            path_epoch_scores = utils.list_files(os.path.join(net_val_dir, epoch_dir), expr='tm_results')
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
