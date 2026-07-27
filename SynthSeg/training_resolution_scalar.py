"""

Trains a per-axis regressor to read the realised voxel spacing (the effective resolution, in mm/axis) of
the synthetic image. It is the bias-field severity regressor (SynthSeg/training_biasfield_scalar.py) with
the target swapped: one severity scalar becomes three per-axis spacings, and nothing else changes.

the target is `resolution`, the per-axis spacing SynthSeg's randomise_res path degraded the content to. the
generation graph draws it (SampleResolution), blurs with it, downsamples the volume to that spacing and
resamples it back up to the 160^3 1mm grid, so the array shape never leaks the label: the file still has
1mm voxels, only its content is coarser. labels_to_image_model exposes the drawn value as the named layer
'resolution' (return_resolution=True), exactly as it exposes 'bias_field_std', so this is the realised
degradation of this image, not a knob that generated a distribution of them.

the target is the resolution deficit relative to the grid the image is stored on, in log:

    y_k = log(s_k / atlas_res_k) / log(max_res / atlas_res_k),  clipped to [0, 1]

log because blur is multiplicative (1 -> 2 mm is the same amount of degradation as 2 -> 4, and a linear
target would spend the whole loss on the 4-8 mm end nobody makes decisions about); normalised to [0, 1] so
the loss and the read-off sit on the tissue-means net's scale. this is a monotone relabelling and ValLoss
saves pred+true raw, so mm, log-MAE or anything else is recomputed a posteriori from the npz.

three outputs and not one: unlike a bias field, resolution is intrinsically per-axis. a scan can be fine
in-plane and coarse through-plane, and that anisotropy IS the QC case (2D multi-slice, failed recons,
resampled-up FLAIR). k=3 is the same head width the tissue-means net ran at.

the primary artefact is the per-epoch validation pred+true that ValLoss saves in val_%03d.npz, and the
primary metric is the mse against var(target) (the score of predicting the mean). everything else --
per-axis breakdowns, the split between native and degraded axes, calibration -- is computed a posteriori
from those files, with no GPU and no re-run.

it departs from the bias-severity net in exactly three places, all forced by the target:
  1. the generator runs the randomise_res path (randomise_res=True is mandatory, return_resolution asserts
     it) and returns the drawn spacing; the bias field is off by default, so resolution is the only
     corruption.
  2. the target is a 3-vector read off the generator, not one scalar, so the head is k=3.
  3. the image is re-normalised after the degradation (see build_generator). this is what keeps the setup
     the same one that worked for the bias field: there, labels_to_image_model applies the field and then
     min-max normalises, so the net sees an image in [0, 1] with the corruption inside it, exactly like a
     real scan normalised at deployment. for resolution the degradation happens after that normalisation
     and nothing normalises again, so without this line the net would train on images spanning ~[0.12,
     0.74] while every deployed image arrives in [0, 1] -- an accidental difference from the bias setup,
     not a deliberate one. same principle as instance norm: train, validate and deploy the same function.
     (instance norm already absorbs much of a global scale change, so this is cheap insurance rather than
     a large effect.) pass renorm=False to turn it off.
the encoder, the head (max pool, two convolutions, spatial mean; the last convolution linear), the
checkpoint guard, the ValLoss that reads the net on each image's own statistics, and the training loop are
all imported from the tissue-means module so the three regressors cannot drift apart. instance norm is the
deployable default here (train == validation == deployment).

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

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
import os
import numpy as np
import tensorflow as tf
from keras import models
import keras.layers as KL
import keras.backend as K

# project imports
from SynthSeg.labels_to_image_model import labels_to_image_model
from SynthSeg.model_inputs import build_model_inputs

# third-party imports
from ext.lab2im import utils

# the target swap is the only real difference, so the encoder + head, the checkpoint guard, the validation
# callback and the training loop are the tissue-means ones, imported rather than copied so the nets can
# never drift apart. build_regression_model with k=3 is that head with a three-channel output.
from SynthSeg.training_tissue_means import (build_regression_model, load_weights_checked, ValLoss,
                                            train_model)

eps = 1e-6

# one name per output axis, in array order. the label is a per-array-axis vector, not an anatomical one:
# the spatial deformation rotates the anatomy inside the array before the degradation, so the degradation
# axes are always the storage axes. mapping to RAS is a deterministic post-step at deployment, not a thing
# the net is asked to learn.
AXIS_NAMES = ('ax0', 'ax1', 'ax2')


def training(labels_dir,
             model_dir,
             generation_labels,
             generation_classes,
             max_res_iso=4.,
             max_res_aniso=8.,
             grid_ablation=None,
             renorm=True,
             holdout=100,
             batchsize=1,
             output_shape=160,
             flipping=True,
             scaling_bounds=.2,
             rotation_bounds=15,
             shearing_bounds=.012,
             translation_bounds=False,
             nonlin_std=4.,
             nonlin_scale=.04,
             bias_field_std=0.,
             bias_scale=.025,
             gamma_std=0.,
             clip=0,
             n_neutral_labels=18,
             n_levels=5,
             nb_conv_per_level=3,
             conv_size=5,
             unet_feat_count=24,
             feat_multiplier=2,
             activation='relu',
             batch_norm=-1,
             instance_norm=False,
             use_residuals=True,
             lr=1e-4,
             clipnorm=0.,
             epochs=100,
             steps_per_epoch=1000,
             validation_steps=100,
             checkpoint=None,
             seed=0):

    """
    :param labels_dir: path of the folder with the training label maps.
    :param model_dir: path of a directory where the models will be saved during training.
    :param generation_labels: path to the 1d array of all the label values in the label maps.
    :param generation_classes: path to the 1d array grouping the labels that share one drawn gaussian. The
    target does not depend on the tissue grouping (it is a property of the acquisition, not of the tissues),
    so the standard SynthSeg classes are used for the richest contrast randomisation.

    # resolution: the target and the only corruption on by default
    :param max_res_iso: (optional) upper bound of the uniform the isotropic branch draws from, U(atlas_res,
    max_res_iso). Default 4, the SynthSeg default. Also sets the target's normalising ceiling together with
    max_res_aniso.
    :param max_res_aniso: (optional) upper bound of the uniform the anisotropic branch draws from for the
    one axis it selects. Default 8, the SynthSeg default.
    The sampling keeps SampleResolution's stock mixture (prob_min 0.05 -> all axes native,
    prob_iso 0.10 -> one shared value, else 0.85 -> one random axis degraded and the other two native). It
    is spiked and axis-coupled: 62% of per-axis targets sit exactly at atlas_res and 0% of images have
    exactly two degraded axes.
    :param grid_ablation: (optional) None keeps the stock resampling. 'blur_only' skips MimicAcquisition so
    only the Gaussian blur cue survives; 'kernel_phase' / 'kernel_random' randomise the resample kernel and
    sub-voxel grid phase. Default None (stock SynthSeg behaviour).
    :param renorm: (optional) re-apply a per-image min-max after the degradation, so the encoder sees an
    image in [0, 1] with the corruption inside it -- the same thing the bias run's net saw, and the same
    thing a real scan normalised at deployment is. Default True. See the module docstring.

    :param holdout: (optional) number of label maps kept out of training, the anatomy the validation loss is
    measured on. Deterministic split on the sorted paths, so the eval scripts agree on which maps were never
    seen: change it in both. Default 100.
    :param batchsize: (optional) images per minibatch. Default 1.
    :param output_shape: (optional) shape of the cropped output image. Default 160.

    # spatial deformation (the dice qc net's defaults; pass False / 0 to turn a term off)
    :param flipping: (optional) random right/left flip. Default True.
    :param scaling_bounds: (optional) scaling factor bounds. Default 0.2.
    :param rotation_bounds: (optional) rotation angle bounds. Default 15.
    :param shearing_bounds: (optional) shearing bounds. Default 0.012.
    :param translation_bounds: (optional) translation bounds. Default False.
    :param nonlin_std: (optional) std of the elastic deformation field. Default 4.
    :param nonlin_scale: (optional) scale of the elastic deformation field. Default 0.04.

    # other intensity corruption (off by default; the resolution is the point)
    :param bias_field_std: (optional) max std of the bias field. Default 0 (no bias field at all), so the
    only thing that varies beyond contrast and anatomy is the resolution.
    :param bias_scale: (optional) smoothness of the bias field, only used when bias_field_std > 0.
    :param gamma_std: (optional) std of the gamma augmentation. Default 0.
    :param clip: (optional) intensity clipping percentile (0 keeps the exact min-max). Default 0.
    :param n_neutral_labels: (optional) number of non-lateral labels in generation_labels. Default 18.

    # architecture (the dice qc net's, unchanged; n_levels sets the downsampling, not output_shape:
    # conv_enc pools after every level but the last, so the encoder divides the volume by 2 ** (n_levels - 1))
    :param n_levels: (optional) number of levels of the encoder. Default 5.
    :param nb_conv_per_level: (optional) convolutions per level. Default 3.
    :param conv_size: (optional) size of the convolution kernels. Default 5.
    :param unet_feat_count: (optional) features at the first level. Default 24.
    :param feat_multiplier: (optional) feature multiplier between levels. Default 2.
    :param activation: (optional) activation function. Default 'relu'.
    :param batch_norm: (optional) axis to batch normalise, or None to turn it off. -1 is the feature axis.
    Default -1. Only used when instance_norm is False.
    :param instance_norm: (optional) per-image normalisation in both train and inference, the deployable
    choice under randomised contrast (train == validation == deployment). Default False; the launcher's
    --norm sets it and defaults it on for this experiment.
    :param use_residuals: (optional) residual connection per level. Default True.

    # training
    :param lr: (optional) learning rate. Default 1e-4.
    :param clipnorm: (optional) gradient norm clipping, 0 to turn it off. Default 0.
    :param epochs: (optional) number of epochs. Default 100.
    :param steps_per_epoch: (optional) steps per epoch (how often the model is saved). Default 1000.
    :param validation_steps: (optional) images drawn from the held-out maps at each epoch end for a val_loss.
    0 turns it off. Default 100.
    :param checkpoint: (optional) path of a saved model to resume from.
    :param seed: (optional) random seed. Default 0.
    """

    # prepare labels
    gen_labels = np.asarray(utils.load_array_if_path(generation_labels)).astype('int32')
    gen_classes = np.asarray(utils.load_array_if_path(generation_classes)).astype('int32')
    assert max(max_res_iso, max_res_aniso) > 0, 'the resolution range is the target of this regressor'

    # hold a few maps out of training. deterministic split on the sorted paths, so training and the eval
    # scripts agree on which maps were never seen.
    labels_paths = sorted(utils.list_images_in_folder(labels_dir))
    n_hold = min(max(holdout, 0), len(labels_paths) - 1)
    train_paths = labels_paths[:len(labels_paths) - n_hold] if n_hold else labels_paths
    val_paths = labels_paths[len(labels_paths) - n_hold:] if n_hold else []

    # a resumed job draws a fresh stream but stays reproducible
    init_epoch = 0 if checkpoint is None else int(os.path.basename(checkpoint).split('rs_')[1][:-3])
    np.random.seed(seed + init_epoch)
    tf.random.set_seed(seed + init_epoch)

    # generation model: outputs[0] image, outputs[1] labels, outputs[2] the drawn per-axis spacing [B, 3].
    # NB the index only holds because the bias std is not returned (labels_to_image_model appends
    # 'bias_field_std' before 'resolution'); build_generator pins return_bias_std=False for that reason.
    labels_shape, _, _, _, _, atlas_res = utils.get_volume_info(train_paths[0], aff_ref=np.eye(4))
    generator = build_generator(labels_shape, atlas_res, gen_labels, output_shape, 2 ** n_levels,
                                n_neutral_labels, scaling_bounds, rotation_bounds, shearing_bounds,
                                translation_bounds, nonlin_std, nonlin_scale, max_res_iso, max_res_aniso,
                                grid_ablation, renorm, bias_field_std, bias_scale, gamma_std, clip, flipping)
    image_shape = generator.outputs[0].get_shape().as_list()[1:]

    # target and prediction. k=3: one spacing per array axis. the head is the tissue-means head, unchanged.
    y_pred = build_regression_model(generator, image_shape, 3, n_levels, nb_conv_per_level, conv_size,
                                    unet_feat_count, feat_multiplier, activation, batch_norm, use_residuals,
                                    instance_norm)
    res_max = float(max(max_res_iso, max_res_aniso))
    y_true = build_target(generator, atlas_res, res_max)
    # the tissue-means ValLoss/probe carry a per-tissue 'present' count that gates its loss; there is no such
    # gate here (an axis always has a resolution), so a constant-ones stand-in keeps the probe signature and
    # the saved npz shape identical without changing anything.
    present = KL.Lambda(lambda s: K.ones_like(s), name='rs_present')(y_true)
    loss = build_loss(y_true, y_pred)
    regression_model = models.Model(generator.inputs, loss)

    # a second read of the same graph (same layer objects, same weights) exposing the prediction, the target
    # and the loss, so ValLoss can report the graph's own mse and keep pred/true per epoch.
    val_probe = models.Model(generator.inputs, [y_pred, y_true, present, loss])
    n_train = int(np.sum([K.count_params(w) for w in regression_model.trainable_weights]))
    print('regressing per-axis spacing  log(s / %s) / log(%.1f) in [0, 1]   trainable params: %d'
          % (np.array(atlas_res), res_max, n_train))

    # input generators. the held-out maps feed a val_loss at each epoch end: every image is drawn fresh, so
    # the loss is already out of sample in contrast and the held-out maps only add unseen anatomy.
    def make_generator(paths):
        model_inputs = build_model_inputs(path_label_maps=paths, n_labels=len(gen_labels),
                                          batchsize=batchsize, n_channels=1,
                                          generation_classes=gen_classes, prior_distributions='uniform')
        return utils.build_training_generator(model_inputs, batchsize)

    input_generator = make_generator(train_paths)
    n_val = validation_steps if (val_paths and validation_steps > 0) else 0
    val_generator = make_generator(val_paths) if n_val else None
    print('  label maps: %d for training, %d held out   validation steps: %d' %
          (len(train_paths), len(val_paths), n_val))
    print('  max_res_iso %.2f   max_res_aniso %.2f   grid_ablation %s   renorm %s   bias_field_std %.2f'
          % (max_res_iso, max_res_aniso, grid_ablation or 'none', renorm, bias_field_std))
    print('  sampling: SampleResolution stock mixture (prob_min .05 / prob_iso .10 / single-aniso-axis .85);'
          ' ~62% of per-axis targets sit exactly at atlas_res.')

    # 'ax0/ax1/ax2' name the three channels in the printed lines and the saved npz; prefix 'rs' names the
    # checkpoints rs_###.h5 (tissue-means defaults to tm_###.h5, bias uses bf_###.h5).
    train_model(regression_model, input_generator, lr, epochs, steps_per_epoch, model_dir, checkpoint,
                init_epoch, clipnorm, val_generator, n_val, val_probe, list(AXIS_NAMES), prefix='rs')


def build_generator(labels_shape, atlas_res, generation_labels, output_shape, output_div_by_n,
                    n_neutral_labels, scaling_bounds, rotation_bounds, shearing_bounds, translation_bounds,
                    nonlin_std, nonlin_scale, max_res_iso, max_res_aniso, grid_ablation, renorm,
                    bias_field_std, bias_scale, gamma_std, clip, flipping=True):

    # randomise_res=True is what draws the spacing at all, and return_resolution asserts it. return_bias_std
    # is pinned False so 'resolution' lands at outputs[2] (labels_to_image_model appends 'bias_field_std'
    # first when it is on, which would silently hand the k=3 head a [B, 1] target).
    gen = labels_to_image_model(labels_shape=labels_shape, n_channels=1,
                               generation_labels=generation_labels, output_labels=generation_labels,
                               n_neutral_labels=n_neutral_labels, atlas_res=atlas_res, target_res=None,
                               output_shape=output_shape, output_div_by_n=output_div_by_n,
                               flipping=flipping, aff=np.eye(4),
                               scaling_bounds=scaling_bounds, rotation_bounds=rotation_bounds,
                               shearing_bounds=shearing_bounds, translation_bounds=translation_bounds,
                               nonlin_std=nonlin_std, nonlin_scale=nonlin_scale,
                               randomise_res=True, max_res_iso=max_res_iso, max_res_aniso=max_res_aniso,
                               grid_ablation=grid_ablation,
                               bias_field_std=bias_field_std, bias_scale=bias_scale,
                               intensity_gamma_std=gamma_std, intensity_clip=clip,
                               return_bias_std=False, return_resolution=True)

    if not renorm:
        return gen

    # re-normalise after the degradation. labels_to_image_model runs IntensityAugmentation(normalise=True)
    # Before the blur/resample block, so in training the volume enters the degradation at exactly [0, 1] and
    # leaves it with its range pulled in by an amount monotone in the blur -- a global "how degraded is this"
    # cue that is an artefact of the operation order, not of the acquisition. at deployment a real scan is
    # min-max normalised after it was acquired, so it always arrives at exactly [0, 1] whatever its true
    # resolution, and a net leaning on that cue reads every real scan as native. done here rather than in
    # labels_to_image_model so no library file changes and the other experiments are untouched.
    def _minmax(x):
        axes = list(range(1, len(x.get_shape().as_list())))
        mn = K.min(x, axis=axes, keepdims=True)
        mx = K.max(x, axis=axes, keepdims=True)
        return (x - mn) / K.maximum(mx - mn, K.epsilon())

    image = KL.Lambda(_minmax, name='rs_renorm')(gen.outputs[0])
    return models.Model(gen.inputs, [image] + gen.outputs[1:])


def build_target(generator, atlas_res, res_max):

    # the drawn per-axis spacing, exposed by the generator as its third output ('resolution', shape [B, 3]),
    # turned into a resolution deficit relative to the grid the image is stored on, on a [0, 1] scale:
    #     y_k = log(s_k / atlas_res_k) / log(res_max / atlas_res_k)
    # log because blur is multiplicative: 1 -> 2 mm and 2 -> 4 mm are the same amount of degradation, and a
    # linear target would put most of its dynamic range in the 4-8 mm end where the image is already
    # unusable and nobody makes a QC call. atlas_res -> 0 and res_max -> 1, and the clip only catches
    # floating-point overshoot at the ends (SampleResolution cannot draw outside [atlas_res, res_max]).
    # monotone relabelling: ValLoss saves pred and true raw, so mm / log-MAE / per-bin metrics are all
    # recomputed a posteriori from val_%03d.npz without re-running anything.
    res = generator.outputs[2]
    lo = np.asarray(utils.reformat_to_list(atlas_res, length=3, dtype='float'), dtype='float32')
    denom = np.log(np.maximum(res_max / lo, 1. + eps)).astype('float32')

    def fn(s):
        return K.clip(tf.math.log(K.maximum(s / lo, eps)) / denom, 0., 1.)

    return KL.Lambda(fn, name='rs_target')(res)


def build_loss(y_true, y_pred):

    # plain mse over the three axes, exactly as the bias-severity net: no present-gating (an axis always has
    # a resolution) and no per-axis weighting.
    def fn(x):
        yt, yp = x
        return K.expand_dims(K.mean(K.square(yt - yp), axis=1), -1)
    loss = KL.Lambda(fn, name='rs_loss')([y_true, y_pred])
    loss._keras_shape = tuple(loss.get_shape().as_list())
    return loss
