"""

Trains a scalar regressor to read the realised bias-field severity of the synthetic image. It is the
per-tissue mean regressor (SynthSeg/training_tissue_means.py) with the target swapped: three tissue means
become one severity scalar, and nothing else changes.

the severity is std_log = std(log B) over the whole image (all voxels of the crop, no brain mask), the
standard deviation of the log bias field. it is computed inside the generation graph by labels_to_image_model
(return_bias_std=True with bias_std_masked=False exposes the named layer 'bias_field_std' = _whole_std(log_bias)),
so it is the severity of the field that was actually drawn and applied to this image, not the knob that
generated it: two images at the same bias_field_std knob get different std_log, and each is labelled by its
own. this is the same move the tissue-means net makes when it regresses the effective mean of the image rather
than the drawn gaussian mean. the whole-image (mask-free) measure is the one a QC head can reproduce at
deployment with no segmentation, and it is defined even on a crop that misses the brain.

the target is normalised to [0, 1] by a fixed ceiling std_log_max (the largest plausible severity: with
bias_field_std = 1 the realised std_log tops out around 0.65), so
the loss and the read-off sit on the tissue-means net's scale: label = clip(std_log / std_log_max, 0, 1).

a fraction (1 - bias_prob) of the images get no bias field at all, so label 0 is a value the target really
takes and the net is asked to certify clean, not only to rank severity. that fraction is honest only because
the log-bias field is zeroed on the same draw that skips it (the fix in ext/lab2im/layers.py, the
return_field branch of BiasFieldCorruption): otherwise a clean image would carry the (unapplied) sampled
field as its label and the net would learn that a bias-free image is severe.

it departs from the tissue-means net in exactly three places, all forced by the target:
  1. the generator returns the bias std (return_bias_std=True) and applies the bias with probability
     bias_prob < 1; the tissue-means generator does neither.
  2. the target is one scalar read off the generator, not three per-tissue masked pools of the image.
  3. the loss is a plain mse (every image has a severity), not a present-gated one (a tissue can be absent
     from a crop; a bias severity never is).
the encoder, the head (max pool, two convolutions, spatial mean; the last convolution linear as it must be
for a target that really reaches 0), the checkpoint guard, the ValLoss that reads the net on each image's
own statistics, and the training loop are all imported from the tissue-means module so the two nets cannot
drift apart. instance norm is the deployable default here (train == validation == deployment).

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
# callback and the training loop are the tissue-means ones, imported rather than copied so the two nets can
# never drift apart. build_regression_model with k=1 is that head with a one-channel output.
from SynthSeg.training_tissue_means import (build_regression_model, load_weights_checked, ValLoss,
                                            train_model)

eps = 1e-6


def training(labels_dir,
             model_dir,
             generation_labels,
             generation_classes,
             std_log_max=0.65,
             bias_field_std=1.0,
             bias_prob=0.9,
             bias_scale=.025,
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
             randomise_res=False,
             max_res_iso=4.,
             max_res_aniso=8.,
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
    :param generation_classes: path to the 1d array grouping the labels that share one drawn gaussian. Unlike
    the tissue-means net, the target here does not depend on this grouping (it is a property of the bias
    field, not of the tissues), so the standard SynthSeg classes are used for the richest contrast
    randomisation rather than the 3-tissue grouping.

    # bias field: the target and the only intensity corruption on by default
    :param std_log_max: (optional) ceiling that normalises the target to [0, 1]: label = clip(std_log /
    std_log_max, 0, 1). The largest plausible realised severity. Default 0.65: at bias_field_std = 1 the
    realised std_log averages ~0.58 and reaches ~0.68 at the top of the field-pattern band. A train-time
    knob, not baked into the graph.
    :param bias_field_std: (optional) max std of the normal the small bias tensor is sampled from; the layer
    draws sigma ~ U(0, bias_field_std) per image. Default 1.0. Set to 0 and there is no target.
    :param bias_prob: (optional) probability of applying the sampled field; the rest are left bias-free with
    target 0. Default 0.9 (~10% clean, so certifying clean is part of the task).
    :param bias_scale: (optional) ratio between the label map size and the small sampled bias tensor (its
    smoothness). Default 0.025, the SynthSeg default.

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

    # other intensity corruption (off by default; the bias field is the point)
    :param randomise_res: (optional) simulate a random acquisition resolution. Default is False.
    :param max_res_iso: (optional) max isotropic resolution. Default 4.
    :param max_res_aniso: (optional) max anisotropic resolution. Default 8.
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
    assert bias_field_std > 0, 'bias_field_std must be > 0, it is the target of this regressor'
    assert std_log_max > 0, 'std_log_max normalises the target, it must be > 0'

    # hold a few maps out of training. deterministic split on the sorted paths, so training and the eval
    # scripts agree on which maps were never seen.
    labels_paths = sorted(utils.list_images_in_folder(labels_dir))
    n_hold = min(max(holdout, 0), len(labels_paths) - 1)
    train_paths = labels_paths[:len(labels_paths) - n_hold] if n_hold else labels_paths
    val_paths = labels_paths[len(labels_paths) - n_hold:] if n_hold else []

    # a resumed job draws a fresh stream but stays reproducible
    init_epoch = 0 if checkpoint is None else int(os.path.basename(checkpoint).split('bf_')[1][:-3])
    np.random.seed(seed + init_epoch)
    tf.random.set_seed(seed + init_epoch)

    # generation model: outputs[0] image, outputs[1] labels, outputs[2] the realised bias std (a scalar)
    labels_shape, _, _, _, _, atlas_res = utils.get_volume_info(train_paths[0], aff_ref=np.eye(4))
    generator = build_generator(labels_shape, atlas_res, gen_labels, output_shape, 2 ** n_levels,
                                n_neutral_labels, scaling_bounds, rotation_bounds, shearing_bounds,
                                translation_bounds, nonlin_std, nonlin_scale, randomise_res, max_res_iso,
                                max_res_aniso, bias_field_std, bias_prob, bias_scale, gamma_std, clip, flipping)
    image_shape = generator.outputs[0].get_shape().as_list()[1:]

    # target and prediction. k=1: one scalar. the head is the tissue-means head with a one-channel output.
    y_pred = build_regression_model(generator, image_shape, 1, n_levels, nb_conv_per_level, conv_size,
                                    unet_feat_count, feat_multiplier, activation, batch_norm, use_residuals,
                                    instance_norm)
    y_true = build_target(generator, std_log_max)
    # the tissue-means ValLoss/probe carry a per-tissue 'present' count that gates its loss; there is no such
    # gate here (a severity is always present), so a constant-ones stand-in keeps the probe signature and the
    # saved npz shape identical without changing anything.
    present = KL.Lambda(lambda s: K.ones_like(s), name='bf_present')(y_true)
    loss = build_loss(y_true, y_pred)
    regression_model = models.Model(generator.inputs, loss)

    # a second read of the same graph (same layer objects, same weights) exposing the prediction, the target
    # and the loss, so ValLoss can report the graph's own mse and keep pred/true per epoch.
    val_probe = models.Model(generator.inputs, [y_pred, y_true, present, loss])
    n_train = int(np.sum([K.count_params(w) for w in regression_model.trainable_weights]))
    print('regressing bias severity  std_log / %.3f in [0, 1]   trainable params: %d' % (std_log_max, n_train))

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
    print('  bias_field_std %.3f   bias_prob %.2f (~%.0f%% clean)   std_log_max %.3f'
          % (bias_field_std, bias_prob, 100 * (1 - bias_prob), std_log_max))

    # 'bias' is the one channel's name in the printed lines and the saved npz; prefix 'bf' names the
    # checkpoints bf_###.h5 (the tissue-means loop defaults to tm_###.h5). The model_dir already keeps the
    # two experiments apart, but the filename prefix labels a checkpoint even out of its folder.
    train_model(regression_model, input_generator, lr, epochs, steps_per_epoch, model_dir, checkpoint,
                init_epoch, clipnorm, val_generator, n_val, val_probe, ['bias'], prefix='bf')


def build_generator(labels_shape, atlas_res, generation_labels, output_shape, output_div_by_n,
                    n_neutral_labels, scaling_bounds, rotation_bounds, shearing_bounds, translation_bounds,
                    nonlin_std, nonlin_scale, randomise_res, max_res_iso, max_res_aniso, bias_field_std,
                    bias_prob, bias_scale, gamma_std, clip, flipping=True):

    # return_bias_std=True adds the scalar severity outputs[2]; bias_prob < 1 leaves a fraction of images
    # bias-free (target 0). output_labels = generation_labels so the mask survives on outputs[1] and the
    # severity is measured over the same brain the net sees (the crop to output_shape happens before the
    # bias, so image, labels and field are all at output_shape).
    return labels_to_image_model(labels_shape=labels_shape, n_channels=1,
                                 generation_labels=generation_labels, output_labels=generation_labels,
                                 n_neutral_labels=n_neutral_labels, atlas_res=atlas_res, target_res=None,
                                 output_shape=output_shape, output_div_by_n=output_div_by_n,
                                 flipping=flipping, aff=np.eye(4),
                                 scaling_bounds=scaling_bounds, rotation_bounds=rotation_bounds,
                                 shearing_bounds=shearing_bounds, translation_bounds=translation_bounds,
                                 nonlin_std=nonlin_std, nonlin_scale=nonlin_scale,
                                 randomise_res=randomise_res, max_res_iso=max_res_iso,
                                 max_res_aniso=max_res_aniso, bias_field_std=bias_field_std,
                                 bias_scale=bias_scale, bias_prob=bias_prob, bias_std_masked=False,
                                 intensity_gamma_std=gamma_std, intensity_clip=clip,
                                 return_bias_std=True, return_resolution=False)


def build_target(generator, std_log_max):

    # the realised severity std_log = std(log B) over the whole image, exposed by the generator as its third
    # output ('bias_field_std', shape [batch, 1]), normalised to [0, 1] by the plausibility ceiling. clip so
    # the rare tail past the ceiling saturates at 1 rather than pushing the target out of range. because the
    # std is mask-free (bias_std_masked=False), target 0 means genuinely no bias field was applied (the
    # bias_prob clean fraction); a crop that misses the brain is not a degenerate 0 (that was the _masked_std
    # failure mode this measure avoids).
    std_log = generator.outputs[2]
    return KL.Lambda(lambda s: K.clip(s / std_log_max, 0., 1.), name='bf_target')(std_log)


def build_loss(y_true, y_pred):

    # plain mse; unlike the tissue-means net there is no present-gating (every image has a severity).
    def fn(x):
        yt, yp = x
        return K.expand_dims(K.mean(K.square(yt - yp), axis=1), -1)
    loss = KL.Lambda(fn, name='bf_loss')([y_true, y_pred])
    loss._keras_shape = tuple(loss.get_shape().as_list())
    return loss
