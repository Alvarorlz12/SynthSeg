"""

Trains a per-axis regressor on the effective resolution of a synthetic scan: the voxel spacing, in mm
per axis, that SynthSeg's randomise_res path degraded the content to.

The generation graph draws the spacing, blurs with it, downsamples the volume and resamples it back up
to the 1 mm grid, so the array shape never carries the label: the file still has 1 mm voxels, only its
content is coarser. labels_to_image_model exposes the drawn value as its 'resolution' output
(return_resolution=True).

The target is the deficit relative to the grid the image is stored on, in millimetres:

    y_k = s_k - atlas_res_k

one-sided: 0 at native, never below the grid, read back as s = atlas_res + pred. Under mse a relative
error is weighted by s^2, so the coarse end carries most of the loss while the QC call is made in
(1, 2] mm.

The spacing is drawn per axis, independently, from a uniform over [atlas_res, max_res], with a fixed
probability of a native (1 mm isotropic) volume. SynthSeg's own sampler couples the axes and is one
flag away (synthseg_sampler), off by default.

k=3, one output channel per array axis: resolution is per axis, and a scan that is fine in-plane and
coarse through-plane (2D multi-slice, resampled-up FLAIR) is the QC case.

The network is the dice qc net's: the same encoder, and a head of a max pool, two k-channel relu
convolutions and a spatial mean. What differs from the bias-severity regressor is the generator, which
runs the randomise_res path (randomise_res=True is mandatory), returns the drawn spacing, and
re-normalises the image after the degradation (see build_generator).

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
from ext.neuron import models as nrn_models

# the checkpoint guard and the training loop are the tissue-means ones. the network is built below
# instead, under names of its own, so that a checkpoint from another head is refused.
from QC.training_tm import load_weights_checked, train_model

eps = 1e-6

# the label is a per-array-axis vector: the deformation rotates the anatomy inside the array before the
# degradation, so the degradation axes are the storage axes and mapping to RAS is a post-step.


def training(labels_dir,
             model_dir,
             generation_labels,
             generation_classes,
             max_res_iso=4.,
             max_res_aniso=8.,
             synthseg_sampler=False,
             res_prob_min=0.2,
             slice_profile='gaussian',
             thickness_min_frac=0.,
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
             checkpoint=None,
             seed=0):

    """
    :param labels_dir: path of the folder with the training label maps.
    :param model_dir: path of a directory where the models will be saved during training.
    :param generation_labels: path to the 1d array of all the label values in the label maps.
    :param generation_classes: path to the 1d array grouping the labels that share one drawn gaussian.

    # resolution: the target, and the only corruption on by default
    :param max_res_iso: (optional) upper bound of the per-axis uniform. Default 4, the SynthSeg default.
    :param max_res_aniso: (optional) upper bound for the axis the stock anisotropic branch selects.
    Default 8, the SynthSeg default. The per-axis sampler draws from the larger of the two bounds.
    :param slice_profile: (optional) shape of the kernel that models the slice profile. 'gaussian' is
    the stock path, whose width doubles as the anti-aliasing filter for the downsampling. 'box' averages
    over the slice thickness and applies no anti-aliasing, which is what a real contiguous acquisition
    does, so the resampling aliases as a real scan does. Default is 'gaussian'.
    :param thickness_min_frac: (optional) lower bound of the slice thickness draw, as a fraction of the
    sampled resolution. 0 is the stock U(atlas_res, resolution); 0.7 covers the range a real gap leaves
    and drops the physically impossible thin slices; 1 forces a contiguous acquisition. Default is 0.
    :param res_prob_min: (optional) probability of drawing the native resolution on every axis, i.e. of a
    1 mm isotropic volume. Default 0.2.
    :param synthseg_sampler: (optional) fall back to SynthSeg's own sampler, which either shares one
    value across the axes or degrades a single axis and pins the other two at atlas_res. Default False,
    i.e. the per-axis uniform, which leaves the three axes independent.
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

    # other intensity corruption (off by default)
    :param bias_field_std: (optional) max std of the bias field. Default 0, i.e. no bias field.
    :param bias_scale: (optional) smoothness of the bias field, only used when bias_field_std > 0.
    :param gamma_std: (optional) std of the gamma augmentation. Default 0.
    :param clip: (optional) intensity clipping percentile (0 keeps the exact min-max). Default 0.
    :param n_neutral_labels: (optional) number of non-lateral labels in generation_labels. Default 18.

    # architecture (the dice qc net's; conv_enc pools after every level but the last, so n_levels
    # divides the volume by 2 ** (n_levels - 1), and output_shape does not)
    :param n_levels: (optional) number of levels of the encoder. Default 5.
    :param nb_conv_per_level: (optional) convolutions per level. Default 3.
    :param conv_size: (optional) size of the convolution kernels. Default 5.
    :param unet_feat_count: (optional) features at the first level. Default 24.
    :param feat_multiplier: (optional) feature multiplier between levels. Default 2.
    :param activation: (optional) activation function. Default 'relu'.
    :param batch_norm: (optional) axis to batch normalise, or None to turn it off. -1 is the feature axis.
    Default -1. Only used when instance_norm is False.
    :param instance_norm: (optional) per-image normalisation, at train and at inference alike. Default
    False; the launcher's --norm sets it.
    :param use_residuals: (optional) residual connection per level. Default True.

    # training
    :param lr: (optional) learning rate. Default 1e-4.
    :param clipnorm: (optional) gradient norm clipping, 0 to turn it off. Default 0.
    :param epochs: (optional) number of epochs. Default 100.
    :param steps_per_epoch: (optional) steps per epoch (how often the model is saved). Default 1000.
    :param checkpoint: (optional) path of a saved model to resume from.
    :param seed: (optional) random seed. Default 0.
    """

    # prepare labels
    gen_labels = np.asarray(utils.load_array_if_path(generation_labels)).astype('int32')
    gen_classes = np.asarray(utils.load_array_if_path(generation_classes)).astype('int32')
    assert max(max_res_iso, max_res_aniso) > 0, 'the resolution range is the target of this regressor'

    # every map in labels_dir trains: labels_dir is already the training partition of a frozen split.
    train_paths = sorted(utils.list_images_in_folder(labels_dir))

    # a resumed job draws a fresh stream but stays reproducible
    init_epoch = 0 if checkpoint is None else int(os.path.basename(checkpoint).split('rs_')[1][:-3])
    np.random.seed(seed + init_epoch)
    tf.random.set_seed(seed + init_epoch)

    # generation model: outputs[0] image, outputs[1] labels, outputs[2] the drawn per-axis spacing [B, 3].
    # that index needs return_bias_std=False, which build_generator pins: labels_to_image_model
    # appends 'bias_field_std' before 'resolution'.
    labels_shape, _, _, _, _, atlas_res = utils.get_volume_info(train_paths[0], aff_ref=np.eye(4))
    generator = build_generator(labels_shape, atlas_res, gen_labels, output_shape, 2 ** n_levels,
                                n_neutral_labels, scaling_bounds, rotation_bounds, shearing_bounds,
                                translation_bounds, nonlin_std, nonlin_scale, max_res_iso, max_res_aniso,
                                bias_field_std, bias_scale, gamma_std, clip, flipping,
                                res_uniform_per_axis=not synthseg_sampler,
                                res_prob_min=res_prob_min,
                                slice_profile=slice_profile,
                                thickness_min_frac=thickness_min_frac)
    image_shape = generator.outputs[0].get_shape().as_list()[1:]

    # target and prediction. k=3: one spacing per array axis.
    y_pred = build_regression_model(generator, image_shape, 3, n_levels, nb_conv_per_level, conv_size,
                                    unet_feat_count, feat_multiplier, activation, batch_norm,
                                    use_residuals, instance_norm)
    res_max = float(max(max_res_iso, max_res_aniso))
    y_true = build_target(generator, atlas_res)
    loss = build_loss(y_true, y_pred)
    regression_model = models.Model(generator.inputs, loss)
    n_train = int(np.sum([K.count_params(w) for w in regression_model.trainable_weights]))

    model_inputs = build_model_inputs(path_label_maps=train_paths, n_labels=len(gen_labels),
                                      batchsize=batchsize, n_channels=1,
                                      generation_classes=gen_classes, prior_distributions='uniform')
    input_generator = utils.build_training_generator(model_inputs, batchsize)

    print('regressing per-axis spacing s - %s in mm   %d label maps   %d params'
          % (np.array(atlas_res), len(train_paths), n_train))
    print('  max_res_iso %.2f  max_res_aniso %.2f  prob_native %.2f  sampler %s  bias_field_std %.2f'
          % (max_res_iso, max_res_aniso, res_prob_min,
             'synthseg' if synthseg_sampler else 'uniform-per-axis', bias_field_std))

    # prefix 'rs' names the checkpoints rs_###.h5.
    train_model(regression_model, input_generator, lr, epochs, steps_per_epoch, model_dir, checkpoint,
                init_epoch, clipnorm, prefix='rs')


def build_generator(labels_shape, atlas_res, generation_labels, output_shape, output_div_by_n,
                    n_neutral_labels, scaling_bounds, rotation_bounds, shearing_bounds, translation_bounds,
                    nonlin_std, nonlin_scale, max_res_iso, max_res_aniso,
                    bias_field_std, bias_scale, gamma_std, clip, flipping=True,
                    res_uniform_per_axis=True, res_prob_min=0.2,
                    slice_profile='gaussian', thickness_min_frac=0.):

    # randomise_res=True is what draws the spacing, and return_resolution asserts it. return_bias_std is
    # pinned False so 'resolution' lands at outputs[2] rather than behind 'bias_field_std'.
    gen = labels_to_image_model(labels_shape=labels_shape, n_channels=1,
                               generation_labels=generation_labels, output_labels=generation_labels,
                               n_neutral_labels=n_neutral_labels, atlas_res=atlas_res, target_res=None,
                               output_shape=output_shape, output_div_by_n=output_div_by_n,
                               flipping=flipping, aff=np.eye(4),
                               scaling_bounds=scaling_bounds, rotation_bounds=rotation_bounds,
                               shearing_bounds=shearing_bounds, translation_bounds=translation_bounds,
                               nonlin_std=nonlin_std, nonlin_scale=nonlin_scale,
                               randomise_res=True, max_res_iso=max_res_iso, max_res_aniso=max_res_aniso,
                               res_uniform_per_axis=res_uniform_per_axis, res_prob_min=res_prob_min,
                               slice_profile=slice_profile,
                               thickness_min_frac=thickness_min_frac,
                               bias_field_std=bias_field_std, bias_scale=bias_scale,
                               intensity_gamma_std=gamma_std, intensity_clip=clip,
                               return_bias_std=False, return_resolution=True)

    # re-normalise after the degradation. labels_to_image_model normalises before the blur/resample block,
    # so the volume leaves it with its range pulled in by an amount monotone in the blur, a cue a real
    # scan never carries. done here rather than in labels_to_image_model so no library file changes.
    def _minmax(x):
        axes = list(range(1, len(x.get_shape().as_list())))
        mn = K.min(x, axis=axes, keepdims=True)
        mx = K.max(x, axis=axes, keepdims=True)
        return (x - mn) / K.maximum(mx - mn, K.epsilon())

    image = KL.Lambda(_minmax, name='rs_renorm')(gen.outputs[0])
    return models.Model(gen.inputs, [image] + gen.outputs[1:])


def build_regression_model(generator, image_shape, k, n_levels, nb_conv_per_level, conv_size, feat_count,
                           feat_multiplier, activation, batch_norm, use_residuals, instance_norm=False):

    # the dice qc net's encoder and head: conv encoder, max pool, two k-channel relu convolutions, average
    # over space. written out here rather than imported so that the layers carry the rs_ prefix.
    enc = nrn_models.conv_enc(input_model=generator, input_shape=image_shape, nb_levels=n_levels,
                              conv_size=conv_size, nb_features=feat_count, feat_mult=feat_multiplier,
                              nb_conv_per_level=nb_conv_per_level, activation=activation,
                              batch_norm=batch_norm, instance_norm=instance_norm,
                              use_residuals=use_residuals, name='rs_enc')
    last = enc.outputs[0]
    conv_kwargs = {'padding': 'same', 'activation': 'relu', 'data_format': 'channels_last'}
    last = KL.MaxPool3D(pool_size=(2, 2, 2), padding='same', name='rs_conv_pool')(last)
    # k channels in both head convs, both relu: the target is >= 0 and is exactly 0 on a native volume,
    # so the relu fits it the way it fits a dice score.
    last = KL.Conv3D(k, kernel_size=5, **conv_kwargs, name='rs_conv0')(last)
    last = KL.Conv3D(k, kernel_size=5, **conv_kwargs, name='rs_conv1')(last)
    return KL.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2, 3]), name='rs_pred')(last)


def build_target(generator, atlas_res):

    # the drawn per-axis spacing (the generator's third output, "resolution", shape [B, 3]) as a deficit
    # relative to the grid the image is stored on: y_k = s_k - atlas_res_k, 0 at native.
    lo = np.asarray(utils.reformat_to_list(atlas_res, length=3, dtype="float"), dtype="float32")
    return KL.Lambda(lambda s: s - lo, name="rs_target")(generator.outputs[2])


def build_loss(y_true, y_pred):

    # plain mse over the three axes: no gating (an axis always has a resolution), no per-axis weighting.
    def fn(x):
        yt, yp = x
        return K.expand_dims(K.mean(K.square(yt - yp), axis=1), -1)
    loss = KL.Lambda(fn, name='rs_loss')([y_true, y_pred])
    loss._keras_shape = tuple(loss.get_shape().as_list())
    return loss
