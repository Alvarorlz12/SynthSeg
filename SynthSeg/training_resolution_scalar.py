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

the target is the resolution deficit relative to the grid the image is stored on, in millimetres:

    y_k = s_k - atlas_res_k

0 at native, read back as s = atlas_res + pred. millimetres and not log because millimetres are what gets
deployed: the head is asked how far from native a scan is, and a small error in log is a large one in mm
at the coarse end. the cost is that under mse a relative error is weighted by s^2, so the coarse end
carries more of the loss than the fine end where the QC call is actually made.

the spacing is drawn per axis, independently, from a uniform over [atlas_res, max_res], with a fixed
probability of a native (1 mm isotropic) volume. SynthSeg's own sampler is one flag away
(synthseg_sampler) but is off by default: it couples the axes, since a coarse axis there implies the
other two are native, which the net can read instead of measuring the blur.

three outputs and not one: unlike a bias field, resolution is intrinsically per-axis. a scan can be fine
in-plane and coarse through-plane, and that anisotropy IS the QC case (2D multi-slice, failed recons,
resampled-up FLAIR). k=3 is one output channel per axis, the way the dice qc net has one per score.

it departs from the bias-severity net in exactly three places, all forced by the target:
  1. the generator runs the randomise_res path (randomise_res=True is mandatory, return_resolution asserts
     it) and returns the drawn spacing; the bias field is off by default, so resolution is the only
     corruption.
  2. the target is a 3-vector read off the generator, not one scalar, so the head is k=3.
  3. the image is re-normalised after the degradation (see build_generator). this is what keeps the
     the same one that worked for the bias field: there, labels_to_image_model applies the field and then
     min-max normalises, so the net sees an image in [0, 1] with the corruption inside it, exactly like a
     real scan normalised at deployment. for resolution the degradation happens after that normalisation
     and nothing normalises again, so without this line the net would train on images spanning ~[0.12,
     0.74] while every deployed image arrives in [0, 1] -- an accidental difference from the bias setup,
     not a deliberate one. same principle as instance norm: train, validate and deploy the same function.
     (instance norm already absorbs much of a global scale change, so this is cheap insurance rather than
     a large effect.)
the network is the dice qc net's, UNCHANGED: same encoder, and the same head of a max pool, two
k-channel relu convolutions and a spatial mean. it is built in this file rather than imported from the
tissue-means module, which departs from that head in two places this target does not want -- a wider
first convolution and a linear last one, both of which suit a target sitting near 0.5 and never near 0.
this one is >= 0 and is exactly 0 on a native volume, so the relu fits it as it fits a dice score. the
checkpoint guard and the training loop are still imported. instance norm is the deployable default
here, so the function that trains is the function that is deployed.

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

# the checkpoint guard, the validation callback and the training loop are the tissue-means ones, imported
# rather than copied. the network itself is built below instead: the two targets do not want the same last
# activation, and a shared builder would make that a change to three heads at once.
from SynthSeg.training_tissue_means import load_weights_checked, train_model

eps = 1e-6

# the label is a per-array-axis vector, not an anatomical one: the spatial deformation rotates the
# anatomy inside the array before the degradation, so the degradation axes are the storage axes.
# mapping to RAS is a deterministic post-step at deployment, not something the net learns.


def training(labels_dir,
             model_dir,
             generation_labels,
             generation_classes,
             max_res_iso=4.,
             max_res_aniso=8.,
             synthseg_sampler=False,
             res_prob_min=0.2,
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
    :param generation_classes: path to the 1d array grouping the labels that share one drawn gaussian. The
    target does not depend on the tissue grouping (it is a property of the acquisition, not of the tissues),
    so the standard SynthSeg classes are used for the richest contrast randomisation.

    # resolution: the target and the only corruption on by default
    :param max_res_iso: (optional) upper bound of the per-axis uniform. Default 4, the SynthSeg default.
    :param max_res_aniso: (optional) upper bound for the axis the stock anisotropic branch selects.
    Default 8, the SynthSeg default. The per-axis sampler draws from the larger of the two bounds.
    :param res_prob_min: (optional) probability of drawing the native resolution on every axis, i.e. of a
    1 mm isotropic volume. Default 0.2.
    :param synthseg_sampler: (optional) fall back to SynthSeg's own sampler, which either shares one
    value across the axes or degrades a single axis and pins the other two at atlas_res. Off by default:
    that coupling means a coarse axis implies the others are native, which the net can read instead of
    measuring the blur, and it leaves 62.8% of per-axis targets sitting on exactly 1 mm.
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
    :param checkpoint: (optional) path of a saved model to resume from.
    :param seed: (optional) random seed. Default 0.
    """

    # prepare labels
    gen_labels = np.asarray(utils.load_array_if_path(generation_labels)).astype('int32')
    gen_classes = np.asarray(utils.load_array_if_path(generation_classes)).astype('int32')
    assert max(max_res_iso, max_res_aniso) > 0, 'the resolution range is the target of this regressor'

    # every map in labels_dir trains. the split is frozen on disk and labels_dir is the training
    # partition already, so carving a second one out here would only shrink it.
    train_paths = sorted(utils.list_images_in_folder(labels_dir))

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
                                bias_field_std, bias_scale, gamma_std, clip, flipping,
                                res_uniform_per_axis=not synthseg_sampler,
                                res_prob_min=res_prob_min)
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

    # prefix 'rs' names the checkpoints rs_###.h5 (tissue-means uses tm_###.h5, bias bf_###.h5).
    train_model(regression_model, input_generator, lr, epochs, steps_per_epoch, model_dir, checkpoint,
                init_epoch, clipnorm, prefix='rs')


def build_generator(labels_shape, atlas_res, generation_labels, output_shape, output_div_by_n,
                    n_neutral_labels, scaling_bounds, rotation_bounds, shearing_bounds, translation_bounds,
                    nonlin_std, nonlin_scale, max_res_iso, max_res_aniso,
                    bias_field_std, bias_scale, gamma_std, clip, flipping=True,
                    res_uniform_per_axis=True, res_prob_min=0.2):

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
                               res_uniform_per_axis=res_uniform_per_axis, res_prob_min=res_prob_min,
                               bias_field_std=bias_field_std, bias_scale=bias_scale,
                               intensity_gamma_std=gamma_std, intensity_clip=clip,
                               return_bias_std=False, return_resolution=True)

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


def build_regression_model(generator, image_shape, k, n_levels, nb_conv_per_level, conv_size, feat_count,
                           feat_multiplier, activation, batch_norm, use_residuals, instance_norm=False):

    # the dice qc net's encoder and head, unchanged: conv encoder, max pool, two k-channel relu
    # convolutions, average over space, which keeps the location until the output. Written out here
    # rather than imported from the tissue-means module because that one departs from it in two places
    # this head does not want, and a shared builder would make either change a change to three heads.
    enc = nrn_models.conv_enc(input_model=generator, input_shape=image_shape, nb_levels=n_levels,
                              conv_size=conv_size, nb_features=feat_count, feat_mult=feat_multiplier,
                              nb_conv_per_level=nb_conv_per_level, activation=activation,
                              batch_norm=batch_norm, instance_norm=instance_norm,
                              use_residuals=use_residuals, name='rs_enc')
    last = enc.outputs[0]
    conv_kwargs = {'padding': 'same', 'activation': 'relu', 'data_format': 'channels_last'}
    last = KL.MaxPool3D(pool_size=(2, 2, 2), padding='same', name='rs_conv_pool')(last)
    # k channels in BOTH head convs, both relu: the dice qc net's head unchanged, so there is nothing to
    # declare about it. It emits one channel per regressed score and so does this, one per array axis.
    # The tissue-means head widens the first conv to max(16, k) and leaves the last one linear, because
    # its target sits near 0.5 and never near 0: a relu there would price in a floor the target never
    # touches, and a channel whose map starts out all negative would have mean exactly 0, gradient exactly
    # 0, and never recover. This target is >= 0 and is exactly 0 on a native volume, a fifth of them, so
    # the relu fits it the same way it fits a dice score.
    # NOTE a checkpoint records neither the last activation nor the widths, so keep runs that differ in
    # them in separate model directories.
    last = KL.Conv3D(k, kernel_size=5, **conv_kwargs, name='rs_conv0')(last)
    last = KL.Conv3D(k, kernel_size=5, **conv_kwargs, name='rs_conv1')(last)
    return KL.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2, 3]), name='rs_pred')(last)


def build_target(generator, atlas_res):

    # the drawn per-axis spacing, exposed by the generator as its third output ("resolution",
    # shape [B, 3]), as a deficit in millimetres relative to the grid the image is stored on:
    #     y_k = s_k - atlas_res_k
    # 0 at native, and read back as s = atlas_res + pred. Under MSE this weights a relative error
    # by s^2, so the coarse end carries more of the loss than the fine one; the fine end is where
    # the QC call is made, but it is also where a millimetre of error matters least in absolute
    # terms, which is the quantity being deployed.
    lo = np.asarray(utils.reformat_to_list(atlas_res, length=3, dtype="float"), dtype="float32")
    return KL.Lambda(lambda s: s - lo, name="rs_target")(generator.outputs[2])


def build_loss(y_true, y_pred):

    # plain mse over the three axes, exactly as the bias-severity net: no present-gating (an axis always has
    # a resolution) and no per-axis weighting.
    def fn(x):
        yt, yp = x
        return K.expand_dims(K.mean(K.square(yt - yp), axis=1), -1)
    loss = KL.Lambda(fn, name='rs_loss')([y_true, y_pred])
    loss._keras_shape = tuple(loss.get_shape().as_list())
    return loss
