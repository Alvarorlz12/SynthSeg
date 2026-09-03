"""

Trains a scalar regressor to predict the bias field severity of the synthetic image it is given. The
network is the SynthSeg QC net (SynthSeg/training_qc.py) with k = 1, so it returns one scalar per image.

The target is std_log, the standard deviation of the log bias field B(x) over the whole crop, with no
brain mask. It is computed inside the generation graph by labels_to_image_model, which returns it as the
named layer 'bias_field_std' when return_bias_std=True, and it is regressed in its own units, with no
ceiling and no rescaling. The std of the field is drawn from U(0, bias_field_std) once per image, so two
images generated with the same bias_field_std have different targets, and the (1 - bias_prob) fraction
left bias-free have target 0. Since std_log is >= 0 and reaches 0, the relu head of the QC net is kept.

std_log is read off the field before the intensity augmentation, so it does not include the gamma applied
afterwards. Raising the normalised intensities to a power k multiplies the log domain by k, so the
severity visible in the image is k * std_log. bias_field_after_gamma swaps the two steps inside
labels_to_image_model, and this file only passes the argument down. That keyword is not implemented there
yet, so any call raises TypeError until it is.

The layers are named bf_*, so load_weights_checked refuses a checkpoint from another head.

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

# the training loop and its checkpoint guard are shared with the other regression heads
from QC.training_tm import train_model


def training(labels_dir,
             model_dir,
             generation_labels,
             generation_classes,
             bias_field_std=0.7,
             bias_prob=0.9,
             bias_scale=.025,
             bias_field_after_gamma=False,
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
             gamma_std=0.5,
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
    The standard SynthSeg classes are used here.

    # bias field
    :param bias_field_std: (optional) max std of the normal the small bias tensor is sampled from; the
    layer draws sigma ~ U(0, bias_field_std) per image. Default is 0.7.
    :param bias_prob: (optional) probability of applying the sampled field; the rest of the images are
    left bias-free, with target 0. Default is 0.9.
    :param bias_scale: (optional) ratio between the label map size and the small sampled bias tensor,
    i.e. how smooth the field is. Default is 0.025.
    :param bias_field_after_gamma: (optional) apply the field after the gamma augmentation instead of
    before it. Default is False, i.e. GMM -> bias -> clip -> min-max -> gamma. Passed straight down to
    labels_to_image_model, which does not implement this argument yet.

    :param batchsize: (optional) number of images per minibatch. Default is 1.
    :param output_shape: (optional) shape of the cropped output image. Default is 160.

    # spatial deformation (pass False or 0 to turn a term off)
    :param flipping: (optional) random right/left flip. Default is True.
    :param scaling_bounds: (optional) scaling factor bounds. Default is 0.2.
    :param rotation_bounds: (optional) rotation angle bounds. Default is 15.
    :param shearing_bounds: (optional) shearing bounds. Default is 0.012.
    :param translation_bounds: (optional) translation bounds. Default is False.
    :param nonlin_std: (optional) std of the elastic deformation field. Default is 4.
    :param nonlin_scale: (optional) scale of the elastic deformation field. Default is 0.04.

    # other intensity augmentation
    :param randomise_res: (optional) simulate a random acquisition resolution. Default is False.
    :param max_res_iso: (optional) max isotropic resolution, only used when randomise_res. Default is 4.
    :param max_res_aniso: (optional) max anisotropic resolution, only read when randomise_res. Default 8.
    :param gamma_std: (optional) std of the gamma augmentation. Default is 0.5.
    :param clip: (optional) intensity clipping percentile, 0 keeps the exact min-max. Default is 0.
    :param n_neutral_labels: (optional) number of non-lateral labels in generation_labels. Default is 18.

    # architecture
    :param n_levels: (optional) number of levels of the encoder. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutions per level. Default is 3.
    :param conv_size: (optional) size of the convolution kernels. Default is 5.
    :param unet_feat_count: (optional) number of features at the first level. Default is 24.
    :param feat_multiplier: (optional) feature multiplier between levels. Default is 2.
    :param activation: (optional) activation function. Default is 'relu'.
    :param batch_norm: (optional) axis to batch normalise, or None to turn it off. -1 is the feature
    axis. Default is -1. Only read when instance_norm is False.
    :param instance_norm: (optional) normalise each image by its own statistics, in training and at
    inference alike. Default is False.
    :param use_residuals: (optional) residual connection per level. Default is True.

    # training
    :param lr: (optional) learning rate. Default is 1e-4.
    :param clipnorm: (optional) gradient norm clipping, 0 to turn it off. Default is 0.
    :param epochs: (optional) number of epochs. Default is 100.
    :param steps_per_epoch: (optional) steps per epoch, i.e. how often the model is saved. Default 1000.
    :param checkpoint: (optional) path of a saved model to resume from. It must be a bf_###.h5: the epoch
    to resume at, and the seed offset that goes with it, are parsed out of that name. Default is None.
    :param seed: (optional) random seed. Default is 0.
    """

    # prepare labels
    gen_labels = np.asarray(utils.load_array_if_path(generation_labels)).astype('int32')
    gen_classes = np.asarray(utils.load_array_if_path(generation_classes)).astype('int32')
    # RandomFlip splits generation_labels into [neutral | left | right] and pairs left[i] with right[i]
    # (layers.py:382-384) using int((n_labels - n_neutral) / 2), which truncates: an odd remainder
    # misaligns left and right silently.
    assert (len(gen_labels) - n_neutral_labels) % 2 == 0, \
        'generation_labels has %d entries and n_neutral_labels is %d, leaving %d lateral labels, which ' \
        'is odd. Left and right must pair up. Did you pass a different generation_labels without ' \
        'updating --neutral_labels?' % (len(gen_labels), n_neutral_labels,
                                        len(gen_labels) - n_neutral_labels)
    assert len(gen_labels) == len(gen_classes), \
        'generation_labels (%d) and generation_classes (%d) must have the same length; they are paired ' \
        'positionally.' % (len(gen_labels), len(gen_classes))
    assert bias_field_std > 0, 'bias_field_std must be > 0, it is the target of this regressor'

    # sorted so that a given seed means the same stream whatever order the filesystem lists the folder in
    labels_paths = sorted(utils.list_images_in_folder(labels_dir))

    # a resumed job draws a fresh stream but stays reproducible
    init_epoch = 0 if checkpoint is None else int(os.path.basename(checkpoint).split('bf_')[1][:-3])
    np.random.seed(seed + init_epoch)
    tf.random.set_seed(seed + init_epoch)

    # generation model: outputs[0] image, outputs[1] labels, outputs[2] the realised bias std [B, 1]
    labels_shape, _, _, _, _, atlas_res = utils.get_volume_info(labels_paths[0], aff_ref=np.eye(4))
    generator = build_generator(labels_shape, atlas_res, gen_labels, output_shape, 2 ** n_levels,
                                n_neutral_labels, scaling_bounds, rotation_bounds, shearing_bounds,
                                translation_bounds, nonlin_std, nonlin_scale, randomise_res, max_res_iso,
                                max_res_aniso, bias_field_std, bias_prob, bias_scale, gamma_std, clip,
                                flipping,
                                bias_field_after_gamma=bias_field_after_gamma)
    image_shape = generator.outputs[0].get_shape().as_list()[1:]

    # target and prediction. k = 1: one scalar per image
    y_pred = build_regression_model(generator, image_shape, 1, n_levels, nb_conv_per_level, conv_size,
                                    unet_feat_count, feat_multiplier, activation, batch_norm,
                                    use_residuals, instance_norm)
    y_true = build_target(generator)
    loss = build_loss(y_true, y_pred)
    regression_model = models.Model(generator.inputs, loss)
    n_train = int(np.sum([K.count_params(w) for w in regression_model.trainable_weights]))

    # input generator
    model_inputs = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels),
                                      batchsize=batchsize, n_channels=1,
                                      generation_classes=gen_classes, prior_distributions='uniform')
    input_generator = utils.build_training_generator(model_inputs, batchsize)

    print('regressing bias severity  std_log in its own units   %d label maps   %d params'
          % (len(labels_paths), n_train))
    print('  bias_field_std %.3f   bias_prob %.2f (~%.0f%% clean)   bias_scale %.3f   field applied %s '
          'the gamma' % (bias_field_std, bias_prob, 100 * (1 - bias_prob), bias_scale,
                         'after' if bias_field_after_gamma else 'before'))
    print('  gamma_std %.2f  clip %d' % (gamma_std, clip))
    # with gamma_std 0 there is no gamma to reorder against, though the clip and the normalisation still move
    if bias_field_after_gamma and (gamma_std <= 0):
        print('  [warn] bias_field_after_gamma is on but gamma_std is 0: no gamma to be applied after')

    train_model(regression_model, input_generator, lr, epochs, steps_per_epoch, model_dir, checkpoint,
                init_epoch, clipnorm, prefix='bf')


def build_generator(labels_shape, atlas_res, generation_labels, output_shape, output_div_by_n,
                    n_neutral_labels, scaling_bounds, rotation_bounds, shearing_bounds, translation_bounds,
                    nonlin_std, nonlin_scale, randomise_res, max_res_iso, max_res_aniso, bias_field_std,
                    bias_prob, bias_scale, gamma_std, clip, flipping=True,
                    bias_field_after_gamma=False):

    # return_bias_std=True adds the scalar severity at outputs[2], and return_resolution stays False so
    # that nothing is appended after it. output_labels = generation_labels keeps the label map on
    # outputs[1], and the crop to output_shape happens before the bias, so image, labels and field all
    # come out at output_shape.
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
                                 bias_scale=bias_scale, bias_prob=bias_prob,
                                 intensity_gamma_std=gamma_std,
                                 intensity_clip=clip, return_bias_std=True, return_resolution=False,
                                 bias_field_after_gamma=bias_field_after_gamma)


def build_regression_model(generator, image_shape, k, n_levels, nb_conv_per_level, conv_size, feat_count,
                           feat_multiplier, activation, batch_norm, use_residuals, instance_norm=False):

    # the QC net's encoder and head
    enc = nrn_models.conv_enc(input_model=generator, input_shape=image_shape, nb_levels=n_levels,
                              conv_size=conv_size, nb_features=feat_count, feat_mult=feat_multiplier,
                              nb_conv_per_level=nb_conv_per_level, activation=activation,
                              batch_norm=batch_norm, instance_norm=instance_norm,
                              use_residuals=use_residuals, name='bf_enc')
    last = enc.outputs[0]
    conv_kwargs = {'padding': 'same', 'activation': 'relu', 'data_format': 'channels_last'}
    # the encoder pools after every level but the last, so this fifth pool is what makes the volume
    # divisible by 2 ** n_levels, the output_div_by_n the generator crops to
    last = KL.MaxPool3D(pool_size=(2, 2, 2), padding='same', name='bf_conv_pool')(last)
    last = KL.Conv3D(k, kernel_size=5, **conv_kwargs, name='bf_conv0')(last)
    last = KL.Conv3D(k, kernel_size=5, **conv_kwargs, name='bf_conv1')(last)
    return KL.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2, 3]), name='bf_pred')(last)


def build_target(generator):

    # the std of the log field the image was multiplied by, exposed by the generator as its third output
    # ('bias_field_std', shape [batch, 1]). The identity gives it a named layer of its own, 'bf_target',
    # so it can be pulled out by name when the graph is rebuilt offline to score a checkpoint.
    std_log = generator.outputs[2]
    return KL.Lambda(lambda s: s, name='bf_target')(std_log)


def build_loss(y_true, y_pred):

    # per-image mse of shape [batch, 1]. IdentityLoss hands the tensor to keras untouched and keras
    # averages it, so the number reported is a mean over the images of the batch.
    def fn(x):
        yt, yp = x
        return K.expand_dims(K.mean(K.square(yt - yp), axis=1), -1)
    loss = KL.Lambda(fn, name='bf_loss')([y_true, y_pred])
    loss._keras_shape = tuple(loss.get_shape().as_list())
    return loss
