"""

The tissue-means regressor as it was trained before the head was brought back to the QC net's, kept
because it is the architecture every tm_*.h5 in models/ holds.

Two layers differ from QC/training_tm.py, both in the head: the first convolution is widened to
max(16, k) channels, and the last one is linear instead of relu. The layer names are the same, tm_conv0
and tm_conv1, so a checkpoint trained here is refused by the current head on its shapes, and the other
way round. Everything else -- the generator, the target, the loss, the training loop and the checkpoint
guard -- is imported from QC.training_tm, so the two cannot drift apart anywhere else.

qc_head=True builds the QC net's head instead, which is what experiments/training_biasfield_scalar.py
passes to get it.

Use QC/training_tm.py for a new run. This file is here to score and resume the old checkpoints:
scripts/experiments/validate_tissue_means.py and scripts/experiments/eval_tissue_means.py build their
graph from it.

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
from SynthSeg.model_inputs import build_model_inputs

# third-party imports
from ext.lab2im import utils
from ext.neuron import models as nrn_models

# the generator, the target, the loss and the training loop are the current head's, unchanged.
# load_weights_checked is imported to be re-exported: the two scorers reach it as tm.load_weights_checked.
from QC.training_tm import (all_tissues, build_generator, build_target, build_loss, build_tissue_lut,
                            check_alignment, load_weights_checked, train_model)


def training(labels_dir,
             model_dir,
             generation_labels,
             generation_classes,
             tissues='CSF,GM,WM',
             holdout=4,
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
             bias_field_std=0.,
             bias_prob=.95,
             gamma_std=0.,
             gamma_prob=1.,
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
             min_vox=8,
             seed=0):

    """
    :param labels_dir: path of the folder with the training label maps.
    :param model_dir: path of a directory where the models will be saved during training.
    :param generation_labels: path to the 1d array of all the label values in the label maps.
    :param generation_classes: path to the 1d array grouping the labels that share one drawn gaussian (this is
    what ties each tissue to a single intensity).

    :param tissues: (optional) comma separated tissues to regress, among CSF, GM, WM. Default is all three.
    :param holdout: (optional) number of label maps kept out of training, i.e. anatomy the run never
    sees. The split is deterministic on the sorted paths, so the evaluation script agrees on
    which maps were never seen: change it in both or the offline probe stops meaning the same thing.
    Default is 4, i.e. 16 for training out of the 20 in the repo.
    :param batchsize: (optional) number of images per minibatch. Default is 1.
    :param output_shape: (optional) shape of the cropped output image. Default is 160.

    # spatial deformation (pass False / 0 to turn a term off)
    :param flipping: (optional) random right/left flip of the anatomy. Default is True.
    :param scaling_bounds: (optional) scaling factor bounds. Default is 0.2.
    :param rotation_bounds: (optional) rotation angle bounds. Default is 15.
    :param shearing_bounds: (optional) shearing bounds. Default is 0.012.
    :param translation_bounds: (optional) translation bounds. Default is False.
    :param nonlin_std: (optional) std of the elastic deformation field. Default is 4.
    :param nonlin_scale: (optional) scale of the elastic deformation field. Default is 0.04.

    # intensity corruption (all off by default, i.e. the clean regime)
    :param randomise_res: (optional) simulate a random acquisition resolution. Default is False.
    :param max_res_iso: (optional) max isotropic resolution. Default is 4.
    :param max_res_aniso: (optional) max anisotropic resolution. Default is 8.
    :param bias_field_std: (optional) std of the bias field. Default is 0.
    :param bias_prob: (optional) fraction of images the bias field is actually applied to. Default .95.
    :param gamma_std: (optional) std of the gamma augmentation. Default is 0.
    :param gamma_prob: (optional) fraction of images the gamma is actually applied to. Default is 1.
    :param clip: (optional) intensity clipping percentile (0 keeps the exact min-max). Default is 0.
    :param n_neutral_labels: (optional) number of non-lateral labels in generation_labels. Default is 18.

    # architecture
    :param n_levels: (optional) number of levels of the encoder. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutions per level. Default is 3.
    :param conv_size: (optional) size of the convolution kernels. Default is 5.
    :param unet_feat_count: (optional) number of features at the first level. Default is 24.
    :param feat_multiplier: (optional) feature multiplier between levels. Default is 2.
    :param activation: (optional) activation function. Default is 'relu'.
    :param batch_norm: (optional) axis to batch normalise, or None to turn batch norm off. It is an axis, not
    a flag: -1 is the feature axis. Default is -1. Only read when instance_norm is False.
    :param instance_norm: (optional) normalise each image by its own statistics, in training and at
    inference alike. Default is False.
    :param use_residuals: (optional) residual connection per level. Default is True.

    # training
    :param lr: (optional) learning rate. Default is 1e-4.
    :param clipnorm: (optional) gradient norm clipping, 0 to turn it off. Default is 0.
    :param epochs: (optional) number of epochs. Default is 100.
    :param steps_per_epoch: (optional) steps per epoch, i.e. how often the model is saved. Default is 1000.
    :param checkpoint: (optional) path of a saved model to resume from.
    :param min_vox: (optional) a tissue with fewer voxels in the crop is not scored in the loss. Default is 8.
    :param seed: (optional) random seed. Default is 0.
    """

    # prepare labels and tissues
    gen_labels = np.asarray(utils.load_array_if_path(generation_labels)).astype('int32')
    gen_classes = np.asarray(utils.load_array_if_path(generation_classes)).astype('int32')
    names = [t.strip().upper() for t in tissues.split(',') if t.strip()]
    assert names and all(t in all_tissues for t in names), 'pick tissues among %s' % all_tissues

    # hold a few maps out of training. the split is deterministic on the sorted paths, so training and
    # the evaluation script agree on which maps were never seen.
    labels_paths = sorted(utils.list_images_in_folder(labels_dir))
    n_hold = min(max(holdout, 0), len(labels_paths) - 1)
    train_paths = labels_paths[:len(labels_paths) - n_hold] if n_hold else labels_paths
    val_paths = labels_paths[len(labels_paths) - n_hold:] if n_hold else []

    # a resumed job draws a fresh stream but stays reproducible
    init_epoch = 0 if checkpoint is None else int(os.path.basename(checkpoint).split('tm_')[1][:-3])
    np.random.seed(seed + init_epoch)
    tf.random.set_seed(seed + init_epoch)

    # generation model: the domain randomisation generator; outputs[0] is the image, outputs[1] the label map
    labels_shape, _, _, _, _, atlas_res = utils.get_volume_info(train_paths[0], aff_ref=np.eye(4))
    generator = build_generator(labels_shape, atlas_res, gen_labels, output_shape, 2 ** n_levels,
                                n_neutral_labels, scaling_bounds, rotation_bounds, shearing_bounds,
                                translation_bounds, nonlin_std, nonlin_scale, randomise_res, max_res_iso,
                                max_res_aniso, bias_field_std, gamma_std, clip, flipping,
                                bias_prob=bias_prob, gamma_prob=gamma_prob)
    image_shape = generator.outputs[0].get_shape().as_list()[1:]

    # target and prediction
    lut, k = build_tissue_lut(gen_labels, names)
    check_alignment(gen_labels, gen_classes, names)
    mu_pred = build_regression_model(generator, image_shape, k, n_levels, nb_conv_per_level, conv_size,
                                     unet_feat_count, feat_multiplier, activation, batch_norm, use_residuals,
                                     instance_norm)
    mu_true, present = build_target(generator, lut, k)
    loss = build_loss(mu_true, mu_pred, present, min_vox)
    regression_model = models.Model(generator.inputs, loss)

    n_train = int(np.sum([K.count_params(w) for w in regression_model.trainable_weights]))
    print('regressing: %s   trainable params: %d' % (', '.join(names), n_train))

    model_inputs = build_model_inputs(path_label_maps=train_paths, n_labels=len(gen_labels),
                                      batchsize=batchsize, n_channels=1,
                                      generation_classes=gen_classes, prior_distributions='uniform')
    input_generator = utils.build_training_generator(model_inputs, batchsize)

    print('  label maps: %d for training, %d held out' % (len(train_paths), len(val_paths)))
    # the intensity regime, printed because it is what the target distribution depends on and the log is the
    # only place a finished run can be asked what it was trained on.
    print('  intensity regime: randomise_res %s   bias_std %.2f (prob %.2f)   gamma_std %.2f (prob %.2f)'
          % (randomise_res, bias_field_std, bias_prob if bias_field_std > 0 else 0.,
             gamma_std, gamma_prob if gamma_std > 0 else 0.))

    train_model(regression_model, input_generator, lr, epochs, steps_per_epoch, model_dir, checkpoint,
                init_epoch, clipnorm)


def build_regression_model(generator, image_shape, k, n_levels, nb_conv_per_level, conv_size, feat_count,
                           feat_multiplier, activation, batch_norm, use_residuals, instance_norm=False,
                           qc_head=False):

    # conv encoder on the image, then the head: max pool, two convolutions, and average over space,
    # which keeps the location until the output.
    enc = nrn_models.conv_enc(input_model=generator, input_shape=image_shape, nb_levels=n_levels,
                              conv_size=conv_size, nb_features=feat_count, feat_mult=feat_multiplier,
                              nb_conv_per_level=nb_conv_per_level, activation=activation,
                              batch_norm=batch_norm, instance_norm=instance_norm,
                              use_residuals=use_residuals, name='tm_enc')
    last = enc.outputs[0]
    conv_kwargs = {'padding': 'same', 'activation': 'relu', 'data_format': 'channels_last'}
    last = KL.MaxPool3D(pool_size=(2, 2, 2), padding='same', name='tm_conv_pool')(last)

    # qc_head is the QC net's head, k channels in both convolutions and relu on both, under names of its
    # own so that a checkpoint from the other head has layers with nowhere to go and load_weights_checked
    # refuses it out loud instead of loading half a net. It is what QC/training_tm.py builds today.
    if qc_head:
        last = KL.Conv3D(k, kernel_size=5, **conv_kwargs, name='tm_qc_conv0')(last)
        last = KL.Conv3D(k, kernel_size=5, **conv_kwargs, name='tm_qc_conv1')(last)
        return KL.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2, 3]), name='tm_pred')(last)

    # the two layers this file exists for: a first convolution of max(16, k) channels, and a linear last
    # one. Both were argued for a target that sits near 0.5 and never reaches 0, so relu would price in a
    # floor the target never touches, and a channel whose whole map starts negative would have a mean of
    # exactly 0, a gradient of exactly 0, and never recover.
    last = KL.Conv3D(max(16, k), kernel_size=5, **conv_kwargs, name='tm_conv0')(last)
    last = KL.Conv3D(k, kernel_size=5, padding='same', activation=None, name='tm_conv1')(last)
    return KL.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2, 3]), name='tm_pred')(last)
