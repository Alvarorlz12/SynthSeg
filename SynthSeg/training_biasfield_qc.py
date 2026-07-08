"""

Trains a regressor to recover the realised bias-field severity (the std of the log-bias
field B(x) over the brain mask) directly from a synthetic image. Cheapest "recover the
injected generation parameter" qc test.

The whole pipeline lives in one TensorFlow/keras graph (SynthSeg style): the model inputs
[labels, means, stds] are turned into a synthetic image in-graph; the same synthesis branch
exposes the realised target std_log as the named layer 'bias_field_std'; a conv encoder
regresses a scalar from the image; and the loss (Huber on the normalised label) is computed
in-graph and returned as the model output, so it compiles with metrics.IdentityLoss()
(inside the reused train_model).

Clone of training_qc.py with the regression target swapped (dice score to bias_std_log) and
the augmentation model replaced by the labels_to_image_model generator (return_bias_std=True).

Clean scope: elastic deformation, randomise_res and gamma are off; only the
bias field is on, with sigma1 ~ U(0, bias_field_std) sampled in-graph.

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
import numpy as np
from keras import models
import keras.layers as KL
import keras.backend as K

# project imports
from SynthSeg.training_qc import train_model          # reused verbatim (compiles with IdentityLoss + Adam)
from SynthSeg.model_inputs import build_model_inputs
from SynthSeg.labels_to_image_model import labels_to_image_model

# third-party imports
from ext.lab2im import utils
from ext.neuron import models as nrn_models


def training(labels_dir,
             model_dir,
             generation_labels,
             output_labels=None,
             generation_classes=None,
             n_neutral_labels=None,
             subjects_prob=None,
             batchsize=1,
             n_channels=1,
             output_shape=160,
             prior_distributions='uniform',
             prior_means=None,
             prior_stds=None,
             # bias field / regression label
             bias_field_std=0.5,
             bias_scale=0.025,
             std_log_max=0.2,
             huber_delta=0.25,
             # encoder architecture
             n_levels=5,
             nb_conv_per_level=3,
             conv_size=5,
             unet_feat_count=24,
             feat_multiplier=2,
             activation='relu',
             # training
             lr=1e-4,
             epochs=300,
             steps_per_epoch=1000,
             checkpoint=None):

    """
    Trains a regressor to recover the realised bias-field severity (std of the log-bias
    field over the brain mask) from a synthetic image, end-to-end in one graph.

    # note: when a parameter has separate values per axis (numpy array or sequence),
    # these refer to the RAS axes.

    :param labels_dir: path of a folder with all the input label maps, or path to a single label map.
    :param model_dir: path of a directory where the models will be saved during training.
    :param generation_labels: list of all the label values in the input label maps. Can be a sequence, a 1d
    numpy array, or the path to such an array.

    # generation parameters
    # label maps parameters
    :param output_labels: (optional) does not affect the QC target (the masked std uses the pre-conversion
    generation labels). Defaults to generation_labels.
    :param generation_classes: (optional) indices regrouping generation labels into classes of the same GMM
    intensity distribution. Same length as generation_labels.
    :param n_neutral_labels: (optional) number of non-sided generation labels. Defaults to all of them (flipping
    is off on this path).
    :param subjects_prob: (optional) relative sampling importance of each label map.

    # output / GMM parameters
    :param batchsize: (optional) number of images generated per mini-batch. Default 1.
    :param n_channels: (optional) number of channels to synthesise. Default 1.
    :param output_shape: (optional) shape of the cropped synthetic image. Default 160.
    :param prior_distributions: (optional) 'uniform' or 'normal' for the GMM priors. Default 'uniform'.
    :param prior_means: (optional) hyper-parameters of the prior over the GMM means.
    :param prior_stds: (optional) hyper-parameters of the prior over the GMM standard deviations.

    # bias field / label parameters
    :param bias_field_std: (optional) upper bound of the in-graph sampled bias std-dev (sigma1 ~ U(0, this)).
    Default 0.5.
    :param bias_scale: (optional) ratio between the label-map size and the small sampled bias tensor. Default 0.025.
    :param std_log_max: (optional) ceiling used to normalise the label: clamp(std_log / std_log_max, 0, 1).
    Default 0.2.
    :param huber_delta: (optional) Huber transition point on the normalised label (in label units;
    0.25 ~= 0.05 in std_log). Below it the loss is quadratic, above it linear. Default 0.25.

    # encoder architecture
    :param n_levels: (optional) number of levels (downsamples) of the convolutional encoder. Default 5.
    :param nb_conv_per_level: (optional) number of convolutions per level. Default 3.
    :param conv_size: (optional) size of the convolution kernels. Default 5.
    :param unet_feat_count: (optional) number of features at the first level. Default 24.
    :param feat_multiplier: (optional) feature multiplier between levels. Default 2.
    :param activation: (optional) activation function ('relu' or 'elu'). Default 'relu'.

    # training parameters
    :param lr: (optional) learning rate. Default 1e-4.
    :param epochs: (optional) number of epochs. Default 300.
    :param steps_per_epoch: (optional) steps per epoch (also the model-saving frequency). Default 1000.
    :param checkpoint: (optional) path of a saved model to load before starting training.
    """

    # prepare data files
    labels_paths = utils.list_images_in_folder(labels_dir)
    generation_labels = utils.load_array_if_path(generation_labels)
    output_labels = generation_labels if output_labels is None else utils.load_array_if_path(output_labels)
    generation_classes = utils.load_array_if_path(generation_classes)
    if n_neutral_labels is None:
        n_neutral_labels = generation_labels.shape[0]

    labels_shape, _, n_dims, _, _, atlas_res = utils.get_volume_info(labels_paths[0], aff_ref=np.eye(4))

    # 1) generator graph: synthesis + in-graph target (exposed as the named layer 'bias_field_std')
    generator = labels_to_image_model(labels_shape=labels_shape,
                                      n_channels=n_channels,
                                      generation_labels=generation_labels,
                                      output_labels=output_labels,
                                      n_neutral_labels=n_neutral_labels,
                                      atlas_res=atlas_res,
                                      target_res=None,
                                      output_shape=output_shape,
                                      output_div_by_n=2 ** n_levels,   # keep the image divisible by the pools
                                      flipping=False, aff=np.eye(4),
                                      scaling_bounds=False, rotation_bounds=False,
                                      shearing_bounds=False, translation_bounds=False,
                                      nonlin_std=0, randomise_res=False,   # clean scope
                                      bias_field_std=bias_field_std, bias_scale=bias_scale,
                                      intensity_gamma_std=0.,   # clean scope: the bias is the only nuisance
                                      return_bias_std=True)

    # 2) regression head on the synthetic image (generator.outputs[0])
    regression_model = build_biasqc_model(input_model=generator,
                                          n_levels=n_levels,
                                          nb_conv_per_level=nb_conv_per_level,
                                          conv_size=conv_size,
                                          unet_feat_count=unet_feat_count,
                                          feat_multiplier=feat_multiplier,
                                          activation=activation)

    # 3) in-graph Huber loss against the normalised label
    qc_model = build_biasqc_loss(generator, regression_model, std_log_max=std_log_max, huber_delta=huber_delta)

    # input generator: GMM means/stds + dummy zero target (IdentityLoss ignores y_true)
    model_inputs = build_model_inputs(path_label_maps=labels_paths,
                                      n_labels=len(generation_labels),
                                      batchsize=batchsize,
                                      n_channels=n_channels,
                                      subjects_prob=subjects_prob,
                                      generation_classes=generation_classes,
                                      prior_means=prior_means,
                                      prior_stds=prior_stds,
                                      prior_distributions=prior_distributions)
    input_generator = utils.build_training_generator(model_inputs, batchsize)

    train_model(qc_model, input_generator, lr, epochs, steps_per_epoch, model_dir, 'qc', checkpoint)


def build_biasqc_model(input_model,
                       n_levels,
                       nb_conv_per_level,
                       conv_size,
                       unet_feat_count,
                       feat_multiplier,
                       activation):

    # the synthetic image is the first output of the generator
    last_tensor = input_model.outputs[0]
    input_shape = last_tensor.get_shape().as_list()[1:]      # e.g. [160, 160, 160, 1]

    # convolutional encoder. batch_norm is off on purpose: the label is the spatial amplitude (std)
    # of the bias field, and with batchsize=1 BatchNormalization(axis=-1) behaves like instance-norm
    # over the spatial dims, so it rescales each channel's spatial variance to ~1, normalising away
    # the very amplitude we need to regress (the net then collapses to predicting the mean label).
    model = nrn_models.conv_enc(input_model=input_model,
                                input_shape=input_shape,
                                nb_levels=n_levels,
                                nb_conv_per_level=nb_conv_per_level,
                                conv_size=conv_size,
                                nb_features=unet_feat_count,
                                feat_mult=feat_multiplier,
                                activation=activation,
                                batch_norm=None,
                                use_residuals=True,
                                name='biasqc')
    feat = model.outputs[0]                                             # [B, w, w, w, F]

    # global mean and std pooling: the std exposes the spatial-variance statistic directly (plain
    # GlobalAveragePooling alone would average the spatial variation away, hiding the signal).
    gmean = KL.GlobalAveragePooling3D(name='biasqc_gmean')(feat)        # [B, F]
    gstd = KL.Lambda(lambda t: K.std(t, axis=[1, 2, 3]), name='biasqc_gstd')(feat)   # [B, F]
    pooled = KL.Concatenate(name='biasqc_pool')([gmean, gstd])          # [B, 2F]
    pred = KL.Dense(1, activation='sigmoid', name='biasqc_pred')(pooled)  # [B, 1] in (0, 1)

    return models.Model(input_model.inputs, pred)


def build_biasqc_loss(generator, regression_model, std_log_max=0.2, huber_delta=0.25):

    # read the target from the generator, not the regression model: conv_enc only consumes outputs[0],
    # so the 'bias_field_std' branch is not on the regression model's inputs-to-prediction path and
    # regression_model.get_layer('bias_field_std') would raise.
    std_log = generator.get_layer('bias_field_std').output             # [B, 1] raw / physical
    label = KL.Lambda(lambda s: K.clip(s / std_log_max, 0., 1.), name='biasqc_label')(std_log)
    pred = regression_model.outputs[0]                                 # [B, 1]

    loss = KL.Lambda(lambda x: huber_loss(x[0], x[1], huber_delta), name='qc_loss')([label, pred])
    loss._keras_shape = tuple(loss.get_shape().as_list())             # keras 2.3 quirk (cf. training_qc.py)

    return models.Model(inputs=generator.inputs, outputs=loss)


def huber_loss(y_true, y_pred, delta=0.1):
    e = K.abs(y_true - y_pred)
    quad = K.minimum(e, delta)        # quadratic part, capped at delta (avoids tf.where)
    lin = e - quad                    # linear excess beyond delta
    return K.mean(0.5 * K.square(quad) + delta * lin)   # scalar, reduces over the batch
