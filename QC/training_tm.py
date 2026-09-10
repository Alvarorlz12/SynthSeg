"""

Trains a regressor to predict the mean intensity of each tissue (CSF / GM / WM) of the synthetic image
it is given. The network is the SynthSeg QC net (SynthSeg/training_qc.py) with k = one output channel
per regressed tissue.

The target is measured on the final normalised image the network sees, in [0, 1], over the true tissue
masks, which the generator returns on its second output. It is well defined because the generation ties
each tissue to a single gaussian (grouped generation_classes), so the mean of a tissue is one intensity
rather than an average of several draws; check_alignment refuses a generation_classes that splits a
regressed tissue across several classes. A tissue with fewer than min_vox voxels in the crop is left out
of the loss, since an absent tissue would give a target of exactly 0, a value the real target never takes.

It is a blind regressor: it reads the image and outputs one number per tissue, it does not segment. The
deliverable built from it is the GM-WM contrast, |GM - WM| / (GM + WM).

The layers are named tm_*, so load_weights_checked refuses a checkpoint from another head.

The training loop, train_model, and the checkpoint guard, load_weights_checked, live here and are shared
with the other two heads.

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
import h5py
import numpy as np
import tensorflow as tf
from keras import models
import keras.layers as KL
import keras.backend as K
import keras.callbacks as KC
from keras.optimizers import Adam

# project imports
from SynthSeg.labels_to_image_model import labels_to_image_model
from SynthSeg.model_inputs import build_model_inputs
from SynthSeg import metrics_model as metrics

# third-party imports
from ext.lab2im import utils
from ext.neuron import models as nrn_models

eps = 1e-6

# the regressed groups, in output order: output j is the j-th key. each group must sit inside a single
# generation class, otherwise its mean would mix intensities from several draws (check_alignment refuses it).
# nine groups since the ungrouping of 2026-09-09; scripts/experiments/make_9groups_classes.py builds the
# matching generation_classes_9groups.npy and prints this dict.
tissue_groups = {
    'csf_ventricular':      [4, 5, 14, 15, 43, 44],   # lateral + inf-lateral ventricles l/r, 3rd, 4th
    'gm_cortex':            [3, 42],
    'gm_cerebellum':        [8, 47],
    'thalamus':             [10, 49],
    'putamen':              [12, 51],
    'pallidum':             [13, 52],
    'hippocampus_amygdala': [17, 18, 53, 54],
    'wm_cerebral':          [2, 41],
    'wm_cerebellum':        [7, 46],
}
# derived, never written twice: the check on --tissues and check_alignment both read it
all_tissues = list(tissue_groups)


def training(labels_dir,
             model_dir,
             generation_labels,
             generation_classes,
             tissues=None,
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

    :param tissues: (optional) comma separated groups to regress, among the keys of tissue_groups, in the
    order the outputs take. Default is None: all of them, in the dict's order.
    :param batchsize: (optional) number of images per minibatch. Default is 1.
    :param output_shape: (optional) shape of the cropped output image. Default is 160.

    # spatial deformation (pass False / 0 to turn a term off)
    :param flipping: (optional) random right/left flip of the anatomy. The grouped generation classes tie the
    left and right label of a tissue to one gaussian and the tissue lut maps both to one index, so the label
    swap that comes with the flip changes neither the intensities nor the target: it is free anatomy, which
    is worth having when only a handful of label maps are recycled all run. Default is True.
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
    :param gamma_prob: (optional) fraction of images the gamma is actually applied to. Default 1, the
    layer's own default, i.e. every image. This target is read off the corrupted image, so with both probs
    at 1 the network never sees the clean end of its own range: lower them to mix clean and corrupted
    images within the epoch instead of training on one regime at a time.
    :param clip: (optional) intensity clipping percentile (0 keeps the exact min-max). Default is 0.
    :param n_neutral_labels: (optional) number of non-lateral labels in generation_labels. Default is 18.

    # architecture (n_levels sets the downsampling, output_shape only the field of view: conv_enc pools
    # after every level but the last, so the encoder divides the volume by 2 ** (n_levels - 1))
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
    :param checkpoint: (optional) path of a saved model to resume from. It must be a tm_###.h5: the epoch to
    resume at, and the seed offset that goes with it, are parsed out of that name. Default is None.
    :param min_vox: (optional) a tissue with fewer voxels in the crop is not scored in the loss. Default is 8.
    :param seed: (optional) random seed. Default is 0.
    """

    # prepare labels and tissues
    gen_labels = np.asarray(utils.load_array_if_path(generation_labels)).astype('int32')
    gen_classes = np.asarray(utils.load_array_if_path(generation_classes)).astype('int32')
    names = list(all_tissues) if tissues is None \
        else [t.strip().lower() for t in tissues.split(',') if t.strip()]
    assert names and all(t in all_tissues for t in names), 'pick tissues among %s' % all_tissues

    # every map in labels_dir trains: labels_dir is already the training partition of a frozen split.
    # sorted so that a given seed means the same stream whatever order the filesystem lists the folder in.
    train_paths = sorted(utils.list_images_in_folder(labels_dir))

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

    model_inputs = build_model_inputs(path_label_maps=train_paths, n_labels=len(gen_labels),
                                      batchsize=batchsize, n_channels=1,
                                      generation_classes=gen_classes, prior_distributions='uniform')
    input_generator = utils.build_training_generator(model_inputs, batchsize)

    print('regressing: %s   %d label maps   %d params' % (', '.join(names), len(train_paths), n_train))
    # the intensity regime, printed because it is what the target distribution depends on and the log is the
    # only place a finished run can be asked what it was trained on.
    print('  intensity regime: randomise_res %s   bias_std %.2f (prob %.2f)   gamma_std %.2f (prob %.2f)'
          % (randomise_res, bias_field_std, bias_prob if bias_field_std > 0 else 0.,
             gamma_std, gamma_prob if gamma_std > 0 else 0.))

    # prefix 'tm' names the checkpoints tm_###.h5.
    train_model(regression_model, input_generator, lr, epochs, steps_per_epoch, model_dir, checkpoint,
                init_epoch, clipnorm)


def build_generator(labels_shape, atlas_res, generation_labels, output_shape, output_div_by_n,
                    n_neutral_labels, scaling_bounds, rotation_bounds, shearing_bounds, translation_bounds,
                    nonlin_std, nonlin_scale, randomise_res, max_res_iso, max_res_aniso, bias_field_std,
                    gamma_std, clip, flipping=True, bias_prob=.95, gamma_prob=1.):

    # output_labels = generation_labels so the label values survive on outputs[1] for the tissue masks
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
                                 bias_prob=bias_prob,
                                 intensity_gamma_std=gamma_std, intensity_gamma_prob=gamma_prob,
                                 intensity_clip=clip,
                                 return_bias_std=False, return_resolution=False)


def build_regression_model(generator, image_shape, k, n_levels, nb_conv_per_level, conv_size, feat_count,
                           feat_multiplier, activation, batch_norm, use_residuals, instance_norm=False):

    # the QC net's encoder and head: conv encoder, max pool, two k-channel relu convolutions, and an
    # average over space, which keeps the location until the output.
    enc = nrn_models.conv_enc(input_model=generator, input_shape=image_shape, nb_levels=n_levels,
                              conv_size=conv_size, nb_features=feat_count, feat_mult=feat_multiplier,
                              nb_conv_per_level=nb_conv_per_level, activation=activation,
                              batch_norm=batch_norm, instance_norm=instance_norm,
                              use_residuals=use_residuals, name='tm_enc')
    last = enc.outputs[0]
    conv_kwargs = {'padding': 'same', 'activation': 'relu', 'data_format': 'channels_last'}
    # the encoder pools after every level but the last, so this fifth pool is what makes the volume
    # divisible by 2 ** n_levels, the output_div_by_n the generator crops to
    last = KL.MaxPool3D(pool_size=(2, 2, 2), padding='same', name='tm_conv_pool')(last)
    last = KL.Conv3D(k, kernel_size=5, **conv_kwargs, name='tm_conv0')(last)
    last = KL.Conv3D(k, kernel_size=5, **conv_kwargs, name='tm_conv1')(last)
    return KL.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2, 3]), name='tm_pred')(last)


def build_target(generator, lut_np, k, suffix=''):

    onehot = labels_to_onehot(generator.outputs[1], lut_np, k, 'tm_onehot_true' + suffix)
    mu_true = masked_moment_pool(onehot, generator.outputs[0], 'tm_pool_true' + suffix)
    present = KL.Lambda(lambda oh: K.sum(oh, axis=[1, 2, 3]), name='tm_present' + suffix)(onehot)
    return mu_true, present


def build_loss(mu_true, mu_pred, present, min_vox):

    # present-gated mse over the regressed tissues (a tissue absent from the crop is not scored)
    def fn(x):
        yt, yp, pres = x
        w = K.cast(pres >= float(min_vox), 'float32')
        e = w * K.square(yt - yp)
        return K.expand_dims(K.sum(e, axis=1) / (K.sum(w, axis=1) + eps), -1)
    loss = KL.Lambda(fn, name='tm_loss')([mu_true, mu_pred, present])
    loss._keras_shape = tuple(loss.get_shape().as_list())
    return loss


def labels_to_onehot(labels, lut_np, k, name):

    lut = tf.constant(lut_np, dtype='int32')

    def fn(lab):
        idx = tf.gather(lut, tf.cast(lab[..., 0], 'int32'))
        return tf.one_hot(idx, depth=k, dtype='float32')
    return KL.Lambda(fn, name=name)(labels)


def masked_moment_pool(p, image, name):

    # per-class mean of the image weighted by p: mu_k = sum(p_k * i) / sum(p_k), scale-free in the volume
    def fn(x):
        pp, img = x
        w = K.sum(pp, axis=[1, 2, 3]) + eps
        return K.sum(pp * img, axis=[1, 2, 3]) / w
    return KL.Lambda(fn, name=name)([p, image])


def build_tissue_lut(gen_labels, names):

    # lut[label] = index of the tissue in names, or len(names) for a label outside them (one-hot all zero)
    k = len(names)
    valid = set(int(x) for x in gen_labels)
    lut = np.full(int(gen_labels.max()) + 1, k, dtype='int32')
    for idx, name in enumerate(names):
        for lab in tissue_groups[name]:
            if lab in valid:
                lut[lab] = idx
            else:
                print('  [warn] label %d (%s) is not in generation_labels; skipped' % (lab, name))
    return lut, k


def check_alignment(gen_labels, gen_classes, names):

    # each regressed tissue must sit inside a single generation class, otherwise its mean mixes several
    # draws. the other tissues are printed too (diagnostic) but do not gate the check.
    lab2gen = {int(l): int(c) for l, c in zip(gen_labels, gen_classes)}
    ok = True
    for name in all_tissues:
        classes = sorted(set(lab2gen[l] for l in tissue_groups[name] if l in lab2gen))
        print('  %-21s -> generation class(es) %s' % (name, classes))
        if name in names and len(classes) > 1:
            ok = False
    if not ok:
        raise ValueError('regressed tissue groups are not aligned with the generation classes (see above)')


def load_weights_checked(model, path):

    # load_weights(by_name) matches on layer names and says nothing about the ones that do not match, so an
    # architecture that is not the checkpoint's loads whatever happens to line up and then runs a different
    # net in silence. the two cases keras misses are both real here: fewer levels makes the model's names a
    # subset of the file's, and turning batch norm off drops layers that the file still has. a shared name
    # with a different shape keras does catch, but as 'axes don't match array', which names nothing, so the
    # shapes are compared here too and the layer is named.
    with h5py.File(path, 'r') as f:
        group = f['model_weights'] if 'model_weights' in f else f
        names = [n.decode() if isinstance(n, bytes) else n for n in group.attrs['layer_names']]
        saved = {}
        for n in names:
            weights = [w.decode() if isinstance(w, bytes) else w for w in group[n].attrs.get('weight_names', [])]
            if weights:
                saved[n] = [tuple(group[n][w].shape) for w in weights]
    wanted = {layer.name: [tuple(int(d) for d in w.shape) for w in layer.weights]
              for layer in model.layers if layer.weights}
    missing, extra = sorted(set(wanted) - set(saved)), sorted(set(saved) - set(wanted))
    mismatched = [n for n in sorted(set(wanted) & set(saved)) if wanted[n] != saved[n]]
    if missing or extra or mismatched:
        detail = []
        if missing:
            detail.append('%d layer(s) this model expects are not in the file (%s)'
                          % (len(missing), ', '.join(missing[:4])))
        if extra:
            detail.append('%d layer(s) in the file have nowhere to go in this model (%s)'
                          % (len(extra), ', '.join(extra[:4])))
        if mismatched:
            detail.append('%d layer(s) have a different shape in the file (%s)'
                          % (len(mismatched), ', '.join('%s: %s in the file, %s here'
                                                        % (n, saved[n], wanted[n]) for n in mismatched[:2])))
        raise ValueError('%s does not match this architecture: %s. the architecture arguments have to be '
                         'the ones the checkpoint was trained with.' % (os.path.basename(path), '; '.join(detail)))
    model.load_weights(path, by_name=True)


def train_model(model, generator, learning_rate, n_epochs, n_steps, model_dir, checkpoint, init_epoch,
                clipnorm, prefix='tm'):

    # prepare model and log folders
    utils.mkdir(model_dir)
    log_dir = os.path.join(model_dir, 'logs')
    utils.mkdir(log_dir)

    # one checkpoint per epoch and a tensorboard log. weights only, so a resumed job rebuilds the model
    # from code and loads the weights by name (no need to serialise the generator and lambda layers).
    save_file_name = os.path.join(model_dir, '%s_{epoch:03d}.h5' % prefix)
    callbacks = [KC.ModelCheckpoint(save_file_name, save_weights_only=True, verbose=1),
                 KC.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)]

    if checkpoint is not None:
        load_weights_checked(model, checkpoint)

    # the QC net does not clip its gradients; clipnorm 0 keeps it that way
    optimizer = Adam(lr=learning_rate, clipnorm=clipnorm) if clipnorm else Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=metrics.IdentityLoss().loss)

    # keras's own defaults, one enqueuer thread and a queue of 10, which is what SynthSeg's training loops
    # use: it reads the next label map off disk while the graph runs. a single thread draws from the global
    # numpy stream, so the seed still fixes the sequence.
    model.fit_generator(generator,
                        epochs=n_epochs,
                        steps_per_epoch=n_steps,
                        callbacks=callbacks,
                        initial_epoch=init_epoch)
