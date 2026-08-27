"""

This function trains a regressor network to predict the mean intensity of each tissue (csf / gm / wm) on
the synthetic image, measured on the final normalised image the network sees (in [0, 1]) over the true
tissue masks. the target is well defined because the generation ties each tissue to a single gaussian
(grouped generation_classes), so the mean of a tissue is one intensity rather than an average of several
draws.

it is a blind regressor: it reads only the image and outputs one number per tissue, it does not segment.
the net is the synthseg dice qc net (training_qc.py) kept as close to identical as the target allows: same
generator, same encoder (five levels, batch norm on, residuals on, relu), same head bar the width of its
first convolution, same in-graph loss, same schedule, one checkpoint per epoch, tensorboard, resume from a
checkpoint.

it departs from that net in four places, all forced by the target rather than chosen:

  1. the input is the image (one channel), not a segmentation (one channel per label).
  2. the generator is labels_to_image_model (it paints an image), not the label-deformation model.
  3. the target is a per-tissue mean of the image (masked pool), not a dice score against a noisy input.
  4. a consequence of 1: conv_enc only builds the residual projection when the input has more than one
     channel (ext/neuron/models.py:334), so at level 0 there is no projection and keras broadcasts the raw
     image onto all 24 feature maps. levels 1-4 are the learned residual the dice qc net has everywhere.
     this is left as it is: the degeneracy comes from the one-channel input, not from a decision here.

two things to watch, both consequences of keeping batch norm on at batchsize 1. it is then instance norm
(one image's statistics per channel), and conv_enc puts a batch norm at the end of every level including
the last, so what the head reads is already normalised, while the target is precisely the per-image level.
and batch norm is the only layer here that behaves differently in training and in inference, so the loss
(batch statistics) and the val_loss (moving averages) are two different functions of the same weights: read
each curve against var(target), not the gap between them.

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

# fixed order 0=csf, 1=gm, 2=wm. each group must sit inside a single generation class, otherwise its mean
# would mix intensities from several draws.
all_tissues = ['CSF', 'GM', 'WM']
tissue_groups = {
    'CSF': [4, 5, 43, 44, 14, 15, 24, 72],   # lateral + inf-lateral ventricles l/r, 3rd/4th/5th, extra-cerebral csf
    'GM': [3, 42, 8, 47],                     # cerebral + cerebellar cortex l/r
    'WM': [2, 41, 7, 46],                     # cerebral + cerebellar white matter l/r
}


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
             validation_steps=100,
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
    :param holdout: (optional) number of label maps kept out of training, i.e. the anatomy the validation
    loss is measured on. The split is deterministic on the sorted paths, so the evaluation script agrees on
    which maps were never seen: change it in both or the val curve and the offline probe stop meaning the
    same thing. Default is 4, i.e. 16 for training out of the 20 in the repo.
    :param batchsize: (optional) number of images per minibatch. Default is 1.
    :param output_shape: (optional) shape of the cropped output image. Default is 160.

    # spatial deformation (same defaults as the dice qc net; pass False / 0 to turn a term off)
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

    # architecture (the dice qc net's, unchanged; note n_levels sets the downsampling, output_shape does not:
    # conv_enc pools after every level but the last, so the encoder divides the volume by 2 ** (n_levels - 1))
    :param n_levels: (optional) number of levels of the encoder. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutions per level. Default is 3.
    :param conv_size: (optional) size of the convolution kernels. Default is 5.
    :param unet_feat_count: (optional) number of features at the first level. Default is 24.
    :param feat_multiplier: (optional) feature multiplier between levels. Default is 2.
    :param activation: (optional) activation function. Default is 'relu'.
    :param batch_norm: (optional) axis to batch normalise, or None to turn batch norm off. It is an axis, not
    a flag: -1 is the feature axis. Default is -1.
    :param use_residuals: (optional) residual connection per level. Default is True.

    # training
    :param lr: (optional) learning rate. Default is 1e-4.
    :param clipnorm: (optional) gradient norm clipping, 0 to turn it off (the dice qc net does not clip).
    Default is 0.
    :param epochs: (optional) number of epochs. Default is 100.
    :param steps_per_epoch: (optional) steps per epoch, i.e. how often the model is saved. Default is 1000.
    :param validation_steps: (optional) images drawn from the held-out label maps at the end of each epoch to
    log a val_loss next to the loss. 0 turns it off. Default is 100.
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

    # the fitted model outputs the loss and nothing else, so the predicted and true means are not readable
    # from it. this second model is the same graph read at a different point: same layer objects, so the same
    # weights, no copy and nothing to keep in sync. the loss tensor is one of its outputs, which is what lets
    # the validation callback report the mse of the graph itself rather than a reimplementation of it, and
    # keep the per-tissue predictions from the same forward pass.
    val_probe = models.Model(generator.inputs, [mu_pred, mu_true, present, loss])
    n_train = int(np.sum([K.count_params(w) for w in regression_model.trainable_weights]))
    print('regressing: %s   trainable params: %d' % (', '.join(names), n_train))

    # input generators. the held-out maps feed a val_loss at the end of each epoch: every image is drawn
    # fresh here, so the loss is already out of sample in contrast and the held-out maps only add unseen
    # anatomy, which is what the val curve measures.
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
    # the intensity regime, printed because it is what the target distribution depends on and the log is the
    # only place a finished run can be asked what it was trained on.
    print('  intensity regime: randomise_res %s   bias_std %.2f (prob %.2f)   gamma_std %.2f (prob %.2f)'
          % (randomise_res, bias_field_std, bias_prob if bias_field_std > 0 else 0.,
             gamma_std, gamma_prob if gamma_std > 0 else 0.))

    train_model(regression_model, input_generator, lr, epochs, steps_per_epoch, model_dir, checkpoint,
                init_epoch, clipnorm, val_generator, n_val, val_probe, names)


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
                           feat_multiplier, activation, batch_norm, use_residuals, instance_norm=False, qc_head=False):

    # conv encoder on the image, then the dice-qc head: max pool, two convolutions, and average over space,
    # which keeps the location until the output.
    enc = nrn_models.conv_enc(input_model=generator, input_shape=image_shape, nb_levels=n_levels,
                              conv_size=conv_size, nb_features=feat_count, feat_mult=feat_multiplier,
                              nb_conv_per_level=nb_conv_per_level, activation=activation,
                              batch_norm=batch_norm, instance_norm=instance_norm,
                              use_residuals=use_residuals, name='tm_enc')
    last = enc.outputs[0]
    conv_kwargs = {'padding': 'same', 'activation': 'relu', 'data_format': 'channels_last'}
    last = KL.MaxPool3D(pool_size=(2, 2, 2), padding='same', name='tm_conv_pool')(last)

    # qc_head keeps the dice qc net's head EXACTLY: k channels in both convolutions and relu on both. That is
    # the right head whenever the target is >= 0 and 0 is a value it really takes -- a dice score, the
    # resolution deficit (0 on a native volume), the bias severity std(B) (exactly 0 on the ~10% of images the
    # bias draw skips). The two deviations in the else branch were argued for the tissue-means target alone,
    # which sits near 0.5 and never reaches 0; carrying them to a dice-shaped target changes the net for
    # nothing. Distinct layer names on purpose: a checkpoint from the other head then has layers with nowhere
    # to go and load_weights_checked refuses it out loud instead of loading half a net.
    if qc_head:
        last = KL.Conv3D(k, kernel_size=5, **conv_kwargs, name='tm_qc_conv0')(last)
        last = KL.Conv3D(k, kernel_size=5, **conv_kwargs, name='tm_qc_conv1')(last)
        return KL.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2, 3]), name='tm_pred')(last)

    # the head's first convolution is the one place it is wider than the dice qc net's, which emits one
    # channel per label in both of its head convs: tying ours to k the same way would make it a rank-1
    # readout whenever a single tissue is regressed.
    last = KL.Conv3D(max(16, k), kernel_size=5, **conv_kwargs, name='tm_conv0')(last)
    # the last convolution is linear where the dice qc net's is relu, and this is the one place its head
    # could not be copied. relu is right for a dice score: it is >= 0 and 0 is a value it really takes, when
    # a label is absent. our target is a mean of a min-max normalised image, it sits near 0.5 and never goes
    # near 0, so the relu prices in a floor the target never touches while keeping its zero-gradient region.
    # with relu here the whole map of a channel can start out negative, and then the mean of it is exactly
    # 0, the gradient is exactly 0, and that tissue never recovers.
    last = KL.Conv3D(k, kernel_size=5, padding='same', activation=None, name='tm_conv1')(last)
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
        print('  %-3s -> generation class(es) %s' % (name, classes))
        if name in names and len(classes) > 1:
            ok = False
    if not ok:
        raise ValueError('regressed tissue groups are not aligned with the generation classes (see above)')


def load_weights_checked(model, path):

    # load_weights(by_name) matches on layer names and says nothing about the ones that do not match, so an
    # architecture that is not the checkpoint's loads whatever happens to line up and then runs a different
    # net in silence. keras only catches the case where a shared name has a different shape. the two it
    # misses are both real here: fewer levels makes the model's names a subset of the file's, and turning
    # batch norm off drops layers that the file still has. require the two sets to be the same.
    with h5py.File(path, 'r') as f:
        group = f['model_weights'] if 'model_weights' in f else f
        names = [n.decode() if isinstance(n, bytes) else n for n in group.attrs['layer_names']]
        saved = set(n for n in names if len(group[n].attrs.get('weight_names', [])))
    wanted = set(layer.name for layer in model.layers if layer.weights)
    missing, extra = sorted(wanted - saved), sorted(saved - wanted)
    if missing or extra:
        detail = []
        if missing:
            detail.append('%d layer(s) this model expects are not in the file (%s)'
                          % (len(missing), ', '.join(missing[:4])))
        if extra:
            detail.append('%d layer(s) in the file have nowhere to go in this model (%s)'
                          % (len(extra), ', '.join(extra[:4])))
        raise ValueError('%s does not match this architecture: %s. the architecture arguments have to be '
                         'the ones the checkpoint was trained with.' % (os.path.basename(path), '; '.join(detail)))
    model.load_weights(path, by_name=True)


class ValLoss(KC.Callback):
    """Mean loss over n images drawn from the held-out label maps, at the end of every epoch, plus the
    per-tissue predicted and true means those images gave, saved next to the checkpoint.

    keras computes a val_loss itself if fit_generator is given validation_data, but its evaluate_generator
    runs every step and then returns the last one (keras/engine/training_generator.py:420 takes
    outs_per_batch[-1], which is only a running mean for a stateful metric, and the loss here is stateless).
    that would put a one-image val_loss next to a loss averaged over a thousand steps, on the same line, and
    the val curve is the point of the run. so average it here, the way validate_qc.py averages its own scores.

    the per-epoch npz is what lets the predictions be looked at as a distribution afterwards (predicted vs
    true, per tissue) instead of only as a summary number. it is the cheap half of the run: the arrays are a
    few hundred floats an epoch, they cannot be recovered once the run is over, and a collapsed regressor is
    something you can see in them directly, without going through a correlation.
    """

    def __init__(self, probe, generator, steps, model_dir, names):
        self.probe = probe
        self.generator = generator
        self.steps = steps
        self.model_dir = model_dir
        self.names = names
        # a batch norm at batchsize 1 normalises each image by its own statistics, so what training fits is
        # an instance-norm net, and that is the function to score. keras's predict runs the other branch,
        # the one with the moving averages, which applies one fixed normalisation to every image: under
        # randomised contrast the per-image statistics are exactly what varies, so that branch is a
        # different function of the same weights. build one function with the learning phase as an explicit
        # input to read both. no updates are passed, so neither call moves the moving averages.
        self.fn = K.function(probe.inputs + [K.learning_phase()], probe.outputs)
        super(ValLoss, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        pred, true, present, losses, frozen = [], [], [], [], []
        for _ in range(self.steps):
            inputs, _ = next(self.generator)
            mu_p, mu_t, pres, loss = self.fn(inputs + [1])   # the net that was trained
            pred.append(mu_p[0])
            true.append(mu_t[0])
            present.append(pres[0])
            losses.append(float(loss[0, 0]))
            # the same graph read the way predict would read it. the generator redraws, so this is a fresh
            # image rather than the same one: it is a second estimate of the same quantity, not a pair.
            frozen.append(float(self.fn(inputs + [0])[3][0, 0]))
        pred, true, present = np.array(pred), np.array(true), np.array(present)
        logs['val_loss'] = float(np.mean(losses))
        logs['val_loss_frozen_bn'] = float(np.mean(frozen))

        np.savez(os.path.join(self.model_dir, 'val_%03d.npz' % (epoch + 1)), pred=pred, true=true,
                 present=present, loss=np.array(losses), loss_frozen_bn=np.array(frozen),
                 tissues=np.array(self.names))

        # keras builds the progress bar before this callback, so it will not show val_loss. print it, or the
        # only place it exists is the tensorboard log and the job's own log never mentions it. the variance of
        # the target goes next to it because the mse only means something against it: a net that has given up
        # and predicts the mean of the target scores mse = var(target), so that is the line to beat, and the
        # spread of the predictions against the spread of the truth says the same thing a second way.
        print('Epoch %05d: val_loss (mse) %.5f   [reference: var(target) %.5f = the score of predicting the '
              'mean]' % (epoch + 1, logs['val_loss'], float(np.mean(true.var(axis=0)))))
        print('           (same weights read with frozen batch norm, as predict would: %.5f)'
              % logs['val_loss_frozen_bn'])
        for j, name in enumerate(self.names):
            print('           %-3s  pred %.3f +- %.3f   true %.3f +- %.3f'
                  % (name, pred[:, j].mean(), pred[:, j].std(), true[:, j].mean(), true[:, j].std()))


def train_model(model, generator, learning_rate, n_epochs, n_steps, model_dir, checkpoint, init_epoch,
                clipnorm, val_generator=None, validation_steps=0, val_probe=None, names=(), prefix='tm'):

    # prepare model and log folders
    utils.mkdir(model_dir)
    log_dir = os.path.join(model_dir, 'logs')
    utils.mkdir(log_dir)

    # one checkpoint per epoch and a tensorboard log. weights only, so a resumed job rebuilds the model
    # from code and loads the weights by name (no need to serialise the generator and lambda layers).
    # ValLoss has to come before the tensorboard callback: it writes val_loss into the epoch's logs and the
    # tensorboard callback is what reads it back out, strips the val_ and files it under 'epoch_loss' in
    # logs/validation next to the training one in logs/train. draw_learning_curve plots both as they are.
    save_file_name = os.path.join(model_dir, '%s_{epoch:03d}.h5' % prefix)
    callbacks = [KC.ModelCheckpoint(save_file_name, save_weights_only=True, verbose=1)]
    if val_generator is not None and validation_steps > 0:
        callbacks.append(ValLoss(val_probe, val_generator, validation_steps, model_dir, names))
    callbacks.append(KC.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False))

    if checkpoint is not None:
        load_weights_checked(model, checkpoint)

    # the dice qc net does not clip its gradients; clipnorm 0 keeps it that way
    optimizer = Adam(lr=learning_rate, clipnorm=clipnorm) if clipnorm else Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=metrics.IdentityLoss().loss)

    # the validation stream is the ValLoss callback, not validation_data (see ValLoss for why). workers=0
    # runs the generator on the main thread: it shares the global numpy stream that picks the label map and
    # draws the gaussians with the validation one, so an enqueuer thread would interleave the two and make
    # the seed meaningless.
    model.fit_generator(generator,
                        epochs=n_epochs,
                        steps_per_epoch=n_steps,
                        callbacks=callbacks,
                        initial_epoch=init_epoch,
                        workers=0)
