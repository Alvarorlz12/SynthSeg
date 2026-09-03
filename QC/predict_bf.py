"""

Deploy the bias-field severity regressor on a real scan: image in, one number out.

This is predict_tm.py with the three-tissue head swapped for the one-channel one. The read is
predict_tm's own preprocess, imported rather than written again, so a volume scored here went through
the pipeline a volume scored by the contrast head goes through: resample to 1 mm, align to RAS, crop
160 around the centre of the volume, normalise p0.5-p99.5 after the crop, pad.

WHAT THE NUMBER IS. The target is std(log B), the standard deviation of the log bias field over the
whole crop, in physical units. Not the b_std the generator draws: the field is sampled at bias_scale
resolution and resized trilinear, and the smoothing costs about a third, so a run with
bias_field_std 0.7 produces targets that stop near 0.45. std_log_max is a declared read-off threshold
and is never applied to the target, so predictions are in the same units.

--preprocessed IS FOR VOLUMES THAT ARE ALREADY PREPARED, and it exists because of one trap. The
corrupted validation set written by make_bf_validation.py is saved after the crop, the normalisation
and the [0,1] ceiling, exactly as the network should see it. Running the normal read over those files
would rescale them a second time, and a p0.5-p99.5 subtracts an offset, which after a multiplicative
field is not the identity: training never does it, so neither does this. On a raw clinical image the
flag is off and the full read applies.

THE OUTPUT IS A SPATIAL MEAN, as in predict_tm: the head ends in a mean over x, y and z, so the net
accepts any shape and answers differently on a larger window. Training cropped to 160, so 160 is the
default, and changing it changes the number.

Usage:
  # deployment
  python scripts/commands/predict_bf.py <images> preds.csv models/biasfield/bf_073.h5

  # the frozen validation set, with its manufactured ground truth
  python scripts/commands/predict_bf.py <data>/qc-data/validation/img_bf preds.csv <model.h5> \
      --preprocessed --gt_csv <data>/qc-data/validation/gt/bf/targets.csv

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
import csv
import numpy as np

# project imports
from SynthSeg.predict import write_csv
from QC.predict_tm import preprocess, prepare_output_files
from QC.training_tm import load_weights_checked
from QC import training_bf as bf

# third-party imports
from ext.lab2im import utils


def predict_bf(path_images,
               path_out,
               path_model,
               gt_csv=None,
               path_resampled=None,
               cropping=160,
               target_res=1.,
               preprocessed=False,
               n_levels=5,
               nb_conv_per_level=3,
               conv_size=5,
               unet_feat_count=24,
               feat_multiplier=2,
               activation='relu',
               norm='instance',
               recompute=True,
               verbose=True):
    """
    Predict the severity of the bias field a real image carries.

    :param path_images: path of an image, a folder of images, or a text file listing them, the three
    forms predict.py accepts.
    :param path_out: path of the output csv. One row per image.
    :param path_model: path of the regressor checkpoint (a bf_*.h5).

    :param gt_csv: (optional) csv of manufactured ground truths, as make_bf_validation.py writes it:
    a 'stem' column and a 'std_log_bias' column. Turns on the true and error columns. An image with no
    row there is scored and left blank rather than dropped.
    :param path_resampled: (optional) folder/path where the resampled images are written, as predict.py
    writes them. Ignored when preprocessed is on, since nothing is resampled.

    :param cropping: (optional) the window the network sees, cropped around the centre of the volume.
    Default is 160, the shape training cropped to. The head is a spatial mean, so this is part of the
    measurement and not a detail.
    :param target_res: (optional) resolution the image is resampled to first. Default is 1.
    :param preprocessed: (optional) the images are already resampled, cropped and normalised, so they
    are fed as they are. Default is False. See the header for why this is not merely a shortcut.

    :param n_levels: (optional) number of levels of the encoder. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutions per level. Default is 3.
    :param conv_size: (optional) size of the convolution kernels. Default is 5.
    :param unet_feat_count: (optional) number of features at the first level. Default is 24.
    :param feat_multiplier: (optional) feature multiplier between levels. Default is 2.
    :param activation: (optional) activation function. Default is 'relu'.
    :param norm: (optional) the normalisation the checkpoint was TRAINED with, among 'instance',
    'batch' and 'none'. It is an architecture argument: a mismatch is refused by load_weights_checked
    rather than loaded half way. Default is 'instance'.
    :param recompute: (optional) whether to overwrite an existing output csv. Default is True.
    :param verbose: (optional) print one line per image. Default is True.
    """

    # prepare input/output filepaths
    path_images, path_out, _, path_resampled = \
        prepare_output_files(path_images, path_out, None, path_resampled)
    if (not recompute) & os.path.isfile(path_out):
        print('%s already exists and recompute is off, nothing to do' % path_out)
        return

    truths = read_gt_csv(gt_csv)

    header = ['std_log_bias']
    if truths is not None:
        header += ['true_std_log_bias', 'abs_err']
    write_csv(path_out, None, True, np.arange(len(header)), np.array(header), skip_first=False)

    # the input shape is left free on the three spatial axes, as predict_tm leaves it: the head is a
    # spatial mean, so one build serves every image.
    net = build_bf_model(path_model=path_model,
                         input_shape=[None] * 3 + [1],
                         n_levels=n_levels,
                         nb_conv_per_level=nb_conv_per_level,
                         conv_size=conv_size,
                         unet_feat_count=unet_feat_count,
                         feat_multiplier=feat_multiplier,
                         activation=activation,
                         norm=norm)

    if cropping is not None:
        cropping = utils.reformat_to_list(cropping, length=3, dtype='int')
        min_pad = cropping
    else:
        min_pad = 128

    loop_info = utils.LoopInfo(len(path_images), 1 if len(path_images) <= 10 else 10, 'predicting', True)
    for i in range(len(path_images)):
        if verbose:
            loop_info.update(i)

        if preprocessed:
            image = load_prepared(path_images[i], n_levels)
        else:
            image = preprocess(path_image=path_images[i], n_levels=n_levels, target_res=target_res,
                               crop=cropping, min_pad=min_pad, path_resample=path_resampled[i])[0]

        pred = float(np.squeeze(np.asarray(net.predict(image))))

        stem = os.path.basename(path_images[i]).split('.nii')[0].replace('.mgz', '')
        row = [stem, '%.6f' % pred]
        if truths is not None:
            t = truths.get(stem)
            row += ['' if t is None else '%.6f' % t, '' if t is None else '%.6f' % abs(pred - t)]
            if t is None:
                print('  [warn] %s: no row in the ground-truth csv, left blank' % stem)
        write_csv(path_out, row, True, np.arange(len(header)), np.array(header), skip_first=False)

    print('\nwrote %s' % path_out)


def read_gt_csv(path):
    """stem -> std(log B). None when there is no ground truth to join."""
    if path is None:
        return None
    assert os.path.isfile(path), 'no such ground-truth csv: %s' % path
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert rows, 'empty ground-truth csv: %s' % path
    for col in ('stem', 'std_log_bias'):
        assert col in rows[0], 'the ground-truth csv needs a %r column, had %s' % (col, list(rows[0]))
    return {r['stem']: float(r['std_log_bias']) for r in rows}


def load_prepared(path_image, n_levels):
    """A volume that is already resampled, cropped and normalised, read and handed over untouched."""
    im, _, aff, n_dims, n_channels, _, _ = utils.get_volume_info(path_image, True)
    if n_dims == 4 and n_channels == 1:
        n_dims, im = 3, im[..., 0]
    assert n_dims == 3, 'input should have 3 dimensions, had %s' % n_dims
    div = 2 ** n_levels
    assert all(s % div == 0 for s in im.shape[:3]), \
        '%s has shape %s, not divisible by %d: it is not a prepared volume' % (path_image, im.shape, div)
    return utils.add_axis(im, axis=[0, -1])


def build_bf_model(path_model, input_shape, n_levels, nb_conv_per_level, conv_size, unet_feat_count,
                   feat_multiplier, activation, norm):
    """predict_tm's build, with the bias head's own graph imported rather than rebuilt. Its layers are
    named bf_*, so a checkpoint from another head is refused instead of loaded by whatever lines up."""

    assert os.path.isfile(path_model), 'The provided model path does not exist.'

    import keras.layers as KL
    import keras.models as KM

    instance_norm = (norm == 'instance')
    batch_norm = -1 if norm == 'batch' else None
    print('architecture: norm=%s' % norm)

    img_in = KL.Input(shape=input_shape, name='val_image_input')
    stand_in = KM.Model(img_in, img_in)
    y = bf.build_regression_model(stand_in, input_shape, 1, n_levels, nb_conv_per_level, conv_size,
                                  unet_feat_count, feat_multiplier, activation, batch_norm, True,
                                  instance_norm)
    net = KM.Model(stand_in.inputs, [y])
    load_weights_checked(net, path_model)
    return net
