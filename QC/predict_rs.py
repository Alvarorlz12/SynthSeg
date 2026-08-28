"""

Deploy the per-axis resolution regressor on a real scan: image in, three spacings out.

This is predict_tm.py with the tissue-means head swapped for the resolution head, which is itself
predict.py with the segmentation head swapped out. Read, resample to 1 mm, align to RAS, crop,
normalise, pad: the same calls in the same order with the same constants, so a volume scored here went
through the pipeline a volume segmented by SynthSeg goes through.

The ground truth is the header. A real scan carries its own per-axis spacing, so there is no --gt and
nothing here needs a segmentation or a second folder. The header is read before the resampling, since
afterwards it says 1 mm on every axis whatever the scan was acquired at. It records the grid the file
is stored on while the network measures the content, so the true_* columns are a proxy for the
acquisition rather than a gold standard.

The resampling to 1 mm is mandatory. Training degrades the content to a spacing s and leaves the volume
on the 1 mm grid it was generated on, which is why the target is a deficit s - atlas_res rather than a
spacing outright: a 3 mm scan stored on a 3 mm grid is not in that domain, the same scan resampled to
1 mm is. --no_resample hands the network a domain it never saw, and exists to measure that.

The three outputs are array axes, and the header's spacings are not in that order. The array is aligned
to RAS, so the network's axes are R, A, S, while the header holds the acquisition order: a sagittal
FLAIR at 1x1x3 mm has its 3 mm on whichever array axis the scanner wrote it to, and comparing without
permuting scores the right number against the wrong axis. The permutation is the same two lines
utils.get_volume_info runs, written out per image in the ras_axes column, where '012' is the identity.

The normalisation is predict.py's p0.5-p99.5, the divisor predict_tm uses and the one SynthSeg deploys
with. Training ends on an exact min-max instead, and --minmax_norm measures that gap.

The output is a spatial mean: the head averages over x, y and z, so the graph accepts any shape and
answers differently for each. Training cropped to 160^3 and 160 is the default, so --cropping is part
of the measurement.

Usage:
  # one image, or a folder, or a text file listing them
  python scripts/commands/predict_rs.py <images> preds.csv models/resolution/rs_extra_uniform_mm_a/rs_033.h5

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

# project imports
from SynthSeg.predict import write_csv
from QC import training_rs as rs

# third-party imports
from ext.lab2im import utils
from ext.lab2im import edit_volumes


# the array axes after the alignment to RAS, which is the order the network's three outputs come in.
AXES = ['R', 'A', 'S']


def predict_rs(path_images,
               path_out,
               path_model,
               path_resampled=None,
               cropping=160,
               target_res=1.,
               minmax_norm=False,
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
    Predict the per-axis voxel spacing the content of a real image sits at.

    :param path_images: path of an image, a folder of images, or a text file listing them, the three
    forms predict.py accepts.
    :param path_out: path of the output csv. One row per image.
    :param path_model: path of the regressor checkpoint (an rs_*.h5).

    :param path_resampled: (optional) folder/path where the resampled images are written, as predict.py
    writes them.

    :param cropping: (optional) the window the network sees, cropped around the centre of the volume as
    predict.py crops it and padded up to the same size when the head is smaller. Default is 160, the
    shape training cropped to. It is the only window knob: min_pad follows it. Rounded up to a multiple
    of 2 ** n_levels. None leaves the volume uncropped and pads to at least 128, SynthSeg's default.
    :param target_res: (optional) resolution the image is resampled to before anything else, and the
    grid the deficit is read against: the network predicts s - target_res and this reads it back as
    s = target_res + prediction. Default is 1., the grid training left its degraded volumes on.
    :param minmax_norm: (optional) normalise with an exact min-max instead of predict.py's p0.5-p99.5.
    Default is False, i.e. the percentile predict_tm and SynthSeg deploy with; the min-max is what
    training ends on, so this flag measures that gap.

    :param n_levels: (optional) number of levels of the encoder. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutions per level. Default is 3.
    :param conv_size: (optional) size of the convolution kernels. Default is 5.
    :param unet_feat_count: (optional) number of features at the first level. Default is 24.
    :param feat_multiplier: (optional) feature multiplier between levels. Default is 2.
    :param activation: (optional) activation function. Default is 'relu'.
    :param norm: (optional) the normalisation the checkpoint was trained with, among 'instance', 'batch'
    and 'none'. It is an architecture argument: a 'none' checkpoint holds no rs_enc_in_down_* layers,
    and a mismatch is refused rather than silently loaded. Default is 'instance'.
    :param recompute: (optional) whether to overwrite an existing output csv. Default is True.
    :param verbose: (optional) print one line per image. Default is True.
    """

    # prepare input/output filepaths
    path_images, path_out, path_resampled = prepare_output_files(path_images, path_out, path_resampled)
    if (not recompute) & os.path.isfile(path_out):
        print('%s already exists and recompute is off, nothing to do' % path_out)
        return

    # the grid the deficit is read against. None keeps each volume on its own grid, and then the
    # per-image spacing is what gets added back.
    atlas_res = None if target_res is None else \
        np.squeeze(utils.reformat_to_n_channels_array(target_res, 3)).astype('float')

    # write the csv header now, so a run that dies halfway still leaves a readable file. the true_*
    # columns are always written: they come out of the header, which every image carries.
    header = ['s_%s' % a for a in AXES] + ['true_%s' % a for a in AXES] + ['err_%s' % a for a in AXES]
    header += ['abs_err', 'ras_axes']
    write_csv(path_out, None, True, np.arange(len(header)), np.array(header), skip_first=False)

    # the input shape is left free on the three spatial axes, as predict.py leaves it: the head is a spatial
    # mean, so one build serves every image.
    net = build_rs_model(path_model=path_model,
                         input_shape=[None] * 3 + [1],
                         n_levels=n_levels,
                         nb_conv_per_level=nb_conv_per_level,
                         conv_size=conv_size,
                         unet_feat_count=unet_feat_count,
                         feat_multiplier=feat_multiplier,
                         activation=activation,
                         norm=norm)

    # the padding follows the crop, so the network sees the window --cropping asks for; a larger min_pad
    # would be clamped.
    if cropping is not None:
        cropping = utils.reformat_to_list(cropping, length=3, dtype='int')
        min_pad = cropping
    else:
        min_pad = 128

    # perform prediction
    if len(path_images) <= 10:
        loop_info = utils.LoopInfo(len(path_images), 1, 'predicting', True)
    else:
        loop_info = utils.LoopInfo(len(path_images), 10, 'predicting', True)
    for i in range(len(path_images)):
        if verbose:
            loop_info.update(i)

        # preprocessing. res_true is the header's spacing, permuted into the network's axis order and read
        # before the resampling.
        image, res_true, ras_axes = preprocess(path_image=path_images[i],
                                     n_levels=n_levels,
                                     target_res=target_res,
                                     crop=cropping,
                                     min_pad=min_pad,
                                     minmax_norm=minmax_norm,
                                     path_resample=path_resampled[i])

        # prediction. the net returns the deficit s - atlas_res, kept at or above 0 by the relu on the
        # last head conv; read it back as a spacing on the grid the image now sits on.
        deficit = np.asarray(net.predict(image))[0]
        s_pred = (atlas_res if atlas_res is not None else res_true) + deficit
        err = np.abs(s_pred - res_true)

        row = [os.path.basename(path_images[i]).replace('.nii.gz', '').replace('.nii', '').replace('.mgz', '')]
        row += ['%.6f' % v for v in s_pred]
        row += ['%.6f' % v for v in res_true]
        row += ['%.6f' % v for v in err]
        row += ['%.6f' % float(err.mean())]
        # the permutation that was applied, as a string so a spreadsheet cannot read it as a number:
        # '012' is the identity, '201' is the R spacing sitting on array axis 2.
        row += [''.join(str(a) for a in ras_axes)]

        # write results to disk
        write_csv(path_out, row, True, np.arange(len(header)), np.array(header), skip_first=False)

    print('\nwrote %s' % path_out)


def prepare_output_files(path_images, out_csv, out_resampled):
    """predict_tm's, with the ground-truth branch dropped: the truth is the header, so there is no
    second folder to pair."""

    # check inputs
    assert path_images is not None, 'please specify an input file/folder (--i)'
    assert out_csv is not None, 'please specify an output csv file (--o)'

    # convert path to absolute paths
    path_images = os.path.abspath(path_images)
    basename = os.path.basename(path_images)
    out_csv = os.path.abspath(out_csv)
    out_resampled = os.path.abspath(out_resampled) if (out_resampled is not None) else out_resampled

    if out_csv[-4:] != '.csv':
        print('output provided without csv extension. Adding csv extension.')
        out_csv += '.csv'
    utils.mkdir(os.path.dirname(out_csv))

    # path_images is a text file
    if basename[-4:] == '.txt':
        if not os.path.isfile(path_images):
            raise Exception('provided text file containing paths of input images does not exist: %s' % path_images)
        with open(path_images, 'r') as f:
            path_images = [line.replace('\n', '') for line in f.readlines() if line != '\n']

    # path_images is a folder
    elif ('.nii.gz' not in basename) & ('.nii' not in basename) & ('.mgz' not in basename) & ('.npz' not in basename):
        if os.path.isfile(path_images):
            raise Exception('Extension not supported for %s, only use: nii.gz, .nii, .mgz, or .npz' % path_images)
        path_images = utils.list_images_in_folder(path_images)

    # path_images is an image
    else:
        assert os.path.isfile(path_images), 'file does not exist: %s \n' \
                                            'please make sure the path and the extension are correct' % path_images
        path_images = [path_images]

    # resampled volumes, named as predict.py names them
    if out_resampled is not None:
        if (out_resampled[-7:] == '.nii.gz') | (out_resampled[-4:] == '.nii') | (out_resampled[-4:] == '.mgz'):
            assert len(path_images) == 1, 'resampled path had a file extension but there are %d input images; ' \
                                          'give a folder' % len(path_images)
            path_resampled = [out_resampled]
            utils.mkdir(os.path.dirname(out_resampled))
        else:
            path_resampled = [os.path.join(out_resampled, os.path.basename(p)) for p in path_images]
            path_resampled = [p.replace('.nii', '_resampled.nii').replace('.mgz', '_resampled.mgz')
                              for p in path_resampled]
            utils.mkdir(out_resampled)
    else:
        path_resampled = [None] * len(path_images)

    return path_images, out_csv, path_resampled


def preprocess(path_image, n_levels, target_res, crop=None, min_pad=None, minmax_norm=False,
               path_resample=None):
    """predict_tm's, minus the second volume, with the header's spacing captured before the resampling
    and permuted into the network's axis order."""

    # read image and corresponding info
    im, _, aff, n_dims, n_channels, h, im_res = utils.get_volume_info(path_image, True)
    if n_dims == 4 and n_channels == 1:
        n_dims = 3
        im = im[..., 0]
    assert n_dims == 3, 'input should have 3 dimensions, had %s' % n_dims
    if n_channels > 1:
        print('WARNING: detected more than 1 channel, only keeping the first channel.')
        im = im[..., 0]

    # the target, read here because after the resampling below the header says 1 mm on every axis
    # whatever the scan was acquired at. permuted into the RAS order the alignment further down puts
    # the array in, with the same two lines utils.get_volume_info runs when given an aff_ref.
    ras_axes = edit_volumes.get_ras_axes(aff, n_dims=n_dims)
    ras_axes_ref = edit_volumes.get_ras_axes(np.eye(4), n_dims=n_dims)
    res_true = np.array(im_res, dtype='float')
    res_true[ras_axes_ref] = res_true[ras_axes]

    # resample image if necessary. this is what puts a real scan in the training domain: coarse content
    # on a 1 mm grid, which is what the deficit is defined against.
    if target_res is not None:
        target_res = np.squeeze(utils.reformat_to_n_channels_array(target_res, n_dims))
        if np.any((im_res > target_res + 0.05) | (im_res < target_res - 0.05)):
            im_res = target_res
            im, aff = edit_volumes.resample_volume(im, aff, im_res)
            if path_resample is not None:
                utils.save_volume(im, aff, h, path_resample)

    # align image
    im = edit_volumes.align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)

    # crop image if necessary, centred on the volume, which is all a deployment has
    if crop is not None:
        crop = utils.reformat_to_list(crop, length=n_dims, dtype='int')
        crop_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in crop]
        im = edit_volumes.crop_volume(im, cropping_shape=crop_shape)

    # normalise, after the crop so the divisor is read off the window the network sees. p0.5-p99.5, as
    # predict.py and predict_tm do; --minmax_norm swaps in the exact min-max training ends on.
    if minmax_norm:
        im = edit_volumes.rescale_volume(im, new_min=0., new_max=1., min_percentile=0, max_percentile=100)
    else:
        im = edit_volumes.rescale_volume(im, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5)

    # pad image
    input_shape = im.shape[:n_dims]
    pad_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in input_shape]
    if min_pad is not None:
        min_pad = utils.reformat_to_list(min_pad, length=n_dims, dtype='int')
        min_pad = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in min_pad]
        pad_shape = np.maximum(pad_shape, min_pad)
    im = edit_volumes.pad_volume(im, padding_shape=pad_shape)

    # add batch and channel axes
    im = utils.add_axis(im, axis=[0, -1])

    return im, res_true, ras_axes


def build_rs_model(path_model, input_shape, n_levels, nb_conv_per_level, conv_size, unet_feat_count,
                   feat_multiplier, activation, norm):
    """predict_tm's build_tm_model, with the resolution head's own graph imported rather than rebuilt.

    k is 3 and not a parameter: a checkpoint with another width is a checkpoint of another head.
    """

    assert os.path.isfile(path_model), "The provided model path does not exist."

    import keras.layers as KL
    import keras.models as KM

    # norm builds the graph: with norm='none' there are no rs_enc_in_down_* layers at all, so a
    # mismatched checkpoint is rejected by load_weights_checked instead of loading by name.
    instance_norm = (norm == 'instance')
    batch_norm = -1 if norm == 'batch' else None
    print('architecture: norm=%s' % norm)

    img_in = KL.Input(shape=input_shape, name='val_image_input')
    stand_in = KM.Model(img_in, img_in)
    y = rs.build_regression_model(stand_in, input_shape, 3, n_levels, nb_conv_per_level, conv_size,
                                  unet_feat_count, feat_multiplier, activation, batch_norm, True,
                                  instance_norm)
    net = KM.Model(stand_in.inputs, [y])
    rs.load_weights_checked(net, path_model)
    return net
