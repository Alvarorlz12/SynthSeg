"""

Deploy the tissue-means regressor on a real scan: image in, one row of numbers out.

This is predict.py with the segmentation head swapped for the regressor. Read, resample to 1 mm, align
to RAS, crop, normalise, pad: all of it calls edit_volumes in predict.py's order and with predict.py's
constants, so a volume scored here went through the pipeline a volume segmented by SynthSeg goes
through. Two things predict.py does not do are taken from elsewhere in the repository rather than
written again: carrying a second volume through the identical crop and pad (predict_group), and the
regressor's own graph and checkpoint guard (QC/training_tm.py).

THE SEGMENTATION NEVER ENTERS THE PREPROCESSING. It is read only for the ground-truth columns; the
window comes from the volume alone, so nothing here needs a segmentation to run. Centring the crop on
the region a segmentation finds, the way predict_qc does, was dropped on purpose: a figure measured
with a brain centre that deployment does not have is not a measurement of what will be deployed.

Three things worth knowing before reading a number out of this.

THE OUTPUT IS A SPATIAL MEAN. The head ends in a mean over x, y and z, so the network is fully
convolutional and will accept a 256^3 conformed head without complaining -- and answer differently,
because the window it averages over then holds far more air. Training cropped to 160^3, so 160 is the
default. Change it and the number changes; that is a property of the head, not a bug.

There is ONE window knob, --cropping. min_pad follows it and could not do otherwise, since predict.py
caps min_pad at cropping: it is a floor for a head smaller than the crop, never a ceiling.

THE ORDER IS resample -> align -> crop -> NORMALISE -> pad. The normalisation lands after the crop on
purpose, so p0.5-p99.5 is read off the window the network sees rather than off the whole head -- and
the generator does the same, cropping before it normalises. That agreement between training and
deployment is why the order is not free. The divisor is predict.py's and not the absolute min-max
training used: it is what SynthSeg does at deployment, and the deliverable is a ratio, so with a floor
at 0 it is invariant to the ceiling by algebra even though the three absolute means are not.

NOTHING HERE MATCHES scratchpad/score_real_tissue_means.py BY CONSTRUCTION. That script centres a hard
160^3 window on the centroid of seg > 0, normalises with its own min-max, and never resamples (it
requires SynthSeg's --resample output instead). Numbers from the two are not interchangeable.

Usage:
  # deployment: an image goes in, a csv comes out, no segmentation anywhere
  python scripts/commands/predict_tm.py <images> preds.csv models/tm_cerebral_instance/tm_149.h5

  # with a ground truth, to score it
  python scripts/commands/predict_tm.py <images> preds.csv <model.h5> --gt <segs_dir>

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
from QC import training_tm as tm

# third-party imports
from ext.lab2im import utils
from ext.lab2im import edit_volumes


def predict_tm(path_images,
               path_out,
               path_model,
               tissues='CSF,GM,WM',
               gt_folder=None,
               path_resampled=None,
               cropping=160,
               target_res=1.,
               n_levels=5,
               nb_conv_per_level=3,
               conv_size=5,
               unet_feat_count=24,
               feat_multiplier=2,
               activation='relu',
               norm='instance',
               min_vox=8,
               recompute=True,
               verbose=True):
    """
    Predict the per-tissue mean intensity, and the GM-WM contrast built from it, on real images.

    :param path_images: path of an image, a folder of images, or a text file listing them. Same three
    forms predict.py accepts, and read by the same helpers.
    :param path_out: path of the output csv. One row per image.
    :param path_model: path of the regressor checkpoint (a tm_*.h5).

    :param tissues: (optional) comma separated tissues to read, among CSF, GM, WM. Must be the ones the
    checkpoint was trained with, since it fixes the width of the output. Default is all three.
    :param gt_folder: (optional) folder of segmentations, one per image, paired in sorted order. Turns on
    the ground truth columns: the same per-tissue means read off the segmentation, on the same crop of
    the same normalised volume, plus the absolute error on the deliverable. Any label map with the
    SynthSeg / FreeSurfer values works; it is resliced onto the image's grid with nearest if it is not
    already on it.
    :param path_resampled: (optional) folder/path where the resampled images are written, exactly as
    predict.py writes them. Costs nothing and is the only way to see what the network was actually fed.

    :param cropping: (optional) the window the network sees, cropped around the centre of the volume as
    predict.py crops it and padded up to the same size when the head is smaller. Default is 160, the
    shape training cropped to. It is the only window knob: min_pad follows it, and predict.py would cap
    it there anyway. Rounded up to a multiple of 2 ** n_levels, as predict.py rounds it. None leaves the
    volume uncropped and pads to at least 128, SynthSeg's own default, which lets the network read the
    whole conformed head instead.
    :param target_res: (optional) resolution the image is resampled to before anything else. Default is
    1., the resolution the training label maps are on. None turns the resampling off.

    :param n_levels: (optional) number of levels of the encoder. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutions per level. Default is 3.
    :param conv_size: (optional) size of the convolution kernels. Default is 5.
    :param unet_feat_count: (optional) number of features at the first level. Default is 24.
    :param feat_multiplier: (optional) feature multiplier between levels. Default is 2.
    :param activation: (optional) activation function. Default is 'relu'.
    :param norm: (optional) the normalisation the checkpoint was TRAINED with, among 'instance', 'batch'
    and 'none'. It is an architecture argument and not a detail: a 'none' checkpoint holds no
    tm_enc_in_down_* layers at all. Default is 'instance'.
    :param min_vox: (optional) a tissue with fewer voxels than this in the crop is left blank in the
    ground truth columns rather than averaged over nothing. Default is 8, training's own gate.
    :param recompute: (optional) whether to overwrite an existing output csv. Default is True.
    :param verbose: (optional) print one line per image. Default is True.
    """

    # prepare input/output filepaths
    path_images, path_out, path_gts, path_resampled = \
        prepare_output_files(path_images, path_out, gt_folder, path_resampled)
    if (not recompute) & os.path.isfile(path_out):
        print('%s already exists and recompute is off, nothing to do' % path_out)
        return

    # prepare the tissue list, and check it against the groups the target is defined on
    tissues = [t.strip() for t in tissues.split(',')] if isinstance(tissues, str) else list(tissues)
    for t in tissues:
        assert t in tm.tissue_groups, 'unknown tissue %r, expected among %s' % (t, list(tm.tissue_groups))

    # prepare the csv header, and write it now so a run that dies halfway still leaves a readable file
    header = list(tissues) + ['contrast']
    if gt_folder is not None:
        header += ['true_%s' % t for t in tissues] + ['true_contrast', 'abs_err']
    write_csv(path_out, None, True, np.arange(len(header)), np.array(header), skip_first=False)

    # the input shape is left free on the three spatial axes, as predict.py leaves it: the head is a
    # spatial mean, so the graph is shape-agnostic and one build serves every image.
    net = build_tm_model(path_model=path_model,
                         input_shape=[None] * 3 + [1],
                         n_tissues=len(tissues),
                         n_levels=n_levels,
                         nb_conv_per_level=nb_conv_per_level,
                         conv_size=conv_size,
                         unet_feat_count=unet_feat_count,
                         feat_multiplier=feat_multiplier,
                         activation=activation,
                         norm=norm)

    # one window knob: the padding follows the crop, so the network always sees the window --cropping
    # asks for whatever the head measured. A larger min_pad would be clamped, not honoured.
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

        # preprocessing
        image, gt, aff, h, im_res, shape, pad_idx, crop_idx = preprocess(path_image=path_images[i],
                                                                         n_levels=n_levels,
                                                                         target_res=target_res,
                                                                         path_gt=path_gts[i],
                                                                         crop=cropping,
                                                                         min_pad=min_pad,
                                                                         path_resample=path_resampled[i])

        # prediction
        mu_pred = np.asarray(net.predict(image))[0]

        # the deliverable, and the same quantities read off the segmentation when there is one. both are
        # computed on the SAME crop of the SAME normalised volume: a ground truth taken over the whole
        # head while the network sees a 160^3 window is a different quantity, not a stricter one.
        row = [os.path.basename(path_images[i]).replace('.nii.gz', '').replace('.nii', '').replace('.mgz', '')]
        row += ['%.6f' % v for v in mu_pred] + ['%.6f' % contrast(mu_pred, tissues)]
        if gt is not None:
            mu_true, counts = tissue_means(image[0, ..., 0], gt, tissues, min_vox)
            c_true = contrast(mu_true, tissues)
            row += ['' if np.isnan(v) else '%.6f' % v for v in mu_true]
            row += ['' if np.isnan(c_true) else '%.6f' % c_true]
            row += ['' if np.isnan(c_true) else '%.6f' % abs(contrast(mu_pred, tissues) - c_true)]
            if np.any(counts < min_vox):
                print('  [warn] %s: %s under %d voxels in the crop, left blank'
                      % (row[0], [t for t, c in zip(tissues, counts) if c < min_vox], min_vox))

        # write results to disk
        write_csv(path_out, row, True, np.arange(len(header)), np.array(header), skip_first=False)

    print('\nwrote %s' % path_out)


def prepare_output_files(path_images, out_csv, gt_folder, out_resampled):
    """predict.py's, kept to its three input forms (a text file, a folder, one image) and cut down to
    the two outputs this script has: one csv for everything, and the resampled volumes."""

    # check inputs
    assert path_images is not None, 'please specify an input file/folder (--i)'
    assert out_csv is not None, 'please specify an output csv file (--o)'

    # convert path to absolute paths
    path_images = os.path.abspath(path_images)
    basename = os.path.basename(path_images)
    out_csv = os.path.abspath(out_csv)
    out_resampled = os.path.abspath(out_resampled) if (out_resampled is not None) else out_resampled
    gt_folder = os.path.abspath(gt_folder) if (gt_folder is not None) else gt_folder

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

    # ground truths, paired with the images in sorted order, which is the pairing evaluate.evaluation
    # uses across this codebase. it is also the one that fails silently, so the first pair is printed and
    # the count is asserted: a folder with one extra file shifts every pairing by one and every number
    # below it stays plausible.
    if gt_folder is not None:
        # a .txt is a LIST, not a segmentation. Without this branch it would be read as one file, paired
        # with every image, and killed by the count assert below -- loudly, but for the wrong reason. The
        # list form is the one that matters on a real tree: BIDS and FreeSurfer do not lay images and
        # anchors out in two flat folders that happen to sort the same way.
        if gt_folder[-4:] == '.txt':
            with open(gt_folder, 'r') as f:
                path_gts = [line.replace('\n', '') for line in f.readlines() if line != '\n']
        elif os.path.isfile(gt_folder):
            path_gts = [gt_folder]
        else:
            path_gts = utils.list_images_in_folder(gt_folder)
        assert len(path_gts) == len(path_images), \
            '%d images but %d ground truths: they are paired in sorted order, so the two folders must ' \
            'hold one file each, in the same order.\n  images: %s\n  gt:     %s' \
            % (len(path_images), len(path_gts), os.path.dirname(path_images[0]), gt_folder)
        print('pairing (sorted order), first pair:\n  image %s\n  gt    %s'
              % (path_images[0], path_gts[0]))
    else:
        path_gts = [None] * len(path_images)

    return path_images, out_csv, path_gts, path_resampled


def preprocess(path_image, n_levels, target_res, path_gt=None, crop=None, min_pad=None,
               path_resample=None):
    """predict.py's, with predict_group's second volume carried through the identical window. The crop
    is chosen from the volume, so the segmentation is never an input to the preprocessing -- only
    something that has to land on the same voxels afterwards."""

    # read image and corresponding info
    im, _, aff, n_dims, n_channels, h, im_res = utils.get_volume_info(path_image, True)
    if n_dims == 4 and n_channels == 1:
        n_dims = 3
        im = im[..., 0]
    assert n_dims == 3, 'input should have 3 dimensions, had %s' % n_dims
    if n_channels > 1:
        print('WARNING: detected more than 1 channel, only keeping the first channel.')
        im = im[..., 0]

    # read the ground truth on ITS OWN grid. loading it as if it shared the image's grid is the mistake
    # this whole script exists to make impossible: a segmentation computed at another resolution then
    # pairs voxel for voxel with the image and every per-tissue mean is taken over the wrong voxels,
    # with nothing in the output looking wrong.
    if path_gt is not None:
        gt, _, aff_gt, _, _, _, _ = utils.get_volume_info(path_gt, True)
    else:
        gt, aff_gt = None, None

    # resample image if necessary
    if target_res is not None:
        target_res = np.squeeze(utils.reformat_to_n_channels_array(target_res, n_dims))
        if np.any((im_res > target_res + 0.05) | (im_res < target_res - 0.05)):
            im_res = target_res
            im, aff = edit_volumes.resample_volume(im, aff, im_res)
            if path_resample is not None:
                utils.save_volume(im, aff, h, path_resample)

    # put the ground truth on the image's grid. nearest and never linear, because a label map
    # interpolated linearly stops being a label map. skipped outright when the two already share a grid,
    # which is the usual case: SynthSeg's own --resample output next to its segmentation, or a
    # FreeSurfer aseg next to the conformed orig it was computed from.
    if gt is not None:
        if (list(gt.shape[:n_dims]) != list(im.shape[:n_dims])) or (np.abs(aff_gt - aff).max() > 1e-4):
            gt = edit_volumes.resample_volume_like(im, aff, gt, aff_gt, interpolation='nearest')
        gt = np.round(gt).astype('int32')

    # align image. the ground truth is aligned with the IMAGE's affine, as predict_group does with its
    # mask: by this point the two are on one grid, so one affine describes both.
    im = edit_volumes.align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)
    if gt is not None:
        gt = edit_volumes.align_volume_to_ref(gt, aff, aff_ref=np.eye(4), n_dims=n_dims, return_copy=False)
    shape = list(im.shape[:n_dims])

    # crop image if necessary, with the second volume carried through the same indices
    if crop is not None:
        crop = utils.reformat_to_list(crop, length=n_dims, dtype='int')
        crop_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in crop]
        # centred on the volume, which is all a deployment has. The ground truth does not choose the
        # window, it only follows it, through the image's own indices.
        im, crop_idx = edit_volumes.crop_volume(im, cropping_shape=crop_shape, return_crop_idx=True)
        if gt is not None:
            gt = edit_volumes.crop_volume_with_idx(gt, crop_idx, n_dims=n_dims)
    else:
        crop_idx = None

    # normalise. p0.5-p99.5 and after the crop, so the divisor is read off the window the network sees
    # rather than off the whole head. See the header for why this is not the min-max training used.
    im = edit_volumes.rescale_volume(im, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5)

    # pad image
    input_shape = im.shape[:n_dims]
    pad_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in input_shape]
    if min_pad is not None:
        min_pad = utils.reformat_to_list(min_pad, length=n_dims, dtype='int')
        min_pad = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in min_pad]
        pad_shape = np.maximum(pad_shape, min_pad)
    im, pad_idx = edit_volumes.pad_volume(im, padding_shape=pad_shape, return_pad_idx=True)
    if gt is not None:
        gt = edit_volumes.pad_volume(gt, padding_shape=pad_shape)

    # add batch and channel axes
    im = utils.add_axis(im, axis=[0, -1])

    return im, gt, aff, h, im_res, shape, pad_idx, crop_idx


def build_tm_model(path_model, input_shape, n_tissues, n_levels, nb_conv_per_level, conv_size,
                   unet_feat_count, feat_multiplier, activation, norm):
    """predict.py's build_model, with the regressor's own graph imported rather than rebuilt."""

    assert os.path.isfile(path_model), "The provided model path does not exist."

    import keras.layers as KL
    import keras.models as KM

    # the normalisation is an ARCHITECTURE argument and not a detail of training: with norm='none' the
    # graph holds no tm_enc_in_down_* layers at all, so loading a 'none' checkpoint into an instance-norm
    # graph fails outright in load_weights_checked. That is the good case -- it fails loudly rather than
    # loading whatever happens to line up by name and running a different network in silence.
    instance_norm = (norm == 'instance')
    batch_norm = -1 if norm == 'batch' else None
    print('architecture: norm=%s' % norm)

    img_in = KL.Input(shape=input_shape, name='val_image_input')
    stand_in = KM.Model(img_in, img_in)
    y = tm.build_regression_model(stand_in, input_shape, n_tissues, n_levels, nb_conv_per_level, conv_size,
                                  unet_feat_count, feat_multiplier, activation, batch_norm, True, instance_norm)
    net = KM.Model(stand_in.inputs, [y])
    tm.load_weights_checked(net, path_model)
    return net


def tissue_means(image, seg, tissues, min_vox=8):
    """The training target, read on a real segmentation.

    The training target is a masked pool, sum(p * i) / sum(p); with a hard one-hot
    p is 0 or 1, so that is exactly image[mask].mean() and this is the same quantity, not an
    approximation of it. The label groups come from tm.tissue_groups, the same dict build_tissue_lut
    reads, so the mask cannot drift from the one training used.

    A tissue with fewer than min_vox voxels in the crop is left as nan rather than averaged over
    nothing: that is training's own gate (an absent tissue would give a target of exactly 0, a value the
    real target never takes). With three coarse groups over a whole brain it does not fire -- the
    smallest count measured over 1500 cells was 87210 voxels -- so a nan here means the crop missed the
    brain, not that the gate is doing its job.
    """
    mu, cnt = np.full(len(tissues), np.nan), np.zeros(len(tissues), 'int64')
    for j, t in enumerate(tissues):
        mask = np.isin(seg, tm.tissue_groups[t])
        cnt[j] = int(mask.sum())
        if cnt[j] >= min_vox:
            mu[j] = float(image[mask].mean())
    return mu, cnt


def contrast(mu, tissues):
    """The deliverable: |GM - WM| / (GM + WM), the normalised contrast between the two tissues.

    A ratio of the two means, so with the normalisation floor at 0 it is invariant to the divisor by
    algebra, which the three absolute means are not.
    """
    if ('GM' not in tissues) or ('WM' not in tissues):
        return np.nan
    gm, wm = mu[tissues.index('GM')], mu[tissues.index('WM')]
    if np.isnan(gm) or np.isnan(wm):
        return np.nan
    return float(abs((gm - wm) / (gm + wm + 1e-9)))
