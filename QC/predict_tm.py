"""

Deploy the tissue-means regressor on a real scan: image in, one row of numbers out.

This is predict.py with the segmentation head swapped for the regressor. Read, resample to 1 mm, align
to RAS, crop, normalise, pad: all of it calls edit_volumes in predict.py's order and with predict.py's
constants, so a volume scored here went through the pipeline a volume segmented by SynthSeg goes
through. Two things predict.py does not do are taken from elsewhere in the repository rather than
written again: carrying a second volume through the identical crop and pad (predict_group), and the
regressor's own graph and checkpoint guard (QC/training_tm.py).

The segmentation is only read for the ground-truth columns: the window comes from the volume alone, so
nothing here needs a segmentation to run. The crop is not centred on the region a segmentation finds, as
predict_qc does, since deployment has no brain centre to centre it on.

The head ends in a mean over x, y and z, so the network accepts any shape, e.g. a 256^3 conformed head,
and answers differently for each, because the window it averages over holds more or less air. Training
cropped to 160^3, so 160 is the default. --cropping is the only window knob: min_pad follows it, since
predict.py caps min_pad at cropping.

The order is resample -> align -> crop -> normalise -> pad. The normalisation comes after the crop, so
p0.5-p99.5 is taken over the window the network sees, as the generator crops before it normalises. The
divisor is predict.py's and not the absolute min-max used in training: the contrast is a ratio, so with a
floor at 0 it does not depend on the ceiling, even though the absolute means do.

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
from QC.predict_rs import default_n_jobs, preprocess_in_order

# third-party imports
from ext.lab2im import utils
from ext.lab2im import edit_volumes


def predict_tm(path_images,
               path_out,
               path_model,
               tissues=None,
               gt_folder=None,
               path_resampled=None,
               cropping=160,
               target_res=1.,
               pad_mode='constant',
               n_levels=5,
               nb_conv_per_level=3,
               conv_size=5,
               unet_feat_count=24,
               feat_multiplier=2,
               activation='relu',
               norm='instance',
               use_cbam=False,
               cbam_ratio=4,
               cbam_kernels=7,
               min_vox=8,
               recompute=True,
               verbose=True,
               prepared=None,
               n_jobs=1):
    """
    Predict the per-tissue mean intensity, and the GM-WM contrast built from it, on real images.

    :param path_images: path of an image, a folder of images, or a text file listing them. Same three
    forms predict.py accepts, and read by the same helpers.
    :param path_out: path of the output csv. One row per image.
    :param path_model: path of the regressor checkpoint (a tm_*.h5).

    :param tissues: (optional) comma separated groups to read, among the keys of QC.training_tm.tissue_groups
    and in the order the checkpoint was trained with, since it fixes the width of the output. Default is
    None: QC.training_tm.all_tissues. A checkpoint from before 2026-10-02 needs legacy_tissues here.
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
    :param pad_mode: (optional) what fills an axis shorter than the window: 'constant' pads with zeros as
    predict.py does, 'edge' with the outermost plane. Training never pads. The ground truth is padded with 0
    either way. Default is 'constant'. The csv gives the voxels padded on each axis (pad_R/A/S).

    :param n_levels: (optional) number of levels of the encoder. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutions per level. Default is 3.
    :param conv_size: (optional) size of the convolution kernels. Default is 5.
    :param unet_feat_count: (optional) number of features at the first level. Default is 24.
    :param feat_multiplier: (optional) feature multiplier between levels. Default is 2.
    :param activation: (optional) activation function. Default is 'relu'.
    :param norm: (optional) the normalisation the checkpoint was trained with, among 'instance', 'batch'
    and 'none'. It is an architecture argument and not a detail: a 'none' checkpoint holds no
    tm_enc_in_down_* layers at all. Default is 'instance'.
    :param use_cbam: (optional) whether the checkpoint was trained with CBAM in the encoder. Architecture
    argument, like norm. Default is False.
    :param cbam_ratio: (optional) the checkpoint's channel attention reduction. Default is 4.
    :param cbam_kernels: (optional) the checkpoint's spatial attention kernels. Default is 7 on every level.
    :param min_vox: (optional) a tissue with fewer voxels than this in the crop is left blank in the
    ground truth columns rather than averaged over nothing. Default is 8, training's own gate.
    :param recompute: (optional) whether to overwrite an existing output csv. Default is True.
    :param verbose: (optional) print one line per image. Default is True.
    :param prepared: (optional) the output of prepare_all for these same images, ground truths and preprocessing
    arguments, one entry per image in the order prepare_output_files pairs them. validate_tm passes it so that
    the resampling, which dominates the cost, is paid once per validation set and not once per checkpoint.
    :param n_jobs: (optional) worker processes for the preprocessing, as in predict_rs. 1 runs it in this process;
    None takes the CPUs of the job but one. Ignored when prepared is given. Default is 1.
    """

    # prepare input/output filepaths
    path_images, path_out, path_gts, path_resampled = \
        prepare_output_files(path_images, path_out, gt_folder, path_resampled)
    if (not recompute) & os.path.isfile(path_out):
        print('%s already exists and recompute is off, nothing to do' % path_out)
        return

    # prepare the tissue list, and check it against the groups the target is defined on
    tissues = resolve_tissues(tissues)

    # prepare the csv header, and write it now so a run that dies halfway still leaves a readable file
    header = list(tissues) + ['contrast']
    if gt_folder is not None:
        header += ['true_%s' % t for t in tissues] + ['true_contrast', 'abs_err']
    header += ['pad_R', 'pad_A', 'pad_S']
    write_csv(path_out, None, True, np.arange(len(header)), np.array(header), skip_first=False)

    cropping, min_pad = window(cropping)
    assert prepared is None or len(prepared) == len(path_images), \
        '%d prepared volumes for %d images' % (len(prepared), len(path_images))

    # the workers are started before the model is built, so no process is forked with a live graph in it
    pool = None
    if prepared is not None:
        volumes = iter(prepared)
    else:
        n_jobs = default_n_jobs() if n_jobs is None else max(int(n_jobs), 1)
        jobs = [dict(path_image=path_images[i], path_gt=path_gts[i], n_levels=n_levels, target_res=target_res,
                     cropping=cropping, min_pad=min_pad, tissues=tissues, min_vox=min_vox,
                     path_resample=path_resampled[i], pad_mode=pad_mode) for i in range(len(path_images))]
        volumes, pool = preprocess_in_order(jobs, n_jobs, fn=_prepare_job)

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
                         norm=norm,
                         use_cbam=use_cbam,
                         cbam_ratio=cbam_ratio,
                         cbam_kernels=cbam_kernels)

    print('preprocessing: cropping=%s  target_res=%s  pad_mode=%s  norm=%s%s'
          % (cropping, target_res, pad_mode, norm,
             '  (prepared once, cached)' if prepared else '  n_jobs=%d' % n_jobs))

    # perform prediction
    if len(path_images) <= 10:
        loop_info = utils.LoopInfo(len(path_images), 1, 'predicting', True)
    else:
        loop_info = utils.LoopInfo(len(path_images), 10, 'predicting', True)
    try:
        for i in range(len(path_images)):
            if verbose:
                loop_info.update(i)

            # preprocessing
            image, truth, pad = next(volumes)

            # prediction
            mu_pred = np.asarray(net.predict(image))[0]

            # prediction and truth are computed on the same crop of the same normalised volume: a truth taken
            # over the whole head while the network sees a 160^3 window would be a different quantity.
            row = [os.path.basename(path_images[i]).replace('.nii.gz', '').replace('.nii', '').replace('.mgz', '')]
            row += ['%.6f' % v for v in mu_pred] + ['%.6f' % contrast(mu_pred, tissues)]
            if truth is not None:
                mu_true, counts = truth
                c_true = contrast(mu_true, tissues)
                row += ['' if np.isnan(v) else '%.6f' % v for v in mu_true]
                row += ['' if np.isnan(c_true) else '%.6f' % c_true]
                row += ['' if np.isnan(c_true) else '%.6f' % abs(contrast(mu_pred, tissues) - c_true)]
                if np.any(counts < min_vox):
                    print('  [warn] %s: %s under %d voxels in the crop, left blank'
                          % (row[0], [t for t, c in zip(tissues, counts) if c < min_vox], min_vox))
            row += ['%d' % v for v in pad]

            # write results to disk
            write_csv(path_out, row, True, np.arange(len(header)), np.array(header), skip_first=False)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    print('\nwrote %s' % path_out)


def resolve_tissues(tissues):
    """None means all_tissues; a comma-separated string or a list is checked against the groups the target
    is defined on."""
    return list(tm.all_tissues) if tissues is None else tm.parse_tissues(tissues)


def window(cropping):
    """One window knob: the padding follows the crop, so the network always sees the window --cropping
    asks for whatever the head measured. A larger min_pad would be clamped, not honoured."""
    if cropping is not None:
        cropping = utils.reformat_to_list(cropping, length=3, dtype='int')
        return cropping, cropping
    return None, 128


def prepare(path_image, path_gt, n_levels, target_res, cropping, min_pad, tissues, min_vox, path_resample=None,
            pad_mode='constant'):
    """One image as predict_tm feeds it to the network, the truth (mu_true, counts) from its segmentation
    (None without one), and the voxels padded on each RAS axis."""
    out = preprocess(path_image=path_image, n_levels=n_levels, target_res=target_res, path_gt=path_gt,
                     crop=cropping, min_pad=min_pad, path_resample=path_resample, pad_mode=pad_mode)
    image, gt, pad_idx = out[0], out[1], out[6]
    pad = [int(s - (pad_idx[3 + j] - pad_idx[j])) for j, s in enumerate(image.shape[1:4])]
    truth = tissue_means(image[0, ..., 0], gt, tissues, min_vox) if gt is not None else None
    return image, truth, pad


def _prepare_job(job):
    return prepare(**job)


def prepare_all(path_images, path_gts, n_levels, target_res, cropping=160, tissues=None, min_vox=8, n_jobs=1,
                pad_mode='constant'):
    """prepare over a list of images, for predict_tm's prepared argument: the same call predict_tm makes
    per image, so a cached volume and its truth are bit-identical to the ones it would compute. Only the
    image and the tissue means are kept, not the segmentation."""
    cropping, min_pad = window(cropping)
    tissues = resolve_tissues(tissues)
    jobs = [dict(path_image=p, path_gt=g, n_levels=n_levels, target_res=target_res, cropping=cropping,
                 min_pad=min_pad, tissues=tissues, min_vox=min_vox, pad_mode=pad_mode)
            for p, g in zip(path_images, path_gts)]
    volumes, pool = preprocess_in_order(jobs, default_n_jobs() if n_jobs is None else max(int(n_jobs), 1),
                                        fn=_prepare_job)
    try:
        return list(volumes)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


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
        # a .txt is a list of segmentations, the form needed on BIDS and FreeSurfer trees, where images and
        # segmentations are not in two flat folders that sort the same way.
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
               path_resample=None, pad_mode='constant'):
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

    # read the ground truth on its own grid: a segmentation computed at another resolution cannot be paired
    # voxel for voxel with the image.
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

    # align image. the ground truth is aligned with the image's affine, as predict_group does with its
    # mask: by this point the two are on one grid.
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

    # normalise, p0.5-p99.5 over the cropped window (see the header)
    im = edit_volumes.rescale_volume(im, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5)

    # pad image. 'edge' fills with the outermost plane instead of zeros, with pad_volume's margins; the
    # ground truth is always padded with 0, so a copied plane never counts as tissue in the truth.
    input_shape = im.shape[:n_dims]
    pad_shape = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in input_shape]
    if min_pad is not None:
        min_pad = utils.reformat_to_list(min_pad, length=n_dims, dtype='int')
        min_pad = [utils.find_closest_number_divisible_by_m(s, 2 ** n_levels, 'higher') for s in min_pad]
        pad_shape = np.maximum(pad_shape, min_pad)
    pad = [max(int(p) - s, 0) for p, s in zip(pad_shape, input_shape)]
    if pad_mode == 'constant':
        im, pad_idx = edit_volumes.pad_volume(im, padding_shape=pad_shape, return_pad_idx=True)
    else:
        pad_idx = np.array([p // 2 for p in pad] + [p // 2 + s for p, s in zip(pad, input_shape)])
        im = np.pad(im, [(p // 2, p - p // 2) for p in pad], mode=pad_mode)
    if gt is not None:
        gt = edit_volumes.pad_volume(gt, padding_shape=pad_shape)

    # add batch and channel axes
    im = utils.add_axis(im, axis=[0, -1])

    return im, gt, aff, h, im_res, shape, pad_idx, crop_idx


def build_tm_model(path_model, input_shape, n_tissues, n_levels, nb_conv_per_level, conv_size,
                   unet_feat_count, feat_multiplier, activation, norm, use_cbam=False, cbam_ratio=4,
                   cbam_kernels=7):
    """predict.py's build_model, with the regressor's own graph imported rather than rebuilt."""

    assert os.path.isfile(path_model), "The provided model path does not exist."

    import keras.layers as KL
    import keras.models as KM

    # the normalisation changes the graph: with norm='none' there are no tm_enc_in_down_* layers, so a
    # checkpoint trained with another norm is refused by load_weights_checked.
    instance_norm = (norm == 'instance')
    batch_norm = -1 if norm == 'batch' else None
    print('architecture: norm=%s%s' % (norm, '  cbam (ratio=%s, kernels=%s)' % (cbam_ratio, cbam_kernels)
                                         if use_cbam else ''))

    img_in = KL.Input(shape=input_shape, name='val_image_input')
    stand_in = KM.Model(img_in, img_in)
    y = tm.build_regression_model(stand_in, input_shape, n_tissues, n_levels, nb_conv_per_level, conv_size,
                                  unet_feat_count, feat_multiplier, activation, batch_norm, True, instance_norm,
                                  use_cbam, cbam_ratio, cbam_kernels)
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
