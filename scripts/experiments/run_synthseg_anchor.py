"""Run SynthSeg over a folder of real scans to build the ground-truth anchor for the tissue-means test.

This is the other half of scratchpad/score_real_tissue_means.py: that script compares the regressor
against a per-tissue mean taken over a segmentation, and this one produces the segmentation.

Why not the stock CLI (scripts/commands/SynthSeg_predict.py):

1. Resumability. The CLI calls predict() with the default recompute=True and does not expose the flag,
   so a run interrupted at volume 25 redoes all 25. predict() already builds a per-file recompute list
   from whether the outputs exist (predict_synthseg.py:419), so passing recompute=False makes the job
   resumable. Relaunch it and it continues.

2. The input layout. --i on a folder uses utils.list_images_in_folder, which is not recursive, and a
   BIDS tree is nested (raw/nifd/bids/sub-X/ses-Y/anat/sub-X_ses-Y_T1w.nii.gz). predict() also accepts
   a .txt of paths, which is what this script writes, so nothing is copied and the originals stay put.

3. The outputs the anchor needs are easy to forget and impossible to add later without re-segmenting:
   the resampled image when there was resampling, --vol and --qc always.

Output layout (the suffixes are what the scorer's find_pairs expects, do not rename):
  <out>/segs/<stem>_synthseg.nii.gz
  <out>/resampled/<stem>_resampled.nii.gz   only for volumes that were not already at 1 mm
  <out>/vol/<stem>_vol.csv                  region volumes in mm3
  <out>/qc/<stem>_qc.csv                    SynthSeg's own predicted Dice per region group
  <out>/{images,segs,resampled,volumes,qc}<run_tag>.txt   the path lists handed to predict()

With a .txt of inputs, --vol and --qc must themselves be .txt lists of one output path per scan
(predict_synthseg.py:319-321 asserts it, and unique_file is False), hence one small csv per volume
rather than one table. Concatenate them once the run is over.

Usage:
  python scripts/experiments/run_synthseg_anchor.py --images <dir> --out <dir>
  python scripts/experiments/run_synthseg_anchor.py --images <dir> --out <dir> --count 1   # smoke test
  python scripts/experiments/run_synthseg_anchor.py --images <dir> --out <dir> \
      --start 11 --count 11                                                                # one array task
"""
import os
import sys
import glob
import time
from collections import Counter
from argparse import ArgumentParser

import nibabel as nib

SYNTHQC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
if SYNTHQC not in sys.path:
    sys.path.insert(0, SYNTHQC)

MODEL_DIR = os.path.join(SYNTHQC, 'models')
LABELS_DIR = os.path.join(SYNTHQC, 'data', 'labels_classes_priors')
# suffixes this script (or a previous run) produces: never feed them back in as inputs
DERIVED = ('_synthseg', '_resampled', '_posteriors')
# predict() resamples a volume only when a voxel size falls outside this band (predict_synthseg.py:449)
RES_TOL = 0.05


def strip_ext(p):
    b = os.path.basename(p)
    for e in ('.nii.gz', '.nii', '.mgz'):
        if b.endswith(e):
            return b[:-len(e)]
    return os.path.splitext(b)[0]


def find_images(root, pattern):
    hits = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
    hits = [h for h in hits if h.endswith(('.nii', '.nii.gz', '.mgz'))]
    return [h for h in hits if not strip_ext(h).endswith(DERIVED)]


def write_list(path, items):
    with open(path, 'w') as f:
        f.write('\n'.join(items) + '\n')
    return path


def main():
    p = ArgumentParser()
    p.add_argument('--images', required=True, help='root to search recursively')
    p.add_argument('--pattern', default=os.path.join('**', '*.nii.gz'),
                   help="glob under --images. Selects the modality too: '**/anat/*_T1w.nii.gz' for one, "
                        "'**/anat/*.nii.gz' for every anatomical contrast in a BIDS tree.")
    p.add_argument('--out', required=True, help='one folder per dataset, see the header for its layout')
    p.add_argument('--threads', type=int, default=4, help='match it to the cpus-per-task you asked Slurm for')
    p.add_argument('--crop', type=int, default=192,
                   help='size of the analysed patch, rounded up to a multiple of 32. predict() defaults '
                        'to cropping=None, which runs the whole volume through a 24-channel first level '
                        'and needs upwards of 13 GB. 192 is the size the stock CLI advertises and holds '
                        'an adult brain; the segmentation is written back into the full volume either way '
                        '(postprocess undoes crop_idx), so the grid the scorer pairs on is unchanged. The '
                        'patch is centred on the volume, not on the brain, so with a smaller value check '
                        'that the segmentation does not reach the edge of the patch.')
    p.add_argument('--fast', action='store_true',
                   help='skip topology postprocessing. Faster, and a slightly worse anchor, so off by '
                        'default: here the segmentation is the ground truth. It also drops the '
                        'left-right flip averaging, which needs n_neutral_labels '
                        '(predict_synthseg.py:88).')
    p.add_argument('--robust', action='store_true',
                   help='SynthSeg-robust, for clinical-grade scans (implies --fast)')
    p.add_argument('--start', type=int, default=0, help='first volume of the slice, for a job array')
    p.add_argument('--count', type=int, default=0, help='0 = to the end')
    p.add_argument('--recompute', action='store_true', help='redo volumes that already have outputs')
    p.add_argument('--run_tag', default='',
                   help='suffix for the four .txt path lists, so concurrent runs over one --out do not '
                        'overwrite each other. The lists are the input predict() reads, so two array '
                        'tasks sharing them process whatever slice happened to be written last. The '
                        'segmentations themselves never collide: the slices are disjoint. Pass the array '
                        'task id.')
    p.add_argument('--gpu', action='store_true',
                   help='segment on the GPU. Off by default because the cpu allocation is a separate '
                        'counter from the gpu one, but the arithmetic only works while the volumes are '
                        'small: a 256 crop costs ~615 s and 8 cores per volume, so a 1159-volume dataset '
                        'burns 1584 core-hours, more than a yearly cpu grant. Note the network is only '
                        'part of the cost, since the flip averaging, the topology postprocessing and the '
                        '--vol/--qc tables stay in numpy on the cpu.')
    p.add_argument('--no_vol_qc', action='store_true',
                   help='skip --vol and --qc. Both ride along in the same pass and cannot be recovered '
                        'later without re-segmenting everything, so they are on by default. The qc score '
                        'is an independent per-scan quality measure, and the one to filter on before '
                        'trusting an anchor.')
    a = p.parse_args()

    images = find_images(os.path.abspath(a.images), a.pattern)
    assert images, 'no images under %s matching %s' % (a.images, a.pattern)
    if a.count:
        images = images[a.start:a.start + a.count]
    elif a.start:
        images = images[a.start:]

    seg_dir = os.path.join(a.out, 'segs')
    res_dir = os.path.join(a.out, 'resampled')
    vol_dir = os.path.join(a.out, 'vol')
    qc_dir = os.path.join(a.out, 'qc')
    # exist_ok, because the tasks of an array start together and all of them see the folders missing:
    # checking and then creating is a race that the first task wins and every other one dies in, with
    # a FileExistsError before a single volume is segmented
    for d in [a.out, seg_dir, res_dir] + ([] if a.no_vol_qc else [vol_dir, qc_dir]):
        os.makedirs(d, exist_ok=True)

    # the outputs are flat and named after the input basename, so two inputs sharing a basename would
    # overwrite each other in silence. A BIDS tree makes the basename unique inside a dataset (subject,
    # session and modality are all in it) and one --out per dataset keeps datasets apart; a FreeSurfer
    # tree does not, since every image is called orig.mgz. Check it rather than assume it.
    dup = sorted(s for s, n in Counter(strip_ext(i) for i in images).items() if n > 1)
    assert not dup, ('%d input basenames repeat and the outputs are named after them, so they would '
                     'overwrite each other: %s. Use one --out per dataset, or rename.' % (len(dup), dup[:5]))

    segs = [os.path.join(seg_dir, strip_ext(i) + '_synthseg.nii.gz') for i in images]
    resa = [os.path.join(res_dir, strip_ext(i) + '_resampled.nii.gz') for i in images]
    vols = [os.path.join(vol_dir, strip_ext(i) + '_vol.csv') for i in images]
    qcs = [os.path.join(qc_dir, strip_ext(i) + '_qc.csv') for i in images]
    todo = sum(not os.path.isfile(s) for s in segs)
    print('%d volumes under %s\n%d still to segment (the rest already have outputs)\nout: %s\n'
          % (len(images), os.path.abspath(a.images), todo, os.path.abspath(a.out)))
    if todo == 0 and not a.recompute:
        print('nothing to do. pass --recompute to redo them.')
        return

    # Ask for the resampled image only if something will actually be resampled. predict() writes that
    # file only when it resamples, so on a dataset already at 1 mm it never appears, and asking for it
    # anyway is not harmless: recompute_list ors the "output missing" flag of every requested output
    # (predict_synthseg.py:419), so every volume looks unfinished for ever and a resumed run redoes the
    # ones it had already done.
    zooms = [nib.load(i).header.get_zooms()[:3] for i in images]
    needs = [any(abs(z - 1.) > RES_TOL for z in zz) for zz in zooms]
    if not any(needs):
        print('none of the %d volumes needs resampling (all within 1 +- %.2f mm): the segmentation lands\n'
              'on the original grid, so pair the scorer with --images, not --resampled.\n'
              % (len(images), RES_TOL))
    elif not all(needs):
        print('mixed: %d of %d need resampling, %d do not. The ones that do not never get a resampled\n'
              'file, so a resumed run redoes them. Split the run by voxel size if that matters.\n'
              % (sum(needs), len(images), len(needs) - sum(needs)))
    else:
        print('all %d volumes will be resampled to 1 mm: pair the scorer with --resampled.\n' % len(images))

    def list_path(name):
        return os.path.join(a.out, '%s%s.txt' % (name, a.run_tag))

    li = write_list(list_path('images'), images)
    ls = write_list(list_path('segs'), segs)
    lr = write_list(list_path('resampled'), resa) if any(needs) else None
    lv = None if a.no_vol_qc else write_list(list_path('volumes'), vols)
    lq = None if a.no_vol_qc else write_list(list_path('qc'), qcs)

    # a resampled list left by an earlier run over this same folder would name files that will never be
    # written, which is exactly the trap the block above avoids
    stale = list_path('resampled')
    if lr is None and os.path.isfile(stale):
        os.remove(stale)
        print('removed a stale %s from a previous run of this folder\n' % os.path.basename(stale))

    if not a.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # level 1 hides tensorflow's INFO lines, which is where the ~500-line dump of allocator bins and
    # chunks that follows every out-of-memory lives, and keeps the W that names the allocation that
    # failed. Level 2 would hide that one too. It has to be set before tensorflow is imported.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(a.threads)
    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    from SynthSeg.predict_synthseg import predict

    robust, fast = a.robust, (a.fast or a.robust)
    model = 'synthseg_robust_2.0.h5' if robust else 'synthseg_2.0.h5'
    # print what tensorflow actually found, not what was asked for: --gpu on a node without one falls
    # back to the cpu in silence, and the only symptom is a run that takes ten times longer than planned
    devices = [d.name.split(':', 2)[-1] for d in tf.config.list_physical_devices('GPU')] or ['CPU']
    print('SynthSeg 2.0%s%s, %d threads, %s\n' % (' robust' if robust else '', ' (fast)' if fast else '',
                                                  a.threads, ', '.join(devices)))

    t0 = time.time()
    predict(path_images=li,
            path_segmentations=ls,
            path_model_segmentation=os.path.join(MODEL_DIR, model),
            labels_segmentation=os.path.join(LABELS_DIR, 'synthseg_segmentation_labels_2.0.npy'),
            robust=robust,
            fast=fast,
            v1=False,
            # 19 neutral labels is the value the stock CLI passes for the 2.0 label set
            # (SynthSeg_predict.py:100); it splits the 55 labels into 19 neutral and 18 left-right pairs,
            # which is what the flip averaging swaps.
            n_neutral_labels=19,
            labels_denoiser=os.path.join(LABELS_DIR, 'synthseg_denoiser_labels_2.0.npy'),
            path_posteriors=None,
            path_resampled=lr,
            path_volumes=lv,
            do_parcellation=False,
            path_model_parcellation=os.path.join(MODEL_DIR, 'synthseg_parc_2.0.h5'),
            labels_parcellation=os.path.join(LABELS_DIR, 'synthseg_parcellation_labels.npy'),
            path_qc_scores=lq,
            path_model_qc=os.path.join(MODEL_DIR, 'synthseg_qc_2.0.h5'),
            labels_qc=os.path.join(LABELS_DIR, 'synthseg_qc_labels_2.0.npy'),
            cropping=a.crop,
            names_segmentation=os.path.join(LABELS_DIR, 'synthseg_segmentation_names_2.0.npy'),
            names_qc=os.path.join(LABELS_DIR, 'synthseg_qc_names_2.0.npy'),
            topology_classes=os.path.join(LABELS_DIR, 'synthseg_topological_classes_2.0.npy'),
            recompute=a.recompute)

    done = sum(os.path.isfile(s) for s in segs)
    dt = time.time() - t0
    print('\n%d/%d segmentations on disk   (%.0f min total, %.0f s/volume over the %d computed)'
          % (done, len(segs), dt / 60., dt / max(todo, 1), todo))


if __name__ == '__main__':
    main()
