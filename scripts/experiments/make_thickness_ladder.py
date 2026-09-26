"""Build a slice-thickness ladder for the resolution head: 1 mm NIFD dev volumes (a T1w and a FLAIR of the same
session) thickened along one axis to k = 2..8 mm, so that the thickness is the only thing changing between steps.

Two kernels:
  * synthseg: the blur and downsampling of the generator (SampleResolution thickness, DynamicGaussianBlur,
    MimicAcquisition), without the upsampling, which predict_rs does. Runs in voxel units of the source.
  * box: the mean of k contiguous slices.

Three geometries, i.e. how the n source slices meet the spacing k:
  * crop: the slice axis is cropped to a multiple of k, so the spacing is exactly k.
  * stretch: int(n / k) samples over the n slices, as MimicAcquisition does, so the spacing is n / int(n / k).
    synthseg kernel only.
  * edge: the first and last slices are repeated up to a multiple of k, then thickened at exactly k.

--coverage trims the thickened axis to that many mm around its centre (144 mm at 6 mm = the 24 slices of the
OASIS-3 FLAIR), so predict_rs has to pad or crop it.

The header is the source's with the slice axis scaled by the spacing, since validate_rs reads the truth from it.
Writes <out>/img/*.nii.gz and <out>/manifest.csv. Existing volumes are not rewritten unless --overwrite.

    python scripts/experiments/make_thickness_ladder.py --out $WORK/qc-data/validation/img_ladder
"""
import csv
import os
import random
import sys
from argparse import ArgumentParser
from collections import defaultdict

import nibabel as nib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import tensorflow as tf  # noqa: E402
import keras.layers as KL  # noqa: E402
from keras.models import Model  # noqa: E402
from ext.lab2im import layers  # noqa: E402
from ext.lab2im import edit_tensors as l2i_et  # noqa: E402

WORK = os.environ.get('WORK', '')
T1_ZOOMS = '1.00|1.00|1.00'
FLAIR_ZOOMS = '1.00|0.98|0.98'
DIRECTIONS = {'axial': 'SI', 'sagittal': 'RL'}   # the RAS letters of the thickened axis
MANIFEST = ['file', 'stem', 'session', 'modality', 'direction', 'kernel', 'geometry', 'k', 'array_axis',
            'thickness_drawn_vox', 'spacing_mm', 'shape', 'zooms', 'coverage_mm']


def select_sessions(inventory, n, seed):
    """NIFD dev sessions with a 1 mm iso T1w and a 0.98 mm FLAIR, one per subject, n at random."""
    by_session = defaultdict(dict)
    for r in csv.DictReader(open(inventory, newline='')):
        if r['dataset'] != 'nifd' or r['split'] != 'dev':
            continue
        if (r['modality'], r['zooms']) in (('T1w', T1_ZOOMS), ('FLAIR', FLAIR_ZOOMS)):
            session = r['stem'].rsplit('_', 1)[0]
            by_session[session][r['modality']] = r['path']
    complete = sorted(s for s, m in by_session.items() if len(m) == 2)
    by_subject = defaultdict(list)
    for s in complete:
        by_subject[s.split('_')[0]].append(s)
    rng = random.Random(seed)
    one_each = [rng.choice(v) for _, v in sorted(by_subject.items())]
    print('NIFD dev: %d complete sessions over %d subjects' % (len(complete), len(one_each)))
    if len(one_each) < n:
        sys.exit('only %d subjects qualify, asked for %d' % (len(one_each), n))
    return [(s, by_session[s]) for s in sorted(rng.sample(one_each, n))]


def slice_axis(aff, direction):
    """The array axis that runs along the RAS direction of the thickened axis."""
    codes = nib.aff2axcodes(aff)
    hits = [i for i, c in enumerate(codes) if c in DIRECTIONS[direction]]
    assert len(hits) == 1, 'axis codes %s give no single %s axis' % (codes, direction)
    return hits[0]


def synthseg_model(shape, axis, k, max_res):
    """Blur and downsampling block of labels_to_image_model (randomise_res branch), in voxel units."""
    ones = np.ones(3)
    image_in = KL.Input(shape=list(shape) + [1])
    res_in = KL.Input(shape=(3,))
    thick_in = KL.Input(shape=(3,))
    sigma = l2i_et.blurring_sigma_for_downsampling(ones, res_in, thickness=thick_in)
    blurred = layers.DynamicGaussianBlur(0.75 * max_res * ones, 1.03)([image_in, sigma])
    down_shape = list(shape)
    down_shape[axis] = shape[axis] // k
    # resample_shape = acquisition shape: the up stage of MimicAcquisition becomes the identity
    down = layers.MimicAcquisition(ones, ones, down_shape, False)([blurred, res_in])
    return Model([image_in, res_in, thick_in], down)


def draw_thickness(res):
    """SampleResolution._sample_thickness as the generator calls it (thickness_min_frac 0)."""
    sr = layers.SampleResolution([1., 1., 1.], max_res_aniso=[8.] * 3, uniform_per_axis=True)
    sr.build(None)
    return sr._sample_thickness(tf.convert_to_tensor(res, dtype='float32')).numpy()


def thicken(vol, axis, k, kernel, seed, max_res, cache, geometry='crop'):
    """Returns the thickened volume, the offset of its first sample in source voxels, the thickness
    synthseg drew (voxels; k for the box) and the spacing written to the header (source voxels)."""
    lead = 0
    if geometry == 'crop':
        n = (vol.shape[axis] // k) * k
        vol = np.take(vol, np.arange(n), axis=axis)
    elif geometry == 'edge':
        # repeat the first and last slices up to the next multiple of k
        extra = -vol.shape[axis] % k
        lead = extra // 2
        pad = [(0, 0)] * vol.ndim
        pad[axis] = (lead, extra - lead)
        vol = np.pad(vol, pad, mode='edge')
    if k == 1:
        return vol, 0., 1., 1.
    if geometry == 'stretch':
        # int(n / k) samples spread over the n slices, as in MimicAcquisition
        assert kernel == 'synthseg', 'stretch is the generator geometry: synthseg kernel only'
        spacing = vol.shape[axis] / float(vol.shape[axis] // k)
    else:
        spacing = float(k)
    if kernel == 'box':
        v = np.moveaxis(vol, axis, 0)
        n = v.shape[0]
        return (np.moveaxis(v.reshape((n // k, k) + v.shape[1:]).mean(axis=1), 0, axis), (k - 1) / 2. - lead,
                float(k), spacing)
    res = np.ones(3, dtype='float32')
    res[axis] = k
    tf.random.set_seed(seed)
    thick = draw_thickness(res)
    key = (vol.shape, axis, k)
    if key not in cache:
        cache.clear()
        cache[key] = synthseg_model(vol.shape, axis, k, max_res)
    out = cache[key].predict([vol[None, ..., None], res[None], thick[None]])[0, ..., 0]
    return out, -float(lead), float(thick[axis]), spacing


def write(vol, img, axis, spacing, offset, path):
    aff = img.affine.copy()
    aff[:3, 3] += aff[:3, axis] * offset
    aff[:3, axis] *= spacing
    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    out = nib.Nifti1Image(vol.astype(np.float32), aff, header=hdr)
    out.set_qform(aff, code=int(img.header['qform_code']) or 1)
    out.set_sform(aff, code=int(img.header['sform_code']) or 1)
    nib.save(out, path)
    return out


def build(a):
    img_dir = os.path.join(a.out, 'img')
    os.makedirs(img_dir, exist_ok=True)
    sessions = select_sessions(a.inventory, a.n, a.seed)
    old = {}
    if os.path.isfile(os.path.join(a.out, 'manifest.csv')):
        old = {r['file']: r for r in csv.DictReader(open(os.path.join(a.out, 'manifest.csv'), newline=''))}
    rows, cache = [], {}
    for i, (session, paths) in enumerate(sessions):
        for m, modality in enumerate(('T1w', 'FLAIR')):
            src = paths[modality]
            img = nib.load(src)
            vol = np.asarray(img.dataobj, dtype=np.float32)
            if vol.ndim == 4:
                vol = vol[..., 0]
            zooms = img.header.get_zooms()[:3]
            stem = '%s_%s' % (session, modality)
            # the seed of each volume depends on its position in this list
            jobs = [('none', 'none', 1, 'crop')] + [(d, kn, k, g) for g in a.geometries for d in a.directions
                                                    for kn in a.kernels for k in a.steps
                                                    if k > 1 and not (g == 'stretch' and kn == 'box')]
            for j, (direction, kernel, k, geometry) in enumerate(jobs):
                axis = slice_axis(img.affine, direction if direction != 'none' else 'axial')
                tail = ('' if geometry == 'crop' else '_' + geometry) + ('_cov%d' % a.coverage if a.coverage else '')
                name = '%s_k1.nii.gz' % stem if k == 1 else '%s_%s_%s%s_k%d.nii.gz' % (stem, direction, kernel, tail, k)
                path = os.path.join(img_dir, name)
                if os.path.isfile(path) and not a.overwrite and name in old:
                    rows.append(old[name])
                    continue
                seed = a.seed * 100000 + i * 1000 + m * 100 + j
                thick_vol, offset, thick, spacing = thicken(vol, axis, k, kernel, seed, a.max_res, cache, geometry)
                if a.coverage and k > 1:
                    n = thick_vol.shape[axis]
                    keep = min(int(a.coverage / (spacing * zooms[axis]) + 1e-6), n)
                    start = (n - keep) // 2
                    thick_vol = np.take(thick_vol, np.arange(start, start + keep), axis=axis)
                    offset += start * spacing
                out = write(thick_vol, img, axis, spacing, offset, path)
                rows.append(dict(file=name, stem=stem, session=session, modality=modality, direction=direction,
                                 kernel=kernel, geometry=geometry, k=k, array_axis=axis,
                                 thickness_drawn_vox='%.3f' % thick,
                                 spacing_mm='%.3f' % (spacing * zooms[axis]),
                                 shape='x'.join(map(str, out.shape[:3])),
                                 zooms='|'.join('%.3f' % z for z in out.header.get_zooms()[:3]),
                                 coverage_mm='%.1f' % (out.shape[axis] * spacing * zooms[axis])))
        print('  %2d/%d %s  (%d volumes so far)' % (i + 1, len(sessions), session, len(rows)), flush=True)
    with open(os.path.join(a.out, 'manifest.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, 'sources.txt'), 'w') as fh:
        for session, paths in sessions:
            fh.write('%s\n%s\n' % (paths['T1w'], paths['FLAIR']))
    print('%d volumes in %s' % (len(rows), img_dir))


def main():
    p = ArgumentParser()
    p.add_argument('--out', required=True, help='ladder root; images go to <out>/img')
    p.add_argument('--inventory', default=os.path.join(WORK, 'qc-data', 'index', 'inventory.csv'))
    p.add_argument('--n', type=int, default=20, help='sessions (each gives one T1w and one FLAIR)')
    p.add_argument('--steps', type=int, nargs='+', default=list(range(1, 9)), help='k: spacing in source voxels')
    p.add_argument('--directions', nargs='+', default=['axial', 'sagittal'], choices=list(DIRECTIONS))
    p.add_argument('--kernels', nargs='+', default=['synthseg', 'box'], choices=['synthseg', 'box'])
    p.add_argument('--geometries', nargs='+', default=['crop'], choices=['crop', 'stretch', 'edge'],
                   help='how the slice axis meets k (see the docstring). stretch skips the box kernel')
    p.add_argument('--coverage', type=float, default=None, help='trim the thickened axis to this many mm')
    p.add_argument('--max_res', type=float, default=8.,
                   help='max(MAX_RES_ISO, MAX_RES_ANISO) of the rs runs: sets the blur window, as in training')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    build(a)


if __name__ == '__main__':
    main()
