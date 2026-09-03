"""Build the small real-data validation set as a self-contained folder.

We currently choose an epoch by eye, and the epoch moves the deliverable a lot (Kirby MAE 0.0300 ->
0.0161 between epochs 100 and 149). Ten real images scored at every checkpoint give a validation curve
on real data, which no synthetic val_*.npz can provide.

Ten is far below this project's error bar for an absolute number, but the curve is PAIRED: the same ten
images at every epoch, so the comparison is within-subject and most of the between-subject variance
cancels. It ranks epochs. It does not measure performance, and no headline should come out of it.

Using an image to choose a checkpoint makes it dev, permanently. So this refuses to pick anything
outside the frozen dev lists. Nothing here touches test.

The layout it writes, and why it is a copy rather than a list of paths into the cohorts:

    <out>/img/<stem>.nii.gz      the ten images
    <out>/gt/fs/<stem>_aseg.mgz  FreeSurfer anchor, for the T1w rows that have one
    <out>/gt/ss/                 SynthSeg 2.0 anchor -- written afterwards, over <out>/img
    <out>/validation.csv         one row per image: cohort, subject, voxel size, and both source paths

Ten volumes are a few hundred MB, and a folder that holds its own images survives the cohort trees
moving, a dev list being regenerated, or an anchor folder being reorganised. The earlier version wrote
two .txt of absolute paths into raw/ and anchors/, which is four ways to silently point at the wrong
file. The csv stays the authority on provenance; the folder is what gets scored.

The SynthSeg anchor is NOT looked up here. Only these ten volumes need one, so it is cheaper to run
SynthSeg over <out> after this than to require a pre-existing segmentation -- which is also the only
way the FLAIR rows can exist at all, since SynthSeg was never run over the 946 FLAIR:

    python scripts/commands/SynthSeg_predict.py --i <out>/img --o <out>/gt/ss --crop 256 \
        --vol <out>/gt/ss/vol.csv --qc <out>/gt/ss/qc.csv

--crop 256 is not optional: the default is 192 and the crop moves the tissue-means divisor by
7.85 %. Nor are --vol and --qc, which ride along in the same pass and cannot be added later without
re-segmenting. img/ holds nothing but the images, so the anchors land beside them rather than among
them, and images are copied under whichever of .nii and .nii.gz they arrived with, so a pass that
picks up only one of the two would leave nine anchors for ten images without a word.

The recipe, and what each row buys:

    1  kirby21  T1w    1.2 x 1.0 x 1.0    a FreeSurfer anchor of its own
    2  ixi      T1w    0.94 and 0.98      two sites; a third of IXI T1 variance is between sites
    1  miriad   T1w    0.94 x 0.94 x 1.5  the only real anisotropy among the T1w; GE; Alzheimer
    2  nifd     T1w    1.0 iso and 1.2    the mixed-resampling case; FTD
    2  nifd     FLAIR  3.00 mm slice      the coarse end, twice
    1  nifd     FLAIR  0.49 in-plane      the fine end
    1  nifd     FLAIR  near-isotropic     FLAIR without an extreme geometry

Six T1w and four FLAIR. The second 3 mm row is not coverage, it is robustness: with only one, the whole
coarse end of the curve rests on a single head. The near-isotropic FLAIR is the row that earns its
place on design rather than on balance -- every other FLAIR here is also an extreme resolution, so
without it the modality and the geometry are confounded, and an odd FLAIR reading could not be
attributed to either.

The cost is real and belongs in the write-up: FLAIR exists only in NIFD in this data, so four FLAIR
rows make the set six tenths NIFD, which is FTD and atrophic -- exactly where the two anchors disagree
most, and the disagreement scales with atrophy. Tolerable here because the curve ranks epochs against
the same anchor at every epoch, so a constant anchor bias cancels. It would not be tolerable for an
absolute number, and no absolute number comes out of ten images anyway.

The two T1w that made room were the second kirby21 and the second miriad: both were exact geometry
duplicates of their own cohort's first row. The two ixi rows look like duplicates by resolution (0.937
against 0.977) and are not -- they are two sites, which is domain rather than geometry.

T2w is deliberately absent: it is held for test, along with the resolutions only it carries.

FLAIR nulls CSF, so a FLAIR CSF tissue mean is not the same quantity as a T1w one. GM and WM carry over
and the deliverable is GM/WM, so FLAIR rows are usable with the CSF column dropped, not explained away.

Run it where the data is:
    python scripts/experiments/make_real_val10.py --root <data>/qc-data

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
"""

import os
import re
import csv
import glob
import shutil
import argparse

import numpy as np
import nibabel as nib

# Geometry filters. spread() maximises the distance between the resolutions it picks, which is the right
# rule when a row just has to be varied -- but it cannot be asked for two of the SAME geometry, and with
# only three FLAIR geometries in dev its fourth pick necessarily duplicates one of them, chosen by a
# tie-break on the path. Naming the band instead says what the row is for. Bands, not exact zooms, so a
# re-run of the inventory that shifts a value by a hundredth does not silently empty a row.
COARSE = lambda z: max(z) >= 2.0                        # noqa: E731  the 3 mm slice
FINE = lambda z: min(z) <= 0.6                          # noqa: E731  the 0.49 in-plane
NEAR_ISO = lambda z: max(z) / min(z) <= 1.1             # noqa: E731  neither extreme

# (dataset, modality, how many, geometry filter or None, what the row is there for).
# The order is the order of the csv, and a subject already taken by an earlier row is not offered again.
RECIPE = [
    ('kirby21', 'T1w', 1, None, 'FreeSurfer anchor of its own, 1.2 mm in-plane'),
    ('ixi', 'T1w', 2, None, 'two sites; a third of IXI T1 variance is between sites'),
    ('miriad', 'T1w', 1, None, 'the only real anisotropy among the T1w (1.5 mm slice), GE 1.5T, Alzheimer'),
    ('nifd', 'T1w', 2, None, 'mixed resampling (1.0 iso and 1.2), FTD'),
    ('nifd', 'FLAIR', 2, COARSE, 'the coarse end, twice: the 3 mm end of the curve rests here'),
    ('nifd', 'FLAIR', 1, FINE, 'the fine end (0.49 in-plane)'),
    ('nifd', 'FLAIR', 1, NEAR_ISO, 'FLAIR without an extreme geometry: separates modality from geometry'),
]

# derived FreeSurfer streams. The longitudinal one re-runs each session from a template built from both,
# so the two stop being independent and the scan-rescan floor is destroyed; the nipype working copies are
# byte-identical duplicates, and pairing a session with its own copy gives perfect agreement. long-* is
# listed explicitly because it carries no session entity of its own, so the entity match below would
# otherwise accept it as a session-less anchor for either session.
FS_DERIVED = ('freesurfer_longitudinal', 'freesurfer_unbiased_template', '/wd', '/ReconAll/',
              '/nipype', '/long-')

ENTITY = re.compile(r'(sub-[A-Za-z0-9]+|ses-[A-Za-z0-9]+)')


def entities(text):
    """The (subject, session) BIDS entities anywhere in a stem or a path, session possibly None.

    The FreeSurfer trees do not agree on a layout -- kirby21 nests subjects/sub-KKI113/ses-01/, nifd
    joins them into sub-XXX_ses-01/ -- so the entities are pulled out of the whole string rather than
    from a fixed position, which covers both.

    They are then compared for EQUALITY. The previous version asked `stem in path`, which fails twice:
    the image stem carries a modality suffix the FreeSurfer path never has (that is why the inventory's
    FreeSurfer column reads zero), and a substring test matches sub-IXI11 inside sub-IXI116, which is a
    wrong anchor rather than a missing one.
    """
    found = ENTITY.findall(text)
    return (next((x for x in found if x.startswith('sub-')), None),
            next((x for x in found if x.startswith('ses-')), None))


def stem_of(path):
    b = os.path.basename(path)
    for e in ('.nii.gz', '.nii', '.mgz'):
        if b.endswith(e):
            return b[:-len(e)]
    return os.path.splitext(b)[0]


def read_dev_stems(root, ds):
    """The frozen dev list. No fallback on purpose: guessing the split is how test gets burned."""
    p = os.path.join(root, 'index', ds, 'splits', 'dev_stems.txt')
    if not os.path.isfile(p):
        raise SystemExit(
            'no dev list for %s at %s.\nThis script will not pick subjects without one: choosing a\n'
            'checkpoint with an image makes that image dev forever, and a wrong guess here silently\n'
            'spends test. Generate the split first (scripts/experiments/make_splits_research_datasets.py).'
            % (ds, p))
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def find_images(root, ds, modality):
    """Every image of one modality under the dataset's BIDS copy. The BIDS tree is the source of images;
    a recon-all tree only holds what recon-all ate."""
    pat = os.path.join(root, 'raw', ds, 'bids', '**', '*%s*.nii*' % modality)
    hits = sorted(p for p in glob.glob(pat, recursive=True) if '.json' not in p)
    if not hits:
        print('[WARNING] %s %s: no images under raw/%s/bids -- the row will come out empty'
              % (ds, modality, ds))
    return hits


def index_fs(root, ds):
    """{(subject, session): aseg path} for one dataset, walking the tree ONCE.

    Not a glob per image: that re-walks the whole tree every time, which on a cohort the size of NIFD
    over a network filesystem is the difference between seconds and hours. Nor a hard-coded 'subjects/' level -- that
    is kirby21's layout, and it is the reason every other cohort read fs=no.
    """
    out = {}
    d = os.path.join(root, 'anchors', 'freesurfer', ds)
    for dirpath, _, files in os.walk(d):
        q = dirpath.replace('\\', '/')
        if 'aseg.mgz' not in files or any(x in q + '/' for x in FS_DERIVED):
            continue
        key = entities(q)
        if key[0] is None:
            continue
        out.setdefault(key, []).append(os.path.join(dirpath, 'aseg.mgz'))
    for key, paths in sorted(out.items()):
        if len(paths) > 1:
            print('  [%s %s: %d asegs, taking the first] %s'
                  % (ds, key, len(paths), sorted(paths)[0]))
        out[key] = sorted(paths)[0]
    return out


def fs_anchor_for(fs_index, image, modality):
    """The FreeSurfer aseg for one image, or None.

    recon-all only ever ate the T1, so a FLAIR or a T2w having none is a property of recon-all and not
    a gap in the tree: those get their anchor from the SynthSeg pass over the finished folder. Handing
    a FLAIR the aseg of its session's T1w would look like an anchor and be a cross-modality transfer --
    a different acquisition, 3 mm slices against 1 mm -- with nothing in the csv to say so.
    """
    if modality != 'T1w':
        return None
    subj, sess = entities(stem_of(image))
    if subj is None:
        return None
    if (subj, sess) in fs_index:
        return fs_index[(subj, sess)]
    if sess is None:                        # a cohort with no sessions: one aseg for the subject
        same = [p for (s, _), p in fs_index.items() if s == subj]
        return same[0] if len(same) == 1 else None
    return None                             # the image names a session; anything else is ambiguous


def zooms(path):
    return tuple(round(float(z), 3) for z in nib.load(path).header.get_zooms()[:3])


def spread(cands, n):
    """Pick n images whose resolutions are as far apart as possible, preferring a candidate that has a
    FreeSurfer anchor when the resolution is a tie, and breaking what is left by path so the selection
    is reproducible.

    Variety is the point of this set, so resolution comes first: picking the first n alphabetically
    would quietly give n copies of the same scanner. But only about half of NIFD's sessions were run
    through recon-all, so at equal resolution an anchored candidate is strictly better -- it is the
    same row plus a second opinion on the ground truth, for free.
    """
    if len(cands) <= n:
        return cands
    cands = sorted(cands, key=lambda c: (c['zooms'], not c['fs'], c['image']))
    picked = [cands[0]]
    while len(picked) < n:
        far = max((c for c in cands if c not in picked),
                  key=lambda c: (min(np.abs(np.array(c['zooms']) - np.array(p['zooms'])).sum()
                                     for p in picked), bool(c['fs']), c['image']))
        picked.append(far)
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True, help='the qc-data root')
    ap.add_argument('--out', default=None, help='default: --root/validation')
    ap.add_argument('--dry_run', action='store_true', help='print the selection, copy nothing')
    a = ap.parse_args()

    out = a.out or os.path.join(a.root, 'validation')
    rows, taken = [], set()

    for ds, modality, n, geometry, why in RECIPE:
        dev = set(read_dev_stems(a.root, ds))
        fs_index = index_fs(a.root, ds) if modality == 'T1w' else {}
        cands = []
        for img in find_images(a.root, ds, modality):
            if not any(s in os.path.basename(img) for s in dev):
                continue                      # not in dev: not a candidate, no discussion
            # the subject is the BIDS entity, NOT the dev-list entry: those lists are per session in
            # some cohorts and per subject in others, so keying on them made two sessions of one head
            # look like two heads, and the whole point of the rule is that they are not
            subject = entities(stem_of(img))[0]
            if subject is None or (ds, subject) in taken:
                continue                      # no subject entity, or an earlier row spent this head
            z = zooms(img)
            if geometry is not None and not geometry(z):
                continue
            cands.append({'dataset': ds, 'modality': modality, 'subject': subject,
                          'stem': stem_of(img), 'image': img,
                          'fs': fs_anchor_for(fs_index, img, modality),
                          'zooms': z})
        # one row per subject: two sessions of the same head are one measurement, and a repeated
        # subject weights that head twice at every epoch of a paired curve
        seen, uniq = set(), []
        for c in sorted(cands, key=lambda c: c['image']):
            if c['subject'] not in seen:
                seen.add(c['subject'])
                uniq.append(c)
        chosen = spread(uniq, n)
        if len(chosen) < n:
            print('[WARNING] %s %s: wanted %d, found %d in dev%s'
                  % (ds, modality, n, len(chosen), ' matching this row\'s geometry' if geometry else ''))
        rows.extend(chosen)
        taken.update((c['dataset'], c['subject']) for c in chosen)
        print('%s %s x%d: %s' % (ds, modality, n, why))     # the prose belongs here, not in the csv
        for c in chosen:
            print('  %-8s %-5s %-30s %-22s fs=%s'
                  % (ds, modality, c['stem'], str(c['zooms']), 'yes' if c['fs'] else 'no'))

    print('\n%d images, %d with a FreeSurfer anchor' % (len(rows), sum(1 for r in rows if r['fs'])))
    print('resolutions covered: %s' % sorted({r['zooms'] for r in rows}))
    if a.dry_run:
        print('\ndry run: nothing copied')
        return

    os.makedirs(os.path.join(out, 'img'), exist_ok=True)
    os.makedirs(os.path.join(out, 'gt', 'fs'), exist_ok=True)
    for r in rows:
        r['image_copy'] = os.path.join(out, 'img', os.path.basename(r['image']))
        shutil.copy2(r['image'], r['image_copy'])
        r['fs_copy'] = ''
        if r['fs']:
            r['fs_copy'] = os.path.join(out, 'gt', 'fs', '%s_aseg.mgz' % r['stem'])
            shutil.copy2(r['fs'], r['fs_copy'])

    # images.txt is every row; the _fs pair is the subset that has a FreeSurfer anchor, IN THE SAME
    # ORDER, which is the pairing predict_tm.py assumes. The SynthSeg pair is written after gt/ss exists.
    with open(os.path.join(out, 'images.txt'), 'w') as f:
        f.write('\n'.join(r['image_copy'] for r in rows) + '\n')
    fs_rows = [r for r in rows if r['fs_copy']]
    for name, key in (('images_fs.txt', 'image_copy'), ('segs_fs.txt', 'fs_copy')):
        with open(os.path.join(out, name), 'w') as f:
            f.write('\n'.join(r[key] for r in fs_rows) + '\n')

    # data only. The voxel size goes in three numeric columns rather than one 'a|b|c' string so it can be
    # sorted, filtered and plotted without being parsed back; what each row is FOR lives in RECIPE and in
    # this file's header, which is where prose can be corrected without rewriting a data file.
    with open(os.path.join(out, 'validation.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'modality', 'subject', 'stem', 'ax1', 'ax2', 'ax3',
                    'image', 'fs', 'source_image', 'source_fs'])
        for r in rows:
            w.writerow([r['dataset'], r['modality'], r['subject'], r['stem'],
                        r['zooms'][0], r['zooms'][1], r['zooms'][2],
                        r['image_copy'], r['fs_copy'], r['image'], r['fs'] or ''])

    print('\n%d images -> %s' % (len(rows), os.path.join(out, 'img')))
    if any(r['modality'] == 'FLAIR' for r in rows):
        print('REMINDER: the FLAIR rows have no usable CSF column -- the sequence nulls CSF. GM, WM and '
              'the deliverable are fine.')
    print('\nnext, the SynthSeg anchor over the images just copied:\n'
          '  python scripts/commands/SynthSeg_predict.py --i %s/img --o %s/gt/ss --crop 256 \\\n'
          '    --vol %s/gt/ss/vol.csv --qc %s/gt/ss/qc.csv' % (out, out, out, out))


if __name__ == '__main__':
    main()
