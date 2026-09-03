"""

Survey what we actually have: for every dataset, modality and native voxel size, how many images, how
they split, and whether they have an anchor to be scored against.

This is meant to run before the small real validation set is chosen, so the choice comes from the data
rather than from the summary lines in DATASETS.md. It reads HEADERS only -- no voxel data -- so a few
thousand files cost minutes, not hours.

Three things it reports that decide whether a row is usable at all:

  split    only dev images can be used to pick a checkpoint; using a test image spends it permanently.
  anchor   an image with no segmentation cannot be scored. FreeSurfer only ever ate the T1, so a T2w or
           a FLAIR can only have a SynthSeg anchor: that is recon-all, not a gap in the tree.
  zooms    the native voxel size, which is the resolution head's target and the reason the set has to
           span more than one value.

The FreeSurfer anchor count skips the derived streams. The longitudinal one re-runs each session from a
template built from both, so the sessions stop being independent, and the nipype working copies are
byte-identical duplicates: counting either inflates the anchor coverage of exactly the cohorts whose
repeat structure is the point.

Usage:
  python scripts/experiments/inventory_real_datasets.py --root <data>/qc-data
  python scripts/experiments/inventory_real_datasets.py --root <data>/qc-data --datasets kirby21,nifd

Writes inventory.csv (one row per image) next to --out, and prints the grouped table.

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
"""

import os
import csv
import glob
import argparse
import collections

import nibabel as nib

MODALITIES = ('T1w', 'T2w', 'FLAIR', 'T2starw', 'PD')
FS_DERIVED = ('freesurfer_longitudinal', 'freesurfer_unbiased_template', '/wd', '/ReconAll/', '/nipype')
VOL_EXT = ('.nii', '.nii.gz', '.mgz')


def stem_of(path):
    b = os.path.basename(path)
    for e in ('.nii.gz', '.nii', '.mgz'):
        if b.endswith(e):
            return b[:-len(e)]
    return os.path.splitext(b)[0]


def index_once(root, ds):
    """Every anchor under the dataset, indexed by the stems it mentions, walking each tree ONCE.

    A glob per image would re-walk the whole tree every time; on a BIDS-sized tree over a network filesystem that is
    the difference between minutes and hours, with nothing printed while it happens.
    """
    ss, fs = [], []
    d = os.path.join(root, 'anchors', 'synthseg-2.0', ds)
    for dirpath, _, files in os.walk(d):
        for f in files:
            if 'synthseg' in f and f.endswith(VOL_EXT):
                ss.append(os.path.join(dirpath, f))
    d = os.path.join(root, 'anchors', 'freesurfer', ds)
    for dirpath, _, files in os.walk(d):
        p = dirpath.replace('\\', '/')
        if any(x in p for x in FS_DERIVED):
            continue
        for f in files:
            if f == 'aseg.mgz':
                fs.append(os.path.join(dirpath, f))
    return ss, fs


def read_splits(root, ds):
    """{stem: split}. Missing lists are reported rather than guessed: picking a validation image without
    knowing its split is how a test set gets spent silently."""
    out = {}
    for name in ('dev', 'test', 'train'):
        p = os.path.join(root, 'index', ds, 'splits', '%s_stems.txt' % name)
        if os.path.isfile(p):
            with open(p) as f:
                for ln in f:
                    if ln.strip():
                        out[ln.strip()] = name
    return out


def modality_of(path):
    b = os.path.basename(path)
    for m in MODALITIES:
        if m in b:
            return m
    return 'other'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True, help='qc-data root')
    ap.add_argument('--datasets', default=None, help='comma separated; default is everything under raw/')
    ap.add_argument('--out', default=None, help='where inventory.csv goes (default: --root/index)')
    a = ap.parse_args()

    raw = os.path.join(a.root, 'raw')
    datasets = ([d.strip() for d in a.datasets.split(',')] if a.datasets
                else sorted(d for d in os.listdir(raw) if os.path.isdir(os.path.join(raw, d))))
    out_dir = a.out or os.path.join(a.root, 'index')
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for ds in datasets:
        bids = os.path.join(raw, ds, 'bids')
        if not os.path.isdir(bids):
            print('[skip] %s: no bids tree at %s' % (ds, bids))
            continue
        splits = read_splits(a.root, ds)
        if not splits:
            print('[WARNING] %s: no split lists under index/%s/splits -- every row will read "unknown", '
                  'and an unknown split cannot be used to pick a checkpoint' % (ds, ds))
        ss, fs = index_once(a.root, ds)
        print('%s: indexing images...' % ds, flush=True)

        imgs = []
        for dirpath, _, files in os.walk(bids):
            for f in files:
                if f.endswith(VOL_EXT):
                    imgs.append(os.path.join(dirpath, f))
        imgs.sort()

        for i, p in enumerate(imgs):
            if i and i % 250 == 0:
                print('  %d / %d' % (i, len(imgs)), flush=True)
            st = stem_of(p)
            try:
                z = tuple(round(float(v), 2) for v in nib.load(p).header.get_zooms()[:3])
            except Exception as e:                       # a corrupt file is a finding, not a crash
                print('  [unreadable] %s: %s' % (p, str(e)[:80]))
                continue
            subj = next((k for k in splits if k in st), None)
            rows.append({
                'dataset': ds,
                'modality': modality_of(p),
                'zooms': '|'.join('%.2f' % v for v in z),
                'iso': 'iso' if len(set(z)) == 1 else 'aniso',
                'split': splits.get(subj, 'unknown'),
                'anchor_synthseg': int(any(st in os.path.basename(x) for x in ss)),
                'anchor_freesurfer': int(any(st in x.replace('\\', '/') for x in fs)),
                'stem': st,
                'path': p,
            })
        print('%s: %d images' % (ds, sum(r['dataset'] == ds for r in rows)), flush=True)

    if not rows:
        raise SystemExit('nothing found under %s' % raw)

    path_csv = os.path.join(out_dir, 'inventory.csv')
    with open(path_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # the grouped table: this is the thing the validation set gets chosen from
    print('\n%-9s %-7s %-18s %-6s %-6s %6s %8s %8s' %
          ('dataset', 'mod', 'zooms', 'iso?', 'split', 'n', 'ss', 'fs'))
    print('-' * 78)
    key = lambda r: (r['dataset'], r['modality'], r['zooms'], r['split'])            # noqa: E731
    grouped = collections.Counter(key(r) for r in rows)
    anch = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        anch[key(r)][0] += r['anchor_synthseg']
        anch[key(r)][1] += r['anchor_freesurfer']
    for (ds, mod, z, sp), n in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1])):
        iso = 'iso' if len(set(z.split('|'))) == 1 else 'aniso'
        print('%-9s %-7s %-18s %-6s %-6s %6d %8d %8d' % (ds, mod, z, iso, sp, n, *anch[(ds, mod, z, sp)]))

    dev = [r for r in rows if r['split'] == 'dev' and (r['anchor_synthseg'] or r['anchor_freesurfer'])]
    print('\n%d images total, %d of them dev AND with an anchor -- those are the only ones the '
          'validation set can draw from.' % (len(rows), len(dev)))
    print('distinct native resolutions among those: %s'
          % sorted({r['zooms'] for r in dev}))
    print('\nwrote %s' % path_csv)


if __name__ == '__main__':
    main()
