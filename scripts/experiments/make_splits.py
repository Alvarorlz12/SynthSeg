"""Split the 992 label maps into train / val / test at the level of PATIENT, not of file.

Two reasons the naive split is wrong, both measured on this dataset:

  1. `training_biasfield_scalar.py:213-216` slices the sorted paths and takes the tail. Sorted,
     ADNI2 goes entirely before HCP, so `holdout=100` puts 100 HCP maps in validation and leaves
     every single ADNI2 map inside training. No clinical anatomy is ever held out.
  2. The 492 ADNI2 files are only 321 patients: 101 subjects have 2 visits, 24 have 3, 6 have 4
     and 1 has 5. Splitting by file puts the same brain in train and in test.

So the unit here is the patient, and a patient is never split across two folders. What to do with the
repeat visits of a held-out patient is a separate choice, `--holdout_visits`:

  all  (default) every visit of a held-out patient joins that patient's split. Nothing is discarded and
       nothing leaks, because the leak is between splits and the whole patient is on one side. The catch
       is weighting: a 5-visit patient would carry 5x the images of a 1-visit one. That is neutralised
       downstream, by giving every held-out patient the SAME number of images in the frozen set and
       spreading the severity grid over whichever visits it has.
  one  only the first visit represents the patient; the rest go to `retest/`. Costs 41 maps out of 992.

Either way training is identical (791 maps): every file of a held-out patient leaves training regardless.

`retest/` is not a bin. Two scans of one brain months apart are the only thing in this dataset that
isolates "same patient, different session", which is the empirical version of the objection that the
severity target depends on the anatomy. Under `--holdout_visits all` those pairs live inside val/test
and the same analysis is still available.

The strata are cohort x site x visit-count. The ADNI site is free: it is the `NNN` of the
`NNN_S_NNNN` subject ID (30 sites, from 27 patients down to 1), and it stands in for the scanner.
The visit-count bucket is in the strata on purpose: in ADNI, patients who drop out or progress
have fewer follow-ups, so holding out single-visit patients preferentially would bias the test set
towards a different disease trajectory. Costs ~43 discarded maps; buys a holdout we can defend.

Usage
-----
  plan   python scripts/make_splits.py plan  --src <dir with *_seg_cerebral.nii.gz> --out splits/
         [--holdout_visits all|one] [--n_val 40] [--n_test 40] [--seed 0]
  apply  python scripts/make_splits.py apply --src <dir> --dst <dir> --manifest splits/manifest.csv
         [--suffix _seg_cerebral.nii.gz] [--mode link|copy|move]     ONE variant per --dst
  stats  python scripts/make_splits.py stats --src <dir> --manifest splits/manifest.csv   (needs nibabel)

`apply` is run once per variant, each into its OWN tree, from the SAME manifest, so a patient lands
on the same side in all of them.
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

import hashlib

import numpy as np

# The identity of a map is its STEM, the part that names subject and visit. Whatever follows is
# a variant suffix that some copies of this dataset carry and some do not:
#   ADNI2_009_S_0842v31_i448580_seg_cerebral.nii.gz  and  ADNI2_009_S_0842v31_i448580.nii.gz
# are the same map. So the patterns are not anchored at the end and every lookup goes through
# the stem, which lets a manifest built under one naming convention drive a folder that uses
# the other.
#   ADNI2_009_S_0842v31_i448580  ->  patient 009_S_0842, site 009, visit v31
#   HCP_100206                   ->  patient HCP_100206, no site, one visit by construction
RE_ADNI = re.compile(r'^ADNI2_((\d+)_S_\d+)([a-z]\d+)_i\d+')
RE_HCP = re.compile(r'^(HCP_\d+)')


def stem_of(fname):
    '''(stem, cohort, patient, site, visit) for one filename, whatever suffix it carries.'''
    m = RE_ADNI.match(fname)
    if m:
        return m.group(0), 'ADNI2', m.group(1), m.group(2), m.group(3)
    m = RE_HCP.match(fname)
    if m:
        return m.group(0), 'HCP', m.group(1), 'HCP', ''
    raise ValueError('filename does not match either cohort pattern: %s' % fname)


def index_by_stem(src, suffix):
    '''stem -> the one file carrying it. Two files for one stem means the folder holds more than
    one variant, and choosing between them silently would be a coin flip that decides which label
    vocabulary the run gets, so it asks for --suffix instead.'''
    out = {}
    for fname in sorted(os.listdir(src)):
        if not (fname.endswith('.nii.gz') and fname.endswith(suffix)):
            continue
        stem = stem_of(fname)[0]
        if stem in out:
            raise SystemExit('%s holds more than one file for %s (%s and %s). Pass --suffix to '
                             'say which variant you mean.' % (src, stem, out[stem], fname))
        out[stem] = fname
    assert out, 'no .nii.gz matching %r found in %s' % (suffix, src)
    return out

SPLITS = ('train', 'val', 'test')


def parse_maps(src, suffix):
    """One row per map: (stem, cohort, patient, site, visit).

    `suffix` picks a variant when the folder holds several. The full archive keeps four maps per
    subject (cerebral, cerebral_lesions, extra_cerebral, extra_cerebral_lesions) side by side, and
    without the filter every patient would read as multi-visit and the visit stratum would become
    meaningless. index_by_stem refuses that case rather than guessing."""
    return [(stem,) + stem_of(fname)[1:]
            for stem, fname in sorted(index_by_stem(src, suffix).items())]


def largest_remainder(counts, total):
    """Allocate `total` items over strata of size `counts` proportionally, exactly. Plain rounding
    does not sum to `total`; the largest-remainder rule does, and it is deterministic."""
    if total <= 0:
        return [0] * len(counts)
    n = sum(counts)
    exact = [total * c / n for c in counts]
    alloc = [int(np.floor(e)) for e in exact]
    order = sorted(range(len(counts)), key=lambda i: (-(exact[i] - alloc[i]), i))
    for i in order[:total - sum(alloc)]:
        alloc[i] += 1
    # never take more patients out of a stratum than it has
    for i, c in enumerate(counts):
        alloc[i] = min(alloc[i], c)
    return alloc


def plan(src, out, n_val, n_test, seed, suffix, holdout_visits, force=False):
    rows = parse_maps(src, suffix)

    visits = defaultdict(list)
    meta = {}
    for stem, cohort, patient, site, visit in rows:
        visits[patient].append(stem)
        meta[patient] = (cohort, site)
    for p in visits:
        visits[p].sort()  # deterministic choice of the visit that represents a held-out patient

    # strata: cohort x site x (1 visit | 2+ visits)
    strata = defaultdict(list)
    for patient, files in visits.items():
        cohort, site = meta[patient]
        strata[(cohort, site, '1' if len(files) == 1 else '2+')].append(patient)

    # the shuffle is a hash of the patient id, not a draw from an RNG. two reasons. numpy only
    # freezes the stream of the legacy RandomState; Generator/PCG64 is explicitly exempt (NEP 19),
    # so a numpy upgrade could silently reshuffle the split. And a hash is a property of the patient
    # rather than of the draw, so the order inside a stratum does not move when the set of maps
    # changes: measured on this dataset, 20 new ADNI patients move 1% of the existing ones instead
    # of the 18-24% either RNG moves. `seed` salts the hash and still selects a different split.
    def hkey(patient):
        return hashlib.md5(('%d/%s' % (seed, patient)).encode()).hexdigest()

    split_of = {}
    for cohort in ('ADNI2', 'HCP'):
        keys = sorted(k for k in strata if k[0] == cohort)
        sizes = [len(strata[k]) for k in keys]
        pools = []
        for k in keys:
            pools.append(sorted(strata[k], key=hkey))
        # val first, then test out of what is left, so both are proportional to the same strata
        for split, want in (('val', n_val), ('test', n_test)):
            avail = [len(p) for p in pools]
            for i, take in enumerate(largest_remainder(avail, want)):
                for _ in range(take):
                    split_of[pools[i].pop()] = split
        assert sum(sizes) - len([p for p in split_of if meta[p][0] == cohort]) >= 0
        for pool in pools:
            for patient in pool:
                split_of[patient] = 'train'

    # the repeat visits of a held-out patient either join it or go to retest; never to training
    manifest = []
    for stem, cohort, patient, site, visit in rows:
        split = split_of[patient]
        if holdout_visits == 'one' and split in ('val', 'test') and stem != visits[patient][0]:
            split = 'retest'
        manifest.append(dict(stem=stem, cohort=cohort, patient=patient, site=site,
                             visit=visit, n_visits=len(visits[patient]), split=split))

    _verify(manifest, holdout_visits)
    os.makedirs(out, exist_ok=True)
    path_manifest = os.path.join(out, 'manifest.csv')
    _refuse_silent_change(path_manifest, manifest, force)
    with open(path_manifest, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['stem', 'cohort', 'patient', 'site', 'visit',
                                          'n_visits', 'split'])
        w.writeheader()
        w.writerows(manifest)
    for split in SPLITS:
        with open(os.path.join(out, 'patients_%s.txt' % split), 'w') as f:
            f.write('\n'.join(sorted({r['patient'] for r in manifest if r['split'] == split})) + '\n')
    _report(manifest)
    print('\nwrote %s and patients_{train,val,test}.txt' % path_manifest)


def _refuse_silent_change(path, manifest, force):
    """A split that changes without anyone noticing invalidates every score taken under the old one,
    and nothing downstream can tell. So overwriting a manifest with a different one is an error, not
    a default."""
    if force or not os.path.isfile(path):
        return
    with open(path) as f:
        old_rows = list(csv.DictReader(f))
    if old_rows and 'stem' not in old_rows[0]:
        raise SystemExit('%s holds a manifest in the older format, keyed by filename instead of by '
                         'stem. Delete it, or pass --force to replace it.' % path)
    old = {r['stem']: r['split'] for r in old_rows}
    new = {r['stem']: r['split'] for r in manifest}
    if old == new:
        print('the manifest already on disk is identical, rewriting it changes nothing')
        return
    shared = set(old) & set(new)
    raise SystemExit(
        'refusing to overwrite %s: it holds a DIFFERENT split (%d of %d maps change side, %d new, '
        '%d gone). Anything already scored under it would stop being comparable. Pass --force if '
        'that is what you mean.' % (path, sum(old[k] != new[k] for k in shared), len(shared),
                                    len(set(new) - set(old)), len(set(old) - set(new))))


def _verify(manifest, holdout_visits):
    """The asserts that make the leak impossible, not merely unlikely. The one that matters is the
    first: no patient in two splits. The per-patient file count is a property of --holdout_visits,
    not an invariant, so it is only checked in the mode that promises it."""
    by_split = defaultdict(set)
    for r in manifest:
        by_split[r['split']].add(r['patient'])
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                shared = by_split[a] & by_split[b]
                assert not shared, 'patient in both %s and %s: %s' % (a, b, sorted(shared)[:5])
    if holdout_visits == 'one':
        for split in ('val', 'test'):
            per_patient = Counter(r['patient'] for r in manifest if r['split'] == split)
            bad = {p: n for p, n in per_patient.items() if n != 1}
            assert not bad, 'with --holdout_visits one each held-out patient gives one map: %s' % bad
    train = {r['patient'] for r in manifest if r['split'] == 'train'}
    held = {r['patient'] for r in manifest if r['split'] in ('val', 'test', 'retest')}
    assert not (train & held), 'patient both in training and held out: %s' % sorted(train & held)[:5]
    assert len({r['stem'] for r in manifest}) == len(manifest), 'duplicate stems'


def _report(manifest):
    print('%-8s %6s %6s %8s %8s' % ('split', 'files', 'patients', 'ADNI2', 'HCP'))
    for split in SPLITS + ('retest',):
        rows = [r for r in manifest if r['split'] == split]
        if not rows:
            continue
        print('%-8s %6d %6d %8d %8d' % (
            split, len(rows), len({r['patient'] for r in rows}),
            sum(r['cohort'] == 'ADNI2' for r in rows), sum(r['cohort'] == 'HCP' for r in rows)))
    print('\nvisit-count balance (ADNI2 patients, %% with 2+ visits) -- these should be close:')
    for split in SPLITS:
        pats = {r['patient']: r['n_visits'] for r in manifest
                if r['split'] == split and r['cohort'] == 'ADNI2'}
        if pats:
            multi = sum(n > 1 for n in pats.values())
            print('  %-6s %3d patients, %4.1f%% multi-visit' % (split, len(pats),
                                                               100 * multi / len(pats)))
    print('\ndistinct ADNI2 sites in each split (of 30): ' + ', '.join(
        '%s %d' % (s, len({r['site'] for r in manifest
                           if r['split'] == s and r['cohort'] == 'ADNI2'}))
        for s in SPLITS))


def apply_(src, dst, manifest_path, suffix, mode):
    """Materialise train/val/test under `dst` for ONE variant, chosen by --suffix.

    One variant per destination tree, always. The four variants do not share a label vocabulary: the
    extra-cerebral maps carry the extra-cranial labels and 531, the cerebral ones do not. A folder
    holding two of them would hand the generator maps its generation_labels array cannot describe,
    and the failure is not an exception, it is a wrong image. So the manifest is built once, on one
    variant, and applied once per variant into its own tree:

        apply --dst $WORK/label_maps_cerebral       --suffix _seg_cerebral.nii.gz
        apply --dst $WORK/label_maps_extra_cerebral --suffix _seg_extra_cerebral.nii.gz

    The same patient is then on the same side in every tree, because it is the same manifest.

    --mode link (default) makes a hard link, not a symlink: a second directory entry for the same
    inode. The bytes are as much "in" train/ as in the archive, there is no indirection and nothing
    to dangle, and it costs no disk. It needs src and dst on the SAME filesystem. --mode copy is the
    fallback across filesystems; --mode move leaves no archive behind, and the manifest makes it
    reversible. Symlinks are not an option: the 20-map run of July used `ln -s` with relative targets
    and broke.

    The parent `dst` is left with no image in it on purpose: `utils.list_images_in_folder` globs one
    level only and asserts non-empty, so pointing a training run at the parent raises instead of
    silently training on everything."""
    import shutil
    with open(manifest_path) as f:
        manifest = list(csv.DictReader(f))
    have = index_by_stem(src, suffix)
    for split in sorted({r['split'] for r in manifest}):
        os.makedirs(os.path.join(dst, split), exist_ok=True)
    place = {'link': os.link, 'copy': shutil.copy2, 'move': shutil.move}[mode]
    n_done, n_missing = Counter(), []
    for r in manifest:
        if r['stem'] not in have:
            n_missing.append((r['stem'], r['split']))
            continue
        fname = have[r['stem']]
        target = os.path.join(dst, r['split'], fname)
        if not os.path.exists(target):
            place(os.path.join(src, fname), target)
        n_done[r['split']] += 1
    print('variant %s, mode %s' % (suffix or '(the only one in the folder)', mode))
    for split, n in sorted(n_done.items()):
        print('  %-8s %5d files' % (split, n))
    if n_missing:
        # a hole in training costs one map out of 790 and nothing else. a hole in val or test means
        # this variant's held-out set is not the same set as the other variants', so a run on one
        # would be scored against different images than a run on the other, and the comparison
        # between them would be quietly wrong. HCP_158136 has no _seg_extra_cerebral_lesions (991 of
        # 992) and it is in train, which is why this is a warning today and not a failure.
        held = [f for f, sp in n_missing if sp != 'train']
        if held:
            raise SystemExit(
                'refusing to build %s: %d map(s) of the held-out sets are absent from %s, so this '
                'variant would carry a different val/test than the others: %s'
                % (dst, len(held), src, held[:5]))
        print('  WARNING: %d file(s) absent from %s, all in train: %s'
              % (len(n_missing), src, [f for f, _ in n_missing][:3]))


def stats(src, manifest_path):
    """One pass over the maps to check the splits are anatomically matched. This is a CHECK, not a
    constraint: the reading noise of the head was measured to be contrast, not anatomy (nb26), so
    this exists to put one defensible sentence in the report, and to catch a pathological draw."""
    import nibabel as nib
    with open(manifest_path) as f:
        manifest = list(csv.DictReader(f))
    ventricles, brain = (4, 43, 5, 44), None
    acc = defaultdict(list)
    for i, r in enumerate(manifest):
        if r['split'] == 'retest':
            continue
        path = os.path.join(src, r['file'])
        if not os.path.isfile(path):
            continue
        vol = np.asarray(nib.load(path).dataobj)
        counts = np.bincount(vol.reshape(-1).astype('int64'))
        total = counts[1:].sum()
        vent = sum(counts[l] for l in ventricles if l < len(counts))
        acc[(r['cohort'], r['split'])].append((total, vent / max(total, 1)))
        if i % 100 == 0:
            print('  %d/%d' % (i, len(manifest)), file=sys.stderr)
    print('%-8s %-6s %5s %14s %16s' % ('cohort', 'split', 'n', 'brain vox (k)', 'ventricle frac'))
    for key in sorted(acc):
        a = np.array(acc[key])
        print('%-8s %-6s %5d  %6.1f +- %-5.1f  %6.4f +- %-6.4f' % (
            key[0], key[1], len(a), a[:, 0].mean() / 1e3, a[:, 0].std() / 1e3,
            a[:, 1].mean(), a[:, 1].std()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('plan')
    q.add_argument('--src', required=True, help='folder of *_seg_cerebral.nii.gz')
    q.add_argument('--out', default='splits')
    q.add_argument('--n_val', type=int, default=40, help='patients per cohort in val')
    q.add_argument('--n_test', type=int, default=40, help='patients per cohort in test')
    q.add_argument('--seed', type=int, default=0, help='salts the per-patient hash')
    q.add_argument('--force', action='store_true',
                   help='overwrite an existing manifest that holds a different split')
    q.add_argument('--holdout_visits', choices=['all', 'one'], default='all',
                   help="'all' keeps every visit of a held-out patient in its split (nothing discarded); "
                        "'one' keeps the first and parks the rest in retest/")
    q.add_argument('--suffix', default='',
                   help="variant filter, e.g. '_seg_cerebral.nii.gz'. Leave it empty when the "
                        'folder already holds exactly one map per subject')
    q = sub.add_parser('apply')
    q.add_argument('--src', required=True, help='folder holding the maps (any or all variants)')
    q.add_argument('--dst', required=True, help='destination tree for ONE variant')
    q.add_argument('--manifest', default='splits/manifest.csv')
    q.add_argument('--suffix', default='',
                   help='variant filter; empty when the folder holds one map per subject')
    q.add_argument('--mode', choices=['link', 'copy', 'move'], default='link',
                   help='link = hard link, no extra disk, needs the same filesystem')
    q = sub.add_parser('stats')
    q.add_argument('--src', required=True)
    q.add_argument('--manifest', default='splits/manifest.csv')
    a = p.parse_args()
    if a.cmd == 'plan':
        plan(a.src, a.out, a.n_val, a.n_test, a.seed, a.suffix, a.holdout_visits, a.force)
    elif a.cmd == 'apply':
        apply_(a.src, a.dst, a.manifest, a.suffix, a.mode)
    else:
        stats(a.src, a.manifest)
