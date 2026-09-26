"""Select and build minival v2: the real-data set every checkpoint of tm and rs is chosen on.

val10 could not choose an rs checkpoint: eight of its ten images sit at 1-1.5 mm, where the curve is flat,
and the other two are 3 mm FLAIR short of the 160 mm window, so the rule was decided by how a checkpoint
reads the zero padding. v2 keeps every image inside the window on every axis (no padding at all) and
spreads the slice thickness from 1 to 5 mm. One image per subject; the rule is a plain mean over images.

    with --scanners (the set in use):
    1  nifd    T1w    1 mm               dev
    1  nifd    T1w    1.2 mm (NIFD2)     dev
    2  ixi     T1w    1.2 mm             moved from test, two sites
    2  miriad  T1w    1.5 mm             moved from test, GE 1.5T
    2  nifd    FLAIR  1 mm               dev
    2  nifd    FLAIR  3 mm               the only two inside the window, both moved from test
    2  adni    FLAIR  3D, 1.1-1.6 mm     ses-M00, distinct sites
    3  adni    FLAIR  2D, 5 mm           ses-M00, distinct sites
    without it, the six T1w are all NIFD (3 at 1 mm, 3 NIFD2 at 1.2 mm, 2 of them moved from test)

Excluded: every NIFD, IXI and MIRIAD subject in a reported table (the scored cohorts, which stay
held-out) and the ADNI subjects of the bias pilot (arms A and B). val10 subjects are not excluded: val10 was dev already and v2 replaces it (the only
free 1.2 mm T1w in dev is a val10 subject). Held out entirely for evaluation: OASIS, HABS, MPI-Leipzig,
the T2starw modality, and the IXI T2w. The tm ground truth is SynthSeg 2.0 for every image (--crop 256), so no
image carries a different anchor from another.

Moving a subject from test to dev is allowed, the reverse never: moved_test_to_dev.txt lists them and has
to be appended to index/nifd/splits/ by hand.

Two steps, because the inputs live in three places:

    # local: small csvs only, writes <out>/manifest.csv, copy_from_icm.txt, moved_test_to_dev.txt
    python scripts/experiments/make_minival_v2.py select --root ../qc-data

    # Jean Zay, once the ADNI files are copied to <adni_dir> keeping their BIDS relative path
    python scripts/experiments/make_minival_v2.py build --root $WORK/qc-data \
        --adni_dir $WORK/qc-data/raw/adni_minival_v2
    python scripts/commands/SynthSeg_predict.py --i $WORK/qc-data/validation/minival_v2/img \
        --o $WORK/qc-data/validation/minival_v2/gt/ss --crop 256 \
        --vol $WORK/qc-data/validation/minival_v2/gt/ss/vol.csv --qc $WORK/qc-data/validation/minival_v2/gt/ss/qc.csv
"""
import argparse
import csv
import glob
import os
import shutil

import numpy as np

SEED = 2026
BOX = 160.
# NIFD subjects moved from test: the only two 3 mm FLAIR inside the window on every axis
FLAIR_3MM = ['sub-NIFD2S0032_ses-M06_FLAIR', 'sub-NIFD1S0018_ses-M00_FLAIR']


def read_csv(path, delimiter=','):
    with open(path, newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def subject(stem):
    return stem.split('_')[0]


def rel_to_root(path):
    """Inventory paths are absolute on Jean Zay; keep what follows qc-data/."""
    return path.split('/qc-data/', 1)[1]


def pick(cands, n, rng, used_subjects, used_sites=None):
    """n candidates at random, one per subject and, when given, one per site."""
    out = []
    for i in rng.permutation(len(cands)):
        c = cands[i]
        if c['subject'] in used_subjects or (used_sites is not None and c['site'] in used_sites):
            continue
        out.append(c)
        used_subjects.add(c['subject'])
        if used_sites is not None:
            used_sites.add(c['site'])
        if len(out) == n:
            return out
    raise RuntimeError('only %d of %d candidates for %s' % (len(out), n, cands[0]['group'] if cands else '?'))


def select(a):
    rng = np.random.default_rng(SEED)
    out = a.out or os.path.join(a.root, 'validation', 'minival_v2')
    os.makedirs(out, exist_ok=True)

    # ---- exclusions: every subject in a reported table (the scored cohorts), and the ADNI bias pilot
    reported = {ds: set() for ds in ('nifd', 'ixi', 'miriad')}
    for ds in reported:
        for p in glob.glob(os.path.join(a.root, 'scores', 'contrast', '*', ds + '*', '*.csv')):
            reported[ds] |= {subject(r['subject']) for r in read_csv(p) if 'subject' in r}
    pilot = {r['participant_id'] for r in read_csv(os.path.join(a.root, 'info', 'adni', 'pairs_pilot.tsv'), '\t')}
    print('excluded: reported subjects %s, %d ADNI bias-pilot subjects'
          % (', '.join('%s %d' % (k, len(v)) for k, v in reported.items()), len(pilot)))

    # ---- NIFD / IXI / MIRIAD candidates: inventory (split, path) x coverage (spacing, extent)
    ixi_site = {r['participant_id']: r['site']
                for r in read_csv(os.path.join(a.root, 'index', 'ixi', 'participants.tsv'), '\t')}
    inv = {r['stem']: r for r in read_csv(os.path.join(a.root, 'index', 'inventory.csv'))}
    cands = []
    for r in read_csv(os.path.join(a.root, 'coverage', 'coverage_all_axes.csv')):
        ds = r['dataset']
        if ds not in reported or r['short_axes'] != '':
            continue
        stem = r['file'].replace('.nii.gz', '').replace('.nii', '')
        sub = subject(stem)
        if sub in reported[ds] or stem not in inv:
            continue
        site = {'nifd': sub[4:9], 'ixi': ixi_site.get(sub, '?'), 'miriad': 'GE1.5T'}[ds]
        cands.append(dict(stem=stem, subject=sub, site=site, modality=r['modality'], dataset=ds,
                          spc=max(float(r['spc_%s' % ax]) for ax in 'RAS'), split=inv[stem]['split'],
                          src=rel_to_root(inv[stem]['path']),
                          spacing='|'.join(r['spc_%s' % ax] for ax in 'RAS'),
                          extent='|'.join(r['ext_%s' % ax] for ax in 'RAS')))

    def group(name, ds, modality, lo, hi, split):
        return [dict(c, group=name) for c in cands
                if c['dataset'] == ds and c['modality'] == modality and lo < c['spc'] <= hi and c['split'] == split]

    used = set()
    rows = []
    if a.scanners:
        # fewer NIFD T1w, and the T1w of other scanners, moved from test (never in a reported table)
        rows += pick(group('T1w 1 mm', 'nifd', 'T1w', 0, 1.1, 'dev'), 1, rng, used)
        rows += pick(group('T1w 1.2 mm', 'nifd', 'T1w', 1.1, 1.6, 'dev'), 1, rng, used)
        rows += pick(group('T1w 1.2 mm', 'ixi', 'T1w', 1.1, 1.6, 'unknown'), 2, rng, used, set())
        rows += pick(group('T1w 1.5 mm', 'miriad', 'T1w', 1.1, 1.6, 'unknown'), 2, rng, used)
    else:
        rows += pick(group('T1w 1 mm', 'nifd', 'T1w', 0, 1.1, 'dev'), 3, rng, used)
        rows += pick(group('T1w 1.2 mm', 'nifd', 'T1w', 1.1, 1.6, 'dev'), 1, rng, used)
        rows += pick(group('T1w 1.2 mm', 'nifd', 'T1w', 1.1, 1.6, 'unknown'), 2, rng, used)   # moved from test
    rows += pick(group('FLAIR 1 mm', 'nifd', 'FLAIR', 0, 1.1, 'dev'), 2, rng, used)
    for stem in FLAIR_3MM:
        c = [dict(x, group='FLAIR 3 mm') for x in cands if x['stem'] == stem]
        assert len(c) == 1 and c[0]['subject'] not in used, stem
        rows += c
        used.add(c[0]['subject'])

    # ---- ADNI candidates: raw FLAIR headers read on ICM (coverage_adni_flair.py), baseline session only
    adni = []
    for r in read_csv(os.path.join(a.root, 'coverage', 'coverage_adni_flair.csv')):
        if r['short_axes'] != '' or r['participant_id'] in pilot or r['session_id'] != 'ses-M00':
            continue
        stem = os.path.basename(r['path']).replace('.nii.gz', '').replace('.nii', '')
        adni.append(dict(stem=stem, subject=r['participant_id'], site=r['site'].split('.')[0], modality='FLAIR',
                         spc=max(float(r['spc_%s' % ax]) for ax in 'RAS'), split='', dataset='adni',
                         src=r['path'], is_3D=r['is_3D'], study=r['original_study'],
                         spacing='|'.join(r['spc_%s' % ax] for ax in 'RAS'),
                         extent='|'.join(r['ext_%s' % ax] for ax in 'RAS')))
    sites = set()
    rows += pick([dict(c, group='FLAIR 3D 1.2 mm') for c in adni if c['is_3D'] == 'True' and 1.1 < c['spc'] <= 1.6],
                 2, rng, used, sites)
    rows += pick([dict(c, group='FLAIR 2D 5 mm') for c in adni if c['is_3D'] == 'False' and abs(c['spc'] - 5.) < .01],
                 3, rng, used, sites)

    # ---- write
    moved = sorted({r['subject'] for r in rows if r['split'] == 'unknown'})
    cols = ['stem', 'dataset', 'subject', 'group', 'modality', 'spacing', 'extent', 'site', 'split',
            'moved_from_test', 'src']
    with open(os.path.join(out, 'manifest.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, moved_from_test=int(r['subject'] in moved)))
    with open(os.path.join(out, 'copy_from_icm.txt'), 'w', newline='\n') as f:
        f.writelines(r['src'] + '\n' for r in rows if r['dataset'] == 'adni')
    with open(os.path.join(out, 'moved_test_to_dev.txt'), 'w', newline='\n') as f:
        f.writelines(s + '\n' for s in moved)

    for r in rows:
        origin = 'test -> dev' if r['subject'] in moved else 'dev' if r['dataset'] != 'adni' else r['study']
        print('%-16s %-6s %-34s %-16s %-7s %s' % (r['group'], r['dataset'], r['stem'], r['spacing'], r['site'], origin))
    print('\n%d images -> %s' % (len(rows), out))


def build(a):
    out = a.out or os.path.join(a.root, 'validation', 'minival_v2')
    rows = read_csv(os.path.join(out, 'manifest.csv'))
    img = os.path.join(out, 'img')
    os.makedirs(img, exist_ok=True)
    for r in rows:
        # ADNI comes from the copy of the ICM tree; NIFD, IXI and MIRIAD are already under <root>/raw
        src = os.path.join(a.adni_dir, r['src']) if r['dataset'] == 'adni' else os.path.join(a.root, r['src'])
        assert os.path.isfile(src), 'missing: %s' % src
        dst = os.path.join(img, os.path.basename(src))
        if not os.path.isfile(dst):
            shutil.copy2(src, dst)
    print('%d images in %s' % (len(os.listdir(img)), img))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['select', 'build'])
    ap.add_argument('--root', required=True, help='the qc-data root')
    ap.add_argument('--out', default=None, help='default: <root>/validation/minival_v2')
    ap.add_argument('--scanners', action='store_true',
                    help='select: swap 4 NIFD T1w for 2 IXI + 2 MIRIAD T1w moved from test (other scanners)')
    ap.add_argument('--adni_dir', default=None, help='build: where the ADNI files were copied, BIDS relative paths kept')
    a = ap.parse_args()
    select(a) if a.step == 'select' else build(a)


if __name__ == '__main__':
    main()
