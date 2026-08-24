"""Split the REAL datasets (IXI, Kirby21) into dev / test, at the level of SUBJECT.

Sibling of make_splits.py, which does the same for the synthetic label maps. Same guarantees --
subject-level unit, hash assignment, a manifest that refuses to change in silence -- and three
differences that follow from these being evaluation-only data:

  1. There is no `train`. The networks are trained 100% on synthetic images, so what a real dataset
     needs is a set you are allowed to look at while deciding things (`dev`) and one that is read
     once, for the report (`test`).
  2. Kirby21 goes entirely to `dev` and is not split. It has 21 subjects, so two halves would both
     be useless; and it is already spent -- the transfer read, the divisor sweep and the --clip 300
     decision were all taken while looking at it. A set already used to decide things is not a test
     set. Its manifest still records the two sessions per subject, because scan-rescan is the one
     real-data metric that needs no anchor at all.
  3. The site is no longer in the filename. The BIDS conversion renamed IXI to `sub-IXINNN` and
     dropped it; it survives in `unorganised/IXI-T{1,2}/IXINNN-<Site>-...`. So the site is recovered
     by joining on the numeric id, and the join is audited in both directions rather than assumed: a
     subject whose site cannot be recovered cannot be stratified, and is reported and left out
     instead of being quietly dropped into some group.

IXI ships no participants.tsv of its own, and the two folders that still carry the site live on a
different cluster from everything else. So the site table is built ONCE, from a listing of those two
folders, and after that nothing reaches across clusters:

  there    ls -1 unorganised/IXI-T1 > t1.txt ; ls -1 unorganised/IXI-T2 > t2.txt   (copy both over)
  here     ... participants --t1_list t1.txt --t2_list t2.txt --bids <.../raw/ixi/bids> \n                            --out <.../index/ixi>
           ... ixi --participants <.../index/ixi/participants.tsv> ...

`participants` also copies the two listings, verbatim, next to the table it derives. The table is a
parse of them, so keeping the input means the parse can be re-checked or redone; a derived file whose
source is gone has to be taken on faith.

Growing the dev set later is safe and shrinking it is not, which is why the default dev is small. The
order inside a stratum is a hash of the subject id and the allocation always takes from the same end,
so raising --n_dev only ever moves subjects from test to dev: measured, dev(100) is a subset of
dev(150) is a subset of dev(290), with nothing leaving dev. The reverse -- a subject that has been in
dev going back to test -- can never be undone, because you have already looked at it.

Usage
-----
  table   python scripts/experiments/make_splits_research_datasets.py participants \
              --t1_list t1.txt [--t2_list t2.txt] --bids <.../raw/ixi/bids> --out <.../index/ixi>

          `source_t1`/`source_t2` in the table are the filenames on the OTHER cluster and do not
          resolve here: they are provenance, and they are where the site comes from. `bids_t1`/
          `bids_t2` are the paths that exist here, and they are what decides which modalities of a
          subject can be scored. The two are reported against each other, because a conversion that
          dropped a volume would otherwise show up as a modality that silently never gets read.

  IXI     python scripts/experiments/make_splits_research_datasets.py ixi \
              (--participants <.../index/ixi/participants.tsv> | --t1 <dir> [--t2 <dir>]) \
              --bids <.../raw/ixi/bids> --segs <.../anchors/synthseg-2.0/ixi/segs> \
              --out <.../index/ixi/splits> [--n_dev 100] [--seed 0] [--force]

  Kirby   python scripts/experiments/make_splits_research_datasets.py kirby21 \
              --index <.../index/kirby21/fsorig> --out <.../index/kirby21/splits> [--force]

The manifest holds one row per SUBJECT, which is one row per decision. `volumes.csv` keeps one row
per (volume x anchor) and picks the split up by joining on `subject`: a derived column there cannot
contradict itself, because the authority is here.
"""

import argparse
import csv
import glob
import hashlib
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_splits import largest_remainder                                    # noqa: E402

RE_IXI_RAW = re.compile(r'^IXI(\d+)-([A-Za-z]+)-(\d+)-T([12])\.nii\.gz$')
RE_IXI_BIDS = re.compile(r'^sub-IXI(\d+)$')
RE_IXI_SEG = re.compile(r'^sub-IXI(\d+)_T([12])w_synthseg\.nii\.gz$')
RE_KIRBY = re.compile(r'^sub-(KKI\d+)_ses-(\d+)_T1w\.mgz$')

FIELDS = ('dataset', 'subject', 'site', 'modalities', 'n_sessions', 'stratum', 'split')
# `source_*` are the filenames on the OTHER cluster: provenance. They carry the site and the scan
# id, and they do NOT resolve here. `bids_*` are the paths that exist on this one, relative to the
# BIDS root, and they are what says whether a modality can actually be scored.
PART_FIELDS = ('participant_id', 'ixi_id', 'site',
               'source_t1', 'source_t2', 'bids_t1', 'bids_t2')


def _key(seed, subject):
    """The order inside a stratum is a hash of the subject id, not a draw from an RNG: it is a
    property of the subject, so it does not move when the set of subjects changes. Same reasoning,
    and same measurement, as in make_splits.py."""
    return hashlib.md5(('%d/%s' % (seed, subject)).encode()).hexdigest()


def _refuse_silent_change(path, manifest, force):
    """Overwriting a manifest with a different one invalidates every score already taken under it,
    and nothing downstream can tell. Same guarantee as make_splits.py, keyed on `subject` because
    the unit here is the subject and not the map stem."""
    if force or not os.path.isfile(path):
        return
    with open(path) as f:
        old = {r['subject']: r['split'] for r in csv.DictReader(f)}
    new = {r['subject']: r['split'] for r in manifest}
    if old == new:
        print('the manifest already on disk is identical, rewriting it changes nothing')
        return
    shared = set(old) & set(new)
    raise SystemExit(
        'refusing to overwrite %s: it holds a DIFFERENT split (%d of %d subjects change side, %d '
        'new, %d gone). Anything already scored under it would stop being comparable. Pass --force '
        'if that is what you mean.'
        % (path, sum(old[k] != new[k] for k in shared), len(shared),
           len(set(new) - set(old)), len(set(old) - set(new))))


def _write(out, manifest, force):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'manifest.csv')
    _refuse_silent_change(path, manifest, force)
    with open(path, 'w', newline='') as f:
        # lineterminator: the csv module's excel dialect writes CRLF, and these tables get read by
        # shell tools too, where a trailing \r makes the last field of every row fail to compare.
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
        w.writeheader()
        for r in manifest:
            w.writerow(r)
    print('\nwrote %s  (%d subjects)' % (path, len(manifest)))


def _scan(folder, regex, what):
    """{id: [match groups]} over the files of `folder` that match, refusing an empty folder."""
    if folder is None:
        return {}
    out = defaultdict(list)
    for name in sorted(os.listdir(folder)):
        m = regex.match(name)
        if m:
            out[m.group(1)].append(m.groups()[1:])
    if not out:
        raise SystemExit('%s: no file matches the expected %s pattern' % (folder, what))
    return dict(out)


# =================================================================================================
def _parse_listing(path, mod):
    """An `ls -1` of one of the source folders -> {id: (site, filename)}.

    Lines that do not match are COUNTED and shown. The failure this guards against is pasting a
    multi-column `ls` instead of `ls -1`: a handful of names would match and the rest would vanish
    without a word, leaving a half-empty site table that looks complete."""
    out, bad = {}, []
    with open(path) as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            m = RE_IXI_RAW.match(name)
            if not m or m.group(4) != mod[-1]:
                bad.append(name)
                continue
            out[m.group(1)] = (m.group(2), name)
    print('  %-8s %4d valid entries from %s' % (mod, len(out), path))
    if bad:
        print('     [WARNING] %d lines do not match IXI###-<Site>-<id>-%s.nii.gz. If there are many, '
              'the file is not an `ls -1`. First: %s' % (len(bad), mod, ', '.join(bad[:3])))
    if not out:
        raise SystemExit('%s yields no valid entry at all' % path)
    return out


def _bids_paths(bids):
    """{id: {'T1': relpath, 'T2': relpath}} by globbing the BIDS tree, layout-agnostic.

    Globbed rather than built from a template because the tree came out of someone else's converter:
    `anat/` may or may not be there, the extension may be .nii or .nii.gz, and a session level would
    give a subject two T1w. More than one match is refused instead of picked, because picking would
    decide in silence which volume every later number is about."""
    if bids is None:
        return {}
    out = {}
    for name in sorted(os.listdir(bids)):
        m = RE_IXI_BIDS.match(name)
        if not m:
            continue
        for mod in ('T1', 'T2'):
            hits = sorted(h for h in glob.glob(os.path.join(bids, name, '**', '*_%sw.nii*' % mod),
                                               recursive=True) if not h.endswith('.json'))
            if len(hits) > 1:
                raise SystemExit('%s holds %d %sw images under %s; one of them would be chosen '
                                 'silently: %s' % (name, len(hits), mod, bids,
                                                   [os.path.relpath(h, bids) for h in hits]))
            if hits:
                out.setdefault(m.group(1), {})[mod] = \
                    os.path.relpath(hits[0], bids).replace(os.sep, '/')
    return out


def build_participants(t1_list, t2_list, bids, out):
    """The site table IXI does not ship, derived from the listing of the source folders and joined to
    what actually exists in the BIDS tree here."""
    print('== source listings ==')
    t1 = _parse_listing(t1_list, 'T1')
    t2 = _parse_listing(t2_list, 'T2') if t2_list else {}

    disagree = {s: (t1[s][0], t2[s][0]) for s in set(t1) & set(t2) if t1[s][0] != t2[s][0]}
    if disagree:
        raise SystemExit('T1 and T2 give different sites for %d subjects; that is an inconsistency '
                         'in the data, not something to choose: %s' % (len(disagree), disagree))

    here = _bids_paths(bids)
    if bids:
        print('  %-8s %4d subjects resolved in the BIDS tree (T1w %d, T2w %d)'
              % ('BIDS', len(here), sum('T1' in v for v in here.values()),
                 sum('T2' in v for v in here.values())))

    os.makedirs(out, exist_ok=True)
    rows, mismatch = [], []
    for sid in sorted(set(t1) | set(t2) | set(here), key=int):
        src = {m: d[sid][1] for m, d in (('T1', t1), ('T2', t2)) if sid in d}
        got = here.get(sid, {})
        site = (t1.get(sid) or t2.get(sid) or ('', ''))[0]
        if bids and set(src) != set(got):
            mismatch.append((sid, '+'.join(sorted(src)) or '-', '+'.join(sorted(got)) or '-'))
        rows.append(dict(participant_id='sub-IXI%s' % sid, ixi_id=sid, site=site,
                         source_t1=src.get('T1', ''), source_t2=src.get('T2', ''),
                         bids_t1=got.get('T1', ''), bids_t2=got.get('T2', '')))

    if mismatch:
        print('\n  [WARNING] %d subjects where the source listing and the BIDS tree disagree on which '
              'modalities exist. What gets scored is the BIDS side; the listing is only where the '
              'site comes from.' % len(mismatch))
        for sid, a, b in mismatch[:10]:
            print('     IXI%s  source=%-5s bids=%s' % (sid, a, b))
        if len(mismatch) > 10:
            print('     ... and %d more' % (len(mismatch) - 10))
    no_site = [r['ixi_id'] for r in rows if not r['site']]
    if no_site:
        print('\n  [WARNING] %d subjects in the BIDS tree with no source listing => no site: %s'
              % (len(no_site), ', '.join('IXI' + x for x in no_site)))

    path = os.path.join(out, 'participants.tsv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=PART_FIELDS, delimiter='\t', lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # the raw listings are kept alongside: the table is a parse of them, and a derived file whose
    # source has gone has to be taken on faith.
    for src, mod in ((t1_list, 'T1'), (t2_list, 'T2')):
        if src:
            with open(src) as a, open(os.path.join(out, 'source_listing_%s.txt' % mod), 'w') as b:
                b.write(a.read())

    sites = defaultdict(int)
    for r in rows:
        sites[r['site']] += 1
    print('\n== participants.tsv ==')
    for s in sorted(sites):
        print('  %-6s %4d' % (s, sites[s]))
    print('  %-6s %4d subjects' % ('TOTAL', len(rows)))
    for label, key in (('source', 'source_%s'), ('bids  ', 'bids_%s')):
        a = sum(1 for r in rows if r[key % 't1'])
        b = sum(1 for r in rows if r[key % 't2'])
        both = sum(1 for r in rows if r[key % 't1'] and r[key % 't2'])
        print('    %s: T1 %d, T2 %d, both %d, T1-only %d, T2-only %d'
              % (label, a, b, both, a - both, b - both))
    print('\nwrote %s' % path)


def _sites_from_participants(path):
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    if not rows or 'site' not in rows[0]:
        raise SystemExit('%s does not look like a participants.tsv (no `site` column)' % path)
    if 'bids_t1' not in rows[0]:
        raise SystemExit('%s predates the bids_* columns and cannot say what exists here. Rebuild it '
                         'with the `participants` subcommand, passing --bids.' % path)
    site, mods = {}, {}
    for r in rows:
        site[r['ixi_id']] = r['site']
        # what can be scored is what exists HERE. The listing is only where the site came from.
        have = [m for m in ('T1', 'T2') if r['bids_%s' % m.lower()]]
        if have and r['site']:
            mods[r['ixi_id']] = {m: r['site'] for m in have}
    return mods, site


# =================================================================================================
def plan_ixi(participants, t1, t2, bids, segs, out, n_dev, seed, require_seg, force):
    if participants:
        raw, _site_of = _sites_from_participants(participants)
    else:
        raw = {}                    # id -> {'T1': site, 'T2': site}
        for folder, mod in ((t1, 'T1'), (t2, 'T2')):
            for sid, hits in _scan(folder, RE_IXI_RAW, 'IXI source').items():
                for site, _scanid, m in hits:
                    raw.setdefault(sid, {})[('T%s' % m)] = site

    bids_ids = set(_scan(bids, RE_IXI_BIDS, 'BIDS'))
    seg_mods = {sid: {('T%s' % g[0]) for g in hits}
                for sid, hits in _scan(segs, RE_IXI_SEG, 'segmentation').items()} if segs else {}

    print('== inventory ==')
    print('  T1 sources      %4d' % sum(1 for v in raw.values() if 'T1' in v))
    print('  T2 sources      %4d' % sum(1 for v in raw.values() if 'T2' in v))
    print('  subjects with at least one source %4d' % len(raw))
    print('  subjects in BIDS %4d' % len(bids_ids))
    if segs:
        print('  segmentations   %4d subjects (T1w %d, T2w %d)'
              % (len(seg_mods), sum('T1' in v for v in seg_mods.values()),
                 sum('T2' in v for v in seg_mods.values())))

    # the join, audited both ways. Nothing is dropped in silence.
    no_site = sorted(bids_ids - set(raw), key=int)
    no_bids = sorted(set(raw) - bids_ids, key=int)
    if no_site:
        print('\n[EXCLUDED] %d subjects in BIDS with no source => no site => cannot be stratified: '
              '%s' % (len(no_site), ', '.join('IXI' + s for s in no_site)))
    if no_bids:
        print('\n[WARNING] %d subjects with a source but no BIDS folder (never scored): %s'
              % (len(no_bids), ', '.join('IXI' + s for s in no_bids)))

    # one site per subject: if T1 and T2 disagree that is not something to choose
    disagree = {s: v for s, v in raw.items() if len(set(v.values())) > 1}
    if disagree:
        raise SystemExit('T1 and T2 give different sites for %d subjects; that is an inconsistency '
                         'in the data, not something to choose: %s' % (len(disagree), sorted(disagree)))

    rows, no_seg = [], []
    for sid in sorted(bids_ids & set(raw), key=int):
        mods_img = set(raw[sid])
        mods = sorted(mods_img & seg_mods.get(sid, mods_img)) if require_seg else sorted(mods_img)
        if not mods:
            no_seg.append(sid)
            continue
        site = next(iter(raw[sid].values()))
        # the stratum is the SITE and not site x modality. Modality availability is ~constant here
        # (577 of 582 have both), so stratifying on it would create cells of 1 and 4 subjects, which
        # stratify nothing and only perturb the allocation. It stays as a reported column.
        rows.append(dict(dataset='ixi', subject='sub-IXI%s' % sid, site=site,
                         modalities='+'.join(mods), n_sessions=1, stratum=site, split=''))
    if no_seg:
        print('\n[EXCLUDED] %d subjects with no usable segmentation: %s'
              % (len(no_seg), ', '.join('IXI' + s for s in no_seg)))

    # dev by strata, with the same exact allocation rule as the synthetic split
    strata = defaultdict(list)
    for r in rows:
        strata[r['stratum']].append(r['subject'])
    keys = sorted(strata)
    pools = [sorted(strata[k], key=lambda s: _key(seed, s)) for k in keys]
    dev = set()
    for i, take in enumerate(largest_remainder([len(p) for p in pools], n_dev)):
        for _ in range(take):
            dev.add(pools[i].pop())
    for r in rows:
        r['split'] = 'dev' if r['subject'] in dev else 'test'

    print('\n== allocation ==')
    print('%-10s %6s %6s %6s' % ('stratum', 'total', 'dev', 'test'))
    for k in keys:
        subs = strata[k]
        d = sum(s in dev for s in subs)
        print('%-10s %6d %6d %6d' % (k, len(subs), d, len(subs) - d))
    print('%-10s %6d %6d %6d' % ('TOTAL', len(rows), len(dev), len(rows) - len(dev)))

    mods = defaultdict(lambda: [0, 0])
    for r in rows:
        mods[r['modalities']][0 if r['split'] == 'dev' else 1] += 1
    print('\n%-10s %6s %6s   (reported, not a stratum)' % ('modalities', 'dev', 'test'))
    for k in sorted(mods):
        print('%-10s %6d %6d' % (k, mods[k][0], mods[k][1]))
    _write(out, rows, force)


# =================================================================================================
def plan_kirby(index, out, force):
    ses = _scan(index, RE_KIRBY, 'Kirby (index/fsorig)')
    rows = [dict(dataset='kirby21', subject='sub-%s' % sid, site='KKI',
                 modalities='T1', n_sessions=len(hits), stratum='KKI', split='dev')
            for sid, hits in sorted(ses.items())]

    print('== Kirby21 ==')
    print('  %d subjects, %d sessions' % (len(rows), sum(r['n_sessions'] for r in rows)))
    by_n = defaultdict(int)
    for r in rows:
        by_n[r['n_sessions']] += 1
    for n in sorted(by_n):
        print('  %d subject(s) with %d session(s)' % (by_n[n], n))
    if set(by_n) != {2}:
        print('  [WARNING] not every subject has 2 sessions: scan-rescan only comes from those that do')
    print('\n  everything goes to `dev`, on purpose: 21 subjects are not worth splitting, and the set '
          'has already been used to decide things (transfer read, divisor sweep, --clip 300).')
    print('  the per-session FreeSurfer runs are <subject>/ses-NN/; the `long-*` folder is the '
          'longitudinal stream and is NOT one of them, and aseg_cs.mgz lives in template space.')
    _write(out, rows, force)


# =================================================================================================
RE_BIDS_SUB = re.compile(r'^(sub-[A-Za-z0-9]+)$')
# BIDS names are <sub>[_<key>-<value>]*_<suffix>.nii[.gz]. Capturing the middle as ONE group of
# key-value entities instead of only ses- is what lets miriad in: its files carry run-1/run-2
# between the session and the suffix, and a regex that expects <sub>_<ses>_<suffix> matches
# nothing at all there -- it fails by finding zero files, not by erroring.
RE_BIDS_IMG = re.compile(r'^(sub-[A-Za-z0-9]+)((?:_[A-Za-z0-9]+-[A-Za-z0-9]+)*)_([A-Za-z0-9]+)\.nii(?:\.gz)?$')
RE_ENTITY_SES = re.compile(r'_(ses-[A-Za-z0-9]+)')


def _fs_sessions(fs, subject, sessions, fs_seg='aseg.mgz'):
    """Sessions of one subject that have a CROSS-SECTIONAL FreeSurfer run.

    Assembled, never globbed. `long-*` is the longitudinal stream and MEASURED on nifd, over ARAMIS'
    own 304 session pairs, it lifts GM Dice from 0.856 to 0.925: it exists to make a subject's sessions
    agree, so anchoring on it measures the regulariser instead of the method.
    """
    hit = set()
    for ses in sessions:
        d = os.path.join(fs, subject, ses or 'ses-01', 't1', 'freesurfer_cross_sectional',
                         '%s_%s' % (subject, ses or 'ses-01'), 'mri', fs_seg)
        if os.path.isfile(d):
            hit.add(ses)
    return hit


def plan_bids(dataset, bids, fs, segs, out, n_dev, seed, force, stratify=('site',)):
    subs = {}                                     # subject -> {modality: {sessions}}
    for entry in sorted(os.listdir(bids)):
        if not RE_BIDS_SUB.match(entry) or not os.path.isdir(os.path.join(bids, entry)):
            continue
        for f in glob.glob(os.path.join(bids, entry, '**', 'anat', '*.nii*'), recursive=True):
            m = RE_BIDS_IMG.match(os.path.basename(f))
            if m and m.group(1) == entry:
                ses = RE_ENTITY_SES.search(m.group(2) or '')
                subs.setdefault(entry, {}).setdefault(m.group(3), set()).add(ses.group(1) if ses else '')
    assert subs, 'no sub-*/**/anat/*.nii* under %s' % bids

    # participants.tsv is read for the WHOLE row, not just the site: --stratify names the columns
    # that go into the stratum. On a single-scanner cohort like miriad the site is constant and
    # stratifying on it does nothing at all, while the diagnosis is what decides comparability.
    meta_of = {}
    part = os.path.join(bids, 'participants.tsv')
    if os.path.isfile(part):
        for r in csv.DictReader(open(part), delimiter='\t'):
            meta_of[r.get('participant_id', '')] = {k: (v or '').strip() for k, v in r.items()}
    if meta_of:
        cols = set(next(iter(meta_of.values())))
        missing = [c for c in stratify if c not in cols]
        assert not missing, ('--stratify names %s, which participants.tsv does not have. It has %s'
                             % (missing, sorted(cols)))
    elif list(stratify) != ['site']:
        raise SystemExit('--stratify needs a participants.tsv under %s' % bids)

    print('== inventory: %s ==' % dataset)
    per_mod = defaultdict(int)
    for v in subs.values():
        for mod, ss in v.items():
            per_mod[mod] += len(ss)
    for mod in sorted(per_mod, key=lambda k: -per_mod[mod]):
        print('  %-8s %5d volumes' % (mod, per_mod[mod]))
    print('  %d subjects' % len(subs))

    rows, fs_subs = [], set()
    for sub in sorted(subs):
        mods = sorted(subs[sub])
        sessions = set().union(*subs[sub].values())
        has_fs = bool(fs) and bool(_fs_sessions(fs, sub, sessions))
        if has_fs:
            fs_subs.add(sub)
        meta = meta_of.get(sub, {})
        site = meta.get('site') or 'NA'
        # the stratum carries FS coverage, not just the site. MEASURED on nifd: 948 T1w exist but only
        # 494 sessions (172 subjects of 346) have a cross-sectional FreeSurfer run, so a split that
        # ignores it lands most of the FS-anchored subjects on one side and leaves that arm with a dev
        # or a test it cannot use.
        key = '|'.join((meta.get(c) or 'NA') for c in stratify)
        stratum = '%s|%s' % (key, 'fs' if has_fs else 'nofs') if fs else key
        rows.append(dict(dataset=dataset, subject=sub, site=site, modalities='+'.join(mods),
                         n_sessions=len(sessions), stratum=stratum, split=''))
    if fs:
        print('  %d of %d subjects have a cross-sectional FreeSurfer run' % (len(fs_subs), len(rows)))

    strata = defaultdict(list)
    for r in rows:
        strata[r['stratum']].append(r['subject'])
    keys = sorted(strata)
    pools = [sorted(strata[k], key=lambda x: _key(seed, x)) for k in keys]
    dev = set()
    for i, take in enumerate(largest_remainder([len(x) for x in pools], n_dev)):
        for _ in range(take):
            dev.add(pools[i].pop())
    for r in rows:
        r['split'] = 'dev' if r['subject'] in dev else 'test'

    print()
    print('== allocation ==')
    print('%-24s %6s %6s %6s' % ('stratum', 'total', 'dev', 'test'))
    for k in keys:
        d = sum(x in dev for x in strata[k])
        print('%-24s %6d %6d %6d' % (k, len(strata[k]), d, len(strata[k]) - d))
    print('%-24s %6d %6d %6d' % ('TOTAL', len(rows), len(dev), len(rows) - len(dev)))
    print()
    print('  the unit is the SUBJECT. %d subjects hold %d sessions, so a statistic over rows is not '
          'over independent points: cluster by subject.'
          % (len(rows), sum(r['n_sessions'] for r in rows)))
    _write(out, rows, force)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    q = sub.add_parser('participants')
    q.add_argument('--t1_list', required=True, help='`ls -1` of unorganised/IXI-T1')
    q.add_argument('--t2_list', default=None, help='`ls -1` of unorganised/IXI-T2')
    q.add_argument('--bids', default=None,
                   help='raw/ixi/bids, to record which modalities exist HERE (strongly recommended)')
    q.add_argument('--out', required=True, help='e.g. index/ixi/')

    q = sub.add_parser('ixi')
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument('--participants', help='participants.tsv built by the subcommand above')
    g.add_argument('--t1', help='unorganised/IXI-T1, when run where the sources are')
    q.add_argument('--t2', default=None, help='unorganised/IXI-T2')
    q.add_argument('--bids', required=True, help='raw/ixi/bids, the sub-IXINNN that get scored')
    q.add_argument('--segs', default=None, help='anchors/synthseg-2.0/ixi/segs')
    q.add_argument('--out', required=True)
    q.add_argument('--n_dev', type=int, default=100, help='subjects in dev; the rest is test')
    q.add_argument('--seed', type=int, default=0, help='salts the per-subject hash')
    q.add_argument('--no_require_seg', action='store_true',
                   help='do not require a segmentation for a modality to count')
    q.add_argument('--force', action='store_true')

    q = sub.add_parser('bids')
    q.add_argument('--dataset', required=True, help='the dataset column, e.g. nifd, miriad')
    q.add_argument('--bids', required=True, help='raw/<ds>/bids')
    q.add_argument('--fs', default=None,
                   help='anchors/freesurfer/<ds>/subjects. Given, FS coverage becomes part of the '
                        'stratum, so dev and test both get a usable FreeSurfer arm.')
    q.add_argument('--segs', default=None, help='anchors/synthseg-2.0/<ds>/<mod>/segs (reported only)')
    q.add_argument('--out', required=True, help='index/<ds>/splits')
    q.add_argument('--n_dev', type=int, default=100, help='subjects in dev; the rest is test')
    q.add_argument('--seed', type=int, default=0, help='salts the per-subject hash')
    q.add_argument('--stratify', default='site',
                   help='comma separated participants.tsv columns that make the stratum. Default '
                        'site. Use diagnosis on a single-scanner cohort, where the site is '
                        'constant and stratifying on it balances nothing.')
    q.add_argument('--force', action='store_true')

    q = sub.add_parser('kirby21')
    q.add_argument('--index', required=True, help='index/kirby21/fsorig')
    q.add_argument('--out', required=True)
    q.add_argument('--force', action='store_true')

    a = p.parse_args()
    if a.cmd == 'participants':
        build_participants(a.t1_list, a.t2_list, a.bids, a.out)
    elif a.cmd == 'ixi':
        plan_ixi(a.participants, a.t1, a.t2, a.bids, a.segs, a.out, a.n_dev, a.seed,
                 not a.no_require_seg, a.force)
    elif a.cmd == 'bids':
        plan_bids(a.dataset, a.bids, a.fs, a.segs, a.out, a.n_dev, a.seed, a.force,
                  tuple(x.strip() for x in a.stratify.split(',') if x.strip()))
    else:
        plan_kirby(a.index, a.out, a.force)
