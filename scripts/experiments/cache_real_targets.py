"""Compute the per-tissue ground truth of the real datasets ONCE and keep it in a table.

The anchor does not depend on the checkpoint, so every number in `targets.csv` survives every model
that will ever be scored against it. What costs time is what produces it -- loading two volumes,
checking they share a grid, centring a 160^3 crop on the brain, normalising, averaging under the
label groups -- and that is exactly the part that was being repeated for each checkpoint.

Nothing here is a second definition of the target. Every computational step is imported from
`scratchpad/score_real_tissue_means.py`: `load_pair`, `crop_pad`, `normalise`, `tissue_means`,
`csf_composition`, `contrast`, and the `GROUPS` themselves. What this file adds is the loop, the
table and the fingerprint. A copy of the arithmetic would be a copy that can drift, and it would
drift towards being wrong in the only place the difference is invisible: the ground truth.

Fingerprint. What decides the contents is written to `targets.meta.json`, and a run whose settings
differ from the one on disk is refused instead of appended to. This mirrors the guarantee in
`SynthSeg/synth_dataset.py`, but the fields are written here rather than through `ds.fingerprint`,
whose field list is read off the synthetic generator's own namespace.

The divisor, and one thing worth being exact about. The per-tissue means are read on the normalised
image, so they depend on it and it is part of the key. The DELIVERABLE is a ratio, and it is invariant
to the divisor ONLY WHEN THE FLOOR IS ZERO:

    mu_t = (x_t - m) / (M - m)   =>   |GM-WM| / (GM+WM) = |x_GM - x_WM| / (x_GM + x_WM - 2m)

so M cancels but m does not. On the synthetic side `intensity_clip=[0, 1e10]` pins m at exactly 0 and
the invariance is real. Nothing pins it on a real scan, and a `pLO-pHI` divisor raises the floor on
purpose. So `deliverable` is stored per divisor, `img_min` is stored per case, and how far the
deliverable moves across divisors is printed at the end: it is a direct readout of how far the floor
is from zero, which is a train/deploy mismatch in the target itself and not an error of the model.

Already-normalised inputs. min-max is invariant to any monotone AFFINE rescaling of the input --
(ax+b-m)/(M-m) does not depend on a or b -- so a volume that someone else already rescaled, to [0,1]
or to anything else, gives the same target as the raw one. What it is NOT invariant to is a
non-affine change: gamma, N4 bias correction, histogram matching. Those cannot be undone here and
would show up as a subject whose target sits away from its cohort. `img_min` and `img_max` are stored
raw, before normalising, so the input range is on record: a FreeSurfer conformed orig.mgz sits at
0-255, a raw scanner T1 at a few hundred to a few thousand, and an img_max of exactly 1.0 means the
volume reached us already normalised.

No GPU and no TensorFlow: this is nibabel and numpy. It is restartable -- rows already in the table
are skipped -- so it can be run in slices on a short queue.

Usage
-----
  IXI     python scripts/experiments/cache_real_targets.py ixi \
              --manifest <.../index/ixi/splits/manifest.csv> \
              --participants <.../index/ixi/participants.tsv> \
              --bids <.../raw/ixi/bids> --segs <.../anchors/synthseg-2.0/ixi/segs> \
              [--resampled <.../anchors/synthseg-2.0/ixi/resampled>] \
              --out <.../index/ixi> [--divisors max,p99.9]

  Kirby   python scripts/experiments/cache_real_targets.py kirby21 \
              --manifest <.../index/kirby21/splits/manifest.csv> \
              --index <.../index/kirby21/fsorig> \
              --fs <.../anchors/freesurfer/kirby21/subjects> \
              --out <.../index/kirby21> [--divisors max,p99.9]

Kirby: the per-session runs are <subject>/ses-NN/mri/. `long-*` is the longitudinal stream, which
is built from both sessions and so destroys the scan-rescan floor that is the reason to use this
dataset at all; and `aseg_cs.mgz` is that result resampled into template space, 5-28 mm off. Neither
is reachable from here, because the pairing goes through the curated index rather than a glob.

CSF is not comparable across anchors. SynthSeg's CSF group is 87-96% label 24, extra-cerebral
fluid; FreeSurfer's aseg barely labels 24, so a CSF read off an aseg is dominated by ventricles and
is a different quantity. `csf24_frac` is stored per row so that can never be read by accident. GM and
WM carry over unchanged, and so does the deliverable.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np

SYNTHQC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (SYNTHQC, os.path.join(SYNTHQC, 'scratchpad')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from score_real_tissue_means import (                                        # noqa: E402
    GROUPS, OUT, TISSUES, contrast, crop_pad, csf_composition, load_pair, normalise, tissue_means)

FIELDS = ('dataset', 'subject', 'session', 'modality', 'split', 'site', 'anchor', 'divisor',
          'mu_CSF', 'mu_GM', 'mu_WM', 'n_CSF', 'n_GM', 'n_WM', 'deliverable',
          'img_min', 'img_max', 'csf24_frac', 'zooms', 'truncated', 'image', 'seg')
KEY = ('subject', 'session', 'modality', 'anchor', 'divisor')


def _meta(anchor, divisors, image_stream, seg_name):
    return dict(anchor=anchor, divisors=list(divisors), image_stream=image_stream,
                seg_name=seg_name, crop=OUT, tissues=list(TISSUES),
                groups={k: sorted(v) for k, v in GROUPS.items()})


def _check_meta(out, meta):
    """Refuse to add rows computed under different settings to a table that already exists. Without
    this the table would silently mix two ground truths, and the mix is invisible downstream."""
    path = os.path.join(out, 'targets.meta.json')
    if not os.path.isfile(path):
        return
    with open(path) as f:
        old = json.load(f)
    differ = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
    if differ:
        raise SystemExit(
            'refusing to write into %s: it already holds targets computed under different settings, '
            'and mixing two ground truths in one table cannot be seen downstream.\n%s\nDelete the '
            'table and its meta, or write somewhere else.'
            % (out, '\n'.join('  %-14s on disk=%r  now=%r' % (k, a, b)
                              for k, (a, b) in sorted(differ.items()))))


def _existing(path):
    if not os.path.isfile(path):
        return set(), 0
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {tuple(r[k] for k in KEY) for r in rows}, len(rows)


def _truncation(brain):
    """A brain pole tapers off; a flat end face means SynthSeg ran with too small a --crop and this
    subject's ground truth is taken over a truncated mask. Measured on IXI002 with --crop 160: total
    volume stayed inside the normal adult range, so a volume check does NOT catch this."""
    bad = []
    for ax in range(3):
        area = brain.sum(axis=tuple(k for k in range(3) if k != ax))
        nz = np.flatnonzero(area)
        if len(nz) and area.max() > 0:
            for end in (nz[0], nz[-1]):
                if area[end] > 0.2 * area.max():
                    bad.append('ax%d@%d=%.0f%%' % (ax, end, 100. * area[end] / area.max()))
    return bad


def run(cases, anchor, divisors, out, image_stream, seg_name, force):
    """`cases` is a list of dicts with subject/session/modality/split/site/image/seg."""
    meta = _meta(anchor, divisors, image_stream, seg_name)
    if force and os.path.isdir(out):
        for f in ('targets.csv', 'targets.meta.json'):
            if os.path.isfile(os.path.join(out, f)):
                os.remove(os.path.join(out, f))
    _check_meta(out, meta)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'targets.csv')
    done, n_before = _existing(path)
    if done:
        print('%d rows already in %s, they are skipped' % (n_before, path))

    new, skipped, unusable, spread = [], 0, [], []
    t_start = time.time()
    for i, c in enumerate(cases):
        keys = [tuple([c['subject'], c['session'], c['modality'], anchor, d]) for d in divisors]
        if all(k in done for k in keys):
            skipped += 1
            continue
        t0 = time.time()
        try:
            vol, seg, zooms = load_pair(c['image'], c['seg'])
        except AssertionError as err:
            print('  [SKIP] %s %s\n         %s' % (c['subject'], c['modality'],
                                                   str(err).replace('\n', '\n         ')))
            unusable.append('%s/%s' % (c['subject'], c['modality']))
            continue
        brain = seg > 0
        if brain.sum() == 0:
            print('  [SKIP] %s %s: empty segmentation' % (c['subject'], c['modality']))
            unusable.append('%s/%s' % (c['subject'], c['modality']))
            continue
        trunc = _truncation(brain)
        if trunc:
            print('  [WARNING] %s %s: segmentation ENDS ABRUPTLY (%s). Re-segment with a larger --crop; '
                  'this subject\'s ground truth is truncated.'
                  % (c['subject'], c['modality'], '; '.join(trunc)))
        centre = np.array(np.where(brain)).mean(1).round().astype(int)
        vol_c = crop_pad(vol, centre, OUT)
        seg_c = crop_pad(seg, centre, OUT, dtype=np.int32)
        csf24 = csf_composition(seg_c, zooms)

        line = '  %-16s %-3s %-4s' % (c['subject'], c['session'], c['modality'])
        deliv = {}
        for d in divisors:
            mu, cnt = tissue_means(normalise(vol_c, d), seg_c)
            deliv[d] = contrast(mu)
            row = dict(c, anchor=anchor, divisor=d, deliverable='%.6f' % deliv[d],
                       img_min='%.4f' % float(vol_c.min()),
                       img_max='%.4f' % float(vol_c.max()),
                       csf24_frac='%.4f' % float(np.asarray(csf24).ravel()[0]),
                       zooms='|'.join('%.3f' % z for z in zooms),
                       truncated=';'.join(trunc))
            for j, t in enumerate(TISSUES):
                row['mu_' + t] = '' if np.isnan(mu[j]) else '%.6f' % mu[j]
                row['n_' + t] = int(cnt[j])
            new.append({k: row.get(k, '') for k in FIELDS})
            line += '  | %s %s' % (d, np.round(mu, 3))
        spread.append((abs(max(deliv.values()) - min(deliv.values())), c['subject'], c['modality'],
                       float(vol_c.min())))
        print(line + '  (%.1fs)' % (time.time() - t0))

    write_header = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        # the csv module's excel dialect writes CRLF; this table gets read by awk and friends too
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
        if write_header:
            w.writeheader()
        for r in new:
            w.writerow(r)
    with open(os.path.join(out, 'targets.meta.json'), 'w') as f:
        json.dump(meta, f, indent=1, sort_keys=True)

    print('\n%d new rows (%d cases skipped as already done) in %.0f s -> %s'
          % (len(new), skipped, time.time() - t_start, path))
    if unusable:
        print('[%d cases UNUSABLE and left out]: %s' % (len(unusable), ', '.join(unusable)))

    # the invariance check, as a measurement rather than an assumption. The deliverable only ignores
    # the divisor when the floor is zero, and on a real scan nothing makes it zero.
    if spread and len(divisors) > 1:
        spread.sort(reverse=True)
        worst = spread[0]
        med = float(np.median([x[0] for x in spread]))
        print('\ndeliverable across %d divisors: median |delta| %.5f, worst %.5f '
              '(%s %s, img_min %.2f)'
              % (len(divisors), med, worst[0], worst[1], worst[2], worst[3]))
        if worst[0] > 0.01:
            print('  [WARNING] the deliverable is NOT stable across divisors here. It is invariant only '
                  'when the floor is 0; these images have a floor far from it, so part of what a '
                  'divisor sweep shows is the anchor moving and not the model.')
    # a table that is short because half the cases failed must not read like a complete one
    print('table now holds %d rows; %d cases x %d divisors were asked for'
          % (n_before + len(new), len(cases), len(divisors)))


def _filter(cases, split):
    if not split:
        return cases
    want = {x.strip() for x in split.split(',') if x.strip()}
    kept = [c for c in cases if c['split'] in want]
    print('split filter %s: %d of %d cases kept' % (sorted(want), len(kept), len(cases)))
    assert kept, 'no case has split in %s; the manifest holds %s' % (
        sorted(want), sorted({c['split'] for c in cases}))
    return kept


def _manifest(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def cases_ixi(manifest, participants, bids, segs, resampled):
    part = {r['participant_id']: r for r in
            csv.DictReader(open(participants), delimiter='\t')}
    out, missing, used = [], [], {'resampled': 0, 'bids': 0}
    for m in _manifest(manifest):
        p = part.get(m['subject'])
        if p is None:
            missing.append('%s (not in participants.tsv)' % m['subject'])
            continue
        for mod in m['modalities'].split('+'):
            rel = p['bids_%s' % mod.lower()]
            if not rel:
                missing.append('%s %s (manifest says it exists, participants.tsv has no path)'
                               % (m['subject'], mod))
                continue
            seg = os.path.join(segs, '%s_%sw_synthseg.nii.gz' % (m['subject'], mod))
            # If SynthSeg resampled, the segmentation lives on ITS grid and the native BIDS image does
            # not: pairing those two aborts in load_pair's affine check on every single case. Globbed
            # rather than assembled, so the extension SynthSeg chose does not have to be guessed.
            img, src = '', 'bids'
            if resampled:
                hits = sorted(h for h in glob.glob(os.path.join(
                    resampled, '%s_%sw_resampled.*' % (m['subject'], mod)))
                    if h.endswith(('.nii', '.nii.gz', '.mgz')))
                if hits:
                    img, src = hits[0], 'resampled'
            # SynthSeg writes no resampled file when the input was already at 1 mm; then the BIDS image
            # is the correctly paired one, and load_pair's grid check is what confirms it.
            if not img:
                img = os.path.join(bids, rel)
            used[src] += 1
            for f in (img, seg):
                if not os.path.isfile(f):
                    missing.append('%s %s (%s)' % (m['subject'], mod, f))
                    break
            else:
                out.append(dict(dataset='ixi', subject=m['subject'], session='', modality=mod,
                                split=m['split'], site=m['site'], image=img, seg=seg))
    # a --resampled that matched nothing would fall back to BIDS on every case and then abort in
    # the grid check, which reads like a data problem instead of a path problem.
    print('image stream: %d resampled, %d native BIDS' % (used['resampled'], used['bids']))
    if resampled and used['resampled'] == 0:
        print('  [WARNING] --resampled was given but matched nothing. Expected '
              '<subject>_<T1|T2>w_resampled.<nii|nii.gz|mgz> in %s' % resampled)
    if missing:
        print('[%d cases could not be paired and are NOT in the table]:' % len(missing))
        for x in missing[:15]:
            print('   %s' % x)
        if len(missing) > 15:
            print('   ... and %d more' % (len(missing) - 15))
    return out


RE_IDX = re.compile(r'^(sub-KKI\d+)_ses-(\d+)_T1w\.(?:mgz|nii|nii\.gz)$')


def cases_kirby(manifest, index, fs, seg_name):
    keep = {r['subject']: r for r in _manifest(manifest)}
    out, missing = [], []
    for name in sorted(os.listdir(index)):
        m = RE_IDX.match(name)
        if not m or m.group(1) not in keep:
            continue
        sub, ses = m.group(1), 'ses-%s' % m.group(2)
        # the pairing goes subject/session -> <fs>/<subject>/<session>/mri/, never a glob: a glob over
        # this root also finds long-* and the Nipype working copies, and pairing a session with its own
        # duplicate would make the test-retest agreement come out perfect.
        seg = os.path.join(fs, sub, ses, 'mri', seg_name)
        img = os.path.join(index, name)
        if not os.path.isfile(seg):
            missing.append('%s %s (%s)' % (sub, ses, seg))
            continue
        out.append(dict(dataset='kirby21', subject=sub, session=ses, modality='T1',
                        split=keep[sub]['split'], site=keep[sub]['site'], image=img, seg=seg))
    if missing:
        print('[%d sessions with no segmentation, NOT in the table]: %s'
              % (len(missing), ', '.join(missing[:10])))
    return out


RE_SES = re.compile(r'(sub-[A-Za-z0-9]+)(?:_(ses-[A-Za-z0-9]+))?_([A-Za-z0-9]+)\.nii(?:\.gz)?$')

# the cross-sectional run of a Clinica CAPS tree. `long-*` is the longitudinal stream, rebuilt from a
# per-subject template, and MEASURED on nifd (ARAMIS' own QC tables, 304 pairs): it lifts GM Dice from
# 0.856 to 0.925 between the same two sessions. Anchoring on it measures that regulariser and not the
# scanner, so the path is assembled and never globbed.
FS_CS = os.path.join('%s', '%s', '%s', 't1', 'freesurfer_cross_sectional', '%s_%s', 'mri')
# recon-all consumed the T1 and nothing else, so no other modality can have an aseg. Asking for one
# would report every FLAIR as a missing case and bury the pairing failures that are real.
FS_MODALITIES = ('T1w', 'T1')


def _bids_files(bids, subject, modality):
    """Every <subject>[_<session>]_<modality>.nii[.gz] under a BIDS tree, as (session, path)."""
    pat = os.path.join(bids, subject, '**', 'anat', '%s*_%s.nii*' % (subject, modality))
    out = []
    for f in sorted(glob.glob(pat, recursive=True)):
        m = RE_SES.search(os.path.basename(f))
        if m and m.group(1) == subject and m.group(3) == modality:
            out.append((m.group(2) or '', f))
    return out


def cases_bids(manifest, dataset, bids, segs, resampled, fs, fs_seg, anchor, modalities=None):
    """One BIDS tree, one row per (subject, session, modality). Works for any dataset whose images
    live in BIDS -- which after 2026-08-24 is all of them, because FreeSurfer is an ANCHOR and not a
    source of images: a recon-all tree only holds what recon-all consumed, so a T2w or a FLAIR is not
    in it and cannot be.

    Two anchors, and they are not interchangeable:
      synthseg     the BIDS image (or SynthSeg's resampled one) under SynthSeg's segmentation.
      freesurfer   orig.mgz under aseg.mgz. BOTH conformed, so this pair is self-consistent -- but it
                   is NOT the BIDS image: aseg lives on the 256^3 1 mm conformed grid and a native
                   oblique BIDS volume does not, so pairing those two aborts in load_pair.
    """
    out, missing, used = [], [], {'resampled': 0, 'bids': 0}
    for m in _manifest(manifest):
        sub = m['subject']
        for mod in m['modalities'].split('+'):
            # the manifest lists every modality the BIDS holds; an anchor run covers one at a
            # time. Without this, scoring T1w against a tree where only T1w was segmented reports
            # every FLAIR as a missing case and buries the pairing failures that are real.
            if modalities and mod not in modalities:
                continue
            hits = _bids_files(bids, sub, mod)
            if not hits:
                missing.append('%s %s (nothing under %s)' % (sub, mod, os.path.join(bids, sub)))
                continue
            for ses, bids_img in hits:
                stem = os.path.basename(bids_img).split('.nii')[0]
                if anchor == 'freesurfer':
                    if not fs:
                        raise SystemExit('--fs is required with --anchor freesurfer')
                    if mod not in FS_MODALITIES:
                        continue
                    s_ = ses or 'ses-01'
                    mri = FS_CS % (fs, sub, s_, sub, s_)
                    img = os.path.join(mri, 'orig.mgz')
                    seg = os.path.join(mri, fs_seg)
                else:
                    seg = os.path.join(segs, stem + '_synthseg.nii.gz')
                    # SynthSeg writes a resampled file ONLY when it resampled. When it did, its
                    # segmentation is on THAT grid and the native BIDS volume is not; and our own
                    # regressor reads a 160^3 crop assuming 1 mm, so the resampled image is also the
                    # only one it sees at the scale it was trained on. Globbed, not assembled, so the
                    # extension SynthSeg chose does not have to be guessed.
                    img, src = '', 'bids'
                    if resampled:
                        cand = sorted(h for h in glob.glob(os.path.join(resampled, stem + '_resampled.*'))
                                      if h.endswith(('.nii', '.nii.gz', '.mgz')))
                        if cand:
                            img, src = cand[0], 'resampled'
                    if not img:
                        img = bids_img
                    used[src] += 1
                for f in (img, seg):
                    if not os.path.isfile(f):
                        missing.append('%s %s %s (%s)' % (sub, ses, mod, f))
                        break
                else:
                    out.append(dict(dataset=dataset, subject=sub, session=ses, modality=mod,
                                    split=m['split'], site=m['site'], image=img, seg=seg))
    if anchor != 'freesurfer':
        print('image stream: %d resampled, %d native BIDS' % (used['resampled'], used['bids']))
        if resampled and used['resampled'] == 0:
            print('  [WARNING] --resampled was given but matched nothing. On a dataset already at '
                  '1 mm that is correct and expected; otherwise it is a path problem.')
    if missing:
        print('[%d cases could not be paired and are NOT in the table]:' % len(missing))
        for x in missing[:15]:
            print('   %s' % x)
        if len(missing) > 15:
            print('   ... and %d more' % (len(missing) - 15))
    # sessions are not independent points. Say the two numbers so no one bootstraps over the rows.
    subs = {c['subject'] for c in out}
    print('%d rows from %d subjects (%.1f per subject): cluster by SUBJECT, not by row'
          % (len(out), len(subs), len(out) / max(len(subs), 1)))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dataset', choices=['ixi', 'kirby21', 'bids'],
                   help="`bids` is the generic route and the one to use for anything new: it "
                        "needs --bids and (--segs | --fs), and takes --name for the dataset "
                        "column. `ixi` and `kirby21` are kept as they were so the tables "
                        "already scored under them stay reproducible.")
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--divisors', default='max,p99.9',
                   help='comma separated. One pass per divisor; the deliverable is stored for each so '
                        'a divisor that deforms the anchor shows up instead of hiding.')
    p.add_argument('--split', default=None,
                   help='only these splits, comma separated (e.g. dev). Computing a target uses no '
                        'model, so doing it for test spends nothing; the filter is for choosing the '
                        'divisor on dev first and writing the real table afterwards.')
    p.add_argument('--force', action='store_true', help='delete the table and start it again')
    p.add_argument('--participants', help='ixi: index/ixi/participants.tsv')
    p.add_argument('--bids', help='ixi: raw/ixi/bids')
    p.add_argument('--segs', help='ixi: anchors/synthseg-2.0/ixi/segs')
    p.add_argument('--resampled', default=None, help="ixi: SynthSeg's --resample folder, if there is one")
    p.add_argument('--index', help='kirby21: index/kirby21/fsorig')
    p.add_argument('--fs', help='kirby21: anchors/freesurfer/kirby21/subjects')
    p.add_argument('--name', default=None,
                   help='bids: the dataset column, e.g. nifd or miriad. Defaults to the '
                        'basename of --bids two levels up.')
    p.add_argument('--modalities', default=None,
                   help='bids: comma separated, e.g. T1w. Restricts the run to the modalities '
                        'that were actually segmented. Leave unset to take every one the '
                        'manifest lists.')
    p.add_argument('--anchor', choices=['synthseg', 'freesurfer'], default='synthseg',
                   help='bids: which anchor to read. One pass each writes two sets of rows into '
                        'the same table, told apart by the `anchor` column, so the two are '
                        'comparable per subject. FreeSurfer reads orig.mgz under aseg.mgz -- '
                        'both conformed -- and NOT the BIDS image, which is on another grid. '
                        'Compare the two on the SCALARS, never with Dice.')
    p.add_argument('--fs_seg', default='aseg.mgz',
                   help='kirby21: keep aseg.mgz. aseg_cs.mgz reads like it but lives in template space')
    a = p.parse_args()
    divisors = [d.strip() for d in a.divisors.split(',') if d.strip()]

    if a.dataset == 'bids':
        assert a.bids, '--bids is required for the generic route'
        name = a.name or os.path.basename(os.path.dirname(os.path.abspath(a.bids)))
        mods = {x.strip() for x in a.modalities.split(',') if x.strip()} if a.modalities else None
        cases = _filter(cases_bids(a.manifest, name, a.bids, a.segs, a.resampled,
                                   a.fs, a.fs_seg, a.anchor, mods), a.split)
        if a.anchor == 'freesurfer':
            # `freesurfer-cs` and not `freesurfer`: the stream is part of what the number means, and
            # `anchor` is in the row key, so a cross-sectional and a longitudinal read could never be
            # confused for one another if the second is ever added.
            run(cases, 'freesurfer-cs', divisors, a.out, 'fs-orig.mgz', a.fs_seg, a.force)
        else:
            run(cases, 'synthseg-2.0', divisors, a.out,
                'synthseg-resampled' if a.resampled else 'bids-native', '*_synthseg.nii.gz', a.force)
    elif a.dataset == 'ixi':
        for need in ('participants', 'bids', 'segs'):
            assert getattr(a, need), '--%s is required for ixi' % need
        cases = _filter(cases_ixi(a.manifest, a.participants, a.bids, a.segs, a.resampled), a.split)
        # which image was read is part of what decides the target: the resampled volume has been
        # interpolated once more than the native one, so the two are not the same ground truth.
        stream = 'synthseg-resampled' if a.resampled else 'bids-T1w/T2w'
        run(cases, 'synthseg-2.0', divisors, a.out, stream, '*_synthseg.nii.gz', a.force)
    else:
        for need in ('index', 'fs'):
            assert getattr(a, need), '--%s is required for kirby21' % need
        cases = _filter(cases_kirby(a.manifest, a.index, a.fs, a.fs_seg), a.split)
        run(cases, 'freesurfer', divisors, a.out, 'fsorig-orig.mgz', a.fs_seg, a.force)
