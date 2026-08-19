"""The fixed synthetic sets the offline validators score against.

This is not a cache. It is the dataset: every number that reaches a table is scored against it, so it
has to outlive the run that made it, and it has to be able to say what it is. Two things follow, and
they are the whole point of this module.

FINGERPRINT. Each set carries the arguments that decided its contents, and is refused if they do not
match the run asking for it. Without that, the failure is silent in the worst possible way: a file
that already exists gets reused whatever you ask for, so a set built from the old tail split scores a
model trained on the new per-patient one, and nothing says a word. `labels_dir` and `holdout` are in
the fingerprint for exactly that reason.

LAYOUT. A path without a .npz extension is a DIRECTORY:

    <set>/  images.npy   (N, ...) float32
            <target>.npy  one per target array, named by the head
            meta.json     the fingerprint, in plain text

An npz is a zip container, so it can be neither memory-mapped nor filled in place: building a
1000-image set at 160^3 and scoring it would both mean holding 16 GB of float32 at once. Separate
.npy files can be allocated on disk and filled one image at a time, and read back with mmap_mode='r',
which is how the scorers already walk them. It is also the shape validate_qc.py works in -- a
directory of things read one at a time -- rather than one blob.

meta.json is written LAST and its presence is what marks the set finished. A job killed halfway leaves
a directory without it, which load_set refuses, instead of reading an images.npy whose header is
perfectly valid and whose tail is zeros.

The .npz form still works and is read whole into ram, so sets written before this exists still load.

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib
"""

import json
import os

import numpy as np
from numpy.lib.format import open_memmap

from ext.lab2im import utils

# arguments whose value is a path. Only the last two components go into the fingerprint: it has to
# survive being typed relative on one machine and absolute on another, and the repo living somewhere
# else on the cluster, while still telling label_maps_cerebral/train from training_label_maps_cerebral
# -- which is precisely the distinction between the new split and the old one.
PATH_KEYS = ('labels_dir', 'generation_labels', 'generation_classes')


def _tail(p):
    parts = os.path.normpath(str(p)).replace('\\', '/').strip('/').split('/')
    return '/'.join(parts[-2:])


def fingerprint(args, keys, n_hold, extra=None):
    """The subset of `args` that decides what the set contains, plus how many maps it was drawn from.

    `extra` is for values that are not command-line arguments but still decide the images: the spatial
    block is hardcoded, identically, in the training script and in the validators, so it cannot diverge
    from a command line today -- but it is written down anyway. If someone edits one of those two copies
    the sets built before stop matching, which is the alarm you want and not a nuisance."""
    d = {k: (_tail(getattr(args, k)) if k in PATH_KEYS else getattr(args, k)) for k in keys}
    d['n_held_out_maps'] = int(n_hold)
    if extra:
        d.update(extra)
    return d


def migrate(path, defaults):
    """Add fingerprint keys that a set predates, and ONLY those. Never touches a key that is already
    there, so it cannot be used to make a real mismatch go away -- which is the one thing that would
    kill this mechanism. Returns what it added.

    It is for exactly one situation: a key becomes part of the fingerprint, the existing sets were built
    with its default, and writing that default down states something that was already true."""
    if _is_npz(path):
        raise SystemExit('migrate only handles the directory form; an npz would have to be rewritten whole')
    # a dataset carries its fingerprint at the top level of meta.json; a directory of scores carries it
    # nested in dataset.json under 'fingerprint', next to the path it was scored against. Both have to
    # move together or the next scoring run is refused by a stamp that predates the new keys.
    meta_path = os.path.join(path, 'meta.json')
    stamp_path = os.path.join(path, 'dataset.json')
    if os.path.isfile(meta_path):
        doc_path, key = meta_path, None
    elif os.path.isfile(stamp_path):
        doc_path, key = stamp_path, 'fingerprint'
    else:
        raise SystemExit('%s holds neither a meta.json nor a dataset.json' % path)
    with open(doc_path) as f:
        doc = json.load(f)
    meta = doc if key is None else doc.setdefault(key, {})
    added = {k: v for k, v in defaults.items() if k not in meta}
    if added:
        meta.update(added)
        with open(doc_path, 'w') as f:
            json.dump(doc, f, indent=2, sort_keys=True)
    print('%s: %s' % (path, ('added ' + ', '.join('%s=%r' % kv for kv in sorted(added.items())))
                            if added else 'nothing missing'))
    return added


def _is_npz(path):
    return path.endswith('.npz')


def open_set(path, n_images, specs):
    """Allocate the arrays. `specs` is [(name, shape_of_one, dtype), ...] with 'images' first.

    On disk and memory-mapped when the set is a directory, so the peak ram while building is one image
    rather than the whole set; in ram when it is an npz, which cannot be filled in place."""
    if path is None or _is_npz(path):
        return [np.empty((n_images,) + tuple(sh), dt) for _, sh, dt in specs]
    utils.mkdir(path)
    return [open_memmap(os.path.join(path, name + '.npy'), mode='w+', dtype=dt,
                        shape=(n_images,) + tuple(sh)) for name, sh, dt in specs]


def close_set(path, names, arrays, meta):
    """Finish the set. meta goes last, so an interrupted write cannot pass for a finished dataset."""
    if path is None:
        return
    if _is_npz(path):
        utils.mkdir(os.path.dirname(path))
        payload = dict(zip(names, arrays))
        payload['meta'] = np.array(json.dumps(meta, sort_keys=True))
        np.savez(path, **payload)
    else:
        for arr in arrays:
            if hasattr(arr, 'flush'):
                arr.flush()
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, sort_keys=True)
    print('  wrote the dataset to %s (%.2f GB, %d images)'
          % (path, arrays[0].nbytes / 1e9, len(arrays[0])))


def load_set(path, names):
    """(list of arrays, meta). The first array comes back memory-mapped in the directory form, so a set
    larger than ram can still be walked one image at a time."""
    if _is_npz(path):
        d = np.load(path)
        meta = json.loads(str(d['meta'])) if 'meta' in d else None
        return [d[n] for n in names], meta
    meta_path = os.path.join(path, 'meta.json')
    if not os.path.isfile(meta_path):
        raise SystemExit('%s has no meta.json, so it is not a finished dataset: either nothing was ever '
                         'written there, or the job writing it died before it could say so. Delete it '
                         'and build it again.' % path)
    with open(meta_path) as f:
        meta = json.load(f)
    arrays = [np.load(os.path.join(path, n + '.npy'), mmap_mode='r' if i == 0 else None)
              for i, n in enumerate(names)]
    return arrays, meta


def load_checked(path, names, want):
    """Load the set and refuse it unless its fingerprint is `want`. The message names the fields that
    differ, because 'the set does not match' is not actionable and 'bias_std file=0.0 now=0.5' is."""
    arrays, have = load_set(path, names)
    if have is None:
        raise SystemExit('%s predates the fingerprint and cannot be checked. Regenerate it, or pass it '
                         'to a run you are sure matches it.' % path)
    differ = {k: (have.get(k), want[k]) for k in want if have.get(k) != want[k]}
    if differ:
        raise SystemExit(
            'refusing to score against %s: it was built with different settings, so its images and its '
            'targets are not the ones this run is asking for.\n%s'
            % (path, '\n'.join('  %-20s file=%r  now=%r' % (k, v[0], v[1])
                               for k, v in sorted(differ.items()))))
    print('  using the dataset at %s (%d images, fingerprint matches)' % (path, len(arrays[0])))
    return arrays


def stamp_validation_dir(validation_dir, dataset_path, meta):
    """Record which dataset a directory of scores was produced against, and refuse to add to it under a
    different one.

    This is needed because the scoring loop SKIPS checkpoints it has already scored. Without the stamp,
    rebuilding a dataset and re-scoring into the same directory leaves every old number in place and the
    curve silently mixes two of them -- the same failure the dataset fingerprint exists to stop, one
    level up.

    Both the path and the fingerprint go in. The path is for whoever reads it later, so they can go and
    look at the set's own meta.json. The fingerprint is what the check runs on, because a path can be
    reused: delete a set and regenerate it at the same location with different settings and the path
    still matches while the contents do not.
    """
    utils.mkdir(validation_dir)
    path = os.path.join(validation_dir, 'dataset.json')
    if os.path.isfile(path):
        with open(path) as f:
            old = json.load(f)
        differ = {k: (old.get('fingerprint', {}).get(k), v)
                  for k, v in meta.items() if old.get('fingerprint', {}).get(k) != v}
        if differ:
            raise SystemExit(
                'refusing to write scores into %s: it already holds scores taken against a different '
                'dataset (%s), and the checkpoints already scored would not be recomputed, so the curve '
                'would mix the two.\n%s\nUse a different --validation_dir, or delete this one.'
                % (validation_dir, old.get('dataset'),
                   '\n'.join('  %-20s there=%r  now=%r' % (k, v[0], v[1])
                             for k, v in sorted(differ.items()))))
        return
    with open(path, 'w') as f:
        json.dump({'dataset': dataset_path, 'fingerprint': meta}, f, indent=2, sort_keys=True)
