"""Build the 3-tissue-grouped generation classes for the extra-cerebral (531) vocabulary.

The tissue-means head does NOT read generation_classes.npy. check_alignment
(SynthSeg/training_tissue_means.py) requires every label of a regressed tissue to sit in ONE
generation class, or the regressed mean averages several GMM draws; that grouping is
generation_classes_3tissues_grouped.npy, in which CSF is class 1. The per-label scheme puts CSF in
class 4, which is why generation_classes_extra531_csf.npy -- correct for its own source array -- makes
this head raise "regressed tissue groups are not aligned with the generation classes".

531 is CSF, so it takes the class of label 24 read out of the grouped array itself rather than
a hard-coded number, and is inserted at index 18 to line up with generation_labels_extra531.npy, which
is shared byte-for-byte by both variants.

Note what this does NOT do: 531 stays out of tissue_groups, so it is generated with the CSF intensity
distribution but is not part of the measured CSF mean. That is deliberate -- the real-data anchor
(SynthSeg / FreeSurfer aseg on IXI and Kirby) has no 531, and E2 and E3 have to regress the same
target for their difference to mean anything.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

PRIORS = 'data/labels_classes_priors/'
OUT_DIR = os.path.join(PRIORS, 'extra_cerebral_531')
OUT = os.path.join(OUT_DIR, 'generation_classes_extra531_3tissues.npy')

# the two shipped arrays this is derived from, and the one it has to line up with
labels = np.load(PRIORS + 'generation_labels.npy')
grouped = np.load(PRIORS + 'generation_classes_3tissues_grouped.npy')
labels531 = np.load(os.path.join(OUT_DIR, 'generation_labels_extra531.npy'))

csf_class = int(grouped[list(labels).index(24)])
out = np.insert(grouped, 18, csf_class).astype('int32')

assert np.array_equal(np.delete(labels531, 18), labels), 'labels531 is not labels + 531 at index 18'
assert int(labels531[18]) == 531 and len(out) == len(labels531) == 55
assert int(out[18]) == csf_class

# the check the trainer will run, run here so a bad array never reaches a 72 h job
tissue_groups = {'CSF': [4, 5, 43, 44, 14, 15, 24, 72], 'GM': [3, 42, 8, 47], 'WM': [2, 41, 7, 46]}
lab2gen = {int(l): int(c) for l, c in zip(labels531, out)}
aligned = {}
for name, group in tissue_groups.items():
    classes = sorted(set(lab2gen[l] for l in group if l in lab2gen))
    assert len(classes) == 1, '%s spans generation classes %s' % (name, classes)
    aligned[name] = classes[0]
    print('  %-3s -> generation class %d' % (name, classes[0]))

np.save(OUT, out)
md5 = hashlib.md5(open(OUT, 'rb').read()).hexdigest()

json.dump({
    'built_by': 'scripts/experiments/make_extra531_grouped_classes.py',
    'generated_utc': datetime.now(timezone.utc).isoformat(),
    'for': 'the tissue-means head (training_tissue_means.py / validate_tissue_means.py), which defaults '
           'to generation_classes_3tissues_grouped.npy',
    'not_to_be_confused_with': 'generation_classes_extra531_csf.npy, built from the per-label '
                               'generation_classes.npy (CSF is class 4 there, class 1 here). Passing that '
                               'one to this head raises "regressed tissue groups are not aligned".',
    'source_generation_classes': {
        'path': PRIORS + 'generation_classes_3tissues_grouped.npy',
        'sha256': hashlib.sha256(open(PRIORS + 'generation_classes_3tissues_grouped.npy', 'rb').read()).hexdigest(),
    },
    'pairs_with': 'generation_labels_extra531.npy (shared, byte-identical for both variants)',
    'new_label': 531, 'inserted_at_index': 18, 'block': 'neutral',
    'assigned_class': csf_class, 'labels_sharing_that_class': 'the CSF group: 4, 5, 43, 44, 14, 15, 24, 72',
    'n_labels': int(len(out)), 'n_classes': int(len(set(out.tolist()))),
    'aligned_tissue_classes': aligned,
    'MUST_PASS_ON_CLI': '--neutral_labels 19',
    'note': '531 is generated with the CSF intensity distribution but is NOT in tissue_groups, so it is '
            'not part of the measured CSF mean: the real-data anchor has no 531, and E2/E3 must regress '
            'the same target. Measured on 6 label maps, 531 is 2.4% of the CSF group volume.',
    'md5_npy': md5,
}, open(os.path.join(OUT_DIR, 'README_extra_cerebral_531_3tissues.json'), 'w'), indent=2)

print('saved %s\n  %d labels, %d classes, md5 %s' % (OUT, len(out), len(set(out.tolist())), md5))
