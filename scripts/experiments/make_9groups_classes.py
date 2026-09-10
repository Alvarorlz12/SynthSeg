"""Build the 9-group generation classes for the tissue-means head (extra-cerebral 531 vocabulary).

Two separate groupings live side by side and must not be confused:

  * GENERATION classes -- which labels share one Gaussian when the image is synthesised. Two labels
    in the same class come out at the same intensity in every sample.
  * REGRESSED groups -- which labels are averaged to produce the numbers the head predicts.

Every regressed group must sit inside one generation class. 

What changes with respect to generation_classes_extra531_3tissues.npy, which regressed three numbers:

  * CSF splits into three classes and only the ventricular one is regressed. Label 24 leaves the
    target.
  * "GM" and "WM" stop being one Gaussian each. Cerebral and cerebellar are now separated on the generation side
    and on the target side together.
  * Hippocampus and amygdala are merged into one class. They are regressed together.
  * Caudate and accumbens are generated but not regressed.

Left/right stay merged, as in the 3-tissue array this replaces. Stock SynthSeg never merges left and
right; that deviation is inherited on purpose so this run changes one thing only, the regrouping.

Run from the SynthQC directory:

    python scripts/experiments/make_9groups_classes.py
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

PRIORS = 'data/labels_classes_priors/'
OUT_DIR = os.path.join(PRIORS, 'extra_cerebral_531')
OUT = os.path.join(OUT_DIR, 'generation_classes_9groups.npy')

# One entry per generation class, in order: a class index is this list's position.
GENERATION_CLASSES = [
    ('background',          [0]),
    # --- CSF: three Gaussians, only the first is regressed
    ('csf_ventricular',     [4, 5, 43, 44, 14, 15]),
    ('csf_extracerebral',   [24, 531]),
    ('csf_fifth_ventricle', [72]),
    # --- GM: eight Gaussians, the first six are regressed
    ('gm_cortex',           [3, 42]),
    ('gm_cerebellum',       [8, 47]),
    ('thalamus',            [10, 49]),
    ('putamen',             [12, 51]),
    ('pallidum',            [13, 52]),
    ('hippocampus_amygdala', [17, 53, 18, 54]),
    ('accumbens',           [26, 58]),
    ('caudate',             [11, 50]),
    # --- WM: five Gaussians, the first two are regressed
    ('wm_cerebral',         [2, 41]),
    ('wm_cerebellum',       [7, 46]),
    ('brainstem',           [16]),
    ('ventral_dc',          [28, 60]),
    ('optic_chiasm',        [85]),
    # --- untouched, exactly as in the 3-tissue array
    ('choroid_plexus',      [136, 137, 163, 164]),
    ('lesion',              [25, 57]),
    ('vessel',              [30, 62]),
] + [('extracerebral_%d' % lab, [lab]) for lab in
     (502, 506, 507, 508, 509, 511, 512, 514, 515, 516, 530)]

# The nine numbers the head predicts, in output order. Each name must be a class above.
REGRESSED = ['csf_ventricular',
             'gm_cortex', 'gm_cerebellum', 'thalamus', 'putamen', 'pallidum', 'hippocampus_amygdala',
             'wm_cerebral', 'wm_cerebellum']

labels = np.load(os.path.join(OUT_DIR, 'generation_labels_extra531.npy'))
by_name = dict(GENERATION_CLASSES)

# build the array, aligned position by position with generation_labels_extra531.npy
label_to_class = {}
for index, (name, group) in enumerate(GENERATION_CLASSES):
    for lab in group:
        assert lab not in label_to_class, 'label %d is in two classes' % lab
        label_to_class[lab] = index

missing = [int(l) for l in labels if int(l) not in label_to_class]
assert not missing, 'labels with no class: %s' % missing
unknown = [l for l in label_to_class if l not in set(int(x) for x in labels)]
assert not unknown, 'classes hold labels outside the vocabulary: %s' % unknown

out = np.array([label_to_class[int(l)] for l in labels], dtype='int32')
assert len(out) == len(labels) == 55
assert int(labels[18]) == 531, 'index 18 must be 531 -- pass --neutral_labels 19'

# brain_generator.py asserts the classes are contiguous 0..K-1
unique = np.unique(out)
assert np.array_equal(unique, np.arange(unique.max() + 1)), 'class indices are not contiguous'

# the check the trainer will run
tissue_groups = {name: sorted(by_name[name]) for name in REGRESSED}
label_to_gen = {int(l): int(c) for l, c in zip(labels, out)}
aligned = {}
for name, group in tissue_groups.items():
    classes = sorted(set(label_to_gen[l] for l in group if l in label_to_gen))
    assert len(classes) == 1, '%s spans generation classes %s' % (name, classes)
    aligned[name] = classes[0]
    print('  %-21s -> generation class %2d  %s' % (name, classes[0], group))

np.save(OUT, out)
md5 = hashlib.md5(open(OUT, 'rb').read()).hexdigest()

json.dump({
    'built_by': 'scripts/experiments/make_9groups_classes.py',
    'generated_utc': datetime.now(timezone.utc).isoformat(),
    'for': 'the tissue-means head (QC/training_tm.py) after the ungrouping agreed on 2026-09-09',
    'replaces': 'generation_classes_extra531_3tissues.npy, which regressed three numbers (CSF, GM, WM)',
    'pairs_with': 'generation_labels_extra531.npy (unchanged, 55 labels, 19 neutral)',
    'MUST_PASS_ON_CLI': '--neutral_labels 19',
    'n_labels': int(len(out)),
    'n_classes': int(len(unique)),
    'n_regressed': len(REGRESSED),
    'generation_classes': {name: sorted(group) for name, group in GENERATION_CLASSES},
    'regressed_groups': tissue_groups,
    'regressed_generation_class': aligned,
    'generated_but_not_regressed': [name for name, _ in GENERATION_CLASSES
                                    if name not in tissue_groups and name != 'background'],
    'md5_npy': md5,
}, open(os.path.join(OUT_DIR, 'README_9groups.json'), 'w'), indent=2)

print('\nsaved %s\n  %d labels, %d classes, %d regressed, md5 %s'
      % (OUT, len(out), len(unique), len(REGRESSED), md5))
print('\npaste into QC/training_tm.py:')
print('tissue_groups = {')
for name in REGRESSED:
    print("    %-24s %s," % ("'%s':" % name, tissue_groups[name]))
print('}')
