"""
Thin launcher for the bias-field severity regressor, mirroring scripts/experiments/training_tissue_means.py.

    python scripts/experiments/training_biasfield_scalar.py <labels_dir> <model_dir> [options]

regresses one scalar per image: the bias severity that survives into the image, std_log, in its own units.
--std_log_max is the plausibility threshold for reading a prediction, not a divisor, so nothing is clipped
and two runs that disagree on it stay comparable. the bias field is the target and the only intensity
corruption on by default;
--bias_prob leaves a fraction of images clean (target 0). deformation is on by default (the dice qc net's
values); pass --no_deform for a local cpu smoke. --norm defaults to instance (train == validation ==
deployment). paths to the label/class arrays resolve from the repo root when relative.

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors
Licensed under the Apache License, Version 2.0; see http://www.apache.org/licenses/LICENSE-2.0
"""


import os
import sys

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from argparse import ArgumentParser
from experiments.training_biasfield_scalar import training

parser = ArgumentParser()

# positional
parser.add_argument('labels_dir', type=str)
parser.add_argument('model_dir', type=str)

# generation. the standard classes (not the 3-tissue grouping): the bias target does not depend on the
# tissue grouping, so richer per-structure contrast is the better domain randomisation here.
parser.add_argument('--generation_labels', type=str, dest='generation_labels',
                    default='data/labels_classes_priors/generation_labels.npy')
parser.add_argument('--generation_classes', type=str, dest='generation_classes',
                    default='data/labels_classes_priors/generation_classes.npy')
parser.add_argument('--holdout', type=int, dest='holdout', default=100)
parser.add_argument('--neutral_labels', type=int, dest='n_neutral_labels', default=18)
parser.add_argument('--output_shape', type=int, dest='output_shape', default=160)

# bias field: the target and its regime
parser.add_argument('--std_log_max', type=float, dest='std_log_max', default=0.65)
parser.add_argument('--bias_std', type=float, dest='bias_field_std', default=1.0)
parser.add_argument('--bias_prob', type=float, dest='bias_prob', default=0.9)
parser.add_argument('--bias_scale', type=float, dest='bias_scale', default=.025)

# other deformation and intensity
parser.add_argument('--no_deform', action='store_true', dest='no_deform')
parser.add_argument('--randomise_res', action='store_true', dest='randomise_res')
parser.add_argument('--gamma_std', type=float, dest='gamma_std', default=0.)
parser.add_argument('--clip', type=int, dest='clip', default=0)
# --qc_head uses the dice qc net's head verbatim (k channels in both head convs, relu on both) instead of the
# tissue-means one (16 channels, then a LINEAR conv). That linear conv was argued for the tissue-means target,
# which sits near 0.5 and never reaches 0; std(B) is >= 0 and is exactly 0 on the ~10% of images bias_prob
# leaves clean. Turn it on with --norm batch: the head deviation is what the July run confounded batch norm
# with. The layer names differ, so a checkpoint from the other head is refused rather than half-loaded.
parser.add_argument('--qc_head', action='store_true', dest='qc_head')

# architecture (defaults are the dice qc net's; n_levels sets the downsampling, output_shape only the fov)
parser.add_argument('--n_levels', type=int, dest='n_levels', default=5)
parser.add_argument('--conv_per_level', type=int, dest='nb_conv_per_level', default=3)
parser.add_argument('--conv_size', type=int, dest='conv_size', default=5)
parser.add_argument('--unet_feat', type=int, dest='unet_feat_count', default=24)
parser.add_argument('--feat_mult', type=int, dest='feat_multiplier', default=2)
parser.add_argument('--activation', type=str, dest='activation', default='relu')
# normalisation. 'instance' = the same per-image normalisation in train and inference (train == validation
# == deployment). 'batch' = BatchNormalization(--batch_norm axis),
# which at batch size 1 is instance norm in training but the moving averages in predict, so train and deploy
# differ. 'none' = no normalisation.
parser.add_argument('--norm', type=str, dest='norm', default='instance', choices=['batch', 'instance', 'none'])
# an axis, not a flag: -1 is the feature axis. only used when --norm batch.
parser.add_argument('--batch_norm', type=str, dest='batch_norm', default='-1')
parser.add_argument('--no_residuals', action='store_true', dest='no_residuals')

# training
parser.add_argument('--lr', type=float, dest='lr', default=1e-4)
parser.add_argument('--clipnorm', type=float, dest='clipnorm', default=0.)
parser.add_argument('--epochs', type=int, dest='epochs', default=100)
parser.add_argument('--steps_per_epoch', type=int, dest='steps_per_epoch', default=1000)
parser.add_argument('--checkpoint', type=str, dest='checkpoint', default=None)
parser.add_argument('--seed', type=int, dest='seed', default=0)

args = vars(parser.parse_args())


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


args['labels_dir'] = resolve(args['labels_dir'])
args['generation_labels'] = resolve(args['generation_labels'])
args['generation_classes'] = resolve(args['generation_classes'])

# --no_deform turns every spatial term off, the flip included; otherwise keep the dice-qc defaults
if args.pop('no_deform'):
    args.update(scaling_bounds=False, rotation_bounds=False, shearing_bounds=False, nonlin_std=0.,
                flipping=False)

# translate --norm into the (batch_norm axis, instance_norm flag) the library takes. --norm is authoritative:
# instance and none both turn batch norm off, batch reads the axis from --batch_norm.
norm = args.pop('norm')
args['instance_norm'] = (norm == 'instance')
if norm == 'batch':
    args['batch_norm'] = None if args['batch_norm'].strip().lower() in ('none', 'off') else int(args['batch_norm'])
else:
    args['batch_norm'] = None
args['use_residuals'] = not args.pop('no_residuals')

training(**args)
