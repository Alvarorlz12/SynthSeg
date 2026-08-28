"""
Thin launcher for the per-axis resolution regressor, mirroring scripts/experiments/training_biasfield_scalar.py.

    python scripts/experiments/training_resolution_scalar.py <labels_dir> <model_dir> [options]

regresses three numbers per image: the per-array-axis voxel spacing the content was degraded to, as a
resolution LOSS relative to the stored 1mm grid, in log, normalised to [0, 1] by max(--max_res_iso,
--max_res_aniso). the resolution is the target and the only corruption on by default (--bias_std 0).
deformation is on by default (the dice qc net's values); pass --no_deform for a local cpu smoke. --norm
defaults to instance (train == validation == deployment). paths to the label/class arrays resolve from the
repo root when relative.

every axis is drawn independently from a uniform over [1 mm, max_res], with a fixed probability of a
native volume. --synthseg_sampler restores SynthSeg's own instead.

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
from QC.training_rs import training

parser = ArgumentParser()

# positional
parser.add_argument('labels_dir', type=str)
parser.add_argument('model_dir', type=str)

# generation. the standard classes (not the 3-tissue grouping): the resolution target does not depend on the
# tissue grouping, so richer per-structure contrast is the better domain randomisation here.
parser.add_argument('--generation_labels', type=str, dest='generation_labels',
                    default='data/labels_classes_priors/generation_labels.npy')
parser.add_argument('--generation_classes', type=str, dest='generation_classes',
                    default='data/labels_classes_priors/generation_classes.npy')
parser.add_argument('--neutral_labels', type=int, dest='n_neutral_labels', default=18)
parser.add_argument('--output_shape', type=int, dest='output_shape', default=160)

# resolution: the target and its regime. the defaults are SynthSeg's.
parser.add_argument('--max_res_iso', type=float, dest='max_res_iso', default=4.)
parser.add_argument('--max_res_aniso', type=float, dest='max_res_aniso', default=8.)
# the sampler. per-axis independent uniform with a fixed probability of a native volume; SynthSeg's own
# is one flag away. Report R2 WITHOUT the native draws: on the stock sampler it read 0.9764 with them and
# 0.9044 without, because most of the points were the same value predicted almost perfectly.
parser.add_argument('--synthseg_sampler', action='store_true', dest='synthseg_sampler',
                    help="use SynthSeg's own resolution sampler (one shared value across the axes, or a "
                         'single degraded axis with the other two at atlas_res) instead of one '
                         'independent uniform draw per axis')
# the dice qc net puts a relu on its last head conv; the tissue-means head is linear because its
# target sits near 0.5. this target is >= 0 and is exactly 0 on a native volume, so relu fits it.
parser.add_argument('--res_prob_min', type=float, dest='res_prob_min', default=0.2,
                    help='probability of drawing a native volume, i.e. 1 mm isotropic. Default 0.2.')
# 'blur_only' drops the resampling grid entirely (only the Gaussian blur cue survives); 'kernel_phase' /
# 'kernel_random' randomise the resample kernel and sub-voxel phase. default none = stock SynthSeg.
parser.add_argument('--no_deform', action='store_true', dest='no_deform')
parser.add_argument('--bias_std', type=float, dest='bias_field_std', default=0.)
parser.add_argument('--bias_scale', type=float, dest='bias_scale', default=.025)
parser.add_argument('--gamma_std', type=float, dest='gamma_std', default=0.)
parser.add_argument('--clip', type=int, dest='clip', default=0)

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
