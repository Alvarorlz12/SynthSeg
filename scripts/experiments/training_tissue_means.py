"""
Thin launcher for the per-tissue mean regressor as it was before the head was brought back to the QC
net's: the head the first experiments and the overfits were run with, and the one every tm_*.h5 already
on disk holds. It runs experiments/training_tissue_means.py, whose first head convolution is widened to
max(16, k) and whose last one is linear; scripts/experiments/overfit_tissue_means_qc.py builds that same
pair, under those same names. Use scripts/commands/training_tm.py for a new run.

    python scripts/experiments/training_tissue_means.py <labels_dir> <model_dir> [options]

deformation is on by default (same values as the dice qc net); pass --no_deform to turn it off for a
local cpu smoke. paths to the label/class arrays are resolved from the repo root when given relative.

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
from experiments.training_tissue_means import training

parser = ArgumentParser()

# positional
parser.add_argument('labels_dir', type=str)
parser.add_argument('model_dir', type=str)

# generation
parser.add_argument('--generation_labels', type=str, dest='generation_labels',
                    default='data/labels_classes_priors/generation_labels.npy')
parser.add_argument('--generation_classes', type=str, dest='generation_classes',
                    default='data/labels_classes_priors/generation_classes_3tissues_grouped.npy')
parser.add_argument('--tissues', type=str, dest='tissues', default='CSF,GM,WM')
parser.add_argument('--holdout', type=int, dest='holdout', default=4)
parser.add_argument('--neutral_labels', type=int, dest='n_neutral_labels', default=18)
parser.add_argument('--output_shape', type=int, dest='output_shape', default=160)

# deformation and intensity
parser.add_argument('--no_deform', action='store_true', dest='no_deform')
parser.add_argument('--randomise_res', action='store_true', dest='randomise_res')
parser.add_argument('--bias_std', type=float, dest='bias_field_std', default=0.)
parser.add_argument('--gamma_std', type=float, dest='gamma_std', default=0.)
# the fraction of images each corruption actually fires on. the target here is read off the corrupted
# image, so at prob 1 there is no clean image anywhere in the stream and the clean end of the range is
# never trained on. lower these to mix the regimes inside the epoch.
parser.add_argument('--bias_prob', type=float, dest='bias_prob', default=.95)
parser.add_argument('--gamma_prob', type=float, dest='gamma_prob', default=1.)
parser.add_argument('--clip', type=int, dest='clip', default=0)

# architecture (defaults are the dice qc net's; n_levels sets the downsampling, output_shape only the fov)
parser.add_argument('--n_levels', type=int, dest='n_levels', default=5)
parser.add_argument('--conv_per_level', type=int, dest='nb_conv_per_level', default=3)
parser.add_argument('--conv_size', type=int, dest='conv_size', default=5)
parser.add_argument('--unet_feat', type=int, dest='unet_feat_count', default=24)
parser.add_argument('--feat_mult', type=int, dest='feat_multiplier', default=2)
parser.add_argument('--activation', type=str, dest='activation', default='relu')
# normalisation. 'batch' = BatchNormalization(axis given by --batch_norm), the faithful default, which at
# batch size 1 is instance norm in training but the moving averages in predict, so train and deploy differ.
# 'instance' = the same per-image normalisation in both, so train, validation and deployment are one function
# (the fix for the randomised-contrast regime). 'none' = no normalisation.
parser.add_argument('--norm', type=str, dest='norm', default='batch', choices=['batch', 'instance', 'none'])
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
