"""

Validate every checkpoint of one bias-field severity training on a fixed validation set, one training
per call. Thin CLI over QC.validate_bf.validate_training, whose parameter names are the argparse dests.

The ground truth is a csv, not a folder of segmentations: the target of this head is the severity the
corruption was applied with, and scripts/experiments/make_bf_validation.py writes the images and that
targets.csv together, joined by stem.

  python scripts/commands/validate_bf.py \
      <data>/qc-data/validation/img_bf \
      <data>/qc-data/validation/gt/bf/targets.csv \
      <models>/biasfield/bf_after_gamma_a \
      <data>/qc-data/validation/scores/bias/bf_a --norm instance

Results land in <validation_main_dir>/bf_<epoch>/bf_results.csv, one per checkpoint. A checkpoint whose
csv exists is skipped unless --recompute, so the job is resumable.

Images written by make_bf_validation.py are already resampled, cropped and normalised, which is what
--preprocessed assumes. Pass --raw for images that are not, and then --cropping and --target_res are
read; on preprocessed images they are ignored.

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
"""

from argparse import ArgumentParser
from QC.validate_bf import validate_training

parser = ArgumentParser(description=__doc__)

# Positional arguments
parser.add_argument("image_dir", type=str, help="folder of corrupted validation images")
parser.add_argument("gt_csv", type=str,
                    help="the targets.csv that came with them, joined to the images by stem")
parser.add_argument("models_dir", type=str, help="folder of bf_*.h5 checkpoints to validate")
parser.add_argument("validation_main_dir", type=str, help="folder the per-epoch subfolders go in")

# What to validate
parser.add_argument("--step_eval", type=int, dest="step_eval", default=1,
                    help="validate one checkpoint every step_eval instead of all of them")

# Preprocessing (must match what predict_bf does at deployment)
parser.add_argument("--raw", action='store_true', dest="raw",
                    help="the images still have to be resampled, cropped and normalised. By default "
                         "they are taken as already preprocessed, which is how make_bf_validation.py "
                         "writes them")
parser.add_argument("--cropping", type=int, dest="cropping", default=160,
                    help="size of the window the network sees, cropped around the centre of the volume "
                         "and padded up if smaller. Only read with --raw. Default 160, the shape "
                         "training cropped to")
parser.add_argument("--target_res", type=float, dest="target_res", default=1.,
                    help="resolution the images are resampled to first. Only read with --raw. Default 1 mm")

# Architecture parameters
parser.add_argument("--n_levels", type=int, dest="n_levels", default=5, help="number of levels of the encoder")
parser.add_argument("--conv_per_level", type=int, dest="nb_conv_per_level", default=3, help="convs per level")
parser.add_argument("--conv_size", type=int, dest="conv_size", default=5, help="size of the convolution kernels")
parser.add_argument("--unet_feat", type=int, dest="unet_feat_count", default=24,
                    help="number of features of the first level")
parser.add_argument("--feat_mult", type=int, dest="feat_multiplier", default=2,
                    help="factor of new feature maps per level")
parser.add_argument("--activation", type=str, dest="activation", default='relu', help="activation function")
parser.add_argument("--norm", type=str, dest="norm", default='instance', choices=['instance', 'batch', 'none'],
                    help="the normalisation the checkpoints were trained with. It describes the "
                         "architecture: the wrong one is refused by load_weights_checked, not loaded")

# Misc
parser.add_argument("--recompute", action='store_true', dest="recompute",
                    help="redo checkpoints that already have a bf_results.csv instead of skipping them")

args = vars(parser.parse_args())
args['preprocessed'] = not args.pop('raw')
validate_training(**args)
