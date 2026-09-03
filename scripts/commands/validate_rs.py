"""

Validate every checkpoint of one resolution training on a real validation set, one training per call.
Thin CLI over QC.validate_rs.validate_training, whose parameter names are the argparse dests.
There is no gt_dir: the truth for this head is the voxel spacing in each image's own header.

  python scripts/commands/validate_rs.py \
      <data>/qc-data/validation/img \
      <repo>/models/resolution/rs_extra_uniform_mm_a \
      <data>/qc-data/validation/scores/resolution/rs_a --norm instance

Results land in <validation_main_dir>/rs_<epoch>/rs_results.csv, one per checkpoint. A checkpoint
whose csv exists is skipped unless --recompute, so the job is resumable.

Do not read the resulting curve in aggregate: the loss weights the coarse end by s^2 while the QC
call is made in (1, 2] mm. The csv keeps true_R/A/S per image, so the bands can be cut afterwards.
See the header of QC/validate_rs.py.

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
"""

from argparse import ArgumentParser
from QC.validate_rs import validate_training

parser = ArgumentParser(description=__doc__)

# Positional arguments
parser.add_argument("image_dir", type=str, help="folder of validation images")
parser.add_argument("models_dir", type=str, help="folder of rs_*.h5 checkpoints to validate")
parser.add_argument("validation_main_dir", type=str, help="folder the per-epoch subfolders go in")

# What to validate
parser.add_argument("--step_eval", type=int, dest="step_eval", default=1,
                    help="validate one checkpoint every step_eval instead of all of them")

# Preprocessing (must match what predict_rs does at deployment)
parser.add_argument("--cropping", type=int, dest="cropping", default=160,
                    help="size of the window the network sees, cropped around the centre of the volume "
                         "and padded up if smaller. Default 160, the shape training cropped to")
parser.add_argument("--target_res", type=float, dest="target_res", default=1.,
                    help="resolution the images are resampled to first, and the grid the predicted "
                         "spacing is read against. Default 1 mm")
parser.add_argument("--no_resample", action='store_true', dest="no_resample",
                    help="score each image on its own grid instead of resampling it, a domain the "
                         "network never saw. Use it to measure that gap, not by default")
parser.add_argument("--minmax_norm", action='store_true', dest="minmax_norm",
                    help="normalise with an exact min-max instead of the p0.5-p99.5 default of "
                         "predict.py. The min-max is what training ends on")

# Architecture parameters
parser.add_argument("--conv_size", type=int, dest="conv_size", default=5, help="size of the convolution kernels")
parser.add_argument("--n_levels", type=int, dest="n_levels", default=5, help="number of levels of the encoder")
parser.add_argument("--conv_per_level", type=int, dest="nb_conv_per_level", default=3, help="convs per level")
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
                    help="redo checkpoints that already have an rs_results.csv instead of skipping them")

args = vars(parser.parse_args())
if args.pop('no_resample'):
    args['target_res'] = None
validate_training(**args)
