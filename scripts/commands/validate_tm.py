"""

Validate every checkpoint of one tissue-means training on the small real validation set.

Thin CLI over QC.validate_tm.validate_training, in the same shape as commands/predict_tm.py: the
argparse dest names are the function's parameter names, so the two cannot drift.

One training per call. --norm is per arm and is an architecture argument, not a preference: a checkpoint
trained with norm='none' holds no tm_enc_in_down_* layers at all, and load_weights_checked refuses the
wrong one rather than loading it badly.

Results land in <validation_main_dir>/tm_<epoch>/tm_results.csv, one per checkpoint, and a checkpoint
whose csv already exists is skipped unless --recompute. The job is therefore resumable: relaunch it and
it continues where the wall clock cut it off.

  python scripts/commands/validate_tm.py \
      <data>/qc-data/validation/img <data>/qc-data/validation/gt/ss/segs \
      <repo>/models/contrast/tm_e2_instance_artefacts \
      <data>/qc-data/validation/scores/tm_e2 --norm instance

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
"""

from argparse import ArgumentParser
from QC.validate_tm import validate_training

parser = ArgumentParser(description=__doc__)

# Positional arguments
parser.add_argument("image_dir", type=str, help="folder of validation images")
parser.add_argument("gt_dir", type=str,
                    help="folder of segmentations, matched to the images by sorting order")
parser.add_argument("models_dir", type=str, help="folder of tm_*.h5 checkpoints to validate")
parser.add_argument("validation_main_dir", type=str, help="folder the per-epoch subfolders go in")

# What to validate
parser.add_argument("--tissues", type=str, dest="tissues", default=None,
                    help="groups the checkpoints were trained with; it fixes the width of the output. "
                         "Default: every key of QC.training_tm.tissue_groups, in its order.")
parser.add_argument("--step_eval", type=int, dest="step_eval", default=1,
                    help="validate one checkpoint every step_eval, to sketch a curve before filling it in")

# Preprocessing, and it must match what predict_tm does at deployment
parser.add_argument("--cropping", type=int, dest="cropping", default=160,
                    help="the window the network sees. Default 160, the shape training cropped to. It is "
                         "the only window knob: the padding follows it")
parser.add_argument("--target_res", type=float, dest="target_res", default=1.,
                    help="resolution the images are resampled to before anything else")
parser.add_argument("--no_resample", action='store_true', dest="no_resample",
                    help="skip the resampling entirely and score each image on its own grid")

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
                    help="the normalisation the checkpoints were TRAINED with. Per arm, and an architecture "
                         "argument: the wrong one is refused by load_weights_checked, not silently loaded.")

# Misc
parser.add_argument("--min_vox", type=int, dest="min_vox", default=8,
                    help="a tissue with fewer voxels than this in the crop is left blank in the gt columns")
parser.add_argument("--recompute", action='store_true', dest="recompute",
                    help="redo checkpoints that already have a tm_results.csv instead of skipping them")

args = vars(parser.parse_args())
if args.pop('no_resample'):
    args['target_res'] = None
validate_training(**args)
