"""
Command-line entry point for the tissue-means regressor, laid out like scripts/commands/predict.py.

  # deployment: an image goes in, a csv comes out, no segmentation anywhere
  python scripts/commands/predict_tm.py <images> preds.csv models/tm_cerebral_instance/tm_149.h5

  # with a ground truth, to score it
  python scripts/commands/predict_tm.py <images> preds.csv <model.h5> --gt <segs_dir>

The crop is always centred on the volume, which is all a deployment has: the segmentation is read for
the ground-truth columns and never to choose the window. Read the header of QC/predict_tm.py
before reading a number out of the csv -- the output is a mean over the window, so --cropping is part
of the measurement and not a memory setting.

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""


# imports
from argparse import ArgumentParser
from QC.predict_tm import predict_tm

parser = ArgumentParser()

# Positional arguments
parser.add_argument("path_images", type=str, help="single image, folder of images, or text file listing them")
parser.add_argument("path_out", type=str, help="output csv, one row per image")
parser.add_argument("path_model", type=str, help="regressor checkpoint (a tm_*.h5)")

# Ground truth
parser.add_argument("--gt", type=str, dest="gt_folder", default=None,
                    help="folder of segmentations, one per image, paired in SORTED ORDER. Adds the "
                         "true_* columns and the error on the deliverable.")
parser.add_argument("--tissues", type=str, dest="tissues", default=None,
                    help="groups the checkpoint regresses, in the order it was trained with. Default: every "
                         "key of QC.training_tm.tissue_groups, in its order.")

# Saving paths
parser.add_argument("--resampled", type=str, dest="path_resampled", default=None,
                    help="path/folder of the images resampled at the given target resolution")

# Processing parameters
parser.add_argument("--cropping", type=int, dest="cropping", default=160,
                    help="the window the network sees, cropped around the centre of the volume and "
                         "padded up to the same size when the head is smaller. Default 160, the shape "
                         "training's RandomCrop used. It is the ONLY window knob: the padding follows "
                         "the crop, as it does in predict_synthseg.py.")
parser.add_argument("--target_res", type=float, dest="target_res", default=1.,
                    help="resolution the image is resampled to before anything else")
parser.add_argument("--no_resample", action='store_true', dest="no_resample",
                    help="skip the resampling entirely and score the image on its own grid")

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
                    help="the normalisation the checkpoint was TRAINED with. It is an architecture "
                         "argument: the wrong one is refused by load_weights_checked, not silently loaded.")

# Misc
parser.add_argument("--min_vox", type=int, dest="min_vox", default=8,
                    help="a tissue with fewer voxels than this in the crop is left blank in the gt columns")
parser.add_argument("--no_recompute", action='store_false', dest="recompute",
                    help="leave an existing output csv alone instead of overwriting it")

args = vars(parser.parse_args())
if args.pop('no_resample'):
    args['target_res'] = None
predict_tm(**args)
