"""
Command-line entry point for the per-axis resolution regressor, laid out like predict_tm.py.

  python scripts/commands/predict_rs.py <images> preds.csv <model.h5>

There is no --gt: the truth for this head is the voxel spacing in each image's own header, so the
true_* columns are always written.

The resampling to 1 mm is what puts a real scan in the training domain, and the header spacing is the
grid the file is stored on, which on --conform'd data is 1 mm whatever it was acquired at. See the
header of QC/predict_rs.py.

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
from QC.predict_rs import predict_rs

parser = ArgumentParser()

# Positional arguments
parser.add_argument("path_images", type=str, help="single image, folder of images, or text file listing them")
parser.add_argument("path_out", type=str, help="output csv, one row per image")
parser.add_argument("path_model", type=str, help="regressor checkpoint (an rs_*.h5)")

# Saving paths
parser.add_argument("--resampled", type=str, dest="path_resampled", default=None,
                    help="path/folder of the images resampled at the given target resolution")

# Processing parameters
parser.add_argument("--cropping", type=int, dest="cropping", default=160,
                    help="size of the window the network sees, cropped around the centre of the volume "
                         "and padded up if smaller. Default 160, the shape training cropped to")
parser.add_argument("--target_res", type=float, dest="target_res", default=1.,
                    help="resolution the image is resampled to first, and the grid the predicted "
                         "spacing is read against. Default 1 mm")
parser.add_argument("--no_resample", action='store_true', dest="no_resample",
                    help="score the image on its own grid instead of resampling it, a domain the "
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
                    help="the normalisation the checkpoint was trained with. It describes the "
                         "architecture: the wrong one is refused by load_weights_checked, not loaded")

# Misc
parser.add_argument("--no_recompute", action='store_false', dest="recompute",
                    help="leave an existing output csv alone instead of overwriting it")

args = vars(parser.parse_args())
if args.pop('no_resample'):
    args['target_res'] = None
predict_rs(**args)
