"""

Train the tissue-means regressor: label maps in, one mean intensity per tissue out. Thin CLI over
QC.training_tm.training, in the same shape as commands/training_bf.py, with the argparse dest names
matching the function's parameter names.

  python scripts/commands/training_tm.py \
      <data>/qc-data/synth/training_label_maps \
      <repo>/models/contrast/tm_9groups \
      --bias_std 0.5 --gamma_std 0.5 --norm instance

--generation_classes has to group the labels of each regressed tissue into one class, otherwise the
target of that tissue mixes several drawn intensities; training refuses a grouping that does not. The
defaults are the extra-cerebral 531 vocabulary with the nine-group classes and their 19 neutral labels;
--generation_labels, --generation_classes and --neutral_labels describe one array and move together.

The target is read off the corrupted image, so with --bias_prob and --gamma_prob both at 1 the network
never sees the clean end of its own range. Lower them to mix the regimes inside the epoch.

There is no online validation. Every label map in labels_dir trains, nothing is held out, and no
val_loss, val_###.npz or logs/validation is written. Checkpoints are scored offline afterwards, with
scripts/commands/validate_tm.py on real images. The two synthetic scorers,
scripts/experiments/eval_tissue_means.py and scripts/experiments/validate_tissue_means.py, build the
head of experiments/training_tissue_means.py and refuse a checkpoint written here.

Relative paths, the two .npy defaults below included, resolve against the current directory, so run it
from the repo root.

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
from QC.training_tm import training

parser = ArgumentParser()

# Positional arguments
parser.add_argument("labels_dir", type=str, help="folder of training label maps")
parser.add_argument("model_dir", type=str, help="folder the tm_###.h5 checkpoints and logs go in")

# Generation parameters
parser.add_argument("--generation_labels", type=str, dest="generation_labels",
                    default='data/labels_classes_priors/extra_cerebral_531/generation_labels_extra531.npy',
                    help="1d array of every label value in the maps")
parser.add_argument("--generation_classes", type=str, dest="generation_classes",
                    default='data/labels_classes_priors/extra_cerebral_531/generation_classes_9groups.npy',
                    help="1d array grouping the labels that share a drawn gaussian. It has to put each "
                         "regressed tissue in a single class.")
parser.add_argument("--tissues", type=str, dest="tissues", default=None,
                    help="comma separated groups to regress, among the keys of QC.training_tm.tissue_groups, "
                         "in output order. It fixes the width of the output. Default: all of them.")
parser.add_argument("--neutral_labels", type=int, dest="n_neutral_labels", default=19,
                    help="number of non-lateral labels in generation_labels. It has to match the array: "
                         "the flip pairs the rest left to right and an odd remainder is refused.")
parser.add_argument("--output_shape", type=int, dest="output_shape", default=160,
                    help="side of the cropped output image")

# Deformation and intensity
parser.add_argument("--no_deform", action='store_true', dest="no_deform",
                    help="turn every spatial term off, the flip included, for a local cpu smoke. "
                         "Otherwise the QC net's deformation defaults are used.")
parser.add_argument("--randomise_res", action='store_true', dest="randomise_res",
                    help="simulate a random acquisition resolution. Off by default, i.e. the clean regime.")
parser.add_argument("--bias_std", type=float, dest="bias_field_std", default=0.,
                    help="max std of the normal the small bias tensor is drawn from; sigma ~ U(0, this), "
                         "once per image. 0 leaves the image bias-free.")
parser.add_argument("--bias_prob", type=float, dest="bias_prob", default=.95,
                    help="fraction of the images the sampled field is actually applied to")
parser.add_argument("--gamma_std", type=float, dest="gamma_std", default=0.,
                    help="std of the gamma augmentation, an exponent drawn as exp(N(0, this))")
parser.add_argument("--gamma_prob", type=float, dest="gamma_prob", default=1.,
                    help="fraction of the images the gamma is actually applied to")
parser.add_argument("--clip", type=int, dest="clip", default=0,
                    help="intensity clipping percentile as an integer, 0 keeps the exact min-max. 300 is "
                         "SynthSeg's own training value.")

# Architecture parameters
parser.add_argument("--n_levels", type=int, dest="n_levels", default=5, help="number of levels of the encoder")
parser.add_argument("--conv_per_level", type=int, dest="nb_conv_per_level", default=3, help="convs per level")
parser.add_argument("--conv_size", type=int, dest="conv_size", default=5, help="size of the convolution kernels")
parser.add_argument("--unet_feat", type=int, dest="unet_feat_count", default=24,
                    help="number of features of the first level")
parser.add_argument("--feat_mult", type=int, dest="feat_multiplier", default=2,
                    help="factor of new feature maps per level")
parser.add_argument("--activation", type=str, dest="activation", default='relu', help="activation function")
parser.add_argument("--norm", type=str, dest="norm", default='instance', choices=['batch', 'instance', 'none'],
                    help="'instance' normalises each image by its own statistics, in training and at "
                         "inference alike. 'batch' is BatchNormalization on the --batch_norm axis, which "
                         "at batch size 1 is instance norm while training but uses the moving averages "
                         "in predict. 'none' is no normalisation. The choice is recorded in the weights "
                         "and refused on resume if it does not match.")
parser.add_argument("--batch_norm", type=str, dest="batch_norm", default='-1',
                    help="axis to batch normalise: -1 is the feature axis. 'none' or 'off' turns it off. "
                         "Only read when --norm batch.")
parser.add_argument("--no_residuals", action='store_true', dest="no_residuals",
                    help="drop the per-level residual connection")

# Training parameters
parser.add_argument("--lr", type=float, dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--clipnorm", type=float, dest="clipnorm", default=0.,
                    help="gradient norm clipping, 0 to leave it off")
parser.add_argument("--epochs", type=int, dest="epochs", default=100, help="number of epochs")
parser.add_argument("--steps_per_epoch", type=int, dest="steps_per_epoch", default=1000,
                    help="steps per epoch, i.e. how often the model is saved")
parser.add_argument("--min_vox", type=int, dest="min_vox", default=8,
                    help="a tissue with fewer voxels than this in the crop is not scored in the loss")
parser.add_argument("--checkpoint", type=str, dest="checkpoint", default=None,
                    help="a tm_###.h5 to resume from. The epoch, and the seed offset that goes with it, "
                         "are parsed out of that filename, so it has to keep the prefix.")
parser.add_argument("--seed", type=int, dest="seed", default=0, help="random seed")

args = vars(parser.parse_args())

# --no_deform turns every spatial term off, the flip included
if args.pop('no_deform'):
    args.update(scaling_bounds=False, rotation_bounds=False, shearing_bounds=False, nonlin_std=0.,
                flipping=False)

# translate --norm into the (batch_norm axis, instance_norm flag) the library takes: instance and none
# both turn batch norm off, batch reads the axis from --batch_norm.
norm = args.pop('norm')
args['instance_norm'] = (norm == 'instance')
if norm == 'batch':
    args['batch_norm'] = None if args['batch_norm'].strip().lower() in ('none', 'off') else int(args['batch_norm'])
else:
    args['batch_norm'] = None
args['use_residuals'] = not args.pop('no_residuals')

training(**args)
