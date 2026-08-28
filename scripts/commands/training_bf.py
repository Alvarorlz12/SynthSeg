"""

Train the bias-field severity regressor: label maps in, one scalar per image out. Thin CLI over
QC.training_bf.training, in the same shape as commands/predict_rs.py and commands/validate_rs.py, with
the argparse dest names matching the function's parameter names.

  python scripts/commands/training_bf.py \
      $WORK/qc-data/synth/training_label_maps \
      $WORK/SynthQC/models/biasfield/bf_after_gamma_a \
      --bias_field_after_gamma --gamma_std 0.5 --clip 300 --norm instance

--bias_field_after_gamma reorders the field against the gamma inside labels_to_image_model. That keyword
is not implemented there yet, so any invocation of this script raises TypeError until it is, with or
without the flag. --gamma_std sets the gamma the reorder is measured against, an exponent drawn as
exp(N(0, gamma_std)) and applied to every image.

There is no online validation. Every label map in labels_dir trains, nothing is held out, and no
val_loss, val_###.npz or logs/validation is written. Checkpoints are scored offline afterwards.

Relative paths, the two .npy defaults below included, resolve against the current directory, so run it
from the repo root, as the slurm launchers do.

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
from QC.training_bf import training

parser = ArgumentParser()

# Positional arguments
parser.add_argument("labels_dir", type=str, help="folder of training label maps")
parser.add_argument("model_dir", type=str, help="folder the bf_###.h5 checkpoints and logs go in")

# Generation parameters
parser.add_argument("--generation_labels", type=str, dest="generation_labels",
                    default='data/labels_classes_priors/generation_labels.npy',
                    help="1d array of every label value in the maps")
parser.add_argument("--generation_classes", type=str, dest="generation_classes",
                    default='data/labels_classes_priors/generation_classes.npy',
                    help="1d array grouping the labels that share a drawn gaussian. The standard "
                         "SynthSeg classes.")
parser.add_argument("--neutral_labels", type=int, dest="n_neutral_labels", default=18,
                    help="number of non-lateral labels in generation_labels. It has to match the array: "
                         "the flip pairs the rest left to right and an odd remainder is refused.")
parser.add_argument("--output_shape", type=int, dest="output_shape", default=160,
                    help="side of the cropped output image")

# Bias field: the target and its regime
parser.add_argument("--bias_std", type=float, dest="bias_field_std", default=0.7,
                    help="max std of the normal the small bias tensor is drawn from; sigma ~ U(0, this), "
                         "once per image. It is the scale of the target.")
parser.add_argument("--bias_prob", type=float, dest="bias_prob", default=0.9,
                    help="probability of applying the sampled field. The rest of the images are left "
                         "clean, with target 0.")
parser.add_argument("--bias_scale", type=float, dest="bias_scale", default=.025,
                    help="ratio between the label map size and the sampled bias tensor, i.e. how smooth "
                         "the field is")
parser.add_argument("--bias_field_after_gamma", action='store_true', dest="bias_field_after_gamma",
                    help="apply the bias field after the gamma augmentation, so that the severity "
                         "recorded is the severity the network sees. The default order is "
                         "GMM -> bias -> clip -> min-max -> gamma.")

# Other deformation and intensity
parser.add_argument("--no_deform", action='store_true', dest="no_deform",
                    help="turn every spatial term off, the flip included, for a local cpu smoke. "
                         "Otherwise the QC net's deformation defaults are used.")
parser.add_argument("--randomise_res", action='store_true', dest="randomise_res",
                    help="also simulate a random acquisition resolution. Off by default, so that the "
                         "bias field is the only intensity corruption.")
parser.add_argument("--gamma_std", type=float, dest="gamma_std", default=0.5,
                    help="std of the gamma augmentation, i.e. the knob --bias_field_after_gamma is "
                         "measured against")
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
parser.add_argument("--checkpoint", type=str, dest="checkpoint", default=None,
                    help="a bf_###.h5 to resume from. The epoch, and the seed offset that goes with it, "
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
