"""

Launch script for the resolution QC regressor (per-axis voxel spacing).
Imports experiments.training_resolution_qc.training and calls it with the experiment parameters.
Self-contained: it puts the repo root on sys.path and resolves data/model paths relative to itself,
so it runs from any working directory (interactive node, nohup, or sbatch).

Usage (needs a GPU):
    python scripts/experiments/train_resolution_qc.py

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors
Licensed under the Apache License, Version 2.0; see http://www.apache.org/licenses/LICENSE-2.0
"""

import os

# pin BLAS to one thread before numpy is imported (OpenBLAS otherwise intermittently deadlocks at 100% CPU
# the first time np.linalg.inv runs inside the keras data-loader worker thread). zero performance cost.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
print(sys.executable)

# repo root = two levels up from scripts/experiments/, put it on sys.path for the imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.training_resolution_qc import training

# paths
DATA = os.path.join(ROOT, 'data')
PRIORS = os.path.join(DATA, 'labels_classes_priors')
path_training_label_maps = os.path.join(DATA, 'training_label_maps')
path_generation_labels = os.path.join(PRIORS, 'generation_labels.npy')
path_generation_classes = os.path.join(PRIORS, 'generation_classes.npy')

# v5 config (post-probe fix for the inverted clean-real confound)
# v5 had two changes left: drop_abs drops the non-transferable lg1/lg2 absolute energies (~84-93% of the
# synthetic-only separation) to force the head onto the transferable g-norm/roll cues. The third one, the
# content-anisotropy augmentation, was removed from labels_to_image_model on 2026-08-28, so this script can
# no longer reproduce the v5 arms that used it.
DROP_ABS = False
path_model_dir = os.path.join(ROOT, 'models',
                              'resolution_qc_v5_aniso' + ('' if DROP_ABS else '_lgkept'))

# resolution sampling / label
max_res_iso = 4.0        # isotropic LR draw U(min_res, 4)
max_res_aniso = 8.0      # single-axis anisotropic LR draw up to 8 mm
#                                  head cannot tell the content aug from real degradation via a grid cheat
huber_delta = 0.1        # transition on the normalized log-spacing label (~23% of spacing)
degraded_weight = 4.0    # soft hurdle against the ~62% min_res point mass (dw8 was a null change, back to v1's 4)

# head architecture (validated config)
no_context = True        # pure directional head (the conv-encoder context collapses the head to the mode)
spectral = False         # v4: replace the amplitude-confounded spectral-shape features with the roll-off below (fork A)
rolloff = True           # v4: per-axis 95% cumulative-energy spectral roll-off, amplitude-invariant bandwidth cue
#                          that attacks the anatomical amplitude-vs-bandwidth confound (clean A-P read as coarse on IXI)
reorient = False         # v4: drop orientation DR, training is already RAS (= real test, verified), reorient only
#                          randomizes that alignment and is incompatible with a per-axis prior (and was a proven null)
hidden = 64
n_levels = 5             # sets output_div_by_n = 32 (160 stays 160)
nb_conv_per_level = 2
conv_size = 3
unet_feat_count = 24
feat_multiplier = 2
activation = 'elu'

# output / GMM
n_neutral_labels = 18
output_shape = 160
n_channels = 1
batchsize = 1
prior_distributions = 'uniform'

# training
lr = 3e-4
epochs = 20
steps_per_epoch = 200    # total = 4000 steps; a checkpoint is saved each epoch (every 200 steps)

# run training

training(path_training_label_maps,
         path_model_dir,
         generation_labels=path_generation_labels,
         generation_classes=path_generation_classes,
         n_neutral_labels=n_neutral_labels,
         batchsize=batchsize,
         n_channels=n_channels,
         output_shape=output_shape,
         prior_distributions=prior_distributions,
         max_res_iso=max_res_iso,
         max_res_aniso=max_res_aniso,
         reorient=reorient,
         huber_delta=huber_delta,
         degraded_weight=degraded_weight,
         no_context=no_context,
         spectral=spectral,
         rolloff=rolloff,
         drop_abs=DROP_ABS,
         hidden=hidden,
         n_levels=n_levels,
         nb_conv_per_level=nb_conv_per_level,
         conv_size=conv_size,
         unet_feat_count=unet_feat_count,
         feat_multiplier=feat_multiplier,
         activation=activation,
         lr=lr,
         epochs=epochs,
         steps_per_epoch=steps_per_epoch)
