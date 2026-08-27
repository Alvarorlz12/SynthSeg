"""
Launch script for the bias-field QC regressor (experiment-1).
Calls experiments.training_biasfield_qc.training with the experiment parameters.
Puts the repo root on sys.path and resolves paths relative to itself, so it
runs from any working directory (interactive node, nohup, or sbatch).

Usage (needs a GPU):
    python scripts/experiments/train_biasfield_qc.py

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors
Licensed under the Apache License, Version 2.0; see http://www.apache.org/licenses/LICENSE-2.0
"""

import os

# Pin BLAS to one thread before numpy is imported. OpenBLAS's threadpool intermittently
# deadlocks at 100% CPU (and the GPU then sits idle) the first time np.linalg.inv runs inside
# the keras data-loader worker thread (get_ras_axes, load_volume). A 4x4 affine inversion is
# single-threaded anyway, so pinning to 1 thread removes the deadlock at zero performance cost.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
print(sys.executable)

# repo root = two levels up from scripts/experiments/, put it on sys.path for the imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.training_biasfield_qc import training

# paths
DATA = os.path.join(ROOT, 'data')
PRIORS = os.path.join(DATA, 'labels_classes_priors')
path_training_label_maps = os.path.join(DATA, 'training_label_maps')
path_generation_labels = os.path.join(PRIORS, 'generation_labels.npy')
path_generation_classes = os.path.join(PRIORS, 'generation_classes.npy')
path_model_dir = os.path.join(ROOT, 'models', 'biasfield_qc_v1')

# bias field / label
bias_field_std = 0.5     # sigma1 ~ U(0, 0.5) sampled in-graph
bias_scale = 0.025
std_log_max = 0.2        # label = clamp(std_log / std_log_max, 0, 1)
huber_delta = 0.25       # transition at ~0.05 std_log: quadratic for typical errors, linear for the saturated tail

# encoder architecture
n_levels = 5
nb_conv_per_level = 3
conv_size = 5
unet_feat_count = 24
feat_multiplier = 2
activation = 'relu'

# output / GMM
n_neutral_labels = 18
output_shape = 160
n_channels = 1
batchsize = 1
prior_distributions = 'uniform'

# training
lr = 1e-4
epochs = 300
steps_per_epoch = 1000

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
         bias_field_std=bias_field_std,
         bias_scale=bias_scale,
         std_log_max=std_log_max,
         huber_delta=huber_delta,
         n_levels=n_levels,
         nb_conv_per_level=nb_conv_per_level,
         conv_size=conv_size,
         unet_feat_count=unet_feat_count,
         feat_multiplier=feat_multiplier,
         activation=activation,
         lr=lr,
         epochs=epochs,
         steps_per_epoch=steps_per_epoch)
