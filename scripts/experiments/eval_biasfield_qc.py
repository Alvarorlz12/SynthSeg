"""

Evaluate a trained bias-field QC checkpoint: how well does the head recover the realised std_log?
Rebuilds the generator + regression head, loads the checkpoint weights by name, runs N synthetic
samples, and reports MAE on physical std_log, correlation, and whether the predictions are ~constant
(collapse) or varying-but-uncorrelated (unstable).

Usage:
    python scripts/experiments/eval_biasfield_qc.py path/to/checkpoint.h5 [N=200]

NOTE: the architecture below must match the checkpoint's (same n_levels/feat/etc. and the same
build_biasqc_model version that produced it).

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors
"""

import os
import sys

# pin BLAS to 1 thread before numpy (same OpenBLAS deadlock guard as training)
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from keras import models

from SynthSeg.labels_to_image_model import labels_to_image_model
from experiments.training_biasfield_qc import build_biasqc_model
from SynthSeg.model_inputs import build_model_inputs
from ext.lab2im import utils

# must match training
STD_LOG_MAX = 0.2
OUTPUT_SHAPE = 160
N_LEVELS = 5
NB_CONV_PER_LEVEL = 3
CONV_SIZE = 5
UNET_FEAT_COUNT = 24
FEAT_MULTIPLIER = 2
ACTIVATION = 'relu'
BIAS_FIELD_STD = 0.5
BIAS_SCALE = 0.025

DATA = os.path.join(ROOT, 'data')
PRIORS = os.path.join(DATA, 'labels_classes_priors')
LABELS_DIR = os.path.join(DATA, 'training_label_maps')

checkpoint = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200

gen_labels = utils.load_array_if_path(os.path.join(PRIORS, 'generation_labels.npy'))
gen_classes = utils.load_array_if_path(os.path.join(PRIORS, 'generation_classes.npy'))
labels_paths = utils.list_images_in_folder(LABELS_DIR)
labels_shape, _, n_dims, _, _, atlas_res = utils.get_volume_info(labels_paths[0], aff_ref=np.eye(4))

generator = labels_to_image_model(labels_shape=labels_shape, n_channels=1,
                                  generation_labels=gen_labels, output_labels=gen_labels,
                                  n_neutral_labels=18, atlas_res=atlas_res, target_res=None,
                                  output_shape=OUTPUT_SHAPE, output_div_by_n=2 ** N_LEVELS,
                                  flipping=False, aff=np.eye(4),
                                  scaling_bounds=False, rotation_bounds=False, shearing_bounds=False,
                                  translation_bounds=False, nonlin_std=0, randomise_res=False,
                                  bias_field_std=BIAS_FIELD_STD, bias_scale=BIAS_SCALE,
                                  intensity_gamma_std=0.,   # match the training scope
                                  return_bias_std=True)
reg = build_biasqc_model(generator, N_LEVELS, NB_CONV_PER_LEVEL, CONV_SIZE,
                         UNET_FEAT_COUNT, FEAT_MULTIPLIER, ACTIVATION)
reg.load_weights(checkpoint, by_name=True)
print('loaded weights:', checkpoint)

# one graph that emits both the prediction and the true std_log from the same forward pass
eval_model = models.Model(generator.inputs,
                          [reg.outputs[0], generator.get_layer('bias_field_std').output])

gen_in = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels), batchsize=1,
                            n_channels=1, generation_classes=gen_classes, prior_distributions='uniform')

preds, trues = [], []
for _ in range(N):
    p, s = eval_model.predict(next(gen_in))
    preds.append(float(p) * STD_LOG_MAX)     # de-normalise pred to physical std_log
    trues.append(float(s))
preds, trues = np.array(preds), np.array(trues)

abs_err = np.abs(preds - trues)
mae = abs_err.mean()
mask = trues < STD_LOG_MAX                    # informative (non-saturated) range
mae_info = abs_err[mask].mean() if mask.any() else float('nan')
corr = float(np.corrcoef(preds, trues)[0, 1])

print('N = %d' % N)
print('true std_log  min/mean/max = %.3f / %.3f / %.3f' % (trues.min(), trues.mean(), trues.max()))
print('pred std_log  min/mean/max = %.3f / %.3f / %.3f  (std=%.4f)' %
      (preds.min(), preds.mean(), preds.max(), preds.std()))
print('MAE (all)              = %.4f' % mae)
print('MAE (true < %.2f)       = %.4f' % (STD_LOG_MAX, mae_info))
print('Pearson r(pred, true)  = %.3f' % corr)
if preds.std() < 1e-3:
    print('>>> predictions ~CONSTANT  -> collapsed (not using the image)')
elif corr > 0.7:
    print('>>> predictions track the target -> it is LEARNING')
else:
    print('>>> predictions vary but weakly correlate -> UNSTABLE / under-trained / miscalibrated')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.scatter(trues, preds, s=8, alpha=0.5)
    lim = float(max(trues.max(), preds.max())) * 1.05
    plt.plot([0, lim], [0, lim], 'r--', lw=1)
    plt.xlabel('true std_log'); plt.ylabel('pred std_log')
    plt.title('bias QC: pred vs true (MAE=%.3f, r=%.2f)' % (mae, corr))
    out = os.path.join(os.path.dirname(os.path.abspath(checkpoint)), 'eval_scatter.png')
    plt.tight_layout(); plt.savefig(out, dpi=110)
    print('scatter ->', out)
except Exception as e:
    print('(no scatter:', e, ')')
