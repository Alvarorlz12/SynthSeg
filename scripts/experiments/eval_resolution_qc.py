"""
Evaluate a trained resolution-QC checkpoint: how well does the head recover the per-axis effective
resolution (voxel spacing, mm)? Rebuilds the generator + directional head, loads the checkpoint weights
by name, runs N fresh synthetic draws and reports the same probe as the overfit diagnostic (per-axis
MAE in mm and log, Pearson r pooled/per-axis, predict-min_res baseline, plus the native/iso/aniso draw
and native/degraded/above-iso axis strata). Use it to pick the best per-epoch checkpoint (train_model
has no online validation) before the real-scan inject gate.

Usage:
    python scripts/experiments/eval_resolution_qc.py path/to/checkpoint.h5 --probe-n 96

The architecture / generation flags below must match the run that produced the checkpoint (defaults
mirror train_resolution_qc.py). Pass overrides if you trained with different values, otherwise
load_weights(by_name=True) silently loads nothing into a mismatched graph.

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors
Licensed under the Apache License, Version 2.0; see http://www.apache.org/licenses/LICENSE-2.0
"""

import os
import sys
import argparse

# pin BLAS to 1 thread before numpy (same OpenBLAS deadlock guard as training)
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# scripts/experiments on the path so we can reuse the overfit diagnostic's probe_report, so the
# eval report stays identical to the overfit harness.
_EXP = os.path.join(ROOT, 'scripts', 'experiments')
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)

import numpy as np
from keras import models

from SynthSeg.labels_to_image_model import labels_to_image_model
from SynthSeg.training_resolution_qc import build_resqc_model     # the production head (matches training)
from SynthSeg.model_inputs import build_model_inputs
from ext.lab2im import utils
from overfit_resolution_qc import probe_report                    # reused metric/report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', help='path to a saved checkpoint (a .h5 written by train_resolution_qc.py)')
    p.add_argument('--probe-n', type=int, default=96)
    p.add_argument('--no-plot', action='store_true')
    # the following must match the training run that produced the checkpoint
    p.add_argument('--output-shape', type=int, default=160)
    p.add_argument('--n-levels', type=int, default=5)
    p.add_argument('--max-res-iso', type=float, default=4.0)
    p.add_argument('--max-res-aniso', type=float, default=8.0)
    p.add_argument('--grid-ablation', choices=['none', 'blur_only', 'kernel_random', 'kernel_phase'], default='none')
    p.add_argument('--context', action='store_true',
                   help='set only if the checkpoint was trained with the conv-encoder context (no_context=False).')
    p.add_argument('--spectral', action='store_true',
                   help='set if the checkpoint was trained WITH the per-axis spectral features (7-feat head).')
    p.add_argument('--reorient', action='store_true',
                   help='set if the checkpoint was trained WITH orientation DR (90-rotations+flips) -> probe on the same distribution.')
    p.add_argument('--rolloff', action='store_true',
                   help='set if the checkpoint was trained WITH the per-axis cumulative-energy roll-off feature (5-feat head).')
    p.add_argument('--drop-abs', action='store_true',
                   help='set if the checkpoint was trained WITHOUT the absolute log-raw energies lg1/lg2 (g-norm+roll head).')
    p.add_argument('--content-aniso-max', type=float, default=0.0,
                   help='set to the value the checkpoint was trained with (e.g. 0.8) to eval on the SAME content-anisotropy-augmented '
                        'distribution the head was trained on; 0 = clean synthetic.')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--ctx-dim', type=int, default=16)
    p.add_argument('--nb-conv-per-level', type=int, default=2)
    p.add_argument('--conv-size', type=int, default=3)
    p.add_argument('--unet-feat-count', type=int, default=24)
    p.add_argument('--feat-multiplier', type=int, default=2)
    p.add_argument('--activation', type=str, default='elu')
    p.add_argument('--n-neutral-labels', type=int, default=18)
    return p.parse_args()


def main():
    a = parse_args()

    DATA = os.path.join(ROOT, 'data')
    PRIORS = os.path.join(DATA, 'labels_classes_priors')
    LABELS_DIR = os.path.join(DATA, 'training_label_maps')

    gen_labels = utils.load_array_if_path(os.path.join(PRIORS, 'generation_labels.npy'))
    gen_classes = utils.load_array_if_path(os.path.join(PRIORS, 'generation_classes.npy'))
    labels_paths = utils.list_images_in_folder(LABELS_DIR)
    labels_shape, _, n_dims, _, _, atlas_res = utils.get_volume_info(labels_paths[0], aff_ref=np.eye(4))

    # label normalization range, identical to training (must match for the de-normalization to be correct)
    min_res = float(np.min(atlas_res))
    max_res = float(max(a.max_res_iso, a.max_res_aniso))
    log_min = float(np.log(min_res))
    log_span = float(np.log(max_res) - log_min)

    # rebuild the same generator + head graph (so layer names match the checkpoint); match training's reorient scope
    reorient_kw = (dict(flipping=True, enable_90_rotations=True)
                   if a.reorient else
                   dict(flipping=False, enable_90_rotations=False))
    generator = labels_to_image_model(labels_shape=labels_shape, n_channels=1,
                                      generation_labels=gen_labels, output_labels=gen_labels,
                                      n_neutral_labels=a.n_neutral_labels, atlas_res=atlas_res,
                                      target_res=None, output_shape=a.output_shape,
                                      output_div_by_n=2 ** a.n_levels,
                                      aff=np.eye(4), scaling_bounds=False, rotation_bounds=False,
                                      shearing_bounds=False, translation_bounds=False, nonlin_std=0, **reorient_kw,
                                      randomise_res=True, max_res_iso=a.max_res_iso, max_res_aniso=a.max_res_aniso,
                                      content_aniso_max=a.content_aniso_max,
                                      bias_field_std=0, return_resolution=True)
    reg = build_resqc_model(generator, n_dims,
                            no_context=not a.context, hidden=a.hidden, ctx_dim=a.ctx_dim, spectral=a.spectral,
                            rolloff=a.rolloff, drop_abs=a.drop_abs,
                            n_levels=a.n_levels, nb_conv_per_level=a.nb_conv_per_level,
                            conv_size=a.conv_size, unet_feat_count=a.unet_feat_count,
                            feat_multiplier=a.feat_multiplier, activation=a.activation)
    reg.load_weights(a.checkpoint, by_name=True)
    print('loaded weights:', a.checkpoint)

    # one graph emitting both the prediction (normalized log-spacing) and the true spacing (mm)
    probe = models.Model(generator.inputs, [reg.outputs[0], generator.get_layer('resolution').output])

    probe_src = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels), batchsize=1,
                                   n_channels=1, generation_classes=gen_classes, prior_distributions='uniform')
    probe_draw = lambda i: next(probe_src)

    print('\n=== EVAL: %d fresh draws ===' % a.probe_n)
    res = probe_report('eval', probe, probe_draw, a.probe_n, log_min, log_span, a.max_res_iso, min_res)

    print('\n=== HEADLINE ===')
    print('pooled r=%.3f (overstates -- bimodal label) | MAE=%.3f mm | log-MAE=%.3f | predict-%gmm baseline %.3f mm'
          % (res['r_pool'], res['mae_mm'], res['mae_log'], min_res, res['mae_base_min']))

    if not a.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            s_pred, trues = res['s_pred'].flatten(), res['trues'].flatten()
            plt.figure(figsize=(5, 5))
            plt.scatter(trues, s_pred, s=8, alpha=0.4)
            lim = float(max(trues.max(), s_pred.max())) * 1.05
            plt.plot([0, lim], [0, lim], 'r--', lw=1)
            plt.xlabel('true spacing (mm)'); plt.ylabel('pred spacing (mm)')
            plt.title('resolution QC: pred vs true (MAE=%.2f mm, r=%.2f)' % (res['mae_mm'], res['r_pool']))
            out = os.path.join(os.path.dirname(os.path.abspath(a.checkpoint)), 'eval_scatter.png')
            plt.tight_layout(); plt.savefig(out, dpi=110)
            print('scatter ->', out)
        except Exception as e:
            print('(no scatter:', e, ')')


if __name__ == '__main__':
    main()
