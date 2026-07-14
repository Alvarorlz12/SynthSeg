"""

Minimal overfit diagnostic + stabilisation A/B for the bias-field QC regressor (experiment-1).

Question: can the head learn image to std(log-bias)? We take the easiest version of the task,
freeze one (or --k) anatomy/contrast tuple and let only the in-graph bias sampler draw a fresh
sigma each step (anatomy constant, bias amplitude is the only varying signal).

The defaults below reproduce the on-disk architecture, whose failure mode is training instability rather
than inexpressivity: the encoder magnitude grows unchecked, the K.std pool feeds that magnitude straight
into the Dense, and the sigmoid saturates. Each lever below breaks one link in that chain.

Stabilisation levers (each a single variable; defaults reproduce the collapse):
  --norm-pool        parameter-free LayerNorm on the pooled [gmean||gstd] vector before the Dense. Kills
                     the degenerate 'scale all features up' direction (scale-invariant) while preserving
                     the cross-channel pattern that carries the signal.
  --output linear    drop the sigmoid (linear output, clip only in the metric): no saturation dead-zone,
                     gradient keeps flowing. Also: softplus.
  --clipnorm 1.0     Adam gradient-norm clipping (cheap insurance against runaway weight growth).
  --no-residuals     turn off the residual stream (a primary explosion path) in the encoder.


Run on a GPU node in the `synthqc` env (full 160^3 needs a GPU). Example:
    srun --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:40:00 --pty bash -lc \
      'module load miniforge; source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate synthqc; \
       cd ~/SynthQC; python scripts/experiments/overfit_biasfield_qc.py --output linear --norm-pool --clipnorm 1.0'

Copyright 2026 Álvaro Ruiz López, Benjamin Billot, and the SynthSeg contributors
Licensed under the Apache License, Version 2.0; see http://www.apache.org/licenses/LICENSE-2.0
"""

import os
import sys
import argparse

# Pin BLAS to one thread before numpy (same OpenBLAS worker-thread deadlock guard as train/eval).
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from keras import models
from keras.optimizers import Adam
import keras.layers as KL
import keras.backend as K

from SynthSeg.labels_to_image_model import labels_to_image_model
from SynthSeg.training_biasfield_qc import huber_loss
from SynthSeg.model_inputs import build_model_inputs
from SynthSeg import metrics_model as metrics
from ext.lab2im import utils
from ext.neuron import models as nrn_models


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # training
    p.add_argument('--steps', type=int, default=600, help='number of train_on_batch steps')
    p.add_argument('--batch', type=int, default=1, help='effective batch size: tile the frozen anatomy to N and '
                   'let the in-graph RNG draw N independent biases (attacks the bs=1 regress-to-mean collapse)')
    p.add_argument('--lr', type=float, default=1e-4, help='Adam learning rate')
    p.add_argument('--clipnorm', type=float, default=0.0, help='Adam gradient-norm clip (0 = off)')
    p.add_argument('--k', type=int, default=1, help='number of FROZEN anatomy/contrast tuples to cycle through')
    p.add_argument('--probe-n', type=int, default=128, help='number of fresh-bias draws used to probe the head')
    p.add_argument('--log-step', type=int, default=20, help='print the running loss every this many steps')
    # stabilisation / head levers (defaults match the on-disk architecture)
    p.add_argument('--output', choices=['sigmoid', 'linear', 'softplus'], default='sigmoid',
                   help='final activation (default sigmoid = on-disk)')
    p.add_argument('--norm-pool', action='store_true', help='LayerNorm the pooled vector (CAUTION: removes the '
                   'between-sample amplitude cue -> tends to regress to the mean)')
    p.add_argument('--logstd-pool', action='store_true', help='log(1+.) compress the pooled stats: bounds the '
                   'dynamic range (tames explosion) but is monotone so it keeps the amplitude ordering')
    p.add_argument('--no-residuals', action='store_true', help='disable residual connections in the encoder')
    p.add_argument('--freeze-encoder', action='store_true', help='LINEAR PROBE: freeze the encoder at init and train '
                   'ONLY the final Dense -> tests whether the signal is linearly decodable from the fixed pooled features')
    p.add_argument('--log-input', action='store_true', help='feed log(image) to the encoder instead of image')
    # architecture (defaults match scripts/experiments/train_biasfield_qc.py)
    p.add_argument('--output-shape', type=int, default=160)
    p.add_argument('--n-levels', type=int, default=5)
    p.add_argument('--nb-conv-per-level', type=int, default=3)
    p.add_argument('--conv-size', type=int, default=5)
    p.add_argument('--unet-feat-count', type=int, default=24)
    p.add_argument('--feat-multiplier', type=int, default=2)
    p.add_argument('--activation', type=str, default='relu')
    # bias / label
    p.add_argument('--bias-field-std', type=float, default=0.5)
    p.add_argument('--bias-scale', type=float, default=0.025)
    p.add_argument('--std-log-max', type=float, default=0.2)
    p.add_argument('--huber-delta', type=float, default=0.25)
    p.add_argument('--n-neutral-labels', type=int, default=18)
    return p.parse_args()


def build_head(input_model, a):
    """Local copy of the regression head so stabilisation levers can be toggled without touching the
    production training_biasfield_qc module. Defaults reproduce build_biasqc_model exactly."""
    last = input_model.outputs[0]
    input_shape = last.get_shape().as_list()[1:]
    enc = nrn_models.conv_enc(input_model=input_model, input_shape=input_shape,
                              nb_levels=a.n_levels, nb_conv_per_level=a.nb_conv_per_level,
                              conv_size=a.conv_size, nb_features=a.unet_feat_count,
                              feat_mult=a.feat_multiplier, activation=a.activation,
                              batch_norm=None, use_residuals=(not a.no_residuals), name='biasqc')
    feat = enc.outputs[0]                                                  # [B, w, w, w, F]
    gmean = KL.GlobalAveragePooling3D(name='biasqc_gmean')(feat)           # [B, F]
    gstd = KL.Lambda(lambda t: K.std(t, axis=[1, 2, 3]), name='biasqc_gstd')(feat)  # [B, F]
    pooled = KL.Concatenate(name='biasqc_pool')([gmean, gstd])             # [B, 2F]
    if a.logstd_pool:
        # gmean (avg of ReLU>=0) and gstd are non-negative; log1p bounds the range, monotone so keeps amplitude
        pooled = KL.Lambda(lambda t: K.log(1. + t), name='biasqc_logpool')(pooled)
    if a.norm_pool:
        # parameter-free LayerNorm over the feature axis: scale-invariant, kills the explosion direction
        pooled = KL.Lambda(lambda t: (t - K.mean(t, -1, keepdims=True)) / (K.std(t, -1, keepdims=True) + 1e-5),
                           name='biasqc_poolnorm')(pooled)
    act = None if a.output == 'linear' else a.output
    pred = KL.Dense(1, activation=act, name='biasqc_pred')(pooled)         # [B, 1]
    return models.Model(input_model.inputs, pred)


def build_loss(generator, reg, a):
    std_log = generator.get_layer('bias_field_std').output                 # [B, 1] raw / physical
    label = KL.Lambda(lambda s: K.clip(s / a.std_log_max, 0., 1.), name='biasqc_label')(std_log)
    pred = reg.outputs[0]
    loss = KL.Lambda(lambda x: huber_loss(x[0], x[1], a.huber_delta), name='qc_loss')([label, pred])
    loss._keras_shape = tuple(loss.get_shape().as_list())
    return models.Model(inputs=generator.inputs, outputs=loss)


def probe_report(tag, probe, dense, frozen_inputs, n_draws, std_log_max):
    """probe outputs [pred, std_log, dense_input]; reconstruct the pre-activation logit from the Dense
    weights so a saturated/exploded readout (extreme logit) is distinguishable from a constant-input head."""
    W, b = dense.get_weights()                          # W: [2F, 1], b: [1]
    preds, trues, pooled_rows, logits = [], [], [], []
    for i in range(n_draws):
        inp = frozen_inputs[i % len(frozen_inputs)]
        pred, std_log, pooled = probe.predict(inp)
        preds.append(float(pred))                       # raw output (sigmoid in (0,1) or linear in R)
        trues.append(float(std_log))
        pooled = np.asarray(pooled).reshape(-1)
        pooled_rows.append(pooled)
        logits.append(float(pooled.dot(W).reshape(-1)[0] + b[0]))
    preds_raw, trues = np.array(preds), np.array(trues)
    preds_phys = np.clip(preds_raw, 0., 1.) * std_log_max   # clip (no-op for sigmoid), physical std_log
    P = np.array(pooled_rows)
    logits = np.array(logits)
    F = P.shape[1] // 2
    gmean, gstd = P[:, :F], P[:, F:]

    def col_corr(M, y):
        Mc = M - M.mean(0, keepdims=True)
        yc = y - y.mean()
        denom = np.sqrt((Mc ** 2).sum(0) * (yc ** 2).sum() + 1e-12)
        return (Mc * yc[:, None]).sum(0) / denom
    r_pool = col_corr(P, trues)
    r_pred = float(np.corrcoef(preds_raw, trues)[0, 1]) if preds_raw.std() > 1e-9 else float('nan')

    print('  [%s] over %d fresh-bias draws on %d frozen anatomy(ies):' % (tag, n_draws, len(frozen_inputs)))
    print('    true std_log   min/mean/max = %.3f / %.3f / %.3f  (std=%.4f)'
          % (trues.min(), trues.mean(), trues.max(), trues.std()))
    print('    pred (raw out) min/mean/max = %.4g / %.4g / %.4g  (std=%.5g)'
          % (preds_raw.min(), preds_raw.mean(), preds_raw.max(), preds_raw.std()))
    print('    pred std_log   min/mean/max = %.3f / %.3f / %.3f  (std=%.5f)'
          % (preds_phys.min(), preds_phys.mean(), preds_phys.max(), preds_phys.std()))
    print('    Pearson r(pred, true)        = %.3f' % r_pred)
    print('    pre-activation logit  min/mean/max = %.4g / %.4g / %.4g' % (logits.min(), logits.mean(), logits.max()))
    print('    pooled vector: mean per-dim std across draws = %.4g  (gmean=%.4g, gstd=%.4g)'
          % (P.std(0).mean(), gmean.std(0).mean(), gstd.std(0).mean()))
    print('    pooled<->true corr: max|r| = %.3f, #dims |r|>0.3 = %d / %d'
          % (np.nanmax(np.abs(r_pool)), int((np.abs(r_pool) > 0.3).sum()), P.shape[1]))
    return {'r_pred': r_pred, 'pred_std': float(preds_phys.std()),
            'logit': (float(logits.min()), float(logits.max())),
            'max_r_pool': float(np.nanmax(np.abs(r_pool))), 'pooled_std': float(P.std(0).mean())}


def main():
    a = parse_args()
    print('config:', vars(a))

    DATA = os.path.join(ROOT, 'data')
    PRIORS = os.path.join(DATA, 'labels_classes_priors')
    LABELS_DIR = os.path.join(DATA, 'training_label_maps')

    gen_labels = utils.load_array_if_path(os.path.join(PRIORS, 'generation_labels.npy'))
    gen_classes = utils.load_array_if_path(os.path.join(PRIORS, 'generation_classes.npy'))
    labels_paths = utils.list_images_in_folder(LABELS_DIR)
    labels_shape, _, n_dims, _, _, atlas_res = utils.get_volume_info(labels_paths[0], aff_ref=np.eye(4))

    generator = labels_to_image_model(labels_shape=labels_shape, n_channels=1,
                                      generation_labels=gen_labels, output_labels=gen_labels,
                                      n_neutral_labels=a.n_neutral_labels, atlas_res=atlas_res,
                                      target_res=None, output_shape=a.output_shape,
                                      output_div_by_n=2 ** a.n_levels,
                                      flipping=False, aff=np.eye(4),
                                      scaling_bounds=False, rotation_bounds=False, shearing_bounds=False,
                                      translation_bounds=False, nonlin_std=0, randomise_res=False,
                                      bias_field_std=a.bias_field_std, bias_scale=a.bias_scale,
                                      intensity_gamma_std=0.,   # clean scope: the bias is the only nuisance
                                      return_bias_std=True)

    if a.log_input:
        log_img = KL.Lambda(lambda x: K.log(K.maximum(x, 1e-6)), name='log_image')(generator.outputs[0])
        head_input_model = models.Model(generator.inputs, log_img)
        print('>>> --log-input ON: head consumes log(image)')
    else:
        head_input_model = generator

    reg = build_head(head_input_model, a)
    qc_model = build_loss(generator, reg, a)

    dense = reg.get_layer('biasqc_pred')
    probe = models.Model(generator.inputs,
                         [reg.outputs[0], generator.get_layer('bias_field_std').output, dense.input])

    base_gen = build_model_inputs(path_label_maps=labels_paths, n_labels=len(gen_labels), batchsize=1,
                                  n_channels=1, generation_classes=gen_classes, prior_distributions='uniform')
    frozen_inputs = [next(base_gen) for _ in range(a.k)]
    print('frozen %d anatomy/contrast tuple(s); only the in-graph bias sigma varies per step.' % a.k)
    # training inputs: tile each frozen tuple to the requested batch (same anatomy, N independent in-graph biases).
    # probing stays at batch 1 (one fresh draw at a time).
    train_inputs = [[np.repeat(x, a.batch, axis=0) for x in f] for f in frozen_inputs]

    print('\n=== PHASE 0: probe at init (no training) ===')
    probe_report('init', probe, dense, frozen_inputs, a.probe_n, a.std_log_max)

    if a.freeze_encoder:
        # linear probe: freeze everything, then re-enable only the final Dense (trainable flags apply at compile)
        for lyr in reg.layers:
            lyr.trainable = False
        reg.get_layer('biasqc_pred').trainable = True
        print('>>> --freeze-encoder ON: only the final Dense trains (linear probe on the fixed init features)')

    opt = Adam(lr=a.lr, clipnorm=a.clipnorm) if a.clipnorm > 0 else Adam(lr=a.lr)
    qc_model.compile(optimizer=opt, loss=metrics.IdentityLoss().loss)
    dummy = np.zeros((a.batch, 1))
    print('\n=== PHASE 1: overfit (%d steps, batch=%d, lr=%.1e, clipnorm=%s, output=%s, norm_pool=%s, residuals=%s) ==='
          % (a.steps, a.batch, a.lr, a.clipnorm or 'off', a.output, a.norm_pool, not a.no_residuals))
    run = None
    for step in range(1, a.steps + 1):
        inp = train_inputs[step % len(train_inputs)]
        loss = float(np.mean(qc_model.train_on_batch(inp, dummy)))
        run = loss if run is None else 0.97 * run + 0.03 * loss
        if step % a.log_step == 0 or step == 1:
            print('    step %4d   loss=%.4f   running=%.4f' % (step, loss, run))

    print('\n=== PHASE 2: probe post-training ===')
    res = probe_report('post', probe, dense, frozen_inputs, a.probe_n, a.std_log_max)

    print('\n=== VERDICT ===')
    if res['pred_std'] < 1e-3:
        if res['max_r_pool'] < 0.2:
            print('CONSTANT prediction AND signal does NOT reach the pool (max|r_pool|=%.2f) -> encoder+pool do '
                  'not encode std(log-bias): architecture cannot express it.' % res['max_r_pool'])
        else:
            print('CONSTANT prediction BUT signal reaches the pool (max|r_pool|=%.2f) -> readout collapsed '
                  '(saturated/exploded logit range [%.3g, %.3g]). Add --norm-pool / --output linear.'
                  % (res['max_r_pool'], res['logit'][0], res['logit'][1]))
    elif res['r_pred'] > 0.7:
        print('LEARNS on the frozen anatomy (r=%.2f, pred-std=%.3f): the head CAN learn std(log-bias).'
              % (res['r_pred'], res['pred_std']))
        print('-> stabilisation worked.')
    else:
        print('PARTIAL: pred varies (std=%.3f) but weakly tracks the target (r=%.2f). Inspect logits/pool above.'
              % (res['pred_std'], res['r_pred']))


if __name__ == '__main__':
    main()
