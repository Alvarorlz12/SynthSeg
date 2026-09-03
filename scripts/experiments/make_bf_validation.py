# -*- coding: utf-8 -*-
"""Build a frozen bias-field validation set from real images.

There is no ground truth for bias severity in any real dataset, so it is manufactured: each image is
read exactly as deployment reads it, a field is applied the way the generator applies it, and the
severity of that field is written down. The corrupted volumes are saved once and reused, for the
reason validate_biasfield_scalar.py already gives about its synthetic set -- a fresh draw per
checkpoint puts the sampling noise of the set on every point of the curve, and that noise is the size
of the effect.

Each anatomy gets ONE field shape, applied at several amplitudes. So severity varies within an image
with the anatomy and the field shape held fixed, which is what makes the curve readable: a real image
carries an unknown residual bias of its own, and in a within-image ladder that unknown is a constant
per anatomy instead of noise across the set.

The pipeline reproduces the generator's, not an approximation of it:

  read (predict_tm.preprocess: 1 mm, 160 crop centred on the volume, p0.5-p99.5 after the crop)
  log_bias ~ N(0, b_std) at bias_scale resolution, resized trilinear, exp     layers.py:1171
  image * exp(log_bias)
  clip to [0, 1]                                                             bias_ceiling, l2im:248
  target = std(log_bias) over the whole crop                                 l2im:235

Two things that are easy to get wrong:

  The target is the std of the UPSAMPLED field, not b_std. The trilinear resize smooths, so the
  realised severity is about 0.61 x b_std: the default ladder reaches 0.70 and the targets stop
  near 0.43.

  The saved volumes are already normalised. Whatever scores them must feed them to the network AS
  THEY ARE. Normalising again would subtract an offset after the field, which training never does.

  # the ladder, for choosing a checkpoint
  python scripts/experiments/make_bf_validation.py \
      --images <data>/qc-data/validation/img \
      --out_img <data>/qc-data/validation/img_bf \
      --out_csv <data>/qc-data/validation/gt/bf/targets.csv

  # one severity per image, for measuring on a cohort
  python scripts/experiments/make_bf_validation.py --sample \
      --images <data>/qc-data/index/gt_pairs/kirby21_bids/images.txt \
      --out_img <data>/qc-data/bf_sets/kirby21/img \
      --out_csv <data>/qc-data/bf_sets/kirby21/targets.csv
"""
import os
import sys
from argparse import ArgumentParser

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

import tensorflow as tf
import keras.backend as K
import keras.layers as KL
import keras.models as KM

from ext.lab2im import utils, layers
import ext.neuron.layers as nrn_layers
from QC.predict_tm import preprocess

LADDER = [0.0, 0.18, 0.35, 0.52, 0.70]


class FixedBiasField(layers.BiasFieldCorruption):
    """BiasFieldCorruption with sigma given instead of drawn, and the shape pinned by the seed, so a
    ladder of amplitudes shares one field."""

    def __init__(self, sigma, seed, bias_scale=0.025, **kwargs):
        # return_field=True or the inherited compute_output_shape hands keras one shape for the two
        # tensors this call returns, and _add_inbound_node raises
        super().__init__(bias_field_std=1.0, bias_scale=bias_scale, same_bias_for_all_channels=False,
                         prob=1.0, return_field=True, **kwargs)
        self.sigma = float(sigma)
        self.seed = int(seed)

    def call(self, inputs, **kwargs):
        if not self.several_inputs:
            inputs = [inputs]
        batchsize = tf.split(tf.shape(inputs[0]), [1, -1])[0]
        bias_shape = tf.concat([batchsize, tf.convert_to_tensor(self.small_bias_shape, 'int32')], 0)
        unit = tf.random.stateless_normal(bias_shape, seed=[self.seed, 0])
        log_bias = nrn_layers.Resize(size=self.inshape[0][1:self.n_dims + 1],
                                     interp_method='linear')(unit * self.sigma)
        out = tf.math.multiply(tf.math.exp(log_bias), inputs[0])
        return [out, log_bias]


def build_corrupter(image_shape, sigma, seed, bias_scale, bias_field_std=0.7, bias_prob=0.9,
                    sample=False):
    """In --sample mode this is the library's own layer, unchanged: it draws b_std ~ U(0, S) and the
    clean fraction itself, and zeroes the returned field on a clean draw, so the set is the training
    distribution and not a reimplementation of it. The ladder needs one field shape held across
    amplitudes, which no stock layer can do, so there it is FixedBiasField."""
    inp = KL.Input(shape=image_shape)
    if sample:
        img, log_bias = layers.BiasFieldCorruption(bias_field_std, bias_scale, False,
                                                   prob=bias_prob, return_field=True)(inp)
    else:
        img, log_bias = FixedBiasField(sigma, seed, bias_scale)(inp)
    img = KL.Lambda(lambda x: tf.clip_by_value(x, 0., 1.), name='bias_ceiling')(img)
    std = KL.Lambda(lambda x: tf.math.reduce_std(x, axis=[1, 2, 3]))(log_bias)
    return KM.Model(inp, [img, std])


def list_images(path):
    if os.path.isdir(path):
        return sorted(utils.list_images_in_folder(path))
    with open(path) as fh:
        return [l.strip() for l in fh if l.strip()]


def main():
    p = ArgumentParser()
    p.add_argument('--images', required=True, help='folder of images, or a file listing them')
    p.add_argument('--out_img', required=True)
    p.add_argument('--out_csv', required=True)
    p.add_argument('--ladder', type=float, nargs='+', default=LADDER,
                   help='b_std values, the sigma of the log-field before it is resized')
    p.add_argument('--sample', action='store_true',
                   help='one severity per image, drawn from U(0, bias_field_std) the way the '
                        'generator draws it, instead of the ladder. For measuring on a cohort, where '
                        'a real image is met once and its own residual bias is part of the answer; '
                        'the ladder is for choosing a checkpoint, where that residual has to cancel')
    p.add_argument('--bias_field_std', type=float, default=0.7,
                   help='upper bound of the draw in --sample, the value training ran with')
    p.add_argument('--bias_prob', type=float, default=0.9,
                   help='probability the layer applies a field at all in --sample, training own '
                        'value. The clean draws are the true zeros of the set, and the layer zeroes '
                        'the returned field on them, so a clean image is not labelled severe')
    p.add_argument('--bias_scale', type=float, default=0.025)
    p.add_argument('--cropping', type=int, default=160)
    p.add_argument('--target_res', type=float, default=1.)
    p.add_argument('--n_levels', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    paths = list_images(a.images)
    assert paths, 'no images in %s' % a.images
    os.makedirs(a.out_img, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)

    if a.sample:
        print('%d images, one severity each, drawn by the library layer: U(0, %.2f), bias_prob %.2f'
              % (len(paths), a.bias_field_std, a.bias_prob))
    else:
        print('%d images x %d severities = %d volumes' % (len(paths), len(a.ladder),
                                                          len(paths) * len(a.ladder)))
    rows = []
    for i, path in enumerate(paths):
        stem = os.path.basename(path).split('.nii')[0]
        im, _, _, _, _, _, _, _ = preprocess(path_image=path, n_levels=a.n_levels,
                                             target_res=a.target_res, crop=a.cropping)
        shape = list(im.shape[1:])
        # one seed per anatomy: the shape of the field is that image's own
        seed = a.seed + 1000 * i
        rungs = [(-1, None)] if a.sample else list(enumerate(a.ladder))
        for k, sigma in rungs:
            # a graph per volume, dropped afterwards. Keeping them costs a few hundred MB each and
            # the run dies around the twentieth: the sigma and the seed are baked into the graph, so
            # there is nothing to reuse anyway.
            K.clear_session()
            # --sample is left to the stateful RNG the library layer uses, unseeded: the set is
            # written once and kept, and the csv records what was applied, so pinning a seed would
            # buy nothing and would tie every volume's field to an arithmetic sequence of seeds.
            # The ladder is unaffected either way, FixedBiasField draws stateless from its own.
            out, std = build_corrupter(shape, sigma, seed, a.bias_scale, a.bias_field_std,
                                       a.bias_prob, a.sample).predict(im)
            out, std = np.squeeze(out), float(np.squeeze(std))
            name = '%s_bf' % stem if k < 0 else '%s_bf%d' % (stem, k)
            utils.save_volume(out, np.eye(4), None, os.path.join(a.out_img, name + '.nii.gz'))
            # b_std is blank in --sample: the layer draws it inside the graph and does not hand it
            # back. std_log_bias is the target either way, and it is the realised field, not the draw
            rows.append((name, stem, k, '' if a.sample else '%d' % seed,
                         '' if sigma is None else '%.4f' % sigma,
                         std, a.bias_scale, a.cropping, a.target_res))
            print('  %-40s b_std %-6s -> std(log B) %.4f'
                  % (name, '?' if sigma is None else '%.2f' % sigma, std))

    with open(a.out_csv, 'w') as fh:
        fh.write('stem,source,level,seed,b_std,std_log_bias,bias_scale,cropping,target_res\n')
        for r in rows:
            fh.write('%s,%s,%d,%s,%s,%.6f,%.4f,%d,%.2f\n' % r)
    print('wrote %s  (%d rows)' % (a.out_csv, len(rows)))

    t = np.array([r[5] for r in rows])
    print('targets: min %.4f  max %.4f  mean %.4f  sd %.4f  zeros %d of %d'
          % (t.min(), t.max(), t.mean(), t.std(), int((t == 0).sum()), len(t)))


if __name__ == '__main__':
    main()
