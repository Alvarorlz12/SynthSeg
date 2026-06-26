"""
QC artifact transforms for the multi-artifact pipeline.

Two engines, one severity contract: we sample the severity ourselves, map it deterministically
to the physical parameter, apply it, and expose the realised severity as the regression label
(the QC target).

  - Section A, in-graph TF/Keras layers (keras `Layer` subclasses) for the intensity artifacts
    (noise, gamma). They run on-GPU inside `labels_to_image_model`. Named by effect
    (`<Effect>Corruption`) like the existing `BiasFieldCorruption` (ext/lab2im/layers.py).

  - Section B, out-of-graph TorchIO callables for the k-space artifacts (motion, ghosting,
    spike). Applied on the numpy image in the generator path. `torch`/`torchio` are
    imported lazily inside `__call__`, so importing this module for the TF layers alone does
    not pull in torch.
"""

# python imports
import numpy as np
import tensorflow as tf
from keras.layers import Layer

# project imports
from ext.lab2im import edit_tensors as l2i_et


# section A: in-graph TF/Keras layers

class NoiseCorruption(Layer):
    """Add zero-mean Gaussian noise at a recorded severity (its std), for QC.

    Mirrors `BiasFieldCorruption`: samples the noise std per batch element from [0, max_std],
    applies it, and can return that std as the label. Apply on the normalised image (after
    `IntensityAugmentation`) so the std is on the [0, 1] scale. The presence gate is per sample,
    so some volumes in a batch carry noise and others do not.

    The input tensor is expected to have shape [batch, *spatial, channels]. The output is the corrupted
    image, or [corrupted_image, severity] if return_severity=True, where severity has shape [batch, 1]
    (the realised noise std, 0.0 when not applied).

    :param max_std: upper bound of the noise std; the realised std ~ U(0, max_std).
    :param prob: per-sample probability of applying noise (severity is 0 otherwise).
    :param return_severity: whether to also output the realised severity (the QC target).
    """

    def __init__(self, max_std=0.1, prob=0.5, return_severity=False, **kwargs):
        self.max_std = max_std
        self.prob = prob
        self.return_severity = return_severity
        self.n_dims = None
        super(NoiseCorruption, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["max_std"] = self.max_std
        config["prob"] = self.prob
        config["return_severity"] = self.return_severity
        return config

    def build(self, input_shape):
        # input_shape = [batch, *spatial, channels], n_dims spatial dims
        self.n_dims = len(input_shape) - 2
        self.built = True
        super(NoiseCorruption, self).build(input_shape)

    def call(self, x, **kwargs):
        # one severity scalar per batch element, of shape [B, 1]
        batchsize = tf.split(tf.shape(x), [1, -1])[0]
        param_shape = tf.concat([batchsize, tf.ones([1], dtype='int32')], axis=0)

        # sample the realised noise std in [0, max_std], gated per sample by `prob`
        std = tf.random.uniform(param_shape, minval=0., maxval=self.max_std)
        keep = tf.cast(tf.random.uniform(param_shape) < self.prob, x.dtype)
        std = std * keep    # severity (0.0 if not applied)

        # add noise scaled by the per-sample std (broadcast over spatial dims and channels)
        std_map = l2i_et.expand_dims(std, axis=[1] * self.n_dims)   # [B, 1, ..., 1]
        noisy = x + tf.random.normal(tf.shape(x), dtype=x.dtype) * std_map

        if self.return_severity:
            return [noisy, std]
        return noisy

    def compute_output_shape(self, input_shape):
        if self.return_severity:
            return [input_shape, (input_shape[0], 1)]
        return input_shape

class GammaContrast(Layer):
    """Apply a random gamma intensity remap at a recorded severity, for QC.

    A gamma transform `out = x ** gamma` reshapes the contrast curve of the normalised image
    (x in [0, 1]): gamma < 1 brightens, gamma > 1 darkens. The layer samples a log-gamma per
    batch element, symmetric around 0, takes `exp` to get gamma, and can return |log gamma| as
    the severity (a sign-agnostic measure of distance from gamma == 1). Apply on the normalised image.

    Contrast is already randomised by the GMM, so this is a confound / sanity baseline the other
    heads must stay selective against, not a primary QC target, hence the name `GammaContrast`
    rather than `*Corruption`. The presence gate is per sample.

    The input tensor is expected to have shape [batch, *spatial, channels]. The output is the remapped
    image, or [remapped_image, severity] if return_severity=True, where severity has shape [batch, 1]
    and equals |log gamma| (0.0 when not applied).

    :param max_log_gamma: bound of the sampled log-gamma; log_gamma ~ U(-max_log_gamma, max_log_gamma),
        so gamma ~ [exp(-m), exp(m)] (e.g. 0.7 gives gamma in ~[0.50, 2.01]).
    :param prob: per-sample probability of applying the gamma shift (severity is 0 otherwise).
    :param return_severity: whether to also output the realised severity (the QC target).
    """

    def __init__(self, max_log_gamma=0.7, prob=0.5, return_severity=False, **kwargs):
        self.max_log_gamma = abs(max_log_gamma)
        self.prob = prob
        self.return_severity = return_severity
        self.n_dims = None
        super(GammaContrast, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["max_log_gamma"] = self.max_log_gamma
        config["prob"] = self.prob
        config["return_severity"] = self.return_severity
        return config

    def build(self, input_shape):
        # input_shape = [batch, *spatial, channels], n_dims spatial dims
        self.n_dims = len(input_shape) - 2
        self.built = True
        super(GammaContrast, self).build(input_shape)

    def call(self, x, **kwargs):
        # one severity scalar per batch element, of shape [B, 1]
        batchsize = tf.split(tf.shape(x), [1, -1])[0]
        param_shape = tf.concat([batchsize, tf.ones([1], dtype='int32')], axis=0)

        # sample the gamma values in [-max_log_gamma, max_log_gamma], gated per sample by `prob`
        g = tf.random.uniform(param_shape, minval=-self.max_log_gamma, maxval=self.max_log_gamma)
        keep = tf.cast(tf.random.uniform(param_shape) < self.prob, x.dtype)
        log_gamma = g * keep    # severity (0.0 if not applied)
        gamma = tf.exp(log_gamma)

        # apply the gamma contrast, broadcast over spatial dims and channels
        out = tf.pow(
            tf.clip_by_value(x, 0.0, 1.0),
            l2i_et.expand_dims(gamma, axis=[1] * self.n_dims)   # [B, 1, ..., 1]
        )

        if self.return_severity:
            return [out, tf.abs(log_gamma)]   # severity = |log(gamma)|, 0.0 if not applied
        return out

    def compute_output_shape(self, input_shape):
        if self.return_severity:
            return [input_shape, (input_shape[0], 1)]
        return input_shape

# section B: out-of-graph TorchIO callables

class ArtefactTransform:
    """Base class for the out-of-graph TorchIO artifact transforms (k-space physics).

    Same severity contract as the section A in-graph layers: we sample a scalar severity ``s``
    ourselves, map it deterministically to the physical parameter(s), drive a TorchIO ``Random*``
    transform with a degenerate range ``(v, v)`` so the realised parameter equals our value
    exactly, and expose ``s`` as the QC label.

    Severity is normalised here (unlike the physical scalar used by Noise/Gamma) because motion
    and ghosting each depend on several physical numbers with no single natural unit, so ``s`` in
    [0, 1] scales every physical parameter as one monotone knob. The physical parameters stay
    recoverable via ``_to_param(s)``. As with noise, the parameter is exact but the realisation
    can vary run-to-run, so the label, not the pixels, is what must be exact.

    Subclasses implement ``_to_param(s)`` (s to physical parameter(s)) and ``_apply(image_np, *p)``
    (the TorchIO call). ``__call__`` handles the absent case (severity == 0 returns the image
    unchanged, bit-identical) and the (W, H, D) to (C, W, H, D) plumbing. ``sample_severity`` draws
    ``s`` with a per-sample presence gate. ``torch``/``torchio`` are imported lazily inside ``_apply``.

    :param prob: per-sample probability of applying the artifact (severity is 0 otherwise).
    :param seed: optional seed for the severity RNG.
    """

    name = None

    def __init__(self, prob=0.5, seed=None):
        self.prob = prob
        self.rng = np.random.default_rng(seed)

    def sample_severity(self):
        """Draw a severity in [0, 1], or 0.0 when the per-sample presence gate is closed."""
        if self.rng.random() >= self.prob:
            return 0.0
        return float(self.rng.random())

    def _to_param(self, severity):
        """Map severity in [0, 1] to the deterministic physical parameter(s). Subclass."""
        raise NotImplementedError

    def _apply(self, image_np, *params):
        """Apply the TorchIO transform to a (W, H, D) numpy image. Subclass."""
        raise NotImplementedError

    def __call__(self, image_np, severity):
        """Apply the artifact at ``severity``. numpy (W, H, D) to numpy (W, H, D).

        ``severity == 0.0`` returns the image unchanged (bit-identical), so an absent artifact
        never perturbs the image even by interpolation round-off.
        """
        if severity <= 0.0:
            return image_np
        params = self._to_param(severity)
        if not isinstance(params, tuple):
            params = (params,)
        return self._apply(image_np, *params)


class MotionArtefact(ArtefactTransform):
    """Rigid-motion artifact (multi-shot k-space corruption) at a controlled severity, for QC.

    Simulates subject motion during acquisition: the volume is acquired as several "shots", each
    at a slightly different rigid pose, recombined in k-space. TorchIO's ``RandomMotion`` composes
    ``num_transforms`` rigid transforms (rotation up to ``degrees``, translation up to
    ``translation`` mm).

    Severity contract: we sample ``s`` in [0, 1] and scale both rotation and translation by it
    (degrees = s * deg_max, translation = s * tr_max), keeping ``num_transforms`` fixed, so ``s``
    is one monotone knob on the motion magnitude. Degenerate ranges make the realised values exact;
    ``s`` is the QC label. The ordinal magnitude transfers to real data but absolute mm does not,
    so the honest target is ``s``.

    :param deg_max: rotation (deg) at s = 1; realised rotation = s * deg_max.
    :param tr_max: translation (mm) at s = 1; realised translation = s * tr_max.
    :param num_transforms: number of rigid "shots" recombined in k-space (fixed).
    :param prob: per-sample probability of applying motion (severity is 0 otherwise).
    :param seed: optional seed for the severity RNG.
    """

    name = "motion"

    def __init__(self, deg_max=10., tr_max=10., num_transforms=2, prob=0.5, seed=None):
        super(MotionArtefact, self).__init__(prob=prob, seed=seed)
        self.deg_max = deg_max
        self.tr_max = tr_max
        self.num_transforms = num_transforms

    def _to_param(self, severity):
        # one severity scalar to (rotation deg, translation mm); both scale with s
        return severity * self.deg_max, severity * self.tr_max

    def _apply(self, image_np, degrees, translation):
        import torch
        import torchio as tio
        transform = tio.RandomMotion(
            degrees=(degrees, degrees), # degenerate range, realised value is exact
            translation=(translation, translation),
            num_transforms=self.num_transforms,
            image_interpolation="linear",
        )
        x = torch.as_tensor(image_np[None].astype("float32"))   # (W,H,D) to (C,W,H,D)
        out = transform(x)
        return out[0].numpy()   # strip the channel axis


class GhostingArtefact(ArtefactTransform):
    """Ghosting artifact (periodic replicas along the phase-encode axis) at a controlled severity.

    Motion/flow during phase encoding produces faint, evenly-spaced copies ("ghosts") of the
    anatomy shifted along the phase-encode axis. TorchIO's ``RandomGhosting`` adds ``num_ghosts``
    replicas at a given ``intensity`` along a chosen ``axis``.

    Severity contract: we sample ``s`` in [0, 1] and map it to the ghost intensity
    (intensity = s * intensity_max), keeping ``num_ghosts`` fixed; ``s`` is the QC label. The
    phase-encode axis is fixed per instance. Degenerate ranges make the realised intensity exact.

    :param intensity_max: ghost intensity at s = 1; realised intensity = s * intensity_max.
    :param num_ghosts: number of ghost replicas (fixed).
    :param axis: phase-encode axis along which the ghosts repeat (0, 1 or 2).
    :param prob: per-sample probability of applying ghosting (severity is 0 otherwise).
    :param seed: optional seed for the severity RNG.
    """

    name = "ghosting"

    def __init__(self, intensity_max=0.8, num_ghosts=4, axis=0, prob=0.5, seed=None):
        super(GhostingArtefact, self).__init__(prob=prob, seed=seed)
        self.intensity_max = intensity_max
        self.num_ghosts = num_ghosts
        self.axis = axis

    def _to_param(self, severity):
        return (severity * self.intensity_max,)

    def _apply(self, image_np, intensity):
        import torch
        import torchio as tio
        transform = tio.RandomGhosting(
            num_ghosts=(self.num_ghosts, self.num_ghosts),
            axes=(self.axis,),
            intensity=(intensity, intensity),   # degenerate range, realised value is exact
            restore=0.02,
        )
        x = torch.as_tensor(image_np[None].astype("float32"))   # (W,H,D) to (C,W,H,D)
        out = transform(x)
        return out[0].numpy()
