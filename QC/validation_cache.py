"""Whether validate_rs / validate_tm can keep the preprocessed validation set in memory across checkpoints.

The preprocessing is the same for every checkpoint of a sweep, and on anisotropic images the resampling to
1 mm dominates the cost (~5 s per volume on CPU against a fraction of a second on the GPU). Keeping the
prepared volumes pays it once per sweep instead of once per checkpoint. The cost is memory: one float32 window
per image, 160^3 x 8 B = 33 MB (preprocess returns float64), known before anything is read, so the decision is taken up front.

The budget is the tightest of what this job may use: the cgroup limit SLURM enforces (v1 or v2), SLURM's own
request (--mem or --mem-per-cpu x cpus), and MemAvailable, minus what the job already holds. Half of it goes to
the cache; the rest is TensorFlow, the model and the volume being read.
"""

import os

import numpy as np

UNKNOWN_BUDGET_CAP = 2e9    # bytes cached when no limit can be read at all


def _read_int(path):
    try:
        with open(path) as f:
            v = f.read().strip()
        return None if v in ('', 'max') else int(v)
    except (OSError, ValueError):
        return None


def _cgroup_limit_and_usage():
    """(limit, usage) in bytes of this process's memory cgroup, or (None, None)."""
    try:
        with open('/proc/self/cgroup') as f:
            lines = f.read().splitlines()
    except OSError:
        return None, None
    for line in lines:
        _, ctrl, path = line.split(':', 2)
        if ctrl == '':                                   # cgroup v2
            base = os.path.join('/sys/fs/cgroup', path.lstrip('/'))
            return _read_int(os.path.join(base, 'memory.max')), _read_int(os.path.join(base, 'memory.current'))
        if 'memory' in ctrl.split(','):                  # cgroup v1
            base = os.path.join('/sys/fs/cgroup/memory', path.lstrip('/'))
            limit = _read_int(os.path.join(base, 'memory.limit_in_bytes'))
            if limit is not None and limit > 2 ** 60:    # v1 writes a huge number for "no limit"
                limit = None
            return limit, _read_int(os.path.join(base, 'memory.usage_in_bytes'))
    return None, None


def _slurm_request():
    mb = os.environ.get('SLURM_MEM_PER_NODE')
    if mb is None and os.environ.get('SLURM_MEM_PER_CPU'):
        cpus = os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_CPUS_ON_NODE') or '1'
        mb = int(os.environ['SLURM_MEM_PER_CPU']) * int(cpus)
    return None if mb is None else int(mb) * 2 ** 20


def _mem_available():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def memory_budget():
    """Bytes this job can still use, and a one-line description of where the number came from."""
    limit, usage = _cgroup_limit_and_usage()
    candidates = {}
    if limit is not None:
        candidates['cgroup'] = limit - (usage or 0)
    request = _slurm_request()
    if request is not None:
        candidates['slurm'] = request - (usage or 0)
    available = _mem_available()
    if available is not None:
        candidates['MemAvailable'] = available
    if not candidates:
        return None, 'no memory limit readable'
    source = min(candidates, key=candidates.get)
    return candidates[source], ', '.join('%s %.1f GB' % (k, v / 1e9) for k, v in candidates.items())


def complete(path_csv, n_images):
    """A results csv counts as done only with one row per image: a checkpoint killed halfway (OOM, time limit)
    leaves a header and a few rows, and resuming must redo it rather than skip it."""
    if not os.path.isfile(path_csv):
        return False
    with open(path_csv) as f:
        n_rows = sum(1 for line in f if line.strip()) - 1
    return n_rows >= n_images


def decide(mode, n_images, cropping):
    """True when the prepared set should be cached. mode is 'on', 'off' or 'auto'."""
    assert mode in ('on', 'off', 'auto'), "cache must be 'on', 'off' or 'auto', got %r" % mode
    if mode != 'auto':
        print('cache: %s (forced)' % mode)
        return mode == 'on'
    if cropping is None:
        print('cache: off (no cropping window, so the size of a prepared volume is not known in advance)')
        return False
    need = n_images * int(np.prod(cropping)) * 8
    budget, where = memory_budget()
    cap = UNKNOWN_BUDGET_CAP if budget is None else 0.5 * budget
    on = need <= cap
    print('cache: %s  (%d images x %s float64 = %.2f GB; budget %s; half of it is the cap)'
          % ('on' if on else 'off', n_images, 'x'.join(str(c) for c in cropping), need / 1e9, where))
    return on
