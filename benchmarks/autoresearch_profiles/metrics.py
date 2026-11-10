"""Fixed physical-coordinate evaluation for exploratory simulation matching."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def seed_for(*parts):
    payload = json.dumps(parts, separators=(',', ':')).encode()
    return 1 + int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % (2**31 - 2)


def bin_piecewise(boundaries, values, edges):
    """Exact integration of a contiguous piecewise constant function; no fill."""
    boundaries, values, edges = map(np.asarray, (boundaries, values, edges))
    if (boundaries.ndim != 1 or values.ndim != 1 or edges.ndim != 1
            or len(boundaries) != len(values) + 1 or len(edges) < 2
            or np.any(np.diff(boundaries) <= 0) or np.any(np.diff(edges) <= 0)
            or not all(np.all(np.isfinite(v)) for v in [boundaries, values, edges])
            or edges[0] < boundaries[0] or edges[-1] > boundaries[-1]):
        raise ValueError('Invalid intervals or incomplete coverage')
    integral = np.r_[0, np.cumsum(np.diff(boundaries) * values)]
    return np.diff(np.interp(edges, boundaries, integral)) / np.diff(edges)


def bin_snps(positions, values, edges):
    positions, values = np.asarray(positions), np.asarray(values)
    if len(positions) < 2 or len(positions) != len(values):
        raise ValueError('Invalid SNP arrays')
    bounds = np.r_[positions[0], (positions[:-1] + positions[1:]) / 2, positions[-1]]
    return bin_piecewise(bounds, values, edges)


def loss(prediction, observation, mask, focal):
    """0.75 spatial log MSE + 0.25 central-200kb log MSE, square rooted.

    No optimized shift, scale, smoothing, selected seed, or target-dependent mask.
    All bins must be finite and positive, including the held-out ones.
    """
    p, y = np.asarray(prediction), np.asarray(observation)
    mask, focal = np.asarray(mask, dtype=bool), np.asarray(focal, dtype=bool)
    if (p.shape != y.shape or p.ndim != 1 or mask.shape != p.shape
            or focal.shape != p.shape or not np.any(mask & focal)
            or not np.all(np.isfinite(p)) or not np.all(np.isfinite(y))
            or np.any(p <= 0) or np.any(y <= 0)):
        raise ValueError('Invalid profile or evaluation mask')
    error = np.log(p / y) ** 2
    return float(np.sqrt(.75 * error[mask].mean() + .25 * error[mask & focal].mean()))


def scores(prediction, target):
    return {name: loss(prediction, target['observed'], target[name], target['focal'])
            for name in ['train', 'holdout', 'all_bins']}
