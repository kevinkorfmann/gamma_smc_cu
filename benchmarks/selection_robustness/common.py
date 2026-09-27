"""Auditable coordinate, provenance and interval-scoring helpers."""
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def seed_for(*parts):
    """Stable independent seed namespace, unaffected by Python hash randomization."""
    h = hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).digest()
    return 1 + int.from_bytes(h[:8], "big") % (2**31 - 2)


def interval_means(left, right, values, edges):
    """Exact overlap-weighted means and covered fraction; gaps are not filled."""
    left, right, values, edges = map(np.asarray, (left, right, values, edges))
    if values.ndim == 1:
        values = values[:, None]
    if (len(left) != len(right) or len(left) != len(values)
            or np.any(right <= left) or np.any(left[1:] < right[:-1])
            or np.any(np.diff(edges) <= 0) or not np.all(np.isfinite(values))):
        raise ValueError("Invalid, overlapping, nonfinite or unsorted intervals")
    total = np.zeros((len(edges) - 1, values.shape[1]))
    covered = np.zeros(len(edges) - 1)
    for lo, hi, value in zip(left, right, values):
        a = max(0, np.searchsorted(edges, lo, side="right") - 1)
        b = min(len(covered), np.searchsorted(edges, hi, side="left"))
        for j in range(a, b):
            weight = max(0, min(hi, edges[j + 1]) - max(lo, edges[j]))
            covered[j] += weight
            total[j] += weight * value
    out = np.full_like(total, np.nan)
    np.divide(total, covered[:, None], out=out, where=covered[:, None] > 0)
    return out, covered / np.diff(edges)


def snp_cells(positions):
    """Nearest-SNP projection with no extrapolation beyond first/last SNP."""
    p = np.asarray(positions)
    if len(p) < 2 or np.any(np.diff(p) <= 0):
        raise ValueError("At least two strictly increasing SNP coordinates required")
    bounds = np.concatenate(([p[0]], (p[:-1] + p[1:]) / 2, [p[-1]]))
    return bounds[:-1], bounds[1:]


def extract_inputs(ts):
    """Same exact binary sites for all methods; no VCF rounding or extra mutations."""
    positions, columns = [], []
    dropped = {"multiallelic_or_missing": 0, "monomorphic": 0}
    for var in ts.variants():
        g = var.genotypes
        if len(var.alleles) != 2 or np.any(g < 0) or np.any(g > 1):
            dropped["multiallelic_or_missing"] += 1
            continue
        if g.min() == g.max():
            dropped["monomorphic"] += 1
            continue
        positions.append(var.site.position)
        columns.append(g)
    if not columns:
        raise ValueError("No usable segregating sites")
    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.uint8), np.array(positions), dropped


def truth_on_grid(ts, pairs, edges):
    left, right, values = [], [], []
    samples = ts.samples()
    for tree in ts.trees():
        if tree.interval.right <= edges[0] or tree.interval.left >= edges[-1]:
            continue
        vals = [tree.tmrca(int(samples[a]), int(samples[b])) for a, b in pairs]
        if not np.all(np.isfinite(vals)) or min(vals) <= 0:
            raise ValueError("Incomplete genealogy or invalid pair/time")
        left.append(tree.interval.left)
        right.append(tree.interval.right)
        values.append(vals)
    truth, coverage = interval_means(left, right, values, edges)
    if not np.allclose(coverage, 1):
        raise ValueError("Truth does not cover scoring grid")
    return truth
