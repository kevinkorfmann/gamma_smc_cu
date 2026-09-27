"""Portable chromosome caches; extract NPZ members once for actual memory mapping."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_chromosome(cache_dir, chr_num):
    root = Path(cache_dir)
    if (root / "parsed").is_dir():
        root /= "parsed"
    folder = root / f"chr{chr_num}"
    if (folder / "READY.json").is_file():
        return {k: np.load(folder / f"{k}.npy", mmap_mode="r" if k != "sample_ids" else None,
                           allow_pickle=k == "sample_ids") for k in ("G", "positions", "sample_ids")}
    with np.load(root / f"chr{chr_num}.npz", allow_pickle=True) as d:
        return {k: d[k] for k in ("G", "positions", "sample_ids")}


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        members = {}
        with zipfile.ZipFile(source) as z:
            for key in ("G", "positions", "sample_ids"):
                name = key + ".npy"
                h = hashlib.sha256()
                with z.open(name) as inp, open(temp / name, "wb") as out:
                    for block in iter(lambda: inp.read(8 << 20), b""):
                        out.write(block)
                        h.update(block)
                members[name] = {"sha256": h.hexdigest(), "bytes": (temp / name).stat().st_size}
        G = np.load(temp / "G.npy", mmap_mode="r")
        positions = np.load(temp / "positions.npy", mmap_mode="r")
        sample_ids = np.load(temp / "sample_ids.npy", allow_pickle=True)
        if G.ndim != 2 or G.shape != (2 * len(sample_ids), len(positions)):
            raise ValueError(f"Inconsistent haplotype/sample/site shapes: {G.shape}")
        if len(set(map(str, sample_ids))) != len(sample_ids):
            raise ValueError("Duplicate sample IDs")
        if not len(positions) or positions[0] < 0 or np.any(np.diff(positions) <= 0):
            raise ValueError("Positions must be nonnegative and strictly increasing")
        for start in range(0, G.shape[0], 32):
            chunk = G[start:start + 32]
            if np.any((chunk != 0) & (chunk != 1)):
                raise ValueError("Genotypes must be binary, complete phased haplotypes")
        record = {"schema": 1, "source_name": source.name, "source_bytes": source.stat().st_size,
                  "source_sha256": sha256(source), "members": members,
                  "shape": list(G.shape), "dtype": str(G.dtype), "samples": len(sample_ids),
                  "first_position": int(positions[0]), "last_position": int(positions[-1]),
                  "coordinate_convention": "unchanged from source NPZ"}
        (temp / "READY.json").write_text(json.dumps(record, indent=2) + "\n")
        os.rename(temp, output)
        print(json.dumps(record), flush=True)
    except BaseException:
        shutil.rmtree(temp)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    prepare(a.source, a.output)


if __name__ == "__main__":
    main()
