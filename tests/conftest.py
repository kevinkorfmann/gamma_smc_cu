"""Pytest config for `tests/betty/` — reproducibility tests that run
gamma_smc_cu against the real 1000G cache on betty.

These tests are skipped on any machine that isn't betty. Run on betty
with:

    ssh betty
    cd /vast/projects/smathi/cohort/kkor/tmrca.cu
    pixi run pytest tests/betty/ -v

To override the skip (e.g., to run locally with your own cache path):

    export GAMMA_SMC_CU_CACHE_DIR=/path/to/parsed
    export GAMMA_SMC_CU_RESULTS_DIR=/path/to/results
    export GAMMA_SMC_CU_SAMPLES_TXT=/path/to/samples.txt
    pixi run pytest tests/betty/ -v
"""
import os
from pathlib import Path

import pytest


DEFAULT_CACHE = Path("/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cache/parsed")
DEFAULT_RESULTS = Path("/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/results")
DEFAULT_SAMPLES = Path("/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/data/samples.txt")


def _path_or_skip(env: str, default: Path, kind: str) -> Path:
    p = Path(os.environ.get(env, default))
    if not p.exists():
        pytest.skip(f"{kind} not available (expected {p}; override via env var {env})")
    return p


@pytest.fixture(scope="session")
def cache_dir() -> Path:
    """Chromosome NPZ cache (GRCh38, bitpacked G + positions + sample_ids)."""
    return _path_or_skip("GAMMA_SMC_CU_CACHE_DIR", DEFAULT_CACHE, "1000G chr NPZ cache")


@pytest.fixture(scope="session")
def results_dir() -> Path:
    """Genome-wide scan outputs: results/chr{N}/{POP}.csv and .npz."""
    return _path_or_skip("GAMMA_SMC_CU_RESULTS_DIR", DEFAULT_RESULTS, "production results dir")


@pytest.fixture(scope="session")
def samples_txt() -> Path:
    """Sample->population map from 1000G."""
    return _path_or_skip("GAMMA_SMC_CU_SAMPLES_TXT", DEFAULT_SAMPLES, "samples.txt")


@pytest.fixture(scope="session")
def pop_map(samples_txt):
    """Map sample_id -> population code (first superpop field)."""
    m = {}
    with open(samples_txt) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                m[parts[0]] = parts[1]
    return m


def pytest_collection_modifyitems(config, items):
    """Auto-skip betty tests if running off-cluster (cache dir missing).

    Use `-m betty` to force-include them; `-m 'not betty'` is default-ish.
    """
    if not DEFAULT_CACHE.exists() and "GAMMA_SMC_CU_CACHE_DIR" not in os.environ:
        skip_betty = pytest.mark.skip(
            reason="betty reproducibility tests: no 1000G cache on this host"
        )
        for item in items:
            if "betty" in item.keywords:
                item.add_marker(skip_betty)
