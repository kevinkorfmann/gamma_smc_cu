"""Path resolution helpers for the stdpopsim parity benchmark harness."""

from __future__ import annotations

import os
import shutil


def _first_existing(*candidates: str | None) -> str | None:
    for candidate in candidates:
        if not candidate:
            continue
        expanded = os.path.expanduser(candidate)
        if os.path.exists(expanded):
            return expanded
    return None


def resolve_gamma_smc_bin(here: str) -> str:
    path = _first_existing(
        os.environ.get("GAMMA_SMC_BIN"),
        os.path.join(here, "gamma_smc", "bin", "gamma_smc"),
        shutil.which("gamma_smc"),
    )
    if path is None:
        raise FileNotFoundError(
            "Could not locate gamma_smc. Set GAMMA_SMC_BIN or populate "
            f"{os.path.join(here, 'gamma_smc', 'bin', 'gamma_smc')}."
        )
    return path


def resolve_flow_field_path(here: str) -> str:
    path = _first_existing(
        os.environ.get("TMRCA_CU_FLOW_FIELD"),
        os.path.join(here, "gamma_smc", "resources", "default_flow_field.txt"),
        "/vast/projects/smathi/cohort/kkor/tmrca.cu/default_flow_field.txt",
    )
    if path is None:
        raise FileNotFoundError(
            "Could not locate default_flow_field.txt. Set TMRCA_CU_FLOW_FIELD "
            "or place the flow field under benchmarks/test_suite_stdpopsim/"
            "gamma_smc/resources/."
        )
    return path
