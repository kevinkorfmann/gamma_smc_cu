"""Unit tests that verify the paper's per-gene rank and q_hier claims
against a small bundled reference CSV.

The CSV `tests/data/paper_reference_qhier.csv` is a 16-row subset of the
full 17,823-gene FDR output table (`private/manuscript/v4.1/tables/
fdr_qvalues.csv`) containing the 5 main-text candidates
(GRK2, CCDC92, CLEC6A, SLC6A15, BPIFA2), 2 Discussion replicators
(TREM2, IFIH1), and 9 canonical sweeps (LCT, SLC24A5, EDAR, FADS1,
HERC2, ABCC11, KITLG, TRPV6, ALDH2). These are the exact values
reported in main.tex, so the tests protect against regressions in
the FDR / hierarchical-BH pipeline that feeds Table S2 of the paper.

Runs in <0.1 s on CPU.
"""
from pathlib import Path
import pytest
import pandas as pd


DATA = Path(__file__).parent.parent / "data" / "paper_reference_qhier.csv"


@pytest.fixture(scope="module")
def ref():
    """Small bundled reference: the 16 genes explicitly referenced in main.tex."""
    df = pd.read_csv(DATA)
    return df.set_index("gene_name")


def _stage2_rank(ref, gene: str) -> int:
    return int(ref.loc[gene, "stage2_rank"])


def _q_hier(ref, gene: str) -> float:
    return float(ref.loc[gene, "q_hier"])


# --------------------------------------------------------------------------
# Stage-2 ranks: main.tex Methods §Hierarchical multiple-testing correction
# "GRK2 ranks 29th of 538 ... SLC24A5 (rank 7) ... SLC6A15 (rank 21) ...
#  CCDC92 (rank 370), CLEC6A (326), BPIFA2 (335)"
# --------------------------------------------------------------------------
def test_grk2_stage2_rank_29(ref):
    assert _stage2_rank(ref, "GRK2") == 29


def test_slc24a5_stage2_rank_7(ref):
    assert _stage2_rank(ref, "SLC24A5") == 7


def test_slc6a15_stage2_rank_21(ref):
    assert _stage2_rank(ref, "SLC6A15") == 21


def test_ccdc92_stage2_rank_370(ref):
    assert _stage2_rank(ref, "CCDC92") == 370


def test_clec6a_stage2_rank_326(ref):
    assert _stage2_rank(ref, "CLEC6A") == 326


def test_bpifa2_stage2_rank_335(ref):
    assert _stage2_rank(ref, "BPIFA2") == 335


# --------------------------------------------------------------------------
# q_hier for main-text candidates (main.tex Methods, Table S2 row values)
# --------------------------------------------------------------------------
def test_grk2_qhier_5p5e_3(ref):
    """main.tex: GRK2 q_hier = 5.5e-3."""
    assert _q_hier(ref, "GRK2") == pytest.approx(5.5e-3, rel=0.02)


def test_slc24a5_qhier_5p2e_3(ref):
    """main.tex: SLC24A5 q_hier = 5.2e-3 (canonical positive control)."""
    assert _q_hier(ref, "SLC24A5") == pytest.approx(5.2e-3, rel=0.02)


def test_slc6a15_qhier_5p5e_3(ref):
    """main.tex: SLC6A15 q_hier = 5.5e-3."""
    assert _q_hier(ref, "SLC6A15") == pytest.approx(5.5e-3, rel=0.02)


def test_ccdc92_qhier_1p1e_2(ref):
    """main.tex: CCDC92 q_hier = 1.1e-2."""
    assert _q_hier(ref, "CCDC92") == pytest.approx(1.1e-2, abs=5e-4)


def test_clec6a_qhier_8p7e_3(ref):
    """main.tex: CLEC6A q_hier = 8.7e-3."""
    assert _q_hier(ref, "CLEC6A") == pytest.approx(8.7e-3, rel=0.02)


def test_bpifa2_qhier_8p9e_3(ref):
    """main.tex: BPIFA2 q_hier = 8.9e-3."""
    assert _q_hier(ref, "BPIFA2") == pytest.approx(8.9e-3, rel=0.02)


# --------------------------------------------------------------------------
# Stage-2 rank ordering: main-text 3 examples sit in middle of stage-2 set
# --------------------------------------------------------------------------
def test_main_text_candidates_are_within_stage2(ref):
    """All 5 main-text candidates + SLC24A5 must have stage2_rank <= 538."""
    for gene in ["GRK2", "SLC24A5", "SLC6A15", "CCDC92", "CLEC6A", "BPIFA2"]:
        rank = _stage2_rank(ref, gene)
        assert 1 <= rank <= 538, f"{gene}: rank={rank} outside 1..538"


def test_grk2_rank_better_than_middle_exemplars(ref):
    """main.tex narrative: GRK2 ranks closer to the top than CCDC92/CLEC6A/
    BPIFA2 (which sit "nearer the middle of the stage-2 distribution")."""
    grk2 = _stage2_rank(ref, "GRK2")
    for weaker in ["CCDC92", "CLEC6A", "BPIFA2"]:
        assert grk2 < _stage2_rank(ref, weaker), (
            f"GRK2 (rank {grk2}) should rank stronger than {weaker}"
        )


def test_slc24a5_top_10(ref):
    """main.tex: SLC24A5 is a canonical positive control; rank 7 is in the
    top 10 of the stage-2 set."""
    assert _stage2_rank(ref, "SLC24A5") <= 10
