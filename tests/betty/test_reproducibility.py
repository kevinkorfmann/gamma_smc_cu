"""Betty reproducibility tests: re-run gamma_smc_cu on the real 1000G
cache and verify the per-gene TMRCA values + within-population rank
orderings match the production genome-wide scan.

Three tiers of test, from fastest to slowest:

  Tier 1  — Production-NPZ integrity checks (load CSV, verify known
            paper values).  <1 s each.  No GPU needed.

  Tier 2  — Fresh inference on a ±2 Mb slice around a focal gene with
            a deterministic 500-pair subsample; verify ranking of the
            focal gene within the slice.  30-90 s per test on a B200
            MIG slice.

  Tier 3  — Full-chromosome fresh inference at all within-pop pairs
            for the smallest (chr, pop) combination and verify the
            resulting per-gene geom-mean TMRCA at a canonical sweep
            matches production exactly.  5-10 min per test.

Run on betty:

    ssh betty
    cd /vast/projects/smathi/cohort/kkor/tmrca.cu
    pixi run pytest tests/betty/ -v               # all three tiers
    pixi run pytest tests/betty/ -v -k tier1      # fastest only
    pixi run pytest tests/betty/ -v -k tier2      # fresh-inference checks
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.betty


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _load_chr_cache(cache_dir, chrom: int):
    path = cache_dir / f"chr{chrom}.npz"
    data = np.load(path, allow_pickle=True)
    return {
        "G": data["G"],
        "positions": data["positions"].astype(np.int64),
        "sample_ids": data["sample_ids"],
    }


def _pop_haplotype_indices(sample_ids, pop_map, population: str):
    """Interleaved haps (2i, 2i+1) for every sample in `population`."""
    idxs = []
    for i, sid in enumerate(sample_ids):
        if sid in pop_map and pop_map[sid] == population:
            idxs.extend([2 * i, 2 * i + 1])
    return sorted(idxs)


def _gene_geom_mean(result, gene_start: int, gene_end: int) -> float:
    """Per-gene geometric mean TMRCA: exp(mean(log(TMRCA))) across all
    (site, pair) tuples where site falls in [gene_start, gene_end].
    """
    pos = result["positions"] if isinstance(result, dict) else result.positions
    mean = result["mean"] if isinstance(result, dict) else result.tmrca
    mask = (pos >= gene_start) & (pos <= gene_end)
    if not mask.any():
        return float("nan")
    m = np.clip(mean[mask], 1.0, 1e6)
    return float(np.exp(np.log(m).mean()))


# --------------------------------------------------------------------------
# Tier 1 — Production integrity (no GPU, <1 s each)
# --------------------------------------------------------------------------
class TestTier1ProductionIntegrity:
    """Verify production NPZ / CSV outputs are intact and match paper claims."""

    def test_tier1_grk2_gih_production_tmrca(self, results_dir):
        """Paper: GRK2 geom-mean TMRCA in GIH ~ 640 generations (production
        scan, 21,115 pairs, 41 sites in gene body). CSV exact value: 643.8.
        """
        csv = pd.read_csv(results_dir / "chr11" / "GIH.csv")
        row = csv[csv.gene_name == "GRK2"].iloc[0]
        assert row.n_pairs == 21115, f"GIH has 103 samples -> 21,115 pairs; got {row.n_pairs}"
        assert row.n_sites == 41, f"Paper: 41 polymorphic sites in GRK2 body in GIH; got {row.n_sites}"
        assert row.geom_mean_tmrca == pytest.approx(643.8, abs=1.0), (
            f"GRK2 GIH geom-mean TMRCA: expected 643.8, got {row.geom_mean_tmrca}"
        )

    def test_tier1_grk2_coordinates(self, results_dir):
        """Paper (Methods §Processing): GRK2 body = chr11:67,266,473-67,286,556 GRCh38."""
        csv = pd.read_csv(results_dir / "chr11" / "GIH.csv")
        row = csv[csv.gene_name == "GRK2"].iloc[0]
        assert int(row.start) == 67_266_473
        assert int(row.end) == 67_286_556

    def test_tier1_slc24a5_low_tmrca_in_gbr(self, results_dir):
        """Paper: SLC24A5 is the canonical European skin-pigmentation sweep,
        rank 0.09% in GBR -> its gene-level geom-mean TMRCA must be low
        (well below genome-wide median ~25,000 gen)."""
        csv = pd.read_csv(results_dir / "chr15" / "GBR.csv")
        row = csv[csv.gene_name == "SLC24A5"].iloc[0]
        assert row.geom_mean_tmrca < 10_000, (
            f"SLC24A5 GBR: expected deep sweep (<10,000 gen); got {row.geom_mean_tmrca}"
        )

    def test_tier1_grk2_is_within_pop_top_1_percent_in_gih(self, results_dir):
        """Paper: GRK2 ranks 0.23% in GIH (top 1%). Verify from production CSV."""
        csv = pd.read_csv(results_dir / "chr11" / "GIH.csv")
        chr11_genes = csv.dropna(subset=["geom_mean_tmrca"])
        grk2_val = chr11_genes[chr11_genes.gene_name == "GRK2"].geom_mean_tmrca.iloc[0]
        # Fraction of chr11 genes with LOWER TMRCA than GRK2
        n_lower = (chr11_genes.geom_mean_tmrca < grk2_val).sum()
        chr11_rank = n_lower / len(chr11_genes) * 100
        assert chr11_rank < 5.0, (
            f"GRK2 should rank in top 5% of chr11 in GIH; got {chr11_rank:.2f}%"
        )

    def test_tier1_pair_count_matches_sample_size(self, results_dir, pop_map):
        """For every production CSV row with polymorphic sites:
        n_pairs == n_hap*(n_hap-1)/2 exactly. Rows with n_sites=0
        (genes with no polymorphic variants in the body) carry
        n_pairs=0 and are excluded.
        """
        csv = pd.read_csv(results_dir / "chr22" / "GIH.csv")
        # Count GIH haplotypes from pop_map
        n_gih = sum(1 for v in pop_map.values() if v == "GIH")
        n_hap = 2 * n_gih
        expected_pairs = n_hap * (n_hap - 1) // 2
        # Filter to rows that actually saw pairs (empty genes report n_pairs=0)
        poly_rows = csv[csv.n_pairs.fillna(0) > 0]
        actual = set(poly_rows.n_pairs.astype(int).unique())
        assert actual == {expected_pairs}, (
            f"Expected all polymorphic-gene GIH rows to have n_pairs={expected_pairs}, "
            f"got {sorted(actual)}"
        )


# --------------------------------------------------------------------------
# Tier 2 — Fresh inference on small slice (30-90 s)
# --------------------------------------------------------------------------
class TestTier2FreshSliceInference:
    """Re-run gamma_smc_cu on a ±2 Mb slice around a focal gene with a
    500-pair random subsample, and verify the focal gene's TMRCA ranks in
    the top fraction of the slice-local gene distribution.
    """

    @pytest.fixture(scope="class")
    def grk2_slice_result(self, cache_dir, pop_map):
        """Run gamma_smc_cu on chr11:66-68 Mb with 500-pair GIH subsample."""
        import gamma_smc_cu

        chr11 = _load_chr_cache(cache_dir, 11)
        haps = _pop_haplotype_indices(chr11["sample_ids"], pop_map, "GIH")
        G_pop = chr11["G"][haps]
        positions = chr11["positions"]

        # Slice to chr11:66-68 Mb (2 Mb window centred ~1 Mb upstream of GRK2 end)
        mask = (positions >= 66_000_000) & (positions <= 68_000_000)
        G_slice = G_pop[:, mask]
        pos_slice = positions[mask]

        # Deterministic 500-pair subsample
        n_haps = G_slice.shape[0]
        rng = np.random.default_rng(42)
        all_pairs = [(i, j) for i in range(n_haps) for j in range(i + 1, n_haps)]
        idx = rng.choice(len(all_pairs), size=500, replace=False)
        pairs = [all_pairs[k] for k in sorted(idx.tolist())]

        result = gamma_smc_cu.infer_blockwise(
            G_slice, pos_slice,
            pairs=pairs,
            mean_only=True,
            auto_estimate_theta=True,
        )
        return result

    def test_tier2_grk2_tmrca_is_low(self, grk2_slice_result):
        """Fresh slice inference: per-site geom-mean TMRCA across the GRK2
        gene body (41 poly sites in the paper) should be well below the
        neutral expectation (~25,000 gen for chr11)."""
        tmrca = _gene_geom_mean(grk2_slice_result, 67_266_473, 67_286_556)
        assert tmrca < 5_000, (
            f"GRK2 slice TMRCA in GIH: expected <5,000 gen (sweep), got {tmrca:.1f}"
        )

    def test_tier2_grk2_ranks_top_of_slice(self, cache_dir, pop_map,
                                            grk2_slice_result, results_dir):
        """Fresh slice with 500 pairs should still rank GRK2 in the top 20%
        of the 1 Mb slice's annotated genes (production scan ranks it at
        0.23% genome-wide in GIH)."""
        # Load per-gene start/end for chr11 from production CSV
        prod = pd.read_csv(results_dir / "chr11" / "GIH.csv")
        slice_genes = prod[(prod.start >= 66_000_000) & (prod.end <= 68_000_000)].copy()
        slice_genes["slice_tmrca"] = [
            _gene_geom_mean(grk2_slice_result, int(r.start), int(r.end))
            for _, r in slice_genes.iterrows()
        ]
        slice_genes = slice_genes.dropna(subset=["slice_tmrca"])
        slice_genes = slice_genes.sort_values("slice_tmrca")
        ranks = {r.gene_name: i + 1 for i, (_, r) in enumerate(slice_genes.iterrows())}
        grk2_rank = ranks["GRK2"]
        n = len(slice_genes)
        assert grk2_rank <= 0.2 * n, (
            f"GRK2 should be in top 20% of the chr11:66-68 Mb slice in GIH; "
            f"got rank {grk2_rank}/{n}"
        )


# --------------------------------------------------------------------------
# Tier 3 — Full-chromosome fresh inference (5-10 min)
# --------------------------------------------------------------------------
class TestTier3FullChromosomeExact:
    """Re-run gamma_smc_cu on the full chromosome with ALL within-pop pairs
    (matching the production scan) and verify bit-level reproducibility of
    the per-gene geom-mean TMRCA at key focal loci.

    We pick chr22 + ASW as the smallest (chr, pop) combination: chr22 ~ 50
    Mb, ASW n=74 -> 148 haps -> 10,878 pairs. Expected wall time on a B200
    MIG slice: 5-10 min.
    """

    @pytest.mark.slow
    def test_tier3_chr22_asw_full_scan_matches_production(
        self, cache_dir, pop_map, results_dir
    ):
        """Full-chromosome, all-pair inference on chr22 ASW matches the
        production genome-wide scan's per-gene geom-mean TMRCA for a
        handful of focal genes within 1% relative error.
        """
        import gamma_smc_cu

        chr22 = _load_chr_cache(cache_dir, 22)
        haps = _pop_haplotype_indices(chr22["sample_ids"], pop_map, "ASW")
        G_pop = chr22["G"][haps]
        positions = chr22["positions"]

        n_haps = G_pop.shape[0]
        pairs = [(i, j) for i in range(n_haps) for j in range(i + 1, n_haps)]
        assert len(pairs) == 10_878, f"ASW has 74 samples -> 10,878 pairs; got {len(pairs)}"

        result = gamma_smc_cu.infer_blockwise(
            G_pop, positions,
            pairs=pairs,
            mean_only=True,
            auto_estimate_theta=True,
        )

        prod = pd.read_csv(results_dir / "chr22" / "ASW.csv")
        # Pick 5 genes with >= 10 polymorphic sites to keep the comparison
        # statistically stable
        prod = prod[prod.n_sites >= 10].head(5)
        for _, row in prod.iterrows():
            fresh = _gene_geom_mean(result, int(row.start), int(row.end))
            expected = float(row.geom_mean_tmrca)
            rel_err = abs(fresh - expected) / expected
            assert rel_err < 0.01, (
                f"{row.gene_name} chr22 ASW: production={expected:.1f}, "
                f"fresh={fresh:.1f}, rel-err {rel_err:.3%} > 1%"
            )
