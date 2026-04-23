"""Unit tests for gamma_smc_cu.infer() top-level API."""

import numpy as np
import pytest
import msprime

import gamma_smc_cu
from gamma_smc_cu.infer import _estimate_scaled_params


@pytest.fixture(scope="module")
def ts():
    ts = msprime.sim_ancestry(
        samples=10, sequence_length=100_000,
        recombination_rate=1e-8, population_size=10000, random_seed=42)
    return msprime.sim_mutations(ts, rate=1.25e-8, random_seed=43)


@pytest.fixture(scope="module")
def genotype_data(ts):
    G = ts.genotype_matrix().T.astype(np.uint8)
    pos = np.array([v.position for v in ts.variants()], dtype=np.float64)
    return G, pos


class TestInferTreeSequence:
    def test_returns_dict(self, ts):
        result = gamma_smc_cu.infer(ts)
        assert isinstance(result, dict)
        assert "mean" in result

    def test_shape_all_pairs(self, ts):
        result = gamma_smc_cu.infer(ts)
        n = ts.num_samples
        S = ts.num_sites
        n_pairs = n * (n - 1) // 2
        assert result["mean"].shape == (S, n_pairs)

    def test_shape_specific_pairs(self, ts):
        pairs = [(0, 1), (2, 3)]
        result = gamma_smc_cu.infer(ts, pairs=pairs)
        assert result["mean"].shape[1] == 2

    def test_returns_pairs(self, ts):
        pairs = [(0, 1), (5, 3)]
        result = gamma_smc_cu.infer(ts, pairs=pairs)
        assert result["pairs"] == pairs

    def test_returns_positions(self, ts):
        result = gamma_smc_cu.infer(ts, pairs=[(0, 1)])
        assert len(result["positions"]) == ts.num_sites

    def test_values_positive(self, ts):
        result = gamma_smc_cu.infer(ts, pairs=[(0, 1)])
        assert np.all(result["mean"] > 0)

    def test_with_ci(self, ts):
        result = gamma_smc_cu.infer(ts, pairs=[(0, 1)], mean_only=False)
        assert "lower" in result
        assert "upper" in result
        assert np.all(result["lower"] <= result["mean"])
        assert np.all(result["upper"] >= result["mean"])


class TestInferGenotypeMatrix:
    def test_returns_dict(self, genotype_data):
        G, pos = genotype_data
        result = gamma_smc_cu.infer(G, pos)
        assert isinstance(result, dict)
        assert "mean" in result

    def test_shape_all_pairs(self, genotype_data):
        G, pos = genotype_data
        n = G.shape[0]
        S = G.shape[1]
        n_pairs = n * (n - 1) // 2
        result = gamma_smc_cu.infer(G, pos)
        assert result["mean"].shape == (S, n_pairs)

    def test_specific_pairs(self, genotype_data):
        G, pos = genotype_data
        result = gamma_smc_cu.infer(G, pos, pairs=[(0, 1)])
        assert result["mean"].shape[1] == 1

    def test_custom_params(self, genotype_data):
        G, pos = genotype_data
        result = gamma_smc_cu.infer(G, pos, mu=1e-8, rho=1e-8, Ne=5000,
                                 pairs=[(0, 1)])
        assert result["mean"].shape[1] == 1
        assert np.all(result["mean"] > 0)

    def test_rejects_mismatched_positions(self, genotype_data):
        G, pos = genotype_data
        with pytest.raises(ValueError, match="positions length"):
            gamma_smc_cu.infer(G, pos[:-1], pairs=[(0, 1)])


class TestAutoEstimateTheta:
    def test_estimate_scaled_params_uses_4ne_convention_for_rho(self):
        G = np.array(
            [
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [0, 0, 0, 1],
                [1, 0, 0, 1],
            ],
            dtype=np.uint8,
        )
        positions = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        Ne = 10_000.0
        mu = 2.0e-8
        rho = 5.0e-8

        eff_mu, eff_rho = _estimate_scaled_params(G, positions, mu, rho, Ne)

        seq_len = positions[-1] + 1.0
        pi_hat = 1.0 / seq_len
        ratio = rho / mu

        np.testing.assert_allclose(4.0 * Ne * eff_mu, pi_hat)
        np.testing.assert_allclose(4.0 * Ne * eff_rho, pi_hat * ratio)

    def test_estimate_scaled_params_counts_leading_span_like_gamma_smc(self):
        G = np.array(
            [
                [0, 1],
                [1, 1],
                [0, 0],
                [0, 1],
            ],
            dtype=np.uint8,
        )
        positions = np.array([100.0, 200.0], dtype=np.float64)
        Ne = 10_000.0
        mu = 2.0e-8
        rho = 5.0e-8

        eff_mu, eff_rho = _estimate_scaled_params(G, positions, mu, rho, Ne)

        pi_hat = 1.0 / (positions[-1] + 1.0)
        ratio = rho / mu

        np.testing.assert_allclose(4.0 * Ne * eff_mu, pi_hat)
        np.testing.assert_allclose(4.0 * Ne * eff_rho, pi_hat * ratio)


class TestInferSegregatingFilter:
    """gamma_smc_cu.infer() must drop monomorphic-in-subset sites before decoding,
    matching the original gamma_smc reference implementation (Schweiger &
    Durbin, 2023). Without the filter, the HMM kernel applies extra
    moment-match transition steps at non-informative sites, which is
    outside the algorithm's validated envelope.
    """

    def _pad_with_monomorphic(self, G, pos):
        """Insert synthetic all-0 and all-1 columns into a genotype matrix."""
        n, S = G.shape
        # Add one all-ref column at the start and one all-alt column at the end.
        all_ref = np.zeros((n, 1), dtype=G.dtype)
        all_alt = np.ones((n, 1), dtype=G.dtype)
        G_padded = np.concatenate([all_ref, G, all_alt], axis=1)
        pos_padded = np.concatenate(
            [[pos[0] - 1.0], pos, [pos[-1] + 1.0]]
        ).astype(np.float64)
        return G_padded, pos_padded

    def test_drops_monomorphic_columns(self, genotype_data):
        G, pos = genotype_data
        G_padded, pos_padded = self._pad_with_monomorphic(G, pos)
        assert G_padded.shape[1] == G.shape[1] + 2

        result = gamma_smc_cu.infer(G_padded, pos_padded, pairs=[(0, 1)])

        # Filtered result must not contain the two synthetic monomorphic sites.
        assert result["mean"].shape[0] == G.shape[1]
        assert len(result["positions"]) == G.shape[1]
        np.testing.assert_array_equal(result["positions"], pos)

    def test_filter_matches_unpadded(self, genotype_data):
        """Decoding padded+filtered input must match decoding the original
        segregating-only input exactly — the filter is the whole invariant."""
        G, pos = genotype_data
        G_padded, pos_padded = self._pad_with_monomorphic(G, pos)
        pairs = [(0, 1), (2, 3)]

        r_orig = gamma_smc_cu.infer(G, pos, pairs=pairs)
        r_padded = gamma_smc_cu.infer(G_padded, pos_padded, pairs=pairs)

        np.testing.assert_array_equal(r_orig["mean"], r_padded["mean"])
        np.testing.assert_array_equal(r_orig["positions"], r_padded["positions"])

    def test_segregating_input_is_noop(self, ts):
        """Simulated tree sequences only carry segregating sites, so the
        filter must be invisible — same shape as before the patch."""
        result = gamma_smc_cu.infer(ts, pairs=[(0, 1)])
        assert result["mean"].shape[0] == ts.num_sites
        assert len(result["positions"]) == ts.num_sites

    def test_blockwise_filter_matches_unpadded(self, genotype_data):
        G, pos = genotype_data
        G_padded, pos_padded = self._pad_with_monomorphic(G, pos)
        pairs = [(0, 1), (2, 3)]

        r_orig = gamma_smc_cu.infer_blockwise(
            G, pos, pairs=pairs,
            core_block_sites=G.shape[1], flank_sites=0,
        )
        r_padded = gamma_smc_cu.infer_blockwise(
            G_padded, pos_padded, pairs=pairs,
            core_block_sites=G.shape[1], flank_sites=0,
        )
        np.testing.assert_array_equal(r_orig["mean"], r_padded["mean"])
        np.testing.assert_array_equal(r_orig["positions"], r_padded["positions"])


class TestInferConsistency:
    def test_ts_and_matrix_agree(self, ts, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3), (4, 5)]
        r_ts = gamma_smc_cu.infer(ts, pairs=pairs)
        r_mat = gamma_smc_cu.infer(G, pos, pairs=pairs)
        np.testing.assert_allclose(r_ts["mean"], r_mat["mean"],
                                    rtol=1e-5, atol=1e-6)

    def test_correlates_with_truth(self, ts):
        pair = (0, 1)
        result = gamma_smc_cu.infer(ts, pairs=[pair])
        est = result["mean"][:, 0]
        pos = result["positions"]

        # Extract truth
        truth = np.empty(len(pos))
        tree_iter = ts.trees(); tree = next(tree_iter)
        for idx, p in enumerate(pos):
            while p >= tree.interval.right:
                tree = next(tree_iter)
            truth[idx] = tree.tmrca(*pair)

        from scipy.stats import pearsonr
        r = pearsonr(np.log(truth), np.log(np.maximum(est, 1)))[0]
        assert r > 0.5, f"Correlation too low: {r}"


class TestInferEdgeCases:
    def test_single_pair(self, ts):
        result = gamma_smc_cu.infer(ts, pairs=[(0, 1)])
        assert result["mean"].shape[1] == 1

    def test_empty_pairs(self, ts):
        result = gamma_smc_cu.infer(ts, pairs=[])
        assert result["mean"].shape[1] == 0


class TestInferModes:
    """All three output modes for gamma_smc_cu.infer()."""

    def test_mean_only(self, ts):
        pairs = [(0, 1), (2, 3)]
        result = gamma_smc_cu.infer(ts, pairs=pairs, mean_only=True)
        assert set(result.keys()) >= {"mean", "pairs", "positions"}
        assert "lower" not in result
        assert "upper" not in result
        assert "posterior_alpha" not in result
        assert "posterior_beta" not in result
        assert result["mean"].shape == (ts.num_sites, 2)
        assert result["mean"].dtype == np.float32
        assert np.all(result["mean"] > 0)

    def test_with_ci(self, ts):
        pairs = [(0, 1), (2, 3)]
        result = gamma_smc_cu.infer(ts, pairs=pairs, mean_only=False)
        assert {"mean", "lower", "upper"} <= set(result.keys())
        assert "posterior_alpha" not in result
        for key in ("lower", "mean", "upper"):
            assert result[key].shape == (ts.num_sites, 2)
            assert result[key].dtype == np.float32
        assert np.all(result["lower"] <= result["mean"] + 1e-3)
        assert np.all(result["mean"] <= result["upper"] + 1e-3)

    def test_with_posterior_mean_only(self, ts):
        pairs = [(0, 1), (2, 3)]
        result = gamma_smc_cu.infer(
            ts, pairs=pairs, mean_only=True, return_posterior=True
        )
        assert {"mean", "posterior_alpha", "posterior_beta"} <= set(result.keys())
        assert "lower" not in result
        assert "upper" not in result
        for key in ("posterior_alpha", "posterior_beta"):
            assert result[key].shape == (ts.num_sites, 2)
            assert result[key].dtype == np.float32
            assert np.all(result[key] > 0)
        # Posterior mean reconstructed from (alpha, beta) in scaled time
        # should match the returned `mean` array within float precision.
        Ne = 10000.0
        recon = (result["posterior_alpha"] / result["posterior_beta"]) * 2.0 * Ne
        np.testing.assert_allclose(recon, result["mean"], rtol=1e-4, atol=1e-2)

    def test_with_posterior_and_ci(self, ts):
        pairs = [(0, 1), (2, 3)]
        result = gamma_smc_cu.infer(
            ts, pairs=pairs, mean_only=False, return_posterior=True
        )
        expected = {"mean", "lower", "upper", "posterior_alpha", "posterior_beta"}
        assert expected <= set(result.keys())
        # All five tmrca arrays have the same shape
        for key in expected:
            assert result[key].shape == (ts.num_sites, 2)
        # Reconstructed posterior mean still matches
        Ne = 10000.0
        recon = (result["posterior_alpha"] / result["posterior_beta"]) * 2.0 * Ne
        np.testing.assert_allclose(recon, result["mean"], rtol=1e-4, atol=1e-2)

    def test_posterior_does_not_change_mean(self, ts):
        pairs = [(0, 1), (2, 3)]
        baseline = gamma_smc_cu.infer(ts, pairs=pairs, mean_only=True)
        with_post = gamma_smc_cu.infer(
            ts, pairs=pairs, mean_only=True, return_posterior=True
        )
        np.testing.assert_array_equal(baseline["mean"], with_post["mean"])


class TestInferBlockwise:
    def test_requires_explicit_pairs(self, genotype_data):
        G, pos = genotype_data
        with pytest.raises(ValueError, match="explicit pairs"):
            gamma_smc_cu.infer_blockwise(
                G,
                pos,
                flow_field_path="dummy-flow-field.txt",
            )

    def test_rejects_invalid_pair_batch_size(self, genotype_data):
        G, pos = genotype_data
        with pytest.raises(ValueError, match="pair_batch_size"):
            gamma_smc_cu.infer_blockwise(
                G,
                pos,
                pairs=[(0, 1)],
                flow_field_path="dummy-flow-field.txt",
                pair_batch_size=0,
            )

    def test_rejects_invalid_max_streams(self, genotype_data):
        G, pos = genotype_data
        with pytest.raises(ValueError, match="max_streams"):
            gamma_smc_cu.infer_blockwise(
                G,
                pos,
                pairs=[(0, 1)],
                flow_field_path="dummy-flow-field.txt",
                max_streams=0,
            )

    def test_passes_blockwise_args(self, monkeypatch, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3)]
        calls = {}

        class FakeFlowContext:
            def __init__(self, G, positions, Ne, mu, rho, flow_field_path, cache_steps):
                calls["init"] = {
                    "shape": G.shape,
                    "positions": len(positions),
                    "Ne": Ne,
                    "mu": mu,
                    "rho": rho,
                    "flow_field_path": flow_field_path,
                    "cache_steps": cache_steps,
                }

            def run_fb_blockwise(
                self,
                pairs,
                core_block_sites,
                flank_sites,
                pair_batch_size,
                max_streams,
                mean_only,
                return_posterior=False,
            ):
                calls["run"] = {
                    "pairs": list(pairs),
                    "core_block_sites": core_block_sites,
                    "flank_sites": flank_sites,
                    "pair_batch_size": pair_batch_size,
                    "max_streams": max_streams,
                    "mean_only": mean_only,
                    "return_posterior": return_posterior,
                }
                return {
                    "mean": np.ones((len(pos), len(pairs)), dtype=np.float32),
                    "blocks": np.array([[0, len(pos), 0, len(pos)]], dtype=np.int32),
                }

        monkeypatch.setattr(gamma_smc_cu._core, "FlowContext", FakeFlowContext)

        result = gamma_smc_cu.infer_blockwise(
            G,
            pos,
            pairs=pairs,
            flow_field_path="dummy-flow-field.txt",
            core_block_sites=256,
            flank_sites=64,
            pair_batch_size=32,
            max_streams=3,
            mean_only=False,
        )

        assert result["mean"].shape == (len(pos), len(pairs))
        assert result["pairs"] == pairs
        np.testing.assert_array_equal(result["positions"], pos)
        np.testing.assert_array_equal(result["blocks"], np.array([[0, len(pos), 0, len(pos)]], dtype=np.int32))
        assert calls["init"]["shape"] == G.shape
        assert calls["run"] == {
            "pairs": pairs,
            "core_block_sites": 256,
            "flank_sites": 64,
            "pair_batch_size": 32,
            "max_streams": 3,
            "mean_only": False,
            "return_posterior": False,
        }

    def test_single_block_matches_full_sequence(self, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3)]

        full = gamma_smc_cu.infer(G, pos, pairs=pairs)
        blockwise = gamma_smc_cu.infer_blockwise(
            G,
            pos,
            pairs=pairs,
            core_block_sites=G.shape[1],
            flank_sites=0,
            pair_batch_size=1,
        )

        np.testing.assert_allclose(blockwise["mean"], full["mean"], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(
            blockwise["blocks"],
            np.array([[0, G.shape[1], 0, G.shape[1]]], dtype=np.int32),
        )

    def test_rejects_zero_flank_with_multi_block(self, genotype_data):
        G, pos = genotype_data
        with pytest.raises(ValueError, match="flank_sites=0"):
            gamma_smc_cu.infer_blockwise(
                G,
                pos,
                pairs=[(0, 1)],
                flow_field_path="dummy-flow-field.txt",
                core_block_sites=10,  # < n_sites -> multi-block
                flank_sites=0,
            )

    def test_rejects_non_int_core_block_sites(self, genotype_data):
        G, pos = genotype_data
        with pytest.raises(TypeError, match="core_block_sites"):
            gamma_smc_cu.infer_blockwise(
                G,
                pos,
                pairs=[(0, 1)],
                flow_field_path="dummy-flow-field.txt",
                core_block_sites="not-a-number",
            )

    def test_auto_core_block_sites_uses_recommendation(self, monkeypatch, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3)]
        captured = {}

        # Pretend there's plenty of GPU memory.
        monkeypatch.setattr(
            gamma_smc_cu._core, "cuda_mem_info", lambda: (32 * 10**9, 40 * 10**9)
        )

        class FakeFlowContext:
            def __init__(self, *_args, **_kwargs):
                pass

            def run_fb_blockwise(self, pairs, **kwargs):
                captured.update(kwargs)
                return {
                    "mean": np.ones((len(pos), len(pairs)), dtype=np.float32),
                    "blocks": np.array([[0, len(pos), 0, len(pos)]], dtype=np.int32),
                }

        monkeypatch.setattr(gamma_smc_cu._core, "FlowContext", FakeFlowContext)

        gamma_smc_cu.infer_blockwise(
            G,
            pos,
            pairs=pairs,
            flow_field_path="dummy-flow-field.txt",
            core_block_sites="auto",
            flank_sites=64,
        )

        # 217 sites + 32GB free GPU -> auto collapses to a single full block.
        assert captured["core_block_sites"] == G.shape[1]
        assert captured["flank_sites"] == 64

    def test_warns_when_block_exceeds_memory(self, monkeypatch, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1)]

        # Pretend almost no GPU memory.
        monkeypatch.setattr(
            gamma_smc_cu._core, "cuda_mem_info", lambda: (1024, 1024)
        )

        class FakeFlowContext:
            def __init__(self, *_args, **_kwargs):
                pass

            def run_fb_blockwise(self, pairs, **kwargs):
                return {
                    "mean": np.ones((len(pos), len(pairs)), dtype=np.float32),
                    "blocks": np.array([[0, len(pos), 0, len(pos)]], dtype=np.int32),
                }

        monkeypatch.setattr(gamma_smc_cu._core, "FlowContext", FakeFlowContext)

        with pytest.warns(UserWarning, match="GPU memory"):
            gamma_smc_cu.infer_blockwise(
                G,
                pos,
                pairs=pairs,
                flow_field_path="dummy-flow-field.txt",
                core_block_sites=8192,
                flank_sites=2048,
            )

    def test_auto_pair_batch_size_capped_by_n_pairs(self, monkeypatch, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3), (4, 5)]
        captured = {}

        monkeypatch.setattr(
            gamma_smc_cu._core, "cuda_mem_info", lambda: (32 * 10**9, 40 * 10**9)
        )

        class FakeFlowContext:
            def __init__(self, *_args, **_kwargs):
                pass

            def run_fb_blockwise(self, pairs, **kwargs):
                captured.update(kwargs)
                return {
                    "mean": np.ones((len(pos), len(pairs)), dtype=np.float32),
                    "blocks": np.array([[0, len(pos), 0, len(pos)]], dtype=np.int32),
                }

        monkeypatch.setattr(gamma_smc_cu._core, "FlowContext", FakeFlowContext)

        gamma_smc_cu.infer_blockwise(
            G,
            pos,
            pairs=pairs,
            flow_field_path="dummy-flow-field.txt",
            pair_batch_size=-1,  # auto -> should be capped by n_pairs in Python
        )
        assert captured["pair_batch_size"] == len(pairs)

    def test_auto_falls_back_when_query_fails(self, monkeypatch, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1)]

        def boom():
            raise RuntimeError("no cuda")

        monkeypatch.setattr(gamma_smc_cu._core, "cuda_mem_info", boom)

        class FakeFlowContext:
            def __init__(self, *_args, **_kwargs):
                pass

            def run_fb_blockwise(self, pairs, **kwargs):
                return {
                    "mean": np.ones((len(pos), len(pairs)), dtype=np.float32),
                    "blocks": np.array([[0, len(pos), 0, len(pos)]], dtype=np.int32),
                }

        monkeypatch.setattr(gamma_smc_cu._core, "FlowContext", FakeFlowContext)

        with pytest.warns(UserWarning, match="could not query GPU"):
            gamma_smc_cu.infer_blockwise(
                G,
                pos,
                pairs=pairs,
                flow_field_path="dummy-flow-field.txt",
                core_block_sites="auto",
            )

    def test_blockwise_mean_only(self, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3)]
        result = gamma_smc_cu.infer_blockwise(
            G, pos, pairs=pairs,
            core_block_sites=G.shape[1], flank_sites=0,
        )
        assert "mean" in result
        assert "lower" not in result
        assert "posterior_alpha" not in result
        assert result["mean"].shape == (G.shape[1], 2)

    def test_blockwise_with_ci(self, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3)]
        result = gamma_smc_cu.infer_blockwise(
            G, pos, pairs=pairs,
            core_block_sites=G.shape[1], flank_sites=0,
            mean_only=False,
        )
        for key in ("mean", "lower", "upper"):
            assert key in result
        assert "posterior_alpha" not in result
        assert np.all(result["lower"] <= result["mean"] + 1e-3)
        assert np.all(result["mean"] <= result["upper"] + 1e-3)

    def test_blockwise_with_posterior_mean_only(self, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3)]
        result = gamma_smc_cu.infer_blockwise(
            G, pos, pairs=pairs,
            core_block_sites=G.shape[1], flank_sites=0,
            return_posterior=True,
        )
        for key in ("mean", "posterior_alpha", "posterior_beta"):
            assert key in result
        assert "lower" not in result
        Ne = 10000.0
        recon = (result["posterior_alpha"] / result["posterior_beta"]) * 2.0 * Ne
        np.testing.assert_allclose(recon, result["mean"], rtol=1e-4, atol=1e-2)

    def test_blockwise_with_posterior_and_ci(self, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3)]
        result = gamma_smc_cu.infer_blockwise(
            G, pos, pairs=pairs,
            core_block_sites=G.shape[1], flank_sites=0,
            mean_only=False,
            return_posterior=True,
        )
        for key in ("mean", "lower", "upper", "posterior_alpha", "posterior_beta"):
            assert key in result
        Ne = 10000.0
        recon = (result["posterior_alpha"] / result["posterior_beta"]) * 2.0 * Ne
        np.testing.assert_allclose(recon, result["mean"], rtol=1e-4, atol=1e-2)

    def test_blockwise_posterior_matches_full_infer(self, genotype_data):
        """Reconstructing posterior_mean from (alpha, beta) must match infer()'s mean."""
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3)]
        full = gamma_smc_cu.infer(G, pos, pairs=pairs)
        blk = gamma_smc_cu.infer_blockwise(
            G, pos, pairs=pairs,
            core_block_sites=G.shape[1], flank_sites=0,
            return_posterior=True,
        )
        np.testing.assert_allclose(blk["mean"], full["mean"], rtol=1e-5, atol=1e-6)
        Ne = 10000.0
        recon = (blk["posterior_alpha"] / blk["posterior_beta"]) * 2.0 * Ne
        np.testing.assert_allclose(recon, full["mean"], rtol=1e-4, atol=1e-2)

    def test_blockwise_posterior_rejects_multi_stream(self, genotype_data):
        G, pos = genotype_data
        with pytest.raises(ValueError, match="return_posterior"):
            gamma_smc_cu.infer_blockwise(
                G, pos, pairs=[(0, 1)],
                flow_field_path="dummy-flow-field.txt",
                core_block_sites=G.shape[1], flank_sites=0,
                return_posterior=True,
                max_streams=2,
            )

    def test_streamed_blockwise_matches_single_stream(self, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3), (4, 5)]

        single_stream = gamma_smc_cu.infer_blockwise(
            G,
            pos,
            pairs=pairs,
            core_block_sites=max(1, G.shape[1] // 2),
            flank_sites=min(64, max(0, G.shape[1] // 4)),
            pair_batch_size=2,
            max_streams=1,
        )
        streamed = gamma_smc_cu.infer_blockwise(
            G,
            pos,
            pairs=pairs,
            core_block_sites=max(1, G.shape[1] // 2),
            flank_sites=min(64, max(0, G.shape[1] // 4)),
            pair_batch_size=2,
            max_streams=2,
        )

        np.testing.assert_allclose(streamed["mean"], single_stream["mean"], rtol=1e-5, atol=1e-6)
