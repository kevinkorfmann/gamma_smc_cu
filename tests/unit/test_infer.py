"""Unit tests for tmrca_cu.infer() top-level API."""

import numpy as np
import pytest
import msprime

import tmrca_cu


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
        result = tmrca_cu.infer(ts)
        assert isinstance(result, dict)
        assert "mean" in result

    def test_shape_all_pairs(self, ts):
        result = tmrca_cu.infer(ts)
        n = ts.num_samples
        S = ts.num_sites
        n_pairs = n * (n - 1) // 2
        assert result["mean"].shape == (S, n_pairs)

    def test_shape_specific_pairs(self, ts):
        pairs = [(0, 1), (2, 3)]
        result = tmrca_cu.infer(ts, pairs=pairs)
        assert result["mean"].shape[1] == 2

    def test_returns_pairs(self, ts):
        pairs = [(0, 1), (5, 3)]
        result = tmrca_cu.infer(ts, pairs=pairs)
        assert result["pairs"] == pairs

    def test_returns_positions(self, ts):
        result = tmrca_cu.infer(ts, pairs=[(0, 1)])
        assert len(result["positions"]) == ts.num_sites

    def test_values_positive(self, ts):
        result = tmrca_cu.infer(ts, pairs=[(0, 1)])
        assert np.all(result["mean"] > 0)

    def test_with_ci(self, ts):
        result = tmrca_cu.infer(ts, pairs=[(0, 1)], mean_only=False)
        assert "lower" in result
        assert "upper" in result
        assert np.all(result["lower"] <= result["mean"])
        assert np.all(result["upper"] >= result["mean"])


class TestInferGenotypeMatrix:
    def test_returns_dict(self, genotype_data):
        G, pos = genotype_data
        result = tmrca_cu.infer(G, pos)
        assert isinstance(result, dict)
        assert "mean" in result

    def test_shape_all_pairs(self, genotype_data):
        G, pos = genotype_data
        n = G.shape[0]
        S = G.shape[1]
        n_pairs = n * (n - 1) // 2
        result = tmrca_cu.infer(G, pos)
        assert result["mean"].shape == (S, n_pairs)

    def test_specific_pairs(self, genotype_data):
        G, pos = genotype_data
        result = tmrca_cu.infer(G, pos, pairs=[(0, 1)])
        assert result["mean"].shape[1] == 1

    def test_custom_params(self, genotype_data):
        G, pos = genotype_data
        result = tmrca_cu.infer(G, pos, mu=1e-8, rho=1e-8, Ne=5000,
                                 pairs=[(0, 1)])
        assert result["mean"].shape[1] == 1
        assert np.all(result["mean"] > 0)

    def test_rejects_mismatched_positions(self, genotype_data):
        G, pos = genotype_data
        with pytest.raises(ValueError, match="positions length"):
            tmrca_cu.infer(G, pos[:-1], pairs=[(0, 1)])


class TestInferConsistency:
    def test_ts_and_matrix_agree(self, ts, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3), (4, 5)]
        r_ts = tmrca_cu.infer(ts, pairs=pairs)
        r_mat = tmrca_cu.infer(G, pos, pairs=pairs)
        np.testing.assert_allclose(r_ts["mean"], r_mat["mean"],
                                    rtol=1e-5, atol=1e-6)

    def test_correlates_with_truth(self, ts):
        pair = (0, 1)
        result = tmrca_cu.infer(ts, pairs=[pair])
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
        result = tmrca_cu.infer(ts, pairs=[(0, 1)])
        assert result["mean"].shape[1] == 1

    def test_empty_pairs(self, ts):
        result = tmrca_cu.infer(ts, pairs=[])
        assert result["mean"].shape[1] == 0


class TestInferModes:
    """All three output modes for tmrca_cu.infer()."""

    def test_mean_only(self, ts):
        pairs = [(0, 1), (2, 3)]
        result = tmrca_cu.infer(ts, pairs=pairs, mean_only=True)
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
        result = tmrca_cu.infer(ts, pairs=pairs, mean_only=False)
        assert {"mean", "lower", "upper"} <= set(result.keys())
        assert "posterior_alpha" not in result
        for key in ("lower", "mean", "upper"):
            assert result[key].shape == (ts.num_sites, 2)
            assert result[key].dtype == np.float32
        assert np.all(result["lower"] <= result["mean"] + 1e-3)
        assert np.all(result["mean"] <= result["upper"] + 1e-3)

    def test_with_posterior_mean_only(self, ts):
        pairs = [(0, 1), (2, 3)]
        result = tmrca_cu.infer(
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
        result = tmrca_cu.infer(
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
        baseline = tmrca_cu.infer(ts, pairs=pairs, mean_only=True)
        with_post = tmrca_cu.infer(
            ts, pairs=pairs, mean_only=True, return_posterior=True
        )
        np.testing.assert_array_equal(baseline["mean"], with_post["mean"])


class TestInferBlockwise:
    def test_requires_explicit_pairs(self, genotype_data):
        G, pos = genotype_data
        with pytest.raises(ValueError, match="explicit pairs"):
            tmrca_cu.infer_blockwise(
                G,
                pos,
                flow_field_path="dummy-flow-field.txt",
            )

    def test_rejects_invalid_pair_batch_size(self, genotype_data):
        G, pos = genotype_data
        with pytest.raises(ValueError, match="pair_batch_size"):
            tmrca_cu.infer_blockwise(
                G,
                pos,
                pairs=[(0, 1)],
                flow_field_path="dummy-flow-field.txt",
                pair_batch_size=0,
            )

    def test_rejects_invalid_max_streams(self, genotype_data):
        G, pos = genotype_data
        with pytest.raises(ValueError, match="max_streams"):
            tmrca_cu.infer_blockwise(
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

        monkeypatch.setattr(tmrca_cu._core, "FlowContext", FakeFlowContext)

        result = tmrca_cu.infer_blockwise(
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

        full = tmrca_cu.infer(G, pos, pairs=pairs)
        blockwise = tmrca_cu.infer_blockwise(
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
            tmrca_cu.infer_blockwise(
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
            tmrca_cu.infer_blockwise(
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
            tmrca_cu._core, "cuda_mem_info", lambda: (32 * 10**9, 40 * 10**9)
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

        monkeypatch.setattr(tmrca_cu._core, "FlowContext", FakeFlowContext)

        tmrca_cu.infer_blockwise(
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
            tmrca_cu._core, "cuda_mem_info", lambda: (1024, 1024)
        )

        class FakeFlowContext:
            def __init__(self, *_args, **_kwargs):
                pass

            def run_fb_blockwise(self, pairs, **kwargs):
                return {
                    "mean": np.ones((len(pos), len(pairs)), dtype=np.float32),
                    "blocks": np.array([[0, len(pos), 0, len(pos)]], dtype=np.int32),
                }

        monkeypatch.setattr(tmrca_cu._core, "FlowContext", FakeFlowContext)

        with pytest.warns(UserWarning, match="GPU memory"):
            tmrca_cu.infer_blockwise(
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
            tmrca_cu._core, "cuda_mem_info", lambda: (32 * 10**9, 40 * 10**9)
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

        monkeypatch.setattr(tmrca_cu._core, "FlowContext", FakeFlowContext)

        tmrca_cu.infer_blockwise(
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

        monkeypatch.setattr(tmrca_cu._core, "cuda_mem_info", boom)

        class FakeFlowContext:
            def __init__(self, *_args, **_kwargs):
                pass

            def run_fb_blockwise(self, pairs, **kwargs):
                return {
                    "mean": np.ones((len(pos), len(pairs)), dtype=np.float32),
                    "blocks": np.array([[0, len(pos), 0, len(pos)]], dtype=np.int32),
                }

        monkeypatch.setattr(tmrca_cu._core, "FlowContext", FakeFlowContext)

        with pytest.warns(UserWarning, match="could not query GPU"):
            tmrca_cu.infer_blockwise(
                G,
                pos,
                pairs=pairs,
                flow_field_path="dummy-flow-field.txt",
                core_block_sites="auto",
            )

    def test_blockwise_mean_only(self, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3)]
        result = tmrca_cu.infer_blockwise(
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
        result = tmrca_cu.infer_blockwise(
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
        result = tmrca_cu.infer_blockwise(
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
        result = tmrca_cu.infer_blockwise(
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
        full = tmrca_cu.infer(G, pos, pairs=pairs)
        blk = tmrca_cu.infer_blockwise(
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
            tmrca_cu.infer_blockwise(
                G, pos, pairs=[(0, 1)],
                flow_field_path="dummy-flow-field.txt",
                core_block_sites=G.shape[1], flank_sites=0,
                return_posterior=True,
                max_streams=2,
            )

    def test_streamed_blockwise_matches_single_stream(self, genotype_data):
        G, pos = genotype_data
        pairs = [(0, 1), (2, 3), (4, 5)]

        single_stream = tmrca_cu.infer_blockwise(
            G,
            pos,
            pairs=pairs,
            core_block_sites=max(1, G.shape[1] // 2),
            flank_sites=min(64, max(0, G.shape[1] // 4)),
            pair_batch_size=2,
            max_streams=1,
        )
        streamed = tmrca_cu.infer_blockwise(
            G,
            pos,
            pairs=pairs,
            core_block_sites=max(1, G.shape[1] // 2),
            flank_sites=min(64, max(0, G.shape[1] // 4)),
            pair_batch_size=2,
            max_streams=2,
        )

        np.testing.assert_allclose(streamed["mean"], single_stream["mean"], rtol=1e-5, atol=1e-6)
