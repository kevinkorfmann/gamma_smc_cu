"""CPU-only regression tests for the scientific benchmark contract."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest
import tskit

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from bench_inputs import materialize_binary_snp_vcf
from run_one import align_reference, blockwise_metrics, pair_metrics, simulate_config, true_t_all_pairs, source_metadata, sha256


def boundary_tree_sequence():
    tables = tskit.TableCollection(sequence_length=10)
    for _ in range(2):
        tables.individuals.add_row()
    for i in range(4):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=i // 2)
    tables.nodes.add_row(time=100)
    tables.nodes.add_row(time=200)
    for node in range(4):
        tables.edges.add_row(0, 4, 4, node)
        tables.edges.add_row(4, 10, 5, node)
    for position, changes in [(0, [(0, 'T')]), (4, [(0, 'T')]), (5, [(5, 'T')]), (6, []), (7, [(0, 'T'), (1, 'G')]), (8, [(0, 'AA')])]:
        site = tables.sites.add_row(position, 'A')
        for node, state in changes:
            tables.mutations.add_row(site=site, node=node, derived_state=state)
    tables.sort()
    return tables.tree_sequence()


def test_vcf_roundtrip_at_zero_and_recombination_boundary(tmp_path):
    ts = boundary_tree_sequence()
    inputs = materialize_binary_snp_vcf(ts, str(tmp_path / 'sites.vcf'))
    # Zero becomes valid VCF POS=1, while the mutation at the topology change
    # remains in the RIGHT-hand tree when converted back to zero-based.
    np.testing.assert_array_equal(inputs.pos, [0, 4])
    np.testing.assert_array_equal(inputs.site_ids, [0, 1])
    fields = [line.split('\t') for line in (tmp_path / 'sites.vcf').read_text().splitlines() if not line.startswith('#')]
    assert [int(f[1]) for f in fields] == [1, 5]
    truth = true_t_all_pairs(ts, [(0, 1)], inputs.pos, inputs.sample_nodes)
    np.testing.assert_array_equal(truth[:, 0], [100, 200])
    assert inputs.n_dropped_nonseg == 1  # all alternate must be dropped
    assert inputs.n_dropped_non_binary == 1  # multiallelic site
    assert inputs.n_dropped_non_snp == 2  # no ALT and insertion
    assert np.all(inputs.G.min(axis=0) == 0) and np.all(inputs.G.max(axis=0) == 1)


def test_single_mutation_layer_and_explicit_configured_rates():
    cfg = dict(species='HomSap', model_id='OutOfAfrica_3G09', pop='YRI', n_hap=4,
               seq_len=10000, mu=1.29e-8, rho=1.2820402396300887e-8, seed=42)
    with pytest.warns(UserWarning, match='mutation rate'):
        ts = simulate_config(cfg)
    provenance = [json.loads(p.record) for p in ts.provenances()]
    mutations = [p for p in provenance if p.get('parameters', {}).get('command') == 'sim_mutations']
    assert len(mutations) == 1
    assert mutations[0]['parameters']['rate'] == cfg['mu']
    assert ts.num_samples == 4


def test_reference_uses_physical_units_and_metadata_pair_order():
    metadata = {'scaled_mutation_rate': .001, 'pairs': [[1, 2], [0, 2], [0, 1]]}
    # alpha/beta=1 coalescent unit: with theta=.001 and mu=1.25e-8,
    # the physical generation scale is 40,000, not fixed-Ne's 20,000.
    means = np.array([[1., 2., 3.], [4., 5., 6.]])
    out = align_reference([0, 4], means, metadata, np.array([4]), [(0, 1), (1, 2)], 1.25e-8)
    np.testing.assert_array_equal(out, [[240000, 160000]])
    with pytest.raises(ValueError, match='no interpolation'):
        align_reference([0, 4], means, metadata, np.array([3]), [(0, 1)], 1.25e-8)


def test_metrics_detect_scale_error_that_correlation_cannot():
    truth = np.array([[100.], [200.], [400.]])
    records = pair_metrics(truth, truth * 2, truth, [(0, 1)])
    assert records[0]['r_gamma_smc_cu'] == pytest.approx(1)
    assert records[0]['rmse_gamma_smc_cu'] == pytest.approx(np.log(2))
    assert records[0]['rmse_gsmc'] == 0
    constant = pair_metrics(np.ones((3, 1)), np.ones((3, 1)), np.ones((3, 1)), [(0, 1)])
    assert constant[0]['r_between_methods'] is None


def test_existing_run_is_never_overwritten(tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    (output / 'evidence').write_text('preserve')
    result = subprocess.run([sys.executable, str(HERE / 'run_one.py'), '--config-idx', '0', '--output-dir', str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert (output / 'evidence').read_text() == 'preserve'
    assert list(output.iterdir()) == [output / 'evidence']


def test_blockwise_diagnostics_localize_boundary_errors():
    full = np.ones((2000, 2))
    blocked = full.copy()
    blocked[1000] = np.exp(.2)
    stats = blockwise_metrics(full, blocked, np.arange(2000), 1000)
    assert stats['n_blocks'] == 2
    assert stats['within_64_sites_of_boundary']['max_abs_log_difference'] == pytest.approx(.2)
    assert stats['more_than_512_sites_from_boundary']['max_abs_log_difference'] == 0


def test_retained_configuration_ids_are_exact():
    output = subprocess.check_output([sys.executable, str(HERE / 'run_one.py'), '--list-configs'], text=True)
    assert output.strip() == '0,1,2,3,4,5,6,7,8,10,11,12,13,14'


def test_exported_snapshot_requires_and_verifies_manifest(tmp_path):
    with pytest.raises(ValueError, match='requires SOURCE_MANIFEST'):
        source_metadata(tmp_path)
    source = tmp_path / 'runner.py'
    source.write_text('print(1)\n')
    (tmp_path / 'SOURCE_MANIFEST.json').write_text(json.dumps({
        'git_commit': 'a' * 40, 'files': {'runner.py': sha256(source)}}))
    meta = source_metadata(tmp_path)
    assert meta['source_manifest_verified'] and meta['git_commit'] == 'a' * 40
    source.write_text('print(2)\n')
    with pytest.raises(ValueError, match='checksum mismatch'):
        source_metadata(tmp_path)
