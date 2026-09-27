from pathlib import Path
import importlib.util
import numpy as np
import pytest

FILE=Path(__file__).resolve().parents[1]/'orthogonal.py'
spec=importlib.util.spec_from_file_location('orthogonal_rerun',FILE)
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)


def test_h12_and_h2h1_known_haplotype_frequencies():
    # Frequencies0.5,0.25,0.25 ->H1=.375,H12=.625,H2/H1=1/3.
    G=np.array([[0,0],[0,0],[1,0],[1,1]],dtype=np.uint8)
    mids,h12,h2=o.h12_track(G,np.array([10,20]),width=2,step=1)
    assert mids.tolist()==[20]
    assert h12[0]==pytest.approx(.625)
    assert h2[0]==pytest.approx(1/3)
    _,h12,h2=o.h12_track(np.zeros((4,2),dtype=np.uint8),np.array([10,20]),2,1)
    assert h12[0]==1 and h2[0]==0


def test_window_slice_preserves_requested_cache_haplotype_order():
    G=np.arange(24).reshape(4,6);p=np.array([10,20,30,40,50,60])
    window,pos=o.window_slice(G,p,[3,1],20,40)
    assert window.tolist()==[[19,20,21],[7,8,9]]
    assert pos.tolist()==[20,30,40]


@pytest.mark.parametrize('header,rows',[('', 'rs1 10 .3 1 2 -1\nrs2 20 .6 2 1 1\n'),('chr id pos freq ihh1 ihh0 ihs\n','1 rs1 10 .3 1 2 -1\n1 rs2 20 .6 2 1 1\n')])
def test_selscan_parser_supports_six_and_seven_column_formats(tmp_path,header,rows):
    p=tmp_path/'scores.out';p.write_text(header+rows)
    d=o.load_selscan(p,'ihs')
    assert d.pos.tolist()==[10,20] and d.ihs.tolist()==[-1,1]
    assert d.freq.tolist()==[.3,.6]


def test_manifest_includes_later_case_studies_and_both_regional_populations(tmp_path):
    a=o.args_parser().parse_args(['manifest','--inputs-root',str(tmp_path/'inputs'),'--output-root',str(tmp_path/'outputs'),'--tools-root',str(tmp_path/'tools')])
    d=o.manifest(a,tmp_path)
    import pandas as pd
    t=pd.read_csv(tmp_path/'tasks.tsv',sep='\t')
    assert d['task_counts']['selscan']==572 and d['task_counts']['h12']==572
    for gene in ['IFIH1','TREM2']:
        for action in ['variant','regional','asmc','xpehh','neighborhood']:
            assert ((t.gene==gene)&(t.action==action)).any()
        assert ((t.gene==gene)&(t.action=='asmc')&(t['pop']=='YRI')).any()


def test_normalization_pools_chromosomes_and_preserves_missing(tmp_path):
    from argparse import Namespace
    import json
    import pandas as pd
    output = tmp_path / 'output'
    genes = tmp_path / 'genes'
    genes.mkdir()
    all_raw = []
    for chrom in range(1, 23):
        root = output / 'selscan' / f'chr{chrom}_YRI'
        root.mkdir(parents=True)
        (root / 'COMPLETE.json').write_text(json.dumps({'complete': True, 'outputs': {}}))
        vals = np.arange(3) + chrom * 10
        all_raw.extend(vals)
        for stat in ['ihs', 'nsl']:
            (root / f'{stat}.{stat}.out').write_text(''.join(
                f'id{i} {10+i} 0.2 1 1 {v}\n' for i, v in enumerate(vals)))
        pd.DataFrame({'gene_name': [f'gene{chrom}', f'missing{chrom}'],
                      'start': [10, 100], 'end': [10, 120]}).to_csv(
                          genes / f'chr{chrom}_genes.tsv', sep='\t', index=False)
    dest = output / 'normalized'
    dest.mkdir()
    a = Namespace(output_root=str(output), genes_dir=str(genes), pop='YRI')
    o.normalize(a, dest)
    result = pd.read_csv(dest / 'genelevel.csv').set_index('gene_name')
    expected = abs((10 - np.mean(all_raw)) / np.std(all_raw))
    assert result.loc['gene1', 'max_abs_ihs_norm'] == pytest.approx(expected)
    assert result.loc['gene1', 'n_ihs_sites'] == 1
    assert result.loc['missing1', 'n_ihs_sites'] == 0
    assert np.isnan(result.loc['missing1', 'max_abs_ihs_norm_rank'])
    bins = pd.read_csv(dest / 'ihs_normalization_bins.csv')
    assert bins['n'].sum() == 66 and bins['usable'].sum() == 1


@pytest.fixture
def native_asmc(monkeypatch):
    """Model the native reference lifetime and seed-before-construction contract."""
    import sys
    from types import ModuleType, SimpleNamespace

    instances = []
    state = SimpleNamespace(positions=np.array([10, 20, 30]),
                            expected_times=np.array([100., 1000., 10000.]),
                            means=None, pair_indices=[(0, 'a', 1, 'b')])

    class Params:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.useKnownSeed = False
            for old, new in [('jobs', 'jobs'), ('job_ind', 'jobInd'),
                             ('using_CSFS', 'usingCSFS'), ('compress', 'compress'),
                             ('skip_CSFS_distance', 'skipCSFSdistance'),
                             ('no_batches', 'noBatches'),
                             ('do_per_pair_posterior_mean', 'doPerPairPosteriorMean'),
                             ('do_posterior_sums', 'doPosteriorSums'),
                             ('do_major_minor_posterior_sums', 'doMajorMinorPosteriorSums'),
                             ('do_per_pair_MAP', 'doPerPairMAP')]:
                setattr(self, new, kwargs[old])
            self.decodingSequence = kwargs['decoding_mode_string'] == 'sequence'

    class Model:
        def __init__(self, params):
            assert params.useKnownSeed is True, 'native seed must be set before construction'
            self.params = params
            self.switches = {}
            self.native_means = np.random.default_rng(1234).uniform(100, 1000, (1, 3)).astype(np.float32)
            instances.append(self)

        def __getattr__(self, name):
            if name.startswith(('set_store_', 'set_write_')):
                return lambda value: self.switches.__setitem__(name, value)
            raise AttributeError(name)

        def get_decoding_params(self): return self.params
        def get_haploid_sample_size(self): return 100
        def get_physical_positions(self): return state.positions
        def get_expected_times(self): return state.expected_times
        def decode_pairs(self, a, b): assert (a, b) == ([0], [1])
        def get_copy_of_results(self): raise AssertionError('avoid copying unused native results')
        def get_ref_of_results(self):
            return SimpleNamespace(per_pair_indices=state.pair_indices,
                                   per_pair_posterior_means=(self.native_means if state.means is None else state.means))

    package = ModuleType('asmc')
    module = ModuleType('asmc.asmc')
    module.ASMC, module.DecodingParams = Model, Params
    package.asmc = module
    monkeypatch.setitem(sys.modules, 'asmc', package)
    monkeypatch.setitem(sys.modules, 'asmc.asmc', module)
    return instances, state


def test_asmc_seeded_means_are_owned_and_state_count_is_not_assumed(native_asmc):
    instances, state = native_asmc
    first, metadata = o.decode_asmc_subset('input', 'dq', [10, 20, 30])
    second, repeated = o.decode_asmc_subset('input', 'dq', [10, 20, 30])
    np.testing.assert_array_equal(first, second)
    assert metadata == repeated
    assert metadata['native_csfs_seed'] == 1234
    assert metadata['state_count'] == 3
    assert metadata['expected_times_generations'] == state.expected_times.tolist()
    params = metadata['native_parameters']
    assert params['jobs'] == params['jobInd'] == 1
    assert params['usingCSFS'] and not params['compress'] and params['skipCSFSdistance'] == 0
    assert instances[0].switches == {
        'set_store_per_pair_posterior_mean': True, 'set_store_per_pair_map': False,
        'set_store_per_pair_posterior': False, 'set_store_sum_of_posterior': False,
        'set_write_per_pair_posterior_mean': False, 'set_write_per_pair_map': False}
    instances[0].native_means[:] = -999
    np.testing.assert_array_equal(first, second)  # returned means survive native buffer reuse/destruction


@pytest.mark.parametrize('field,value,message', [
    ('means', np.array([[1., 0., 3.]]), 'nonpositive'),
    ('means', np.array([[1., np.nan, 3.]]), 'nonfinite'),
    ('means', np.array([1., 2., 3.]), 'Incomplete'),
    ('pair_indices', [(1, 'b', 0, 'a')], 'unexpected focal'),
    ('positions', np.array([10, 21, 30]), 'position grid'),
    ('expected_times', np.array([100., 100., 1000.]), 'state grid'),
])
def test_asmc_rejects_incomplete_or_misaligned_results(native_asmc, field, value, message):
    _, state = native_asmc
    setattr(state, field, value)
    with pytest.raises(ValueError, match=message):
        o.decode_asmc_subset('input', 'dq', [10, 20, 30])


@pytest.mark.parametrize('external_regional_root', [False, True])
def test_regional_asmc_preserves_subset_seed_pairs_and_window(tmp_path, monkeypatch, native_asmc, external_regional_root):
    from argparse import Namespace
    from types import SimpleNamespace
    import importlib.metadata

    prior_root = tmp_path/'previous-output' if external_regional_root else tmp_path
    prior = prior_root/'regional'/'GENE_YRI'
    prior.mkdir(parents=True)
    (prior/'COMPLETE.json').write_text('{}')
    pairs = np.array([[0, 109], [4, 7]])
    haps = np.arange(120)
    positions = np.array([10, 20, 30])
    G = np.random.default_rng(1).integers(0, 2, size=(120, 3), dtype=np.uint8)
    np.savez(prior/'results.npz', pairs=pairs, population_hap_indices=haps)
    dq = tmp_path/'decoding.gz'
    dq.write_bytes(b'test decoding quantities')
    dest = tmp_path/'asmc'
    dest.mkdir()
    writes = []
    def write_input(root, chrom, pos, subset, samples):
        writes.append((chrom, pos.copy(), subset.copy(), samples))
    monkeypatch.setattr(o, 'legacy', lambda _: SimpleNamespace(write_asmc_input=write_input))
    monkeypatch.setattr(o, 'require_complete', lambda _: {})
    monkeypatch.setattr(o, 'chromosome', lambda _: ({'G': G, 'positions': positions}, {}, haps, {}))
    monkeypatch.setattr(o, 'gene', lambda _: (15, 25, 20, 10, 30))
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '1.4.0' if name == 'asmc-asmc' else None)
    args = Namespace(decoding_quantities=str(dq), tools_root='', output_root=str(tmp_path),
                     regional_root=str(prior_root) if external_regional_root else None,
                     gene='GENE', pop='YRI', chr=6)
    metadata = o.asmc(args, dest)
    rng = np.random.default_rng(123)
    expected_subsets = []
    for first, second in pairs:
        others = [i for i in range(120) if i not in (first, second)]
        expected_subsets.append([first, second]+sorted(rng.choice(others, 98, replace=False).tolist()))
    with np.load(dest/'results.npz') as result:
        np.testing.assert_array_equal(result['pairs'], pairs)
        np.testing.assert_array_equal(result['positions'], positions)
        np.testing.assert_array_equal(result['subsets'], expected_subsets)
        assert result['mean'].shape == (2, 3)
    for (chrom, pos, subset, samples), selected in zip(writes, expected_subsets):
        assert chrom == 6 and samples == 50
        np.testing.assert_array_equal(pos, positions)
        np.testing.assert_array_equal(subset, G[selected])
    assert metadata['asmc_version'] == '1.4.0'
    assert metadata['csfs_haploid_sample_count'] == 50
    assert metadata['native_csfs_seed'] == 1234 and metadata['subset_seed'] == 123
    assert metadata['state_count'] == 3
    assert metadata['regional_dependency']['path'] == str(prior/'COMPLETE.json')


@pytest.mark.parametrize('external_regional_root', [False, True])
def test_asmc_identity_tracks_the_selected_regional_root(tmp_path, monkeypatch, external_regional_root):
    output = tmp_path/'new-output'
    regional = tmp_path/'previous-output' if external_regional_root else output
    argv = ['asmc', '--inputs-root', str(tmp_path/'inputs'), '--output-root', str(output),
            '--tools-root', str(tmp_path/'tools'), '--cache-dir', str(tmp_path/'cache'),
            '--samples', str(tmp_path/'samples'), '--genes-dir', str(tmp_path/'genes'),
            '--chr', '6', '--pop', 'YRI', '--gene', 'GENE']
    if external_regional_root:
        argv += ['--regional-root', str(regional)]
    args = o.args_parser().parse_args(argv)
    monkeypatch.setattr(o, 'fingerprint', lambda path: {'path': str(path), 'sha256': 'test'})
    ident = o.identity(args)
    assert ident['inputs']['regional_marker']['path'] == str(regional/'regional'/'GENE_YRI'/'COMPLETE.json')
    assert o.output_folder(args) == output/'asmc'/'GENE_YRI'
