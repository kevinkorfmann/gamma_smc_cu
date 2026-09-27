"""Reject incomplete or mismatched scientific runtime comparisons."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('runtime_plot',
    Path(__file__).parents[1]/'plot_matched_runtime.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def records():
    return [dict(series=series, panel_manifest_sha256='panel', n_sites=100,
                 n_haplotypes=20, host='same-host', n_pairs=n,
                 pair_set='all' if n==190 else 'within', pairs_sha256=f'pairs-{n}', status='ok',
                 is_extrapolated=False, metric='native_total', includes_output=True,
                 seconds=[1., 2., 3.])
            for series in module.SERIES for n in (10, 190)]


def test_complete_matched_results(records):
    assert set(module.validate_records(records)) == set(module.SERIES)


@pytest.mark.parametrize('key,value', [
    ('panel_manifest_sha256', 'other'), ('host', 'other-host'),
    ('pairs_sha256', 'different-pairs'), ('n_sites', 99),
    ('is_extrapolated', True), ('status', 'failed'),
    ('metric', 'decode'), ('includes_output', False),
    ('seconds', [1, 2]), ('seconds', [1, float('nan'), 3]),
])
def test_invalid_comparisons_rejected(records, key, value):
    rows = deepcopy(records)
    rows[0][key] = value
    with pytest.raises(ValueError):
        module.validate_records(rows)


def test_missing_point_and_duplicate_point_rejected(records):
    with pytest.raises(ValueError):
        module.validate_records(records[1:])
    with pytest.raises(ValueError):
        module.validate_records(records+[records[0]])


def test_incomplete_endpoint_cannot_be_called_all_pairs(records):
    for row in records:
        if row['n_pairs'] == 190:
            row['n_pairs'] = 100
    with pytest.raises(ValueError, match='every distinct pair'):
        module.validate_records(records)
