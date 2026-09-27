"""Path-independent scientific identity for completed chromosome summaries."""
from copy import deepcopy


PATH_OR_TASK_CONFIG = {'cache_dir', 'samples', 'output_dir', 'resume', 'chr', 'populations',
                       'lead_variants', 'leads'}
REQUIRED_CONFIG = ('mu', 'rho', 'core_block_sites', 'flank_sites', 'pair_chunk')


def scientific_identity(metadata):
    if not metadata.get('complete') or metadata.get('calibration', {}).get('time_units') != 'generations':
        raise ValueError('Result must be complete and calibrated in generations')
    identity = metadata['identity']
    config = identity['configuration']
    if any(key not in config for key in REQUIRED_CONFIG):
        raise ValueError('Missing scientific inference settings')
    inputs = identity['inputs']
    cache = inputs['cache']
    if cache.get('format') != 'extracted_npy' or 'record' not in cache:
        raise ValueError('A documented READY.json NPY input cache is required')
    record = cache['record']
    members = {name: entry['sha256'] for name, entry in record['members'].items()}
    if set(members) != {'G.npy', 'positions.npy', 'sample_ids.npy'} or not record.get('source_sha256'):
        raise ValueError('Missing source VCF or cache-member hashes')
    if record.get('positions_base') != 1 or record.get('positions_dtype') != 'int64':
        raise ValueError('Expected exact one-based int64 VCF cache coordinates')
    filters = deepcopy(record.get('filters', {}))
    filters.pop('chromosome', None)
    sources = identity.get('source_sha256')
    if not sources:
        raise ValueError('Missing inference source hashes')
    common = dict(
        configuration={k: v for k, v in config.items() if k not in PATH_OR_TASK_CONFIG and k != 'sequence_length'},
        sources=sources,
        samples_sha256=inputs['samples']['sha256'],
        input_schema=dict(format=cache['format'], schema=record['schema'], dtype=record['dtype'],
                          positions_dtype=record['positions_dtype'], positions_base=record['positions_base'],
                          coordinate_convention=record['coordinate_convention'], filters=filters),
        calibration_method=metadata['calibration'].get('calibration'),
        denominator_convention=metadata['calibration'].get('denominator_convention'),
        time_units='generations')
    for optional in ('lead_variants', 'leads'):
        if optional in inputs:
            common[optional + '_sha256'] = inputs[optional]['sha256']
    chromosome = dict(source_vcf_sha256=record['source_sha256'], cache_members=members,
                      cache_shape=record['shape'], genes_sha256=inputs['genes']['sha256'],
                      sequence_length=config.get('sequence_length'))
    return common, chromosome
