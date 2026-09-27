"""Controlled ASMC tuning on one frozen full marker panel and distinct pairs.

Version 2 fixes the upstream CSFS sampling seed and reconstructs convenience
defaults explicitly (native batch 64). Version 1's unseeded baseline is invalid
for numerical parity and must remain a separate, unselected diagnostic record.
Runs each profile in its own subprocess (independent peak RSS), with one warmup
and three measured repetitions by default. No genotype/marker thinning occurs.
The optional compressed-emission profile changes the model and is excluded from
the main timing comparison. This script does not launch Slurm jobs.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def check_panel(panel):
    panel = Path(panel).resolve()
    for name, entry in json.loads((panel / 'manifest.json').read_text()).items():
        path = (panel / name).resolve()
        if not path.is_relative_to(panel) or sha256(path) != entry['sha256']:
            raise ValueError(f'Frozen panel checksum mismatch: {name}')
    info = json.loads((panel / 'panel.json').read_text())
    if info['positions_base'] != 0:
        raise ValueError('Expected zero-based frozen-panel positions')
    return info


def select_pairs(n_haplotypes, count, supplied=None):
    if supplied is None:
        if n_haplotypes % 2 or not 1 <= count <= n_haplotypes // 2:
            raise ValueError('Default pairs require count <= diploid sample count')
        # Spread within-individual pairs across the cohort without refiltering it.
        individual = np.linspace(0, n_haplotypes // 2 - 1, count, dtype=np.int64)
        pairs = np.column_stack([2 * individual, 2 * individual + 1])
    else:
        pairs = np.asarray(supplied)
        if pairs.dtype.kind not in 'iu' or pairs.shape != (count, 2):
            raise ValueError('Supplied pairs must have shape (count, 2) and integer dtype')
    if (np.any(pairs < 0) or np.any(pairs >= n_haplotypes)
            or np.any(pairs[:, 0] == pairs[:, 1])
            or len({tuple(sorted(p)) for p in pairs.tolist()}) != len(pairs)):
        raise ValueError('Pairs must be distinct, nonself and within the full panel')
    return pairs.astype(np.int64)


def profiles(batch_sizes, native_batch_size=32, include_compress=False):
    result = [dict(name='v2_reconstructed_defaults_copy_b32_n64', constructor='reconstructed_defaults',
                   extraction='copy', outer_batch=32, native_batch=64, changed_model=False)]
    if native_batch_size != 64:
        # Hold legacy flags constant to separate native allocation size from the
        # cost of unused summaries. The first profile matches the actual default.
        result.append(dict(name=f'v2_reconstructed_defaults_copy_b32_n{native_batch_size}',
                           constructor='reconstructed_defaults', extraction='copy', outer_batch=32,
                           native_batch=native_batch_size, changed_model=False))
    for batch in batch_sizes:
        for extraction in ['copy', 'ref']:
            result.append(dict(name=f'v2_means_only_{extraction}_b{batch}_n{native_batch_size}', constructor='explicit',
                               extraction=extraction, outer_batch=batch,
                               native_batch=native_batch_size, changed_model=False))
    if include_compress:
        result.append(dict(name=f'v2_changed_model_compressed_ref_b128_n{native_batch_size}', constructor='explicit',
                           extraction='ref', outer_batch=128, native_batch=native_batch_size,
                           changed_model=True))
    return result


def create_asmc(module, profile, panel, dq, output):
    if profile['constructor'] not in ('explicit', 'reconstructed_defaults'):
        raise ValueError('Unseeded convenience constructor is excluded from controlled comparisons')
    changed = profile['changed_model']
    defaults = profile['constructor'] == 'reconstructed_defaults'
    params = module.DecodingParams(
        in_file_root=str(panel / 'asmc'), dq_file=str(dq), out_file_root=str(output / 'asmc'),
        jobs=1, job_ind=1, decoding_mode_string='sequence', using_CSFS=not changed, compress=changed,
        skip_CSFS_distance=float('nan') if changed else 0.0, no_batches=False,
        do_posterior_sums=defaults, do_major_minor_posterior_sums=defaults,
        do_per_pair_posterior_mean=False, do_per_pair_MAP=defaults)
    params.batchSize = profile['native_batch']
    # Data.cpp seeds C rand with1234 before randomized CSFS hypergeometric
    # projection. Set before construction, including in every independent worker.
    params.useKnownSeed = True
    obj = module.ASMC(params)
    obj.set_store_per_pair_posterior_mean(True)
    p = obj.get_decoding_params()
    recorded = {name: getattr(p, name) for name in ['batchSize', 'noBatches', 'decodingSequence',
                'usingCSFS', 'compress', 'skipCSFSdistance', 'doPosteriorSums',
                'doMajorMinorPosteriorSums', 'doPerPairMAP', 'useKnownSeed', 'jobs', 'jobInd']}
    recorded['fixed_upstream_seed'] = 1234
    if not recorded['useKnownSeed'] or recorded['jobs'] != 1 or recorded['jobInd'] != 1:
        raise ValueError('Exact pair decoding requires fixed CSFS seed and full-panel jobs=jobInd=1')
    if recorded['batchSize'] != profile['native_batch']:
        raise ValueError('Actual native SIMD batch differs from the declared profile')
    if not recorded['decodingSequence'] or recorded['noBatches']:
        raise ValueError('ASMC must retain vectorized sequence-mode decoding')
    if profile['constructor'] == 'explicit' and (recorded['doPosteriorSums'] or recorded['doMajorMinorPosteriorSums']):
        raise ValueError('Means-only profile unexpectedly enables unused summaries')
    if not profile['changed_model'] and (not recorded['usingCSFS'] or recorded['compress'] or recorded['skipCSFSdistance'] != 0):
        raise ValueError('Primary profiles must preserve full CSFS emissions')
    recorded['skipCSFSdistance'] = str(recorded['skipCSFSdistance']) if math.isinf(recorded['skipCSFSdistance']) else recorded['skipCSFSdistance']
    return obj, recorded


def comparison(values, reference, rtol, atol):
    """Bound temporary memory while checking every posterior mean."""
    result = dict(n_values=values.size, n_outside_tolerance=0, exactly_equal=True,
                  maximum_absolute_difference=0.0, maximum_relative_difference=0.0)
    for start in range(0, values.shape[1], 8192):
        block = np.asarray(values[:, start:start + 8192], dtype=np.float64)
        if not np.isfinite(block).all() or np.any(block <= 0):
            raise ValueError('Posterior means must be finite and strictly positive')
        if reference is None:
            continue
        expected = np.asarray(reference[:, start:start + 8192], dtype=np.float64)
        if not np.isfinite(expected).all() or np.any(expected <= 0):
            raise ValueError('Reference posterior means must be finite and positive')
        difference = np.abs(block - expected)
        result['n_outside_tolerance'] += int(np.count_nonzero(difference > atol + rtol * np.abs(expected)))
        result['exactly_equal'] &= bool(np.array_equal(block, expected))
        result['maximum_absolute_difference'] = max(result['maximum_absolute_difference'], float(difference.max()))
        result['maximum_relative_difference'] = max(result['maximum_relative_difference'], float((difference / np.abs(expected)).max()))
    result['compared_to_reference'] = reference is not None
    result['within_tolerance'] = result['n_outside_tolerance'] == 0
    return result


def distribution(records, key):
    values = np.asarray([r[key] for r in records if not r['warmup']])
    return dict(seconds=values.tolist(), median=float(np.median(values)), minimum=float(values.min()),
                maximum=float(values.max()), q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)))


def run_profile(spec, profile, output, reference_path=None, module=None):
    module = module or importlib.import_module('asmc.asmc')
    panel, dq = Path(spec['panel_dir']), Path(spec['decoding_quantities'])
    info = spec['panel']; pairs = np.load(spec['pairs_file']); positions = np.load(panel / 'positions.npy')
    records = []
    for repetition in range(-1, spec['repeats']):
        directory = output / ('warmup' if repetition < 0 else f'rep{repetition:02d}')
        directory.mkdir()
        start = time.perf_counter()
        t0 = time.perf_counter(); obj, params = create_asmc(module, profile, panel, dq, directory)
        setup = time.perf_counter() - t0
        t0 = time.perf_counter()
        if obj.get_haploid_sample_size() != info['n_haplotypes'] or obj.get_num_sites() != info['n_sites']:
            raise ValueError('ASMC input workload changed')
        np.testing.assert_array_equal(obj.get_physical_positions(), positions + 1)
        expected_times = np.asarray(obj.get_expected_times())
        if not np.isfinite(expected_times).all() or np.any(expected_times <= 0):
            raise ValueError('Invalid ASMC expected times')
        validation = time.perf_counter() - t0
        prediction_path = directory / 'posterior_means.npy'
        t0 = time.perf_counter()
        predictions = np.lib.format.open_memmap(prediction_path, mode='w+', dtype=np.float32,
                                               shape=(len(pairs), info['n_sites']))
        output_time = time.perf_counter() - t0
        reference = np.load(reference_path, mmap_mode='r') if reference_path else None
        native_decode = extraction = 0.0
        checks = []; ownership = []; calls = []
        for lo in range(0, len(pairs), profile['outer_batch']):
            subset = pairs[lo:lo + profile['outer_batch']]
            a, b = subset[:, 0].tolist(), subset[:, 1].tolist()
            t0 = time.perf_counter(); obj.decode_pairs(a, b)
            native_decode += time.perf_counter() - t0
            t0 = time.perf_counter()
            result = obj.get_copy_of_results() if profile['extraction'] == 'copy' else obj.get_ref_of_results()
            values = np.asarray(result.per_pair_posterior_means)
            returned = np.asarray([(row[0], row[2]) for row in result.per_pair_indices], dtype=np.int64)
            extraction += time.perf_counter() - t0
            t0 = time.perf_counter()
            if values.shape != (len(subset), info['n_sites']) or values.dtype != np.float32:
                raise ValueError('Unexpected ASMC posterior mean shape/dtype')
            np.testing.assert_array_equal(returned, subset)
            checks.append(comparison(values, None if reference is None else reference[lo:lo + len(subset)],
                                     spec['rtol'], spec['atol']))
            ownership.append(dict(numpy_owns_data=bool(values.flags.owndata),
                                  contiguous=bool(values.flags.c_contiguous), has_base=values.base is not None))
            validation += time.perf_counter() - t0
            t0 = time.perf_counter(); predictions[lo:lo + len(subset)] = values
            output_time += time.perf_counter() - t0
            calls.append(len(subset))
            # All views are consumed before the next decode can replace native buffers.
            del values, result, returned
        t0 = time.perf_counter(); predictions.flush(); del predictions
        output_time += time.perf_counter() - t0
        elapsed = time.perf_counter() - start
        del reference, obj; gc.collect()
        passed = all(c['within_tolerance'] for c in checks)
        record = dict(repetition=repetition, warmup=repetition < 0, setup_seconds=setup,
                      native_decode_seconds=native_decode, extraction_seconds=extraction,
                      decode_and_extract_seconds=native_decode + extraction,
                      validation_seconds=validation, output_seconds=output_time,
                      total_seconds=elapsed, params=params, native_states=len(expected_times),
                      outer_call_pair_counts=calls, array_ownership=ownership,
                      comparisons=checks, same_model_parity_passed=passed,
                      predictions_sha256=sha256(prediction_path), predictions_bytes=prediction_path.stat().st_size)
        records.append(record); write_json(directory / 'timing.json', record)
        if reference_path is None:
            # First baseline warmup is the immutable numerical reference for every profile.
            reference_path = str(prediction_path)
        elif not spec['retain_predictions']:
            prediction_path.unlink()
        write_json(output / 'progress.json', records)
    measured = [r for r in records if not r['warmup']]
    parity = all(r['same_model_parity_passed'] for r in records)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = dict(profile=profile, status='ok' if parity or profile['changed_model'] else 'parity_failed',
                  eligible_for_main_comparison=not profile['changed_model'] and parity,
                  reference_path=reference_path, comparisons_include_warmup=True,
                  repetitions=records, peak_process_rss_bytes=int(rss if sys.platform == 'darwin' else rss * 1024),
                  memory_scope='whole profile subprocess high-water mark, including setup, validation, outputs and result copies',
                  timing_scope='native_decode includes ASMC decode_pairs and native posterior-mean materialization; extraction, validation and binary output are separately timed',
                  timing={key: distribution(measured, key) for key in ['setup_seconds', 'native_decode_seconds', 'decode_and_extract_seconds',
                          'extraction_seconds', 'validation_seconds', 'output_seconds', 'total_seconds']})
    write_json(output / 'result.json', result)
    return result


def environment():
    module = importlib.import_module('asmc.asmc')
    files = {}
    for name in ['asmc', 'asmc.asmc', module.ASMC.__module__]:
        imported = importlib.import_module(name)
        if getattr(imported, '__file__', None):
            path = Path(imported.__file__).resolve()
            for source in [path, *path.parent.glob('*.so')]:
                files[str(source)] = sha256(source)
    def captured(command):
        try:
            return subprocess.run(command, capture_output=True, text=True, timeout=15).stdout
        except OSError:
            return 'unavailable'
    return dict(hostname=socket.gethostname(), platform=platform.platform(), python=sys.version,
                executable=sys.executable, asmc_version=importlib.metadata.version('asmc-asmc'),
                module_files_sha256=files, lscpu=captured(['lscpu']),
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                allocation={k:v for k,v in os.environ.items() if k.startswith('SLURM_') or k in
                            ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
                script_sha256=sha256(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel-dir'); p.add_argument('--decoding-quantities')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--pairs', type=int, default=64, help='Distinct pairs; default spread within-individual pairs')
    p.add_argument('--pairs-file', help='Optional exact integer pair array, shape (--pairs, 2)')
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[8,32,64,128,256,512], help='Python API call sizes; native SIMD batch remains independent')
    p.add_argument('--native-batch-size', type=int, default=32)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--rtol', type=float, default=1e-6); p.add_argument('--atol', type=float, default=1e-5)
    p.add_argument('--include-compress', action='store_true', help='Separate changed-emission-model experiment, excluded from main comparison')
    p.add_argument('--retain-predictions', action='store_true')
    p.add_argument('--timeout-per-profile', type=float, default=1800)
    p.add_argument('--profile-spec', help=argparse.SUPPRESS); p.add_argument('--profile-index', type=int, help=argparse.SUPPRESS)
    a = p.parse_args(); output = Path(a.output_dir).resolve(); output.mkdir(parents=True, exist_ok=False)
    if a.profile_spec:
        spec = json.loads(Path(a.profile_spec).read_text()); profile = spec['profiles'][a.profile_index]
        reference = None if a.profile_index == 0 else spec['reference_path']
        result = run_profile(spec, profile, output, reference)
        return 0 if result['status'] == 'ok' else 3
    if (not a.panel_dir or not a.decoding_quantities or min(a.pairs, a.repeats, a.native_batch_size, *a.batch_sizes) < 1
            or len(set(a.batch_sizes)) != len(a.batch_sizes) or min(a.rtol, a.atol) < 0
            or not np.isfinite([a.rtol, a.atol, a.timeout_per_profile]).all() or a.timeout_per_profile <= 0):
        p.error('Provide panel/DQ paths, positive counts/timeouts and unique batch sizes, and finite nonnegative tolerances')
    panel = Path(a.panel_dir).resolve(); dq = Path(a.decoding_quantities).resolve(); info = check_panel(panel)
    pairs = select_pairs(info['n_haplotypes'], a.pairs, np.load(a.pairs_file) if a.pairs_file else None)
    pairs_file = output / 'pairs.npy'; np.save(pairs_file, pairs)
    configurations = profiles(a.batch_sizes, a.native_batch_size, a.include_compress)
    spec = dict(schema_version=2, diagnostic_version=2, panel_dir=str(panel), decoding_quantities=str(dq), panel=info,
                panel_manifest_sha256=sha256(panel / 'manifest.json'), dq_sha256=sha256(dq),
                pairs_file=str(pairs_file), pairs_sha256=sha256(pairs_file), n_pairs=len(pairs),
                repeats=a.repeats, rtol=a.rtol, atol=a.atol, profiles=configurations,
                retain_predictions=a.retain_predictions,
                reference_path=str(output / configurations[0]['name'] / 'warmup/posterior_means.npy'),
                caveat='Outer batches >= total pairs have identical actual call shapes; one CPU process per profile; no genotype or marker thinning')
    spec_file = output / 'specification.json'; write_json(spec_file, spec); write_json(output / 'environment.json', environment())
    results = []
    for index, profile in enumerate(configurations):
        folder = output / profile['name']
        with open(output / f"{profile['name']}.log", 'w') as log:
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--profile-spec', str(spec_file),
                                   '--profile-index', str(index), '--output-dir', str(folder)],
                                  stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout_per_profile)
        if not (folder / 'result.json').exists():
            raise RuntimeError(f'Profile failed with exit {proc.returncode}; inspect {profile["name"]}.log')
        result = json.loads((folder / 'result.json').read_text()); results.append(result)
        write_json(output / 'progress.json', results)
        print(json.dumps({'profile':profile['name'], 'status':result['status'],
                          'median_decode_and_extract_seconds':result['timing']['decode_and_extract_seconds']['median'],
                          'peak_process_rss_bytes':result['peak_process_rss_bytes']}), flush=True)
        if proc.returncode not in (0,3):
            raise RuntimeError(f'Unexpected profile exit {proc.returncode}')
    check_panel(panel)
    if sha256(dq) != spec['dq_sha256']:
        raise ValueError('Decoding quantities changed during diagnostic')
    eligible = [r for r in results if r['eligible_for_main_comparison']]
    eligible.sort(key=lambda r:r['timing']['decode_and_extract_seconds']['median'])
    passed = all(r['status'] == 'ok' for r in results if not r['profile']['changed_model'])
    write_json(output / 'result.json', dict(status='ok' if passed else 'parity_failed', profiles=results,
               ranking_same_model_decode_plus_extraction=[r['profile']['name'] for r in eligible],
               source_script_sha256=sha256(__file__), specification_sha256=sha256(spec_file)))
    return 0 if passed else 3


if __name__ == '__main__':
    raise SystemExit(main())
