"""Portable timings on one immutable marker panel and exact distinct pair sets.

Prepare once, then run each method/pair-set on its allocated node. Native timing
scopes are recorded explicitly; this script never fabricates all-pairs timings
from repeated copies of one pair or silently extrapolates a small run.
"""
from __future__ import annotations
import argparse
import gzip
import importlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'benchmarks/test_suite_stdpopsim'))
from run_one import captured, sha256, source_metadata, write_json


def manifest(folder):
    return {str(p.relative_to(folder)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
            for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'manifest.json'}


def check_panel(folder):
    entries = json.loads((folder / 'manifest.json').read_text())
    for name, entry in entries.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or sha256(path) != entry['sha256']:
            raise ValueError(f'panel checksum mismatch or missing file: {name}')
    return json.loads((folder / 'panel.json').read_text())


def pair_set(n_hap, specification):
    """Match upstream gamma_smc ordering for -w, -S/-T, or full decoding."""
    if n_hap < 2 or n_hap % 2:
        raise ValueError('an even positive number of haplotypes is required')
    n = n_hap // 2
    if specification == 'within':
        pairs = [(2*i, 2*i+1) for i in range(n)]
        first, second = list(range(n)), []
    elif specification == 'all':
        pairs = []
        for i in range(n):
            pairs.append((2*i, 2*i+1))
            for j in range(i+1, n):
                pairs.extend([(2*i, 2*j), (2*i, 2*j+1), (2*i+1, 2*j), (2*i+1, 2*j+1)])
        first, second = list(range(n)), []
    elif specification.startswith('cross:'):
        k = int(specification.split(':')[1])
        if not 0 < k < n:
            raise ValueError('cross:k requires 0 < k < number of diploid samples')
        first, second = list(range(k, n)), list(range(k))
        pairs = [(a, b) for i in first for j in second for a in (2*i, 2*i+1) for b in (2*j, 2*j+1)]
    else:
        raise ValueError('pair-set must be within, cross:k, or all')
    if len({tuple(sorted(p)) for p in pairs}) != len(pairs) or any(a == b for a, b in pairs):
        raise ValueError('pair set is not a collection of distinct nonself pairs')
    return np.asarray(pairs, dtype=np.int32), first, second


def prepare(args, output):
    start = time.perf_counter()
    if getattr(args, 'cache_dir', None):
        sys.path.insert(0, str(REPO/'analysis/genome_wide'))
        from rerun_support import load_input
        data, source = load_input(args.cache_dir, args.chromosome)
        if source.get('format') == 'extracted_npy':
            record = source['record']
            recorded_base = record.get('positions_base')
            if recorded_base is not None and recorded_base != args.positions_base:
                raise ValueError('requested positions base differs from immutable cache metadata')
            folder = Path(source['ready_manifest']['path']).parent
            for name, entry in record['members'].items():
                if sha256(folder/name) != entry['sha256']:
                    raise ValueError(f'source cache checksum mismatch: {name}')
    else:
        with np.load(args.parsed_npz, allow_pickle=True) as archive:
            data = {key: archive[key] for key in ['G', 'positions', 'sample_ids']}
        source = {'format': 'npz', 'path': str(Path(args.parsed_npz).resolve()), 'sha256': sha256(args.parsed_npz)}
    G, positions, ids = data['G'], data['positions'], data['sample_ids'].astype(str)
    if G.ndim != 2 or G.shape[0] != 2*len(ids) or len(set(ids)) != len(ids):
        raise ValueError('inconsistent or duplicate sample identities')
    populations = {}
    with open(args.samples_file) as stream:
        for line in stream:
            fields = line.split()
            if len(fields) >= 7:
                populations[fields[1]] = fields[5]
    selected = [i for i, name in enumerate(ids) if populations.get(name) == args.population]
    if not selected:
        raise ValueError(f'no {args.population} samples found')
    G = np.asarray(G[np.asarray([2*i+j for i in selected for j in (0, 1)])])
    positions = np.asarray(positions)
    if G.shape[1] != len(positions) or not np.all(np.isin(G, [0, 1])):
        raise ValueError('parsed panel must contain complete phased binary genotypes')
    G = G.astype(np.int8)
    if not np.all(positions == np.floor(positions)):
        raise ValueError('integer genomic positions are required')
    positions = positions.astype(np.int64) - args.positions_base
    if np.any(positions < 0) or np.any(np.diff(positions) <= 0):
        raise ValueError('positions must be strictly increasing and nonnegative after base conversion')
    n_before = len(positions)
    # Exactly one population-level filter; never refilter when changing pairs.
    keep = (G.min(axis=0) == 0) & (G.max(axis=0) == 1)
    G, positions = np.ascontiguousarray(G[:, keep]), positions[keep]
    if len(positions) < 2:
        raise ValueError('at least two retained markers are required')
    names = [str(ids[i]) for i in selected]
    np.save(output / 'G.npy', G)
    np.save(output / 'positions.npy', positions)
    np.save(output / 'source_column_indices.npy', np.flatnonzero(keep))
    write_json(output / 'sample_ids.json', names)
    if any(re.search(r'\s', name) for name in names):
        raise ValueError('sample IDs may not contain whitespace')
    conversion_start = time.perf_counter()
    vcf = output / 'panel.vcf'
    with open(vcf, 'w') as stream:
        stream.write('##fileformat=VCFv4.2\n')
        stream.write(f'##contig=<ID={args.chromosome},length={int(positions[-1])+1}>\n')
        stream.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotype">\n')
        stream.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(names) + '\n')
        for site, position in enumerate(positions):
            gt = '\t'.join(f'{G[2*i,site]}|{G[2*i+1,site]}' for i in range(len(names)))
            stream.write(f'{args.chromosome}\t{position+1}\tSNP_{site}\tA\tT\t.\tPASS\t.\tGT\t{gt}\n')
    with open(output / 'panel.vcf.gz', 'wb') as stream:
        subprocess.run(['bgzip', '-c', str(vcf)], stdout=stream, check=True)
    subprocess.run(['tabix', '-p', 'vcf', str(output / 'panel.vcf.gz')], check=True)
    vcf.unlink()
    vcf_seconds = time.perf_counter() - conversion_start
    conversion_start = time.perf_counter()
    root = output / 'asmc'
    with gzip.open(str(root)+'.hap.gz', 'wt') as haps, gzip.open(str(root)+'.map.gz', 'wt') as genetic_map:
        for site, position in enumerate(positions):
            # ASMC/Oxford and VCF use one-based positions; GPU/CPU use zero-based.
            pos1 = int(position) + 1
            alleles = ' '.join(str(int(x)) for x in G[:, site])
            haps.write(f'{args.chromosome}:{pos1}_1_2 SNP_{site} {pos1} 1 2 {alleles}\n')
            genetic_map.write(f'{args.chromosome}\tSNP_{site}\t{pos1*args.rho*100:.14g}\t{pos1}\n')
    with open(str(root)+'.samples', 'w') as stream:
        stream.write('ID_1 ID_2 missing\n0 0 0\n')
        for name in names:
            stream.write(f'{name} {name} 0\n')
    asmc_seconds = time.perf_counter() - conversion_start
    length = int(positions[-1]) + 1
    theta = float(np.mean(np.count_nonzero(G[0::2] != G[1::2], axis=1) / length))
    if theta <= 0:
        raise ValueError('within-diploid theta must be positive')
    info = {'schema_version': 1, 'population': args.population, 'chromosome': args.chromosome,
            'n_haplotypes': len(G), 'n_sites': len(positions), 'n_sites_before_population_filter': n_before,
            'n_all_pairs': len(G)*(len(G)-1)//2, 'positions_base': 0,
            'input_positions_base': args.positions_base, 'physical_mu': args.mu, 'physical_rho': args.rho,
            'theta': theta, 'scaled_recombination_rate': theta*args.rho/args.mu,
            'generations_per_coalescent_unit': theta/(2*args.mu), 'calibration_sequence_length': length,
            'filter': 'complete binary genotypes, both alleles observed in full population; fixed across all methods and pair counts',
            'source_cache': source,
            'samples_file': {'path': str(Path(args.samples_file).resolve()), 'sha256': sha256(args.samples_file)},
            'prepare_seconds': time.perf_counter()-start, 'vcf_conversion_seconds': vcf_seconds,
            'asmc_conversion_seconds': asmc_seconds, 'source': source_metadata()}
    write_json(output / 'panel.json', info)
    write_json(output / 'manifest.json', manifest(output))
    return info


def timer_distribution(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'seconds': a.tolist(), 'median': float(np.median(a)), 'min': float(a.min()),
            'max': float(a.max()), 'q25': float(np.quantile(a, .25)), 'q75': float(np.quantile(a, .75))}


def optional_timer_distribution(records, key):
    values = [r.get(key) for r in records]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f'inconsistent timing availability: {key}')
    return timer_distribution(values)


def parse_cpu_timers(stdout, n_hap, n_sites, n_pairs):
    clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', stdout)
    for expression, value in [(r'Read (\d+) samples', n_hap//2), (r'Read (\d+) segregating sites', n_sites),
                               (r'Applying to (\d+) haplotype pairs', n_pairs)]:
        matches = re.findall(expression, clean)
        if len(matches) != 1 or int(matches[0]) != value:
            raise ValueError(f'CPU input-workload mismatch for {expression}: {matches}, expected {value}')
    timers = {}
    for label in ['Emissions preparation', 'Forward pass', 'Backward pass', 'Output']:
        matches = re.findall(re.escape(label)+r' time:\s*([0-9.]+) secs', clean)
        if len(matches) != 1:
            raise ValueError(f'CPU timing missing: {label}')
        timers[label.lower().replace(' ', '_')+'_seconds'] = float(matches[0])
    timers['decode_seconds'] = sum(timers[k] for k in ['emissions_preparation_seconds', 'forward_pass_seconds', 'backward_pass_seconds'])
    return timers


def cpu_repetition(args, panel, info, pairs, first, second, directory):
    binary, flow = Path(args.gamma_smc_bin).resolve(), Path(args.flow_field).resolve()
    command = [str(binary), '-i', str(panel / 'panel.vcf.gz'), '-o', str(directory / 'posteriors.zst'),
               '-m', str(info['theta']), '-r', str(info['scaled_recombination_rate']), '-f', str(flow), '-h']
    if args.pair_set == 'within':
        command.append('-w')
    elif second:
        names = json.loads((panel / 'sample_ids.json').read_text())
        for filename, indices in [('first.txt', first), ('second.txt', second)]:
            (directory / filename).write_text(''.join(names[i]+'\n' for i in indices))
        command += ['-S', str(directory/'first.txt'), '-T', str(directory/'second.txt')]
    start = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    elapsed = time.perf_counter()-start
    (directory/'stdout.txt').write_text(process.stdout)
    (directory/'stderr.txt').write_text(process.stderr)
    write_json(directory/'command.json', {'command': command, 'returncode': process.returncode})
    if process.returncode:
        raise RuntimeError(f'CPU subprocess failed; inspect {directory}')
    timers = parse_cpu_timers(process.stdout, info['n_haplotypes'], info['n_sites'], len(pairs))
    meta = json.loads((directory/'posteriors.zst.meta').read_text())
    np.testing.assert_array_equal(meta['output_positions'], np.load(panel/'positions.npy'))
    np.testing.assert_array_equal(meta['pairs'], pairs)
    if not np.isclose(meta['scaled_mutation_rate'], info['theta'], rtol=1e-6, atol=5.1e-11):
        raise ValueError('CPU scaled mutation rate differs from common panel parameter')
    if not np.isclose(meta['scaled_recombination_rate'], info['scaled_recombination_rate'], rtol=1e-6, atol=5.1e-11):
        raise ValueError('CPU scaled recombination rate differs from common panel parameter')
    return {'native_total_seconds': elapsed, **timers,
            'setup_and_uninstrumented_seconds': elapsed-timers['decode_seconds']-timers['output_seconds'],
            'timing_scope': 'subprocess wall includes input parsing, flow initialization, posterior decoding and zstd file output; upstream decode excludes output',
            'posterior_format': 'upstream compressed float32 alpha/beta, coalescent units'}


def gpu_repetition(args, panel, info, pairs, directory):
    sys.path.insert(0, str(REPO/'python'))
    if args.gpu_mode == 'export':
        return gpu_export_repetition(args, panel, info, pairs, directory)
    import gamma_smc_cu
    if args.gpu_mode == 'streaming':
        return gpu_streaming_repetition(args, panel, info, pairs, directory)
    start = time.perf_counter()
    G, positions = np.load(panel/'G.npy'), np.load(panel/'positions.npy')
    load_seconds = time.perf_counter()-start
    decode_seconds, output_seconds, metadata = 0., 0., None
    n_batches = (len(pairs)+args.gpu_pair_batch-1)//args.gpu_pair_batch
    for batch, lo in enumerate(range(0, len(pairs), args.gpu_pair_batch)):
        subset = pairs[lo:lo+args.gpu_pair_batch].tolist()
        kwargs = dict(mu=info['theta']/40000, rho=info['scaled_recombination_rate']/40000,
                      Ne=10000, physical_mu=info['physical_mu'], auto_estimate_theta=False,
                      pairs=subset, flow_field_path=args.flow_field, mean_only=True)
        if args.gpu_mode == 'blockwise':
            kwargs.update(core_block_sites=args.core_sites, flank_sites=args.flank_sites,
                          pair_batch_size=args.gpu_pair_batch, max_streams=1)
            inference = gamma_smc_cu.infer_blockwise
        else:
            inference = gamma_smc_cu.infer
        t0 = time.perf_counter()
        result = inference(G, positions, **kwargs)
        decode_seconds += time.perf_counter()-t0
        np.testing.assert_array_equal(result['positions'], positions)
        values = np.asarray(result['mean'])
        if values.shape != (len(positions), len(subset)) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError('invalid GPU posterior means')
        metadata = result['metadata']
        if metadata.get('time_units') != 'generations':
            raise ValueError('requires physically calibrated GPU API')
        t0 = time.perf_counter()
        np.save(directory/f'mean_batch{batch:05d}.npy', values)
        output_seconds += time.perf_counter()-t0
        del result, values
    elapsed = time.perf_counter()-start
    write_json(directory/'calibration.json', metadata)
    return {'native_total_seconds': elapsed, 'input_load_seconds': load_seconds,
            'decode_seconds': decode_seconds, 'output_seconds': output_seconds, 'n_batches': n_batches,
            'timing_scope': 'wall from prepared .npy load through host posterior means and .npy writes; decode sums public API calls including preprocessing, allocations, decoding and host transfer; no kernel-only timer',
            'posterior_format': 'float posterior mean .npy per pair batch, generations'}


def gpu_export_repetition(args, panel, info, pairs, directory):
    import gamma_smc_cu
    start = time.perf_counter()
    G, positions = np.load(panel/'G.npy'), np.load(panel/'positions.npy')
    load_seconds = time.perf_counter()-start
    if args.core_sites < len(positions) or args.flank_sites != 0:
        raise ValueError('export requires full-sequence context and zero flanks')
    exported = gamma_smc_cu.export_dense(
        G, positions, pairs=pairs, output_dir=directory,
        pair_batch_size=args.gpu_pair_batch, gpu_ids=args.gpu_ids,
        tile_sites=args.checkpoint_sites or 4096, staging_buffers=args.export_buffers,
        workers_per_gpu=args.export_workers_per_gpu,
        numa_local=not args.no_numa_local, durable=args.durable_output,
        mu=info['theta']/40000, rho=info['scaled_recombination_rate']/40000, Ne=10000,
        physical_mu=info['physical_mu'], auto_estimate_theta=False,
        flow_field_path=args.flow_field)
    elapsed = time.perf_counter()-start
    if exported['shape'] != [len(positions), len(pairs)] or exported['calibration']['time_units'] != 'generations':
        raise ValueError('Unexpected exported shape or physical calibration')
    write_json(directory/'calibration.json', exported['calibration'])
    return dict(native_total_seconds=elapsed, input_load_seconds=load_seconds,
                decode_seconds=None, output_seconds=None, n_batches=len(exported['files']),
                export=exported,
                timing_scope='wall includes input load, context setup, full-context inference, GPU validation, transfers and all NPY writes; stage timers overlap; fsync included only when durable_output is true',
                posterior_format='float32 posterior mean .npy per pair batch, generations')


def gpu_streaming_repetition(args, panel, info, pairs, directory):
    from contextlib import closing
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    import gamma_smc_cu
    start = time.perf_counter()
    G, positions = np.load(panel/'G.npy'), np.load(panel/'positions.npy')
    load_seconds = time.perf_counter()-start
    if args.core_sites < len(positions) or args.flank_sites != 0:
        raise ValueError('streaming requires one full-sequence block and zero flanks')
    stream = gamma_smc_cu.iter_infer_batches(
        G, positions, pairs=pairs.tolist(), pair_batch_size=args.gpu_pair_batch,
        gpu_ids=args.gpu_ids, checkpoint_sites=args.checkpoint_sites,
        mu=info['theta']/40000,
        rho=info['scaled_recombination_rate']/40000, Ne=10000,
        physical_mu=info['physical_mu'], auto_estimate_theta=False,
        flow_field_path=args.flow_field, mean_only=True)
    output_seconds, wait_seconds, n_batches, metadata = 0., 0., 0, None
    pending_writes = deque()

    def write_batch(path, values):
        t0 = time.perf_counter()
        np.save(path, values)
        return time.perf_counter()-t0

    with closing(stream), ThreadPoolExecutor(max_workers=args.output_workers) as writers:
        while True:
            t0 = time.perf_counter()
            try:
                result = next(stream)
            except StopIteration:
                break
            wait_seconds += time.perf_counter()-t0
            values = np.asarray(result['mean'])
            np.testing.assert_array_equal(result['positions'], positions)
            offset = result['pair_offset']
            np.testing.assert_array_equal(result['pairs'], pairs[offset:offset+values.shape[1]])
            if (values.shape[0] != len(positions) or
                    (not result.get('posterior_means_validated', False) and
                     (not np.isfinite(values).all() or np.any(values <= 0)))):
                raise ValueError('invalid GPU posterior means')
            metadata = result['metadata']
            if metadata['time_units'] != 'generations':
                raise ValueError('requires physically calibrated GPU API')
            pending_writes.append(writers.submit(write_batch,
                directory/f'mean_batch{n_batches:05d}.npy', values))
            if len(pending_writes) >= args.output_workers:
                output_seconds += pending_writes.popleft().result()
            n_batches += 1
            del result, values
        while pending_writes:
            output_seconds += pending_writes.popleft().result()
    elapsed = time.perf_counter()-start
    write_json(directory/'calibration.json', metadata)
    return dict(native_total_seconds=elapsed, input_load_seconds=load_seconds,
                decode_seconds=None, consumer_wait_seconds=wait_seconds,
                output_seconds=output_seconds, n_batches=n_batches,
                timing_scope='wall includes input load, context setup, full-context inference, host transfer, validation and all NPY writes; output_seconds sums writer time and may overlap; consumer wait is not decode time',
                posterior_format='float posterior mean .npy per pair batch, generations')


def shard_pairs(pairs, workers, n_haplotypes):
    """Contiguous, balanced shards preserve every original oriented pair once."""
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu':
        raise ValueError('pairs require an integer array with shape (n, 2)')
    if workers < 1 or workers > len(pairs):
        raise ValueError('workers must be positive and cannot exceed pair count')
    if (np.any(pairs < 0) or np.any(pairs >= n_haplotypes)
            or np.any(pairs[:, 0] == pairs[:, 1])
            or len({tuple(sorted(p)) for p in pairs.tolist()}) != len(pairs)):
        raise ValueError('pairs must be distinct, nonself and within the full cohort')
    return [part.copy() for part in np.array_split(pairs, workers)]


def select_physical_cpus(topology, allowed, workers):
    """One allowed logical CPU per (socket, physical core), never its SMT peer."""
    allowed = set(allowed)
    by_cpu = {}
    for line in topology.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.strip().split(',')
        if len(fields) != 3:
            raise ValueError('expected lscpu -p=CPU,CORE,SOCKET topology')
        try:
            cpu, core, socket = map(int, fields)
        except ValueError as exc:
            raise ValueError('physical CPU topology is unavailable') from exc
        if cpu in by_cpu:
            raise ValueError('duplicate CPU in topology')
        by_cpu[cpu] = (socket, core)
    if not allowed or not allowed.issubset(by_cpu):
        raise ValueError('topology does not describe all CPUs in the allowed affinity')
    seen, chosen = set(), []
    for cpu in sorted(allowed):
        identity = by_cpu[cpu]
        if identity not in seen:
            seen.add(identity)
            chosen.append(cpu)
    if workers < 1 or len(chosen) < workers:
        raise ValueError(f'insufficient allocated physical cores: {len(chosen)} available, {workers} workers requested')
    return chosen[:workers]


def asmc_affinity_plan(workers):
    if not hasattr(os, 'sched_getaffinity') or not hasattr(os, 'sched_setaffinity'):
        if workers != 1:
            raise ValueError('multiple ASMC workers require enforceable physical-core affinity')
        return {'selected_cpus': [None], 'allowed_cpus': None,
                'status': 'single worker; CPU affinity unavailable on this platform'}
    allowed = sorted(os.sched_getaffinity(0))
    topology = subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET'], text=True)
    selected = select_physical_cpus(topology, allowed, workers)
    # A Slurm task may have a wider affinity than its requested CPU count.
    # Never infer a larger allocation from that broader mask.
    requested = os.environ.get('SLURM_CPUS_PER_TASK')
    if requested and workers > int(requested):
        raise ValueError('ASMC workers exceed SLURM_CPUS_PER_TASK')
    return {'selected_cpus': selected, 'allowed_cpus': allowed, 'lscpu_parse': topology,
            'status': 'one allowed logical CPU per distinct physical core'}


def asmc_binary_provenance(module):
    files = {}
    for name in {module.__name__, module.ASMC.__module__}:
        imported = importlib.import_module(name)
        if getattr(imported, '__file__', None):
            path = Path(imported.__file__).resolve()
            for source in [path, *path.parent.glob('*.so'), *path.parent.glob('*.pyd')]:
                files[str(source)] = sha256(source)
    try:
        version = importlib.metadata.version('asmc-asmc')
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {'asmc_version': version, 'module_files_sha256': files,
            'constructor_source_sha256': sha256(HERE/'tune_asmc.py')}


def validate_asmc_means(values, returned_pairs, subset, n_sites):
    if values.shape != (len(subset), n_sites) or values.dtype != np.float32:
        raise ValueError('ASMC posterior means have unexpected shape or dtype')
    np.testing.assert_array_equal(returned_pairs, subset)
    for lo in range(0, n_sites, 8192):
        block = values[:, lo:lo+8192]
        if not np.isfinite(block).all() or np.any(block <= 0):
            raise ValueError('ASMC posterior means must be finite and positive')


def asmc_worker(job, module=None):
    """Decode only this pair shard, retaining the full input cohort and markers."""
    from tune_asmc import create_asmc
    previous_affinity = None
    if job['cpu'] is not None:
        previous_affinity = os.sched_getaffinity(0)
        if job['cpu'] not in previous_affinity:
            raise ValueError('requested worker CPU is outside inherited allocation affinity')
        os.sched_setaffinity(0, {job['cpu']})
        if os.sched_getaffinity(0) != {job['cpu']}:
            os.sched_setaffinity(0, previous_affinity)
            raise ValueError('worker CPU affinity was not applied')
    try:
        module = module or importlib.import_module('asmc.asmc')
        provenance = asmc_binary_provenance(module)
        panel, directory = Path(job['panel']), Path(job['directory'])
        info, pairs = job['info'], np.load(job['pairs_file'])
        profile = {'constructor': 'explicit', 'extraction': 'ref', 'changed_model': False,
                   'outer_batch': job['outer_batch'], 'native_batch': job['native_batch']}
        start, cpu_start = time.perf_counter(), time.process_time()
        asmc, params = create_asmc(module, profile, panel, Path(job['dq']), directory)
        setup_seconds = time.perf_counter()-start
        if not params.get('useKnownSeed') or params.get('jobs') != 1 or params.get('jobInd') != 1:
            raise ValueError('ASMC workers require a fixed CSFS seed and jobs=1/jobInd=1 on the full cohort')
        t0 = time.perf_counter()
        if asmc.get_haploid_sample_size() != info['n_haplotypes'] or asmc.get_num_sites() != info['n_sites']:
            raise ValueError('ASMC loaded a different cohort or marker count')
        np.testing.assert_array_equal(asmc.get_physical_positions(), np.load(panel/'positions.npy')+1)
        expected_times = np.asarray(asmc.get_expected_times())
        if expected_times.ndim != 1 or not len(expected_times) or not np.isfinite(expected_times).all() or np.any(expected_times <= 0):
            raise ValueError('invalid ASMC expected coalescent times')
        validation_seconds = time.perf_counter()-t0
        native_decode = native_decode_cpu = extraction = output_seconds = 0.
        call_sizes = []
        for batch, lo in enumerate(range(0, len(pairs), job['outer_batch'])):
            subset = pairs[lo:lo+job['outer_batch']]
            a, b = subset[:, 0].tolist(), subset[:, 1].tolist()
            t0, cpu0 = time.perf_counter(), time.process_time()
            asmc.decode_pairs(a, b)
            native_decode += time.perf_counter()-t0
            native_decode_cpu += time.process_time()-cpu0
            t0 = time.perf_counter()
            result = asmc.get_ref_of_results()
            values = np.asarray(result.per_pair_posterior_means)
            # Binding returns (hap_i, id_i, hap_j, id_j), not two columns.
            returned = np.asarray([(row[0], row[2]) for row in result.per_pair_indices], dtype=np.int64)
            extraction += time.perf_counter()-t0
            t0 = time.perf_counter()
            validate_asmc_means(values, returned, subset, info['n_sites'])
            validation_seconds += time.perf_counter()-t0
            t0 = time.perf_counter()
            np.save(directory/f'mean_batch{batch:05d}.npy', values.T)
            output_seconds += time.perf_counter()-t0
            call_sizes.append(len(subset))
            # get_ref borrows buffers: save and release every view before decode.
            del values, returned, result
        t0 = time.perf_counter()
        np.save(directory/'expected_coalescent_times.npy', expected_times)
        output_seconds += time.perf_counter()-t0
        elapsed, cpu_work = time.perf_counter()-start, time.process_time()-cpu_start
        return {'worker_index': job['worker_index'], 'pid': os.getpid(),
                'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                'thread_environment': {key: os.environ.get(key) for key in
                                       ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
                'pair_start': job['pair_start'], 'n_pairs': len(pairs), 'pairs_sha256': sha256(job['pairs_file']),
                'native_total_seconds': elapsed, 'setup_seconds': setup_seconds,
                'native_decode_seconds': native_decode, 'extraction_seconds': extraction,
                'decode_seconds': native_decode+extraction, 'validation_seconds': validation_seconds,
                'output_seconds': output_seconds, 'cpu_work_seconds': cpu_work,
                'native_decode_cpu_seconds': native_decode_cpu, 'actual_params': params,
                'native_states': len(expected_times), 'provenance': provenance,
                'outer_call_pair_counts': call_sizes, 'profile': profile}
    finally:
        if previous_affinity is not None:
            os.sched_setaffinity(0, previous_affinity)


def asmc_process_entry(job, results):
    try:
        results.put((job['worker_index'], asmc_worker(job), None))
    except BaseException:
        results.put((job['worker_index'], None, traceback.format_exc()))
        raise


def spawned_asmc_workers(jobs, timeout):
    """Dedicated spawned processes, one per shard; fail the whole repetition."""
    context = multiprocessing.get_context('spawn')
    results = context.Queue()
    children, completed = [], {}
    deadline = time.monotonic()+timeout
    thread_keys = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
    old_environment = {key: os.environ.get(key) for key in thread_keys}
    try:
        # Applied before fresh interpreters import NumPy; parent is restored below.
        os.environ.update({key: '1' for key in thread_keys})
        for job in jobs:
            process = context.Process(target=asmc_process_entry, args=(job, results))
            process.start()
            children.append(process)
        while len(completed) != len(jobs):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('ASMC worker wall-time limit exceeded')
            try:
                index, record, error = results.get(timeout=min(1., remaining))
            except queue.Empty:
                if any(process.exitcode not in (None, 0) for process in children):
                    raise RuntimeError('ASMC worker exited without a successful result')
                if all(process.exitcode is not None for process in children):
                    raise RuntimeError('ASMC worker exited without reporting a result')
                continue
            if error:
                raise RuntimeError(f'ASMC worker {index} failed:\n{error}')
            if index in completed or index not in range(len(jobs)):
                raise RuntimeError('invalid or duplicate ASMC worker result')
            completed[index] = record
        for process in children:
            process.join(max(0., deadline-time.monotonic()))
            if process.exitcode != 0:
                raise RuntimeError('ASMC worker failed or did not exit within wall-time limit')
        return [completed[i] for i in range(len(jobs))]
    finally:
        for process in children:
            if process.is_alive():
                process.terminate()
            process.join()
        results.close()
        results.join_thread()
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def summarize_asmc_workers(records, parallel_wall_seconds, affinity):
    if not records:
        raise ValueError('no ASMC worker results')
    first = records[0]
    for record in records[1:]:
        for key in ['actual_params', 'native_states', 'provenance']:
            if record[key] != first[key]:
                raise ValueError(f'ASMC workers differ in {key}')
    parallel = len(records) > 1
    stage_keys = ['setup_seconds', 'native_decode_seconds', 'extraction_seconds',
                  'decode_seconds', 'validation_seconds', 'output_seconds']
    return {'native_total_seconds': parallel_wall_seconds if parallel else first['native_total_seconds'],
            **{key: None if parallel else first[key] for key in stage_keys},
            'worker_stage_wall_sums': {key: sum(r[key] for r in records) for key in stage_keys},
            'cpu_work_seconds': sum(r['cpu_work_seconds'] for r in records),
            'native_decode_cpu_seconds': sum(r['native_decode_cpu_seconds'] for r in records),
            'workers': records, 'n_workers': len(records), 'affinity_plan': affinity,
            'execution': 'spawned_processes' if parallel else 'in_process_single_worker',
            'actual_params': first['actual_params'], 'native_states': first['native_states'],
            'provenance': first['provenance'],
            'timing_scope': ('parent wall from process spawn through successful worker joins; includes interpreter/import startup, provenance checks, full-panel initialization per worker, decoding, validation and NPY writes; stage sums are worker work, NOT parallel elapsed time'
                             if parallel else 'wall from ASMC initialization on the prepared full Oxford panel through means, validation and NPY writes; decode is native_decode plus borrowed-result extraction; setup, validation and output are separate'),
            'cpu_work_scope': 'sum of worker process_time during native_total scope; CPU seconds, not wall time; excludes spawned interpreter/import startup',
            'posterior_format': 'posterior mean .npy (sites, pairs) per worker/batch, native DQ generation units'}


def asmc_repetition(args, panel, info, pairs, directory):
    workers = args.asmc_workers
    shards = shard_pairs(pairs, workers, info['n_haplotypes'])
    affinity = asmc_affinity_plan(workers)
    jobs, offset = [], 0
    for index, (shard, cpu) in enumerate(zip(shards, affinity['selected_cpus'])):
        folder = directory/f'worker{index:03d}'
        folder.mkdir()
        np.save(folder/'pairs.npy', shard)
        jobs.append({'worker_index': index, 'cpu': cpu, 'pair_start': offset,
                     'panel': str(panel), 'info': info, 'directory': str(folder),
                     'pairs_file': str(folder/'pairs.npy'), 'dq': str(Path(args.decoding_quantities).resolve()),
                     'outer_batch': args.asmc_pair_batch, 'native_batch': args.asmc_native_batch})
        offset += len(shard)
    start = time.perf_counter()
    records = [asmc_worker(jobs[0])] if workers == 1 else spawned_asmc_workers(jobs, args.timeout)
    wall_seconds = time.perf_counter()-start
    for job, record in zip(jobs, records):
        if (record['worker_index'] != job['worker_index'] or record['pair_start'] != job['pair_start']
                or record['n_pairs'] != len(np.load(job['pairs_file']))
                or record['pairs_sha256'] != sha256(job['pairs_file'])):
            raise ValueError('ASMC worker result does not match assigned pair shard')
    return summarize_asmc_workers(records, wall_seconds, affinity)


def run(args, output):
    panel = Path(args.panel_dir).resolve()
    info = check_panel(panel)
    pairs, first, second = pair_set(info['n_haplotypes'], args.pair_set)
    np.save(output/'pairs.npy', pairs)
    environment = {'source': source_metadata(), 'cpu': captured(['lscpu']),
        'gpu': captured(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv']),
        'hostname': captured(['hostname']), 'python': sys.version, 'python_executable': sys.executable,
        'versions': {}, 'external_files': {}}
    environment['scheduler'] = {k: v for k,v in os.environ.items() if k.startswith('SLURM_') or k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}
    for package in ['numpy', 'asmc-asmc', 'gamma-smc-cu']:
        try:
            environment['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment['versions'][package] = None
    for name in ['flow_field', 'gamma_smc_bin', 'decoding_quantities']:
        value = getattr(args, name)
        if value:
            environment['external_files'][name] = {'path': str(Path(value).resolve()), 'sha256': sha256(value)}
    for path in (REPO/'python/gamma_smc_cu').glob('*.so'):
        environment['external_files'][str(path)] = {'sha256': sha256(path)}
    write_json(output/'environment.json', environment)
    specification = {'schema_version': 2, 'panel_manifest_sha256': sha256(panel/'manifest.json'),
        'panel': info, 'pair_set': args.pair_set, 'pairs_sha256': sha256(output/'pairs.npy'),
        'n_distinct_pairs': len(pairs), 'coverage': 'all_pairs' if len(pairs)==info['n_all_pairs'] else 'measured_distinct_pair_subset',
        'is_extrapolated': False, 'method': args.method, 'arguments': vars(args),
        'native_timing_caveat': 'setup/output formats and API boundaries differ; preserve and report native_total and decode scopes separately, never call decode timings identical kernel measurements'}
    write_json(output/'specification.json', specification)
    runner = {'cpu': cpu_repetition, 'gpu': gpu_repetition, 'asmc': asmc_repetition}[args.method]
    records = []
    for repetition in range(-1, args.repeats):
        directory = output/('warmup' if repetition == -1 else f'rep{repetition:02d}')
        if args.method != 'gpu' or args.gpu_mode != 'export':
            directory.mkdir()
        if args.method == 'cpu':
            record = runner(args, panel, info, pairs, first, second, directory)
        else:
            record = runner(args, panel, info, pairs, directory)
        record['repetition'] = repetition
        record['warmup'] = repetition == -1
        record['artifacts'] = manifest(directory)
        write_json(directory/'timing.json', record)
        records.append(record)
        # Always retain one measured set of predictions; other repetitions have
        # native output generated/timed/hashed identically before removing it.
        if not args.retain_all_predictions and repetition != 0:
            for path in directory.rglob('mean_batch*.npy'):
                path.unlink()
            if (directory/'posteriors.zst').exists():
                (directory/'posteriors.zst').unlink()
        write_json(output/'progress.json', records)
        print(json.dumps({'method': args.method, 'pair_set': args.pair_set, 'repetition': repetition,
                          'native_total_seconds': record['native_total_seconds']}), flush=True)
    measured = [record for record in records if not record['warmup']]
    result = {**specification, 'status': 'ok', 'warmup': records[0], 'repetitions': measured,
              'native_total': timer_distribution([r['native_total_seconds'] for r in measured]),
              'decode': optional_timer_distribution(measured, 'decode_seconds'),
              'output': optional_timer_distribution(measured, 'output_seconds')}
    if args.method == 'asmc':
        result['asmc_timing'] = {key: optional_timer_distribution(measured, key)
                                 for key in ['setup_seconds', 'native_decode_seconds', 'extraction_seconds',
                                             'validation_seconds', 'cpu_work_seconds', 'native_decode_cpu_seconds']}
        result['parallel_timing_caveat'] = 'Multicore decode/output stage wall summaries are null; only native_total is measured parallel elapsed time. Worker stage sums and process CPU seconds are not parallel wall times.'
    write_json(output/'result.json', result)
    write_json(output/'manifest.json', manifest(output))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    prep = actions.add_parser('prepare', help='freeze full marker panel once before choosing pairs/methods')
    inputs = prep.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--parsed-npz', help='legacy input only if complete, strictly increasing and duplicate-free')
    inputs.add_argument('--cache-dir', help='new cache root containing parsed/chrN/READY.json and memory-mapped arrays')
    prep.add_argument('--samples-file', required=True)
    prep.add_argument('--positions-base', required=True, type=int, choices=[0, 1])
    prep.add_argument('--population', default='YRI')
    prep.add_argument('--chromosome', default='22')
    prep.add_argument('--mu', type=float, default=1.25e-8)
    prep.add_argument('--rho', type=float, default=1e-8)
    execution = actions.add_parser('run', help='one method and exact pair set, warmup plus timed repetitions')
    execution.add_argument('--panel-dir', required=True)
    execution.add_argument('--method', required=True, choices=['gpu','cpu','asmc'])
    execution.add_argument('--pair-set', default='within', help='within, cross:k, or all; no repeated pairs')
    execution.add_argument('--repeats', type=int, default=3)
    execution.add_argument('--flow-field')
    execution.add_argument('--gamma-smc-bin')
    execution.add_argument('--decoding-quantities')
    execution.add_argument('--gpu-mode', choices=['full','blockwise','streaming','export'], default='full')
    execution.add_argument('--gpu-ids', nargs='+', type=int,
                           help='Visible CUDA devices for streaming/export; default is current device')
    execution.add_argument('--gpu-pair-batch', type=int, default=256)
    execution.add_argument('--checkpoint-sites', type=int,
                           help='Exact checkpoint/replay tile size for streaming/export')
    execution.add_argument('--output-workers', type=int, default=1,
                           help='Bounded concurrent NPY writers in GPU streaming mode')
    execution.add_argument('--export-buffers', type=int, default=3,
                           help='Pinned staging slots per export worker (2..8)')
    execution.add_argument('--export-workers-per-gpu', type=int, default=2,
                           help='Concurrent batch files per GPU in export mode')
    execution.add_argument('--no-numa-local', action='store_true',
                           help='Disable GPU-local CPU placement in export mode')
    execution.add_argument('--durable-output', action='store_true',
                           help='Include file/directory fsync in export mode')
    execution.add_argument('--asmc-pair-batch', type=int, default=32)
    execution.add_argument('--asmc-native-batch', type=int, default=32,
                           help='ASMC internal SIMD batch, independently configured from Python call size')
    execution.add_argument('--asmc-workers', type=int, default=1,
                           help='Spawned ASMC pair shards pinned to distinct allocated physical CPU cores')
    execution.add_argument('--core-sites', type=int, default=65536)
    execution.add_argument('--flank-sites', type=int, default=8192)
    execution.add_argument('--timeout', type=float, default=86400)
    execution.add_argument('--retain-all-predictions', action='store_true')
    for command in [prep, execution]:
        command.add_argument('--output-dir', required=True, help='new directory only; never overwritten')
    args = parser.parse_args()
    if args.action == 'prepare' and (args.mu <= 0 or args.rho < 0):
        parser.error('mu must be positive and rho nonnegative')
    if args.action == 'run':
        if args.gpu_ids is not None and (args.method != 'gpu' or args.gpu_mode not in ('streaming', 'export')):
            parser.error('--gpu-ids requires GPU streaming or export mode')
        if args.checkpoint_sites is not None and (args.checkpoint_sites < 1 or
                args.method != 'gpu' or args.gpu_mode not in ('streaming', 'export')):
            parser.error('--checkpoint-sites requires a positive value and GPU streaming or export mode')
        if not 2 <= args.export_buffers <= 8:
            parser.error('--export-buffers must be between 2 and 8')
        if args.export_workers_per_gpu < 1:
            parser.error('--export-workers-per-gpu must be positive')
        if (args.durable_output or args.no_numa_local or args.export_buffers != 3
                or args.export_workers_per_gpu != 2) and (
                args.method != 'gpu' or args.gpu_mode != 'export'):
            parser.error('Direct export options require --method gpu --gpu-mode export')
        if args.output_workers < 1 or (args.output_workers != 1 and
                (args.method != 'gpu' or args.gpu_mode != 'streaming')):
            parser.error('--output-workers requires a positive value and GPU streaming mode')
        if min(args.repeats, args.gpu_pair_batch, args.asmc_pair_batch, args.asmc_native_batch, args.asmc_workers, args.core_sites) < 1 or args.flank_sites < 0:
            parser.error('positive repeats, batch sizes, and core sites; nonnegative flanks required')
        if not np.isfinite(args.timeout) or args.timeout <= 0:
            parser.error('timeout must be finite and positive')
        required = {'gpu': ['flow_field'], 'cpu': ['flow_field','gamma_smc_bin'], 'asmc': ['decoding_quantities']}[args.method]
        for name in required:
            if not getattr(args, name):
                parser.error(f'{args.method} requires --{name.replace("_", "-")}')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        (prepare if args.action == 'prepare' else run)(args, output)
    except BaseException:
        (output/'FAILED.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
