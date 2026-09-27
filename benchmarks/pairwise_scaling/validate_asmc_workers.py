"""Gate native ASMC process parallelism using completed within-pair runs.

Compares every retained rep00 posterior mean in original pair order without
loading chromosome-sized prediction arrays into RAM. Never reruns inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Run:
    def __init__(self, root, label, report):
        self.root, self.label, self.report = Path(root).resolve(), label, report
        require(not (self.root/'FAILED.txt').exists(), f'{label}: run has FAILED.txt')
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        report['input_provenance'][label] = {
            'root': str(self.root), 'manifest_sha256': sha256(self.root/'manifest.json'),
            'verified_artifacts': {}}
        self.result = json.loads(self.verified('result.json').read_text())
        self.environment = json.loads(self.verified('environment.json').read_text())
        result = self.result
        require(result['status'] == 'ok' and result['method'] == 'asmc', f'{label}: not a successful ASMC run')
        require(result.get('is_extrapolated') is False and result['pair_set'] == 'within',
                f'{label}: requires measured within-pair runs')
        self.pairs = np.load(self.verified('pairs.npy'), allow_pickle=False)
        require(sha256(self.root/'pairs.npy') == result['pairs_sha256'], f'{label}: input pair hash mismatch')
        nh, self.ns = result['panel']['n_haplotypes'], result['panel']['n_sites']
        require(nh > 1 and nh % 2 == 0 and self.ns > 0, f'{label}: invalid full panel shape')
        require(self.pairs.dtype.kind in 'iu' and self.pairs.shape == (nh//2, 2), f'{label}: invalid within-pair shape')
        require(np.array_equal(self.pairs, np.arange(nh).reshape(-1, 2)), f'{label}: incomplete or reordered within pairs')
        require(len(self.pairs) == result['n_distinct_pairs'], f'{label}: distinct pair count mismatch')
        self.record = json.loads(self.verified('rep00/timing.json').read_text())
        saved = [r for r in result['repetitions'] if r.get('repetition') == 0]
        require(len(saved) == 1 and saved[0] == self.record and self.record['warmup'] is False,
                f'{label}: retained repetition does not match completed result')
        self.n_workers = self.record['n_workers']
        workers = self.record['workers']
        require(self.n_workers == len(workers) == result['arguments']['asmc_workers'], f'{label}: worker count mismatch')
        self.segments, self.grid, offset = [], None, 0
        for index, worker in enumerate(workers):
            prefix = f'rep00/worker{index:03d}'
            require(worker['worker_index'] == index and worker['pair_start'] == offset,
                    f'{label}: missing, overlapping or out-of-order worker shard')
            require(worker['n_pairs'] > 0, f'{label}: empty worker shard')
            pair_path = self.verified(prefix+'/pairs.npy')
            pairs = np.load(pair_path, allow_pickle=False)
            require(sha256(pair_path) == worker['pairs_sha256'], f'{label}: worker pair hash mismatch')
            stop = offset+worker['n_pairs']
            require(np.array_equal(pairs, self.pairs[offset:stop]), f'{label}: worker pairs differ from exact input order')
            params = worker['actual_params']
            expected = dict(useKnownSeed=True, fixed_upstream_seed=1234, jobs=1, jobInd=1,
                            decodingSequence=True, usingCSFS=True, compress=False, skipCSFSdistance=0,
                            noBatches=False, doPosteriorSums=False, doMajorMinorPosteriorSums=False,
                            doPerPairMAP=False)
            require(all(params.get(key) == value for key, value in expected.items()),
                    f'{label}: worker must use seed1234 and full-CSFS means-only sequence decoding')
            require(params.get('batchSize', 0) > 0, f'{label}: invalid native batch')
            require(params == self.record['actual_params'], f'{label}: workers use different parameters')
            require(worker['provenance'].get('asmc_version') == '1.4.0', f'{label}: ASMC 1.4.0 required')
            require(bool(worker['provenance'].get('module_files_sha256')), f'{label}: missing native binary provenance')
            grid = np.load(self.verified(prefix+'/expected_coalescent_times.npy'), allow_pickle=False)
            require(grid.ndim == 1 and len(grid) == worker['native_states'] == self.record['native_states']
                    and np.isfinite(grid).all() and np.all(grid > 0), f'{label}: invalid state grid')
            require(self.grid is None or np.array_equal(grid, self.grid), f'{label}: workers use different state grids')
            self.grid = grid
            sizes = worker['outer_call_pair_counts']
            require(sizes and all(isinstance(n, int) and n > 0 for n in sizes) and sum(sizes) == len(pairs),
                    f'{label}: output batch counts do not cover the shard')
            paths = sorted((self.root/prefix).glob('mean_batch*.npy'))
            expected_names = [f'mean_batch{i:05d}.npy' for i in range(len(sizes))]
            require([p.name for p in paths] == expected_names, f'{label}: missing or extra posterior batches')
            for path, count in zip(paths, sizes):
                path = self.verified(str(path.relative_to(self.root)))
                matrix = np.load(path, mmap_mode='r', allow_pickle=False)
                require(matrix.shape == (self.ns, count) and matrix.dtype == np.float32,
                        f'{label}: posterior shape/dtype mismatch: {path.name}')
                self.segments.append((offset, offset+count, path))
                offset += count
                del matrix
            require(offset == stop, f'{label}: shard output coverage mismatch')
        require(offset == len(self.pairs), f'{label}: not all input pairs have outputs')
        self.dq_hash = self.environment['external_files']['decoding_quantities']['sha256']
        require(isinstance(self.dq_hash, str) and len(self.dq_hash) == 64, f'{label}: missing DQ hash')
        report['input_provenance'][label].update(
            panel_manifest_sha256=result['panel_manifest_sha256'], pairs_sha256=result['pairs_sha256'],
            decoding_quantities_sha256=self.dq_hash, n_workers=self.n_workers,
            n_pairs=len(self.pairs), n_sites=self.ns, native_states=len(self.grid),
            worker_provenance=[w['provenance'] for w in workers], actual_params=self.record['actual_params'])

    def verified(self, relative):
        path = (self.root/relative).resolve()
        require(path.is_relative_to(self.root), f'{self.label}: artifact escapes run directory')
        entry = self.manifest.get(relative)
        require(entry is not None and path.is_file(), f'{self.label}: missing manifest artifact {relative}')
        digest = sha256(path)
        require(digest == entry['sha256'] and path.stat().st_size == entry['bytes'],
                f'{self.label}: artifact checksum mismatch: {relative}')
        self.report['input_provenance'][self.label]['verified_artifacts'][relative] = digest
        return path


def compare(single, parallel, rtol, atol):
    require(single.n_workers == 1 and parallel.n_workers > 1, 'requires one single worker run and one parallel run')
    require(single.result['panel_manifest_sha256'] == parallel.result['panel_manifest_sha256'], 'panel hash differs')
    require(single.result['pairs_sha256'] == parallel.result['pairs_sha256']
            and np.array_equal(single.pairs, parallel.pairs), 'exact input pairs differ')
    require(single.ns == parallel.ns and single.dq_hash == parallel.dq_hash, 'site count or DQ hash differs')
    require(np.array_equal(single.grid, parallel.grid), 'single/parallel state grids differ')
    metrics = dict(n_values=0, n_outside_tolerance=0, exactly_equal=True,
                   maximum_absolute_difference=0., maximum_relative_difference=0.)
    ai = bi = position = 0
    while position < len(single.pairs):
        alo, ahi, ap = single.segments[ai]
        blo, bhi, bp = parallel.segments[bi]
        stop = min(ahi, bhi)
        a, b = np.load(ap, mmap_mode='r'), np.load(bp, mmap_mode='r')
        for pair_lo in range(position, stop, 32):
            pair_hi = min(pair_lo+32, stop)
            for site_lo in range(0, single.ns, 4096):
                x = np.asarray(a[site_lo:site_lo+4096, pair_lo-alo:pair_hi-alo], dtype=np.float64)
                y = np.asarray(b[site_lo:site_lo+4096, pair_lo-blo:pair_hi-blo], dtype=np.float64)
                require(np.isfinite(x).all() and np.isfinite(y).all() and np.all(x > 0) and np.all(y > 0),
                        'posterior means must all be finite and strictly positive')
                delta = np.abs(y-x)
                metrics['n_values'] += x.size
                metrics['n_outside_tolerance'] += int(np.count_nonzero(delta > atol+rtol*np.abs(x)))
                metrics['exactly_equal'] &= bool(np.array_equal(x, y))
                metrics['maximum_absolute_difference'] = max(metrics['maximum_absolute_difference'], float(delta.max()))
                metrics['maximum_relative_difference'] = max(metrics['maximum_relative_difference'], float((delta/np.abs(x)).max()))
        del a, b
        position = stop
        ai += position == ahi
        bi += position == bhi
    require(metrics['n_values'] == single.ns*len(single.pairs), 'comparison did not cover every posterior mean')
    return metrics


def validate(single, parallel, output, rtol=1e-6, atol=1e-5):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, passed=False, rtol=rtol, atol=atol,
                  compared_repetition='rep00', input_provenance={}, errors=[],
                  validator_sha256=sha256(__file__), memory_scope='memory-mapped inputs; float64 comparison blocks of at most 4096 sites by 32 pairs')
    try:
        require(np.isfinite([rtol, atol]).all() and rtol >= 0 and atol >= 0, 'invalid tolerances')
        one, many = Run(single, 'single', report), Run(parallel, 'parallel', report)
        report['comparison'] = compare(one, many, rtol, atol)
        report['passed'] = report['comparison']['n_outside_tolerance'] == 0
        if not report['passed']:
            report['errors'].append('posterior means differ beyond tolerance')
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    temporary = output/'validation.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(output/'validation.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single', required=True)
    parser.add_argument('--parallel', required=True)
    parser.add_argument('--output-dir', required=True, help='new immutable output directory')
    args = parser.parse_args()
    report = validate(args.single, args.parallel, args.output_dir)
    print(json.dumps({'passed': report['passed'], 'errors': report['errors']}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
