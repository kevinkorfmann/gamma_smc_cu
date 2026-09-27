#!/usr/bin/env python3
"""Collect first validated genome winners from already synchronized host roots.

No networking or job execution occurs here. Existing winners are immutable;
run again to add newly completed chromosome/population tasks.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'genome_wide'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from campaign_identity import scientific_identity
from full_chromosome_protocol import FullProtocol, PROTOCOL as FULL_PROTOCOL

POPS = sorted('ACB ASW ESN GWD LWK MSL YRI CEU FIN GBR IBS TSI CDX CHB CHS JPT KHV BEB GIH ITU PJL STU CLM MXL PEL PUR'.split())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_source_files(root, source_directory):
    source = (root/source_directory).resolve()
    frozen = read_json(source/'SOURCE_MANIFEST.json')
    if not re.fullmatch(r'[0-9a-fA-F]{40}', frozen.get('git_commit', '')) or not frozen.get('files'):
        raise ValueError('Frozen source requires a full Git commit and nonempty file hashes')
    hashes = {}
    for name, entry in frozen['files'].items():
        path = (source/name).resolve()
        digest = entry.get('sha256') if isinstance(entry, dict) else entry
        if not path.is_relative_to(source) or not path.is_file() or sha(path) != digest:
            raise ValueError(f'Frozen source file missing or changed: {name}')
        hashes[name] = digest
    return hashes


def verify_local_gpu_gate(root, validation, source_hash):
    """Check actual retained local evidence; do not claim cross-hardware parity."""
    if validation != (root/'validation/production-approved.json').resolve():
        raise ValueError('Local gate must be this synchronized root\'s validation/production-approved.json')
    gate = read_json(validation)
    if gate.get('passed') is not True or gate.get('source_manifest_sha256') != source_hash:
        raise ValueError('Local GPU approval must have passed=true and match the exact frozen source')
    preflight = root/'validation/preflight-v2/validation.json'
    ready = root/'inputs/cache/parsed/chr22/READY.json'
    exit_file = root/'validation/gpu-tests.exit'
    if sha(preflight) != gate.get('preflight_sha256') or sha(ready) != gate.get('chromosome22_READY_sha256'):
        raise ValueError('Local approval preflight or chromosome22 READY hash mismatch')
    if exit_file.read_text().strip() != '0':
        raise ValueError('Actual local GPU test exit record must equal zero')
    d = read_json(preflight)
    if d.get('passed') is not True or not d.get('hostname') or d['hostname'] != gate.get('hostname'):
        raise ValueError('Local preflight must pass on the approval host')
    settings = d['settings']
    if settings.get('chr') != 22 or settings.get('pop') != 'ASW':
        raise ValueError('Expected the actual chromosome22 ASW GPU preflight')
    core, flank = settings['core_sites'], settings['flank_sites']
    if core <= 0 or flank < 0 or d['retained_sites'] <= core:
        raise ValueError('Local preflight did not exercise multiple core blocks')
    limits = dict(site_p99_abs_log=.01, site_max_abs_log=.1, gene_max_abs_log=.01)
    if d.get('diagnostic_limits') != limits:
        raise ValueError('Unexpected or weakened local diagnostic limits')
    quantiles = d['site_abs_log_error']
    ordered = [quantiles[k] for k in ('median', 'p95', 'p99', 'max')]
    if (not np.isfinite(ordered).all() or ordered != sorted(ordered) or ordered[0] < 0
            or quantiles['p99'] > .01 or quantiles['max'] > .1):
        raise ValueError('Recorded full-chromosome GPU comparison did not pass')
    cal = d['calibration']
    for key in ('full_metadata', 'blockwise_metadata'):
        if d.get(key, {}).get('time_units') != 'generations':
            raise ValueError('Local GPU diagnostic is not calibrated in generations')
    if cal.get('time_units') != 'generations' or cal.get('physical_mu') != 1.25e-8 or cal.get('physical_rho') != 1e-8:
        raise ValueError('Unexpected local preflight time calibration')
    sample_path = root/'validation/preflight-v2/cross_host.npz'
    with np.load(sample_path, allow_pickle=False) as sample:
        positions, pairs = sample['positions'], sample['pairs']
        x, y = sample['full'].astype(float), sample['blockwise'].astype(float)
        if (positions.ndim != 1 or len(positions) != min(2000, d['retained_sites'])
                or not np.isfinite(positions).all() or np.any(positions < 0)
                or np.any(positions != np.floor(positions)) or np.any(np.diff(positions) <= 0)
                or pairs.shape != (20, 2) or pairs.dtype.kind not in 'iu'
                or not np.array_equal(pairs, d['pairs']) or np.any(pairs < 0)
                or np.any(pairs >= 2*d['samples']) or np.any(pairs[:, 0] == pairs[:, 1])
                or len({tuple(sorted(p)) for p in pairs.tolist()}) != 20
                or x.shape != (len(positions), 20) or y.shape != x.shape
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.any(x <= 0) or np.any(y <= 0)):
            raise ValueError('Invalid retained local GPU comparison arrays')
        error = np.abs(np.log(x)-np.log(y))
        sample_p99, sample_max = map(float, np.quantile(error, [.99, 1]))
        # A deterministic subset can overrepresent rare errors, so its p99 is
        # descriptive rather than a second test of the dense-array percentile.
        if sample_max > .1 or sample_max > quantiles['max']+1e-8:
            raise ValueError('Retained local GPU arrays contradict the passed diagnostic')
    gene_path = root/'validation/preflight-v2/gene_comparison.csv'
    with gene_path.open() as stream:
        genes = list(csv.DictReader(stream))
    if not genes:
        raise ValueError('Missing local GPU gene comparisons')
    gene_errors = []
    for row in genes:
        full, block, reported = map(float, (row['full'], row['blockwise'], row['abs_log_error']))
        if not np.isfinite([full, block, reported]).all() or min(full, block) <= 0 or int(row['n_sites']) < 2:
            raise ValueError('Invalid local GPU gene comparison')
        actual = abs(math.log(block/full))
        if not math.isclose(actual, reported, rel_tol=1e-7, abs_tol=1e-12):
            raise ValueError('Local GPU gene error does not match retained means')
        gene_errors.append(actual)
    gene_max = max(gene_errors)
    if gene_max > .01 or not math.isclose(gene_max, d['gene_max_abs_log_error'], rel_tol=1e-7, abs_tol=1e-12):
        raise ValueError('Local GPU gene comparisons did not pass')
    evidence = {str(path.relative_to(root)): sha(path) for path in
                [validation, preflight, ready, exit_file, sample_path, gene_path]}
    return dict(artifact_sha256=evidence, tested_hostname=d['hostname'], gpu=d.get('gpu'),
                core_block_sites=core, flank_sites=flank, physical_mu=cal['physical_mu'],
                physical_rho=cal['physical_rho'], ready_identity=read_json(ready),
                retained_sample_p99_abs_log=sample_p99, retained_sample_max_abs_log=sample_max,
                recomputed_gene_max_abs_log=gene_max,
                scope='Local GPU tests exit=0 and hash-bound full/blockwise preflight; retained sample and gene means independently rechecked. No cross-host or cross-hardware parity claim; full dense site arrays were not retained.')


def local_artifact(root, done, name):
    """Map recorded remote task paths into an explicitly synchronized root."""
    recorded = Path(done['output'])
    suffix = Path('outputs/tasks') / done['task_id'] / 'attempt-001'
    if list(recorded.parts[-len(suffix.parts):]) != list(suffix.parts):
        raise ValueError('Unexpected worker attempt output layout')
    remote_root = recorded.parents[3]
    values = dict(root=str(remote_root), source=str(remote_root / 'source'),
                  inputs=str(remote_root / 'inputs'), tools=str(remote_root / 'inputs/tools'),
                  output=str(recorded), python='unused')
    path = Path(name.format(**values))
    if not path.is_absolute():
        path = recorded / path
    relative = path.relative_to(remote_root)
    result = (root / relative).resolve()
    if root not in result.parents:
        raise ValueError('Artifact is outside its synchronized host root')
    return result


def inspect_candidate(host, root, chrom, pop, source_hash, protocol=None):
    task = protocol.tasks[chrom, pop] if protocol is not None else None
    task_id = task['id'] if task is not None else f'genome_chr{chrom}_{pop}'
    state = root / 'state'
    marker = state / (task_id + '.done.json')
    if (state / (task_id + '.failed.json')).exists():
        return None, 'failed task marker'
    if not marker.is_file():
        return None, 'not completed'
    done = read_json(marker)
    if done.get('source_manifest_sha256') != source_hash:
        raise ValueError(f'Completed task uses a different frozen source: {host}/{task_id}')
    if done.get('task_id') != task_id or done.get('returncode') != 0 or done.get('error'):
        return None, 'invalid successful-completion record'
    if protocol is not None:
        protocol.check_receipt(done, task)
    epoch = done.get('completed_epoch')
    if not isinstance(epoch, (int, float)) or not math.isfinite(epoch) or epoch <= 0:
        return None, 'missing/invalid completion epoch'
    artifacts = {}
    try:
        for name, detail in done['outputs'].items():
            path = local_artifact(root, done, name)
            if (not path.is_file() or path.stat().st_size != detail['bytes']
                    or sha(path) != detail['sha256']):
                return None, f'missing or mismatched artifact: {name}'
            artifacts[name] = dict(local_path=str(path), sha256=detail['sha256'], bytes=detail['bytes'])
        metas = [Path(entry['local_path']) for entry in artifacts.values()
                 if Path(entry['local_path']).name == pop + '.metadata.json']
        if len(metas) != 1:
            return None, 'exactly one gene metadata artifact is required'
        metadata_path = metas[0]
        meta = read_json(metadata_path)
        if meta['identity'].get('chromosome') != chrom or meta['identity'].get('population') != pop:
            return None, 'metadata chromosome/population mismatch'
        common, chrom_inputs = scientific_identity(meta)
        files = {'.metadata.json': metadata_path}
        for ext in ('.csv', '.npz'):
            path = metadata_path.with_name(pop + ext)
            digest = meta['output_sha256'][ext]
            if not path.is_file() or sha(path) != digest:
                return None, f'metadata output hash mismatch: {path.name}'
            if not any(entry['local_path'] == str(path) and entry['sha256'] == digest for entry in artifacts.values()):
                return None, f'output absent from worker validation: {path.name}'
            files[ext] = path
    except (KeyError, OSError, json.JSONDecodeError, ValueError) as exc:
        return None, f'invalid artifact metadata: {exc}'
    if protocol is not None:
        protocol.check_metadata(meta, done, task)
    return dict(host=host, chromosome=chrom, population=pop, task_id=task_id,
                completed_epoch=epoch, source_manifest_sha256=source_hash,
                **({'task_manifest_sha256': protocol.manifest_sha256} if protocol is not None else {}),
                done_sha256=sha(marker), done_path=str(marker),
                common_identity=common, chromosome_identity=chrom_inputs,
                files={ext: str(path) for ext, path in files.items()}, artifacts=artifacts), None


def verify_registry(output, registry):
    registered = {name for winner in registry['winners'].values() for name in winner['selected_artifacts']}
    actual = {str(path.relative_to(output)) for path in (output/'genome').rglob('*') if path.is_file()}
    if actual != registered:
        raise ValueError('Unregistered/stale partial files in selected genome results')
    for task_id, winner in registry['winners'].items():
        for relative, detail in winner['selected_artifacts'].items():
            path = output / relative
            if not path.is_file() or path.stat().st_size != detail['bytes'] or sha(path) != detail['sha256']:
                raise ValueError(f'Existing selected winner changed or is incomplete: {task_id}: {path}')


def collect(hosts, output, validation=None, source_directory="source", *, local_validation=None,
            full_task_manifest=None):
    hosts = {name: Path(path).resolve() for name, path in hosts.items()}
    output = Path(output).resolve()
    if (validation is None) == (local_validation is None):
        raise ValueError('Choose exactly one cross-host or local GPU validation gate')
    if full_task_manifest is not None and local_validation is not None:
        raise ValueError('Full-chromosome collection requires its explicit cross-host gate')
    mode = 'single_host_local_gpu' if local_validation is not None else 'cross_host'
    if full_task_manifest is not None:
        mode = 'full_chromosome_cross_host'
    validation = Path(local_validation if local_validation is not None else validation).resolve()
    source_directory = Path(source_directory)
    if source_directory.is_absolute() or '..' in source_directory.parts:
        raise ValueError('Source directory must be a safe relative path')
    if not hosts:
        raise ValueError('At least one synchronized host root is required')
    if mode == 'single_host_local_gpu' and (len(hosts) != 1 or str(source_directory) != 'source-v2'):
        raise ValueError('Local GPU mode requires exactly one host and --source source-v2; no derived/runtime-v3 approvals')
    gate = read_json(validation)
    if gate.get('passed') is not True:
        raise ValueError('Numerical validation must explicitly have passed=true')
    if gate.get('protocol') == FULL_PROTOCOL and full_task_manifest is None:
        raise ValueError('Full-chromosome gate requires its explicit full task manifest mapping')
    gate_sha = sha(validation)
    source_hashes = {name: sha(root / source_directory / 'SOURCE_MANIFEST.json') for name, root in hosts.items()}
    if len(set(source_hashes.values())) != 1:
        raise ValueError('Host source manifests differ; refusing to combine campaigns')
    source_hash = next(iter(source_hashes.values()))
    if gate.get('source_manifest_sha256') != source_hash:
        raise ValueError('Validation is missing its source binding or refers to a different frozen source')
    source_files = {name: verify_source_files(root, source_directory) for name, root in hosts.items()}
    protocol = (FullProtocol(full_task_manifest, hosts, gate, source_hash, source_directory)
                if full_task_manifest is not None else None)
    task_manifest_hash = protocol.manifest_sha256 if protocol is not None else None
    evidence = (verify_local_gpu_gate(next(iter(hosts.values())), validation, source_hash)
                if mode == 'single_host_local_gpu' else None)
    if any(output == root or root in output.parents or output in root.parents for root in hosts.values()):
        raise ValueError('Selected output must be separate from synchronized host roots')
    output.mkdir(parents=True, exist_ok=True)
    with open(output / '.collector.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry_path = output / 'winner_registry.json'
        if registry_path.exists():
            registry = read_json(registry_path)
            if (registry['source_manifest_sha256'] != source_hash
                    or registry.get('validation_mode', 'cross_host') != mode
                    or registry.get('validation_sha256', registry.get('cross_host_validation_sha256')) != gate_sha
                    or registry.get('validation_evidence') != evidence
                    or registry.get('task_manifest_sha256') != task_manifest_hash
                    or (mode == 'single_host_local_gpu' and registry.get('host_names') != sorted(hosts))
                    or registry.get('source_directory', 'source') != str(source_directory)):
                raise ValueError('Existing selection uses another source campaign or validation gate')
            verify_registry(output, registry)
        else:
            if any(p.name != '.collector.lock' for p in output.iterdir()):
                raise ValueError('Unregistered/stale partial collection directory; inspect it before proceeding')
            registry = dict(schema=2, source_manifest_sha256=source_hash, source_directory=str(source_directory),
                            validation_mode=mode, validation_sha256=gate_sha, validation=gate,
                            validation_evidence=evidence, host_names=sorted(hosts),
                            common_identity=None, chromosome_identities={}, winners={}, complete=False)
            if protocol is not None:
                registry.update(task_manifest_sha256=task_manifest_hash,
                                protocol=protocol.manifest['protocol'],
                                scientific_settings=protocol.manifest['scientific_settings'])
            if mode in ('cross_host', 'full_chromosome_cross_host'):
                registry.update(cross_host_validation_sha256=gate_sha, cross_host_validation=gate)
        choices, rows = [], []
        for chrom in range(1, 23):
            for pop in POPS:
                task_id = protocol.tasks[chrom, pop]['id'] if protocol is not None else f'genome_chr{chrom}_{pop}'
                valid, errors = [], {}
                for host, root in sorted(hosts.items()):
                    candidate, error = inspect_candidate(host, root, chrom, pop, source_hash, protocol)
                    if candidate:
                        valid.append(candidate)
                    else:
                        errors[host] = error
                for candidate in valid:
                    common = candidate['common_identity']; chrom_input = candidate['chromosome_identity']
                    if any(source_files[candidate['host']].get(name) != digest for name, digest in common['sources'].items()):
                        raise ValueError(f'Metadata inference source differs from frozen source: {candidate["host"]}/{task_id}')
                    if evidence is not None:
                        config = common['configuration']
                        expected = dict(mu=evidence['physical_mu'], rho=evidence['physical_rho'],
                                        core_block_sites=evidence['core_block_sites'], flank_sites=evidence['flank_sites'])
                        if any(config.get(key) != value for key, value in expected.items()):
                            raise ValueError(f'Scientific settings differ from actual local GPU gate: {task_id}')
                        ready = evidence['ready_identity']
                        if chrom == 22 and (chrom_input['source_vcf_sha256'] != ready['source_sha256']
                                or chrom_input['cache_members'] != {k:v['sha256'] for k,v in ready['members'].items()}
                                or chrom_input['cache_shape'] != ready['shape']):
                            raise ValueError(f'Chromosome22 input differs from actual local GPU gate: {task_id}')
                    if registry['common_identity'] is None:
                        registry['common_identity'] = common
                    elif registry['common_identity'] != common:
                        raise ValueError(f'Scientific settings/source/schema mismatch: {candidate["host"]}/{task_id}')
                    old = registry['chromosome_identities'].get(str(chrom))
                    if old is not None and old != chrom_input:
                        raise ValueError(f'Input VCF/cache/annotation mismatch for chromosome {chrom}: {candidate["host"]}/{task_id}')
                    registry['chromosome_identities'][str(chrom)] = chrom_input
                current = registry['winners'].get(task_id)
                if current:
                    status, winner_host, epoch = 'selected', current['host'], current['completed_epoch']
                elif valid:
                    winner = min(valid, key=lambda item: (item['completed_epoch'], item['host']))
                    choices.append(winner)
                    status, winner_host, epoch = 'selected', winner['host'], winner['completed_epoch']
                else:
                    status, winner_host, epoch = ('invalid' if any(e != 'not completed' for e in errors.values()) else 'missing'), '', ''
                rows.append(dict(chr=chrom, pop=pop, task_id=task_id, status=status, winner_host=winner_host,
                                 completed_epoch=epoch, validated_candidates=len(valid), errors=json.dumps(errors, sort_keys=True)))
        # Validate the entire campaign before copying any newly selected artifact.
        for winner in choices:
            task_id = winner['task_id']; chrom, pop = winner['chromosome'], winner['population']
            destination = output / 'genome' / f'chr{chrom}'
            destination.mkdir(parents=True, exist_ok=True)
            selected = {}
            for ext in winner['files']:
                if (destination / (pop + ext)).exists():
                    raise ValueError(f'Unregistered/stale partial winner: {task_id}')
            with tempfile.TemporaryDirectory(prefix='.collect-', dir=output) as temporary:
                for ext in ('.npz', '.csv', '.metadata.json'):
                    original = winner['files'][ext]
                    tmp = Path(temporary) / (pop + ext)
                    shutil.copyfile(original, tmp)
                    verified = next(item['sha256'] for item in winner['artifacts'].values() if item['local_path'] == original)
                    if sha(tmp) != verified:
                        raise ValueError(f'Artifact changed while copying: {original}')
                    final = destination / tmp.name
                    os.link(tmp, final)  # Atomic creation; never overwrite an existing winner.
                    tmp.unlink()
                    selected[str(final.relative_to(output))] = dict(sha256=sha(final), bytes=final.stat().st_size)
            winner['selected_artifacts'] = selected
            registry['winners'][task_id] = winner
            registry['complete'] = len(registry['winners']) == 572
            atomic_json(registry_path, registry)
        # Also retain the validated campaign when no new tasks are available.
        registry['complete'] = len(registry['winners']) == 572
        atomic_json(registry_path, registry)
        report = output / 'inspection.tsv'
        temp = report.with_suffix('.tsv.tmp')
        with temp.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
            writer.writeheader(); writer.writerows(rows)
        os.replace(temp, report)
        summary = dict(selected=len(registry['winners']), expected=572, complete=registry['complete'],
                       newly_selected=len(choices), postprocessing_ready=registry['complete'],
                       validation_mode=mode,
                       cross_host_parity_validated=(mode in ('cross_host', 'full_chromosome_cross_host')))
        atomic_json(output / 'collection_status.json', summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-root', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--source', default='source', help='Frozen source directory within each host root; use source-v2 for production')
    parser.add_argument('--output', required=True, help='Separate selected-results directory; can be incrementally revisited')
    parser.add_argument('--full-task-manifest', help='Explicit 572-task full_chromosome_v1 manifest; requires its cross-host gate and source-v2')
    gates = parser.add_mutually_exclusive_group(required=True)
    gates.add_argument('--cross-host-validation')
    gates.add_argument('--local-validation', help='One host only: its actual source-v2 validation/production-approved.json')
    args = parser.parse_args()
    hosts = {}
    for entry in args.host_root:
        name, path = entry.split('=', 1)
        if not re.fullmatch('[A-Za-z0-9_-]+', name) or name in hosts:
            raise ValueError('Host names must be unique simple identifiers')
        hosts[name] = path
    print(json.dumps(collect(hosts, args.output, args.cross_host_validation, args.source,
                             local_validation=args.local_validation,
                             full_task_manifest=args.full_task_manifest), sort_keys=True))


if __name__ == '__main__':
    main()
