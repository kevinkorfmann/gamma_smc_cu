"""Extend audited full-chromosome Gamma profiles to a 10-Mb display window.

Only reads the independent regional rerun's cache. Writes under the isolated
selection benchmark. Reuses the exact 20 pairs and physical calibration, and
requires agreement with the already archived 1-Mb result before completion.
"""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
from common import digest, write_json


def main():
    root = Path.cwd()
    refs = root / 'references/trem2_empirical_comparison_v1'
    out = root / 'results/stronger_bgs_v1/empirical_context_10Mb_v1'
    out.mkdir(exist_ok=True)
    api = root / 'study/spatial_time_calibration_v1/current_infer.py'
    validation = json.loads((root / 'results/stronger_bgs_v1/spatial_time_calibration_v1/validation.json').read_text())
    assert validation['passed'] and digest(api) == validation['corrected_api_sha256']
    spec = importlib.util.spec_from_file_location('current_calibrated_infer', api)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    import gamma_smc_cu
    from gamma_smc_cu import _core
    records = []
    for pop in ['IBS', 'YRI']:
        marker = out / f'{pop}_complete.json'
        if marker.exists():
            rec = json.loads(marker.read_text())
            assert digest(out / f'{pop}.npz') == rec['output_sha256']
            records.append(rec); continue
        prior_path = refs / f'{pop}_results.npz'
        old = json.loads((refs / f'{pop}_COMPLETE.json').read_text())
        assert old['complete'] and digest(prior_path) == old['outputs']['results.npz']
        prior = np.load(prior_path)
        cache = Path(old['identity']['configuration']['cache_dir']) / 'parsed/chr6'
        expected = old['identity']['inputs']['cache_manifest']['sha256']
        assert digest(cache / 'READY.json') == expected
        ready = json.loads((cache / 'READY.json').read_text())
        for name in ['positions.npy', 'sample_ids.npy']:
            assert digest(cache / name) == ready['members'][name]['sha256']
        pos = np.load(cache / 'positions.npy', mmap_mode='r')
        raw = np.load(cache / 'G.npy', mmap_mode='r')
        assert list(raw.shape) == ready['shape']
        hap = prior['population_hap_indices']
        G = np.ascontiguousarray(raw[hap, :])
        c = old['details']['calibration']
        theta = np.mean([np.count_nonzero(G[i] != G[i+1]) / (pos[-1]+1) for i in range(0, len(G), 2)])
        np.testing.assert_allclose(theta, c['theta'], rtol=1e-12)
        start = time.monotonic()
        with (root / 'runs/stronger_bgs_v1/gamma.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            free, total = _core.cuda_mem_info()
            if free < 8 * 1024**3:
                raise RuntimeError('Fewer than 8 GiB free on GPU; leave other projects alone and retry later')
            result = mod.infer_blockwise(G, pos, pairs=prior['pairs'].tolist(),
                Ne=c['calibrated_Ne'], mu=c['physical_mu'], rho=c['physical_rho'],
                physical_mu=c['physical_mu'], auto_estimate_theta=False,
                flow_field_path=str(Path(gamma_smc_cu.__file__).parent/'default_flow_field.txt'),
                core_block_sites=6000000, flank_sites=0, pair_batch_size=20,
                max_streams=1, mean_only=True)
        p = result['positions']; y = result['mean']
        ix = np.searchsorted(p, prior['gamma_positions'])
        np.testing.assert_array_equal(p[ix], prior['gamma_positions'])
        np.testing.assert_allclose(y[ix], prior['gamma_mean'], rtol=2e-5, atol=.02)
        discrepancy = float(np.max(np.abs(y[ix]/prior['gamma_mean']-1)))
        # Empirical VCF POS is one-based. Export zero-based positions explicitly.
        p0 = p-1
        keep = (p0 >= 36160846) & (p0 < 46160846)
        np.savez_compressed(out/f'{pop}.npz', positions0=p0[keep], mean=y[keep],
            pairs=prior['pairs'], population_hap_indices=hap)
        rec = dict(population=pop, output_sha256=digest(out/f'{pop}.npz'),
            prior_result_sha256=digest(prior_path), prior_complete_sha256=digest(refs/f'{pop}_COMPLETE.json'),
            cache_manifest_sha256=expected, cache_G_hash_from_audited_ready=ready['members']['G.npy']['sha256'],
            cache_G_full_hash_recomputed=False, selected_genotypes_sha256=__import__('hashlib').sha256(G).hexdigest(),
            api_sha256=digest(api), source_sha256=digest(__file__), pairs=20, haplotypes=len(G),
            input_sites=len(pos), inference_context='full chromosome; one block',
            display_interval0=[36160846,46160846], calibration=c, metadata=result['metadata'],
            prior_1Mb_max_relative_difference=discrepancy, seconds=time.monotonic()-start)
        write_json(marker, rec); records.append(rec)
        del G, raw, result, y
        print(json.dumps(rec), flush=True)
    write_json(out/'completion.json', dict(complete=True, records=records,
        caveat='Same historical 20-pair samples; descriptive population profiles, not independent biological replicates.'))


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    main()
