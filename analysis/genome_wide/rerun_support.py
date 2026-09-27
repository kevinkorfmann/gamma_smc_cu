"""Portable, explicit configuration for reproducible scientific reruns."""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys
import numpy as np


def add_arguments(parser):
    for option, env, description in [
        ('cache-dir', 'GAMMA_CACHE_DIR', 'Read-only cache with parsed/ and genes/'),
        ('samples', 'GAMMA_SAMPLES', 'Sample membership file'),
        ('output-dir', 'GAMMA_OUTPUT_DIR', 'Separate new results directory'),
    ]:
        parser.add_argument('--'+option, default=os.environ.get(env),
                            required=not os.environ.get(env), help=description+'; env '+env)
    parser.add_argument('--mu', type=float, default=1.25e-8, help='Physical mutation rate per bp/generation')
    parser.add_argument('--rho', type=float, default=1e-8, help='Physical recombination rate per bp/generation')
    parser.add_argument('--core-block-sites', type=int, default=4096)
    parser.add_argument('--flank-sites', type=int, default=2048)
    parser.add_argument('--pair-chunk', type=int, default=250)
    parser.add_argument('--sequence-length', type=float, help='Heterozygosity denominator in bp; default last chromosome position + 1')
    parser.add_argument('--resume', action='store_true', help='Skip only completed outputs with identical input/configuration fingerprints')


def check_arguments(args):
    if args.mu <= 0 or args.rho < 0 or args.core_block_sites <= 0 or args.flank_sites < 0 or args.pair_chunk <= 0:
        raise ValueError('Invalid physical rates or block/chunk settings')
    if args.sequence_length is not None and args.sequence_length <= 0:
        raise ValueError('Sequence length must be positive')
    cache = Path(args.cache_dir).resolve(); output = Path(args.output_dir).resolve()
    if output == cache or cache in output.parents or output in cache.parents:
        raise ValueError('Output and input cache must be separate, non-nested directories')
    if not cache.is_dir() or not Path(args.samples).is_file():
        raise FileNotFoundError('Cache directory and sample file must exist')


def fingerprint(path):
    path = Path(path).resolve(); h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return {'path':str(path), 'size':path.stat().st_size, 'sha256':h.hexdigest()}


def calibrate(G, positions, mu, rho, sequence_length=None):
    """Estimate chromosome-wide 4Ne*mu from paired diploid heterozygosity.

    Calibration uses the full population chromosome, before local window slices.
    Passing Ne_hat and physical mu/rho to auto_estimate_theta=False gives both
    the intended scaled kernel rates and physical-generation output units.
    """
    if G.ndim != 2 or G.shape[0] < 2 or G.shape[0] % 2 or G.shape[1] != len(positions) or len(positions)==0:
        raise ValueError('Calibration requires nonempty paired diploid haplotypes')
    length=float(sequence_length if sequence_length is not None else positions[-1]+1)
    if length <= positions[-1] or length <= 0:
        raise ValueError('Sequence length must extend beyond the final site')
    theta=float((G[0::2] != G[1::2]).sum(axis=1,dtype=np.int64).mean()/length)
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError('Nonpositive/nonfinite chromosome heterozygosity: cannot calibrate physical time')
    ne=theta/(4*mu)
    return {'theta':theta, 'calibrated_Ne':ne, 'physical_mu':float(mu), 'physical_rho':float(rho),
            'sequence_length_bp':length,'denominator_convention':'explicit length' if sequence_length is not None else 'last position + 1',
            'generations_per_coalescent_unit':2*ne, 'scaled_recombination_rate':theta*rho/mu,
            'calibration':'mean within-diploid heterozygosity over the full population chromosome',
            'time_units':'generations'}


def validate_posterior_means(mean, n_sites=None, n_pairs=None):
    """Reject decoding failures without clipping real positive young times.

    Validate in bounded row batches, so the boolean temporary does not scale
    with the entire chromosome-by-pair output matrix.
    """
    mean = np.asarray(mean)
    if mean.ndim != 2 or (n_sites is not None and len(mean) != n_sites) or (n_pairs is not None and mean.shape[1] != n_pairs):
        raise ValueError('Posterior means have an unexpected site/pair shape')
    step = max(1, 1_000_000 // max(1, mean.shape[1]))
    for start in range(0, len(mean), step):
        block = mean[start:start+step]
        if not np.all(np.isfinite(block)) or np.any(block <= 0):
            raise ValueError('Posterior means must be finite and strictly positive; no floor is applied')


def per_pair_moments(mean, site_indices):
    """Float64 site reductions, with a temporary only for one gene/window."""
    values = np.array(mean[site_indices, :], dtype=np.float64, copy=True)
    if not len(values):
        raise ValueError('At least one selected site is required for moments')
    validate_posterior_means(values)
    linear = values.mean(axis=0, dtype=np.float64)
    np.log(values, out=values)
    logarithmic = values.mean(axis=0, dtype=np.float64)
    return linear, logarithmic


def pair_distribution_counts(per_pair_lin, per_pair_log, edges):
    """Finite histogram with explicit tails and unbinned <1000 counters.

    Histogram values summarize each pair's geometric mean across sites. Exact
    here means counting decoded pair summaries, not estimating from bin centers.
    Threshold comparisons are strict (<), in physical generations.
    """
    histogram, _ = np.histogram(per_pair_log, bins=edges)
    return {
        'histogram': histogram,
        'histogram_underflow': int(np.count_nonzero(per_pair_log < edges[0])),
        'histogram_overflow': int(np.count_nonzero(per_pair_log > edges[-1])),
        'n_geom_lt_1000': int(np.count_nonzero(per_pair_log < np.log(1000.0))),
        'n_arith_lt_1000': int(np.count_nonzero(per_pair_lin < 1000.0)),
    }


def run_identity(args, inputs, extra=None):
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip()
    except (OSError,subprocess.CalledProcessError): revision=None
    config={k:v for k,v in vars(args).items() if k not in ('resume','populations')}
    sources={str(p.relative_to(Path(__file__).resolve().parents[2])):fingerprint(p)['sha256'] for p in [Path(__file__),Path(sys.argv[0]).resolve()]}
    return {'configuration':config, 'inputs':inputs, 'git_revision':revision, 'source_sha256':sources, **(extra or {})}


def completed_or_reserve(out_dir, pop, identity, resume=False):
    paths=[Path(out_dir)/(pop+ext) for ext in ('.csv','.npz','.metadata.json')]
    if any(p.exists() for p in paths):
        if resume and all(p.exists() for p in paths):
            metadata=json.loads(paths[-1].read_text())
            if metadata.get('identity')==identity and metadata.get('complete'):
                # Validate retained outputs against the completion marker.
                if all(fingerprint(p)['sha256']==metadata['output_sha256'][p.suffix] for p in paths[:2]):return True
        raise FileExistsError(f'Refusing to replace incomplete, mismatched or existing outputs for {pop} in {out_dir}')
    return False


def finish(out_dir, pop, identity, calibration, decoder_metadata):
    path=Path(out_dir)/(pop+'.metadata.json')
    metadata={'complete':True,'identity':identity,'calibration':calibration,'decoder_metadata':decoder_metadata,
              'python':sys.version,'platform':platform.platform(),
              'output_sha256':{ext:fingerprint(Path(out_dir)/(pop+ext))['sha256'] for ext in ('.csv','.npz')}}
    temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(metadata,indent=2)+'\n');temporary.replace(path)


def load_input(cache_dir, chromosome):
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    from analysis.rerun.cache import load_chromosome
    root=Path(cache_dir)
    if (root/'parsed').is_dir():root=root/'parsed'
    ready=root/f'chr{chromosome}'/'READY.json'
    if ready.exists():
        record=json.loads(ready.read_text())
        for name, member in record['members'].items():
            member_path=ready.parent/name
            if not member_path.is_file() or member_path.stat().st_size != member['bytes']:
                raise ValueError(f'Cache member size mismatch: {member_path}')
        source={'format':'extracted_npy', 'ready_manifest':fingerprint(ready), 'record':record}
    else:source={'format':'npz', **fingerprint(root/f'chr{chromosome}.npz')}
    return load_chromosome(cache_dir,chromosome),source
