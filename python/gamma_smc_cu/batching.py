"""Bounded, pipelined inference with one persistent context per GPU."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import operator
import numpy as np

from gamma_smc_cu import _core
from gamma_smc_cu.infer import (
    _calibrate_result, _coerce_inputs, _empty_result, _filter_segregating,
    _inference_calibration, _normalize_pairs, _resolve_flow_field_path,
    _validate_rates,
)


class RegionMomentContext:
    """Full-context inference returning each pair's region moments.

    The GPU computes arithmetic means and mean log times in float64, after
    applying the same float32 generation calibration as dense inference.
    Keeping the pair dimension preserves downstream histograms, quantiles,
    minima and threshold counts of these region summaries. It does not retain
    individual site posteriors. Region bounds passed to ``run`` are half-open
    indices into ``positions``, the retained, segregating positions attribute.
    """

    def __init__(self, G_or_ts, positions=None, *, mu=1.25e-8, rho=1e-8,
                 Ne=10000, auto_estimate_theta=True, physical_mu=None,
                 flow_field_path=None, gpu_id=None):
        _validate_rates(Ne, mu, rho)
        G, positions = _coerce_inputs(G_or_ts, positions)
        G, self.positions, _ = _filter_segregating(G, positions)
        if not len(self.positions):
            raise ValueError("Region moments require at least one segregating site.")
        kernel_mu, kernel_rho, self.metadata = _inference_calibration(
            G, self.positions, mu, rho, Ne, auto_estimate_theta, physical_mu)
        self.n_haplotypes = G.shape[0]
        previous = _core.get_device()
        device = previous if gpu_id is None else operator.index(gpu_id)
        if not 0 <= device < _core.get_device_count():
            raise ValueError("gpu_id must identify a visible CUDA device.")
        try:
            _core.set_device(device)
            self._context = _core.FlowContext(
                G, self.positions, float(Ne), kernel_mu, kernel_rho,
                _resolve_flow_field_path(flow_field_path), 0)
        finally:
            _core.set_device(previous)

    def run(self, pairs, regions, *, tile_sites=4096):
        pairs = _normalize_pairs(pairs, self.n_haplotypes)
        regions = [(operator.index(a), operator.index(b)) for a, b in regions]
        tile_sites = operator.index(tile_sites)
        if tile_sites < 1:
            raise ValueError('tile_sites must be positive.')
        if regions:
            result = self._context.run_fb_region_moments(
                pairs, regions, tile_sites=tile_sites,
                calibration_factor=self.metadata['generation_rescaling_factor'])
        else:
            result = {key: np.empty((0, len(pairs)), dtype=np.float64)
                      for key in ('region_mean', 'region_mean_log')}
        result.update(pairs=pairs, positions=self.positions, regions=regions,
                      metadata=self.metadata.copy())
        return result


def iter_infer_batches(
    G_or_ts, positions=None, *, pairs, pair_batch_size=256, gpu_ids=None,
    mu=1.25e-8, rho=1e-8, Ne=10000, flow_field_path=None,
    mean_only=True, return_posterior=False, auto_estimate_theta=True,
    physical_mu=None, checkpoint_sites=None,
):
    """Yield full-context posterior batches in the supplied pair order.

    Validate and filter the panel, calibrate time, and upload the flow context
    once per device. A persistent worker per GPU decodes the next batch while
    the caller consumes or writes the preceding result. Each result has the
    usual ``mean``, ``positions``, ``pairs`` and ``metadata`` fields, plus
    ``pair_offset``. All sequence sites remain in one forward/backward block;
    only pairs are partitioned. Optional CI and posterior parameters retain
    the same semantics as ``infer_blockwise``.

    ``gpu_ids=None`` uses the current CUDA device. Multiple devices must be
    explicitly selected, e.g. ``gpu_ids=[0, 1]``. IDs refer to visible devices.
    At most one future per device is queued; discard batches after consuming
    them to keep host memory bounded. The caller owns yielded arrays and may
    retain them if desired. When stopping early, close the iterator (or use
    ``contextlib.closing``) to wait for outstanding work and release contexts.

    Allocation and validation happen on iteration. No results are yielded for
    an empty pair list. Monomorphic panels yield correctly shaped empty arrays.
    Batch size affects throughput and memory, but does not change inference.
    ``checkpoint_sites`` enables exact checkpoint/replay to bound GPU forward
    state memory by tile size. This adds a forward pass; measure whether the
    larger feasible pair batches compensate on your GPU and workload.
    """
    pair_batch_size = operator.index(pair_batch_size)
    if pair_batch_size < 1:
        raise ValueError("pair_batch_size must be positive.")
    if checkpoint_sites is not None:
        checkpoint_sites = operator.index(checkpoint_sites)
        if checkpoint_sites < 1:
            raise ValueError("checkpoint_sites must be positive.")
    _validate_rates(Ne, mu, rho)
    G, positions = _coerce_inputs(G_or_ts, positions)
    G, positions, _ = _filter_segregating(G, positions)
    kernel_mu, kernel_rho, metadata = _inference_calibration(
        G, positions, mu, rho, Ne, auto_estimate_theta, physical_mu)
    pairs = _normalize_pairs(pairs, G.shape[0])
    if not pairs:
        return
    if not len(positions):
        for offset in range(0, len(pairs), pair_batch_size):
            batch = pairs[offset:offset + pair_batch_size]
            result = _calibrate_result(
                _empty_result(positions, batch, mean_only, return_posterior,
                              blockwise=True), metadata.copy())
            result["pair_offset"] = offset
            yield result
        return

    devices = [_core.get_device()] if gpu_ids is None else [operator.index(x) for x in gpu_ids]
    if (not devices or len(set(devices)) != len(devices)
            or any(x < 0 or x >= _core.get_device_count() for x in devices)):
        raise ValueError("gpu_ids must contain distinct visible CUDA device IDs.")
    devices = devices[:(len(pairs) + pair_batch_size - 1) // pair_batch_size]
    contexts = []
    previous_device = _core.get_device()
    try:
        for device in devices:
            _core.set_device(device)
            contexts.append(_core.FlowContext(
                G, positions, float(Ne), kernel_mu, kernel_rho,
                _resolve_flow_field_path(flow_field_path), 0))
    finally:
        _core.set_device(previous_device)

    def decode(index, offset):
        batch = pairs[offset:offset + pair_batch_size]
        if checkpoint_sites is None:
            result = contexts[index].run_fb_blockwise(
                batch, core_block_sites=len(positions), flank_sites=0,
                pair_batch_size=pair_batch_size, max_streams=1,
                mean_only=mean_only, return_posterior=return_posterior)
        else:
            result = contexts[index].run_fb_checkpointed(
                batch, tile_sites=checkpoint_sites, mean_only=mean_only,
                return_posterior=return_posterior,
                calibration_factor=metadata['generation_rescaling_factor'])
        result.update(positions=positions, pairs=batch, pair_offset=offset)
        if checkpoint_sites is not None:
            result['metadata'] = metadata.copy()
            return result
        return _calibrate_result(result, metadata.copy())

    offsets = iter(range(0, len(pairs), pair_batch_size))
    pending = deque()
    try:
        with ThreadPoolExecutor(max_workers=len(contexts)) as pool:
            for index in range(len(contexts)):
                pending.append((index, pool.submit(decode, index, next(offsets))))
            while pending:
                index, future = pending.popleft()
                result = future.result()
                offset = next(offsets, None)
                if offset is not None:
                    pending.append((index, pool.submit(decode, index, offset)))
                yield result
    finally:
        pending.clear()
        contexts.clear()
