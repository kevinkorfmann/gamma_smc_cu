"""Bounded direct export of exact dense posterior means to standard NPY files."""

from concurrent.futures import ThreadPoolExecutor
import errno
import json
import operator
import os
from pathlib import Path
import sys
import time

import numpy as np
from gamma_smc_cu import _core
from gamma_smc_cu.infer import (
    _coerce_inputs, _filter_segregating, _inference_calibration,
    _normalize_pairs, _resolve_flow_field_path, _validate_rates,
)


def _gpu_local_cpus(device):
    """Intersect PCI-local CPUs with the calling thread's existing allocation."""
    if not hasattr(os, 'sched_getaffinity'):
        return None
    try:
        bus = _core.device_pci_bus_id(device).lower()
        ranges = (Path('/sys/bus/pci/devices') / bus / 'local_cpulist').read_text().strip()
        cpus = set()
        for item in ranges.split(','):
            bounds = item.split('-')
            cpus.update(range(int(bounds[0]), int(bounds[-1]) + 1))
        return sorted(cpus & os.sched_getaffinity(0)) or None
    except (OSError, ValueError, AttributeError, RuntimeError):
        return None


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _preallocate_file(fd, size):
    """Reserve space where supported without glibc's slow zero-write emulation."""
    try:
        if sys.platform.startswith('linux'):
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            allocate = libc.fallocate
            allocate.argtypes = [ctypes.c_int, ctypes.c_int,
                                 ctypes.c_longlong, ctypes.c_longlong]
            allocate.restype = ctypes.c_int
            if allocate(fd, 0, 0, size) != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
        elif hasattr(os, 'posix_fallocate'):
            os.posix_fallocate(fd, 0, size)
    except OSError as e:
        if e.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            raise


def export_dense(
    G_or_ts, positions=None, *, pairs, output_dir, pair_batch_size=8192,
    gpu_ids=None, tile_sites=4096, staging_buffers=3, mu=1.25e-8, rho=1e-8,
    Ne=10000, physical_mu=None, auto_estimate_theta=True, flow_field_path=None,
    numa_local=True, preallocate=True, durable=False, workers_per_gpu=2,
):
    """Export every calibrated float32 posterior mean with full chromosome context.

    The output is a new directory containing ``positions.npy``, ``pairs.npy``,
    and ``mean_batchNNNNN.npy`` arrays of shape (retained sites, batch pairs).
    Pair order is unchanged. ``manifest.json`` is published only after all
    files finish successfully. Refuse an existing directory; a failed run may
    leave valid batches, but has no complete manifest. It is not resumable.

    Exact checkpoint/replay overlaps GPU computation, asynchronous transfers
    into a small pinned ring, and direct writes to the NPY payload. Host staging
    scales with ``tile_sites * pair_batch_size * staging_buffers * 4`` bytes
    per worker, not chromosome length. Multiple workers per GPU overlap writes
    to separate batch files. GPU free memory is checked before decoding.
    Confidence intervals and posterior parameters are not exported by this API.

    Devices are explicit and relative to CUDA_VISIBLE_DEVICES; None uses the
    current GPU. When available, NUMA placement confines each export worker and
    its helper threads to GPU-local CPUs within its existing CPU allocation.
    This does not change the calling thread's affinity. Disable with numa_local=False.
    Preallocation falls back to sparse sizing on unsupported filesystems; disk
    space and I/O errors still propagate. With durable=True, file and directory
    fsync calls finish before return; the default matches ordinary NPY writes.
    Returned stage timers overlap and must not be summed into wall time.
    """
    start = time.perf_counter()
    if os.name != 'posix':
        raise OSError('Direct dense export requires POSIX file descriptors.')
    pair_batch_size = operator.index(pair_batch_size)
    tile_sites = operator.index(tile_sites)
    staging_buffers = operator.index(staging_buffers)
    workers_per_gpu = operator.index(workers_per_gpu)
    if pair_batch_size < 1 or tile_sites < 1 or not 2 <= staging_buffers <= 8:
        raise ValueError('Positive batch/tile sizes and 2..8 staging buffers required.')
    if workers_per_gpu < 1:
        raise ValueError('workers_per_gpu must be positive.')
    _validate_rates(Ne, mu, rho)
    G, positions = _coerce_inputs(G_or_ts, positions)
    G, positions, _ = _filter_segregating(G, positions)
    kernel_mu, kernel_rho, calibration = _inference_calibration(
        G, positions, mu, rho, Ne, auto_estimate_theta, physical_mu)
    pairs = _normalize_pairs(pairs, G.shape[0])
    devices = [_core.get_device()] if gpu_ids is None else [operator.index(x) for x in gpu_ids]
    if (not devices or len(set(devices)) != len(devices)
            or any(x < 0 or x >= _core.get_device_count() for x in devices)):
        raise ValueError('gpu_ids must contain distinct visible CUDA device IDs.')
    devices = devices[:max(1, (len(pairs) + pair_batch_size - 1) // pair_batch_size)]
    worker_devices = (devices * workers_per_gpu)[:max(1, (len(pairs) + pair_batch_size - 1) // pair_batch_size)]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for name, values in [('positions.npy', positions),
                         ('pairs.npy', np.asarray(pairs, dtype=np.int32).reshape(-1, 2))]:
        with (output / name).open('xb') as f:
            np.save(f, values, allow_pickle=False)
            if durable:
                f.flush()
                os.fsync(f.fileno())

    flow_path = _resolve_flow_field_path(flow_field_path)
    contexts = []
    previous = _core.get_device()
    try:
        if len(pairs) and len(positions):
            for device in worker_devices:
                _core.set_device(device)
                contexts.append(_core.FlowContext(G, positions, float(Ne), kernel_mu,
                                                  kernel_rho, flow_path, 0))
    finally:
        _core.set_device(previous)
    preparation = time.perf_counter() - start

    def worker(index, device):
        original_cpus = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
        cpus = _gpu_local_cpus(device) if numa_local else None
        records = []
        try:
            if cpus:
                os.sched_setaffinity(0, cpus)
            for batch_index in range(index, (len(pairs)+pair_batch_size-1)//pair_batch_size, len(worker_devices)):
                offset = batch_index * pair_batch_size
                batch = pairs[offset:offset+pair_batch_size]
                shape = (len(positions), len(batch))
                path = output / f'mean_batch{batch_index:05d}.npy'
                temporary = path.with_suffix('.npy.partial')
                try:
                    with temporary.open('xb') as f:
                        np.lib.format.write_array_header_1_0(f, dict(
                            descr=np.lib.format.dtype_to_descr(np.dtype(np.float32)),
                            fortran_order=False, shape=shape))
                        data_offset = f.tell()
                        f.flush()
                        size = data_offset + shape[0] * shape[1] * 4
                        os.ftruncate(f.fileno(), size)
                        if preallocate:
                            _preallocate_file(f.fileno(), size)
                        if shape[0]:
                            timing = contexts[index].export_mean_fd(
                                batch, f.fileno(), data_offset, tile_sites=tile_sites,
                                calibration_factor=calibration['generation_rescaling_factor'],
                                buffers=staging_buffers)
                        else:
                            timing = dict(output_bytes=0, posterior_means_validated=True)
                        if durable:
                            os.fsync(f.fileno())
                    temporary.replace(path)
                    records.append(dict(file=path.name, pair_offset=offset, n_pairs=len(batch),
                                        gpu_id=device, cpu_affinity=cpus, **timing))
                finally:
                    temporary.unlink(missing_ok=True)
            return records
        finally:
            if cpus and original_cpus is not None:
                os.sched_setaffinity(0, original_cpus)

    try:
        with ThreadPoolExecutor(max_workers=len(worker_devices)) as pool:
            futures = [pool.submit(worker, i, device) for i, device in enumerate(worker_devices)]
            records = [r for future in futures for r in future.result()]
    finally:
        contexts.clear()
    records.sort(key=lambda r: r['pair_offset'])
    result = dict(complete=True, format='npy', dtype=np.dtype(np.float32).str,
                  shape=[len(positions), len(pairs)], calibration=calibration,
                  gpu_ids=devices, pair_batch_size=pair_batch_size, tile_sites=tile_sites,
                  staging_buffers=staging_buffers, durable=bool(durable),
                  workers_per_gpu=workers_per_gpu, active_workers=len(worker_devices),
                  preparation_seconds=preparation, elapsed_seconds=time.perf_counter()-start,
                  positions_file='positions.npy', pairs_file='pairs.npy', files=records)
    temporary = output / 'manifest.json.partial'
    with temporary.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
        if durable:
            f.flush()
            os.fsync(f.fileno())
    temporary.replace(output / 'manifest.json')
    if durable:
        _sync_directory(output)
        _sync_directory(output.parent)
    return result
