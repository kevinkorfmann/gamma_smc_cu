"""
MultiGPUFlowContext: thread-based multi-GPU using per-context device management.
Each FlowContext stores its device_id and cache pointers, and calls
cudaSetDevice at the start of each method.  The GIL is released during
GPU work so threads truly run concurrently.

Splits pairs evenly across available GPUs, runs kernels concurrently via
a persistent thread pool, and concatenates results.
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gamma_smc_cu import _core


class MultiGPUFlowContext:
    """Distributes FlowContext work across multiple GPUs.

    Parameters
    ----------
    G : np.ndarray, shape (n, S), dtype uint8
        Haploid genotype matrix.
    positions : np.ndarray, shape (S,), dtype float64
        Physical positions of segregating sites.
    Ne, mu, rho : float
        Demographic and rate parameters.
    flow_field_path : str
        Path to Schweiger's default_flow_field.txt.
    gpu_ids : list[int] or None
        GPU device IDs to use.  None = all available.
    cache_steps : int
        Flow field cache depth (0 = auto from data).
    """

    def __init__(self, G, positions, Ne, mu, rho, flow_field_path,
                 gpu_ids=None, cache_steps=0):
        if gpu_ids is None:
            gpu_ids = list(range(_core.get_device_count()))
        self.gpu_ids = gpu_ids
        self.n_gpus = len(gpu_ids)
        self.S = len(positions)

        self.contexts = []
        for gid in gpu_ids:
            _core.set_device(gid)
            ctx = _core.FlowContext(G, positions, float(Ne), mu, rho,
                                    flow_field_path, cache_steps)
            self.contexts.append(ctx)
        _core.set_device(gpu_ids[0])

        # Persistent thread pool avoids per-call ThreadPoolExecutor overhead
        self._pool = ThreadPoolExecutor(max_workers=self.n_gpus)

    def __del__(self):
        self._pool.shutdown(wait=False)

    def _split_pairs(self, pairs):
        """Split pairs list into per-GPU chunks."""
        n_pairs = len(pairs)
        chunk = (n_pairs + self.n_gpus - 1) // self.n_gpus
        gpu_pairs = []
        for i in range(self.n_gpus):
            s = i * chunk
            e = min(s + chunk, n_pairs)
            gpu_pairs.append(pairs[s:e] if s < e else [])
        return gpu_pairs

    # ------------------------------------------------------------------
    # run_fb_summary: aggregate per-site statistics across GPUs
    # ------------------------------------------------------------------
    def run_fb_summary(self, pairs, chunk_size=10000):
        n_pairs = len(pairs)
        if n_pairs == 0:
            return {"site_mean": np.zeros(self.S, dtype=np.float32),
                    "n_pairs": 0, "n_gpus": 0}

        gpu_pairs = self._split_pairs(pairs)

        def run_gpu(gpu_idx, gpairs):
            if not gpairs:
                return np.zeros(self.S, dtype=np.float64), 0
            ctx = self.contexts[gpu_idx]
            site_sum = np.zeros(self.S, dtype=np.float64)
            total = 0
            for i in range(0, len(gpairs), chunk_size):
                chunk = gpairs[i:i + chunk_size]
                r = ctx.run_fb_summary(chunk)
                site_sum += r["site_mean"].astype(np.float64) * len(chunk)
                total += len(chunk)
            return site_sum, total

        futures = [self._pool.submit(run_gpu, i, gp)
                   for i, gp in enumerate(gpu_pairs)]

        total_sum = np.zeros(self.S, dtype=np.float64)
        total_pairs = 0
        for f in futures:
            ss, c = f.result()
            total_sum += ss
            total_pairs += c

        return {"site_mean": (total_sum / total_pairs).astype(np.float32),
                "n_pairs": total_pairs, "n_gpus": self.n_gpus}

    # ------------------------------------------------------------------
    # run_fwd: forward-only across GPUs, returns per-pair results
    # ------------------------------------------------------------------
    def run_fwd(self, pairs, mean_only=True):
        """Forward-only filtering distributed across GPUs.

        Returns dict with 'mean' [S, n_pairs] (and 'lower'/'upper' if
        not mean_only).
        """
        n_pairs = len(pairs)
        if n_pairs == 0:
            d = {"mean": np.empty((self.S, 0), dtype=np.float32)}
            if not mean_only:
                d["lower"] = np.empty((self.S, 0), dtype=np.float32)
                d["upper"] = np.empty((self.S, 0), dtype=np.float32)
            return d

        gpu_pairs = self._split_pairs(pairs)

        def run_gpu(gpu_idx, gpairs):
            if not gpairs:
                return None
            return self.contexts[gpu_idx].run_fwd(gpairs, mean_only)

        futures = [self._pool.submit(run_gpu, i, gp)
                   for i, gp in enumerate(gpu_pairs)]
        results = [f.result() for f in futures]

        means = [r["mean"] for r in results if r is not None]
        out = {"mean": np.concatenate(means, axis=1)}
        if not mean_only:
            out["lower"] = np.concatenate([r["lower"] for r in results if r is not None], axis=1)
            out["upper"] = np.concatenate([r["upper"] for r in results if r is not None], axis=1)
        return out

    # ------------------------------------------------------------------
    # run_fb: forward-backward across GPUs, returns per-pair results
    # ------------------------------------------------------------------
    def run_fb(self, pairs, mean_only=True):
        """Forward-backward smoothing distributed across GPUs.

        Returns dict with 'mean' [S, n_pairs] (and 'lower'/'upper' if
        not mean_only).
        """
        n_pairs = len(pairs)
        if n_pairs == 0:
            d = {"mean": np.empty((self.S, 0), dtype=np.float32)}
            if not mean_only:
                d["lower"] = np.empty((self.S, 0), dtype=np.float32)
                d["upper"] = np.empty((self.S, 0), dtype=np.float32)
            return d

        gpu_pairs = self._split_pairs(pairs)

        def run_gpu(gpu_idx, gpairs):
            if not gpairs:
                return None
            return self.contexts[gpu_idx].run_fb(gpairs, mean_only)

        futures = [self._pool.submit(run_gpu, i, gp)
                   for i, gp in enumerate(gpu_pairs)]
        results = [f.result() for f in futures]

        means = [r["mean"] for r in results if r is not None]
        out = {"mean": np.concatenate(means, axis=1)}
        if not mean_only:
            out["lower"] = np.concatenate([r["lower"] for r in results if r is not None], axis=1)
            out["upper"] = np.concatenate([r["upper"] for r in results if r is not None], axis=1)
        return out

    @property
    def device_ids(self):
        return self.gpu_ids
