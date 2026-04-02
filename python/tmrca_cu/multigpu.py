"""
MultiGPUFlowContext: thread-based multi-GPU using per-context device management.
Each FlowContext stores its device_id and cache pointers, and calls
cudaSetDevice at the start of each method.
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tmrca_cu import _core


class MultiGPUFlowContext:
    def __init__(self, G, positions, Ne, mu, rho, flow_field_path, gpu_ids=None):
        if gpu_ids is None:
            gpu_ids = list(range(_core.get_device_count()))
        self.gpu_ids = gpu_ids
        self.n_gpus = len(gpu_ids)
        self.S = len(positions)
        
        self.contexts = []
        for gid in gpu_ids:
            _core.set_device(gid)
            ctx = _core.FlowContext(G, positions, float(Ne), mu, rho, flow_field_path, 0)
            self.contexts.append(ctx)
        _core.set_device(gpu_ids[0])
    
    def run_fb_summary(self, pairs, chunk_size=10000):
        n_pairs = len(pairs)
        if n_pairs == 0:
            return {"site_mean": np.zeros(self.S, dtype=np.float32), "n_pairs": 0, "n_gpus": 0}
        
        chunk_per_gpu = (n_pairs + self.n_gpus - 1) // self.n_gpus
        
        def run_gpu(args):
            gpu_idx, gpairs = args
            if not gpairs:
                return np.zeros(self.S, dtype=np.float64), 0
            ctx = self.contexts[gpu_idx]
            site_sum = np.zeros(self.S, dtype=np.float64)
            total = 0
            for i in range(0, len(gpairs), chunk_size):
                chunk = gpairs[i:i+chunk_size]
                r = ctx.run_fb_summary(chunk)  # cudaSetDevice called internally
                site_sum += r["site_mean"].astype(np.float64) * len(chunk)
                total += len(chunk)
            return site_sum, total
        
        gpu_pairs = []
        for i in range(self.n_gpus):
            s = i * chunk_per_gpu; e = min(s + chunk_per_gpu, n_pairs)
            gpu_pairs.append(pairs[s:e] if s < e else [])
        
        with ThreadPoolExecutor(max_workers=self.n_gpus) as ex:
            results = list(ex.map(run_gpu, enumerate(gpu_pairs)))
        
        total_sum = np.zeros(self.S, dtype=np.float64); total_pairs = 0
        for ss, c in results:
            total_sum += ss; total_pairs += c
        
        return {"site_mean": (total_sum/total_pairs).astype(np.float32),
                "n_pairs": total_pairs, "n_gpus": self.n_gpus}
