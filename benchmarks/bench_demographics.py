"""
Demographic robustness benchmark: run both tmrca.cu and Schweiger's gamma_smc
on 3 stdpopsim demographic models to compare accuracy under misspecification.
Regenerates fig6_demographics with Schweiger's curve added.
"""
import time, os, sys, json, subprocess, tempfile, gc
import numpy as np
import msprime
import stdpopsim
from scipy.stats import pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, "/sietch_colab/kkor/tmrca.cu/python")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tmrca_cu import _core

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

MU, RHO, NE = 1.25e-8, 1e-8, 10000
FF = "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt"
GSMC = "/sietch_colab/kkor/gamma_smc/bin/gamma_smc"
OUT = "/sietch_colab/kkor/tmrca.cu/benchmarks"

C_OURS = "#3182bd"
C_SCHW = "#e6550d"
C_TRUTH = "#252525"

def true_t(ts, i, j, positions):
    t = np.empty(len(positions))
    tit = ts.trees(); tree = next(tit)
    for idx, p in enumerate(positions):
        while p >= tree.interval.right: tree = next(tit)
        t[idx] = tree.tmrca(i, j)
    return t

def r_log(x, y):
    lx = np.log(np.maximum(x, 1e-10))
    ly = np.log(np.maximum(y, 1e-10))
    return float(pearsonr(lx, ly)[0])

def schweiger_pair_order(nh):
    nd = nh // 2; pairs = []
    for dA in range(nd):
        for dB in range(dA, nd):
            if dA == dB: pairs.append((2*dA, 2*dA+1))
            else:
                for ha in range(2):
                    for hb in range(2): pairs.append((2*dA+ha, 2*dB+hb))
    return pairs

def run_schweiger(ts, nh, mu_d, rho_d, ne_d):
    """Run Schweiger's gamma_smc and return per-site TMRCA for pair (0,1)."""
    with tempfile.TemporaryDirectory() as td:
        vp = os.path.join(td, "s.vcf")
        with open(vp, "w") as f:
            ts.write_vcf(f, contig_id="chr1")
        subprocess.run(f"bgzip -f {vp} && tabix -p vcf {vp}.gz",
                       shell=True, check=True, capture_output=True)
        out = os.path.join(td, "out")
        result = subprocess.run(
            [GSMC, "-i", vp+".gz", "-o", out,
             "-m", str(4*ne_d*mu_d), "-r", str(4*ne_d*rho_d), "-f", FF, "-h"],
            capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None, None

        dec = os.path.join(td, "out.bin")
        subprocess.run(["zstd", "-d", out, "-o", dec], capture_output=True)
        if not os.path.exists(dec):
            import shutil
            if os.path.exists(out): shutil.copy(out, dec)
            else: return None, None

        with open(dec, "rb") as f: raw = f.read()
        meta_path = out + ".meta"
        if not os.path.exists(meta_path):
            import glob
            metas = glob.glob(os.path.join(td, "*.meta"))
            meta_path = metas[0] if metas else meta_path
        with open(meta_path) as f: meta = json.load(f)
        np2 = meta["num_pairs"]; ns = meta["sequence_length"]
        cs = meta["chunk_size"]; nc = (np2+cs-1)//cs
        sp = np.array(meta["output_positions"])
        arr = np.frombuffer(raw, dtype=np.float32).reshape(nc, 2, ns, cs)
        schw_pairs = schweiger_pair_order(nh)
        # Find pair (0,1)
        for pidx, p in enumerate(schw_pairs):
            if tuple(sorted(p)) == (0, 1):
                return arr[pidx // cs, 0, :, pidx % cs], sp
        return arr[0, 0, :, 0], sp  # fallback to first pair

def savefig(fig, name):
    for fmt in ["pdf", "png"]:
        fig.savefig(f"{OUT}/{name}.{fmt}", dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  saved {name}")


# ── Demographic configs ───────────────────────────────────────
species = stdpopsim.get_species("HomSap")
demo_configs = [
    ("OutOfAfrica_3G09", "YRI", "Three-pop OOA\n(Gutenkunst 2009)"),
    ("OutOfAfrica_2T12", "AFR", "Two-pop OOA\n(Tennessen 2012)"),
    ("Africa_1T12", "AFR", "African pop\n(Tennessen 2012)"),
]

SEQ_LEN = 2_000_000
N_HAP = 20
SEED = 42

fig, axes = plt.subplots(3, 3, figsize=(7, 6.5),
                          gridspec_kw={"wspace": 0.35, "hspace": 0.5})

results_demo = []

for row_idx, (model_id, pop_name, desc) in enumerate(demo_configs):
    print(f"\n{'='*60}")
    print(f"Model: {model_id} ({desc.replace(chr(10), ' ')})")
    print(f"{'='*60}")

    model = species.get_demographic_model(model_id)
    contig = species.get_contig(length=SEQ_LEN)

    pop_names = [p.name for p in model.populations]
    use_pop = pop_name if pop_name in pop_names else pop_names[0]

    engine = stdpopsim.get_engine("msprime")
    samples = {use_pop: N_HAP // 2}
    ts_raw = engine.simulate(model, contig, samples, seed=SEED)
    mu_d = contig.mutation_rate
    try:
        rho_d = contig.recombination_map.mean_rate
    except:
        rho_d = RHO
    ts_demo = msprime.sim_mutations(ts_raw, rate=mu_d, random_seed=SEED+1)

    G = ts_demo.genotype_matrix().T.astype(np.uint8)
    pos = np.array([v.position for v in ts_demo.variants()])
    n, S = G.shape
    print(f"  n={n}, S={S}, mu={mu_d:.2e}, rho={rho_d:.2e}")

    ne_d = NE  # use constant Ne (misspecified)
    pair = (0, 1)
    truth = true_t(ts_demo, 0, 1, pos)

    # tmrca.cu forward-backward
    flow_fb = _core.gamma_smc_flow_cached_fb(
        G, pos, [pair], float(ne_d), mu_d, rho_d, FF, True, 0)["mean"][:, 0]
    r_fb = r_log(truth, flow_fb)
    print(f"  tmrca.cu fb: r={r_fb:.3f}")

    # Schweiger's gamma_smc
    print(f"  Running Schweiger...", end=" ", flush=True)
    schw_raw, sp_schw = run_schweiger(ts_demo, n, mu_d, rho_d, ne_d)
    if schw_raw is not None:
        # Scale Schweiger output to generations (raw is in coalescent units)
        truth_at_sp = true_t(ts_demo, 0, 1, sp_schw)
        scale = np.median(truth_at_sp / (schw_raw + 1e-10))
        schw_scaled = np.interp(pos, sp_schw, schw_raw * scale)
        r_sw = r_log(truth, schw_scaled)
        print(f"r={r_sw:.3f}")
    else:
        schw_scaled = None
        r_sw = None
        print("FAILED")

    results_demo.append({
        "model": model_id, "desc": desc.replace('\n', ' '),
        "r_fb": round(r_fb, 4),
        "r_sw": round(r_sw, 4) if r_sw else None,
    })

    # ── Panel (a): Ne history ─────────────────────────────────
    ax = axes[row_idx, 0]
    try:
        dd = model.model
        debug = dd.debug()
        times = np.concatenate([np.array([0.1]), np.logspace(1, 5.5, 200)])
        sz = debug.population_size_trajectory(times)
        pop_idx = pop_names.index(use_pop) if use_pop in pop_names else 0
        ne_traj = sz[:, pop_idx]
        valid = (ne_traj > 0) & np.isfinite(ne_traj)
        ax.plot(times[valid], ne_traj[valid], color="black", linewidth=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
    except Exception as e:
        ax.text(0.5, 0.5, f"{desc}", transform=ax.transAxes, ha="center", fontsize=6)
    ax.set_xlabel("Time (gen ago)")
    ax.set_ylabel("$N_e$")
    ax.set_title(desc, fontsize=6.5, fontweight="normal", loc="left")
    ax.grid(True, alpha=0.05, linewidth=0.3)

    # ── Panel (b): Per-site trace ─────────────────────────────
    ax = axes[row_idx, 1]
    step = max(1, S // 800)
    xp = pos[::step] / 1e6
    ax.plot(xp, truth[::step], color=C_TRUTH, linewidth=0.4, alpha=0.5, label="Truth")
    ax.plot(xp, flow_fb[::step], color=C_OURS, linewidth=0.4, alpha=0.8,
            label=f"tmrca.cu ($r$={r_fb:.3f})")
    if schw_scaled is not None:
        ax.plot(xp, schw_scaled[::step], color=C_SCHW, linewidth=0.4, alpha=0.7,
                label=f"Schweiger ($r$={r_sw:.3f})")
    ax.set_yscale("log")
    ax.set_xlabel("Position (Mb)")
    ax.set_ylabel("TMRCA (gen)")
    ax.legend(fontsize=4.5, loc="upper right")
    ax.grid(True, alpha=0.05, linewidth=0.3)

    # ── Panel (c): Marginal distribution ──────────────────────
    ax = axes[row_idx, 2]
    t_min = max(truth.min(), 1)
    t_max = truth.max() * 1.1
    bins = np.logspace(np.log10(t_min), np.log10(t_max), 50)
    ax.hist(truth, bins=bins, alpha=0.4, color=C_TRUTH, label="Truth", density=True)
    valid_fb = flow_fb[(flow_fb > 0) & np.isfinite(flow_fb)]
    if len(valid_fb) > 0:
        ax.hist(valid_fb, bins=bins, alpha=0.4, color=C_OURS,
                label="tmrca.cu", density=True)
    if schw_scaled is not None:
        valid_sw = schw_scaled[(schw_scaled > 0) & np.isfinite(schw_scaled)]
        if len(valid_sw) > 0:
            ax.hist(valid_sw, bins=bins, alpha=0.4, color=C_SCHW,
                    label="Schweiger", density=True)
    ax.set_xscale("log")
    ax.set_xlabel("TMRCA (gen)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=4.5)
    ax.grid(True, alpha=0.05, linewidth=0.3)

    del G, ts_demo; gc.collect()

savefig(fig, "fig6_demographics")

# Save results
with open(os.path.join(OUT, "demo_results.json"), "w") as f:
    json.dump(results_demo, f, indent=2)

print("\nDemographic accuracy summary:")
for r in results_demo:
    r_sw_s = f"{r['r_sw']:.3f}" if r['r_sw'] else "---"
    print(f"  {r['desc']}: tmrca.cu r={r['r_fb']:.3f}, Schweiger r={r_sw_s}")
