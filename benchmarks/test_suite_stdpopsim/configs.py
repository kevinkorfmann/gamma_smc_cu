"""
Hand-picked stdpopsim demographic models for cross-species benchmarking of
tmrca.cu vs gamma_smc (Schweiger and Durbin, 2023).

Running this module as __main__ resolves each (species, model) to a concrete
config dict and writes configs.json next to this file. run_one.py consumes
the JSON; slurm_array.sh reads its length to size the array.
"""
import json
import os
import sys
import warnings

import stdpopsim

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "configs.json")

# Hand-picked (species_id, model_id). Rationale: programmatic iteration over
# stdpopsim.all_species() sweeps up species with no models and contigs whose
# default length is tens of Mb, which blows up simulation wall time. This list
# covers phylogenetically diverse demographic histories while keeping each
# simulation under ~30 s at 5 Mb, 20 haplotypes.
CONFIGS = [
    # HomSap --- 6 configs spanning classic OOA variants, admixture, sawtooth
    ("HomSap", "OutOfAfrica_3G09"),
    ("HomSap", "OutOfAfrica_2T12"),
    ("HomSap", "Africa_1T12"),
    ("HomSap", "AmericanAdmixture_4B18"),
    ("HomSap", "Zigzag_1S14"),
    ("HomSap", "OutOfAfricaExtendedNeandertalAdmixturePulse_3I21"),
    # Non-human primates
    ("PonAbe", "TwoSpecies_2L11"),
    ("PanTro", "BonoboGhost_4K19"),
    # Drosophila
    ("DroMel", "African3Epoch_1S16"),
    ("DroMel", "OutOfAfrica_2L06"),
    # Arabidopsis
    ("AraTha", "SouthMiddleAtlas_1D17"),
    ("AraTha", "African2Epoch_1H18"),
    # Mosquito
    ("AnoGam", "GabonAg1000G_1A17"),
    # Domestic mammals
    ("CanFam", "EarlyWolfAdmixture_6F14"),
    ("BosTau", "HolsteinFriesian_1M13"),
]

N_HAP = 20
SEQ_LEN = 5_000_000
SEED = 42
RHO_DEFAULT = 1e-8   # fallback when stdpopsim contig has no scalar rate
MU_DEFAULT = 1.25e-8


def _pick_present_day_pop(model, needed_diploids):
    """Return the name of the first population sampled at time 0 with enough lineage slots."""
    pop_names = [p.name for p in model.populations]
    # Prefer populations whose default sampling time is 0 (present-day).
    candidates = []
    for p in model.populations:
        t = getattr(p, "default_sampling_time", None)
        if t is None or t == 0:
            candidates.append(p.name)
    if not candidates:
        candidates = pop_names
    # The first candidate is adequate; stdpopsim models rarely have per-pop caps
    # that matter at the low sample sizes we use here.
    return candidates[0] if candidates else None


def expand_configs():
    """Resolve CONFIGS into a list of dicts usable by run_one.py."""
    assert N_HAP % 2 == 0, "n_hap must be even (VCF ploidy requirement)"

    out = []
    for idx, (sp_id, model_id) in enumerate(CONFIGS):
        try:
            species = stdpopsim.get_species(sp_id)
        except Exception as e:
            warnings.warn(f"[skip] {sp_id}/{model_id}: unknown species ({e})")
            continue

        try:
            model = species.get_demographic_model(model_id)
        except Exception as e:
            warnings.warn(f"[skip] {sp_id}/{model_id}: unknown model ({e})")
            continue

        pop = _pick_present_day_pop(model, N_HAP // 2)
        if pop is None:
            warnings.warn(f"[skip] {sp_id}/{model_id}: no present-day population")
            continue

        try:
            contig = species.get_contig(length=SEQ_LEN)
        except Exception as e:
            warnings.warn(f"[skip] {sp_id}/{model_id}: cannot build contig ({e})")
            continue

        mu = getattr(contig, "mutation_rate", None) or MU_DEFAULT
        rho_source = "model"
        try:
            rho = float(contig.recombination_map.mean_rate)
            if not (rho > 0):
                rho, rho_source = RHO_DEFAULT, "default"
        except Exception:
            rho, rho_source = RHO_DEFAULT, "default"

        out.append({
            "config_idx": idx,
            "species": sp_id,
            "model_id": model_id,
            "pop": pop,
            "n_hap": N_HAP,
            "seq_len": SEQ_LEN,
            "seed": SEED,
            "mu": float(mu),
            "rho": float(rho),
            "rho_source": rho_source,
        })
    return out


def _print_table(configs):
    header = f"{'idx':>3}  {'species':<8} {'model':<48} {'pop':<8} {'mu':>10} {'rho':>10} {'rho_src':<8}"
    print(header)
    print("-" * len(header))
    for c in configs:
        print(
            f"{c['config_idx']:>3}  "
            f"{c['species']:<8} "
            f"{c['model_id']:<48} "
            f"{c['pop']:<8} "
            f"{c['mu']:>10.2e} "
            f"{c['rho']:>10.2e} "
            f"{c['rho_source']:<8}"
        )
    print(f"\n{len(configs)} configs; seq_len={SEQ_LEN:,} bp, n_hap={N_HAP}")


def main():
    configs = expand_configs()
    if not configs:
        sys.exit("No configs resolved; check stdpopsim installation and CONFIGS list.")
    with open(OUT_JSON, "w") as f:
        json.dump(configs, f, indent=2)
    _print_table(configs)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
