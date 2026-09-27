import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("rerun_recombination", Path(__file__).resolve().parents[2] /
                                            "analysis/rerun/recombination.py")
recomb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recomb)
from infer_maps import sha, validate_completed


def test_target_union_replaces_old_candidates_and_refreshes_population(monkeypatch):
    monkeypatch.setattr(recomb, "FIXED", {"CASE": "GIH"})
    ranks = pd.DataFrame([
        dict(gene_name=name, gene_id=name, chr=2, start=1000001, end=1000100,
             min_pop="IBS", min_rank=.001) for name in ("CASE", "NEW", "OLD", "KNOWN")])
    old = pd.DataFrame([dict(gene="CASE", sources="main.tex"),
                        dict(gene="KNOWN", sources="tables/table1.tex"),
                        dict(gene="OLD", sources="Figure 2 highlighted locus")])
    result = recomb.target_union(ranks, [("new candidates", pd.DataFrame({"gene_name": ["NEW"]}))], old).set_index("gene")
    assert set(result.index) == {"CASE", "NEW", "KNOWN"}
    assert result.loc["CASE"].populations == "GIH,YRI"
    assert result.loc["KNOWN"].populations == "IBS,YRI"
    assert result.loc["NEW"].gene_start0 == 1000000
    assert result.loc["NEW"].extract_start0 == 0


def test_completed_map_rejects_changed_outputs_or_settings(tmp_path):
    (tmp_path / "maps").mkdir()
    (tmp_path / "cache").mkdir()
    cache, data = tmp_path / "cache/G.npz", tmp_path / "maps/G.YRI.intervals.npz"
    cache.write_bytes(b"input")
    data.write_bytes(b"output")
    provenance = dict(script_sha256="code", manifest_sha256="targets", panel_sha256="panel",
                      model_sha256={"model": "weights"}, fastrho_version="0.1.1", seed=42)
    record = dict(provenance, gene="G", input_cache_sha256=sha(cache), output_sha256={data.name: sha(data)})
    (tmp_path / "maps/G.YRI.json").write_text(json.dumps(record))
    prefix = tmp_path / "maps/G.YRI"
    assert validate_completed(prefix, provenance, tmp_path)
    with pytest.raises(ValueError, match="different seed"):
        validate_completed(prefix, dict(provenance, seed=43), tmp_path)
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed completed output"):
        validate_completed(prefix, provenance, tmp_path)


def test_fixed_BGS_is_reannotated_with_partial_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(recomb, "checked_bgs_reference", lambda _: {})
    (tmp_path / "inputs").mkdir()
    (tmp_path / "bgs").mkdir()
    (tmp_path / "bgs/buffalo_kern2024_provenance.json").write_text("{}")
    pd.DataFrame([dict(gene_id="g1", gene_name="FIRST", chr=1, start=1, end=10),
                  dict(gene_id="g2", gene_name="SECOND", chr=1, start=11, end=20)]).to_csv(
                      tmp_path / "inputs/genome_wide_ranks.csv", index=False)
    pd.DataFrame({"gene": ["FIRST"]}).to_csv(tmp_path / "inputs/targets.tsv", sep="\t", index=False)
    for pop in ("YRI", "CEU"):
        pd.DataFrame(dict(chrom=["chr1", "chr1"], start=[0, 15], end=[5, 20], Bprime=[.2, .8])).to_csv(
            tmp_path / f"bgs/buffalo_kern2024_{pop}_CADD6_100kb.tsv.gz", sep="\t", index=False)
    from types import SimpleNamespace
    recomb.annotate_bgs(SimpleNamespace(root=tmp_path))
    result = pd.read_csv(tmp_path / "bgs_annotations/all_gene_Bprime.csv.gz")
    np.testing.assert_allclose(result.Bprime_YRI, [.2, .8])
    np.testing.assert_allclose(result.Bprime_YRI_coverage, [.5, .5])
    target = pd.read_csv(tmp_path / "bgs_annotations/target_gene_Bprime.csv")
    assert target.gene_name.tolist() == ["FIRST"]
