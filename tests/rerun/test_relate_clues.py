"""Run with pytest --confcutdir=tests/rerun tests/rerun/test_relate_clues.py."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("relate_clues", Path(__file__).resolve().parents[2] /
                                            "analysis/rerun/relate_clues.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)


@pytest.mark.parametrize("ancestor,ref,alt,derived", [("A", "A", "C", "C"), ("C", "A", "C", "A")])
def test_polarization_tracks_ancestry_not_majority(ancestor, ref, alt, derived):
    assert rc.polarize(ancestor, ref, alt) == derived


@pytest.mark.parametrize("ancestor,ref,alt", [("N", "A", "C"), ("G", "A", "C"),
                                           ("A", "A", "AT"), ("A", "A", "A")])
def test_ambiguous_or_unmatched_ancestry_is_rejected(ancestor, ref, alt):
    with pytest.raises(ValueError):
        rc.polarize(ancestor, ref, alt)


def test_only_focal_trees_enter_converter(tmp_path):
    source, output = tmp_path / "sampled.newick", tmp_path / "focal.newick"
    source.write_text("#chrom\tchromStart\tchromEnd\tMCMC_sample\ttree\n"
                      "chr\t0\t100\t0\t(0:1,1:1);\n"
                      "chr\t100\t200\t0\t(0:2,1:2);\n"
                      "chr\t100\t200\t1\t(0:3,1:3);\n")
    assert rc.select_focal_trees(source, output, 100) == 2
    assert "(0:1,1:1)" not in output.read_text()
    with open(source, "a") as f:
        f.write("chr\t100\t200\t1\t(0:3,1:3);\n")
    with pytest.raises(ValueError, match="Multiple focal"):
        rc.select_focal_trees(source, output, 100)


def test_trajectory_uses_normalized_grid_quantiles_and_generation_units(tmp_path):
    prefix = tmp_path / "result"
    np.savetxt(str(prefix) + "_freqs.txt", [0, 0.5, 1], delimiter=",")
    np.savetxt(str(prefix) + "_post.txt", [[0.01, 0.25], [0.98, 0.5], [0.01, 0.25]], delimiter=",")
    rc.summarize_trajectory(prefix, 28)
    rows = np.loadtxt(str(prefix) + "_trajectory95.tsv", skiprows=1)
    np.testing.assert_allclose(rows[0], [0, 0, 0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(rows[1], [1, 28, 0.5, 0.5, 0, 1])


@pytest.mark.parametrize("bad", ["none", "order", "flips"])
def test_locus_validates_vcf_ancestry_order_and_converter_changes(tmp_path, monkeypatch, bad):
    pysam = pytest.importorskip("pysam")
    fasta = tmp_path / "ancestor.fa"
    fasta.write_text(">2\n" + "A" * 200 + "\n")
    vcf = tmp_path / "input.vcf"
    vcf.write_text("##fileformat=VCFv4.2\n##contig=<ID=2,length=200>\n"
                   '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
                   "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
                   "2\t100\trs1\tA\tC\t.\tPASS\t.\tGT\t0|1\t1|0\n")
    pysam.tabix_compress(str(vcf), str(vcf) + ".gz")
    pysam.tabix_index(str(vcf) + ".gz", preset="vcf")
    prefix = tmp_path / "popsize"
    (tmp_path / "popsize.mut").write_text("id;pos\n0;100\n")
    labels = tmp_path / "subtree.poplabels"
    labels.write_text("sample population group sex\nS1 IBS EUR 0\nS2 IBS EUR 0\n")
    m = dict(output=str(tmp_path / "output"), relate_root=str(tmp_path), clues_root=str(tmp_path),
             seed=42, mutation_rate=1.25e-8, branch_samples=2, max_converter_flip_fraction=0,
             t_cutoff=2000, df=200, selection_bound=0.1, integration_points=100,
             years_per_generation=28, groups={"chr2_EUR": dict(chromosome=2, vcf=str(vcf) + ".gz", ancestor=str(fasta))},
             loci={"TEST": dict(group="chr2_EUR", population="IBS", target=100, max_distance=0)})
    runner = rc.Runner(m, "locus", "TEST")
    monkeypatch.setattr(runner, "dependency", lambda stage, task:
                        dict(ancestor=str(fasta)) if stage == "prepare" else dict(prefix=str(prefix), poplabels=str(labels)))

    def command(argv, name):
        if name == "sample":
            alleles = "ACAC" if bad == "order" else "ACCA"
            (runner.work / "sampled.sites").write_text("NAMES\t0\t1\t2\t3\nREGION\tchr\t90\t110\n100\t" + alleles + "\n")
            (runner.work / "sampled.newick").write_text("#chrom\tchromStart\tchromEnd\tMCMC_sample\ttree\n"
                "chr\t90\t110\t0\t((0:1,1:1):1,(2:1,3:1):1);\n"
                "chr\t90\t110\t1\t((0:1,1:1):1,(2:1,3:1):1);\n")
        elif name == "convert_clues" and bad == "flips":
            np.savetxt(runner.work / "converted_derived.txt", [0, 1, 1, 1], fmt="%d")
        elif name == "inference":
            assert float(argv[argv.index("--popFreq") + 1]) == 0.5
            np.savetxt(runner.work / "result_freqs.txt", [0, 0.5, 1], delimiter=",")
            np.savetxt(runner.work / "result_post.txt", [[0, 0], [1, 1], [0, 0]], delimiter=",")
            (runner.work / "result_inference.txt").write_text("logLR\tSelectionMLE1\n1\t0.01\n")
    monkeypatch.setattr(runner, "command", command)
    if bad == "none":
        result = runner.locus()
        audit = json.loads(Path(result["polarization"]).read_text())
        assert audit["derived"] == "C" and audit["vector"] == [0, 1, 1, 0]
        assert audit["derived_frequency"] == 0.5 and audit["converter_flipped_leaves"] == []
    else:
        with pytest.raises(ValueError, match="order disagrees|beyond the declared tolerance"):
            runner.locus()
