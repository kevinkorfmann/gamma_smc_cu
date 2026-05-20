# Literature dive: are the 5 v4.1 "novel" candidates really novel?

Done **2026-04-11**, before manuscript submission, to verify we are not over-claiming novelty for the five replicated population-specific sweep candidates reported in v4.1.

Method: five parallel literature searches (PubMed, Google Scholar, the major selection-scan databases, and the main published genome-wide selection scans) to find any prior report of positive selection at each gene in any human population.

**Top-line result:** All five candidates pass the negative-literature-search test. None appears as a positive-selection candidate in any prior published scan we (or the agents) could locate. **The negative result is weak evidence on its own** — it does not exclude a hit buried in a large supplementary table — and so all manuscript wording should be hedged as "to our knowledge, not previously reported in genome-wide scans" rather than asserted absolutely.

## Sources checked for every gene

For each candidate, all five of the following resources were queried:

- **1000 Genomes Selection Browser** (Pybus et al. 2014, *NAR* 42:D903) — CEU, YRI, CHB; iHS, XP-EHH, Tajima's D, Fay-Wu H, XP-CLR, FST, CLR, ΔDAF, ΔiHH. Note: lacks SAS.
- **dbPSHP** (Li et al. 2014, *NAR* 42:D910) — 15,472 curated literature loci.
- **PopHumanScan** (Murga-Moreno et al. 2019, *NAR* 47:D1080) — 22 1KG pops × 8 statistics.
- **Voight et al. 2006** (*PLOS Biol*, iHS, HapMap), **Sabeti et al. 2007** (*Nature*, LRH/XP-EHH), **Pickrell et al. 2009** (*Genome Res*, HGDP CMS/XP-EHH), **Akey et al. 2009**, **Grossman et al. 2013** (CMS, *Cell*), **Field et al. 2016** (SDS, *Science*), **Mathieson et al. 2015** (aDNA, *Nature*).
- Population-specific scans where applicable (see per-gene notes).

## Per-gene findings

### GRK2 / ADRBK1 (chr11q13.2, shared West Eurasian, 9/10 W Eurasian pops <1%)

**CORRECTION added 2026-04-11 after cross-population check:** GRK2 is not a "GIH-specific novel candidate". It is a **shared West Eurasian sweep** at <1% rank in 5/5 South Asian populations (GIH 0.23%, BEB 0.28%, PJL 0.29%, ITU 0.30%, STU 0.32%) AND 4/5 European populations (IBS 0.29%, TSI 0.30%, GBR 0.38%, CEU 0.47%; FIN borderline at 0.74%), and absent (>4%) in all AFR, EAS, and AMR populations. Nine populations across two continents replicate this signal. The framing in this notes file originally treated it as GIH-specific — that was wrong.

**Critical orthogonal observation (2026-04-11 selscan run):** GRK2 in GIH has **zero polymorphic sites inside the gene body** in the within-GIH selscan hap matrix. The site density jumps from pos 67,261,997 directly to pos 67,296,338 — a 34 kb gap that completely spans GRK2 (67,266,473–67,286,556). This is the signature of a near-completed hard sweep: loss of diversity. Within-GIH iHS cannot be computed because there is nothing to compute on. But in BEB (the next-most-depleted SAS population), 5 sites in the gene body remain, and iHS fraction-extreme ranks GRK2 at the 0.44th percentile of the BEB genome. CEU has 5 sites at iHS fraction-extreme rank 0.62%. PJL 1.69%, FIN 2.29%, IBS 4.74%. All <5% in 5/9 W Eurasian populations where iHS is measurable.

**Variant-level evidence (2026-04-11 Hudson FST run):**
- n_variants in ±500 kb window: 24,473
- SAS-depleted vs SAS-enriched (relative to non-SAS superpopulations): 21,445 : 3,028 = 7.08:1
- Max Hudson FST: 0.43
- Mean of top-10 variant FSTs: 0.41
- Most-differentiated variant: chr11:67,407,126, 1.9% in GIH vs 47% in non-SAS (AF diff −0.45)

**Prior selection report: NONE found.**

Population-specific scans checked:
- **Metspalu et al. 2011** (*AJHG*, "Shared and Unique Components … South Asia", PMC3234374) — most directly relevant SAS scan. GRK2/ADRBK1 not mentioned. Their top SAS hits: MSTN, DOK5, CLOCK, PPARA.
- **Johnson & Voight 2018** (PMC5866773) — iHS on all 26 1KG pops including GIH, ITU, STU, BEB, PJL. GRK2/ADRBK1 not flagged.

Functional context: GRK2 is a major cardiovascular regulator (β-adrenergic desensitization, upregulated in heart failure). The GRK2-cardiovascular literature is **entirely pharmacogenomic/disease-association**, not selection (e.g., Jacobson et al. 2015, PMC4581348, rs1894111 + hydrochlorothiazide BP response in Whites; rSNP panel characterized in African Americans). GRK2 is **not** a major lead gene in the 1M-individual BP GWAS (Evangelou et al. 2018, *Nat Genet*).

**Caveat for the manuscript:** 11q13 is gene-dense and recombination-active. Neighbors include POLD3, RPS3, SF1, MAP4K2, CCS, PPP2R5B, CFL1, MUS81. Before claiming GRK2 specifically, verify the actual TMRCA-minimum / most-differentiated variant lies inside the GRK2 gene body and is not LD-tagging a neighbor.

### BPIFA2 / SPLUNC2 / PLUNC2 (chr20q11.21, GIH/SAS, TMRCA rank ~0.005)

**Prior selection report: NONE found.**

Population-specific scans checked:
- **Bhattacharjee et al. 2022** (*PLOS ONE*, "Identifying signatures of natural selection in Indian populations") — 107 regions / 434 genes via PBS + XP-EHH + CLR. Main text does not mention BPIFA2 or chr20q11. **Supplementary tables not directly enumerated — should be manually grepped before submission.**
- **Mondal et al. 2016** (*Nat Genet*, Andamanese + mainland India) — focuses on body-size genes; BPIFA2 not mentioned.
- **Metspalu et al. 2011** — no BPIFA2.

The entire 20q11 BPI-fold cluster (BPIFA1/SPLUNC1, BPIFB1/LPLUNC1, BPIFA3, BPIFB2–6) has **zero prior selection reports** in humans across the searches performed.

Related precedent (does **not** implicate BPIFA2 specifically):
- MUC7 has documented positive selection in the primate lineage (anti-fungal domain remodeled, Xu et al. 2016 *MBE*).
- AMY1 copy-number variation is the textbook salivary-selection case (Perry et al. 2007, *Nat Genet*).

**Action item before submission:** download Bhattacharjee 2022 supp tables S1/S2 + Metspalu 2011 supp tables, search for "BPIFA2", "SPLUNC2", "C20orf70", and the chr20:31.8–31.9 Mb (GRCh38) window.

### SLC6A15 / SBAT1 / B0AT2 (chr12q21.3, CHS/EAS, TMRCA rank ~0.005)

**Prior selection report: NONE found.**

Population-specific scans checked:
- **Han Chinese genome-wide selection scan (2024, *Sci Bull / Innovation*, "Recent positive selection signatures reveal phenotypic evolution in the Han Chinese population")** — multi-statistic iHS / XP-EHH / PBS / nSL. Identifies 24 EAS-selected loci including MHC, IGH, STING1, PSG, ADH1B, ALDH2, ALDH3B2, OR4C16. **SLC6A15 is NOT among them, and no chr12q21 locus is listed.** This is the strongest piece of negative evidence we have for any of the five candidates.

Functional context (do **not** cite as selection evidence):
- SLC6A15 has a well-established **EUR-only** GWAS link to major depressive disorder via rs1545843 (~690 kb downstream), with eQTL evidence in hippocampus and lymphoblastoid lines (Kohli et al. 2011, *Neuron* 70:252; Quast et al. 2013, *PLOS One* 8:e68645; Schuhmacher et al. 2013, *IJNP* 16:83). Not East-Asia-specific. Functional rationale exists; selection rationale does not.

12q21.3 is otherwise unremarkable in selection-scan literature.

### CCDC92 (chr12q24.31, CDX/EAS, TMRCA rank ~0.003)

**Prior selection report: NONE found.**

Population-specific scans checked:
- **Han Chinese 2024 *Sci Bull* scan** — lists ALDH2, ADH1B, ALDH3B2 at 12q24, but **CCDC92 is NOT among the 24 regions**.
- **Ayub et al. 2014** (PMC3113599, T2D selection in EAS) — concentrated on HHEX, THADA, KCNJ11. **Explicitly not** CCDC92, even though they scanned T2D GWAS loci.
- **PopHumanScan** — CCDC92 / ZNF664 appear only as GWAS pleiotropy annotations, not as selection candidates.

**Critical: CCDC92 is NOT in LD with the ALDH2/SH2B3 East Asian sweep.**
- CCDC92 (GRCh38): chr12:**123,935,626–123,972,831**
- ALDH2 (GRCh38): chr12:**111,766,933–111,817,532**
- SH2B3/BRAP/ATXN2 cluster: ~111.4–112.0 Mb

Distance CCDC92 → ALDH2 ≈ **12.1 Mb**. The known 12q24 LD block harbouring the ALDH2/SH2B3 East Asian sweep spans ~1 Mb (Koyama et al.; Ayub et al.). CCDC92 sits an order of magnitude beyond that block — **no plausible mechanism for hitchhiking on rs671**. The CDX signal is independent. **Add this sentence to the manuscript explicitly to preempt the obvious reviewer objection.**

Functional context (relevant but not selection):
- rs11057401 (p.Ser70Cys) and rs825476: CAD, T2D, insulin resistance, WHR-adjusted-BMI in multiple GWAS.
- Female-specific WHR/adipose-tissue signal; CCDC92 KO mice show reduced obesity and improved insulin sensitivity (Wang et al. 2022, *iScience*, PMC9804112).
- TWAS hit for visceral adipose / T2D (Diabetologia 2023).

These establish that CCDC92 is functionally relevant to a metabolic phenotype that is **plausibly under selection** in a rice-cultivating Tai-Kadai population (CDX), strengthening rather than weakening the novel-sweep hypothesis.

### CLEC6A / Dectin-2-like (chr12p13.31, CDX/EAS, TMRCA rank ~0.003)

**Prior selection report: NONE found** for CLEC6A specifically, nor for cluster neighbours CLEC4A/C/D/E or CLEC7A.

Population-specific scans checked:
- **Deschamps et al. 2016** (*AJHG*, "Genomic Signatures of Selective Pressures and Introgression from Archaic Hominins at Human Innate Immunity Genes", PMC4711739) — main 1KG innate-immunity scan, 57 positive-selection hits, highlights TLR6-TLR1-TLR10. **No CLEC6A or chr12 Dectin cluster hit.**
- **Quintana-Murci & Clark 2013** (*Nature Reviews Immunology*) — emphasizes TLRs and NLRs; C-type lectins described as evolving under weaker/redundant selection.
- **Barreiro & Quintana-Murci 2010** — same.
- **Dannemann et al. 2016 / Deschamps 2016** archaic introgression at innate immunity — highlights TLR1/6/10, OAS, STAT2; **no chr12 Dectin cluster introgression**.

**Important do-not-overclaim:**
- The chr12 Dectin cluster has zero prior selection reports.
- BUT the **chr19 DC-SIGN / CD209 cluster** (a different C-type lectin cluster on a different chromosome) **does** have prior selection reports (Ortiz et al. 2008, SARS-CoV susceptibility paper).
- **Therefore: do not write "first selection signal in the C-type lectin family". Keep the novelty claim specific to the chr12 Dectin cluster.**

Functional context (not selection): only CLEC6A variant in the literature is rs12099687 (intronic, invasive aspergillosis susceptibility, Fisher et al. 2017, *Br J Haematol*).

## Suggested manuscript wording (the formula all five agents converged on)

> "*To our knowledge, [GENE] has not been reported as a positive-selection candidate in prior genome-wide scans, including iHS- and XP-EHH-based maps (Voight et al. 2006; Sabeti et al. 2007; Pickrell et al. 2009), the 1000 Genomes Selection Browser (Pybus et al. 2014), the South-Asia–specific scan of Metspalu et al. (2011) [for SAS candidates], the Indian-population scan of Bhattacharjee et al. (2022) [for SAS candidates], the Han-Chinese-specific scan of [Sci Bull 2024] [for EAS candidates], the Composite of Multiple Signals scan (Grossman et al. 2013), the Singleton Density Score (Field et al. 2016), nor the curated dbPSHP and PopHumanScan catalogues.*"

Pair-specific add-ons:
- **CCDC92**: "*CCDC92 lies ~12 Mb distal to the ALDH2/SH2B3 East Asian sweep at chr12q24.12, well outside any plausible LD block, and therefore represents an independent signal rather than hitchhiking on the rs671 sweep.*"
- **CLEC6A**: keep the novelty claim **specific to the chr12 Dectin cluster**; do not generalise to "first C-type-lectin sweep" because the chr19 DC-SIGN cluster has prior reports (Ortiz et al. 2008).
- **GRK2**: verify the actual TMRCA-minimum variant falls inside GRK2's gene body, not in a neighbor (POLD3, MAP4K2, CCS).
- **All five**: drop any wording stronger than "*to our knowledge … not reported …*"; do not write "first" or "previously unknown to selection".

## Outstanding action items before submission

1. **Manually grep the supplementary gene lists** of:
   - Metspalu et al. 2011 (AJHG)
   - Bhattacharjee et al. 2022 (PLOS ONE)
   - Han Chinese 2024 *Sci Bull* scan
   - Pybus et al. 2014 (1000 Genomes Selection Browser)
   - Pickrell et al. 2009 (Genome Res, supp)
   - Grossman et al. 2013 (CMS, supp)
   - Field et al. 2016 (SDS, supp)
   - Mathieson et al. 2015 (aDNA, supp)

   Search terms per gene:
   - GRK2 → "GRK2", "ADRBK1", "11q13.2", "chr11:67,0–67,2 Mb"
   - BPIFA2 → "BPIFA2", "SPLUNC2", "PLUNC2", "C20orf70", "chr20:31.8–31.9 Mb"
   - SLC6A15 → "SLC6A15", "SBAT1", "B0AT2", "12q21.3"
   - CCDC92 → "CCDC92", "ZNF664", "12q24.31"
   - CLEC6A → "CLEC6A", "CLECSF10", "Dectin-2", "12p13.31"

   Estimated time: ~30 min for all 5 genes × 8 supplements.

2. **Verify the TMRCA-minimum variant for GRK2 actually sits in the GRK2 gene body** rather than a 11q13 neighbor.

3. **Cross-check with the orthogonal selscan run** (Phase 2, currently running on betty as array 5223911 + retry 5224208). If iHS or nSL also flags the gene at >99th percentile in the focal population, that's an independent confirmation that strengthens the novelty claim (it would mean two methods agree but no prior scan reported it).

## Source URLs

### Selection scan databases
- [1000 Genomes Selection Browser (Pybus 2014)](https://academic.oup.com/nar/article/42/D1/D903/1058421)
- [dbPSHP (Li 2014)](https://academic.oup.com/nar/article/42/D1/D910/1045338)
- [PopHumanScan (Murga-Moreno 2019)](https://academic.oup.com/nar/article/47/D1/D1080/5134333)

### Classical genome-wide scans
- [Voight 2006 iHS (PLOS Biol)](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0040072)
- [Sabeti 2007 (Nature)](https://www.nature.com/articles/nature06250)
- [Pickrell 2009 (Genome Res)](https://genome.cshlp.org/content/19/5/826.long)
- [Grossman 2013 CMS (Cell)](https://www.sciencedirect.com/science/article/pii/S0092867412015042)
- [Field 2016 SDS (Science)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5182071/)
- [Mathieson 2015 aDNA (Nature)](https://www.nature.com/articles/nature16152)

### Population-specific scans
- [Metspalu 2011 South Asia (AJHG)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3234374/)
- [Mondal 2016 Andamanese + India (Nat Genet)](https://www.nature.com/articles/ng.3621)
- [Bhattacharjee 2022 Indian populations (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0271767)
- [Johnson & Voight 2018 shared signatures (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5866773/)
- [Han Chinese 2024 selection signatures (Sci Bull)](https://www.sciencedirect.com/science/article/pii/S2095927323005558)
- [Ayub 2014 T2D selection in East Asians (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3113599/)
- [Deschamps 2016 innate immunity selection (AJHG)](https://www.cell.com/fulltext/S0002-9297(15)00485-1)

### Per-gene functional context
- [Jacobson 2015 ADRBK1 rSNPs (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4581348/)
- [Evangelou 2018 1M-person BP GWAS (Nat Genet)](https://pubmed.ncbi.nlm.nih.gov/30224653/)
- [Kohli 2011 SLC6A15 + MDD (Neuron)](https://pubmed.ncbi.nlm.nih.gov/21521612/)
- [Quast 2013 SLC6A15 functional variants (PLOS One)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0068645)
- [Wang 2022 Ccdc92 KO metabolic phenotype (iScience)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9804112/)
- [Nandula 2020 BPIFA2 salivary surfactant (Exp Physiol)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9484039/)
