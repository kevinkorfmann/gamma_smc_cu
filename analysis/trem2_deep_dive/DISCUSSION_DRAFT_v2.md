# Revised Discussion paragraph — v2 (NOT yet in main.tex)

Framing: pan-OoA sweep at the **TREML1–TREM2 locus**, missed by classical haplotype scans due to the compact gene bodies. Replaces the earlier "IBS-specific TREM2" and abandoned "FOXP4" drafts.

## Gene content at the sweep peak (chr6:41.15 Mb)

The H12 peak at 41.150 Mb sits **inside TREML1** (41,149,167--41,155,553; 6.4 kb), with **TREM2 immediately adjacent** (41,158,488--41,163,186; 4.7 kb) 3 kb downstream. An unannotated lncRNA (ENSG00000290034) spans the TREML1--TREM2 intergenic region. The two paralogs are co-regulated and functionally coupled through DAP12-based myeloid-cell signalling. Attribution between TREML1 vs TREM2 cannot be made from the haplotype data alone — fine-mapping and eQTL colocalization would be needed to distinguish them. We therefore refer to the signal as the **TREML1/TREM2 locus** throughout.

## Draft paragraph

> **A previously unreported out-of-Africa sweep at the TREML1/TREM2 locus that classical haplotype scans miss.**
>
> The strongest gene-level signal we recover at 6p21.1 lies at the TREML1/TREM2 locus, two adjacent and functionally coupled 6.4 kb and 4.7 kb myeloid-immunoreceptor genes whose within-population pairwise-TMRCA ranks below the 1\% genome-wide tail in every non-African 1000 Genomes population (IBS 0.34\%, FIN 0.33\%, GBR 0.37\%, CEU 0.50\%, TSI 0.47\%, JPT 0.41\%, CHB 0.72\%, KHV 0.96\%, YRI 1.57\%, LWK 1.45\%). Haplotype identity analysis (H12 in 400-SNP windows) places a prominent peak at chr6:41.150 Mb, inside the TREML1 gene body with TREM2 3 kb downstream, reaching H12 = 0.58--0.72 in every non-African panel and fading to 0.18 in YRI (Table~S-x). The most-differentiated variant across this block, chr6:41,166,068 (G>A), drops from 95\% frequency in Africans to 43\% in non-Africans ($|\Delta\mathrm{AF}| = 0.52$) --- a classical pan-OoA shift. No IBS-specific allele-frequency anomaly survives within the swept region: the maximum $|\Delta\mathrm{AF}\text{(IBS vs.\ CEU/FIN/GBR/TSI)}|$ among common variants is 0.06, indistinguishable from noise.
>
> This locus is conspicuously absent from canonical iHS, XP-EHH, and CMS catalogs (Voight 2006; Sabeti 2007; Pickrell 2009; Grossman 2013; Field 2016), and from published Sprime/Skov archaic-introgression maps. Our own iHS scan clarifies why: both TREML1 (6.4 kb) and TREM2 (4.7 kb) have zero sites with computable iHS in every European and East Asian panel --- the gene bodies are too compact to sustain the extended-haplotype-homozygosity decay that iHS requires once post-sweep recombination has broken down linked haplotypes. The same scan does flag TREML1 and TREM2 at genome-wide top 0.1--0.4\% in Bengali (BEB), Sri Lankan Tamil (STU), and Indian Telugu (ITU) panels, where longer haplotype blocks preserve the sweep signature. The pairwise-TMRCA statistic, by contrast, depends on coalescent relatedness within a window and is insensitive to gene length, recovering the sweep across every non-African panel where classical scans lose power. Akbari et al.'s aDNA time-series flags no variant in the region at FDR $\leq$ 0.01 (max POSTERIOR = 0.14 among gene-body variants passing filter), consistent with an ancient sweep whose frequency trajectory is pre-Holocene and therefore largely invisible to recent aDNA panels.
>
> The functional target is biologically coherent across the two paralogs. TREM2 encodes a microglial and macrophage immunoreceptor that binds anionic phospholipids, mycobacterial mycolic acids, and apoptotic-cell ligands (Colonna 2023); complete loss of TREM2 causes Nasu--Hakola disease, and partial loss-of-function variants (R47H, R62H) elevate Alzheimer's risk 2--4-fold (Jonsson 2013; Guerreiro 2013). TREML1 is expressed primarily on megakaryocytes and platelets, binds apoptotic cells, and modulates thrombo-inflammatory responses at sites of vascular damage (Washington et al. 2004, 2009). Both paralogs signal through the DAP12 ITAM adaptor. The strong purifying-selection load against TREM2 loss-of-function, together with the absence of coding variants among the most-differentiated sites, makes a regulatory sweep in the shared 5$'$ regulatory region or the TREML1--TREM2 intergenic block the most parsimonious mechanism. Fine-mapping and cell-type-resolved eQTL colocalization in microglia, monocytes, and megakaryocytes will be needed to attribute the selected regulatory effect to TREM2, TREML1, or both. A residual allele-frequency feature at chr6:41,485,209 in IBS ($|\Delta\mathrm{AF}| = 0.40$ vs.\ non-IBS) does not colocalize with the sweep block (pairwise $r^2 \leq 0.03$, H12 = 0.01) and likely reflects background-selection structure or cryptic demographic substructure rather than an independent sweep.

## Structural notes

- Drops all "IBS-specific" language.
- Drops the FOXP4 hypothesis entirely.
- Attributes the signal to **TREML1/TREM2** jointly, not TREM2 alone — H12 peak is inside TREML1, with TREM2 3 kb away.
- Frames the key point as "classical scans have a gene-size blind spot; our method closes it" --- now doubly justified (both TREML1 at 6.4 kb and TREM2 at 4.7 kb are below iHS's effective resolution in short-LD panels).
- The 41.47 Mb leftover signal gets one honest sentence at the end --- acknowledged, not claimed.

## Downstream edits required

1. **Figure captions** --- `fig_trem2_dive`: panel (a) title must change from "IBS-specific" to "pan-non-AFR"; panel (b) focal window should shift from 41.47 Mb to 41.15 Mb; **panel (h) "TREM cluster paralog layout" should be re-emphasized** since TREML1 is now a co-lead rather than a bystander; panel (f) haplotype-sharing at 41,470,132 is no longer the right anchor.
2. **Figure file names** --- `fig_trem2_dive.pdf` could be renamed `fig_treml1_trem2_dive.pdf`, but "trem2_dive" is fine as a stable internal name; titles/captions are what matter for readers.
3. **Abstract/keywords** --- replace "IBS" in any TREM2 context with "pan-non-AFR"; replace "TREM2" with "TREML1/TREM2" where describing the selection signal.
4. **Main Results section** --- the "IBS case study" heading needs a pivot. Propose: "A short-gene sweep at the TREML1/TREM2 locus invisible to iHS."
5. **Manhattan / novelty-audit labels** --- `TREM2 IBS` dot relabel to `TREML1/TREM2 (pan-non-AFR)` or similar.
6. **Novelty-audit memory file** --- `project_grk2_not_novel.md` claim ("only TREM2 (IBS) is fully clean gene+pop-novel") needs updating to reflect the pan-OoA, TREML1-co-implicated reframing.
7. **Akbari 3-method figure caption** --- if it shows TREM2 across populations, the story is now "gene-size power limit at the TREML1/TREM2 locus," not "IBS-specific."
8. **fig_trem2_dive.py** --- panel (h) TREM cluster layout should highlight BOTH TREML1 and TREM2 bounding the H12 peak; currently TREML1 is one of six paralogs with no special emphasis.

## Still open

- Wait for CLUES v2 (job 5485000, chained behind converged popsize 5484977) --- gives us sweep age and strength for the pan-OoA variant. The v1 CLUES (5484941) failed at chr6:41,166,068 on infinite-sites; v2 walks a ranked candidate list and uses the converged popsize.
- If the sweep dates < 20 kya in CLUES, that would be a surprise and would warrant a second, more pointed paragraph. If > 40 kya, the "ancient OoA sweep" framing stands as drafted.
- Decision on whether to include a supplementary figure demonstrating the gene-size/iHS power limit with simulations (would take 2--3 days to make compelling).
- eQTL colocalization between the swept haplotype and TREML1 vs TREM2 regulatory variants --- the key experiment for resolving gene attribution.
