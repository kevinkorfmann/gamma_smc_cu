# Discussion paragraph — DRAFT for review (NOT YET IN main.tex)

Per CLAUDE.md / memory rule: exploratory text stays here until you approve integration.

## Option A — TREM–FOXP4 cluster framing (recommended: safest, most defensible)

> **A regulatory sweep in the 6p21.1 TREM–FOXP4 cluster in Iberian populations.**
>
> The strongest gene-level signal in our IBS scan falls in a 1 Mb block spanning the TREM immunoreceptor cluster (TREML1, TREM2, TREML2/4, TREM1, NCR2) and the transcription factor FOXP4. Pairwise TMRCA in IBS is collapsed relative to non-African populations (bootstrap ratio $r_\text{IBS/non-AFR}$ = 1.41, 95\% CI [1.35, 1.49]), with a sweep-allele frequency of 82\% in IBS vs.\ 56\% in other non-African populations at the most-differentiated variant (chr6:41,485,209; $\Delta$AF $=-0.40$, max $F_\text{ST}=0.29$). The focal sweep variant (chr6:41,470,132) sits $\sim$60 kb upstream of FOXP4 and $>$300 kb downstream of TREM2, placing the peak in an intergenic regulatory region rather than inside the TREM2 coding locus --- so we refer to this as a TREM–FOXP4 cluster sweep. This region is absent from canonical haplotype-based selection scans (iHS/XP-EHH/CMS/SDS; Voight 2006, Sabeti 2007, Grossman 2013, Field 2016) and from published Sprime/Skov archaic-introgression catalogs. Akbari et al.'s aDNA time-series assigns no variant in the window POSTERIOR $\geq 0.99$; the highest subthreshold posterior (0.75) falls at rs11760063, an intronic variant in FOXP4, consistent with a sweep that is either too recent or too geographically restricted for their pooled ancient-European panel to resolve.
>
> Biologically, the locus is triply anchored. FOXP4 is the lead GWAS gene for both severe and long COVID-19 (rs9367106, OR\,=\,1.63 for long COVID; Kousathanas et al.\ 2022; Lammi et al.\ 2023), is a cis-eQTL target in lung and brain, and is a direct regulator of lung secretory epithelial cell fate. The neighbouring TREM cluster encodes myeloid immunoreceptors whose functional dosage sits under strong purifying selection --- complete loss of TREM2 causes Nasu–Hakola disease and heterozygous partial loss-of-function variants confer a 2--4$\times$ Alzheimer's risk (Jonsson et al.\ 2013; Guerreiro et al.\ 2013) --- ruling out a coding sweep on the TREM2 side and leaving a regulatory mechanism as the most parsimonious model. Iberian populations have carried disproportionately high historical burdens of respiratory infection (plague, influenza, tuberculosis) and the locus is mechanistically compatible with selection acting through airway myeloid immunity, but we stop short of proposing a single historical driver. Definitive gene attribution will require fine-mapping and eQTL colocalization between the swept haplotype and the FOXP4 regulatory block (rs9367106 / rs2496644).

## Option B — FOXP4 framing (more committal, higher payoff if colocalization confirms)

> **A recent regulatory sweep at FOXP4 in Iberian populations.**
>
> […same opening; then drop the "we refer to…" sentence and replace with:] The proximity of the sweep peak to FOXP4 (60 kb) versus TREM2 (>300 kb), the Akbari subthreshold signal inside a FOXP4 intron, and FOXP4's independent eQTL and GWAS footprint make FOXP4 the most plausible functional target. […rest as above, removing caveats about TREM cluster.]

## Recommendation

**Use Option A in the current draft.** We have the proximity, eQTL, and Akbari-subthreshold evidence to name FOXP4 as the leading candidate, but we do not yet have fine-mapping / colocalization in hand, and the reviewer-facing risk of committing to FOXP4 prematurely is higher than the clarity benefit. Option A lets us discuss both genes honestly while centering FOXP4 as the probable driver. One follow-up experiment (colocalization of the sweep haplotype with the rs9367106 FOXP4 eQTL block in IBS) is enough to upgrade A → B in a revision.

## Downstream edits the new framing will require

Before swapping anything into `main.tex`, expect to touch:

1. **Section titles / subsection headings** — any "TREM2 case study" / "TREM2 deep dive" phrasing.
2. **`fig_trem2_dive` caption** — must state the peak is 60 kb from FOXP4, not inside TREM2. The haplotype-sharing panel is still valid; just re-label.
3. **`gen_fig_trem2_dive.py`** — add FOXP4 gene body to the gene-track panel alongside TREM2; consider renaming output `fig_foxp4_trem2_dive.pdf`.
4. **Manhattan / novelty-audit figure labels** — the `TREM2 IBS` dot becomes `TREM-FOXP4 IBS` (or a two-line label).
5. **`test_trem2_*` unit tests** — file names are fine (internal), but any assertion that the peak is "within TREM2" must be updated.
6. **Keywords / abstract** — "TREM2 / microglia" phrasing → "6p21.1 TREM–FOXP4 cluster / myeloid immunity / respiratory infection".
7. **Memory `project_grk2_not_novel.md`** — update: "TREM2 (IBS)" → "TREM–FOXP4 cluster (IBS)", note that FOXP4 is the leading candidate driver.

## What I recommend next

1. You read Options A vs B and tell me which framing to commit to.
2. I run LD/colocalization between our most-differentiated variant (41,485,209) and rs9367106 on the local `trem2_pm500kb.vcf.gz` in IBS (~15 min).
3. If colocalization is clean → upgrade to Option B. If ambiguous → lock Option A.
4. Then (and only then) apply the cascade of edits above to `main.tex`, figures, and tests.
