# Findings

**Corrected release · 27 September 2026**

The human atlas summarizes coalescence times across 26 populations in the
high-coverage 1000 Genomes panel: 3,202 individuals, 19,119 gene rows, and
143 candidate loci from the gene-level screen.

[Download the 143-locus list](https://raw.githubusercontent.com/kevinkorfmann/gamma_smc_cu/findings-2026-09-27/findings_database/releases/2026-09-27/stage5_loci.csv) ·
[Download the full gene atlas](https://raw.githubusercontent.com/kevinkorfmann/gamma_smc_cu/findings-2026-09-27/findings_database/releases/2026-09-27/genome_wide_ranks.csv.gz) ·
[All tables and data dictionary](https://github.com/kevinkorfmann/gamma_smc_cu/tree/findings-2026-09-27/findings_database/releases/2026-09-27)

## Browse the loci

Search by gene, population or chromosome. Rank is the representative gene's
lowest within-population percentile; smaller values indicate shallower
coalescence. Gene groups adjacent to one another are reported as one locus.

```{raw} html
:file: _generated/loci.html
```

## Highlights

**GRK2** has coalescence ranks of 0.19–1.16% across South Asian and European
populations. Its GIH gene body retains a coalescence signal where gene-body
iHS scores are unavailable under the analysis's frequency filter.

**TREML1 / TREM2** is a regional case study with shared shallow coalescence
across non-African populations. TREM2's ranks are 1.07–5.95%; it is studied
separately from the 1% candidate list above.

## Performance

The reported chromosome-22 benchmark processes **63,190 pairs in 15.91 seconds**
on one V100S, including buffered dense output. The CPU reference takes
1,076.74 seconds, a 67.7× runtime ratio for those native workflows.
Across 14 simulation configurations, median correlation with true TMRCA is
0.833 for CUDA and 0.832 for the CPU reference.

[Benchmark tables](https://github.com/kevinkorfmann/gamma_smc_cu/tree/findings-2026-09-27/findings_database/releases/2026-09-27)
and [methods](methods.md) give the settings and comparison scope.

## Work with the full atlas

From a checkout of the repository, with pandas installed:

```python
from pathlib import Path
import pandas as pd

release = Path("findings_database/releases/2026-09-27")
atlas = pd.read_csv(release / "genome_wide_ranks.csv.gz")
print(atlas.loc[atlas.gene_name.eq("GRK2"),
                ["gene_name", "GIH_tmrca", "GIH_rank"]])
```

`*_tmrca` is in generations and `*_rank` is a fraction: multiply by 100 for
percent. Coordinates are GRCh38. The list prioritizes regions for follow-up;
interpretation and analysis details are in [Methods](methods.md).
