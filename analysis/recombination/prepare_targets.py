#!/usr/bin/env python3
"""Build an auditable GRCh38 manifest from the current study revision."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd

OVERRIDES = dict(GRK2='GIH', TREM2='IBS', TREML1='IBS', IFIH1='IBS',
                 CCDC92='CDX', CLEC6A='CDX', SLC6A15='CHS', BPIFA2='GIH', LCT='CEU', ADAM22='CDX', CCDC70='GIH',
                 TMEM30A='CDX', ZNF420='CDX', RAB11FIP3='GIH', C11orf65='GIH')
ALIASES = {'ADRBK1':'GRK2', 'Ccdc92':'CCDC92', 'SPLUNC2':'BPIFA2', 'C20orf70':'SMIM28'}
NON_GENES = {'ARG-Needle','GENE','Relate','asmc','cxt','selscan','tsdate','tsinfer','CYP3A'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--revision',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    stats=pd.read_csv(a.revision/'data/genome_wide_stats.csv').set_index('gene_name')
    paths=[a.revision/x for x in ['main.tex','si.tex','tables/table1_known_sweeps.tex','tables/table2_novel_findings.tex']]
    sources={}
    for path in paths:
        for name in re.findall(r'\\textit\{([A-Za-z0-9/\-]+)\}',path.read_text()):
            if name not in NON_GENES:
                sources.setdefault(ALIASES.get(name,name),set()).add(str(path.relative_to(a.revision)))
    display=a.revision/'data/figure2_manhattan/display_classification.csv';paths.append(display)
    for name in pd.read_csv(display).gene_name:
        sources.setdefault(name,set()).add('Figure 2 highlighted locus')
    sources.setdefault('FOXP4',set()).add('TREM-region neighbour discussed in analysis')
    rows=[];unresolved=[]
    for name,source in sources.items():
        if name not in stats.index:
            unresolved.append({'gene':name,'sources':sorted(source),'reason':'Absent from protein-coding gene rank table; TREML3P lies inside TREM2 regional map.'});continue
        r=stats.loc[name]
        if not all(pd.notna(r[x]) for x in ['chr','start','end','min_pop']):
            raise ValueError(f'Missing coordinates/population: {name}')
        start,end=int(r.start),int(r.end)
        lo,hi=max(0,start-1-500000),end+500000
        if name in ['GRK2','TREM2']:
            import numpy as np
            lo,hi=map(int,np.load(a.revision/f'data/figure45_case_studies/{name}_plot_data.npz')['window'])
            lo-=1  # display inputs are 1-based positions
        pop=OVERRIDES.get(name,r.min_pop)
        priority=0 if name in ['GRK2','TREM2'] else 1 if name in OVERRIDES else 2 if any(x.endswith('.tex') for x in source) else 3
        rows.append(dict(gene=name,gene_id=r.gene_id,chrom=f'chr{int(r.chr)}',gene_start0=start-1,gene_end0=end,
                         display_start0=lo,display_end0=hi,extract_start0=max(0,lo-500000),extract_end0=hi+500000,
                         focal_population=pop,populations=','.join(dict.fromkeys([pop,'YRI'])),priority=priority,
                         minimum_tmrca_rank=float(r.min_rank),sources='; '.join(sorted(source))))
    rows.sort(key=lambda r:(r['priority'],r['gene']))
    pd.DataFrame(rows).to_csv(a.output/'targets.tsv',sep='\t',index=False)
    paths.append(a.revision/'data/genome_wide_stats.csv')
    report=dict(assembly='GRCh38',coordinates='0-based half-open',n_genes=len(rows),n_maps=sum(len(r['populations'].split(',')) for r in rows),
                aliases=ALIASES,unresolved=unresolved,source_sha256={str(p.relative_to(a.revision)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                population_rule='Case-study population where specified, otherwise minimum-rank population, plus YRI. Phase-3 sample membership in GRCh38 high-coverage VCF; no pooling across populations.',
                interpretation='Exploratory LD-derived maps; selection/demography can affect inference. Comparison with existing TMRCA is descriptive, not a causal test.')
    (a.output/'target_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
