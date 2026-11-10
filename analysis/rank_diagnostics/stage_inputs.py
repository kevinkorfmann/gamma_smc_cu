"""Freeze small, local manuscript sources and exact inference accumulators."""
from pathlib import Path
import argparse
import shutil
import numpy as np
import pandas as pd
import stdpopsim
from common import POPS,sha,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    manuscript=a.workspace/'manuscript/v5.2_revisions'
    records={}
    def copy(source,name):
        dest=a.output/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);records[name]={'source':str(source),'sha256':sha(dest)}
    copy(manuscript/'data/genome_wide_ranks.csv','atlas.csv')
    copy(manuscript/'data/gamma_rerun_2026/stage5_loci.csv','stage5.csv')
    copy(manuscript/'data/bgs_2026/gene_bgs_annotations.csv.gz','barroso.csv.gz')
    for pop in ['YRI','CEU']:
        copy(manuscript/f'data/recombination_2026/bgs/buffalo_kern2024_{pop}_CADD6_100kb.tsv.gz',f'buffalo_{pop}.tsv.gz')
    copy(manuscript/'data/recombination_2026/bgs/buffalo_kern2024_provenance.json','buffalo_provenance.json')
    root=a.workspace/'archive/reruns/review-20260925/selected-full-chromosome-v1'
    copy(root/'winner_registry.json','winner_registry.json')
    tables=[];sources=[]
    for c in range(1,23):
        for pop in POPS:
            f=root/f'genome/chr{c}/{pop}.npz'
            with np.load(f,allow_pickle=False) as z:
                tables.append(pd.DataFrame({'gene_id':z['gene_id'],'pop':pop,'chr':c,
                  'n_sites':z['n_sites_per_gene'],'pairs':z['count'],
                  'log_sum':z['log_sum'],'n_pairs_total':int(z['n_pairs_total'])}))
            sources.append({'chr':c,'pop':pop,'sha256':sha(f)})
    pd.concat(tables,ignore_index=True).to_csv(a.output/'site_counts.csv.gz',index=False)
    records['site_counts.csv.gz']={'sha256':sha(a.output/'site_counts.csv.gz'),'sources':sources}
    for mapid in ['DeCodeSexAveraged_GRCh38','PyrhoYRI_GRCh38']:
        m=stdpopsim.get_species('HomSap').get_genetic_map(mapid)
        if not m.is_cached():m.download()
        for c in range(1,23):
            source=m.map_cache_dir/m.file_pattern.format(id=str(c))
            copy(source,f'{mapid}/chr{c}.txt')
        records[mapid]={'url':m.url,'archive_sha256':m.sha256,'assembly':'GRCh38','file_pattern':m.file_pattern}
    for pop in POPS:
        copy(a.workspace/f'gamma_smc_cu/analysis/orthogonal_v41/selscan_genelevel/{pop}.csv',f'historical_ihs/{pop}.csv')
    write_json(a.output/'SOURCES.json',records)
    print(f'Frozen {len(tables)} inference files and {len(records)} input records.')

if __name__=='__main__':main()
