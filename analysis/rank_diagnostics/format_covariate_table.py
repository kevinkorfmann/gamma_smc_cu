"""Format verified estimates for the manuscript; no statistical recomputation."""
from pathlib import Path
import argparse
import csv
import json
from common import POPS,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.data/'covariate_correlations.csv'
    receipt=json.loads((a.data/'comparison_verification.json').read_text())
    assert sha(source)==receipt['outputs'][source.name]
    rows=list(csv.DictReader(source.open()))
    lookup={(r['pop'],r['kind'],r['covariate']):r for r in rows}
    out=[f'% Source SHA-256: {sha(source)}',r'\begin{table}[H]',r'\centering',
         r'\caption{\textbf{Genome-wide covariate dependence of TMRCA rank.} Marginal and partial Spearman correlations with increasing TMRCA rank. Within each population, all estimates use the same $n$ complete-covariate non-SD genes. Partial correlations control the other three ranked covariates. Sites denotes retained-site count; recombination uses the deCODE reference map and BGS uses YRI $B^{\prime}$. Full chromosome-jackknife intervals and annotation sensitivities accompany the machine-readable source tables for Supplementary Fig.~\ref{fig:rank_diagnostics}.}',
         r'\label{tab:rank_covariates}',r'\small',r'\setlength{\tabcolsep}{3.3pt}',
         r'\begin{tabular}{lr*{8}{r}}',r'\toprule',
         r'Population & $n$ & \multicolumn{4}{c}{Marginal correlation} & \multicolumn{4}{c}{Partial correlation} \\',
         r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}',
         r' & & Length & Sites & Recomb. & BGS & Length & Sites & Recomb. & BGS \\',r'\midrule']
    for pop in POPS:
        entries=[lookup[(pop,k,c)] for k in ['marginal','partial'] for c in ['length_bp','n_sites','recombination','bgs']]
        assert len({r['n'] for r in entries})==1
        out.append(' & '.join([pop,f"{int(entries[0]['n']):,}"]+[f"{float(r['rho']):.3f}" for r in entries])+r' \\')
    out.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    a.output.write_text('\n'.join(out)+'\n')

if __name__=='__main__':main()
