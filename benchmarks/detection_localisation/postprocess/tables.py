"""Format complete held-out neutral rates and prespecified score sensitivities."""
import argparse, hashlib, json
from pathlib import Path
import pandas as pd

METHODS=['Gamma','iHS body','XP-EHH body','iHS flanks','XP-EHH flanks','iHS maximum','XP-EHH maximum']
NAMES=['Gamma-SMC','iHS fraction','XP-EHH fraction','iHS, 100-kb flanks','XP-EHH, 100-kb flanks','Maximum absolute iHS','Maximum XP-EHH']
REGIMES=['incomplete','near_fixed','completed','older_completed']
RNAME=['Incomplete','Near fixation','Completed','Older completed']

def pct(r):return f'{100*r.rate:.1f} ({100*r.ci_low:.1f}--{100*r.ci_high:.1f})'

def run(a):
    p=pd.read_csv(a.summary/'performance.csv');a.out.mkdir(exist_ok=True,parents=True)
    def row(w,m,e,r):
        b=p[(p.width==w)&(p.method==m)&(p.endpoint==e)&(p.regime==r)]
        assert len(b)==1
        return b.iloc[0]
    null=[r'\begin{table}[H]',r'\centering',r'\caption{\textbf{Held-out neutral call rates in the controlled benchmark.} Percentages and Wilson 95\% confidence intervals among 200 independent neutral evaluation replicates. Each width, statistic and endpoint has a separate threshold from 200 neutral calibration replicates, targeting 5\% with strict exceedance. Discrete score ties can make this rule conservative. Missing scores count as non-calls. Target intervals are centred on random pseudo-targets; scans use the strongest window in the central 500\,kb.}',r'\label{tab:detection_null}',r'\footnotesize',r'\begin{tabular}{lrcc}',r'\toprule',r'Statistic & Width (kb) & Target calls (\%) & Scan calls (\%) \\',r'\midrule']
    cells=[]
    for w in [5000,20000]:
        for m,name in zip(METHODS,NAMES):
            target=row(w,m,'target','neutral');scan=row(w,m,'scan','neutral')
            null.append(f'{name} & {w//1000} & {pct(target)} & {pct(scan)} '+r'\\')
            cells.extend([dict(target),dict(scan)])
        if w==5000:null.append(r'\midrule')
    null.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (a.out/'table_detection_null.tex').write_text('\n'.join(null)+'\n')
    alt=[r'\begin{table}[H]',r'\centering',r'\caption{\textbf{Detection with alternative haplotype summaries.} Percentages and Wilson 95\% intervals across 100 independent replicates per selected history. Maximum scores use absolute iHS or signed XP-EHH inside the target window; flank scores use extreme-core fractions in disjoint 100-kb intervals on both sides, excluding the target window. All thresholds use independent neutral calibration. Missing scores count as non-detections. Corresponding achieved neutral rates are in Table~\ref{tab:detection_null}; these summaries have different detection and localisation properties.}',r'\label{tab:detection_alternatives}',r'\footnotesize',r'\setlength{\tabcolsep}{3pt}']
    for e in ['target','scan']:
        alt.extend([r'\medskip',r'\par\textbf{'+('Target-interval detection' if e=='target' else 'Scan-level detection')+r'}\par\smallskip',r'\begin{tabular}{rlcccc}',r'\toprule',r'Width & History & iHS max. & XP-EHH max. & iHS flanks & XP-EHH flanks \\',r'(kb) & & (\%) & (\%) & (\%) & (\%) \\',r'\midrule'])
        for w in [5000,20000]:
            for regime,name in zip(REGIMES,RNAME):
                values=[row(w,m,e,regime) for m in ['iHS maximum','XP-EHH maximum','iHS flanks','XP-EHH flanks']]
                alt.append(f'{w//1000} & {name} & '+' & '.join(pct(r) for r in values)+r' \\')
                cells.extend(dict(r) for r in values)
            if w==5000:alt.append(r'\midrule')
        alt.extend([r'\bottomrule',r'\end{tabular}'])
    alt.append(r'\end{table}')
    (a.out/'table_detection_alternatives.tex').write_text('\n'.join(alt)+'\n')
    loc=[r'\begin{table}[H]',r'\centering',r'\caption{\textbf{Localisation under primary and alternative summaries.} Median error (interquartile range) in kb across all computable selected replicates; the subscript gives the number of computable replicates out of 100. Each replicate retains all tied strongest windows and uses their median absolute distance from the true target. Windows scan the central 500\,kb in 5-kb steps. These summaries are not conditioned on detection. Flank scores use disjoint 100-kb regions on each side of a window, while the other scores use the window itself.}',r'\label{tab:detection_localisation}',r'\footnotesize',r'\setlength{\tabcolsep}{3pt}']
    for w in [5000,20000]:
        loc.extend([r'\medskip',r'\par\textbf{'+f'{w//1000}-kb windows'+r'}\par\smallskip',r'\begin{tabular}{lcccc}',r'\toprule',r'Statistic & Incomplete & Near fixation & Completed & Older completed \\',r'\midrule'])
        for m,name in zip(METHODS,NAMES):
            values=[row(w,m,'scan',r) for r in REGIMES]
            def error(r):
                if not r.computable:return r'---$_{0}$'
                return f'{r.localisation_median/1000:.1f} ({r.localisation_q25/1000:.1f}--{r.localisation_q75/1000:.1f})'+f'$_{{{int(r.computable)}}}$'
            loc.append(name+' & '+' & '.join(error(r) for r in values)+r' \\')
            cells.extend(dict(r) for r in values)
        loc.extend([r'\bottomrule',r'\end{tabular}'])
    loc.append(r'\end{table}')
    (a.out/'table_detection_localisation.tex').write_text('\n'.join(loc)+'\n')
    (a.out/'detection_table_values.json').write_text(json.dumps({'source_sha256':hashlib.sha256((a.summary/'performance.csv').read_bytes()).hexdigest(),'rows':cells},indent=2,default=lambda v:v.item())+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
