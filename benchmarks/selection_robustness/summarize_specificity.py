"""Combine completed map sensitivities and estimate ratios of ensemble means."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import digest, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',action='append',required=True,
        help='Map label=analysis directory');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(4381637);results=[];provenance={}
    for arg in a.analysis:
        label,directory=arg.split('=',1);directory=Path(directory)
        s=json.loads((directory/'summary.json').read_text())
        with (directory/'replicates.csv').open() as f:rows=list(csv.DictReader(f))
        metrics=['focal_pi_ratio','focal_tmrca_ratio','focal_B']
        neutral=np.array([[float(r[k]) for k in metrics] for r in rows if r['regime']=='neutral'])
        for group in ['all_positive','fixed','lost']:
            selected=np.array([[float(r[k]) for k in metrics] for r in rows
                               if r['regime']=='positive' and (group=='all_positive' or r['outcome']==group)])
            if len(selected)<2:raise ValueError('Too few replicates for ratio interval')
            bsel=selected[rng.integers(len(selected),size=(10000,len(selected)))].mean(axis=1)
            bneu=neutral[rng.integers(len(neutral),size=(10000,len(neutral)))].mean(axis=1)
            ratio=100*(selected.mean(axis=0)/neutral.mean(axis=0)-1)
            ci=np.quantile(100*(bsel/bneu-1),[.025,.975],axis=0)
            row=dict(map=label,group=group,n_selected=len(selected),n_neutral=len(neutral))
            for j,k in enumerate(metrics):
                row.update({k+'_selected_mean':float(selected[:,j].mean()),
                    k+'_neutral_mean':float(neutral[:,j].mean()),
                    k+'_relative_percent_change':float(ratio[j]),
                    k+'_relative_ci95_lo':float(ci[0,j]),k+'_relative_ci95_hi':float(ci[1,j])})
            for k in ['focal_B','focal_minus_flanks_B','outside_region_mean_B']:
                v=s['contrasts'][group][k]
                row.update({k+'_absolute_change':v['estimate'],k+'_absolute_ci95_lo':v['ci95'][0],
                            k+'_absolute_ci95_hi':v['ci95'][1]})
            results.append(row)
        fixed=s['outcomes'].get('fixed',0);total=sum(s['outcomes'].get(k,0) for k in ['fixed','lost','segregating'])
        f=fixed/total;z=1.95996398454;center=(f+z*z/(2*total))/(1+z*z/total)
        width=z*np.sqrt(f*(1-f)/total+z*z/(4*total*total))/(1+z*z/total)
        ages=[float(r['fixation_age']) for r in rows if r['outcome']=='fixed']
        provenance[label]=dict(summary_sha256=digest(directory/'summary.json'),
            replicates_sha256=digest(directory/'replicates.csv'),outcomes=s['outcomes'],
            fixation_fraction=f,fixation_wilson_ci95=[center-width,center+width],
            fixation_age_mean=float(np.mean(ages)),fixation_age_range=[min(ages),max(ages)])
    with (a.out/'combined_effects.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=results[0].keys());writer.writeheader();writer.writerows(results)
    write_json(a.out/'combined_effects.json',dict(effects=results,provenance=provenance,
        bootstrap_replicates=10000,source_sha256=digest(__file__),
        ratios='Ratios of independent ensemble means, not means of per-replicate ratios.'))


if __name__=='__main__':main()
