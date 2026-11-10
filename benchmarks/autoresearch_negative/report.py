"""Audit complete negative-selection fits and compare the unchanged positive fit."""
import argparse,json
from pathlib import Path
import numpy as np
from metrics import digest,write_json,scores


def main():
    p=argparse.ArgumentParser()
    for n in ['run','targets','positive-run','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--sensitivity',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    completion=json.loads((a.run/'completion.json').read_text());assert completion['complete']
    manifest=json.loads((a.run/'manifest.json').read_text());report={}
    for key,summary in completion['results'].items():
        target=dict(np.load(a.targets/key/'target.npz'));best=np.load(a.run/key/'validation_best.npz')
        checked=[]
        for f in sorted((a.run/key).glob('**/complete.json')):
            d=json.loads(f.read_text())
            assert digest(f.parent/'profile.npz')==d['profile_sha256']
            assert digest(f.parent/'ancestry.trees')==d['ancestry_sha256']
            assert digest(f.parent/'variants.trees')==d['variants_sha256']
            if d['config']['model']=='negative':
                assert d['config']['s']<0 and all(x<0 for x in d['negative_s_range'])
            checked.append(d)
        pred=best['inferred'].mean(0)
        assert np.isclose(scores(pred,target)['all_bins'],summary['validation']['best']['all_bins'])
        positive=json.loads((a.positive_run/key/'summary.json').read_text())
        rng=np.random.default_rng(739290);neutral=np.load(a.run/key/'validation_tuned_neutral.npz')['inferred']
        boot=[]
        for i in range(2000):
            ix=rng.integers(0,len(best['inferred']),len(best['inferred']))
            nerr=scores(neutral[ix].mean(0),target)['all_bins'];err=scores(best['inferred'][ix].mean(0),target)['all_bins']
            boot.append(100*(1-err/nerr))
        report[key]=dict(summary=summary,previous_positive_error=positive['validation']['best']['all_bins'],
            validated_replicate_files=len(checked),
            max_uncoalesced_fraction=max(d.get('raw_diagnostics',{}).get('uncoalesced_fraction',0) for d in checked),
            improvement_vs_neutral_95pct_mc_interval=np.quantile(boot,[.025,.975]).tolist(),
            parameter_interpretation='Synthetic selected blocks are flexible fitting assumptions; not estimates of human function')
    sens=None
    if a.sensitivity and (a.sensitivity/'completion.json').exists():sens=json.loads((a.sensitivity/'completion.json').read_text())
    write_json(a.out/'summary.json',dict(results=report,sensitivity=sens,
        campaign_manifest_sha256=digest(a.run/'manifest.json'),source_sha256=digest(__file__)))
    lines=['# Negative-selection profile fitting','',
        'Independent SLiM optimization against the frozen GRK2–GIH and TREM2–IBS targets. '
        'All observed bins were fitted; only simulation random seeds are held out. '
        'There are 24 fresh validation replicates per model. The initial campaign uses approximate '
        'Q=10 rescaling and 10Ne forward burn-in. There are no beneficial mutations or conditioned sweeps.','',
        '| Target | Negative Gamma error | True TMRCA error | Neutral-marker Gamma error | Fitted neutral error | Previous positive error |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key,r in report.items():
        v=r['summary']['validation']
        lines.append(f"| {key} | {v['best']['all_bins']:.3f} | {v['genealogical_truth']['all_bins']:.3f} | {v['neutral_reporter']['all_bins']:.3f} | {v['tuned_neutral']['all_bins']:.3f} | {r['previous_positive_error']:.3f} |")
    lines += ['', 'The primary Gamma assay includes selected and neutral variants. '
        'The neutral-marker assay uses the same genealogies with an independent neutral mutation process. '
        'Differences between these curves distinguish variant depletion from changes in linked genealogy. '
        'The previous positive fit used neutral markers and a different random-seed campaign; this is a '
        'descriptive comparison, not evidence for the preferred biological mechanism.','']
    for key,r in report.items():
        lines += ['## '+key,'','```json',json.dumps(r['summary']['chosen'],indent=2),'```','',
            f"Simulation-bootstrap interval for improvement over the tested neutral comparator: "
            f"{r['improvement_vs_neutral_95pct_mc_interval'][0]:.1f}% to {r['improvement_vs_neutral_95pct_mc_interval'][1]:.1f}%. "
            'This interval quantifies Monte Carlo uncertainty only.','']
    lines += ['## Scaling and burn-in','']
    if sens:
        lines += ['Fresh-seed checks keep the winning configuration fixed. Q=5 reduces acceleration; '
            '20Ne doubles burn-in. Four replicates per check are a diagnostic, not proof of convergence or Q=1 equivalence.','',
            '| Target | Check | Gamma error | Weighted log change vs Q10 mean |','| --- | --- | ---: | ---: |']
        for key,checks in sens['results'].items():
            for name,r in checks.items():lines.append(f"| {key} | {name} | {r['error']:.3f} | {r['profile_change']:.3f} |")
    else:lines += ['Smaller-Q and longer-burn checks are pending. Results are preliminary.']
    lines += ['','## Reproduction and limitations','',
        'The observed means, deCODE map, physical mutation rate, chromosome Gamma calibration, '
        '20 pair identities and metric remain fixed. All negative coefficients and saved artifact hashes '
        'were audited. Large tree files remain on Sesame.','',
        'Parameters were optimized for resemblance. A synthetic selected block is not an empirical '
        'functional annotation. Constant population size, a single fixed deleterious effect, approximate '
        'SLiM rescaling and 3-Mb inference context limit biological interpretation. No population-specific '
        'demography, phasing error or accessibility masks are fitted.','',
        'Methods: [SLiM](https://messerlab.org/slim/), '
        '[pyslim workflow](https://tskit.dev/pyslim/docs/stable/tutorial.html), '
        '[Karpathy autoresearch](https://github.com/karpathy/autoresearch).']
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(a.out),'validated':sum(r['validated_replicate_files'] for r in report.values())}))

if __name__=='__main__':main()
