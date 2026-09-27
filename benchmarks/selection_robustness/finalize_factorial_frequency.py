"""Reproducible compact calibration summary, SI diagnostics, and final report."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import digest, write_json


def calibration(root):
    x=json.loads((root/'analysis/focal_replicates.json').read_text())
    out=[];rng=np.random.default_rng(620926)
    for bg in ['neutral','bgs']:
        fs=sorted({r['family'] for r in x if r['background']==bg})
        w=rng.multinomial(len(fs),np.ones(len(fs))/len(fs),size=4000)
        for stage in ['control','f25','f50','f75','f100','fixed_plus250','fixed_plus1000']:
            rows=[r for r in x if r['background']==bg and r['stage']==stage]
            sums={m:np.array([sum(r[m] for r in rows if r['family']==f) for f in fs]) for m in ['truth','auto','fixed']}
            for method in ['auto','fixed']:
                ratio=(w@sums[method])/(w@sums['truth'])
                out.append(dict(background=bg,stage=stage,method=method,
                    mean_true=float(sums['truth'].sum()/len(rows)),
                    mean_inferred=float(sums[method].sum()/len(rows)),
                    inferred_over_true=float(sums[method].sum()/sums['truth'].sum()),
                    ci95=np.quantile(ratio,[.025,.975]).tolist()))
    write_json(root/'analysis/focal_calibration.json',out)
    return out


def diagnostic_figure(root,s,a,c,fonts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from plot_factorial_frequency import STYLE,COLORS
    if fonts:
        for p in fonts.glob('*.ttf'):font_manager.fontManager.addfont(p)
    plt.rcParams.update(STYLE)
    fig,axs=plt.subplots(2,2,figsize=(6.5,4.6))
    fig.subplots_adjust(left=.17,right=.98,bottom=.12,top=.92,hspace=.65,wspace=.43)
    metrics=['focal100kb_tmrca','focal100kb_neutral_pi','genome_tmrca','genome_neutral_pi']
    labels=['Focal TMRCA','Focal neutral diversity','10-Mb TMRCA','10-Mb neutral diversity']
    for ax,name,letter,title,xlabel in [
        (axs[0,0],'BGS / neutral control','a','BGS-only effect','BGS / neutral control'),
        (axs[0,1],'BGS burn 10Ne / 5Ne','b','Burn-in sensitivity','100,000 / 50,000 generations')]:
        rows=[next(r for r in a['contrasts'] if r['contrast']==name and r['metric']==m) for m in metrics]
        for i,r in enumerate(rows):
            ax.plot(r['ci95'],[i,i],color=COLORS['bgs'],lw=.8)
            ax.plot(r['ratio'],i,'o',color=COLORS['bgs'],ms=3)
        ax.set_yticks(range(4),labels);ax.invert_yaxis()
        if letter=='b':ax.tick_params(labelleft=False)
        ax.axvline(1,color='#69737D',lw=.6,ls=':')
        ax.set(xlim=(.62,1.2),xlabel=xlabel)
        ax.set_title(letter,loc='left',pad=9)
        ax.text(.52,1.095,title,transform=ax.transAxes,ha='center',fontsize=7.2)
    stages=['f25','f50','f75','f100'];xx=[25,50,75,100]
    ax=axs[1,0]
    for bg,color in [('neutral',COLORS['sweep']),('bgs',COLORS['bgs_sweep'])]:
        rows=[next(r for r in c if r['background']==bg and r['stage']==stage and r['method']=='auto') for stage in stages]
        yy=[r['inferred_over_true'] for r in rows];ci=np.array([r['ci95'] for r in rows])
        ax.fill_between(xx,ci[:,0],ci[:,1],color=color,alpha=.12,lw=0)
        ax.plot(xx,yy,'o-',color=color,ms=3,lw=.9)
    ax.axhline(1,color='#69737D',ls=':',lw=.6)
    ax.set(xticks=xx,xlabel='Selected allele frequency (%)',ylabel='Inferred / true focal TMRCA',ylim=(.35,1.35))
    ax.set_title('c',loc='left',pad=9)
    ax.text(.5,1.095,'Gamma-SMC-CU calibration (auto)',transform=ax.transAxes,ha='center',fontsize=7.2)
    raw=json.loads((root/'analysis/inference_diagnostics.json').read_text())
    ax=axs[1,1]
    for bg,color in [('neutral',COLORS['sweep']),('bgs',COLORS['bgs_sweep'])]:
        for crop,ls,marker in [('auto_6000000','-','o'),('auto_3016000','--','s')]:
            yy=[100*np.median([r['context'][crop]['median_relative_change'] for r in raw if r['background']==bg and r['stage']==stage]) for stage in stages]
            ax.plot(xx,yy,color=color,ls=ls,marker=marker,ms=3,lw=.9)
    ax.set(xticks=xx,xlabel='Selected allele frequency (%)',ylabel='Median change from 10 Mb (%)',ylim=(0,30))
    ax.set_title('d',loc='left',pad=9)
    ax.text(.5,1.095,'Inference context (auto)',transform=ax.transAxes,ha='center',fontsize=7.2)
    ax.legend([Line2D([],[],color='black',ls='-'),Line2D([],[],color='black',ls='--')],['6 Mb','3.016 Mb'],frameon=False,loc='upper left',fontsize=6.1)
    fig.legend([Line2D([],[],color=COLORS['sweep']),Line2D([],[],color=COLORS['bgs_sweep'])],
        ['Neutral background','BGS background'],loc='lower center',bbox_to_anchor=(.57,.003),ncol=2,frameon=False)
    for ext in ['pdf','png','svg']:fig.savefig(root/f'figures/factorial_frequency_diagnostics.{ext}',dpi=400)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--fonts',type=Path)
    args=p.parse_args();root=args.root
    s=json.loads((root/'analysis/summary.json').read_text())
    a=json.loads((root/'audit/audit_summary.json').read_text());assert a['checks_passed'] and s['complete']
    c=calibration(root);diagnostic_figure(root,s,a,c,args.fonts)
    fmt=lambda r:f"{r['ratio']:.3f} ({r['ci95'][0]:.3f}–{r['ci95'][1]:.3f})"
    effect=lambda bg,stage,method:next(r for r in s['effects'] if r['background']==bg and r['stage']==stage and r['method']==method)
    lines=['# Completed TREM2 matched-map selection benchmark','',
        'The completed experiment separates a moderate BGS-only effect from a much deeper sweep-associated TMRCA trough. '
        'On the same recombination map, fixation leaves about 4% of the corresponding no-sweep focal TMRCA, with or without BGS. '
        'Gamma-SMC-CU recovers the frequency-dependent trough but overstates its depth: its focal estimates at fixation are about half the true mean pairwise TMRCA. '
        'This is a mechanism and inference-robustness result, not empirical model selection at TREM2 and not a refit of the published B′ map.','',
        '## Completed bank and sampling','',
        f"All {a['simulations']} simulations and {a['inference_snapshots']} inference snapshots completed, with no failed families. "
        'The bank contains 30 independent founders per background, one no-sweep control and 12 unconditioned introductions per founder: 60 founders, 60 controls and 720 introductions. '
        'Each snapshot uses 50 haplotypes and the same 1,225 pairs for inference and truth. Eight family workers and a serialized GPU were used on Sesame; no new bank or Betty job was launched during final review.','',
        '| Background | Introductions | Fixed | Lost | Founders with attained stages |',
        '|---|---:|---:|---:|---:|']
    for bg in ['neutral','bgs']:
        count=next(r for r in s['counts'] if r['background']==bg and r['stage']=='f100')
        outcome=s[bg]['outcomes']
        lines.append(f"| {bg} | 360 | {outcome['fixed']} | {outcome['lost']} | {count['contributing_families']} |")
    lines+=['','Every trajectory reaching 25% also reached fixation and both recovery snapshots in this realized bank. '
        'Consequently the fixed-cohort and stage-attainment analyses are identical. These are conditional effects among successful trajectories, not an average over all new mutations. '
        'Confidence intervals resample founder families (4,000 draws); neither descendants nor pairs are treated as independent replicates. '
        'The biological models are constant Ne=10,000, Q=1, μ=1.29×10⁻⁸, a single-copy beneficial allele with s=0.1 and h=0.5, '
        'and Gamma_K17 deleterious selection on 365,046 bp of Havana exons across the same 10-Mb deCODE map. No selection parameters were fitted to empirical TREM2.','',
        '## Effect sizes','',
        'Focal summaries cover 100 kb centred on chr6:41,160,846 (zero based). The following ratios use each sweep background’s own final no-sweep control.','',
        '| Stage | True, neutral background | Gamma auto, neutral background | True, BGS background | Gamma auto, BGS background |',
        '|---|---:|---:|---:|---:|']
    for stage,label in [('f25','25%'),('f50','50%'),('f75','75%'),('f100','Fixation'),('fixed_plus250','Fixation +250 generations'),('fixed_plus1000','Fixation +1,000 generations')]:
        cells=[]
        for bg in ['neutral','bgs']:
            for method in ['truth','auto']:
                r=effect(bg,stage,method);cells.append(f"{r['focal100kb_retained_tmrca']:.3f} ({r['ci95'][0]:.3f}–{r['ci95'][1]:.3f})")
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines+=['','The 95% intervals are founder-cluster bootstrap intervals. At 25% on the BGS background, the interval includes no added reduction. '
        'The 75% and fixation stages produce clear additional troughs. At +1,000 generations, true focal TMRCA still retains only 7–9% of baseline. '
        'A sensitivity analysis weighting controls by the number of successful descendants from each founder leaves the fixation result unchanged in substance '
        '(retained true TMRCA 0.038 on neutral and 0.042 on BGS backgrounds). This does not remove conditioning on successful establishment.','',
        '| No-sweep BGS / neutral contrast | Retained fraction (95% CI) |','|---|---:|']
    for r in a['contrasts']:
        if r['contrast']=='BGS / neutral control':lines.append(f"| {r['metric']} | {fmt(r)} |")
    lines+=['','The focal BGS effect is about 19%; averaged over 10 Mb it is only about 4%. '
        'The exon-masked neutral-reporter assay agrees with TMRCA: at fixation it retains 0.037 of its neutral-background control and 0.043 of its BGS-background control. '
        'Reporter diversity is computed outside selected exons and is not labelled a B′ estimate.','',
        '## Inference limitations','',
        '| Background at fixation | True focal TMRCA | Gamma auto | Inferred / true (95% CI) |','|---|---:|---:|---:|']
    for r in c:
        if r['stage']=='f100' and r['method']=='auto':
            lines.append(f"| {r['background']} | {r['mean_true']:.1f} generations | {r['mean_inferred']:.1f} generations | {r['inferred_over_true']:.3f} ({r['ci95'][0]:.3f}–{r['ci95'][1]:.3f}) |")
    lines+=['','Known-generative fixed scaling does not correct the trough: fixation estimates retain 0.492 and 0.457 of truth on neutral and BGS backgrounds. '
        'Median focal SNP counts fall from 249.5/210 in controls to 22/20 at fixation (neutral/BGS). The minimum counts at fixation are 9/11. '
        'Mean posterior TMRCA therefore detects the effect without providing accurate absolute calibration at the deepest trough.','',
        'At fixation, cropping the same genotypes from 10 Mb to 6 Mb changes auto-scaled estimates by a median 6.5%/5.3% across the scored pair-window entries; '
        '3.016-Mb crops change them by 23.9%/19.8%. These are medians across replicates of within-replicate median absolute relative change. '
        'The corresponding median within-replicate 95th-percentile changes are 10.7%/8.9% for 6 Mb and 42.3%/34.9% for 3.016 Mb. '
        'Some replicates change substantially more; full distributions are archived. Fixed-scaling crop differences are negligible on the assessed interiors '
        '(the largest within-snapshot 95th-percentile change is 0.029% across all 1,320 cropped snapshots). '
        'Crop coordinates were rebased to zero. These checks measure inference context, not biological convergence as the simulated chromosome becomes longer. '
        'Gamma-SMC-CU receives the common scalar mean recombination rate, while SLiM receives the full heterogeneous map.','',
        '## Burn-in and controls','',
        '| BGS 10Ne / 5Ne | Ratio (95% CI) |','|---|---:|']
    for r in a['contrasts']:
        if r['contrast']=='BGS burn 10Ne / 5Ne':lines.append(f"| {r['metric']} | {fmt(r)} |")
    lines+=['','Mean pre-recapitation uncoalesced span falls from 22.56% at 5Ne to 1.66% at 10Ne (maximum 3.44% across the 30 final BGS founders). '
        'The 5Ne/10Ne point estimates show no substantial systematic shift, particularly across 10 Mb; focal intervals remain broad and do not prove exact equilibrium. '
        'The final BGS no-sweep continuation has focal TMRCA 1.040 times the founder (95% CI 1.001–1.082), versus 1.003 across 10 Mb (0.997–1.010). '
        'This modest local drift is retained as a limitation. Controls sampled at +100, +300 and +1,000 generations are also archived; controls are not exact first-passage-time matches.','',
        '## Interpretation and scope','',
        'The benchmark supports the narrow claim that positive selection can generate a pronounced TMRCA/diversity dip on top of modest BGS, and that Gamma-SMC-CU can recover that pattern while exaggerating its depth. '
        'It does not show that the published B′ estimator changed in response to the sweep. '
        'The exon-only DFE does not recreate the CADD6 B′ model, population-specific demography, noncoding selected sites, or a strong-BGS GRK2 null. '
        'Therefore it neither establishes positive selection at empirical TREM2 nor excludes BGS as an explanation at GRK2. '
        'ASMC/cxt production comparisons, empirical ABC, and a balancing-selection production comparison remain unfinished and are not included in this bank.','',
        '## Deliverables and verification','',
        '[Frequency and recovery figure](figures/factorial_frequency_effects.pdf), '
        '[four-arm genomic profiles, Gamma auto](figures/factorial_frequency_gamma_auto.pdf), '
        '[fixed-scaling sensitivity](figures/factorial_frequency_gamma_fixed.pdf), '
        '[diagnostic panels](figures/factorial_frequency_diagnostics.pdf). All also have PNG and SVG exports.','',
        'The audit checked all scientific configurations, map identity, simulator and inference sources, unique seeds, founder parent hashes, final ticks, '
        'single-copy introductions, natural first passages, post-fixation timing, sample/pair counts and complete scoring-interval projection support. '
        'Six production snapshots spanning both backgrounds and control/25%/fixation stages independently matched all-pair truth to branch diversity and reporter genealogies at relative tolerance 10⁻¹². '
        'The earlier same-seed observer/no-observer replay retained identical biological tables.','',
        'Final review repaired a directory glob that had treated sweep driver logs as run directories. '
        'The frozen scientific scripts, all simulations, genotypes and inference outputs were preserved; only analysis and figure generation changed. '
        'The failed builder log and initial successful figures remain archived on Sesame. '
        'The final artifact review record records visual inspection, file hashes and exact script versions.','',
        'Machine-readable data: analysis/summary.json, analysis/focal_calibration.json, audit/audit_summary.json, '
        'audit/all_introductions.json, audit/biological_measurements.json, audit/all_inference_metrics.json. '
        'The compact archive retains manifests and map inputs; large native trees and pair-level predictions remain on Sesame.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (root/'analysis/RESULTS.md').write_text(
        '# Final reviewed result\n\nSee [the completed report](../RESULTS.md) for results, '
        'diagnostics, limitations and figures. All effect estimates remain in summary.json.\n')
    (root/'figures/diagnostics_caption.txt').write_text(
        'Diagnostic panels. (a) No-sweep BGS/neutral ratios for focal (100-kb) and whole-simulation (10-Mb) TMRCA and exon-masked neutral diversity. '
        '(b) Paired 10Ne/5Ne BGS burn-in ratios. Points and bars are estimates and 95% founder-cluster bootstrap intervals. '
        '(c) Gamma-SMC-CU auto-scaled focal TMRCA divided by exact truth, with founder-bootstrap ribbons. '
        '(d) Median across attained replicates of within-replicate median absolute relative differences between cropped and 10-Mb auto-scaled inference. '
        'Lines in (d) are descriptive medians, without confidence intervals; fixed-scaling differences are negligible on the assessed interior. '
        'Both backgrounds use the same map. Connecting lines guide the eye; stage means are conditional on attainment.\n')
    write_json(root/'analysis/finalization_provenance.json',dict(source_sha256=digest(__file__),
        calibration_bootstrap_seed=620926,calibration_bootstrap_replicates=4000,
        input_summary_sha256=digest(root/'analysis/summary.json'),input_audit_sha256=digest(root/'audit/audit_summary.json')))


if __name__=='__main__':main()
