"""Restyle validated physical-time profiles without recomputing the analysis."""
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from common import digest,write_json
from plot_selection_spatial_presentation_v4 import make_figure,make_comparison

root=Path.cwd();src=root/'results/stronger_bgs_v1/spatial_v3';out=root/'results/stronger_bgs_v1/spatial_v4'
out.mkdir(exist_ok=True)
s=json.loads((src/'summary.json').read_text())
assert s['time_units']=='physical generations' and json.loads((src/'audit.json').read_text())['passed']
for f in (root/'references/figure_fonts').glob('*.ttf'):font_manager.fontManager.addfont(f)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7.2,
    'axes.labelsize':7.5,'axes.titlesize':8.1,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.6,
    'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.6,'ytick.major.size':2.6,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
m=json.loads((root/'runs/stronger_bgs_v1/calibration_v1/manifest.json').read_text())
models={x['name']:x for x in m['models']};groups=[];stats={}
with np.load(src/'profiles_and_bias.npz') as z:
    for row in s['groups']:
        g=dict(row);name=g['name'];model=name.removeprefix('spatial_');old=model.startswith('original_')
        region=Path(models['annotated_gamma' if old else model]['region'])
        g['region']=json.loads((region/'region.json').read_text());g['maps']=np.load(region/'maps.npz')
        if old:
            bg=model.split('_')[1]
            g['intervals']=g['maps']['exons'] if bg=='bgs' else np.empty((0,2))
            g['subtitle']=f"Original bank · {g['snapshots']} fixed sweeps from {g['contributing_founders']}/30 simulation replicates · 50 haplotypes / 1,225 pairs"
            g['reference_label']='Matched no-sweep truth'
            g['caution']='Conditional on natural fixation; no-sweep controls weighted by fixed introductions within each simulation replicate.'
        else:
            g['intervals']=g['maps']['selected_intervals']
            g['subtitle']='BGS only · 6 independent simulation replicates · 50 haplotypes and 1,225 pairs per replicate'
            g['reference_label']='Neutral truth reference'
            g['failed_burn']=model=='density30_s1e3'
        groups.append(g);stats[name]={}
        for width in [50000,10000]:
            prefix=name+'_'+str(width)+'_'
            stats[name][width]={k.removeprefix(prefix):z[k].copy() for k in z.files if k.startswith(prefix)}
with PdfPages(out/'spatial_dips_and_biases_all_models.pdf') as pdf:
    for g in groups:make_figure(g,stats[g['name']],g['maps'],g['intervals'],g['region']['left'],out,pdf)
for zoom in [True,False]:make_comparison(groups,stats,out,zoom)
shutil.copy2(src/'profiles_and_bias.npz',out/'profiles_and_bias.npz')
shutil.copy2(src/'audit.json',out/'audit.json')
shutil.copy2(root/'results/stronger_bgs_v1/spatial_time_calibration_v1/validation.json',out/'time_calibration_validation.json')
s.update(presentation_source_sha256=digest(Path(__file__).with_name('plot_selection_spatial_presentation_v4.py')),
    rendering_source_sha256=digest(__file__),previous_analysis_summary_sha256=digest(src/'summary.json'),
    presentation_changes='Founders labelled simulation replicates; focal scoring window explicit; larger filled deCODE map, consistent rate axes; no smoothing or scientific changes')
write_json(out/'summary.json',s)
print('Rendered spatial_v4 from validated, unchanged physical-time profiles')
