"""Lead-variant windows summarized from the same full-chromosome posteriors.

This module loads the GRCh38, one-based lead table and extracts already computed
lead arrays from genome-wide NPZ outputs; extraction never launches inference.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rerun_support import fingerprint

POPS = ['ACB','ASW','BEB','CDX','CEU','CHB','CHS','CLM','ESN','FIN','GBR','GIH',
        'GWD','IBS','ITU','JPT','KHV','LWK','MSL','MXL','PEL','PJL','PUR','STU','TSI','YRI']


def load_leads(path, chromosome):
    if path is None:
        return []
    data = pd.read_csv(path, sep='\t', dtype={'CHROM':str, 'RSID':str})
    required = {'CHROM','POS','RSID','X','S','POSTERIOR'}
    if not required.issubset(data):
        raise ValueError(f'Lead table lacks columns: {sorted(required-set(data))}')
    chrom = data.CHROM.str.removeprefix('chr')
    data = data.loc[chrom == str(chromosome)]
    records = []
    for row in data.itertuples(index=False):
        center = int(row.POS)
        if not np.isfinite(float(row.POS)) or center != float(row.POS) or center < 1:
            raise ValueError('Lead positions must be positive one-based integers')
        name = str(row.RSID) if pd.notna(row.RSID) else f'chr{chromosome}_{center}'
        records.append({'rsid':name, 'chrom':str(chromosome), 'center_pos':center,
                        'akbari_X':float(row.X), 'akbari_S':float(row.S),
                        'akbari_posterior':float(row.POSTERIOR)})
    keys = [(r['rsid'],r['center_pos']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate lead ID/position rows in the lead table')
    return records


def lead_arrays(leads, half_bp, arrays, offset, enabled):
    result = {'lead_schema_version':np.int64(1), 'lead_enabled':np.bool_(enabled),
              'lead_window_half_bp':np.int64(half_bp),
              'lead_inference_context':np.array('same full-chromosome posterior as gene summaries')}
    for key in ['rsid','chrom','center_pos','akbari_X','akbari_S','akbari_posterior']:
        dtype = str if key in ['rsid','chrom'] else (np.int64 if key=='center_pos' else np.float64)
        result['lead_'+key] = np.asarray([row[key] for row in leads],dtype=dtype)
    for key,value in arrays.items():
        result['lead_'+key] = value[offset:]
    return result


def extract(results_dir, output_dir, chromosomes=range(1,23), populations=POPS, allow_partial=False):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError('Choose a new lead extraction output directory')
    rows, inputs, missing = [], [], []
    expected_definition = {}
    source_lead_hash = None
    for chromosome in chromosomes:
        for population in populations:
            path = Path(results_dir)/f'chr{chromosome}'/f'{population}.npz'
            marker = path.with_suffix('.metadata.json')
            if not path.is_file() or not marker.is_file():
                missing.append(str(path)); continue
            metadata = json.loads(marker.read_text())
            if not metadata.get('complete') or metadata.get('calibration',{}).get('time_units') != 'generations':
                raise ValueError(f'Incomplete or uncalibrated input {path}')
            artifact = fingerprint(path)
            if artifact['sha256'] != metadata['output_sha256']['.npz']:
                raise ValueError(f'Input checksum mismatch {path}')
            lead_input = metadata['identity']['inputs'].get('lead_variants')
            if not lead_input:
                raise ValueError(f'Genome scan did not record a lead-variant input: {path}')
            if source_lead_hash is None:
                source_lead_hash = lead_input['sha256']
            elif source_lead_hash != lead_input['sha256']:
                raise ValueError('Cannot mix summaries from different lead tables')
            with np.load(path,allow_pickle=False) as z:
                if not bool(z['lead_enabled']) or int(z['lead_schema_version']) != 1:
                    raise ValueError(f'Lead summaries not enabled/recognized: {path}')
                n_leads = len(z['lead_rsid'])
                definition = [(str(z['lead_rsid'][i]),int(z['lead_center_pos'][i]),int(z['lead_window_half_bp'])) for i in range(n_leads)]
                if chromosome in expected_definition and expected_definition[chromosome] != definition:
                    raise ValueError('Lead identities or windows differ between populations')
                expected_definition[chromosome] = definition
                count = z['lead_count']
                eligible = z['lead_n_sites'] >= 2
                if not np.array_equal(count>0,eligible) or np.any(count[eligible] != int(z['n_pairs_total'])):
                    raise ValueError(f'Incomplete lead pair accumulation: {path}')
                for i in range(n_leads):
                    n = int(count[i])
                    gm = float(np.exp(z['lead_log_sum'][i]/n)) if n else np.nan
                    am = float(z['lead_lin_sum'][i]/n) if n else np.nan
                    if n and (not np.isfinite(gm) or gm <= 0 or not np.isfinite(am) or am <= 0):
                        raise ValueError(f'Invalid physical lead summary: {path}')
                    rows.append(dict(rsid=str(z['lead_rsid'][i]),chrom=str(chromosome),population=population,
                        center_pos=int(z['lead_center_pos'][i]),window_half_bp=int(z['lead_window_half_bp']),
                        akbari_X=float(z['lead_akbari_X'][i]),akbari_S=float(z['lead_akbari_S'][i]),
                        akbari_posterior=float(z['lead_akbari_posterior'][i]),geom_mean_tmrca=gm,
                        arith_mean_tmrca=am,min_tmrca=float(z['lead_min_lin'][i]) if n else np.nan,
                        n_pairs=n,n_sites=int(z['lead_n_sites'][i]),
                        frac_pairs_geom_lt_1000=float(z['lead_n_geom_lt_1000'][i]/n) if n else np.nan,
                        frac_pairs_arith_lt_1000=float(z['lead_n_arith_lt_1000'][i]/n) if n else np.nan))
            inputs.append({'npz':artifact,'metadata':fingerprint(marker)})
    if missing and not allow_partial:
        raise FileNotFoundError(f'Missing {len(missing)} full-chromosome outputs; first: {missing[0]}')
    if not rows:
        raise ValueError('No lead summaries found')
    table = pd.DataFrame(rows)
    output.mkdir(parents=True)
    table.to_csv(output/'lead_tmrca.csv',index=False)
    keys = ['chrom','center_pos','rsid','window_half_bp']
    wide = table.pivot(index=keys,columns='population',values='geom_mean_tmrca').reset_index()
    wide.columns.name = None
    wide.to_csv(output/'lead_tmrca_wide.csv',index=False)
    for (chromosome,population), group in table.groupby(['chrom','population'],sort=False):
        directory = output/f'chr{chromosome}'
        directory.mkdir(exist_ok=True)
        group.drop(columns='population').to_csv(directory/f'{population}.csv',index=False)
    manifest = {'schema_version':1,'status':'diagnostic_partial' if missing else 'complete',
                'time_units':'generations','context':'same full-chromosome posterior as gene summaries; no regional inference rerun',
                'source_lead_table_sha256':source_lead_hash,'inputs':inputs,'missing_inputs':missing,
                'n_rows':len(table),'n_leads':len(wide),
                'outputs':{str(p.relative_to(output)):fingerprint(p) for p in output.rglob('*.csv')}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--chromosomes',nargs='+',type=int,default=list(range(1,23)))
    parser.add_argument('--populations',nargs='+',default=POPS)
    parser.add_argument('--allow-partial',action='store_true',help='mark incomplete collections as diagnostic only')
    args=parser.parse_args()
    extract(args.results_dir,args.output_dir,args.chromosomes,args.populations,args.allow_partial)


if __name__=='__main__':
    main()
