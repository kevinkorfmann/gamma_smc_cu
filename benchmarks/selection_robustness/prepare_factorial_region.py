"""Create a generative regional map/annotation snapshot without importing a B prior."""
import argparse
from pathlib import Path
import numpy as np
import stdpopsim
from common import digest, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--left',type=int,default=36160846)
    p.add_argument('--right',type=int,default=46160846)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    species=stdpopsim.get_species('HomSap')
    gmap=species.get_genetic_map('DeCodeSexAveraged_GRCh38')
    rate=gmap.get_chromosome_map('6').slice(a.left,a.right,trim=True)
    if np.any(~np.isfinite(rate.rate)) or np.any(rate.rate<0):raise ValueError('Invalid map coverage')
    ann=species.get_annotations('ensembl_havana_104_exons')
    exons=ann.get_chromosome_annotations('6')
    exons=exons[(exons[:,1]>a.left)&(exons[:,0]<a.right)]
    exons=np.clip(exons-a.left,0,a.right-a.left)
    if np.any(exons[1:,0]<exons[:-1,1]):raise ValueError('Overlapping exons')
    np.savez_compressed(a.out/'maps.npz',map_position=rate.position,map_rate=rate.rate,exons=exons)
    write_json(a.out/'region.json',dict(assembly='GRCh38',chrom='6',left=a.left,right=a.right,
        length=a.right-a.left,interval_convention='zero-based half-open',map_id=gmap.id,
        map_url=gmap.url,map_archive_sha256=gmap.sha256,map_total_morgans=rate.total_mass,
        mean_r=rate.mean_rate,annotation_id=ann.id,exon_bp=int(np.diff(exons,axis=1).sum()),
        maps_sha256=digest(a.out/'maps.npz'),source_sha256=digest(__file__),
        focal_position0=41160846-a.left,scoring_interval0=[39648000-a.left,42664000-a.left],
        external_B_prior_used_in_simulation=False))


if __name__=='__main__':main()
