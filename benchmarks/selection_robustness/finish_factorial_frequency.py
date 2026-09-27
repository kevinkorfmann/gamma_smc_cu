"""Generate analysis artifacts once the independently running bank finishes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from common import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--region',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fonts',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    while not (a.bank/'completion.json').exists():time.sleep(60)
    result=json.loads((a.bank/'completion.json').read_text())
    failed=[r for r in result['results'] if r['status']!='completed']
    if failed:
        write_json(a.out/'NEEDS_ATTENTION.json',dict(failed_families=failed,
            note='Final scientific plots withheld until failed families are accounted for.'))
        raise SystemExit(1)
    here=Path(__file__).resolve().parent
    subprocess.run([sys.executable,str(here/'analyze_factorial_frequency.py'),
        '--bank',str(a.bank),'--out',str(a.out/'analysis')],check=True)
    subprocess.run([sys.executable,str(here/'plot_factorial_frequency.py'),
        '--analysis',str(a.out/'analysis'),'--region',str(a.region),
        '--out',str(a.out/'figures'),'--fonts',str(a.fonts)],check=True)
    write_json(a.out/'READY_FOR_REVIEW.json',dict(artifact_generation_complete=True,
        visual_review_complete=False,burn_in_and_context_diagnostics_reviewed=False,
        report=str(a.out/'analysis/RESULTS.md')))


if __name__=='__main__':main()
