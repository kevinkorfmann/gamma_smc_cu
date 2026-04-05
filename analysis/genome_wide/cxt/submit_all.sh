#!/bin/bash
# Submit all 6 cxt region jobs in parallel on betty
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p ${DIR}/logs

echo "Submitting 6 cxt jobs..."
for script in \
    slurm_sh2b3_aldh2.sh \
    slurm_cyp3a.sh \
    slurm_fads1.sh \
    slurm_clec6a.sh \
    slurm_abcc11.sh \
    slurm_trpv6.sh \
; do
    JID=$(sbatch --parsable ${DIR}/${script})
    echo "  ${script} -> job ${JID}"
done

echo "All submitted. Monitor with: squeue -u \$USER"
