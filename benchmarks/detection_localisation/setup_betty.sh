#!/bin/bash
#SBATCH --job-name=detection-setup
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=00:30:00
set -euo pipefail
RUN=/vast/projects/smathi/cohort/kkor/tmrca.cu/reruns/detection-localisation-20260927
export PATH=/vast/projects/smathi/cohort/kkor/tmrca.cu/gamma_smc_cu/.pixi/envs/default/bin:$PATH
export LD_LIBRARY_PATH=/vast/projects/smathi/cohort/kkor/tmrca.cu/gamma_smc_cu/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$RUN/tools"
cd "$RUN/tools"
if [ ! -x slim43/slim ]; then
  curl -fL --retry 2 https://github.com/MesserLab/SLiM/archive/refs/tags/v4.3.tar.gz -o slim-v4.3.tar.gz
  sha256sum slim-v4.3.tar.gz > slim-v4.3.sha256
  tar xzf slim-v4.3.tar.gz
  cmake -S SLiM-4.3 -B slim43 -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  cmake --build slim43 --target slim --parallel 6
fi
"$RUN/tools/slim43/slim" -v
