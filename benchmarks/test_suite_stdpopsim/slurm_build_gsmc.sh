#!/bin/bash
#SBATCH --job-name=build_gsmc
#SBATCH --partition=b200-mig90
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=/vast/projects/smathi/cohort/kkor/tmrca.cu/benchmarks/test_suite_stdpopsim/logs/build_gsmc_%j.log

set -euo pipefail

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/benchmarks/test_suite_stdpopsim
PIXI_ENV=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default

mkdir -p "${BASE}/logs"
cd "${BASE}"

# gamma_smc uses a plain Makefile and links against htslib + zstd. Both are
# already present in the repo's pixi env, so we point CPPFLAGS/LDFLAGS there.
export CPPFLAGS="-I${PIXI_ENV}/include"
export LDFLAGS="-L${PIXI_ENV}/lib -Wl,-rpath,${PIXI_ENV}/lib"
export CPATH="${PIXI_ENV}/include:${CPATH:-}"
export LIBRARY_PATH="${PIXI_ENV}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}"

if [ ! -d gamma_smc ]; then
    git clone https://github.com/regevs/gamma_smc.git
fi

cd gamma_smc

# Clean any stale build artifacts from a previous failed attempt.
rm -f src/*.o bin/gamma_smc
mkdir -p bin

# The Makefile pulls CPPFLAGS/LDFLAGS from the environment. Override MARCH
# too so the binary runs on any node in the partition, not just the builder.
make CXX=g++ MARCH=x86-64-v3 bin/gamma_smc

test -x bin/gamma_smc || { echo "FATAL: bin/gamma_smc missing after build"; exit 1; }
ldd bin/gamma_smc | head -20
./bin/gamma_smc --help 2>&1 | head -5 || true
echo "build OK: $(realpath bin/gamma_smc)"
