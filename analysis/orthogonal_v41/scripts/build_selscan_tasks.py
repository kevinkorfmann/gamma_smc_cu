#!/usr/bin/env python
"""Build the selscan SLURM task list (chr, pop).

Writes one line per task to analysis/orthogonal_v41/scripts/selscan_tasks.txt.
22 chromosomes x 26 1KG populations = 572 tasks.
"""

import os

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
OUT = os.path.join(REPO, "analysis/orthogonal_v41/scripts/selscan_tasks.txt")

POPS = [
    "ACB", "ASW", "BEB", "CDX", "CEU", "CHB", "CHS", "CLM", "ESN", "FIN",
    "GBR", "GIH", "GWD", "IBS", "ITU", "JPT", "KHV", "LWK", "MSL", "MXL",
    "PEL", "PJL", "PUR", "STU", "TSI", "YRI",
]

CHRS = list(range(1, 23))


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w") as f:
        for chr_num in CHRS:
            for pop in POPS:
                f.write(f"{chr_num}\t{pop}\n")
                n += 1
    print(f"Wrote {n} tasks -> {OUT}")


if __name__ == "__main__":
    main()
