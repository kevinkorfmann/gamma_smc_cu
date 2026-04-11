"""Generate analysis/orthogonal_v41/scripts/tasks.txt from gene_list.

Lines 1-15: focal pop tasks (5 novel + 5 positive + 5 neutral)
Lines 16-30: YRI control tasks (same 15 genes, pop=YRI)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gene_list

OUT = os.path.join(HERE, "tasks.txt")

genes = gene_list.all_genes()  # 15 entries
with open(OUT, "w") as f:
    # Focal pop entries
    for gene, chr_, pop, group in genes:
        f.write(f"{gene}\t{chr_}\t{pop}\t{group}\n")
    # YRI control entries (re-tag group with "_yri" suffix in saved file)
    for gene, chr_, pop, group in genes:
        f.write(f"{gene}\t{chr_}\tYRI\tcontrol\n")

print(f"Wrote {OUT} ({2 * len(genes)} tasks)")
with open(OUT) as f:
    for i, line in enumerate(f, start=1):
        print(f"  {i:2d}: {line.strip()}")
