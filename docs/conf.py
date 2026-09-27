"""Small, CPU-only documentation build: no CUDA imports or generated figures."""
import csv
import html
from pathlib import Path

project = "Gamma-SMC CUDA"
author = "Kevin Korfmann and Sara Mathieson"
copyright = "2026, Kevin Korfmann and Sara Mathieson"
extensions = ["myst_parser"]
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3
exclude_patterns = ["_build", "_generated", "requirements.txt"]
html_theme = "furo"
html_title = "Gamma-SMC CUDA"
html_static_path = ["_static"]
html_css_files = ["minimal.css"]
html_js_files = ["findings.js"]
html_theme_options = {
    "light_css_variables": {"color-brand-primary": "#224b67", "color-brand-content": "#224b67"},
    "dark_css_variables": {"color-brand-primary": "#9bc7e5", "color-brand-content": "#9bc7e5"},
    "source_repository": "https://github.com/kevinkorfmann/gamma_smc_cu/",
    "source_branch": "main", "source_directory": "docs/",
}
html_show_sphinx = False
html_show_copyright = False

# Render the browsing table from the same versioned CSV offered for download.
root = Path(__file__).resolve().parent
_release_dir = root.parent / "findings_database/releases/2026-09-27"
with (_release_dir / "stage5_loci.csv").open(newline="") as stream:
    loci = list(csv.DictReader(stream))
rows = []
for locus in loci:
    values = [locus["representative_gene"], locus["locus_id"],
              locus["focal_population"], f'{100 * float(locus["min_rank"]):.3f}',
              locus["cluster_size"]]
    search = " ".join(values + [locus["cluster_members"]])
    rows.append('<tr data-search="' + html.escape(search, quote=True) + '">' +
                ''.join('<td>' + html.escape(v) + '</td>' for v in values) + '</tr>')
table = '''<div class="findings-browser">
<label for="locus-search">Search the findings</label>
<input id="locus-search" type="search" placeholder="Gene, population, or chromosome" autocomplete="off">
<p id="locus-count" role="status" aria-live="polite">143 loci</p>
<div class="locus-table-wrap"><table id="locus-table">
<thead><tr><th scope="col">Representative gene</th><th scope="col">GRCh38 locus</th>
<th scope="col">Focal population</th><th scope="col">Minimum rank (%)</th>
<th scope="col">Genes</th></tr></thead><tbody>'''
table += '\n'.join(rows) + '</tbody></table></div></div>'
(root / '_generated').mkdir(exist_ok=True)
(root / '_generated/loci.html').write_text(table)
