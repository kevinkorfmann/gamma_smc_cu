"""Sphinx config for gamma_smc_cu docs."""

project = "gamma_smc_cu"
author = "Kevin Korfmann"
copyright = "2026, Kevin Korfmann"

extensions = [
    "myst_parser",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]

source_suffix = {".md": "markdown"}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "gamma_smc_cu"
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#3182bd",
        "color-brand-content": "#3182bd",
    },
    "dark_css_variables": {
        "color-brand-primary": "#58a6ff",
        "color-brand-content": "#58a6ff",
    },
}
