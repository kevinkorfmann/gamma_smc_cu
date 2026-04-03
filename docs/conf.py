project = "tmrca.cu"
author = "Kevin Korfmann"
extensions = ["myst_parser"]
templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "furo"
html_title = "tmrca.cu"
html_static_path = ["_static"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#3182bd",
        "color-brand-content": "#3182bd",
    },
    "dark_css_variables": {
        "color-brand-primary": "#58a6ff",
        "color-brand-content": "#58a6ff",
    },
}
