"""
Configuration file for the Sphinx documentation builder.
"""
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "omfiles"
copyright = "2025, Open-Meteo"
author = "terraputix"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", "https://docs.python.org/3/objects.inv"),
    "xarray": ("https://docs.xarray.dev/en/stable/", "https://docs.xarray.dev/en/stable/objects.inv"),
    "numpy": ("https://numpy.org/doc/stable/", "https://numpy.org/doc/stable/objects.inv"),
    "fsspec": (
        "https://filesystem-spec.readthedocs.io/en/latest/",
        "https://filesystem-spec.readthedocs.io/en/latest/objects.inv",
    ),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
