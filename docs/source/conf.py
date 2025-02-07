# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SIMOD'
copyright = '2025, UT Information Systems Research Group'
author = 'UT Information Systems Research Group'
release = '5.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

# Get the absolute path of the project's root directory
sys.path.insert(0, os.path.abspath("../../src"))  # Adjust if necessary

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx"
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.9", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

templates_path = ['_templates']
exclude_patterns = []
autodoc_class_attributes = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Automatically generate summaries
autosummary_generate = True
