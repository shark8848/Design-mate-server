# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

#project = 'Apoco Intelligent Analysis'
#copyright = '2023, SHARK8848'
#author = 'SHARK8848'
#release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#extensions = []

#templates_path = ['_templates']
#exclude_patterns = []

#language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
#html_static_path = ['_static']

import os
import sys

# Set the path for the Flask app
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = 'Apoco Intelligent Analysis'
copyright = '2023, SHARK8848'
author = 'SHARK8848'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
'''
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]
'''
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax'
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_theme_options = {
    "style_nav_header_background": "#343131",
}

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "Apoco Intelligent Analysis Appdoc"

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# -- Options for autodoc extension -------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "private-members": True,
    "special-members": "__init__",
    "show-inheritance": True,
}

