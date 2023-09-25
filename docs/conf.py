# Name: Paul Setinek
# Github: acse-pms122

import sys
import os


# Set path to parent directory
sys.path.insert(0, os.path.abspath(os.sep.join((os.curdir, "../"))))

project = "Subsea-NeRF"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.video"
]

bibtex_bibfiles = ["references.bib"]
html_theme = "sphinx_rtd_theme"
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build"]
autoclass_content = "both"
author = "Paul Setinek"
