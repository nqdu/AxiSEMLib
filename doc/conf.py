"""Sphinx configuration for the AxiSEMLib guides."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axisemlib import __version__  # noqa: E402


project = "AxiSEMLib"
release = __version__
extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
myst_heading_anchors = 3
master_doc = "index"
exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
html_title = "AxiSEMLib"
html_baseurl = "https://nqdu.github.io/AxiSEMLib/"
