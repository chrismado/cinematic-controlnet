from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "cinematic-controlnet"
author = "Chris Matteau"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]
templates_path = ["_templates"]
exclude_patterns = ["_build"]
autodoc_mock_imports = [
    "accelerate",
    "cv2",
    "diffusers",
    "huggingface_hub",
    "numpy",
    "torch",
    "transformers",
    "wandb",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable/", {}),
    "torch": ("https://pytorch.org/docs/stable/", {}),
}
html_theme = "alabaster"
