"""Test-only loader for notebook setup and definitions, without launching the study."""

from __future__ import annotations

import json
from pathlib import Path


def setup_cells(document):
    """Read the setup portion preceding the notebook's ordinary study sequence."""
    result = []
    for cell in document["cells"]:
        if cell["id"] == "study-start":
            return result
        result.append(cell)
    raise ValueError("Notebook study section is missing")


_notebook_path = Path(__file__).resolve().parents[1] / "chexpert_ssl.ipynb"
_notebook_document = json.loads(_notebook_path.read_text(encoding="utf-8"))
for _cell in setup_cells(_notebook_document):
    if _cell["cell_type"] == "code":
        exec(  # noqa: S102 -- trusted local setup/definition cells only
            compile("".join(_cell["source"]), f"{_notebook_path}#{_cell['id']}", "exec"), globals()
        )
