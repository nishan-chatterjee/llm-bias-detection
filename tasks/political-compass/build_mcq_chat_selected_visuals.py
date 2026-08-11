#!/usr/bin/env python3
"""Run the selected MCQ/chat visual generation notebook.

This keeps the paper-facing selected visuals reproducible without manually
stepping through the notebook UI. The notebook remains the readable source for
the plotting code and captions; this script executes the data/plot cells that
write files under ``selected-visuals/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "mcq-chat-analysis-selected-visuals.ipynb"


def main() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = str(Path(conda_prefix) / "lib")
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if conda_lib not in current_ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = conda_lib + (":" + current_ld if current_ld else "")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-selected")

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {}

    # Cells 2-5 contain imports/constants, scoring helpers, data loading, and
    # selected visual generation. Cell 7 is optional Playwright HTML export and
    # should be run interactively when browser dependencies are available.
    for cell_index in [1, 2, 3, 4]:
        source = "".join(notebook["cells"][cell_index]["source"])
        print(f"Executing notebook cell {cell_index + 1}...", flush=True)
        exec(compile(source, f"{NOTEBOOK}:cell-{cell_index + 1}", "exec"), namespace)


if __name__ == "__main__":
    main()
