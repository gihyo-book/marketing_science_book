import os
from pathlib import Path

import papermill as pm
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    REPO_ROOT / "main_314.ipynb",
    *sorted(REPO_ROOT.glob("[0-1][0-9]/*.ipynb")),
]
CHAPTER = os.environ.get("NOTEBOOK_CHAPTER")
MAIN_NOTEBOOKS = [
    str(path.relative_to(REPO_ROOT))
    for path in NOTEBOOKS
    if CHAPTER is None
    or path.parent.name == CHAPTER
    or (CHAPTER == "00" and path.name == "main_314.ipynb")
]


@pytest.mark.parametrize("notebook_relpath", MAIN_NOTEBOOKS)
def test_main_notebooks_execute(
    notebook_relpath: str, repo_root, tmp_path: Path
):
    input_notebook = repo_root / notebook_relpath
    output_notebook = tmp_path / f"{input_notebook.stem}_executed.ipynb"
    notebook_cwd = input_notebook.parent

    assert input_notebook.exists(), f"Notebook not found: {input_notebook}"

    pm.execute_notebook(
        input_path=str(input_notebook),
        output_path=str(output_notebook),
        kernel_name="marketing-science-src",
        parameters={"CI_SMOKE_TEST": True}
        if notebook_relpath == "12/12_main.ipynb"
        else {},
        cwd=str(notebook_cwd),
        progress_bar=False,
        log_output=False,
    )
