from pathlib import Path

import papermill as pm
import pytest

MAIN_NOTEBOOKS = [
    "02/02_main.ipynb",
    "03/03_main.ipynb",
    "04/04_main.ipynb",
    "05/05_main.ipynb",
    "06/06_main.ipynb",
    "07/07_main.ipynb",
    "08/08_main.ipynb",
    "09/09_main.ipynb",
    "10/10_main.ipynb",
    "12/12_main.ipynb",
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
        kernel_name="python3",
        cwd=str(notebook_cwd),
        progress_bar=False,
        log_output=False,
    )
