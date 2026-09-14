from pathlib import Path

import papermill as pm
import pytest

CH11_NOTEBOOKS = [
    "11/11_main.ipynb",
]


@pytest.mark.parametrize("notebook_relpath", CH11_NOTEBOOKS)
def test_ch11_notebook_execute(
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
