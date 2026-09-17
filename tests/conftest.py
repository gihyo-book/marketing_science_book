import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _configure_notebook_test_env(tmp_path_factory):
    # Avoid GUI backend dependency during CI/local test runs.
    os.environ.setdefault("MPLBACKEND", "Agg")

    ipython_dir = tmp_path_factory.mktemp("ipython")
    os.environ.setdefault("IPYTHONDIR", str(ipython_dir))
    os.environ.setdefault(
        "JUPYTER_DATA_DIR", str(tmp_path_factory.mktemp("jupyter"))
    )
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            "marketing-science-src",
            "--display-name",
            "Marketing Science (Python 3.14)",
        ],
        check=True,
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
