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

    subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            "python3",
            "--display-name",
            "python3",
        ],
        check=True,
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
