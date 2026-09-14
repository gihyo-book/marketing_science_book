from pathlib import Path

from IPython.display import HTML, display

DATAFRAME_STYLE = """
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
        padding: 15px;
    }

    .dataframe tbody tr th {
        vertical-align: top;
        padding: 15px;
    }

    .dataframe tbody tr:hover {
        background-color: #ffff99;
    }

    .dataframe {
        background-color: white;
        color: black;
        font-size: 17px;
    }
</style>
"""


def apply_dataframe_style():
    display(HTML(DATAFRAME_STYLE))


def get_data_path(filename):
    candidates = []
    for path in [Path.cwd(), *Path.cwd().parents]:
        data_dirs = [path / "data", *path.glob("*/data")]
        candidates.extend(data_dir / filename for data_dir in data_dirs)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    lower_filename = filename.lower()
    for data_dir in {
        candidate.parent
        for candidate in candidates
        if candidate.parent.exists()
    }:
        for candidate in data_dir.iterdir():
            if candidate.name.lower() == lower_filename:
                return candidate

    raise FileNotFoundError(f"Data file not found: {filename}")
