from dataclasses import asdict, dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_file_path: str
    config_file_path: str
    default_columns: list[str]

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config_file_path: str
    module_file_path: str
    class_name: str

    def to_dict(self):
        return asdict(self)


DATASET_SPECS = {
    "online-retail": DatasetSpec(
        name="online-retail",
        input_file_path=str(
            BASE_DIR.parents[1] / "02/data/online_retail.csv.gz"
        ),
        config_file_path=str(BASE_DIR / "config/dataset/online-retail.yaml"),
        default_columns=[
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "CustomerID",
        ],
    ),
}


MODEL_SPECS = {
    "BPR": ModelSpec(
        name="BPR",
        config_file_path=str(BASE_DIR / "config/model/BPR.yaml"),
        module_file_path=str(BASE_DIR / "models/BPR.py"),
        class_name="BPR",
    ),
    "NGCF": ModelSpec(
        name="NGCF",
        config_file_path=str(BASE_DIR / "config/model/NGCF.yaml"),
        module_file_path=str(BASE_DIR / "models/NGCF.py"),
        class_name="NGCF",
    ),
    "LightGCN": ModelSpec(
        name="LightGCN",
        config_file_path=str(BASE_DIR / "config/model/LightGCN.yaml"),
        module_file_path=str(BASE_DIR / "models/LightGCN.py"),
        class_name="LightGCN",
    ),
}


def get_dataset_info(dataset_name: str) -> dict:
    try:
        return DATASET_SPECS[dataset_name].to_dict()
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from exc


def get_model_info(model_name: str) -> dict:
    try:
        return MODEL_SPECS[model_name].to_dict()
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_name}") from exc
