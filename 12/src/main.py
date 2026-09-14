import argparse
import datetime
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from dataset.dataset import Dataset
from info import BASE_DIR, get_dataset_info, get_model_info
from loguru import logger
from util.logger import init_logger
from util.seed import set_seed
from util.util import JST, convert_to_num

COMMON_CONFIG_PATH = BASE_DIR / "config/common.yaml"
RESULT_DIR = BASE_DIR / "result"
DEFAULT_DATASET = "online-retail"
SUPPORTED_MODELS = ("BPR", "NGCF", "LightGCN")
SKIP_SERIALIZATION = object()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=[DEFAULT_DATASET],
    )
    parser.add_argument(
        "-m", "--model", type=str, default="LightGCN", choices=SUPPORTED_MODELS
    )

    args, unknown_args = parser.parse_known_args()
    return args, parse_unknown_args(unknown_args)


def parse_unknown_args(unknown_args):
    unknown_args_dict = {}
    for unknown_arg in unknown_args:
        if unknown_arg.startswith("--"):
            key, value = unknown_arg[2:].split("=", maxsplit=1)
        elif unknown_arg.startswith("-"):
            key, value = unknown_arg[1:].split("=", maxsplit=1)
        else:
            raise ValueError(f"Unsupported argument format: {unknown_arg}")

        unknown_args_dict[key] = convert_to_num(value)
    return unknown_args_dict


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f.read())


def build_config(args, unknown_args_dict, dataset_info, model_info):
    config = load_yaml(COMMON_CONFIG_PATH)
    dataset_config = load_yaml(Path(dataset_info["config_file_path"]))
    model_config = load_yaml(Path(model_info["config_file_path"]))

    config.update(dataset_config)
    config.update(model_config)
    config.update(unknown_args_dict)
    config.update(vars(args))

    if "dataset_seed" not in config:
        config["dataset_seed"] = config["seed"]

    return config


def load_model_class(model_info):
    module_path = Path(model_info["module_file_path"])
    module_name = f"runtime_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, model_info["class_name"])


def create_model(config, dataset, model_info):
    model_class = load_model_class(model_info)
    device = torch.device(
        f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu"
    )
    return model_class(config, dataset).to(device)


def to_json_serializable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, torch.Tensor):
        return SKIP_SERIALIZATION
    if isinstance(value, dict):
        serialized_dict = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                return SKIP_SERIALIZATION
            serialized_item = to_json_serializable(item)
            if serialized_item is SKIP_SERIALIZATION:
                return SKIP_SERIALIZATION
            serialized_dict[str(key)] = serialized_item
        return serialized_dict
    if isinstance(value, (list, tuple)):
        if len(value) > 100:
            return SKIP_SERIALIZATION
        serialized_items = []
        for item in value:
            serialized_item = to_json_serializable(item)
            if serialized_item is SKIP_SERIALIZATION:
                return SKIP_SERIALIZATION
            serialized_items.append(serialized_item)
        return serialized_items
    return SKIP_SERIALIZATION


def collect_serializable_model_attributes(model):
    model_attr_dict = {}
    for key, value in model.__dict__.items():
        if not isinstance(key, str):
            continue
        if key.startswith("_"):
            continue
        serialized_value = to_json_serializable(value)
        if serialized_value is SKIP_SERIALIZATION:
            continue
        model_attr_dict[key] = serialized_value
    return model_attr_dict


def collect_dataset_summary(dataset):
    return {
        "user_interaction_mean": dataset.user_interaction_mean,
        "item_interaction_mean": dataset.item_interaction_mean,
        "sparsity": dataset.sparsity,
        "density": dataset.density,
        "user_num": dataset.user_num,
        "item_num": dataset.item_num,
        "nnz": dataset.nnz,
        "train_nnz": dataset.train_nnz,
        "val_nnz": dataset.val_nnz,
        "test_nnz": dataset.test_nnz,
        "orig_df_before_filtering_shape": dataset.orig_df_before_filtering_shape,
        "orig_df_after_filtering_shape": dataset.orig_df_after_filtering_shape,
        "set_user_idx2neg_item_idx_set_dict_time": dataset.set_user_idx2neg_item_idx_set_dict_time,
    }


def save_result(config, model, dataset, elapsed_time_minutes):
    timestamp = datetime.datetime.now(tz=JST)
    yyyymmddhhmmss = timestamp.strftime("%Y%m%d%H%M%S")
    yyyymmdd_dir = RESULT_DIR / timestamp.strftime("%Y%m%d")
    yyyymmdd_dir.mkdir(parents=True, exist_ok=True)

    result_file_name_postfix = str(config["result_file_name_postfix"])
    file_name = f"{yyyymmddhhmmss}_{config['dataset']}_{config['model']}_{result_file_name_postfix}.json"
    result_path = yyyymmdd_dir / file_name

    ret_dict = {}
    ret_dict.update(collect_serializable_model_attributes(model))
    ret_dict.update(collect_dataset_summary(dataset))
    ret_dict.update({"elapsed_time_minutes": elapsed_time_minutes})

    temporary_result_path = result_path.with_suffix(".json.tmp")
    with temporary_result_path.open(mode="w") as f:
        json.dump(ret_dict, f, ensure_ascii=False, indent=2)
    temporary_result_path.replace(result_path)

    logger.info(f"result file name: {file_name}")
    logger.info(f"successfully saved    : {result_path}")
    logger.info(
        f"successfully finished : total elapsed_time: {elapsed_time_minutes:.2f} min"
    )


def main(
    args: argparse.Namespace, unknown_args_dict: dict
) -> tuple[dict, dict]:
    total_start_time = time.time()

    dataset_info = get_dataset_info(args.dataset)
    model_info = get_model_info(args.model)
    config = build_config(args, unknown_args_dict, dataset_info, model_info)

    init_logger(config)
    logger.info(f"config : {json.dumps(config, indent=2)}")

    logger.info("1. set seed")
    set_seed(seed=config["seed"])

    logger.info("2. set dataset")
    dataset = Dataset(config, dataset_info)
    logger.info(f"user_num  : {dataset.user_num:,}")
    logger.info(f"item_num  : {dataset.item_num:,}")
    logger.info(f"inter_num : {dataset.nnz:,}")

    logger.info("3. set model")
    model_start_time = time.time()
    model = create_model(config, dataset, model_info)
    logger.debug(
        f"end model initialization: {time.time() - model_start_time:.2f} sec"
    )
    logger.info(model)

    logger.info("4. training and evaluate")
    model.fit()

    logger.info("5. save result")
    elapsed_time_minutes = (time.time() - total_start_time) / 60
    save_result(config, model, dataset, elapsed_time_minutes)
    return model.best_valid_result, model.test_result


if __name__ == "__main__":
    parsed_args, parsed_unknown_args = parse_args()
    main(parsed_args, parsed_unknown_args)
