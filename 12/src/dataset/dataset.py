import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from util.logger import logger


class Dataset:
    def __init__(self, config, dataset_info) -> None:
        self.file_name = dataset_info["input_file_path"]
        self.config = config

        self.usecols = self.config["usecols"]
        self.name = self.config["dataset"]
        self.lower = float(self.config["dataset_lower"])
        self.upper = float(self.config["dataset_upper"])
        self.train_batch_size = self.config["train_batch_size"]
        self.eval_batch_size = self.config["eval_batch_size"]
        self.random_state = self.config["dataset_seed"]
        self.is_save_data = self.config["is_save_data"]
        self.data_split = self.config["data_split"]
        self.positive_rating_threshold = self.config.get(
            "positive_rating_threshold"
        )
        self.source_format = self.config.get("source_format", "standard")
        self.split_group_name = self.config.get("split_group_name")
        self.min_user_groups = self.config.get("min_user_groups")
        self.user_id_name = self.config["user_id_name"]
        self.item_id_name = self.config["item_id_name"]
        self.user_idx_name = self.config["user_idx_name"]
        self.item_idx_name = self.config["item_idx_name"]
        self.neg_item_idx_name = self.config["neg_item_idx_name"]

        self.train_size, self.val_size, self.test_size = self.config[
            "train_val_test"
        ]

        self.orig_df = self._load_source_dataframe()
        self.df = self.get_filtered_df()

        self._log_filtered_dataframe_summary()
        self._split_train_val_test_by_config()
        self._filter_unseen_evaluation_entities()
        self.df = pd.concat(
            [self.train_df, self.val_df, self.test_df], ignore_index=True
        )
        self._encode_interaction_ids()
        self.set_basic_info()

        self.all_item_idx_set = set(range(self.item_num))
        self._prepare_negative_sampling()
        self._prepare_evaluation_indices()

        if self.is_save_data:
            self.save_train_valid_test_data()

    def _load_source_dataframe(self) -> pd.DataFrame:
        orig_df = pd.read_csv(self.file_name, usecols=self.usecols).reset_index(
            drop=True
        )
        self.orig_df_before_filtering_shape = orig_df.shape
        logger.debug(f"orig_df shape : {orig_df.shape}")

        if self.source_format == "online_retail":
            return self._prepare_online_retail_dataframe(orig_df)

        if self.positive_rating_threshold is not None:
            orig_df = orig_df[
                orig_df["rating"] >= self.positive_rating_threshold
            ].reset_index(drop=True)

        logger.debug("drop duplicate")
        orig_df = orig_df.drop_duplicates(
            subset=[self.user_id_name, self.item_id_name], keep="last"
        ).reset_index(drop=True)
        self.orig_df_after_filtering_shape = orig_df.shape
        logger.debug(f"orig_df shape : {orig_df.shape}")
        return orig_df

    def _prepare_online_retail_dataframe(
        self, orig_df: pd.DataFrame
    ) -> pd.DataFrame:
        non_product_codes = {"POST", "M", "C2", "DOT", "BANK CHARGES"}
        dataframe = orig_df[
            orig_df["CustomerID"].notna()
            & (orig_df["Quantity"] > 0)
            & (orig_df["UnitPrice"] > 0)
            & ~orig_df["InvoiceNo"].astype(str).str.upper().str.startswith("C")
            & ~orig_df["StockCode"].astype(str).isin(non_product_codes)
        ].copy()

        dataframe[self.user_id_name] = (
            dataframe["CustomerID"].astype("int64").astype(str)
        )
        dataframe[self.item_id_name] = (
            dataframe["StockCode"].astype(str).str.strip()
        )
        dataframe["invoice_id"] = dataframe["InvoiceNo"].astype(str).str.strip()
        dataframe["timestamp"] = pd.to_datetime(
            dataframe["InvoiceDate"], errors="coerce"
        )
        dataframe = dataframe.dropna(
            subset=[
                self.user_id_name,
                self.item_id_name,
                "invoice_id",
                "timestamp",
            ]
        )
        dataframe = (
            dataframe.drop_duplicates()
            .sort_values(["timestamp", "invoice_id", self.item_id_name])
            .drop_duplicates(
                subset=[self.user_id_name, self.item_id_name], keep="first"
            )
            .reset_index(drop=True)
        )
        dataframe = dataframe[
            [self.user_id_name, self.item_id_name, "invoice_id", "timestamp"]
        ]
        self.orig_df_after_filtering_shape = dataframe.shape
        logger.debug(f"orig_df shape : {dataframe.shape}")
        return dataframe

    def _log_filtered_dataframe_summary(self):
        logger.debug(f"user_num   : {self.df[self.user_id_name].nunique():,}")
        logger.debug(f"item_num   : {self.df[self.item_id_name].nunique():,}")
        logger.debug(f"df.head(5) : \n{self.df.head(5)}")
        logger.debug(f"df.tail(5) : \n{self.df.tail(5)}")

    def _encode_interaction_ids(self):
        user_idx, _ = pd.factorize(self.df[self.user_id_name], sort=True)
        item_idx, _ = pd.factorize(self.df[self.item_id_name], sort=True)

        self.user_id2user_idx_dict = dict(
            zip(self.df[self.user_id_name], user_idx)
        )
        self.item_id2item_idx_dict = dict(
            zip(self.df[self.item_id_name], item_idx)
        )
        self.user_idx2user_id_dict = dict(
            zip(user_idx, self.df[self.user_id_name])
        )
        self.item_idx2item_id_dict = dict(
            zip(item_idx, self.df[self.item_id_name])
        )

        self.df[self.user_idx_name] = self.df[self.user_id_name].map(
            self.user_id2user_idx_dict
        )
        self.df[self.item_idx_name] = self.df[self.item_id_name].map(
            self.item_id2item_idx_dict
        )

        for dataframe in (self.train_df, self.val_df, self.test_df):
            dataframe[self.user_idx_name] = dataframe[self.user_id_name].map(
                self.user_id2user_idx_dict
            )
            dataframe[self.item_idx_name] = dataframe[self.item_id_name].map(
                self.item_id2item_idx_dict
            )

    def _split_train_val_test_by_config(self):
        logger.debug("data split")
        if self.data_split == "sort":
            self.split_train_val_test_by_sort()
            return
        if self.data_split == "group_sort":
            self.split_train_val_test_by_group_sort()
            return
        if self.data_split == "no_sort":
            self.split_train_val_test()
            return
        raise ValueError(f"Unsupported data_split: {self.data_split}")

    def _filter_unseen_evaluation_entities(self):
        train_user_ids = set(self.train_df[self.user_id_name])
        train_item_ids = set(self.train_df[self.item_id_name])
        for split_name in ("val_df", "test_df"):
            dataframe = getattr(self, split_name)
            dataframe = dataframe[
                dataframe[self.user_id_name].isin(train_user_ids)
                & dataframe[self.item_id_name].isin(train_item_ids)
            ].reset_index(drop=True)
            if dataframe.empty:
                raise ValueError(f"{split_name} is empty after filtering.")
            setattr(self, split_name, dataframe)

    def _prepare_negative_sampling(self):
        logger.debug("start set_user_idx2neg_item_idx_set_dict")
        start_time = time.time()
        self.set_user_idx2neg_item_idx_set_dict()
        elapsed_time = time.time() - start_time
        self.set_user_idx2neg_item_idx_set_dict_time = elapsed_time
        logger.debug(
            f"end set_user_idx2neg_item_idx_set_dict: {elapsed_time:.2f} sec"
        )

    def _prepare_evaluation_indices(self):
        logger.debug("set_train_viewed_item_idx")
        self.set_train_viewed_item_idx()
        self.set_train_val_viewed_item_idx()

    def _build_user_item_index_dict(
        self, dataframe: pd.DataFrame
    ) -> dict[int, np.ndarray]:
        grouped_dict = {}
        for user_idx, group_df in dataframe.groupby(self.user_idx_name):
            grouped_dict[user_idx] = group_df[self.item_idx_name].values
        return grouped_dict

    def _build_eval_tensor_dataset(self, dataframe: pd.DataFrame):
        user_idx2item_idx_dict = self._build_user_item_index_dict(dataframe)
        user_idx_list = list(user_idx2item_idx_dict.keys())
        dataset = TensorDataset(torch.tensor(user_idx_list, dtype=torch.long))
        return dataset, user_idx2item_idx_dict

    def _build_sparse_interaction_matrix(self, matrix_type: str):
        matrix_cls = sp.coo_matrix if matrix_type == "coo" else sp.csr_matrix
        return matrix_cls(
            (
                np.ones(self.train_nnz, dtype=np.float32),
                (
                    self.train_df[self.user_idx_name].values,
                    self.train_df[self.item_idx_name].values,
                ),
            ),
            shape=(self.user_num, self.item_num),
        ).astype(np.float32)

    def _build_train_subset_dataframe(
        self, user_values, item_values
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.user_idx_name: user_values.values,
                self.item_idx_name: item_values.values,
            }
        )

    def _filter_group_by_interaction_count(
        self, dataframe: pd.DataFrame, group_key: str, target_key: str
    ):
        return (
            dataframe.groupby(group_key)
            .filter(
                lambda group: (
                    self.lower <= group[target_key].count() <= self.upper
                )
            )
            .reset_index(drop=True)
        )

    def _validate_sorted_split(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        train_groupby = train_df.groupby(self.user_id_name)
        val_groupby = val_df.groupby(self.user_id_name)
        test_groupby = test_df.groupby(self.user_id_name)
        assert (
            sum(
                train_groupby.max()["timestamp"]
                > val_groupby.min()["timestamp"]
            )
            == 0
        )
        assert (
            sum(
                val_groupby.max()["timestamp"] > test_groupby.min()["timestamp"]
            )
            == 0
        )

    def set_basic_info(self):
        self.user_num = self.df[self.user_idx_name].nunique()
        self.item_num = self.df[self.item_idx_name].nunique()
        self.nnz = self.df.shape[0]

        self.density = self.nnz / self.user_num / self.item_num
        self.sparsity = 1 - self.density

        self.user_interaction_mean = (
            self.df.groupby(self.user_idx_name)
            .count()[self.item_idx_name]
            .mean()
        )
        self.item_interaction_mean = (
            self.df.groupby(self.item_idx_name)
            .count()[self.user_idx_name]
            .mean()
        )

        self.train_user_num = self.train_df[self.user_idx_name].nunique()
        self.train_item_num = self.train_df[self.item_idx_name].nunique()
        self.train_nnz = self.train_df.shape[0]

        self.val_user_num = self.val_df[self.user_idx_name].nunique()
        self.val_item_num = self.val_df[self.item_idx_name].nunique()
        self.val_nnz = self.val_df.shape[0]

        self.test_user_num = self.test_df[self.user_idx_name].nunique()
        self.test_item_num = self.test_df[self.item_idx_name].nunique()
        self.test_nnz = self.test_df.shape[0]

    def get_filtered_df(self) -> pd.DataFrame:
        dataframe = self.orig_df.copy(deep=True)

        logger.info("dataset filter loop start")
        while True:
            logger.info("inside filtering loop")
            start_time = time.time()

            logger.debug(f"  {self.user_id_name} groupby")
            dataframe = self._filter_group_by_interaction_count(
                dataframe,
                self.user_id_name,
                self.item_id_name,
            )

            logger.debug(f"  {self.item_id_name} groupby")
            dataframe = self._filter_group_by_interaction_count(
                dataframe,
                self.item_id_name,
                self.user_id_name,
            )

            if self.split_group_name and self.min_user_groups:
                dataframe = (
                    dataframe.groupby(self.user_id_name)
                    .filter(
                        lambda group: (
                            group[self.split_group_name].nunique()
                            >= self.min_user_groups
                        )
                    )
                    .reset_index(drop=True)
                )

            if dataframe.empty:
                raise ValueError("df data length is 0.")

            item_groupby = dataframe.groupby(self.item_id_name).count()
            user_groupby = dataframe.groupby(self.user_id_name).count()

            logger.debug(
                f"  groupby_item_id max : {item_groupby.max().iloc[0]}"
            )
            logger.debug(
                f"  groupby_item_id min : {item_groupby.min().iloc[0]}"
            )
            logger.debug(
                f"  groupby_user_id max : {user_groupby.max().iloc[0]}"
            )
            logger.debug(
                f"  groupby_user_id min : {user_groupby.min().iloc[0]}"
            )

            if (
                item_groupby.max().iloc[0] <= self.upper
                and item_groupby.min().iloc[0] >= self.lower
                and user_groupby.max().iloc[0] <= self.upper
                and user_groupby.min().iloc[0] >= self.lower
            ):
                break

            elapsed_time = time.time() - start_time
            logger.info(f"elapsed_time : {elapsed_time:.3f}")

        logger.info("dataset filter loop end")
        return dataframe

    def split_train_val_test(self):
        row_indices = np.arange(len(self.df))
        train_indices, test_indices = train_test_split(
            row_indices,
            test_size=self.test_size,
            stratify=self.df[self.user_id_name],
            random_state=self.random_state,
        )

        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=self.val_size / (1 - self.test_size),
            stratify=self.df.iloc[train_indices][self.user_id_name],
            random_state=self.random_state,
        )

        self.train_df = (
            self.df.iloc[train_indices].copy().reset_index(drop=True)
        )
        self.val_df = self.df.iloc[val_indices].copy().reset_index(drop=True)
        self.test_df = self.df.iloc[test_indices].copy().reset_index(drop=True)

    def split_train_val_test_by_sort(self):
        train_parts = []
        val_parts = []
        test_parts = []

        sorted_df = self.df.sort_values([self.user_id_name, "timestamp"])
        for _, group in sorted_df.groupby(self.user_id_name, sort=False):
            item_count = len(group)
            train_end = min(
                max(1, int(item_count * self.train_size)), item_count - 2
            )
            val_end = min(
                max(
                    train_end + 1,
                    int(item_count * (self.train_size + self.val_size)),
                ),
                item_count - 1,
            )
            train_parts.append(group.iloc[:train_end])
            val_parts.append(group.iloc[train_end:val_end])
            test_parts.append(group.iloc[val_end:])

        train_df = pd.concat(train_parts)
        val_df = pd.concat(val_parts)
        test_df = pd.concat(test_parts)

        self._validate_sorted_split(train_df, val_df, test_df)

        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

    def split_train_val_test_by_group_sort(self):
        if not self.split_group_name:
            raise ValueError("split_group_name is required for group_sort.")

        train_parts = []
        val_parts = []
        test_parts = []
        sorted_df = self.df.sort_values(
            [
                self.user_id_name,
                "timestamp",
                self.split_group_name,
                self.item_id_name,
            ]
        )

        for _, group in sorted_df.groupby(self.user_id_name, sort=False):
            groups = (
                group.groupby(self.split_group_name, as_index=False)[
                    "timestamp"
                ]
                .min()
                .sort_values(["timestamp", self.split_group_name])
            )
            group_count = len(groups)
            train_end = min(
                max(1, int(group_count * self.train_size)), group_count - 2
            )
            val_end = min(
                max(
                    train_end + 1,
                    int(group_count * (self.train_size + self.val_size)),
                ),
                group_count - 1,
            )
            train_group_ids = set(
                groups.iloc[:train_end][self.split_group_name]
            )
            val_group_ids = set(
                groups.iloc[train_end:val_end][self.split_group_name]
            )
            test_group_ids = set(groups.iloc[val_end:][self.split_group_name])
            train_parts.append(
                group[group[self.split_group_name].isin(train_group_ids)]
            )
            val_parts.append(
                group[group[self.split_group_name].isin(val_group_ids)]
            )
            test_parts.append(
                group[group[self.split_group_name].isin(test_group_ids)]
            )

        train_df = pd.concat(train_parts)
        val_df = pd.concat(val_parts)
        test_df = pd.concat(test_parts)
        self._validate_sorted_split(train_df, val_df, test_df)

        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

    def set_user_idx2neg_item_idx_set_dict(self):
        self.user_idx2neg_item_idx_set_dict = {}

        for user_idx, group_df in self.train_df.groupby(self.user_idx_name):
            positive_item_set = set(group_df[self.item_idx_name].values)
            self.user_idx2neg_item_idx_set_dict[user_idx] = (
                self.all_item_idx_set - positive_item_set
            )

    def set_user_idx2pos_item_idx_set_dict(self):
        self.user_idx2pos_item_idx_set_dict = {}
        for user_idx, group_df in self.train_df.groupby(self.user_idx_name):
            self.user_idx2pos_item_idx_set_dict[user_idx] = set(
                group_df[self.item_idx_name].values
            )

    def get_user_idx2neg_item_idx_set_dict(self, user_idx_name):
        return self.all_item_idx_set - self.user_idx2pos_item_idx_set_dict.get(
            user_idx_name, set()
        )

    def get_dataloader(self):
        train_ds = TensorDataset(
            torch.tensor(
                self.train_df[self.user_idx_name].values, dtype=torch.long
            ),
            torch.tensor(
                self.train_df[self.item_idx_name].values, dtype=torch.long
            ),
        )

        val_ds, self.val_user_idx2item_idx_dict = (
            self._build_eval_tensor_dataset(self.val_df)
        )
        test_ds, self.test_user_idx2item_idx_dict = (
            self._build_eval_tensor_dataset(self.test_df)
        )

        train_dl = DataLoader(
            train_ds, batch_size=self.train_batch_size, shuffle=True
        )
        val_dl = DataLoader(
            val_ds, batch_size=self.eval_batch_size, shuffle=False
        )
        test_dl = DataLoader(
            test_ds, batch_size=self.eval_batch_size, shuffle=False
        )
        return train_dl, val_dl, test_dl

    def get_train_coo_matrix(self):
        return self._build_sparse_interaction_matrix("coo")

    def get_train_csr_matrix(self):
        return self._build_sparse_interaction_matrix("csr")

    def set_train_viewed_item_idx(self):
        self.train_user_idx2viewed_idx_dict = self._build_user_item_index_dict(
            self.train_df
        )

    def set_train_val_viewed_item_idx(self):
        train_val_df = pd.concat([self.train_df, self.val_df]).reset_index(
            drop=True
        )
        self.train_val_user_idx2viewed_idx_dict = (
            self._build_user_item_index_dict(train_val_df)
        )

    def save_train_valid_test_data(self):
        output_dir = (
            Path(self.config.get("checkpoint_dir_name", "./checkpoint"))
            / "dataset_cache"
            / self.name
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self.train_df.to_csv(output_dir / "train.csv", index=False)
        self.val_df.to_csv(output_dir / "val.csv", index=False)
        self.test_df.to_csv(output_dir / "test.csv", index=False)
