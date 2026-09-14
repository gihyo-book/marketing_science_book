import os
import time

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from util.util import create_dir, get_yyyymmddhhmmss, is_train_stop


class Evaluator:
    def __init__(
        self,
        val_user_idx,
        score_matrix,
        user_idx2viewed_idx_dict,
        val_user_idx2item_idx_dict,
        topk,
        gpu=0,
    ) -> None:
        self.val_user_idx = val_user_idx
        self.score_matrix = score_matrix
        self.user_idx2viewed_idx_dict = user_idx2viewed_idx_dict
        self.val_user_idx2item_idx_dict = val_user_idx2item_idx_dict
        self.topk = topk
        self.device = torch.device(
            f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        )

        self._mask_viewed_items()
        self.topk_idx_matrix = torch.topk(
            self.score_matrix, self.topk, dim=1
        ).indices
        self.val_binary_matrix = self._build_validation_binary_matrix()
        self.topk_binary_matrix = torch.gather(
            self.val_binary_matrix, dim=1, index=self.topk_idx_matrix
        )
        self.topk_ndcg_score_matrix = self._build_ndcg_score_matrix()

    def _mask_viewed_items(self):
        viewed_row_idx_list = []
        viewed_col_idx_list = []
        for row_idx, user_idx in enumerate(self.val_user_idx):
            viewed_item_idx = self.user_idx2viewed_idx_dict[user_idx.item()]
            viewed_row_idx_list.extend([row_idx] * len(viewed_item_idx))
            viewed_col_idx_list.extend(viewed_item_idx)

        viewed_idx_tuple = (viewed_row_idx_list, viewed_col_idx_list)
        self.score_matrix[viewed_idx_tuple] = -np.inf

    def _build_validation_binary_matrix(self):
        val_binary_matrix = torch.zeros(
            self.score_matrix.shape, device=self.device
        )
        val_matrix_row_idx_list = []
        val_matrix_col_idx_list = []

        for row_idx, user_idx in enumerate(self.val_user_idx):
            val_item_idx = self.val_user_idx2item_idx_dict[user_idx.item()]
            val_matrix_row_idx_list.extend([row_idx] * len(val_item_idx))
            val_matrix_col_idx_list.extend(val_item_idx)

        val_binary_matrix[
            (val_matrix_row_idx_list, val_matrix_col_idx_list)
        ] = 1
        return val_binary_matrix

    def _build_ndcg_score_matrix(self):
        ndcg_score_list = []
        for user_idx in self.val_user_idx:
            val_item_idx = self.val_user_idx2item_idx_dict[user_idx.item()]
            valid_item_count = min(self.topk, len(val_item_idx))
            ndcg_score_list.append(
                [
                    1 / np.log2(2 + rank) if rank < valid_item_count else 0
                    for rank in range(self.topk)
                ]
            )
        return torch.tensor(
            ndcg_score_list, dtype=torch.float32, device=self.device
        )

    def get_recall(self):
        return self.topk_binary_matrix.sum(dim=1) / self.val_binary_matrix.sum(
            dim=1
        )

    def get_precision(self):
        return self.topk_binary_matrix.sum(dim=1) / self.topk

    def get_ndcg(self):
        discount = torch.tensor(
            [
                [1 / np.log2(2 + rank) for rank in range(self.topk)]
                for _ in range(self.topk_binary_matrix.shape[0])
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return torch.mul(self.topk_binary_matrix, discount).sum(
            dim=1
        ) / self.topk_ndcg_score_matrix.sum(dim=1)

    def get_mrr(self):
        rank_matrix = torch.tensor(
            [
                [
                    rank
                    for rank in range(self.topk_binary_matrix.shape[1], 0, -1)
                ]
                for _ in range(self.topk_binary_matrix.shape[0])
            ],
            dtype=torch.long,
            device=self.device,
        )
        rank_matrix = torch.mul(rank_matrix, self.topk_binary_matrix)
        rank_matrix = torch.cat(
            [
                torch.zeros(
                    self.topk_binary_matrix.shape[0], 1, device=self.device
                ),
                rank_matrix,
            ],
            dim=1,
        )
        _, rank_indices = torch.topk(rank_matrix, k=1, dim=1)
        rank_indices = rank_indices.to(torch.double)
        return torch.reciprocal(
            torch.where(rank_indices == 0, float("inf"), rank_indices)
        )

    def get_map(self):
        cumulative_hits = torch.mul(
            self.topk_binary_matrix,
            torch.cumsum(self.topk_binary_matrix, dim=1),
        )
        reciprocal_ranks = torch.reciprocal(
            torch.tensor(
                [
                    [i for i in range(1, self.topk_binary_matrix.shape[1] + 1)]
                    for _ in range(self.topk_binary_matrix.shape[0])
                ],
                dtype=torch.float32,
                device=self.device,
            )
        )
        valid_item_counts = (
            self.val_binary_matrix.sum(dim=1).reshape(-1, 1).to(torch.long)
        )
        topk_counts = torch.full(
            (self.val_binary_matrix.shape[0], 1),
            self.topk,
            dtype=torch.long,
            device=self.device,
        )
        denominator, _ = torch.min(
            torch.cat([valid_item_counts, topk_counts], dim=1), dim=1
        )
        return torch.mul(cumulative_hits, reciprocal_ranks).sum(
            dim=1
        ) / torch.where(denominator == 0, 1, denominator)


class Trainer:
    def __init__(self, config, model, dataset) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config
        self.epochs = self.config["epochs"]
        self.topk = self.config["topk"]
        self.seed = self.config["seed"]
        self.dataset_seed = self.config["dataset_seed"]
        self.show_progress = self.config.get("show_progress", True)

        self.checkpoint_dir = self.config["checkpoint_dir_name"]
        create_dir(self.checkpoint_dir)

        self.checkpoint_model_file, self.checkpoint_model_file_2 = (
            self._build_checkpoint_paths()
        )
        self.gpu = self.config["gpu"]
        self.device = torch.device(
            f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu"
        )

    def _build_checkpoint_paths(self):
        seed_suffix = f"_seed_{self.seed}_dataset_seed_{self.dataset_seed}"
        checkpoint_model_file = os.path.join(
            self.checkpoint_dir,
            f"{get_yyyymmddhhmmss()}_{self.config['model']}_{self.dataset.name}{seed_suffix}.pt",
        )
        checkpoint_model_file_2 = os.path.join(
            self.checkpoint_dir,
            f"{self.dataset.name}-seed-{self.seed}-dataset-seed-{self.dataset_seed}.pt",
        )
        return checkpoint_model_file, checkpoint_model_file_2

    def _initialize_training_state(self):
        self.best_valid_metric = 0
        self.best_valid_epoch = -1
        self.best_valid_recall = 0
        self.best_valid_precision = 0
        self.best_valid_ndcg = 0
        self.best_valid_mrr = 0
        self.best_valid_map = 0

    def _sample_negative_items(self, user_idx_batch, pos_item_idx_batch):
        user_idx_array = np.array(user_idx_batch)
        pos_item_idx_array = np.array(pos_item_idx_batch)
        pending_indices = list(range(len(user_idx_array)))

        ret_user_idx_list = []
        ret_pos_item_idx_list = []
        ret_neg_item_idx_list = []

        while pending_indices:
            temp_negative_item_idx = np.random.choice(
                self.dataset.item_num, size=len(pending_indices)
            )
            next_pending_indices = []

            for batch_idx, user_idx, pos_item_idx, neg_item_idx in zip(
                pending_indices,
                user_idx_array[pending_indices],
                pos_item_idx_array[pending_indices],
                temp_negative_item_idx,
            ):
                if (
                    neg_item_idx
                    in self.dataset.user_idx2neg_item_idx_set_dict[user_idx]
                ):
                    ret_user_idx_list.append(user_idx)
                    ret_pos_item_idx_list.append(pos_item_idx)
                    ret_neg_item_idx_list.append(neg_item_idx)
                else:
                    next_pending_indices.append(batch_idx)

            pending_indices = next_pending_indices

        return ret_user_idx_list, ret_pos_item_idx_list, ret_neg_item_idx_list

    def _build_training_batch(self, user_idx_batch, pos_item_idx_batch):
        ret_user_idx_list, ret_pos_item_idx_list, ret_neg_item_idx_list = (
            self._sample_negative_items(
                user_idx_batch,
                pos_item_idx_batch,
            )
        )
        return {
            "user_idx": torch.tensor(
                ret_user_idx_list, dtype=torch.long, device=self.device
            ),
            "item_idx": torch.tensor(
                ret_pos_item_idx_list, dtype=torch.long, device=self.device
            ),
            "neg_item_idx": torch.tensor(
                ret_neg_item_idx_list, dtype=torch.long, device=self.device
            ),
        }

    def _run_train_batch(self, user_idx_batch, pos_item_idx_batch):
        batch = self._build_training_batch(user_idx_batch, pos_item_idx_batch)
        self.model.optimizer.zero_grad()
        loss = self.model.compute_training_loss(batch)
        loss.backward()
        self.model.optimizer.step()

    def _run_train_epoch(self, epoch):
        self.model.train()
        train_dl = tqdm(
            self.train_dl,
            total=len(self.train_dl),
            ncols=100,
            desc=f"\033[31mTrain {epoch:>4}\033[0m",
            disable=not self.show_progress,
        )

        start_time = time.time()
        for user_idx_batch, pos_item_idx_batch in train_dl:
            self._run_train_batch(user_idx_batch, pos_item_idx_batch)
        return time.time() - start_time

    def _initialize_metric_sums(self):
        return {"recall": 0, "precision": 0, "ndcg": 0, "mrr": 0, "map": 0}

    def _accumulate_metric_sums(self, metric_sums, evaluator, user_count):
        metric_sums["recall"] += (
            evaluator.get_recall().sum().item() / user_count
        )
        metric_sums["precision"] += (
            evaluator.get_precision().sum().item() / user_count
        )
        metric_sums["ndcg"] += evaluator.get_ndcg().sum().item() / user_count
        metric_sums["mrr"] += evaluator.get_mrr().sum().item() / user_count
        metric_sums["map"] += evaluator.get_map().sum().item() / user_count

    def _evaluate_split(
        self,
        dataloader,
        viewed_idx_dict,
        target_idx_dict,
        user_count,
        desc=None,
    ):
        metric_sums = self._initialize_metric_sums()
        all_user_embeddings, all_item_embeddings = self.model()
        iterator = dataloader
        if desc is not None:
            iterator = tqdm(
                dataloader,
                total=len(dataloader),
                ncols=100,
                desc=desc,
                disable=not self.show_progress,
            )

        for user_idx_batch in iterator:
            user_indices = user_idx_batch[0].to(self.device)
            score_matrix = torch.matmul(
                all_user_embeddings[user_indices],
                all_item_embeddings.transpose(0, 1),
            )
            evaluator = Evaluator(
                user_idx_batch[0],
                score_matrix,
                viewed_idx_dict,
                target_idx_dict,
                self.topk,
                self.gpu,
            )

            self._accumulate_metric_sums(metric_sums, evaluator, user_count)

        return metric_sums

    def _log_epoch_metrics(self, epoch, metrics):
        logger.info(
            f"epoch : {epoch}, recall@{self.topk} : {metrics['recall']:.4f}, precision@{self.topk} : {metrics['precision']:.4f}, ndcg@{self.topk} : {metrics['ndcg']:.4f}, mrr@{self.topk} : {metrics['mrr']:.4f}, map@{self.topk} : {metrics['map']:.4f}"
        )

    def _update_best_validation(self, epoch, metrics):
        self.best_valid_epoch = epoch
        self.best_valid_recall = metrics["recall"]
        self.best_valid_precision = metrics["precision"]
        self.best_valid_ndcg = metrics["ndcg"]
        self.best_valid_mrr = metrics["mrr"]
        self.best_valid_map = metrics["map"]

        logger.info(f"best valid score improved : {metrics['recall']:.4f}")
        logger.info(f"save checkpoint file : {self.checkpoint_model_file}")
        torch.save(self.model.state_dict(), self.checkpoint_model_file)

    def _store_training_metadata(
        self, training_elapsed_time_list, current_epoch, total_epoch
    ):
        self.model.training_average_elapsed_time_sec = float(
            np.mean(training_elapsed_time_list)
        )
        self.model.early_stop_epoch = current_epoch
        self.model.total_epoch = total_epoch

    def _load_best_checkpoint(self):
        logger.info("load best valid model")
        self.model.load_state_dict(
            torch.load(self.checkpoint_model_file, map_location=self.device)
        )

    def _build_metric_result(self, metrics):
        return {
            f"recall@{self.topk}": metrics["recall"],
            f"precision@{self.topk}": metrics["precision"],
            f"ndcg@{self.topk}": metrics["ndcg"],
            f"mrr@{self.topk}": metrics["mrr"],
            f"map@{self.topk}": metrics["map"],
        }

    def _evaluate_validation(self, epoch):
        self.model.eval()
        with torch.no_grad():
            metrics = self._evaluate_split(
                self.val_dl,
                self.dataset.train_user_idx2viewed_idx_dict,
                self.dataset.val_user_idx2item_idx_dict,
                self.dataset.val_user_num,
                desc=f"\033[31mValid {epoch:>4}\033[0m",
            )
        self._log_epoch_metrics(epoch, metrics)
        return metrics

    def _evaluate_test(self):
        self.model.eval()
        with torch.no_grad():
            self._load_best_checkpoint()
            metrics = self._evaluate_split(
                self.test_dl,
                self.dataset.train_val_user_idx2viewed_idx_dict,
                self.dataset.test_user_idx2item_idx_dict,
                self.dataset.test_user_num,
            )
        return metrics

    def train(self):
        self.train_dl, self.val_dl, self.test_dl = self.dataset.get_dataloader()
        self._initialize_training_state()

        training_elapsed_time_list = []
        current_epoch = 0
        total_epoch = 0

        for epoch in range(self.epochs):
            total_epoch += 1
            training_elapsed_time_list.append(self._run_train_epoch(epoch))

            validation_metrics = self._evaluate_validation(epoch)
            if validation_metrics["recall"] > self.best_valid_metric:
                self._update_best_validation(epoch, validation_metrics)

            is_stop, current_epoch, self.best_valid_metric = is_train_stop(
                validation_metrics["recall"],
                self.best_valid_metric,
                current_epoch,
                self.model.early_stop_num,
            )
            if is_stop:
                logger.info("=== early stop break ===")
                logger.info(f"best valid score : {self.best_valid_metric:.4f}")
                break

        self._store_training_metadata(
            training_elapsed_time_list, current_epoch, total_epoch
        )
        test_metrics = self._evaluate_test()

        self.test_recall = test_metrics["recall"]
        self.test_precision = test_metrics["precision"]
        self.test_ndcg = test_metrics["ndcg"]
        self.test_mrr = test_metrics["mrr"]
        self.test_map = test_metrics["map"]

        logger.info(
            f"BEST VALID RESULT : recall@{self.topk} : {self.best_valid_recall:.4f}, precision@{self.topk} : {self.best_valid_precision:.4f}, ndcg@{self.topk} : {self.best_valid_ndcg:.4f}, mrr@{self.topk} : {self.best_valid_mrr:.4f}, map@{self.topk} : {self.best_valid_map:.4f}"
        )
        logger.info(
            f"TEST RESULT       : recall@{self.topk} : {self.test_recall:.4f}, precision@{self.topk} : {self.test_precision:.4f}, ndcg@{self.topk} : {self.test_ndcg:.4f}, mrr@{self.topk} : {self.test_mrr:.4f}, map@{self.topk} : {self.test_map:.4f}"
        )

        best_valid_metrics = {
            "recall": self.best_valid_recall,
            "precision": self.best_valid_precision,
            "ndcg": self.best_valid_ndcg,
            "mrr": self.best_valid_mrr,
            "map": self.best_valid_map,
        }
        return self._build_metric_result(
            best_valid_metrics
        ), self._build_metric_result(test_metrics)
