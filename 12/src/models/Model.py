from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from util.loss import bpr_loss, squared_embedding_penalty


class Model(nn.Module, metaclass=ABCMeta):
    def __init__(self, config, dataset):
        super().__init__()

        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        self.user_id_name = config["user_id_name"]
        self.item_id_name = config["item_id_name"]

        self.user_idx_name = config["user_idx_name"]
        self.item_idx_name = config["item_idx_name"]
        self.neg_item_idx_name = config["neg_item_idx_name"]

        self.early_stop_num = config["early_stop_num"]
        self.gpu = config["gpu"]
        self.device = self._build_device(self.gpu)

    @staticmethod
    def _build_device(gpu_index):
        return torch.device(
            f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu"
        )

    def _count_trainable_parameters(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def compute_training_loss(self, batch):
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def set_optimizer(self):
        raise NotImplementedError

    def _get_training_indices(self, batch):
        return (
            batch[self.user_idx_name],
            batch[self.item_idx_name],
            batch[self.neg_item_idx_name],
        )

    @staticmethod
    def _dot_product(left_embeddings, right_embeddings):
        return torch.einsum("bd,bd->b", left_embeddings, right_embeddings)

    def _compute_bpr_loss(
        self,
        user_embeddings,
        positive_item_embeddings,
        negative_item_embeddings,
        penalty_embeddings=None,
    ):
        positive_scores = self._dot_product(
            user_embeddings, positive_item_embeddings
        )
        negative_scores = self._dot_product(
            user_embeddings, negative_item_embeddings
        )
        if penalty_embeddings is None:
            penalty_embeddings = (
                user_embeddings,
                positive_item_embeddings,
                negative_item_embeddings,
            )
        return bpr_loss(
            positive_scores, negative_scores
        ) + self.reg_weight * squared_embedding_penalty(*penalty_embeddings)

    def __str__(self):
        """Return the module representation with trainable parameter count."""
        return f"{super().__str__()}\nTrainable parameters: {self._count_trainable_parameters()}"
