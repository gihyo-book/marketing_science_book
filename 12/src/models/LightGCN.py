import numpy as np
import torch
from scipy import sparse
from torch import nn, optim
from trainer.Trainer import Trainer
from util.util import initialize_with_xavier_uniform

from models.Model import Model


class LightGCN(Model):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.dataset = dataset
        self.epochs = config["epochs"]
        self.topk = config["topk"]

        self.interaction_matrix = self.dataset.get_train_coo_matrix()

        self.embedding_size = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.learning_rate = config["learning_rate"]
        self.reg_weight = config["reg_weight"]

        self.user_embedding = nn.Embedding(
            num_embeddings=self.user_num,
            embedding_dim=self.embedding_size,
        ).to(self.device)
        self.item_embedding = nn.Embedding(
            num_embeddings=self.item_num,
            embedding_dim=self.embedding_size,
        ).to(self.device)
        self.register_buffer(
            "norm_adj_matrix", self._build_normalized_adjacency()
        )

        self.apply(initialize_with_xavier_uniform)
        self.trainable_parameters = self._count_trainable_parameters()

        self.set_optimizer()

    def _build_normalized_adjacency(self):
        total_nodes = self.user_num + self.item_num
        interaction_matrix = self.interaction_matrix
        user_indices = np.asarray(interaction_matrix.row, dtype=np.int64)
        item_indices = (
            np.asarray(interaction_matrix.col, dtype=np.int64) + self.user_num
        )

        adjacency_rows = np.concatenate((user_indices, item_indices))
        adjacency_cols = np.concatenate((item_indices, user_indices))
        adjacency = sparse.coo_array(
            (
                np.ones(adjacency_rows.size, dtype=np.float32),
                (adjacency_rows, adjacency_cols),
            ),
            shape=(total_nodes, total_nodes),
        )
        adjacency.sum_duplicates()
        adjacency.data.fill(1.0)

        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inverse_sqrt_degree = np.power(degree + 1e-7, -0.5)
        degree_matrix = sparse.diags_array(inverse_sqrt_degree)
        normalized_adjacency = (
            degree_matrix @ adjacency @ degree_matrix
        ).tocoo()

        return self._scipy_sparse_to_torch_sparse(normalized_adjacency)

    def _scipy_sparse_to_torch_sparse(self, matrix):
        row = np.asarray(matrix.row, dtype=np.int64)
        col = np.asarray(matrix.col, dtype=np.int64)
        indices = np.vstack((row, col))
        index_tensor = torch.from_numpy(indices).to(self.device)
        value_tensor = torch.from_numpy(matrix.data.astype(np.float32)).to(
            self.device
        )
        return torch.sparse_coo_tensor(
            indices=index_tensor,
            values=value_tensor,
            size=matrix.shape,
            dtype=torch.float32,
            device=self.device,
        ).coalesce()

    def _get_initial_node_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        return torch.cat([user_embeddings, item_embeddings], dim=0)

    def _split_user_item_embeddings(self, all_embeddings):
        return torch.split(all_embeddings, [self.user_num, self.item_num])

    def forward(self):
        all_embeddings = self._get_initial_node_embeddings()
        embedding_sum = all_embeddings
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(
                self.norm_adj_matrix, all_embeddings
            )
            embedding_sum = embedding_sum + all_embeddings
        mean_embeddings = embedding_sum / (self.n_layers + 1)
        return self._split_user_item_embeddings(mean_embeddings)

    def compute_training_loss(self, batch):
        user_indices, positive_item_indices, negative_item_indices = (
            self._get_training_indices(batch)
        )
        all_user_embeddings, all_item_embeddings = self()
        user_initial_embeddings = self.user_embedding(user_indices)
        positive_item_initial_embeddings = self.item_embedding(
            positive_item_indices
        )
        negative_item_initial_embeddings = self.item_embedding(
            negative_item_indices
        )
        return self._compute_bpr_loss(
            all_user_embeddings[user_indices],
            all_item_embeddings[positive_item_indices],
            all_item_embeddings[negative_item_indices],
            penalty_embeddings=(
                user_initial_embeddings,
                positive_item_initial_embeddings,
                negative_item_initial_embeddings,
            ),
        )

    def set_optimizer(self):
        self.optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=0.0
        )

    def fit(self):
        trainer = Trainer(self.config, self, self.dataset)
        self.best_valid_result, self.test_result = trainer.train()
