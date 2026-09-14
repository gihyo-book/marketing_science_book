import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from torch import nn, optim
from trainer.Trainer import Trainer
from util.util import initialize_with_xavier_normal

from models.Model import Model


class NGCF(Model):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.dataset = dataset
        self.epochs = config["epochs"]
        self.topk = config["topk"]

        self.interaction_matrix = self.dataset.get_train_coo_matrix()

        self.embedding_size = config["embedding_size"]
        self.hidden_size_list = [self.embedding_size] + config[
            "hidden_size_list"
        ]
        self.node_dropout = config["node_dropout"]
        self.message_dropout = config["message_dropout"]
        self.reg_weight = config["reg_weight"]
        self.learning_rate = config["learning_rate"]

        self.user_embedding = nn.Embedding(
            self.user_num, self.embedding_size
        ).to(self.device)
        self.item_embedding = nn.Embedding(
            self.item_num, self.embedding_size
        ).to(self.device)
        self.gnn_layers = self._build_gnn_layers().to(self.device)
        self.message_activation = nn.LeakyReLU(negative_slope=0.2)
        self.message_dropout_layer = nn.Dropout(self.message_dropout)

        self.register_buffer(
            "norm_adj_matrix", self._build_normalized_adjacency()
        )

        self.apply(initialize_with_xavier_normal)
        self.trainable_parameters = self._count_trainable_parameters()

        self.set_optimizer()

    def _build_gnn_layers(self):
        layers = nn.ModuleList()
        for input_size, output_size in zip(
            self.hidden_size_list[:-1], self.hidden_size_list[1:]
        ):
            layers.append(
                BiGNNLayer(input_size, output_size, device=self.device)
            )
        return layers

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
        inverse_degree = np.power(degree + 1e-7, -1.0)
        degree_matrix = sparse.diags_array(inverse_degree)
        normalized_adjacency = degree_matrix @ adjacency + sparse.eye_array(
            total_nodes,
            dtype=np.float32,
        )
        normalized_adjacency = normalized_adjacency.tocoo()

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

    def _drop_adjacency_values(self):
        if not self.training or self.node_dropout == 0:
            return self.norm_adj_matrix
        dropped_values = F.dropout(
            self.norm_adj_matrix.values(), p=self.node_dropout, training=True
        )
        return torch.sparse_coo_tensor(
            self.norm_adj_matrix.indices(),
            dropped_values,
            self.norm_adj_matrix.shape,
            device=self.device,
        ).coalesce()

    def _update_embeddings(self, gnn_layer, normalized_adjacency, embeddings):
        embeddings = gnn_layer(normalized_adjacency, embeddings)
        embeddings = self.message_activation(embeddings)
        embeddings = self.message_dropout_layer(embeddings)
        return F.normalize(embeddings, p=2, dim=1)

    def _propagate_embeddings(self, normalized_adjacency, all_embeddings):
        layer_outputs = [all_embeddings]
        for gnn_layer in self.gnn_layers:
            all_embeddings = self._update_embeddings(
                gnn_layer, normalized_adjacency, all_embeddings
            )
            layer_outputs.append(all_embeddings)
        return torch.cat(layer_outputs, dim=1)

    def _split_user_item_embeddings(self, all_embeddings):
        return torch.split(all_embeddings, [self.user_num, self.item_num])

    def forward(self):
        normalized_adjacency = self._drop_adjacency_values()
        all_embeddings = self._get_initial_node_embeddings()
        ngcf_all_embeddings = self._propagate_embeddings(
            normalized_adjacency, all_embeddings
        )
        return self._split_user_item_embeddings(ngcf_all_embeddings)

    def compute_training_loss(self, batch):
        user_indices, positive_item_indices, negative_item_indices = (
            self._get_training_indices(batch)
        )
        all_user_embeddings, all_item_embeddings = self()
        return self._compute_bpr_loss(
            all_user_embeddings[user_indices],
            all_item_embeddings[positive_item_indices],
            all_item_embeddings[negative_item_indices],
        )

    def set_optimizer(self):
        self.optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=0.0
        )

    def fit(self):
        trainer = Trainer(self.config, self, self.dataset)
        self.best_valid_result, self.test_result = trainer.train()


class BiGNNLayer(nn.Module):
    r"""Propagate a layer of Bi-interaction GNN.

    .. math::
        output = SEW_1 + ((SE) \otimes E)W_2
    """

    def __init__(self, in_dim, out_dim, device):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim).to(
            device
        )
        self.interaction_transform = nn.Linear(
            in_features=in_dim, out_features=out_dim
        ).to(device)

    def forward(self, propagation_matrix, features):
        propagated = torch.sparse.mm(propagation_matrix, features)
        linear_part = self.linear(propagated)
        interaction_part = self.interaction_transform(
            torch.mul(propagated, features)
        )
        return linear_part + interaction_part
