from torch import nn, optim
from trainer.Trainer import Trainer
from util.util import initialize_with_xavier_normal

from models.Model import Model


class BPR(Model):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.dataset = dataset
        self.epochs = config["epochs"]
        self.topk = config["topk"]

        self.embedding_size = config["embedding_size"]
        self.learning_rate = config["learning_rate"]
        self.reg_weight = config["reg_weight"]

        self.user_embedding = nn.Embedding(
            self.user_num, self.embedding_size
        ).to(self.device)
        self.item_embedding = nn.Embedding(
            self.item_num, self.embedding_size
        ).to(self.device)
        self.apply(initialize_with_xavier_normal)
        self.trainable_parameters = self._count_trainable_parameters()

        self.set_optimizer()

    def forward(self):
        return self.user_embedding.weight, self.item_embedding.weight

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
