import torch
from torch import nn


class EventEncoder(nn.Module):
    def __init__(
        self,
        cat_feature_indexes: list,
        vocab_sizes: list,
        cont_feature_indexes: list,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.cat_feature_indexes = cat_feature_indexes
        self.vocab_sizes = vocab_sizes
        self.cont_feature_indexes = cont_feature_indexes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.vocab_sizes[i], self.hidden_dim)
                for i in range(len(self.vocab_sizes))
            ]
        )

        self.batch_norm = nn.BatchNorm1d(len(self.cont_feature_indexes))
        self.cont_embeddings = nn.Linear(
            len(self.cont_feature_indexes), self.hidden_dim
        )

        self.global_linear = nn.Linear(
            (len(self.vocab_sizes) + 1) * self.hidden_dim, self.output_dim
        )

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        :praram input_features:
        :returns:
        """

        cat_embeddings = torch.cat(
            [
                self.cat_embeddings[i](
                    input_features[:, :, self.cat_feature_indexes[i]]
                )
                for i in range(len(self.vocab_sizes))
            ],
            dim=2,
        )

        cont_embeddings = self.cont_embeddings(
            self.batch_norm(input_features[:, :, self.cont_feature_indexes])
        )

        whole_embeddings = torch.cat([cat_embeddings, cont_embeddings], dim=2)

        out = self.global_linear(whole_embeddings)

        return out
