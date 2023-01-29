import math

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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TrainablePositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.max_len = max_len

        self.embedding = nn.Embedding(num_embeddings=self.max_len, embedding_dim=self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        B, T, H = x.size()

        position = torch.arange(self.max_len).expand(B, -1)
        x = self.embedding(position)

        return self.dropout(x)