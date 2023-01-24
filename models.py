import torch
from torch import nn

from layers import EventEncoder


class LSTMModel(nn.Module):
    def __init__(self,
                 cat_feature_indexes: list,
                 vocab_sizes: list,
                 cont_feature_indexes: list,
                 encoder_hidden_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3,
                 bias: bool = True,
                 batch_first: bool = True,
                 bidirectional: bool = False,
                 dropout: float = 0.1):
        super().__init__()

        self.cat_feature_indexes = cat_feature_indexes
        self.vocab_sizes = vocab_sizes
        self.cont_feature_indexes = cont_feature_indexes
        self.encoder_hidden_dim = encoder_hidden_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.event_embedding = EventEncoder(
            cat_feature_indexes=self.cat_feature_indexes,
            vocab_sizes=self.vocab_sizes,
            cont_feature_indexes=self.cont_feature_indexes,
            hidden_dim=self.encoder_hidden_dim,
            output_dim=self.hidden_dim
        )

        self.seq2seq = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=self.batch_first,
            bidirectional=self.bidirectional,
            dropout=self.dropout
        )

        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        event_embeddings = self.event_embedding(input_features)
        x, (h, c) = self.seq2seq(event_embeddings)
        out = self.out(h[:, :, -1])

        return out