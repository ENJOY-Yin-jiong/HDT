import torch
import torch.nn as nn
import torch.nn.functional as F

from .operation import Conv1D


class PositionalEmbedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, embedding_dim, num_embeddings):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        bsz, seq_length = inputs.shape[:2]
        position_ids = torch.arange(seq_length,
                                    dtype=torch.long,
                                    device=inputs.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings


class Projection(nn.Module):
    def __init__(self, in_dim, dim, drop_rate=0.0):
        super(Projection, self).__init__()
        self.drop = nn.Dropout(p=drop_rate)
        self.projection = Conv1D(in_dim=in_dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 bias=True,
                                 padding=0)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, input_features):
        # the input feature with shape (batch_size, seq_len, in_dim)
        input_features = self.drop(input_features)
        output = self.projection(input_features)  # (batch_size, seq_len, dim)
        output = self.layer_norm(output)
        return output


class Prediction(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, drop_rate=0.):
        super(Prediction, self).__init__()
        self.fc1 = Conv1D(in_dim=in_dim,
                          out_dim=hidden_dim,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc2 = Conv1D(in_dim=hidden_dim,
                          out_dim=out_dim,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True)

    def forward(self, input_feature):
        output = self.fc1(input_feature)
        output = F.gelu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output