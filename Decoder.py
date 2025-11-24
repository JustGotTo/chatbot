import torch as t
from torch import nn
import math


from Things import MultiHeadAttention, PositionalEncoding, TextEmbedding

class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads, dropout):
        super(DecoderBlock, self).__init__()

        self.self_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)

        self.attention_l2 = MultiHeadAttention(embedding_dim, num_heads)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.layer_norm3 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, encoder_output, mask=None):
        seq_length = X.size(1)
        norm1 = self.layer_norm1(X)
        attention, _ = self.self_attention(norm1, norm1, norm1, mask=mask)
        X = X + self.dropout(attention)

        norm2 = self.layer_norm2(X)
        attention2, _ = self.attention_l2(norm2, encoder_output, encoder_output)
        X = X + self.dropout(attention2)

        X = self.layer_norm3(X)
        feed = self.feedforward(X)
        X = X + self.dropout(feed)

        return X


class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_heads, num_layers=5, dropout = 0.2, embedding_module = None):
        super().__init__()

        self.text_embedding = embedding_module if embedding_module is not None else TextEmbedding(vocab_size, embedding_dim, dropout)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=512, dropout=dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(embedding_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, encoder_output, mask=None):
        X = self.text_embedding(token_ids)
        X = self.positional_encoding(X)
        X = self.dropout(X)

        if mask is None:
            seq_len = token_ids.size(1)
            mask = t.tril(t.ones(seq_len, seq_len, device=token_ids.device)).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            X = layer(X, encoder_output, mask)

        final_output = self.linear(X)
        return final_output