from torch import nn
from Things import TextEmbedding, PositionalEncoding, MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_heads, dropout = 0.2):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, X, mask=None):

        norm_X = self.layer_norm1(X)
        attn_out, _ = self.attention(norm_X, mask=mask)
        X = X + self.dropout(attn_out)

        norm2_O = self.layer_norm2(X)
        feed = self.feedforward(norm2_O)
        X = X + self.dropout(feed)
        return X

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_heads, dropout = 0.2, shared_embedding = None):
        super().__init__()
        self.textEmbed = shared_embedding or TextEmbedding(vocab_size, embedding_dim, dropout)
        self.positionalEncod = PositionalEncoding(embedding_dim, max_len=512, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderBlock(embedding_dim, hidden_dim, num_heads, dropout) for _ in range(5)
        ])
    def forward(self, ids):
        X = self.textEmbed(ids)
        X = self.positionalEncod(X)
        for layer in self.layers:
            X = layer(X)
        return X

