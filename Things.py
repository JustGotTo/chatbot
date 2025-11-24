import torch as t
from torch import nn
import re
import math

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def tokenize(self, filename, limit=None):
        vocab = {}
        with open(filename, "r", encoding="utf-8") as f:

            text = f.read().lower()
            tokens = re.findall(r"\w+", text)
            vocab = {"<pad>": 0}
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
                    if limit is not None and len(vocab) >= limit:
                        break
            vocab["<unk>"] = len(vocab)
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

        return vocab

    def text_to_ids(self, text):
        tokens = re.findall(r"\w+", text.lower())
        ids = [self.vocab.get(token, self.vocab.get("<unk>", 0)) for token in tokens]
        return [min(i, self.embedding.num_embeddings - 1) for i in ids]

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)
        return self.dropout(embeddings)

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=512, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        posenc = t.zeros(max_len, embedding_dim)
        position = t.arange(0, max_len, dtype=t.float).unsqueeze(1)
        division = t.exp(t.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        posenc[:, 0::2] = t.sin(position * division)
        posenc[:, 1::2] = t.cos(position * division)
        posenc = posenc.unsqueeze(0)
        self.register_buffer('posenc', posenc)

    def forward(self, token_embeddings):
        seq_len = token_embeddings.size(1)
        token_embeddings = token_embeddings + self.posenc[:, :seq_len, :]
        return self.dropout(token_embeddings)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embedding_dim, key_dim, value_dim):
        super().__init__()
        self.key_dim = key_dim
        self.Q1 = nn.Linear(embedding_dim, key_dim)
        self.K1 = nn.Linear(embedding_dim, key_dim)
        self.V1 = nn.Linear(embedding_dim, value_dim)
    def forward(self, X):
        Q = self.Q1(X)
        K = self.K1(X)
        V = self.V1(X)
        score = t.matmul(Q,K.transpose(-1, -2)) / math.sqrt(self.key_dim)
        weights = t.softmax(score, dim=-1)
        output = t.matmul(weights, V)
        return output, weights
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.per_head_dim = embedding_dim // num_heads

        assert embedding_dim % num_heads == 0, "Make sure embedding_dim is divisible by num_heads"

        self.Q1 = nn.Linear(embedding_dim, embedding_dim)
        self.K1 = nn.Linear(embedding_dim, embedding_dim)
        self.V1 = nn.Linear(embedding_dim, embedding_dim)

        self.concat = nn.Linear(embedding_dim, embedding_dim)
    def forward(self, query, key=None, value=None, mask=None):
        if key is None:
            key = query
        if value is None:
            value = query
        Q = self.Q1(query)
        K = self.K1(key)
        V = self.V1(value)

        batch_size, seqlen, _ = query.size()

        Q = Q.view(batch_size, -1, self.num_heads, self.per_head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.per_head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.per_head_dim).transpose(1, 2)


        score = t.matmul(Q,K.transpose(-1, -2)) / math.sqrt(self.per_head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))

        weights = t.softmax(score, dim=-1)
        output = t.matmul(weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)
        output = self.concat(output)

        return output, weights
