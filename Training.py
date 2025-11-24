import torch as t
from torch import nn, optim
import math
from torch.utils.data import Dataset

from Things import TextEmbedding
from Transformer import Transformer
from Decoder import Decoder
from Encoder import Encoder

embedding_dim = 256
hidden_dim = 512
num_heads = 8
vocab_size = 100000
num_layers = 5
dropout = 0.2
learning_rate = 1e-3
epochs = 100
path = r"C:\Users\Vova\Desktop\gooaq_pairs.jsonl"
device = 'cuda' if t.cuda.is_available() else 'cpu'

import json

class GooaqPairsDataset(Dataset):
    def __init__(self, filename, text_embedding):
        self.datalist = []
        with open(filename, 'r', encoding='utf-8') as f:
            data = f.read()
            for line in data.strip().split('\n'):
                qa_pair = json.loads(line)

                self.datalist.append([qa_pair[0], qa_pair[1]])

        self.text_embedding = text_embedding
    def __len__(self):
        return len(self.datalist)
    def __getitem__(self, idx):
        q,a = self.datalist[idx]

        q_ids = t.tensor(self.text_embedding.text_to_ids(q), dtype=t.long)
        a_ids = t.tensor(self.text_embedding.text_to_ids(a), dtype=t.long)

        return q_ids, a_ids

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    q, a = zip(*batch)
    questions_padded = pad_sequence(q, batch_first=True, padding_value=0)
    answers_padded = pad_sequence(a, batch_first=True, padding_value=0)
    return questions_padded, answers_padded


from torch.utils.data import DataLoader

shared_text_embed = TextEmbedding(1, embedding_dim, dropout)  # temporary
vocab = shared_text_embed.tokenize(path, limit=vocab_size)

actual_vocab_size = len(shared_text_embed.vocab)

shared_text_embed = TextEmbedding(actual_vocab_size, embedding_dim, dropout)
shared_text_embed.vocab = vocab
shared_text_embed.inv_vocab = {v: k for k, v in vocab.items()}

dataset = GooaqPairsDataset(r"C:\Users\Vova\Desktop\gooaq_pairs.jsonl", text_embedding=shared_text_embed)
loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)

criterion = nn.CrossEntropyLoss(ignore_index=0)
decoder = Decoder(embedding_dim, hidden_dim, actual_vocab_size, num_heads, num_layers, dropout, shared_text_embed).to(device)
encoder = Encoder(embedding_dim, hidden_dim, actual_vocab_size, num_heads, dropout, shared_embedding=shared_text_embed).to(device)
model = Transformer(encoder, decoder).to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(epochs):
    total_loss = 0
    for q,a in loader:
        q,a = q.to(device), a.to(device)

        optimizer.zero_grad()

        input_target = a[:, :-1]
        expected_target = a[:, 1:]

        seq_len = input_target.size(1)
        mask = t.tril(t.ones(seq_len, seq_len, device=input_target.device)).unsqueeze(0).unsqueeze(0)

        logits = model(q, input_target, target_mask=mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), expected_target.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print('Epoch: {}, Loss: {:.4f}'.format(epoch + 1, avg_loss))

