from Transformer import Transformer
from Encoder import Encoder
from Decoder import Decoder
from shape_check import ShapeChecker
import torch
#DUMMY DATA
embedding_dim = 256
hidden_dim = 512
num_heads = 8
vocab_size = 50000
encoder = Encoder(embedding_dim, hidden_dim, vocab_size, num_heads)
decoder = Decoder(embedding_dim, hidden_dim, vocab_size, num_heads)
model = Transformer(encoder, decoder)
checker = ShapeChecker(model, verbose=True)
checker.enable()
x = torch.randint(0, vocab_size, (4, 10))
y = torch.randint(0, vocab_size, (4, 10))
out = model(x, y)
print("Final output:", out.shape)
checker.disable()