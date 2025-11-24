import torch as t
from torch import nn
import math

from Encoder import Encoder
from Decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, encoder : Encoder, decoder : Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, ids_input, ids_target, target_mask=None, device=None):
        if device is None:
            device = next(self.parameters()).device

        ids_input = ids_input.to(device)
        ids_target = ids_target.to(device)

        encoder_output = self.encoder(ids_input)
        logits = self.decoder(ids_target, encoder_output, mask=target_mask)
        return logits

    @t.no_grad()
    def generate(self, ids_input, length, start_ids, end_ids, device=None):
        if device is None:
            device = ids_input.device

        encoder_output = self.encoder(ids_input.to(device))
        batch_size = ids_input.size(0)

        generated_output = t.full((batch_size, 1), start_ids, device=device, dtype=t.long)

        finished = t.zeros(batch_size, dtype=t.bool, device=device)

        for step in range(length - 1):
            decoder_output = self.decoder(generated_output, encoder_output)
            next_token = decoder_output[:, -1, :]
            next_ids = t.argmax(next_token, dim=-1, keepdim=True)
            generated_output = t.cat([generated_output, next_ids], dim=-1)

            finished |= (next_ids.squeeze(1) == end_ids)
            if finished.all():
                break

        return generated_output
