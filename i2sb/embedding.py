from torch import nn
from .x_transformer import Encoder, TransformerWrapper

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class RainfallEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=100, max_seq_len=24,
                 device="cuda",embedding_dropout=0.0):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, rainfall):
        z = self.transformer(rainfall, return_embeddings=True)
        return z
    
    def encode(self, rainfall):
        return self.forward(rainfall)
