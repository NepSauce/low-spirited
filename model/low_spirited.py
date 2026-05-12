import torch
import torch.nn as nn

class LowSpiritedModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, idx):
        x = self.embedding(idx)

        return x