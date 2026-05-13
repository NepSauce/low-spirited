import torch
import torch.nn as nn
import torch.nn.functional as F

class LowSpiritedModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, targets)



        return logits, loss