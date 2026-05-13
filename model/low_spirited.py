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
        B, T, C = idx.shape
        x = self.embedding(idx)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        attn_scores = q @ k.transpose(-2, -1)
        attn_scores = attn_scores / (k.shape[-1] ** 0.5)

        # Causal masking
        mask = torch.tril(torch.ones(T, T, device=x.device))
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = attn_weights @ v
        logits = self.lm_head(out)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, targets)

        return logits, loss