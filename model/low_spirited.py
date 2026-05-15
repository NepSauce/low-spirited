import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Measures similarity between tokens
        attn = q @ k.transpose(-2, -1)
        attn = attn / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(T, T, device=x.device))

        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        # Multi-head self attention
        self.attn = MultiHeadAttention(embed_dim, num_heads)

        # Layer normalization stabilizes activations
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Residual connection around attention
        #
        # x = old information
        # attn(...) = new contextual information
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))

        return x


class LowSpiritedModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.Sequential(
            Block(embed_dim, num_heads),
            Block(embed_dim, num_heads),
            Block(embed_dim, num_heads),
            Block(embed_dim, num_heads)
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.position_embedding = nn.Embedding(1024, embed_dim)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.embedding(idx)
        positions = torch.arange(T, device=idx.device)
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings
        x = self.blocks(x)
        
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)

            # Focus on the last time step
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx