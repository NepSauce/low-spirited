import torch
import torch.optim as optim

from utils.tokenizer import CharTokenizer
from model.low_spirited import LowSpiritedModel


with open("data/input.txt", "r", encoding="utf-8") as f:
    data = f.read()

tokenizer = CharTokenizer(data)
data = torch.tensor(tokenizer.encode(data), dtype=torch.long)

batch_size = 32
block_size = 64
embed_dim = 4
num_heads = 4
lr = 3e-4

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y

model = LowSpiritedModel(
    vocab_size=tokenizer.vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads
) 

optimizer = optim.Adam(model.parameters(), lr=lr)

for step in range(2000):
    x, y = get_batch()

    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
