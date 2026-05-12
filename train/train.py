import torch
from utils.tokenizer import CharTokenizer

with open("data/input.txt", "r", encoding="utf-8") as f:
    data = f.read()

tokenizer = CharTokenizer(data)
data = torch.tensor(tokenizer.encode(data), dtype=torch.long)

block_size = 8;
x = data[:block_size]
y = data[1:block_size+1]

print(x)
print(y)