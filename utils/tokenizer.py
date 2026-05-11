class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])
    
    @property
    def vocab_size(self):
        return len(self.stoi)