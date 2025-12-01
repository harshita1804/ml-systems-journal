import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 200
n_embd = 384
num_heads = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# --- Data Loading ---
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda d: ''.join([itos[i] for i in d])

# Train/Val Split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout= nn.Dropout(dropout)

    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim = -1)  # concatenate across dim C -- channels
        out = self.proj(out)
        out = self.dropout(out)
        return out
class FeedForward(nn.Module):
    def __init__(self, n_embd): # layer followed by relu (non linearity)
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x): # this is on a per token basis
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x+ self.sa(self.ln1(x))
        x = x+ self.ffwd(  self.ln2(x))
        return x

# --- Model Definition ---
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, n_embd) 
        # self.sa_head = MultiHeadAttention(4, n_embd//4)
        # self.ffwd_head = FeedForward(n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for i in range(n_layer)],
            # Block(n_embd, 4), 
            # Block(n_embd, 4),
            # Block(n_embd, 4), 
            # Block(n_embd, 4),
            nn.LayerNorm(n_embd))
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        token_emb = self.token_embedding_table(idx)  # (B,T,C, this C is n_embd) # we have encoded  indices and encoded the identity of these tokens to n_embd dimensions
        # now we will also encode positions of these tokens 
        pos_emb = self.position_embedding_table(torch.arange(T, device = DEVICE)) #T,C
        # now we add the pos embedding to the embeding of incoming idx
        x = token_emb + pos_emb #B,T,C
        # x = self.sa_hea≥
        # pos embedding will not help a bigram modfel since they are translation invariant 
        # but it will help an n gram model and also a trasformer

        x = self.blocks(x)
        logits = self.lm_head(x)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # (B, C) 
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q@k.transpose(-2, -1) * C**-0.5

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)

        out = wei @ v
        return out 

# --- Main Training Loop ---
if __name__ == "__main__":
    model = BigramLanguageModel()
    m = model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iter in range(MAX_ITERS):
        # Every once in a while evaluate the loss on train and val sets
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sample a batch of data
        xb, yb = get_batch('train')

        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
