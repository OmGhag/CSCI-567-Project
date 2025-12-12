import torch
from transformer_model import BigramLanguageModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Test parameters
vocab_size = 65
n_embd = 64
block_size = 32
n_head = 4
n_layer = 4

# Create model
model = BigramLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer)
model = model.to(device)

# Test forward pass
idx = torch.randint(0, vocab_size, (4, block_size)).to(device)
targets = torch.randint(0, vocab_size, (4, block_size)).to(device)

logits, loss = model(idx, targets)
print(f"✓ Forward pass works! Loss: {loss.item():.4f}")

# Test generation
context = torch.zeros((1, 1), dtype=torch.long).to(device)
generated = model.generate(context, max_new_tokens=50)
print(f"✓ Generation works! Generated shape: {generated.shape}")

print("\n✓ All basic tests passed!")