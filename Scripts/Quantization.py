import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from torch.quantization import QuantStub, DeQuantStub, prepare, convert, default_qconfig
from torch.ao.quantization.qconfig import float_qparams_weight_only_qconfig
import cProfile
import pstats

def main():
    # Hyperparameters
    b_size = 64 
    blk_size = 512 
    max_steps = 5000
    eval_freq = 500
    lr = 3e-4
    device = 'cpu'  # Quantization is supported on CPU
    eval_steps = 200
    emb_dim = 512
    num_heads = 8
    num_layers = 8
    drop_prob = 0.2
    
    torch.manual_seed(1337)
    
    with open('Data/Harry_Potter_all_books_preprocessed.txt', 'r', encoding='utf-8') as f:
        dataset = f.read()
        
    unique_chars = sorted(list(set(dataset)))
    vocab_len = len(unique_chars)
    
    char_to_int = { ch:i for i,ch in enumerate(unique_chars) }
    int_to_char = { i:ch for i,ch in enumerate(unique_chars) }
    encode_text = lambda s: [char_to_int[c] for c in s] 
    decode_text = lambda l: ''.join([int_to_char[i] for i in l])
    
    data = torch.tensor(encode_text(dataset), dtype=torch.long)
    n = int(0.9*len(data)) 
    train_set = data[:n]
    val_set = data[n:]
    
    def get_data_batch(split):
        data = train_set if split == 'train' else val_set
        indices = torch.randint(len(data) - blk_size, (b_size,))
        x = torch.stack([data[i:i+blk_size] for i in indices])
        y = torch.stack([data[i+1:i+blk_size+1] for i in indices])
        x, y = x.to(device), y.to(device)
        return x, y
    
    @torch.no_grad()
    def calc_loss():
        result = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_steps)
            for k in range(eval_steps):
                X, Y = get_data_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            result[split] = losses.mean()
        model.train()
        return result
    
    class SelfAttentionHead(nn.Module):
        """ one head of self-attention """
    
        def __init__(self, head_dim):
            super().__init__()
            self.key = nn.Linear(emb_dim, head_dim, bias=False)
            self.query = nn.Linear(emb_dim, head_dim, bias=False)
            self.value = nn.Linear(emb_dim, head_dim, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(blk_size, blk_size)))
    
            self.dropout = nn.Dropout(drop_prob)
    
        def forward(self, x):
            B,T,C = x.shape
            k = self.key(x)  
            q = self.query(x) 
            wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            v = self.value(x)
            out = wei @ v 
            return out
    
    class MultiHeadSelfAttention(nn.Module):
        """ multiple heads of self-attention in parallel """
    
        def __init__(self, num_heads, head_dim):
            super().__init__()
            self.heads = nn.ModuleList([SelfAttentionHead(head_dim) for _ in range(num_heads)])
            self.proj = nn.Linear(head_dim * num_heads, emb_dim)
            self.dropout = nn.Dropout(drop_prob)
    
        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out
    
    class FeedForward(nn.Module):
        """ a simple linear layer followed by a non-linearity """
    
        def __init__(self, emb_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(emb_dim, 4 * emb_dim),
                nn.ReLU(),
                nn.Linear(4 * emb_dim, emb_dim),
                nn.Dropout(drop_prob),
            )
    
        def forward(self, x):
            return self.net(x)
    
    class TransformerBlock(nn.Module):
        """ Transformer block: communication followed by computation """
    
        def __init__(self, emb_dim, num_heads):
            super().__init__()
            head_dim = emb_dim // num_heads
            self.sa = MultiHeadSelfAttention(num_heads, head_dim)
            self.ffwd = FeedForward(emb_dim)
            self.ln1 = nn.LayerNorm(emb_dim)
            self.ln2 = nn.LayerNorm(emb_dim)
    
        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x
    
    class GPTLanguageModel(nn.Module):
    
        def __init__(self):
            super().__init__()
            self.quant = QuantStub()
            self.dequant_emb = DeQuantStub()  # Dequantize before embedding layers
            self.token_embedding_table = nn.Embedding(vocab_len, emb_dim)
            self.position_embedding_table = nn.Embedding(blk_size, emb_dim)
            self.blocks = nn.Sequential(*[TransformerBlock(emb_dim, num_heads=num_heads) for _ in range(num_layers)])
            self.ln_f = nn.LayerNorm(emb_dim) 
            self.lm_head = nn.Linear(emb_dim, vocab_len)
            self.dequant = DeQuantStub()
            self.apply(self._init_weights)
    
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
        def forward(self, idx, targets=None):
            idx = self.quant(idx.float())  # Convert to float for quantization
            idx = self.dequant_emb(idx)  # Dequantize before embedding layers
            B, T = idx.shape
            tok_emb = self.token_embedding_table(idx.long())  # Convert back to long for embedding
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = tok_emb + pos_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            logits = self.dequant(logits)
    
            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)
    
            return logits, loss
    
        def generate(self, idx, max_new_tokens):
            idx = self.quant(idx.float())  # Convert to float for quantization
            idx = self.dequant_emb(idx)  # Dequantize before embedding layers
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -blk_size:]
                logits, loss = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return self.dequant(idx)
    
    model = GPTLanguageModel()
    model_path = os.path.join(os.getcwd(), 'model2.pth')
    state_dict = torch.load(model_path, map_location=device)
    
    # Remove "module." prefix if it exists
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # Print the weights matrix of the model before quantization
    print('Weights before quantization')
    print(model.token_embedding_table.weight)
    print(model.token_embedding_table.weight.dtype)
    
    # Print the size of the model before quantization
    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp_delme.p")
        print('Size (MB):', os.path.getsize("temp_delme.p") / 1e6)
        os.remove("temp_delme.p")
    
    print('Size of the model before quantization')
    print_size_of_model(model)
    
    # Prepare the model for quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model.token_embedding_table.qconfig = float_qparams_weight_only_qconfig
    model.position_embedding_table.qconfig = float_qparams_weight_only_qconfig
    
    # Skip LayerNorm quantization
    for module_name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            module.qconfig = None
    
    prepare(model, inplace=True)
    
    # Calibrate the model with representative data
    calibration_steps = 150  # Adjust the number of calibration steps as needed
    for _ in range(calibration_steps):
        x, y = get_data_batch('train')
        model(x.float())  # Convert input to float for calibration
    
    # Convert the model to quantized version
    convert(model, inplace=True)
    
    # Print model weights and sizes
    print("Weights after quantization:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}... | dtype: {param.dtype}")
    
    # Print the size of the model after quantization
    print("Model size after quantization:")
    print_size_of_model(model)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()
