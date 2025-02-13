import torch
import torch.nn as nn
import math

# RMSNorm is a normalization technique that normalizes the input by dividing by the square root of the variance plus a small number to prevent division by zero
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5): # the number of features/dimensions/embeddings in the input, eps is a small number to prevent division by zero
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # weight is a learnable parameter that scales the input
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).sqrt() + self.eps # compute the norm of the input
        return x / norm * self.weight # normalize the input by dividing by the norm and scale it by the weight parameter


# RotaryEmbedding is a technique that rotates the input by a learnable angle
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, device=None): # dim is the number of features/dimensions/embeddings in the input, base is a base number for the frequency, device is the device to store the buffer
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim)) # compute the inverse frequency
        self.register_buffer("inv_freq", inv_freq) # register the inverse frequency as a buffer

    def forward(self, x, seq_len):
        seq_len = seq_len.to(x.device) # convert seq_len to the device of the input 
        t = torch.arange(seq_len, device=x.device) # create a tensor of the sequence length
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # compute the frequency by taking the dot product of the sequence length and the inverse frequency
        emb = torch.cat((freqs, freqs), dim=-1) # concatenate the frequency with itself
        return emb

class LlamaMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False) # create the gate projection layer with the input dimension and the hidden dimension
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False) # create the up projection layer with the input dimension and the hidden dimension
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False) # create the down projection layer with the hidden dimension and the output dimension
        self.act_fn = nn.SiLU() # create the activation function

    def forward(self, x):
        gated = self.gate_proj(x) # apply the gate projection to the input
        hidden = self.up_proj(x) # apply the up projection to the input
        return self.down_proj(self.act_fn(gated * hidden)) # apply the activation function to the gated and hidden values and then apply the down projection
    
class LlamaAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, dim = x.size() # [batch_size, seq_len, dim] -> [4, 128, 576]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)


        # Split heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, num_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)

        # Combine heads
        context = context.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.o_proj(context)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads):
        super().__init__()
        self.self_attn = LlamaAttention(dim, num_heads)
        self.mlp = LlamaMLP(dim, hidden_dim)
        self.input_layernorm = LlamaRMSNorm(dim)
        self.post_attention_layernorm = LlamaRMSNorm(dim)

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = x + residual

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + residual
        return x


class LlamaModel(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, hidden_dim, num_heads):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = LlamaRMSNorm(dim)
        self.rotary_emb = LlamaRotaryEmbedding(dim)

    def forward(self, x):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class LlamaForCausalLM(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, hidden_dim, num_heads):
        super().__init__()
        self.model = LlamaModel(vocab_size, dim, num_layers, hidden_dim, num_heads)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.model(x)
        return self.lm_head(x)

def get_model(tokenizer):
    vocab_size = tokenizer.vocab_size  # Use actual tokenizer vocab size
    return LlamaForCausalLM(
        vocab_size=vocab_size,
        dim=576,
        num_layers=30,
        hidden_dim=1536,
        num_heads=8
    )

# model = get_model()
# print(model)

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_scheduler
from torch.optim import AdamW
#import wandb
import os
#from model import get_model

#wandb.init(project="smollm-training", name="llama-smollm-corpus")

BATCH_SIZE = 8
SEQ_LEN = 256
LEARNING_RATE = 1e-4
EPOCHS = 5
WARMUP_STEPS = 1000
GRADIENT_CLIP_VAL = 1.0
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def generate_text(
    model, tokenizer, prompt, max_length=50, temperature=0.7, top_k=50, device=DEVICE
):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature

            # Apply top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)

            # Sample from the filtered distribution
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[0, next_token_idx[0]]

            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    model.train()
    return generated_text


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "loss": loss,
            "step": step,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler):
    if os.path.exists(path):
        # path = './checkpoints/checkpoint_step_5000.pt'
        # print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"], checkpoint["step"]
    return 0, 0


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
if tokenizer.pad_token is None:
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.resize_token_embeddings(len(tokenizer))

dataset = load_dataset(
    "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True, split="train"
)


def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=SEQ_LEN, padding="max_length"
    )


tokenized_dataset = dataset.map(tokenize_function, batched=True)


def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor(
        [item["attention_mask"] for item in batch], dtype=torch.long
    )
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


train_loader = DataLoader(
    tokenized_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn
)

# Initialize model, optimizer, and scheduler
model = get_model(tokenizer)
model.to(DEVICE)

# Print model parameters
# total_params = count_parameters(model)
# print(f"\nModel Statistics:")
# print(f"Total Parameters: {total_params:,}")
# print(f"Model Size: {total_params * 4 / (1024 * 1024):.2f} MB")  # Assuming float32 (4 bytes)
# print(f"Device: {DEVICE}")
# print(f"Batch Size: {BATCH_SIZE}")
# print(f"Sequence Length: {SEQ_LEN}")
# print(f"Learning Rate: {LEARNING_RATE}")
# print("-" * 50 + "\n")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    total_steps=10000,
    pct_start=0.1,
    anneal_strategy="cos",
    cycle_momentum=False,
)

# Load checkpoint if exists
start_epoch, global_step = load_checkpoint(
    f"{CHECKPOINT_DIR}/latest_checkpoint.pt", model, optimizer, lr_scheduler
)

# Sample prompts for evaluation
sample_prompts = [
    "The future of artificial intelligence",
    "The most important thing in life",
    "The best way to learn programming",
]

model.train()
try:
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        for step, batch in enumerate(train_loader, start=global_step):
            # Move batch to device
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass
            outputs = model(input_ids)
            logits = outputs.view(-1, tokenizer.vocab_size)

            # Calculate loss with label smoothing
            loss = torch.nn.functional.cross_entropy(
                logits, labels.view(-1), label_smoothing=0.1  # Add label smoothing
            )

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VAL)
            optimizer.step()
            lr_scheduler.step()

            # Logging
            if step % 10 == 0:
                print(
                    f"Step {step}, Loss: {loss.item():.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                )
                #wandb.log(
                #    {
                #        "loss": loss.item(),
                #        "lr": lr_scheduler.get_last_lr()[0],
                #        "step": step,
                #        "epoch": epoch,
                #    }
                #)

            # Save checkpoint every 100 steps
            #if step % 100 == 0:
            #    save_checkpoint(
            #        model,
            #        optimizer,
            #        lr_scheduler,
            #        epoch,
            #        step,
            #        loss.item(),
            #        f"{CHECKPOINT_DIR}/latest_checkpoint.pt",
            #    )

                # Also save numbered checkpoint every 1000 steps
                if step % 1000 == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step,
                        loss.item(),
                        f"{CHECKPOINT_DIR}/checkpoint_step.pt",
                    )

            # Generate sample text every 500 steps with different temperatures
            if step % 500 == 0:
                print("\n=== Generating Sample Texts ===")
                for temp in [0.7, 1.0]:  # Try different temperatures
                    for prompt in sample_prompts:
                        generated = generate_text(
                            model,
                            tokenizer,
                            prompt,
                            temperature=temp,
                            max_length=100,  # Increased max length
                        )
                        print(f"\nPrompt: {prompt}")
                        print(f"Temperature: {temp}")
                        print(f"Generated: {generated}")
                        #wandb.log(
                        #    {
                        #        f"generated_text_temp_{temp}_{prompt[:20]}": wandb.Html(
                        #            generated
                        #        )
                        #    }
                        #)
                print("\n=== End of Samples ===\n")
                model.train()

        # Save epoch checkpoint
        #save_checkpoint(
        #    model,
        #    optimizer,
        #    lr_scheduler,
        #    epoch,
        #    step,
        #    loss.item(),
        #    f"{CHECKPOINT_DIR}/checkpoint_epoch.pt",
        #)

except KeyboardInterrupt:
    print("\nTraining interrupted! Saving checkpoint...")
    save_checkpoint(
        model,
        optimizer,
        lr_scheduler,
        epoch,
        step,
        loss.item(),
        f"{CHECKPOINT_DIR}/interrupted_checkpoint.pt",
    )

print("Training complete!")
#wandb.finish()
