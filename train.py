import torch
from torch.utils.data import Dataset, DataLoader
from model import TransformerModel, config
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from torchsummary import summary
from tqdm import tqdm
import time

# Function to train a new tokenizer
def train_tokenizer(data_files, vocab_size, save_path):
    """
    Args:
        data_files: List of file paths containing text data.
        vocab_size: Desired vocabulary size.
        save_path: Path to save the trained tokenizer.
    """
    tokenizer = Tokenizer(WordPiece())
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
    vocab_size=vocab_size,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    min_frequency=1  # Lower threshold to include more tokens
    )

    tokenizer.train(data_files, trainer)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    return tokenizer

# Function to save a checkpoint
def save_checkpoint(model, optimizer, scheduler, current_batch, checkpoint_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'current_batch': current_batch
    }
    torch.save(checkpoint, checkpoint_path)
    #print(f"Checkpoint saved at batch {current_batch} to {checkpoint_path}")

# Function to load a checkpoint
def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        current_batch = checkpoint['current_batch']
        print(f"Checkpoint loaded from {checkpoint_path} at batch {current_batch}")
        return current_batch
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting fresh training.")
        return 0

class DataLoaderLite:
    def __init__(self, tokenizer, data_files, B, T):
        self.B = B
        self.T = T

        self.tokenizer = tokenizer

        # Load and tokenize data
        self.data = []
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = tokenizer.encode(text).ids
                self.data.extend(tokens)

        print(f'loaded {len(self.data)} tokens')
        print(f'1 epoch = {len(self.data) // (B * T)} batches')

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.data[self.current_position: self.current_position + B * T + 1]
        
        # Convert list to tensor
        buf = torch.tensor(buf, dtype=torch.long)

        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)   # targets
        
        # advance the position in the tensor
        self.current_position += B * T

        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.data):
            self.current_position = 0

        return x, y

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# SEED
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

data_files = ["/kaggle/input/input1-txt/input.txt"]  # Replace with your text file paths
vocab_size = 49152
tokenizer_save_path = "/kaggle/working/"

# Train a new tokenizer if not already trained
if not os.path.exists(os.path.join(tokenizer_save_path, "tokenizer.json")):
    tokenizer = train_tokenizer(data_files, vocab_size, tokenizer_save_path)
else:
    tokenizer = Tokenizer.from_file(os.path.join(tokenizer_save_path, "tokenizer.json"))

batch_size = 16
tokens_per_batch = 256

train_loader = DataLoaderLite(tokenizer,data_files,B = batch_size, T = tokens_per_batch)

# Function to generate predictions
def generate_predictions(model, tokenizer, text, max_tokens=100, top_k=50):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(text).ids
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # Add batch dimension

        generated_ids = input_ids
        for _ in range(max_tokens):
            outputs = model(generated_ids)
            next_token_logits = outputs[:, -1, :]  # Get logits for the last token
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

            if top_k is not None:
                # Top-k sampling
                top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                next_token = top_k_indices.gather(
                    dim=-1, index=torch.multinomial(top_k_probs, num_samples=1)
                )
            else:
                # Greedy decoding
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # Ensure next_token has the correct shape for concatenation
            next_token = next_token.view(1, 1)  # (batch_size=1, sequence_length=1)

            # Concatenate next_token with generated_ids
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Stop if end-of-sequence token is generated
            if next_token.item() == tokenizer.token_to_id("[SEP]"):
                break

        generated_text = tokenizer.decode(generated_ids.squeeze().tolist())
    model.train()
    return generated_text



config["vocab_size"] = len(tokenizer.get_vocab())
print(f"New Vocab Size is: {config['vocab_size']}")
# Initialize model, optimizer, and scheduler
model = TransformerModel(config)
model.to(device)

def initialize_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Embedding):
        torch.nn.init.xavier_uniform_(m.weight)

model.apply(initialize_weights)

# Print the model architecture
print(model)

# Calculate and print the total number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {trainable_params}")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.65)

# Define checkpoint path
checkpoint_path = "/kaggle/working/checkpoint.pth"

# Load checkpoint if exists
start_batch = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

# Fixed text for predictions
fixed_text = "He that will give good words to thee will flatter"

# Training loop with tqdm progress bar
total_batches = 5000
checkpoint_interval = len(train_loader.data) // (batch_size * tokens_per_batch)

for i in range(start_batch, total_batches, checkpoint_interval):
    loss_list = []
    start_time = time.time()
    with tqdm(total=checkpoint_interval, desc=f"Training Batches {i}-{i+checkpoint_interval-1}", unit="batch") as pbar:
        for batch_idx in range(checkpoint_interval):
            global_batch = i + batch_idx
            if global_batch >= total_batches:
                break

            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            pred = model(x)

            # Reshape predictions and targets for loss computation
            pred = pred.view(-1, pred.size(-1))  # Flatten predictions
            y = y.view(-1)  # Flatten targets

            # Compute loss
            loss = criterion(pred, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar with current loss
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
            loss_list.append(loss.item())

            # Generate text and save checkpoint at the last batch of the interval
            if (batch_idx + 1) == checkpoint_interval or global_batch == total_batches - 1:
                generated_text = generate_predictions(model, tokenizer, fixed_text, max_tokens=100)
                print(f"\nGenerated text at batch {global_batch}: {generated_text}")
                save_checkpoint(model, optimizer, scheduler, global_batch, checkpoint_path)

    # Scheduler step
    scheduler.step()
    elapsed_time = time.time() - start_time
    epoch_loss = sum(loss_list) / len(loss_list)
    print(f"Loss: {epoch_loss:.4f}, Time: {elapsed_time:.2f}s")