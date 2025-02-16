import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import math
import os
from numpy import arange

# Import the LlamaModel from model_manual.py
#from model_manual import LlamaModel

# Function to generate text from the model
def generate_text(model, input_text, vocab, id_to_token, device, max_length=50, temperature=0.7):
    model.eval()
    input_ids = torch.tensor([[vocab.get(token, vocab['<|endoftext|>']) for token in input_text.split()]], dtype=torch.long).to(device)
    generated_tokens = input_ids.tolist()[0]

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)[:, -1, :]
            logits = logits / temperature
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).squeeze().item()

            if next_token == vocab['<|endoftext|>']:
                break

            generated_tokens.append(next_token)
            input_ids = torch.tensor([generated_tokens], dtype=torch.long).to(device)

    return ' '.join(id_to_token[token] for token in generated_tokens if token in id_to_token)


class DataloaderLite:
    def __init__(self, file_path, seq_len, batch_size):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer", add_prefix_space=True)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.tokenizer.resize_token_embeddings(len(self.tokenizer))

        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.epochs = len(self.text) // (self.seq_len * self.batch_size)
        self.current_position = 0
        self.padded_chunks = []
        self.max_len = 0
        
    def get_max_length(self):
        return self.max_len

    def next_batch(self):

        self.chunks = [self.text[(self.current_position + i):(self.current_position + i + self.seq_len)] for i in range(0, self.seq_len*self.batch_size, self.seq_len)]
        self.current_position = self.current_position + self.seq_len*self.batch_size
        if self.current_position + (self.seq_len*self.batch_size + 1) > len(self.text):
            self.current_position = 0
        self.encoded_chunks = [self.tokenizer(chunk, return_tensors='pt', truncation=True, max_length=self.seq_len) for chunk in self.chunks]

        self.max_len = max(chunk['input_ids'].shape[1] for chunk in self.encoded_chunks)
        self.padded_chunks = []
        for chunk in self.encoded_chunks:
            input_ids = torch.cat((chunk['input_ids'], torch.full((1, self.max_len - chunk['input_ids'].shape[1]), self.tokenizer.pad_token_id, dtype=torch.long)), dim=1)
            attention_mask = torch.cat((chunk['attention_mask'], torch.zeros((1, self.max_len - chunk['attention_mask'].shape[1]), dtype=torch.long)), dim=1)
            self.padded_chunks.append(input_ids.squeeze(0))
            self.padded_chunks.append(attention_mask.squeeze(0))
        return self.padded_chunks

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_masks = torch.stack([item[1] for item in batch])
    return input_ids, attention_masks

def train_model(config, train_file, steps, output_dir):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.resize_token_embeddings(len(tokenizer))

    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    dataloader = DataloaderLite(train_file, SEQ_LEN, BATCH_SIZE)
    #padded_chunks = dataloader.next_batch()
    #print(padded_chunks[0])
    #print(padded_chunks[1])
    #print(padded_chunks[2])
    #print(padded_chunks[3])

    model = LlamaModel(config).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    progress_bar = tqdm(range(steps), desc="Training")
    
    PROMPT = "inventory to particularise their abundance"
    
    step = 0
    while step < steps:
        input_tokens = dataloader.next_batch()
        max_len = dataloader.get_max_length()
        input_list = []
        attenttion_list = []
        for i in arange(BATCH_SIZE):
            input_list.append(input_tokens[2*i])
            attenttion_list.append(input_tokens[2*i+1])

        input_ids = torch.stack(input_list)
        inputs = input_ids[:, :-1]  # Keep batch dim and remove last token
        targets = input_ids[:, 1:]  # Keep batch dim and remove first token
        attention_mask = torch.stack(attenttion_list)
        attentions = attention_mask[:, :-1]
        device = next(model.parameters()).device
        inputs, attentions = inputs.to(device), attentions.to(device)
        positions = torch.arange(0, inputs.size(1), dtype=torch.long).unsqueeze(0).repeat(inputs.size(0), 1).to(device)
        #print(inputs.shape)
        #print(attentions.shape)
        #print(positions.shape)
    
        optimizer.zero_grad()
        logits = model(inputs, attentions,position_ids=positions)
    
        labels = targets.to(device)  # Move labels to the same device as the model and inputs
        # Create a mask based on counts
        mask = attentions.bool()  # The attention mask already has the correct shape
        logits_masked = logits[mask].contiguous().view(-1, config.vocab_size)
        labels_masked = labels[mask].contiguous().view(-1)
        
        loss = loss_fn(logits_masked, labels_masked)
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item())
    
        #if step % 10 == 0:
        #    torch.save({
        #        'step': step,
        #        'model_state_dict': model.state_dict(),
        #        'optimizer_state_dict': optimizer.state_dict(),
        #        'scheduler_state_dict': scheduler.state_dict()
        #    }, os.path.join(output_dir, f'checkpoint_{step}.pt'))
        #    #generated_text = generate_text(model, PROMPT, vocab, id_to_token, model.device)
        #    #print(f"\nGenerated text at step {step}: {generated_text}\n")
    
        step += 1
        if step >= steps:
            break

class Config:
    pass

if __name__ == "__main__":
    config = Config()
    config.vocab_size = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer").vocab_size
    config.num_layers = 30
    config.hidden_size = 576
    config.num_attention_heads = 8
    config.rms_norm_eps = 1e-5
    config.max_position_embeddings = 2048
    config.rope_theta = 500000.0
    config.hidden_act = False
    config.intermediate_size = 1536

    BATCH_SIZE = 8
    SEQ_LEN = 256

    train_model(config, '/kaggle/input/assign13-era-v3-dataset/input.txt', 1000, './output')
