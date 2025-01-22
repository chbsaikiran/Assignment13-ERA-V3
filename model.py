import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        
        # Load model configuration from YAML
        self.hidden_size = config['hidden_size']
        self.vocab_size = config['vocab_size']
        self.num_hidden_layers = config['num_hidden_layers']
        self.num_attention_heads = config['num_attention_heads']
        self.intermediate_size = config['intermediate_size']
        self.max_position_embeddings = config['max_position_embeddings']
        self.rms_norm_eps = config['rms_norm_eps']
        self.tie_word_embeddings = config['tie_word_embeddings']

        # Embedding layers
        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(self.hidden_size, self.num_attention_heads, self.intermediate_size, self.rms_norm_eps)
            for _ in range(self.num_hidden_layers)
        ])

        # Output layer
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)
        if self.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, input_ids):
        # Input embeddings
        seq_length = input_ids.size(1)
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        x = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)

        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final linear layer
        logits = self.lm_head(x)
        return logits

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, rms_norm_eps):
        super(TransformerLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.head_dim = hidden_size // num_attention_heads

        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"

        # Self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads, batch_first=True)

        # Feedforward layers
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.activation = F.silu  # Activation function (SiLU)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

        # Normalization
        self.norm1 = nn.LayerNorm(hidden_size, eps=rms_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, x):
        # Self-attention block
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output  # Residual connection
        x = self.norm1(x)

        # Feedforward block
        ff_output = self.linear2(self.activation(self.linear1(x)))
        x = x + ff_output  # Residual connection
        x = self.norm2(x)

        return x

# Load model configuration from YAML-like dictionary (parsed from YAML)
config = {
    "hidden_size": 576,
    "vocab_size": 49152,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "intermediate_size": 1536,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": True
}

# Initialize the model
#model = TransformerModel(config)
#
# Example usage
#input_ids = torch.randint(0, config['vocab_size'], (1, 128))  # Batch size 1, sequence length 128
#output = model(input_ids)
#print(output.shape)  # Should be (1, 128, vocab_size)
