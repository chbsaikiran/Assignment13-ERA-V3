import torch
from torch import Tensor,nn
from typing import Dict, List, Optional, Union

class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False):
        if prenorm and residual is not None:
            input = input + residual.float() if residual_in_fp32 else input + residual

        variance = input.pow(2).mean(-1, keepdim=True)
        input = input / torch.sqrt(variance + self.eps)
        output = self.weight * input

        if dropout_p > 0.0:
            dropout_mask = torch.bernoulli(torch.full_like(output, 1 - dropout_p)) / (1 - dropout_p)
            output = output * dropout_mask
            if return_dropout_mask:
                return output, dropout_mask

        return output

class TritonLayerNorm(nn.LayerNorm):
    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):
        from flash_attn.ops.triton.layer_norm import layer_norm_fn

        return layer_norm_fn(
            input,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=False,
            return_dropout_mask=return_dropout_mask,
        )

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, end: int, theta: float = 500000.0):
        super().__init__()
        self.dim = dim
        self.end = end
        self.theta = theta
        self.init_rotary_embeddings()

    def init_rotary_embeddings(self):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Optional[torch.LongTensor]):
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[1], -1, 1)  # seq_len
        position_ids_expanded = position_ids[:, None, :].float().transpose(0, 2)  # shape correction
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def rotate_half(self, x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=2):
        cos = cos.unsqueeze(unsqueeze_dim)  # Add an extra dimension
        sin = sin.unsqueeze(unsqueeze_dim)
        
        # Expand along the batch_size and n_heads dimensions
        cos = cos.expand(q.shape[0], q.shape[1], q.shape[2], -1)
        sin = sin.expand(q.shape[0], q.shape[1], q.shape[2], -1)
        
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.n_heads = config.num_attention_heads
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_model = config.hidden_size
        #self.is_using_mup = config.is_using_mup

        self.qkv_proj = nn.Linear(self.d_model, 3 * self.n_heads * self.d_qk, bias=False)
        self.rotary_embedding = LlamaRotaryEmbedding(dim=self.d_qk, end=config.max_position_embeddings, theta=config.rope_theta)
        self.o_proj = nn.Linear(self.n_heads * self.d_qk, self.d_model, bias=False)

    def forward(self, hidden_states, sequence_mask, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(0, hidden_states.size(0), dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        qkv_states = self.qkv_proj(hidden_states)
        q_length, batch_size, _ = hidden_states.shape  # Add this line
        
        query_states, key_states, value_states = torch.chunk(qkv_states, 3, dim=-1)
        query_states = query_states.view(q_length, batch_size, self.n_heads, self.d_qk).transpose(0, 1)
        key_states = key_states.view(q_length, batch_size, self.n_heads, self.d_qk).transpose(0, 1)
        value_states = value_states.view(q_length, batch_size, self.n_heads, self.d_qk).transpose(0, 1)
        
        cos, sin = self.rotary_embedding(value_states, position_ids)
        query_states, key_states = self.rotary_embedding.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / (self.d_qk ** 0.5)
        #print(attn_scores.shape)
        #print(sequence_mask.shape)
        #attn_scores = attn_scores.masked_fill(sequence_mask[:, None, None, :] == 0, float('-inf'))
        attn_scores = attn_scores.permute(1, 2, 3, 0)  # (8, 8, 8, 256)
        attn_scores = attn_scores.masked_fill(sequence_mask[:, None, None, :].expand(attn_scores.shape) == 0, float('-inf'))
        attn_scores = attn_scores.permute(3, 0, 1, 2)  # Restore original shape


        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
        attention_output = torch.matmul(attn_probs, value_states)

        attention_output = attention_output.transpose(0, 1).contiguous().view(q_length, batch_size, self.n_heads * self.d_qk)
        output = self.o_proj(attention_output)

        return {"hidden_states": output, "sequence_mask": sequence_mask}

class ColumnLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


class RowLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)

class GLUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.functional.silu

    def forward(self, merged_states: torch.Tensor):
        gate_states, up_states = torch.split(merged_states, merged_states.shape[-1] // 2, dim=-1)
        return self.act(gate_states) * up_states


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gate_up_proj = ColumnLinear(config.hidden_size, 2*config.intermediate_size, bias=False)
        self.down_proj = RowLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.split_silu_mul = GLUActivation()

    def forward(self, hidden_states):
        merged_states = self.gate_up_proj(hidden_states)
        hidden_states = self.down_proj(self.split_silu_mul(merged_states))
        return {"hidden_states": hidden_states}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.MLP = MLP(config)

    def forward(self, hidden_states, attention_mask, position_ids=None):
        normed_hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(normed_hidden_states, attention_mask, position_ids)["hidden_states"]
        hidden_states = hidden_states + attn_output
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.MLP(hidden_states)["hidden_states"]
        hidden_states = hidden_states + mlp_output
        return hidden_states

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_layers)])
        self.final_layer_norm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnLinear(config.hidden_size, config.vocab_size, bias=False)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_layers)])

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

class CrossEntropyLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
        log_probs = logits - torch.log(sum_exp_logits)

        loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        ctx.save_for_backward(exp_logits / sum_exp_logits, target)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target = ctx.saved_tensors
        grad_input = softmax.clone()
        grad_input.scatter_add_(dim=-1, index=target.unsqueeze(-1), src=-1.0)
        grad_input.mul_(grad_output.unsqueeze(-1))
        return grad_input, None


def cross_entropy_loss(logits, target, dtype: torch.dtype = None):
    if dtype is not None:
        logits = logits.to(dtype=dtype)
    return CrossEntropyLossFunction.apply(logits, target)


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, label_ids: torch.Tensor, label_mask: torch.Tensor) -> dict:
        loss = cross_entropy_loss(logits, label_ids.transpose(0, 1).contiguous(), dtype=torch.float).transpose(0, 1)
        loss = (loss * label_mask).sum() / label_mask.sum()
        return {"loss": loss}
