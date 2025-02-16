import torch
from torch import Tensor,nn
from typing import Dict, List, Optional, Union

import dataclasses


@dataclasses.dataclass
class TensorPointer:
    """Dataclass specifying from which rank we need to query a tensor from in order to access data"""

    # Needed to understand from which rank to get the tensor
    # TODO @thomasw21: Maybe add which group it belongs to as well? Typically this is highly correlated to `p2p.pg`
    group_rank: int
    # TODO @thomasw21: Maybe add a tag (torch.distributed.send/recv allow for tagging)

class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(
        self, input, residual=None, dropout_p=0.0, prenorm=False, residual_in_fp32=False, return_dropout_mask=False
    ):
        from flash_attn.ops.triton.layer_norm import layer_norm_fn

        return layer_norm_fn(
            input,
            self.weight,
            None,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True,
            return_dropout_mask=return_dropout_mask,
        )

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, end: int, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.end = end
        self.theta = theta
        self.register_buffer(
            "freqs_cis",
            self._compute_freqs(end, dim, theta),
            persistent=False,
        )

    def _compute_freqs(self, end, dim, theta):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs)
        complex_freqs = torch.polar(torch.ones_like(freqs), freqs)
        return torch.view_as_real(complex_freqs)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.LongTensor]):
        batch_size, seq_length, num_heads, inner_dim = x.shape
        dtype = x.dtype
        x = x.view(batch_size, seq_length, num_heads, inner_dim // 2, 2)
        complex_x = torch.view_as_complex(x.float())

        if position_ids is None:
            freqs_cis = self.freqs_cis[:seq_length]
        else:
            freqs_cis = self.freqs_cis[position_ids]

        complex_freqs = torch.view_as_complex(freqs_cis)
        x_out = torch.view_as_real(complex_x * complex_freqs).view(batch_size, seq_length, num_heads, inner_dim)
        return x_out.type(dtype)

class CoreAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0, "Hidden size must be divisible by number of attention heads."
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        #self.is_using_mup = config.is_using_mup

    def forward(self, query_states, key_states, value_states, q_sequence_mask, kv_sequence_mask):
        cu_seqlens_q = torch.cumsum(q_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32)
        cu_seqlens_k = torch.cumsum(kv_sequence_mask.sum(-1, dtype=torch.int32), dim=0, dtype=torch.int32)
        causal = q_sequence_mask.shape[1] != 1
        #softmax_scale = 1 / query_states.shape[-1] if self.is_using_mup else None

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=None, dropout_p=0.0
        )
        return attn_output

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding
        
        self.n_q_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else config.num_attention_heads
        self.n_repeats = config.num_attention_heads // self.n_kv_heads
        self.is_gqa = config.num_attention_heads != self.n_kv_heads
        self.d_qk = config.hidden_size // config.num_attention_heads
        self.d_v = config.hidden_size // config.num_attention_heads
        self.d_model = config.hidden_size

        self.qkv_proj = nn.Linear(self.d_model, config.num_attention_heads * self.d_qk + 2 * self.n_kv_heads * self.d_qk, bias=False)
        self.rotary_embedding = RotaryEmbedding(dim=self.d_qk, end=config.max_position_embeddings, theta=config.rope_theta)
        self.flash_rotary_embedding = FlashRotaryEmbedding(dim=self.d_qk, base=config.rope_theta, interleaved=config.rope_interleaved)
        self.o_proj = nn.Linear(config.num_attention_heads * self.d_qk, self.d_model, bias=False)
        self.attention = CoreAttention(config)

    def forward(self, hidden_states, sequence_mask):
        qkv_states = self.qkv_proj(hidden_states)
        q_length, batch_size, _ = qkv_states.shape

        if self.is_gqa:
            query_states, key_states, value_states = torch.split(
                qkv_states,
                [self.n_q_heads * self.d_qk, self.n_kv_heads * self.d_qk, self.n_kv_heads * self.d_qk],
                dim=-1,
            )
        else:
            query_states, key_states, value_states = qkv_states.view(q_length, batch_size, 3, self.n_q_heads, self.d_qk).permute(2, 1, 0, 3, 4).contiguous()

        query_states, key_value_states = self.flash_rotary_embedding(query_states, kv=torch.stack([key_states, value_states], dim=2))
        key_states, value_states = torch.chunk(key_value_states, 2, dim=2)

        attention_output = self.attention(
            query_states=query_states.view(batch_size * q_length, self.n_q_heads, self.d_qk),
            key_states=key_states.view(batch_size * q_length, self.n_kv_heads, self.d_qk),
            value_states=value_states.view(batch_size * q_length, self.n_kv_heads, self.d_v),
            q_sequence_mask=sequence_mask,
            kv_sequence_mask=sequence_mask,
        )

        attention_output = attention_output.contiguous().view(batch_size, q_length, self.n_q_heads * self.d_v).transpose(0, 1)
        output = self.o_proj(attention_output)

        return {"hidden_states": output, "sequence_mask": sequence_mask}

class ColumnLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = x.detach()  # Detach from computation graph
        return nn.functional.linear(x, self.weight, self.bias)


class RowLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = x.detach()  # Detach from computation graph
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
        self.attn = CausalSelfAttention(config)
        self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def _core_forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
    ) -> List[Union[torch.Tensor, TensorPointer]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        output = self.attn(hidden_states=hidden_states, sequence_mask=sequence_mask)
        hidden_states = output["hidden_states"]
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)["hidden_states"]
        hidden_states = hidden_states + residual

        return hidden_states, output["sequence_mask"]

    def forward(
        self,
        hidden_states: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:


        hidden_states, sequence_mask = self._core_forward(hidden_states, sequence_mask)

        return {
            "hidden_states": hidden_states,
            "sequence_mask": sequence_mask,
        }

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        #layer_idx = config.num_layers
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_layers)])
        self.final_layer_norm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnLinear(config.hidden_size, config.vocab_size, bias=False)
        self.cast_to_fp32 = lambda: lambda x: x.float()

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        input_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
    ):
        return self.forward_with_hidden_states(input_ids=input_ids, input_mask=input_mask)[0]

    def forward_with_hidden_states(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        input_mask: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
    ):
        # Format input in `[seq_length, batch_size]` to support high TP with low batch_size
        input_ids = input_ids.transpose(0, 1)
        hidden_states = self.embed_tokens(input_ids)

        hidden_encoder_states = {
            "hidden_states": hidden_states,
            "sequence_mask": input_mask,
        }

        for encoder_block in self.layers:
            hidden_encoder_states = encoder_block(**hidden_encoder_states)

        hidden_encoder_states["hidden_states"] = self.final_layer_norm(hidden_encoder_states["hidden_states"])

        sharded_logits = self.lm_head(x=hidden_encoder_states["hidden_states"])

        fp32_sharded_logits = sharded_logits.float()

        return fp32_sharded_logits, hidden_states

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
