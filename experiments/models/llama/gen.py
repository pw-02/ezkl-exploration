import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyLlamaConfig:
    block_size: int = 64
    vocab_size: int = 65
    n_layer: int = 6
    n_embd: int = 128
    n_head: int = 4
    n_kv_head: int = 2
    intermediate_size: int = 352
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    bias: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * norm


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000.0):
        super().__init__()

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )

        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)

        emb = torch.repeat_interleave(freqs, 2, dim=-1)

        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def forward(self, q, k):
        T = q.size(-2)

        cos = self.cos[:, :, :T, :]
        sin = self.sin[:, :, :T, :]

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        return q, k


class TinyLlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(
            config.n_embd,
            config.n_head * self.head_dim,
            bias=config.bias,
        )
        self.k_proj = nn.Linear(
            config.n_embd,
            config.n_kv_head * self.head_dim,
            bias=config.bias,
        )
        self.v_proj = nn.Linear(
            config.n_embd,
            config.n_kv_head * self.head_dim,
            bias=config.bias,
        )
        self.o_proj = nn.Linear(
            config.n_head * self.head_dim,
            config.n_embd,
            bias=config.bias,
        )

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.block_size,
            base=config.rope_theta,
        )

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.block_size, config.block_size),
        )

    def repeat_kv(self, x):
        # x: [B, n_kv_head, T, head_dim]
        if self.n_rep == 1:
            return x

        B, H, T, D = x.shape
        x = x[:, :, None, :, :]
        x = x.expand(B, H, self.n_rep, T, D)
        return x.reshape(B, H * self.n_rep, T, D)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        att = q @ k.transpose(-2, -1)
        att = att * (1.0 / math.sqrt(self.head_dim))

        att = att.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            -1e4,
        )

        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.o_proj(y)

        return y


class TinyLlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gate_proj = nn.Linear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
        )
        self.up_proj = nn.Linear(
            config.n_embd,
            config.intermediate_size,
            bias=config.bias,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.n_embd,
            bias=config.bias,
        )

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        x = F.silu(gate) * up
        x = self.down_proj(x)

        return x


class TinyLlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_layernorm = RMSNorm(
            config.n_embd,
            eps=config.rms_norm_eps,
        )
        self.self_attn = TinyLlamaAttention(config)

        self.post_attention_layernorm = RMSNorm(
            config.n_embd,
            eps=config.rms_norm_eps,
        )
        self.mlp = TinyLlamaMLP(config)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TinyLlama(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.n_embd,
        )

        self.layers = nn.ModuleList([
            TinyLlamaBlock(config)
            for _ in range(config.n_layer)
        ])

        self.norm = RMSNorm(
            config.n_embd,
            eps=config.rms_norm_eps,
        )

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

        print(f"number of parameters: {self.get_num_params()}")

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


if __name__ == "__main__":
    OUT_DIR = "experiments/models/llama"
    os.makedirs(OUT_DIR, exist_ok=True)

    config = TinyLlamaConfig(
        block_size=64,
        vocab_size=65,
        n_layer=6,
        n_embd=128,
        n_head=4,
        n_kv_head=2,
        intermediate_size=352,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        bias=False,
    )

    model = TinyLlama(config)
    model.eval()

    x = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, config.block_size),
        dtype=torch.long,
    )

    with torch.no_grad():
        y = model(x)

    onnx_path = os.path.join(
        OUT_DIR,
        "tiny_llama_6_layers_128_embd.onnx",
    )

    input_json_path = os.path.join(
        OUT_DIR,
        "input.json",
    )

    torch.onnx.export(
        model,
        x,
        onnx_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        external_data=False,
    )

    data = {
        "input_shapes": [[1, config.block_size]],
        "input_data": [
            x.detach().numpy().reshape([-1]).tolist()
        ],
        "output_data": [
            y.detach().numpy().reshape([-1]).tolist()
        ],
    }

    with open(input_json_path, "w") as f:
        json.dump(data, f)

    print("Saved:")
    print(f"  ONNX:  {onnx_path}")
    print(f"  Input: {input_json_path}")