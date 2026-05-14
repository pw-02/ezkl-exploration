import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Config
# -------------------------

@dataclass
class TinyPythiaConfig:
    block_size: int = 64
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = False



# -------------------------
# Layers
# -------------------------

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
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


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.query_key_value = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
        )

        self.dense = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.block_size,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.query_key_value(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k = self.rope(q, k)

        att = q @ k.transpose(-2, -1)
        att = att * (1.0 / math.sqrt(self.head_dim))

        att = att.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            -1e4,
        )

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dense(y)
        y = self.resid_dropout(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense_h_to_4h = nn.Linear(
            config.n_embd,
            4 * config.n_embd,
            bias=config.bias,
        )

        self.dense_4h_to_h = nn.Linear(
            4 * config.n_embd,
            config.n_embd,
            bias=config.bias,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.dense_h_to_4h(x)
        x = F.gelu(x)
        x = self.dense_4h_to_h(x)
        x = self.dropout(x)
        return x


class PythiaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_layernorm = LayerNorm(config.n_embd, bias=config.bias)
        self.post_attention_layernorm = LayerNorm(config.n_embd, bias=config.bias)

        self.attention = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attention(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# -------------------------
# Model
# -------------------------

class TinyPythia(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.n_embd)

        self.layers = nn.ModuleList([
            PythiaBlock(config)
            for _ in range(config.n_layer)
        ])

        self.final_layer_norm = LayerNorm(config.n_embd, bias=config.bias)

        self.embed_out = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        self.embed_out.weight = self.embed_in.weight

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
        B, T = input_ids.shape

        assert T <= self.config.block_size

        x = self.embed_in(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.final_layer_norm(x)
        logits = self.embed_out(x)

        return logits


# -------------------------
# Export
# -------------------------

if __name__ == "__main__":
    OUT_DIR = "experiments/models/pythia"
    os.makedirs(OUT_DIR, exist_ok=True)

    n_layer = 4
    n_embd = 64

    config = TinyPythiaConfig(
        block_size=64,
        vocab_size=50304,
        n_layer=6,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=True,
    )

    model = TinyPythia(config)
    model.eval()

    shape = [1, config.block_size]

    x = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, config.block_size),
        dtype=torch.long,
    )

    with torch.no_grad():
        torch_out = model(x)

    onnx_path = os.path.join(
        OUT_DIR,
        f"tiny_pythia_{n_layer}_layers_{n_embd}_embd.onnx",
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

    d = x.detach().numpy().reshape([-1]).tolist()

    data = dict(
        input_shapes=[shape],
        input_data=[d],
        output_data=[
            torch_out.detach().numpy().reshape([-1]).tolist()
        ],
    )

    input_json_path = os.path.join(OUT_DIR, "input.json")

    with open(input_json_path, "w") as f:
        json.dump(data, f)

    print("Saved:")
    print(f"  ONNX:  {onnx_path}")
    print(f"  Input: {input_json_path}")