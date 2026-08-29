"""GPT baseline used for MosaicFormer comparisons.

Copyright (c) 2026 Bharat Gusaiwal. All rights reserved.
"""

import torch
import torch.nn as nn

from model import Feedforward, LayerNorm


class MultiheadAttention(nn.Module):
    """Full causal self-attention baseline, matching the GPT notebook style."""
    def __init__(self, cfg):
        super().__init__()
        assert cfg["emb_dim"] % cfg["n_heads"] == 0

        self.d_out = cfg["emb_dim"]
        self.num_heads = cfg["n_heads"]
        self.head_dim = self.d_out // self.num_heads
        self.w_query = nn.Linear(cfg["emb_dim"], self.d_out, bias=cfg["qkv_bias"])
        self.w_key = nn.Linear(cfg["emb_dim"], self.d_out, bias=cfg["qkv_bias"])
        self.w_value = nn.Linear(cfg["emb_dim"], self.d_out, bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(self.d_out, self.d_out)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg["context_length"], cfg["context_length"]))
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        keys = self.w_key(x)
        values = self.w_value(x)
        queries = self.w_query(x)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1, 2).contiguous()
        context_vector = context_vector.view(batch_size, num_tokens, self.d_out)
        return self.out_proj(context_vector)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiheadAttention(cfg)
        self.ff = Feedforward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.Lnorm = LayerNorm(cfg["emb_dim"])
        self.output_layer = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, input_index):
        batch_size, num_tokens = input_index.shape
        tok_emb = self.token_embedding(input_index)
        pos_emb = self.position_embedding(torch.arange(num_tokens, device=input_index.device))
        x = self.dropout(tok_emb + pos_emb)
        x = self.transformer_blocks(x)
        x = self.Lnorm(x)
        return self.output_layer(x)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_gpt_parameter_count(cfg):
    """Estimate GPTModel parameters without constructing the model."""
    vocab_size = cfg["vocab_size"]
    context_length = cfg["context_length"]
    emb_dim = cfg["emb_dim"]
    n_layers = cfg["n_layers"]
    qkv_bias = cfg["qkv_bias"]

    embeddings = (vocab_size + context_length) * emb_dim
    output_head = vocab_size * emb_dim
    final_norm = 2 * emb_dim

    qkv = 3 * emb_dim * emb_dim
    if qkv_bias:
        qkv += 3 * emb_dim
    attention_out = emb_dim * emb_dim + emb_dim
    feedforward = (emb_dim * 4 * emb_dim) + (4 * emb_dim) + (4 * emb_dim * emb_dim) + emb_dim
    block_norms = 4 * emb_dim
    block = qkv + attention_out + feedforward + block_norms
    return embeddings + output_head + final_norm + (n_layers * block)


def make_baseline_config(mosaic_cfg, emb_dim=None, n_layers=None):
    """Build a GPT baseline config from the shared experiment settings."""
    cfg = mosaic_cfg.copy()
    cfg["emb_dim"] = emb_dim if emb_dim is not None else cfg["emb_dim"]
    cfg["n_layers"] = n_layers if n_layers is not None else cfg["n_layers"]
    if cfg["emb_dim"] % cfg["n_heads"] != 0:
        raise ValueError("baseline emb_dim must be divisible by n_heads")
    return cfg


def find_parameter_matched_baseline_config(mosaic_cfg, target_params, max_emb_dim=512):
    """Find a GPT config with parameter count close to a MosaicFormer model.

    This keeps context length, number of heads, and optimizer-facing settings
    fixed, then searches embedding sizes that remain divisible by n_heads.
    """
    n_heads = mosaic_cfg["n_heads"]
    start_dim = max(n_heads, mosaic_cfg["emb_dim"] // 2)
    candidates = []

    for emb_dim in range(start_dim, max_emb_dim + 1):
        if emb_dim % n_heads != 0:
            continue
        cfg = make_baseline_config(mosaic_cfg, emb_dim=emb_dim)
        params = estimate_gpt_parameter_count(cfg)
        gap = abs(params - target_params)
        candidates.append((gap, params, cfg))

    if not candidates:
        raise ValueError("No valid parameter-matched baseline candidates found")

    _, estimated_params, cfg = min(candidates, key=lambda item: item[0])
    params = count_parameters(GPTModel(cfg))
    cfg["matched_to_params"] = target_params
    cfg["matched_param_count"] = params
    cfg["matched_param_count_estimate"] = estimated_params
    cfg["matched_param_gap_ratio"] = abs(params - target_params) / target_params
    return cfg
