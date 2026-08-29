"""MosaicFormer model architecture.

Copyright (c) 2026 Bharat Gusaiwal. All rights reserved.
"""

import math

import torch
import torch.nn as nn


class Gelu(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class LayerNorm(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(input_dim))
        self.shift = nn.Parameter(torch.zeros(input_dim))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * x_norm + self.shift


class Feedforward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            Gelu(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


def build_local_causal_mask(num_tokens, window_size, device):
    """Token j is visible to token i when j <= i and it is in the local window."""
    positions = torch.arange(num_tokens, device=device)
    causal_mask = positions.unsqueeze(0) <= positions.unsqueeze(1)
    local_mask = positions.unsqueeze(0) > (positions.unsqueeze(1) - window_size)
    return causal_mask & local_mask


class LocalCausalAttention(nn.Module):
    """The detailed short-term pathway of MosaicFormer."""
    def __init__(self, cfg):
        super().__init__()
        assert cfg["emb_dim"] % cfg["n_heads"] == 0

        self.num_heads = cfg["n_heads"]
        self.head_dim = cfg["emb_dim"] // cfg["n_heads"]
        self.window_size = cfg["window_size"]
        self.emb_dim = cfg["emb_dim"]

        self.qkv = nn.Linear(cfg["emb_dim"], 3 * cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        batch_size, num_tokens, emb_dim = x.shape

        queries, keys, values = self.qkv(x).chunk(3, dim=-1)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)
        attention_scores = attention_scores / (self.head_dim ** 0.5)

        mask = build_local_causal_mask(num_tokens, self.window_size, x.device)
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values
        context_vector = context_vector.transpose(1, 2).contiguous()
        context_vector = context_vector.view(batch_size, num_tokens, emb_dim)
        return self.out_proj(context_vector)


class MemoryCompressor(nn.Module):
    """Compresses one old chunk into a few learned memory tokens.

    M = softmax(P K^T / sqrt(d)) V
    P is a small matrix of learned memory queries.
    """
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.memory_queries = nn.Parameter(
            torch.randn(cfg["num_memory_tokens"], emb_dim) * 0.02
        )
        self.w_key = nn.Linear(emb_dim, emb_dim, bias=cfg["qkv_bias"])
        self.w_value = nn.Linear(emb_dim, emb_dim, bias=cfg["qkv_bias"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, chunk_states):
        batch_size, num_tokens, emb_dim = chunk_states.shape
        queries = self.memory_queries.unsqueeze(0).expand(batch_size, -1, -1)
        keys = self.w_key(chunk_states)
        values = self.w_value(chunk_states)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores = attention_scores / (emb_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        return attention_weights @ values


class MemoryCrossAttention(nn.Module):
    """The current chunk asks questions to the compressed history."""
    def __init__(self, cfg):
        super().__init__()
        emb_dim = cfg["emb_dim"]
        self.w_query = nn.Linear(emb_dim, emb_dim, bias=cfg["qkv_bias"])
        self.w_key = nn.Linear(emb_dim, emb_dim, bias=cfg["qkv_bias"])
        self.w_value = nn.Linear(emb_dim, emb_dim, bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x, memory):
        emb_dim = x.shape[-1]
        queries = self.w_query(x)
        keys = self.w_key(memory)
        values = self.w_value(memory)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores = attention_scores / (emb_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        memory_context = attention_weights @ values
        return self.out_proj(memory_context)


class GatedFusion(nn.Module):
    """Decides whether local detail or historical memory matters more."""
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(2 * cfg["emb_dim"], cfg["emb_dim"])

    def forward(self, local_context, memory_context):
        gate_input = torch.cat([local_context, memory_context], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        return gate * local_context + (1.0 - gate) * memory_context


class MosaicTransformerBlock(nn.Module):
    """Local attention + optional memory retrieval + feedforward network."""
    def __init__(self, cfg):
        super().__init__()
        self.use_memory = cfg.get("use_memory", True)
        self.use_gate = cfg.get("use_gate", True)

        self.local_attention = LocalCausalAttention(cfg)
        if self.use_memory:
            self.memory_cross_attention = MemoryCrossAttention(cfg)
            self.memory_compressor = MemoryCompressor(cfg)
            if self.use_gate:
                self.gated_fusion = GatedFusion(cfg)
        self.feedforward = Feedforward(cfg)

        self.norm1 = LayerNorm(cfg["emb_dim"])
        if self.use_memory:
            self.norm2 = LayerNorm(cfg["emb_dim"])
        self.norm3 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x, memory=None):
        local_context = self.local_attention(self.norm1(x))

        if self.use_memory and memory is not None and memory.shape[1] > 0:
            memory_context = self.memory_cross_attention(self.norm2(x), memory)
            if self.use_gate:
                attention_output = self.gated_fusion(local_context, memory_context)
            else:
                attention_output = 0.5 * (local_context + memory_context)
        else:
            attention_output = local_context

        x = x + self.dropout(attention_output)
        x = x + self.dropout(self.feedforward(self.norm3(x)))

        new_memory = self.memory_compressor(x) if self.use_memory else None
        return x, new_memory


class MosaicFormerModel(nn.Module):
    """Decoder-only language model that processes an input in chunks.

    Each layer stores memory separately. A later chunk can retrieve the last
    `max_memory_chunks` compressed summaries from that layer.
    """
    def __init__(self, cfg):
        super().__init__()
        assert cfg["context_length"] % cfg["chunk_size"] == 0, (
            "context_length must be divisible by chunk_size"
        )
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.transformer_blocks = nn.ModuleList(
            [MosaicTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.Lnorm = LayerNorm(cfg["emb_dim"])
        self.output_layer = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # self.output_layer.weight = self.token_embedding.weight # here we have 

    def forward(self, input_index):
        batch_size, num_tokens = input_index.shape
        if num_tokens > self.cfg["context_length"]:
            raise ValueError("input length is larger than context_length")

        token_embeddings = self.token_embedding(input_index)
        positions = torch.arange(num_tokens, device=input_index.device)
        position_embeddings = self.position_embedding(positions)
        x = self.dropout(token_embeddings + position_embeddings)

        chunk_size = self.cfg["chunk_size"]
        max_memory_chunks = self.cfg["max_memory_chunks"]
        chunks = x.split(chunk_size, dim=1)
        memory_banks = [[] for _ in self.transformer_blocks]
        output_chunks = []

        for chunk in chunks:
            x = chunk
            for layer_index, block in enumerate(self.transformer_blocks):
                recent_memories = memory_banks[layer_index][-max_memory_chunks:]
                memory = torch.cat(recent_memories, dim=1) if recent_memories else None
                x, new_memory = block(x, memory)
                if new_memory is not None:
                    memory_banks[layer_index].append(new_memory)
            output_chunks.append(x)

        x = torch.cat(output_chunks, dim=1)
        x = self.Lnorm(x)
        return self.output_layer(x)


def make_mosaic_config(vocab_size, use_memory=True, use_gate=True, size="small"):
    """Build a MosaicFormer config.

    The default is the small 4GB-GPU friendly setup used in the notebooks.
    Use size="gpt2" only when you intentionally want a much larger model.
    """
    if size == "small":
        return {
            "vocab_size": vocab_size,
            "context_length": 256,
            "emb_dim": 128,
            "n_heads": 4,
            "n_layers": 4,
            "dropout": 0.1,
            "qkv_bias": False,
            "chunk_size": 64,
            "window_size": 64,
            "num_memory_tokens": 8,
            "max_memory_chunks": 4,
            "use_memory": use_memory,
            "use_gate": use_gate,
        }

    if size == "gpt2":
        return {
            "vocab_size": vocab_size,
            "context_length": 1024,
            "emb_dim": 768,
            "n_heads": 12,
            "n_layers": 12,
            "dropout": 0.1,
            "qkv_bias": False,
            "chunk_size": 128,
            "window_size": 128,
            "num_memory_tokens": 16,
            "max_memory_chunks": 8,
            "use_memory": use_memory,
            "use_gate": use_gate,
        }

    raise ValueError("size must be either 'small' or 'gpt2'")


def override_config(cfg, **updates):
    """Return a copy of cfg with updates, and check the chunking invariants."""
    new_cfg = cfg.copy()
    new_cfg.update(updates)
    if new_cfg["emb_dim"] % new_cfg["n_heads"] != 0:
        raise ValueError("emb_dim must be divisible by n_heads")
    if new_cfg["context_length"] % new_cfg["chunk_size"] != 0:
        raise ValueError("context_length must be divisible by chunk_size")
    if new_cfg["window_size"] > new_cfg["chunk_size"]:
        raise ValueError("window_size should be <= chunk_size for clear local-only ablations")
    return new_cfg


def make_gpt2_like_mosaic_config(vocab_size, use_memory=True, use_gate=True):
    """Compatibility helper for old notebooks that expected the 1024/768 config."""
    return make_mosaic_config(
        vocab_size,
        use_memory=use_memory,
        use_gate=use_gate,
        size="gpt2",
    )


def _old_large_config_reference(vocab_size, use_memory=True, use_gate=True):
    """Kept only as a readable reference while studying older notebook outputs."""
    return {
        "vocab_size": vocab_size,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "dropout": 0.1,
        "qkv_bias": False,
        "chunk_size": 128,      # 1024 / 8 = 128
        "window_size": 128,
        "num_memory_tokens": 16,
        "max_memory_chunks": 8,
        "use_memory": use_memory,
        "use_gate": use_gate,
    }
