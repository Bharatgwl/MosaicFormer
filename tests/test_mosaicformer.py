import unittest

import torch

from baseline import GPTModel, find_parameter_matched_baseline_config
from model import (
    LocalCausalAttention,
    MemoryCompressor,
    MosaicFormerModel,
    MosaicTransformerBlock,
    build_local_causal_mask,
    make_mosaic_config,
    override_config,
)


def tiny_config(vocab_size=64, use_memory=True, use_gate=True):
    cfg = make_mosaic_config(vocab_size, use_memory=use_memory, use_gate=use_gate)
    return override_config(
        cfg,
        context_length=16,
        chunk_size=4,
        window_size=4,
        emb_dim=32,
        n_heads=4,
        n_layers=2,
        num_memory_tokens=2,
        max_memory_chunks=4,
        dropout=0.0,
    )


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


class MosaicFormerTests(unittest.TestCase):
    def test_local_causal_mask_has_no_future_tokens(self):
        mask = build_local_causal_mask(6, window_size=3, device=torch.device("cpu"))
        self.assertFalse(mask[0, 1].item())
        self.assertFalse(mask[2, 3].item())
        self.assertTrue(mask[3, 3].item())
        self.assertTrue(mask[3, 1].item())
        self.assertFalse(mask[3, 0].item())

    def test_memory_shapes_are_correct(self):
        cfg = tiny_config()
        x = torch.randn(2, cfg["chunk_size"], cfg["emb_dim"])

        local_attention = LocalCausalAttention(cfg)
        local_output = local_attention(x)
        self.assertEqual(local_output.shape, x.shape)

        compressor = MemoryCompressor(cfg)
        memory = compressor(x)
        self.assertEqual(memory.shape, (2, cfg["num_memory_tokens"], cfg["emb_dim"]))

        block = MosaicTransformerBlock(cfg)
        block_output, new_memory = block(x, memory)
        self.assertEqual(block_output.shape, x.shape)
        self.assertEqual(new_memory.shape, (2, cfg["num_memory_tokens"], cfg["emb_dim"]))

        model = MosaicFormerModel(cfg)
        input_ids = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]))
        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, cfg["context_length"], cfg["vocab_size"]))

    def test_local_only_removes_memory_parameters(self):
        full_model = MosaicFormerModel(tiny_config(use_memory=True, use_gate=True))
        local_model = MosaicFormerModel(tiny_config(use_memory=False, use_gate=False))

        local_parameter_names = [name for name, _ in local_model.named_parameters()]
        self.assertFalse(any("memory_" in name for name in local_parameter_names))
        self.assertFalse(any("gated_fusion" in name for name in local_parameter_names))
        self.assertLess(count_parameters(local_model), count_parameters(full_model))

    def test_model_does_not_leak_future_tokens(self):
        torch.manual_seed(123)
        cfg = tiny_config()
        model = MosaicFormerModel(cfg)
        model.eval()

        tokens_a = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]))
        tokens_b = tokens_a.clone()
        first_changed_position = 10
        tokens_b[:, first_changed_position:] = torch.randint(
            0,
            cfg["vocab_size"],
            (1, cfg["context_length"] - first_changed_position),
        )

        with torch.no_grad():
            logits_a = model(tokens_a)
            logits_b = model(tokens_b)

        self.assertTrue(
            torch.allclose(
                logits_a[:, :first_changed_position, :],
                logits_b[:, :first_changed_position, :],
                atol=1e-5,
            )
        )

    def test_parameter_matched_baseline_is_close_to_mosaicformer(self):
        cfg = tiny_config(vocab_size=128)
        mosaic_params = count_parameters(MosaicFormerModel(cfg))
        baseline_cfg = find_parameter_matched_baseline_config(
            cfg,
            target_params=mosaic_params,
            max_emb_dim=128,
        )
        baseline_params = count_parameters(GPTModel(baseline_cfg))
        gap_ratio = abs(baseline_params - mosaic_params) / mosaic_params
        self.assertLess(gap_ratio, 0.05)


if __name__ == "__main__":
    unittest.main()
