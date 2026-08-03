import copy
import unittest

import torch
from torch import nn

from cosyvoice.utils.lora import (
    LoRALinear,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
)


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.ff = nn.Linear(4, 4)
        self.llm_decoder = nn.Linear(4, 4)
        self.speech_embedding = nn.Embedding(8, 4)

    def forward(self, x):
        return self.llm_decoder(self.q_proj(x) + self.ff(x))


class NativeLoRATest(unittest.TestCase):
    def test_injection_freezes_base_and_keeps_adaptation_heads_trainable(self):
        model = TinyLanguageModel()
        count = inject_lora(model, rank=2, alpha=4, dropout=0.0)

        self.assertEqual(count, 1)
        self.assertIsInstance(model.q_proj, LoRALinear)
        self.assertFalse(model.q_proj.base.weight.requires_grad)
        self.assertTrue(model.q_proj.lora_A.requires_grad)
        self.assertTrue(model.llm_decoder.weight.requires_grad)
        self.assertTrue(model.speech_embedding.weight.requires_grad)
        self.assertFalse(model.ff.weight.requires_grad)

    def test_initial_adapter_is_a_noop_and_state_round_trips(self):
        model = TinyLanguageModel()
        baseline = copy.deepcopy(model)
        inject_lora(model, rank=2, alpha=4, dropout=0.0)
        sample = torch.randn(3, 4)

        torch.testing.assert_close(model(sample), baseline(sample))
        state = lora_state_dict(model)
        restored = TinyLanguageModel()
        inject_lora(restored, rank=2, alpha=4, dropout=0.0)
        load_lora_state_dict(restored, state)
        for name, value in state.items():
            torch.testing.assert_close(restored.state_dict()[name], value)


if __name__ == "__main__":
    unittest.main()
