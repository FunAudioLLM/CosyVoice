import unittest

import torch
import torch.nn.functional as F

from cosyvoice.hifigan.generator import ResBlock
from cosyvoice.transformer.convolution import CausalConv1d


class TestCausalConv1d(unittest.TestCase):
    def test_length_and_padding_direction(self):
        x = torch.arange(20, dtype=torch.float32).reshape(1, 1, -1)
        for kernel_size, dilation in [(2, 2), (4, 2), (4, 4), (3, 2), (4, 1), (4, 3)]:
            for side in ['left', 'right']:
                with self.subTest(kernel_size=kernel_size, dilation=dilation, side=side):
                    conv = CausalConv1d(1, 1, kernel_size, dilation=dilation, causal_type=side, bias=False)
                    with torch.no_grad():
                        conv.weight.copy_(torch.arange(1, kernel_size + 1).reshape(1, 1, -1))
                    padding = dilation * (kernel_size - 1)
                    padded = F.pad(x, (padding, 0) if side == 'left' else (0, padding))
                    expected = F.conv1d(padded, conv.weight, dilation=dilation)
                    actual = conv(x)
                    self.assertEqual(actual.shape, x.shape)
                    torch.testing.assert_close(actual, expected)

    def test_context_does_not_cross_causal_direction(self):
        x = torch.arange(20, dtype=torch.float32).reshape(1, 1, -1)
        for side in ['left', 'right']:
            with self.subTest(side=side):
                conv = CausalConv1d(1, 1, 4, dilation=2, causal_type=side, bias=False)
                with torch.no_grad():
                    conv.weight.fill_(1)
                changed = x.clone()
                if side == 'left':
                    changed[:, :, 10:] += 100
                    unaffected = slice(None, 10)
                else:
                    changed[:, :, :10] += 100
                    unaffected = slice(10, None)
                torch.testing.assert_close(conv(changed)[:, :, unaffected], conv(x)[:, :, unaffected])

    def test_explicit_context_matches_full_sequence(self):
        x = torch.arange(24, dtype=torch.float32).reshape(1, 1, -1)
        for side in ['left', 'right']:
            with self.subTest(side=side):
                conv = CausalConv1d(1, 1, 4, dilation=2, causal_type=side, bias=False)
                with torch.no_grad():
                    conv.weight.fill_(1)
                full = conv(x)
                if side == 'left':
                    chunk = conv(x[:, :, 12:], cache=x[:, :, 6:12])
                    expected = full[:, :, 12:]
                else:
                    chunk = conv(x[:, :, :12], cache=x[:, :, 12:18])
                    expected = full[:, :, :12]
                torch.testing.assert_close(chunk, expected)

    def test_causal_resblock_forward_and_backward(self):
        torch.manual_seed(0)
        block = ResBlock(channels=2, kernel_size=4, dilations=[1, 2, 4], causal=True)
        x = torch.randn(1, 2, 32, requires_grad=True)
        output = block(x)
        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for name, parameter in block.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == '__main__':
    unittest.main()
