import unittest

import torch
import torch.nn.functional as F

from cosyvoice.hifigan.generator import _linear_interpolate_1d_fallback


class LinearInterpolate1dTest(unittest.TestCase):

    def test_cosyvoice3_downsampling_coordinates(self):
        input = torch.linspace(-1, 1, 6720 * 9).reshape(1, 6720, 9).transpose(1, 2)

        expected = F.interpolate(input, scale_factor=1 / 480, mode='linear')
        actual = _linear_interpolate_1d_fallback(input, scale_factor=1 / 480)

        torch.testing.assert_close(actual, expected)

    def test_cosyvoice3_upsampling_coordinates(self):
        input = torch.linspace(-1, 1, 14 * 9).reshape(1, 14, 9).transpose(1, 2)

        expected = F.interpolate(input, scale_factor=480, mode='linear')
        actual = _linear_interpolate_1d_fallback(input, scale_factor=480)

        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
