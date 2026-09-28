import unittest

from cosyvoice.utils.frontend_utils import replace_blank


class ReplaceBlankTests(unittest.TestCase):
    def test_doubled_space_between_numbers_stays_a_space(self):
        self.assertEqual(replace_blank("版本 12  34"), "版本12 34")

    def test_single_space_between_numbers_stays(self):
        self.assertEqual(replace_blank("版本 12 34"), "版本12 34")

    def test_space_beside_chinese_is_still_removed(self):
        self.assertEqual(replace_blank("你 a"), "你a")

    def test_single_space_between_ascii_words_stays(self):
        self.assertEqual(replace_blank("a b"), "a b")

    def test_doubled_space_between_ascii_words_collapses_to_one(self):
        self.assertEqual(replace_blank("a  b"), "a b")


if __name__ == "__main__":
    unittest.main()
