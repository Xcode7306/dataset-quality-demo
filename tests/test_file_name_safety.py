"""跨平台安全文件名的字符与字节边界测试。"""

import unittest

from src.upload_service import sanitize_file_name


class FileNameSafetyTests(unittest.TestCase):
    def test_chinese_name_respects_character_and_utf8_byte_limits(self):
        safe_name = sanitize_file_name(
            "数" * 200 + ".json",
            safe_extension=".json",
        )

        self.assertLessEqual(len(safe_name), 120)
        self.assertLessEqual(len(safe_name.encode("utf-8")), 255)
        self.assertTrue(safe_name.endswith(".json"))
        self.assertEqual(safe_name.removesuffix(".json"), "数" * 83)

    def test_emoji_name_is_cut_only_between_unicode_scalars(self):
        emoji = chr(0x1F600)
        safe_name = sanitize_file_name(
            emoji * 200 + ".json",
            safe_extension=".json",
        )

        self.assertLessEqual(len(safe_name), 120)
        self.assertLessEqual(len(safe_name.encode("utf-8")), 255)
        self.assertTrue(safe_name.endswith(".json"))
        self.assertEqual(safe_name.removesuffix(".json"), emoji * 62)
        safe_name.encode("utf-8", errors="strict")

    def test_custom_byte_limit_still_preserves_safe_extension(self):
        emoji = chr(0x1F600)
        safe_name = sanitize_file_name(
            emoji * 10 + ".json",
            safe_extension=".json",
            max_length=20,
            max_bytes=14,
        )

        self.assertEqual(safe_name, emoji * 2 + ".json")
        self.assertEqual(len(safe_name.encode("utf-8")), 13)

    def test_byte_limit_must_leave_room_for_stem(self):
        with self.assertRaisesRegex(ValueError, "max_bytes"):
            sanitize_file_name(
                "dataset.json",
                safe_extension=".json",
                max_bytes=5,
            )


if __name__ == "__main__":
    unittest.main()
