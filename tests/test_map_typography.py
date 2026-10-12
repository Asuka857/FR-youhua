import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MapTypographyTest(unittest.TestCase):
    def test_main_map_typography_settings(self):
        source = (ROOT / "mypackage" / "picture.py").read_text(encoding="utf-8")

        self.assertIn("ax.set_title(title, fontsize=26", source)
        self.assertIn("labelsize=16", source)
        self.assertIn("fontsize=19", source)
        self.assertRegex(
            source,
            re.compile(
                r"ax\.text\(\s*0\.5,\s*0\.25,\s*text_str,"
                r".*?fontsize=22,.*?ha='center'",
                re.S,
            ),
        )

    def test_comparison_map_typography_settings(self):
        source = (ROOT / "mypackage" / "W_comparison_map_tif.py").read_text(encoding="utf-8")

        self.assertIn("'Y': 'Yield'", source)
        self.assertIn("'CH4': 'CH₄'", source)
        self.assertIn("'N2O': 'N₂O'", source)
        self.assertIn("ax.set_title(title, fontsize=26", source)
        self.assertIn("labelsize=16", source)
        self.assertIn("fontsize=19", source)
        self.assertRegex(
            source,
            re.compile(
                r"ax\.text\(\s*0\.5,\s*0\.25,"
                r".*?text_str,.*?fontsize=22,.*?ha='center'",
                re.S,
            ),
        )
        self.assertNotIn("title=f'{title_name} - Optimal'", source)
        self.assertNotIn("title=f'{title_name} - Control'", source)
        self.assertNotIn("title=f'{title_name} - Change'", source)

    def test_ratio_map_title_uses_scientific_indicator_label(self):
        comparison_source = (ROOT / "mypackage" / "W_comparison_map_tif.py").read_text(encoding="utf-8")
        ratio_source = (ROOT / "mypackage" / "W_ratio_analysis.py").read_text(encoding="utf-8")

        self.assertIn("'CH4': 'CH₄'", ratio_source)
        self.assertIn("'N2O': 'N₂O'", ratio_source)
        self.assertNotIn("ratio_title = f'{title_name} ratio'", comparison_source)
        self.assertNotIn("title=f'{title_name} ratio'", ratio_source)

    def test_ratio_global_mean_uses_signed_percent_change(self):
        from mypackage.W_comparison_map_tif import format_global_mean_text

        comparison_source = (ROOT / "mypackage" / "W_comparison_map_tif.py").read_text(encoding="utf-8")
        ratio_source = (ROOT / "mypackage" / "W_ratio_analysis.py").read_text(encoding="utf-8")

        self.assertEqual(format_global_mean_text(0.82, as_percentage=True), "Global Mean: -18.0%")
        self.assertEqual(format_global_mean_text(1.08, as_percentage=True), "Global Mean: +8.0%")
        self.assertIn("global_mean_as_percentage=True", comparison_source)
        self.assertIn("global_mean_as_percentage=True", ratio_source)


if __name__ == "__main__":
    unittest.main()
