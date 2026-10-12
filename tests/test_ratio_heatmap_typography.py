import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RatioHeatmapTypographyTest(unittest.TestCase):
    def test_heatmap_font_sizes_are_enlarged(self):
        source = (ROOT / "mypackage" / "W_ratio_heatmap.py").read_text(encoding="utf-8")

        self.assertIn("plt.rcParams['font.size'] = 15", source)
        self.assertIn("'size': 13", source)
        self.assertIn("fontsize=16", source)
        self.assertIn("fontsize=14", source)
        self.assertIn("fontsize=18", source)
        self.assertIn("cbar.ax.tick_params(labelsize=14)", source)

    def test_heatmap_removes_axis_titles_and_bolds_bottom_variables(self):
        source = (ROOT / "mypackage" / "W_ratio_heatmap.py").read_text(encoding="utf-8")

        self.assertIn("ax.set_ylabel('')", source)
        self.assertIn("ax.set_xlabel('')", source)
        self.assertIn("ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=14, weight='bold')", source)
        self.assertNotIn("ax.set_ylabel('Continents'", source)
        self.assertNotIn("ax.set_xlabel('Indicators'", source)

    def test_heatmap_adds_global_row_and_removes_ratio_label(self):
        source = (ROOT / "mypackage" / "W_ratio_heatmap.py").read_text(encoding="utf-8")

        self.assertIn("global_row = merged_df.groupby('indicator')['optimal_ck_ratio'].mean().to_frame().T", source)
        self.assertIn("continents = ['Global', 'Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']", source)
        self.assertIn("cbar.ax.set_ylabel('')", source)
        self.assertNotIn("'label': 'ratio'", source)
        self.assertNotIn("cbar.ax.set_ylabel('ratio'", source)

    def test_heatmap_uses_subscripts(self):
        source = (ROOT / "mypackage" / "W_ratio_heatmap.py").read_text(encoding="utf-8")

        self.assertIn("'CH4': 'CH₄'", source)
        self.assertIn("'N2O': 'N₂O'", source)

    def test_heatmap_crop_order_is_maize_wheat_rice(self):
        source = (ROOT / "mypackage" / "W_ratio_heatmap.py").read_text(encoding="utf-8")

        self.assertIn("crops=['maize', 'wheat', 'rice']", source)
        self.assertIn("for crop in ['maize', 'wheat', 'rice']:", source)
        self.assertIn("ax2.set_xticklabels(['Maize', 'Wheat', 'Rice'], fontsize=18, weight='bold')", source)


if __name__ == "__main__":
    unittest.main()
