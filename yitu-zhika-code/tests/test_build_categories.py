# P1-A：标签规则回归测试。不使用 tempfile（避免沙箱权限问题）。
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.build_categories import classify_ingredient, classify_ingredient_legacy, match_ingredient


class CategoryRuleTests(unittest.TestCase):
    def test_documented_false_hits_fixed(self):
        # 方案 §4.2 的四个规则错例：旧规则误匹配，新规则应正确
        cases = {
            "apple pie": "dessert",
            "eggplant": "vegetable",
            "graham cracker": "grain",
            "cream of mushroom soup": "soup_stew",
        }
        for ingr, expected in cases.items():
            self.assertEqual(classify_ingredient(ingr), expected, f"{ingr} 归为 {classify_ingredient(ingr)}，应为 {expected}")
            # 同时确认旧规则确实错了（证明这是真问题而非过度修复）
            self.assertNotEqual(classify_ingredient_legacy(ingr), expected, f"{ingr} 旧规则竟正确？（不该）")

    def test_word_boundary_no_substring(self):
        # 词边界：egg 不是 eggplant 的子串匹配；ham 不是 graham 的子串匹配
        self.assertEqual(classify_ingredient("eggplant"), "vegetable")
        self.assertEqual(classify_ingredient("graham cracker"), "grain")
        self.assertEqual(classify_ingredient("scrambled egg"), "egg")

    def test_head_noun_priority(self):
        # 右端/头名词优先：cream of mushroom soup → soup(soup_stew)，不是 cream(dairy)
        self.assertEqual(classify_ingredient("mushroom soup"), "soup_stew")
        self.assertEqual(classify_ingredient("chicken soup"), "soup_stew")
        self.assertEqual(classify_ingredient("apple pie"), "dessert")

    def test_override_regression(self):
        # 人工覆盖：明确已知的复合食材
        self.assertEqual(classify_ingredient("ice cream"), "dessert")      # 旧：dairy (ice cream 在 dairy)
        self.assertEqual(classify_ingredient("sour cream"), "dairy")
        self.assertEqual(classify_ingredient("sweet potato"), "vegetable")

    def test_returns_keyword_for_audit(self):
        cat, kw = match_ingredient("grilled chicken")
        self.assertEqual(cat, "meat")
        self.assertEqual(kw, "chicken")


if __name__ == "__main__":
    unittest.main()
