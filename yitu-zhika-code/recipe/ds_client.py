"""DeepSeek-V3 API 封装。
- 只生成"菜名+克数"草稿，不输出营养数字
- D5: 新增 suggest_substitutions 为每个食物建议 1-2 个替换

环境变量：DEEPSEEK_API_KEY
超时/重试：3 次指数退避；失败抛 RuntimeError，让上层降级到纯规则模式
"""
import json
import os
import time

import requests


class DSClient:
    BASE_URL = "https://api.deepseek.com/v1/chat/completions"
    MODEL = "deepseek-chat"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "response_format": {"type": "json_object"},
        }
        for attempt in range(3):
            try:
                r = requests.post(self.BASE_URL, headers=headers,
                                  json=payload, timeout=60)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def plan_draft(self, profile_summary: str, target_kcal: int,
                   macros: dict, allergies: list) -> dict:
        """生成 7 天 x 3 餐草稿，仅菜名 + 克数。"""
        sys = (
            "你是营养规划助手。只输出 JSON，结构：\n"
            '{"daily_plans":[{"day":1,"meals":[{"meal":"breakfast",'
            '"items":[{"name":"...","grams":N,"cooked_or_raw":"..."}]}]}]}\n'
            "硬约束：\n"
            "1. 只输出菜名和克数，不要输出任何营养数字\n"
            f"2. 过敏忌口必须避开：{json.dumps(allergies, ensure_ascii=False)}\n"
            "3. 7 天总热量接近目标 ±10%\n"
            "4. 份量给生重或熟重，标注 cooked_or_raw"
        )
        user = (
            f"用户档案：{profile_summary}\n"
            f"目标热量：{target_kcal} kcal/天\n"
            f"目标宏量：{macros}\n"
            f"过敏忌口：{allergies}\n"
            f"输出 7 天 x 3 餐 JSON。"
        )
        return json.loads(self.generate(sys, user, max_tokens=4096))

    def suggest_substitutions(self, food_name: str, category: str = "",
                              allergies: list = None,
                              target_kcal_per_100g: float = None,
                              count: int = 2) -> dict:
        """D5: 为单个食物建议替换选项。

        返回 JSON：
        {
            "original": food_name,
            "category": category,
            "substitutions": [
                {"name": "...", "grams_swap": N, "reason": "...",
                 "kcal_per_100g_estimate": N}
            ]
        }

        grams_swap: 等价热量替换时，原 grams 对应的新克数（DS 估算）
        reason: 替换原因（同蛋白/低脂/低卡/同类别等）
        """
        allergies = allergies or []
        sys = (
            "你是营养替换助手。只输出 JSON，结构：\n"
            '{"substitutions":[{"name":"...","grams_swap":N,'
            '"reason":"...","kcal_per_100g_estimate":N}]}\n'
            "硬约束：\n"
            f"1. 输出 {count} 个替换选项，必须可等价替换原食物\n"
            f"2. 避开过敏源：{json.dumps(allergies, ensure_ascii=False)}\n"
            "3. grams_swap 是与原 100g 等价热量的新食物克数\n"
            "4. reason 必须简短（<20 字），说明替换理由\n"
            "5. 不输出任何其他营养数字（蛋白/碳水/脂肪由后端 RAG 回填）"
        )
        user = (
            f"原食物：{food_name}\n"
            f"类别：{category or 'unknown'}\n"
        )
        if target_kcal_per_100g is not None:
            user += f"原热量密度：{target_kcal_per_100g} kcal/100g\n"
        user += f"过敏忌口：{allergies}\n"
        user += f"输出 {count} 个替换选项 JSON。"
        try:
            resp = self.generate(sys, user, max_tokens=512)
            data = json.loads(resp)
            if "substitutions" not in data:
                data = {"substitutions": []}
            return {
                "original": food_name,
                "category": category,
                "substitutions": data["substitutions"][:count],
            }
        except Exception as e:
            # 降级：返回空替换，不阻断主流程
            return {
                "original": food_name,
                "category": category,
                "substitutions": [],
                "error": str(e),
            }
