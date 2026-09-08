"""
食谱推荐 — LLM润色 (路线B)
=============================
在规则引擎(路线A)生成的食谱骨架基础上，使用大语言模型润色：
- 生成具体做法步骤
- 添加口味描述
- 补充烹饪小贴士

关键原则:
    - 保留营养数据不改动，只润色文字
    - 支持多种LLM后端 (OpenAI / 本地模型)
    - 食谱骨架的营养数值作为约束传给LLM
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any


class LLMPolisher:
    """LLM食谱润色器

    Args:
        backend: LLM后端 ("openai" / "local" / "custom")
        model: 模型名称
        api_key: API密钥 (从环境变量或传入)
        api_base: API基础URL
        temperature: 生成温度
        max_tokens: 最大token数
    """

    # 系统提示词
    SYSTEM_PROMPT = """你是一位专业的中式菜谱润色师。你的任务是为食谱添加详细的烹饪步骤、口味描述和小贴士。

重要规则:
1. 保留原有的食材和份量信息，不得修改
2. 保留原有的营养数据(卡路里、蛋白质等)，不得修改
3. 生成的做法步骤要具体、可操作
4. 口味描述要生动、有食欲
5. 小贴士要实用，可以包含: 食材替代、火候控制、时间建议
6. 使用中文输出"""

    # 单餐润色模板
    MEAL_POLISH_TEMPLATE = """请为以下餐食润色，添加做法、口味描述和小贴士。

## 餐食信息
餐次: {meal_name}
食材: {foods_str}
营养: {nutrition_str}

请输出JSON格式:
{{
    "dish_name": "菜品名称(自拟，体现特色)",
    "cooking_steps": ["步骤1", "步骤2", ...],
    "flavor_description": "口味描述(2-3句话)",
    "tips": "烹饪小贴士",
    "estimated_time": "预计烹饪时间(如: 20分钟)"
}}"""

    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # API配置
        if backend == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            self.api_base = api_base or os.environ.get(
                "OPENAI_API_BASE", "https://api.openai.com/v1"
            )
        elif backend == "local":
            # 本地模型（如Ollama）
            self.api_base = api_base or "http://localhost:11434/v1"
            self.api_key = api_key or "local"
        elif backend == "custom":
            self.api_base = api_base or ""
            self.api_key = api_key or ""
        else:
            raise ValueError(f"不支持的LLM后端: {backend}")

    def polish_meal(self, meal_info: Dict[str, Any]) -> Dict[str, Any]:
        """润色单餐食谱

        Args:
            meal_info: 餐食信息 {
                'meal_name': str,
                'foods': [{'name': str, 'grams': int}, ...],
                'nutrition': {'calories': float, ...}
            }

        Returns:
            润色后的餐食信息（保留原始营养数据）
        """
        # 构建prompt
        foods_str = "、".join([f"{f['name']}{f['grams']}g" for f in meal_info["foods"]])
        nutrition_str = "、".join([f"{k}: {v}" for k, v in meal_info["nutrition"].items()])

        prompt = self.MEAL_POLISH_TEMPLATE.format(
            meal_name=meal_info["meal_name"],
            foods_str=foods_str,
            nutrition_str=nutrition_str,
        )

        # 调用LLM
        response_text = self._call_llm(prompt)

        # 解析响应
        polished = self._parse_response(response_text)

        # 合并结果（保留原始营养数据）
        result = dict(meal_info)
        if polished:
            result["dish_name"] = polished.get("dish_name", meal_info["meal_name"])
            result["cooking_steps"] = polished.get("cooking_steps", [])
            result["flavor_description"] = polished.get("flavor_description", "")
            result["tips"] = polished.get("tips", "")
            result["estimated_time"] = polished.get("estimated_time", "")
        else:
            # LLM调用失败，使用模板化润色
            result.update(self._template_polish(meal_info))

        return result

    def polish_weekly_recipe(self, recipe: List[Dict]) -> List[Dict]:
        """润色整周食谱

        Args:
            recipe: 规则引擎生成的周食谱

        Returns:
            润色后的周食谱
        """
        polished_recipe = []

        for day_info in recipe:
            polished_day = {"day": day_info["day"], "meals": []}

            for meal in day_info["meals"]:
                try:
                    polished_meal = self.polish_meal(meal)
                except Exception as e:
                    print(f"润色失败(第{day_info['day']}天{meal['meal_name']}): {e}")
                    polished_meal = dict(meal)
                    polished_meal.update(self._template_polish(meal))

                polished_day["meals"].append(polished_meal)

            polished_recipe.append(polished_day)

        return polished_recipe

    def _call_llm(self, prompt: str) -> str:
        """调用LLM API

        Args:
            prompt: 用户提示词

        Returns:
            LLM响应文本
        """
        if not self.api_key and self.backend != "local":
            raise ValueError("API key未设置")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"LLM API调用失败: {e}")
            return ""

    def _parse_response(self, response_text: str) -> Optional[Dict]:
        """解析LLM响应中的JSON"""
        if not response_text:
            return None

        # 尝试从响应中提取JSON
        try:
            # 直接解析
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 块
        try:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
        except (json.JSONDecodeError, AttributeError):
            pass

        # 尝试提取花括号内容
        try:
            import re
            brace_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if brace_match:
                return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, AttributeError):
            pass

        return None

    def _template_polish(self, meal_info: Dict) -> Dict:
        """模板化润色（LLM不可用时的降级方案）"""
        foods = meal_info.get("foods", [])
        food_names = "、".join([f["name"] for f in foods])
        meal_name = meal_info.get("meal_name", "")

        # 根据餐次生成模板化描述
        templates = {
            "早餐": {
                "dish_name": f"营养{food_names}早餐",
                "cooking_steps": [
                    "准备食材，清洗干净",
                    "按食材特性分别烹饪",
                    "摆盘上桌，搭配饮品",
                ],
                "flavor_description": "清爽营养的早餐组合，为新的一天注入活力。",
                "tips": "早餐建议搭配一杯温水，有助于肠胃健康。",
                "estimated_time": "15分钟",
            },
            "午餐": {
                "dish_name": f"丰盛{food_names}午餐",
                "cooking_steps": [
                    "准备食材，肉类提前腌制10分钟",
                    "热锅下油，先炒荤菜再炒素菜",
                    "调味出锅，搭配主食",
                ],
                "flavor_description": "荤素搭配的午餐组合，营养均衡又美味。",
                "tips": "午餐后可适当散步，有助消化。",
                "estimated_time": "30分钟",
            },
            "晚餐": {
                "dish_name": f"清淡{food_names}晚餐",
                "cooking_steps": [
                    "准备食材，以清淡烹饪为主",
                    "少油少盐，蒸煮优先",
                    "搭配主食，温热上桌",
                ],
                "flavor_description": "清淡少油的晚餐组合，有助于消化和睡眠。",
                "tips": "晚餐建议在睡前3小时完成，避免增加肠胃负担。",
                "estimated_time": "25分钟",
            },
        }

        return templates.get(meal_name, {
            "dish_name": f"{food_names}组合",
            "cooking_steps": ["准备食材", "烹饪", "上桌"],
            "flavor_description": "营养均衡的餐食组合。",
            "tips": "注意控制油盐用量。",
            "estimated_time": "20分钟",
        })


if __name__ == "__main__":
    # 测试模板化润色
    polisher = LLMPolisher(backend="local")

    meal = {
        "meal_name": "午餐",
        "foods": [
            {"name": "白米饭", "grams": 200},
            {"name": "宫保鸡丁", "grams": 150},
            {"name": "清炒西兰花", "grams": 100},
        ],
        "nutrition": {"calories": 580.0, "protein": 25.0, "carb": 65.0, "fat": 18.0, "fiber": 3.0},
    }

    polished = polisher.polish_meal(meal)
    print(f"菜品名: {polished.get('dish_name')}")
    print(f"做法: {polished.get('cooking_steps')}")
    print(f"口味: {polished.get('flavor_description')}")
    print(f"营养(不变): {polished.get('nutrition')}")
