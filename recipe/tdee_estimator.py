"""
TDEE估算器
===========
基于用户个人信息计算每日总能量消耗(TDEE)，
作为食谱推荐的热量目标。

计算公式:
    1. BMR (基础代谢率) - Mifflin-St Jeor公式:
       男性: BMR = 10 × 体重(kg) + 6.25 × 身高(cm) - 5 × 年龄 + 5
       女性: BMR = 10 × 体重(kg) + 6.25 × 身高(cm) - 5 × 年龄 - 161

    2. TDEE = BMR × 活动系数:
       久坐: 1.2
       轻度运动(1-3天/周): 1.375
       中度运动(3-5天/周): 1.55
       高度运动(6-7天/周): 1.725
       极高强度(体力劳动): 1.9

    3. 目标调整:
       减脂: TDEE × 0.8
       维持: TDEE
       增肌: TDEE × 1.2
"""

from typing import Optional, Dict
from enum import Enum
from dataclasses import dataclass


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"


class ActivityLevel(Enum):
    SEDENTARY = "sedentary"           # 久坐
    LIGHT = "light"                   # 轻度运动
    MODERATE = "moderate"             # 中度运动
    ACTIVE = "active"                 # 高度运动
    VERY_ACTIVE = "very_active"       # 极高强度


class Goal(Enum):
    LOSE = "lose"           # 减脂
    MAINTAIN = "maintain"   # 维持
    GAIN = "gain"           # 增肌


# 活动系数映射
ACTIVITY_MULTIPLIERS = {
    ActivityLevel.SEDENTARY: 1.2,
    ActivityLevel.LIGHT: 1.375,
    ActivityLevel.MODERATE: 1.55,
    ActivityLevel.ACTIVE: 1.725,
    ActivityLevel.VERY_ACTIVE: 1.9,
}

# 目标调整系数
GOAL_MULTIPLIERS = {
    Goal.LOSE: 0.8,
    Goal.MAINTAIN: 1.0,
    Goal.GAIN: 1.2,
}

# 活动等级中文描述
ACTIVITY_LABELS = {
    ActivityLevel.SEDENTARY: "久坐不动",
    ActivityLevel.LIGHT: "轻度运动(1-3天/周)",
    ActivityLevel.MODERATE: "中度运动(3-5天/周)",
    ActivityLevel.ACTIVE: "高度运动(6-7天/周)",
    ActivityLevel.VERY_ACTIVE: "极高强度(体力劳动)",
}


@dataclass
class UserProfile:
    """用户档案"""
    height_cm: float = 170.0         # 身高(cm)
    weight_kg: float = 65.0          # 体重(kg)
    age: int = 30                     # 年龄
    gender: Gender = Gender.MALE      # 性别
    activity_level: ActivityLevel = ActivityLevel.MODERATE  # 活动量
    goal: Goal = Goal.MAINTAIN        # 目标


class TDEEEstimator:
    """TDEE估算器

    基于Mifflin-St Jeor公式计算每日热量需求。

    Args:
        profile: 用户档案
    """

    def __init__(self, profile: Optional[UserProfile] = None):
        self.profile = profile or UserProfile()

    def calculate_bmr(self, profile: Optional[UserProfile] = None) -> float:
        """计算BMR (基础代谢率)

        使用Mifflin-St Jeor公式:
        男性: BMR = 10 × 体重 + 6.25 × 身高 - 5 × 年龄 + 5
        女性: BMR = 10 × 体重 + 6.25 × 身高 - 5 × 年龄 - 161

        Args:
            profile: 用户档案 (默认使用初始化时的profile)

        Returns:
            BMR (kcal/day)
        """
        p = profile or self.profile
        bmr = 10 * p.weight_kg + 6.25 * p.height_cm - 5 * p.age
        if p.gender == Gender.MALE:
            bmr += 5
        else:
            bmr -= 161
        return max(bmr, 800)  # 最低保障

    def calculate_tdee(self, profile: Optional[UserProfile] = None) -> float:
        """计算TDEE (每日总能量消耗)

        TDEE = BMR × 活动系数 × 目标系数

        Args:
            profile: 用户档案

        Returns:
            TDEE (kcal/day)
        """
        p = profile or self.profile
        bmr = self.calculate_bmr(p)
        activity_mult = ACTIVITY_MULTIPLIERS.get(p.activity_level, 1.55)
        goal_mult = GOAL_MULTIPLIERS.get(p.goal, 1.0)
        tdee = bmr * activity_mult * goal_mult
        return round(tdee)

    def calculate_macros(self, tdee: Optional[float] = None) -> Dict[str, float]:
        """计算宏量营养素建议量

        默认比例: 碳水50%、蛋白质25%、脂肪25%

        Args:
            tdee: TDEE值 (默认自动计算)

        Returns:
            {carb_g, protein_g, fat_g, carb_kcal, protein_kcal, fat_kcal}
        """
        if tdee is None:
            tdee = self.calculate_tdee()

        # 热量分配
        carb_kcal = tdee * 0.50
        protein_kcal = tdee * 0.25
        fat_kcal = tdee * 0.25

        # 转换为克数 (碳水/蛋白质: 4kcal/g, 脂肪: 9kcal/g)
        carb_g = carb_kcal / 4
        protein_g = protein_kcal / 4
        fat_g = fat_kcal / 9

        return {
            "carb_g": round(carb_g),
            "protein_g": round(protein_g),
            "fat_g": round(fat_g),
            "carb_kcal": round(carb_kcal),
            "protein_kcal": round(protein_kcal),
            "fat_kcal": round(fat_kcal),
        }

    def get_full_report(self, profile: Optional[UserProfile] = None) -> Dict[str, any]:
        """获取完整的能量评估报告

        Returns:
            完整报告字典
        """
        p = profile or self.profile
        bmr = self.calculate_bmr(p)
        tdee = self.calculate_tdee(p)
        macros = self.calculate_macros(tdee)

        return {
            "profile": {
                "性别": "男" if p.gender == Gender.MALE else "女",
                "身高": f"{p.height_cm}cm",
                "体重": f"{p.weight_kg}kg",
                "年龄": f"{p.age}岁",
                "活动量": ACTIVITY_LABELS.get(p.activity_level, "未知"),
                "目标": {"lose": "减脂", "maintain": "维持", "gain": "增肌"}.get(p.goal.value, "维持"),
            },
            "bmr": round(bmr),
            "tdee": tdee,
            "macros": macros,
        }


if __name__ == "__main__":
    # 测试TDEE估算
    profiles = [
        UserProfile(height_cm=175, weight_kg=70, age=25, gender=Gender.MALE,
                    activity_level=ActivityLevel.MODERATE, goal=Goal.MAINTAIN),
        UserProfile(height_cm=162, weight_kg=55, age=30, gender=Gender.FEMALE,
                    activity_level=ActivityLevel.LIGHT, goal=Goal.LOSE),
        UserProfile(height_cm=180, weight_kg=80, age=35, gender=Gender.MALE,
                    activity_level=ActivityLevel.ACTIVE, goal=Goal.GAIN),
    ]

    estimator = TDEEEstimator()
    for p in profiles:
        report = estimator.get_full_report(p)
        print(f"\n{report['profile']}")
        print(f"  BMR: {report['bmr']} kcal → TDEE: {report['tdee']} kcal")
        print(f"  碳水: {report['macros']['carb_g']}g, "
              f"蛋白质: {report['macros']['protein_g']}g, "
              f"脂肪: {report['macros']['fat_g']}g")
