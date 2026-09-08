"""TDEE 估算与用户档案。Mifflin-St Jeor 公式，无 LLM。"""
from dataclasses import dataclass, field
from enum import Enum


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"


class ActivityLevel(Enum):
    SEDENTARY = 1.2
    LIGHT = 1.375
    MODERATE = 1.55
    ACTIVE = 1.725
    VERY_ACTIVE = 1.9


class Goal(Enum):
    LOSE = 0.8
    MAINTAIN = 1.0
    GAIN = 1.15


@dataclass
class UserProfile:
    """用户档案。special_population=True 时路由人工审核，不自动生成食谱。"""
    height_cm: float = 170.0
    weight_kg: float = 65.0
    age: int = 30
    gender: Gender = Gender.MALE
    activity_level: ActivityLevel = ActivityLevel.MODERATE
    goal: Goal = Goal.MAINTAIN
    allergies: list = field(default_factory=list)
    preferences: list = field(default_factory=list)
    special_population: bool = False  # 孕妇/糖尿病/肾病等 -> 路由人工


class TDEEEstimator:
    def __init__(self, profile: UserProfile):
        self.profile = profile

    def estimate(self) -> dict:
        p = self.profile
        # Mifflin-St Jeor
        bmr = (10 * p.weight_kg + 6.25 * p.height_cm
               - 5 * p.age + (5 if p.gender == Gender.MALE else -161))
        tdee = bmr * p.activity_level.value * p.goal.value
        return {
            "bmr": round(bmr, 1),
            "tdee": round(tdee, 1),
            "target_calories": round(tdee, 1),
            "macros_g": {
                "carb": round(tdee * 0.50 / 4, 1),
                "protein": round(tdee * 0.25 / 4, 1),
                "fat": round(tdee * 0.25 / 9, 1),
            },
            "special_population": p.special_population,
        }