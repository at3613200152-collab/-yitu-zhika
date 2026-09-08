"""DS+RAG 营养师模块。

数据流：
  UserProfile -> TDEE -> 目标宏量 -> DS 生成 7天x3餐 草稿（菜名+克数）
              -> NutritionDB 检索每条菜名的精确营养 -> 回填 + SHA256 溯源
              -> 输出 weekly_plan，未知项保持未知不臆造

约束（delivery_scope.md L33）：
  - 可追溯营养数据：每条数字带 source_row_id + source_sha256
  - 缺失配料/油量保持未知，不把图片估计当精确摄入
  - 特殊人群/疾病方案不自动生成，路由人工审核
"""
from .tdee_estimator import TDEEEstimator, UserProfile, Gender, ActivityLevel, Goal
from .weekly_planner import WeeklyPlanner
__all__ = ["TDEEEstimator", "UserProfile", "Gender", "ActivityLevel", "Goal", "WeeklyPlanner"]