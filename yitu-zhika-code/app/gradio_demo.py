"""
Gradio演示界面 (本地适配版)
============================
一图知卡的交互式演示界面。

功能:
    1. 上传食物图片 → 显示: 原图/预测NIR图/食物类别/卡路里
    2. 侧栏: 输入用户信息 → 计算TDEE
    3. 生成周食谱按钮

使用方法:
    cd C:\\Users\\user\\Desktop\\yitu-zhika\\yitu-zhika-code
    C:\\Users\\user\\miniconda3\\envs\\yitu\\python.exe app\\gradio_demo.py
"""

import os
import sys
import glob as _glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)
os.chdir(PROJECT_ROOT)

import gradio as gr
import numpy as np
from PIL import Image


def create_demo():
    pipeline = None

    def init_pipeline():
        nonlocal pipeline
        if pipeline is None:
            from app.experiment_pipeline import ExperimentPipeline
            pipeline = ExperimentPipeline(device='cpu')
        return pipeline

    def analyze_food(image):
        if image is None:
            return None, "请上传图片", ""

        try:
            pipe = init_pipeline()
            pil_image = Image.fromarray(image)
            result = pipe.predict(pil_image)

            from app.experiment_pipeline import format_prediction
            return format_prediction(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"分析失败: {str(e)}", ""

    def calculate_tdee(height, weight, age, gender, activity, goal):
        try:
            from recipe.tdee_estimator import TDEEEstimator, UserProfile, Gender, ActivityLevel, Goal

            activity_map = {
                "久坐不动": ActivityLevel.SEDENTARY,
                "轻度运动(1-3天/周)": ActivityLevel.LIGHT,
                "中度运动(3-5天/周)": ActivityLevel.MODERATE,
                "高度运动(6-7天/周)": ActivityLevel.ACTIVE,
                "极高强度(体力劳动)": ActivityLevel.VERY_ACTIVE,
            }
            goal_map = {"减脂": Goal.LOSE, "维持": Goal.MAINTAIN, "增肌": Goal.GAIN}

            profile = UserProfile(
                height_cm=height, weight_kg=weight, age=int(age),
                gender=Gender.MALE if gender == "男" else Gender.FEMALE,
                activity_level=activity_map.get(activity, ActivityLevel.MODERATE),
                goal=goal_map.get(goal, Goal.MAINTAIN),
            )
            estimator = TDEEEstimator(profile)
            report = estimator.estimate()
            tdee = int(report['target_calories'])
            macros = report['macros_g']

            result_text = (
                f"## 能量评估结果\n"
                f"| 指标 | 数值 |\n"
                f"|------|------|\n"
                f"| 基础代谢(BMR) | {report['bmr']} kcal/天 |\n"
                f"| 每日总消耗(TDEE) | **{tdee} kcal/天** |\n"
                f"| 碳水建议 | {macros['carb']}g |\n"
                f"| 蛋白质建议 | {macros['protein']}g |\n"
                f"| 脂肪建议 | {macros['fat']}g |\n"
            )
            return result_text, tdee
        except ImportError:
            return "## TDEE计算\n\n食谱模块未安装，跳过。", 2000
        except Exception as e:
            return f"计算失败: {str(e)}", 2000

    def generate_recipe(tdee, height, weight, age, gender, activity, goal):
        try:
            from recipe.weekly_planner import WeeklyPlanner
            from recipe.tdee_estimator import UserProfile, Gender, ActivityLevel, Goal

            activity_map = {
                "久坐不动": ActivityLevel.SEDENTARY,
                "轻度运动(1-3天/周)": ActivityLevel.LIGHT,
                "中度运动(3-5天/周)": ActivityLevel.MODERATE,
                "高度运动(6-7天/周)": ActivityLevel.ACTIVE,
                "极高强度(体力劳动)": ActivityLevel.VERY_ACTIVE,
            }
            goal_map = {"减脂": Goal.LOSE, "维持": Goal.MAINTAIN, "增肌": Goal.GAIN}

            profile = UserProfile(
                height_cm=height, weight_kg=weight, age=int(age),
                gender=Gender.MALE if gender == "男" else Gender.FEMALE,
                activity_level=activity_map.get(activity, ActivityLevel.MODERATE),
                goal=goal_map.get(goal, Goal.MAINTAIN),
            )

            # 跳过无 DS API key 的情况，降级到纯 TDEE 提示
            import os
            if not os.environ.get('DEEPSEEK_API_KEY'):
                return f"## 食谱生成\n\n目标: {tdee} kcal/天\n\n未配置 DEEPSEEK_API_KEY，跳过 DS 营养师规划。请配置后重试。"

            planner = WeeklyPlanner(llm_backend="ds")
            result = planner.generate(user_profile=profile, use_llm_polish=False, calorie_tolerance=200)

            if result.get('status') == 'refused':
                return f"## 食谱生成\n\n特殊人群方案需营养师人工审核，不自动生成。"

            recipe_text = f"## 每周食谱 (目标: {tdee} kcal/天)\n\n"
            for day in result['daily_recipes']:
                recipe_text += f"### 第{day['day']}天 ({day.get('date', '')})\n"
                for meal in day['meals']:
                    recipe_text += f"**{meal['meal_name']}**\n"
                    for food in meal['foods']:
                        src = food.get('source')
                        src_tag = f" [溯源:{src['row_id']}]" if src else " [未入库]"
                        recipe_text += f"  - {food['name']}: {food['grams']}g{src_tag}\n"
                    recipe_text += f"  → 热量: {meal['nutrition'].get('calories', 0):.0f}kcal\n\n"
                dev = day.get('deviation_kcal', 0)
                ok = '✓' if day.get('within_tolerance') else '✗'
                recipe_text += f"日总: {day['day_total_kcal']}kcal (偏差 {dev:+.0f}) {ok}\n---\n"
            if result.get('unknown_foods'):
                recipe_text += f"\n⚠️ {len(result['unknown_foods'])} 项未入库食物，营养数据缺失\n"
            return recipe_text
        except ImportError:
            return "## 食谱生成\n\n食谱模块未安装，跳过。"
            return "## 食谱生成\n\n食谱模块未安装，跳过。"
        except Exception as e:
            return f"食谱生成失败: {str(e)}"

    with gr.Blocks(title="一图知卡 - 食物热量分析", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🍽️ 一图知卡 — 基于近红外光谱的实时食物热量分析\n"
            "上传食物照片，对比 RGB 与 RGB＋预测 NIR 的热量/重量估计。"
            "当前使用已核验的新权重及官方测试 ID 子集；外部基线通过训练与审计后接入。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📸 食物分析")
                input_image = gr.Image(label="上传食物照片", type="numpy")
                analyze_btn = gr.Button("🔍 分析食物", variant="primary")
                with gr.Row():
                    nir_output = gr.Image(label="预测NIR图像")
                    category_output = gr.Textbox(label="分类结果", lines=2)
                nutrition_output = gr.Markdown(label="营养估计")

            with gr.Column(scale=1):
                gr.Markdown("## 👤 用户信息")
                with gr.Row():
                    height_input = gr.Number(label="身高(cm)", value=170)
                    weight_input = gr.Number(label="体重(kg)", value=65)
                with gr.Row():
                    age_input = gr.Number(label="年龄", value=30)
                    gender_input = gr.Radio(["男", "女"], label="性别", value="男")
                activity_input = gr.Dropdown(
                    choices=["久坐不动", "轻度运动(1-3天/周)", "中度运动(3-5天/周)",
                             "高度运动(6-7天/周)", "极高强度(体力劳动)"],
                    label="活动量", value="中度运动(3-5天/周)")
                goal_input = gr.Radio(choices=["减脂", "维持", "增肌"], label="目标", value="维持")
                tdee_btn = gr.Button("📊 计算每日热量需求", variant="secondary")
                tdee_output = gr.Markdown()
                tdee_value = gr.State(2000)
                recipe_btn = gr.Button("🥗 生成周食谱", variant="primary")
                recipe_output = gr.Markdown()

        analyze_btn.click(fn=analyze_food, inputs=[input_image],
                          outputs=[nir_output, category_output, nutrition_output], api_name='analyze_food')
        tdee_btn.click(fn=calculate_tdee,
                       inputs=[height_input, weight_input, age_input, gender_input,
                               activity_input, goal_input],
                       outputs=[tdee_output, tdee_value])
        recipe_btn.click(fn=generate_recipe,
                           inputs=[tdee_value, height_input, weight_input,
                                   age_input, gender_input, activity_input, goal_input],
                           outputs=[recipe_output])

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
