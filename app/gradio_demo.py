"""
Gradio演示界面
===============
一图知卡的交互式演示界面。

功能:
    1. 上传食物图片 → 显示: 原图/预测NIR图/食物类别/卡路里/与基线对比
    2. 侧栏: 输入用户信息(身高体重年龄性别活动量) → 计算TDEE
    3. 生成周食谱按钮 → 展示7天食谱(营养表+LLM润色做法)

界面语言: 中文

使用方法:
    python app/gradio_demo.py
"""

import os
import sys
import gradio as gr
import numpy as np
from PIL import Image
from typing import Optional

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def create_demo():
    """创建Gradio演示界面"""

    # ========= 推理管线 (延迟初始化) =========
    pipeline = None

    def init_pipeline():
        nonlocal pipeline
        if pipeline is None:
            from app.pipeline import InferencePipeline

            # 自动检测checkpoint
            import glob as _glob
            gen_ckpt = None
            for pattern in ['checkpoints/phase1/final_model.pth',
                            'checkpoints/phase1/epoch_*.pth']:
                matches = sorted(_glob.glob(pattern))
                if matches:
                    gen_ckpt = matches[-1]  # 取最新的
                    break

            mt_ckpt = None
            for pattern in ['checkpoints/multitask/best_model.pt',
                            'checkpoints/multitask/checkpoint_epoch_*.pt']:
                matches = sorted(_glob.glob(pattern))
                if matches:
                    mt_ckpt = matches[0] if 'best' in matches[0] else matches[-1]
                    break

            pipeline = InferencePipeline(
                generator_ckpt=gen_ckpt,
                multitask_ckpt=mt_ckpt,
                device="auto",
            )
        return pipeline

    # ========= 食物分析函数 =========
    def analyze_food(image):
        """分析食物图片

        Args:
            image: 输入图像 (numpy array)

        Returns:
            nir_image, category, nutrition_text, comparison_text
        """
        if image is None:
            return None, "请上传图片", "", ""

        try:
            pipe = init_pipeline()
            pil_image = Image.fromarray(image)
            result = pipe.predict(pil_image)

            # NIR图像
            nir_img = result.get('nir_image')
            if nir_img is not None:
                nir_rgb = np.stack([nir_img] * 3, axis=-1)  # 灰度转RGB
            else:
                nir_rgb = np.zeros_like(image)

            # 分类和营养
            category = f"食物类别: 类别{result.get('category_idx', '?')}"
            category_prob = result.get('category_prob', 0)
            if category_prob > 0:
                category += f" (置信度: {category_prob:.1%})"

            calories = result.get('calories', 0)
            weight = result.get('weight', 0)

            nutrition_text = (
                f"## 营养估计\n"
                f"| 指标 | 预测值 |\n"
                f"|------|--------|\n"
                f"| 卡路里 | **{calories:.0f} kcal** |\n"
                f"| 重量 | **{weight:.0f} g** |\n"
            )

            # 基线对比
            comparison_text = ""
            baseline_cal = result.get('baseline_calories')
            if baseline_cal is not None:
                diff = calories - baseline_cal
                comparison_text = (
                    f"## 与CalorieCLIP基线对比\n"
                    f"| 方法 | 卡路里估计 |\n"
                    f"|------|------------|\n"
                    f"| 本方法(Ours) | {calories:.0f} kcal |\n"
                    f"| CalorieCLIP | {baseline_cal:.0f} kcal |\n"
                    f"| 差异 | {diff:+.0f} kcal |\n"
                )

            return nir_rgb, category, nutrition_text, comparison_text

        except Exception as e:
            return None, f"分析失败: {str(e)}", "", ""

    # ========= TDEE计算 =========
    def calculate_tdee(height, weight, age, gender, activity, goal):
        """计算TDEE"""
        try:
            from recipe.tdee_estimator import TDEEEstimator, UserProfile, Gender, ActivityLevel, Goal

            activity_map = {
                "久坐不动": ActivityLevel.SEDENTARY,
                "轻度运动(1-3天/周)": ActivityLevel.LIGHT,
                "中度运动(3-5天/周)": ActivityLevel.MODERATE,
                "高度运动(6-7天/周)": ActivityLevel.ACTIVE,
                "极高强度(体力劳动)": ActivityLevel.VERY_ACTIVE,
            }
            goal_map = {
                "减脂": Goal.LOSE,
                "维持": Goal.MAINTAIN,
                "增肌": Goal.GAIN,
            }

            profile = UserProfile(
                height_cm=height,
                weight_kg=weight,
                age=int(age),
                gender=Gender.MALE if gender == "男" else Gender.FEMALE,
                activity_level=activity_map.get(activity, ActivityLevel.MODERATE),
                goal=goal_map.get(goal, Goal.MAINTAIN),
            )

            estimator = TDEEEstimator(profile)
            report = estimator.get_full_report()

            tdee = report['tdee']
            macros = report['macros']

            result_text = (
                f"## 能量评估结果\n"
                f"| 指标 | 数值 |\n"
                f"|------|------|\n"
                f"| 基础代谢(BMR) | {report['bmr']} kcal/天 |\n"
                f"| 每日总消耗(TDEE) | **{tdee} kcal/天** |\n"
                f"| 碳水建议 | {macros['carb_g']}g ({macros['carb_kcal']}kcal) |\n"
                f"| 蛋白质建议 | {macros['protein_g']}g ({macros['protein_kcal']}kcal) |\n"
                f"| 脂肪建议 | {macros['fat_g']}g ({macros['fat_kcal']}kcal) |\n"
            )

            return result_text, tdee

        except Exception as e:
            return f"计算失败: {str(e)}", 2000

    # ========= 食谱生成 =========
    def generate_recipe(tdee):
        """生成周食谱"""
        try:
            from recipe.weekly_planner import WeeklyPlanner
            from recipe.tdee_estimator import UserProfile, Gender, ActivityLevel, Goal

            planner = WeeklyPlanner(llm_backend="local")
            profile = UserProfile(
                activity_level=ActivityLevel.MODERATE,
                goal=Goal.MAINTAIN,
            )

            result = planner.generate(
                user_profile=profile,
                use_llm_polish=False,  # 先用模板润色
                calorie_tolerance=200,
            )

            # 格式化输出
            recipe_text = f"## 每周食谱 (目标: {tdee} kcal/天)\n\n"

            for day in result['daily_recipes']:
                recipe_text += f"### 第{day['day']}天\n"
                for meal in day['meals']:
                    recipe_text += f"**{meal['meal_name']}**\n"
                    for food in meal['foods']:
                        recipe_text += f"  - {food['name']}: {food['grams']}g\n"
                    nut = meal['nutrition']
                    recipe_text += f"  → 热量: {nut.get('calories', 0):.0f}kcal, "
                    recipe_text += f"蛋白质: {nut.get('protein', 0):.1f}g, "
                    recipe_text += f"碳水: {nut.get('carb', 0):.1f}g, "
                    recipe_text += f"脂肪: {nut.get('fat', 0):.1f}g\n\n"

                recipe_text += "---\n"

            summary = result['weekly_summary']
            recipe_text += (
                f"\n### 周汇总\n"
                f"日均热量: {summary['avg_daily_calories']} kcal\n"
                f"日均蛋白质: {summary['avg_daily_protein_g']}g\n"
                f"日均碳水: {summary['avg_daily_carb_g']}g\n"
                f"日均脂肪: {summary['avg_daily_fat_g']}g\n"
                f"偏差天数: {summary['deviation_days']}/{summary['total_days']}\n"
            )

            return recipe_text

        except Exception as e:
            return f"食谱生成失败: {str(e)}"

    # ========= 构建Gradio界面 =========
    with gr.Blocks(title="一图知卡 - 食物热量分析", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🍽️ 一图知卡 — 基于近红外光谱的实时食物热量分析\n"
            "上传食物照片，AI自动识别食物类别并估算卡路里，还能为你生成个性化周食谱！"
        )

        with gr.Row():
            # 左侧：图片上传和分析
            with gr.Column(scale=1):
                gr.Markdown("## 📸 食物分析")
                input_image = gr.Image(label="上传食物照片", type="numpy")

                with gr.Row():
                    analyze_btn = gr.Button("🔍 分析食物", variant="primary")

                with gr.Row():
                    with gr.Column():
                        nir_output = gr.Image(label="预测NIR图像")
                    with gr.Column():
                        category_output = gr.Textbox(label="分类结果", lines=2)

                nutrition_output = gr.Markdown(label="营养估计")
                comparison_output = gr.Markdown(label="基线对比")

            # 右侧：用户信息和食谱
            with gr.Column(scale=1):
                gr.Markdown("## 👤 用户信息")
                with gr.Row():
                    height_input = gr.Number(label="身高(cm)", value=170)
                    weight_input = gr.Number(label="体重(kg)", value=65)
                with gr.Row():
                    age_input = gr.Number(label="年龄", value=30)
                    gender_input = gr.Radio(["男", "女"], label="性别", value="男")
                activity_input = gr.Dropdown(
                    choices=[
                        "久坐不动",
                        "轻度运动(1-3天/周)",
                        "中度运动(3-5天/周)",
                        "高度运动(6-7天/周)",
                        "极高强度(体力劳动)",
                    ],
                    label="活动量",
                    value="中度运动(3-5天/周)",
                )
                goal_input = gr.Radio(
                    choices=["减脂", "维持", "增肌"],
                    label="目标",
                    value="维持",
                )

                tdee_btn = gr.Button("📊 计算每日热量需求", variant="secondary")
                tdee_output = gr.Markdown()
                tdee_value = gr.State(2000)

                recipe_btn = gr.Button("🥗 生成周食谱", variant="primary")
                recipe_output = gr.Markdown()

        # ========= 事件绑定 =========
        analyze_btn.click(
            fn=analyze_food,
            inputs=[input_image],
            outputs=[nir_output, category_output, nutrition_output, comparison_output],
        )

        tdee_btn.click(
            fn=calculate_tdee,
            inputs=[height_input, weight_input, age_input, gender_input,
                    activity_input, goal_input],
            outputs=[tdee_output, tdee_value],
        )

        recipe_btn.click(
            fn=generate_recipe,
            inputs=[tdee_value],
            outputs=[recipe_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
