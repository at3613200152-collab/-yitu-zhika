# App 模块

一图知卡前端应用。

## 模块列表

| 文件 | 描述 |
|------|------|
| [pipeline.py](pipeline.py) | 端到端推理管线，自动检测生成器架构 |
| [gradio_demo.py](gradio_demo.py) | Gradio Web界面 (食物分析+TDEE+食谱推荐) |

## 快速启动

```bash
# 确保checkpoints在 checkpoints/phase1/ 和 checkpoints/multitask/ 下
python app/gradio_demo.py
# 浏览器访问 http://localhost:7860
```

## 功能

1. **食物分析**: 上传RGB图 → 生成NIR → 预测类别/卡路里/重量
2. **TDEE计算**: 输入身高体重年龄 → 计算每日热量需求
3. **食谱推荐**: 基于TDEE生成7天食谱
