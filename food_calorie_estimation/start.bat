@echo off
chcp 65001 >nul
echo ========================================
echo   食物卡路里估计小程序 - 启动脚本
echo ========================================
echo.

cd /d %~dp0

echo [1/3] 检查 Python 依赖...
python -c "import flask, torch, PIL" 2>nul
if errorlevel 1 (
    echo 正在安装依赖...
    pip install -r requirements.txt
) else (
    echo 依赖已安装
)

echo.
echo [2/3] 检查模型权重...
if exist "checkpoints\nir_generator\best.pth" (
    echo   [OK] NIR 生成器权重
) else (
    echo   [!] 未找到 NIR 生成器权重, 将使用随机初始化
)
if exist "checkpoints\multitask\best.pth" (
    echo   [OK] 多任务网络权重
) else (
    echo   [!] 未找到多任务网络权重, 将自动 fallback 到 ImageNet+知识库
)

echo.
echo [3/3] 启动后端服务 (Flask)...
echo   - 访问地址: http://localhost:5000
echo   - 按 Ctrl+C 停止服务
echo.

python app/server.py

pause
