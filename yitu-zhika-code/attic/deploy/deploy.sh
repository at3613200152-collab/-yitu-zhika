# 部署脚本（Ubuntu，在 yitu-zhika-code 目录下以 root 或 sudo 执行）
# 前置：已把代码、模型权重、results 审计文件、data/dishes_verified.csv 放到服务器
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== 1/4 创建虚拟环境 =="
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip

echo "== 2/4 安装依赖（安装 PyTorch CPU 版，节省体积）=="
# 如无 GPU，装 CPU 版 torch；有 GPU 则改用官网命令
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo "== 3/4 校验文件存在 =="
test -f checkpoints/meal_rgb_official_v1/best.pt || { echo "缺少主模型 best.pt"; exit 1; }
test -f checkpoints/meal_nir_official_v1/best.pt || { echo "缺少 NIR 模型 best.pt"; exit 1; }
test -f data/nutrition5k/dishes_verified.csv || { echo "缺少营养库 CSV"; exit 1; }
echo "权重与数据文件 OK"

echo "== 4/4 冒烟自检（导入并加载模型）=="
python -c "import sys; sys.path.insert(0,'.'); from app.inference_service import get_pipeline; p=get_pipeline(); print('pipeline loaded:', p is not None)"

echo "完成。启动命令："
echo "  source venv/bin/activate && gunicorn -c deploy/gunicorn.conf.py app.inference_service:app"
echo "生产请用 systemd（deploy/systemd/yitu-zhika.service）守护，并用 nginx 反代 + HTTPS。"
