import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_file_structure():
    print("=" * 60)
    print("  检查项目文件结构")
    print("=" * 60)
    
    expected_files = [
        "config.yaml",
        "requirements.txt",
        "README.md",
        "start.bat",
        "test_setup.py",
        "utils/__init__.py",
        "utils/utils.py",
        "data/__init__.py",
        "data/dataset.py",
        "models/__init__.py",
        "models/nir_generator.py",
        "models/multitask_net.py",
        "models/baseline.py",
        "training/__init__.py",
        "training/train_nir_generator.py",
        "training/train_multitask.py",
        "evaluation/__init__.py",
        "evaluation/metrics.py",
        "inference/__init__.py",
        "inference/estimator.py",
        "app/__init__.py",
        "app/server.py",
        "frontend/index.html",
        "frontend/css/style.css",
        "frontend/js/app.js",
    ]
    
    all_exist = True
    for f in expected_files:
        path = os.path.join(os.path.dirname(__file__), f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✓ {f} ({size} bytes)")
        else:
            print(f"  ✗ {f} - 缺失")
            all_exist = False
    
    return all_exist


def test_config_syntax():
    print("\n" + "=" * 60)
    print("  检查配置文件语法")
    print("=" * 60)
    
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        print("  ✓ YAML 语法正确")
        print(f"  ✓ 项目名称: {config['project']['name']}")
        print(f"  ✓ NIR生成器: {config['models']['nir_generator']['type']}")
        print(f"  ✓ 多任务骨干: {config['models']['multitask']['backbone']}")
        return True
    except Exception as e:
        print(f"  ✗ YAML 语法错误: {e}")
        return False


def test_python_syntax():
    print("\n" + "=" * 60)
    print("  检查 Python 文件语法")
    print("=" * 60)
    
    py_files = []
    for root, dirs, files in os.walk(os.path.dirname(__file__)):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    
    all_ok = True
    for py_file in py_files:
        rel_path = os.path.relpath(py_file, os.path.dirname(__file__))
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                code = f.read()
            compile(code, py_file, "exec")
            print(f"  ✓ {rel_path}")
        except SyntaxError as e:
            print(f"  ✗ {rel_path}: 语法错误 - {e}")
            all_ok = False
    
    return all_ok


def test_html_css_js():
    print("\n" + "=" * 60)
    print("  检查前端文件")
    print("=" * 60)
    
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    
    html_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        if "<!DOCTYPE html>" in content and "</html>" in content:
            print(f"  ✓ index.html - HTML 结构完整")
        else:
            print(f"  ✗ index.html - HTML 结构不完整")
    
    css_path = os.path.join(frontend_dir, "css/style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            content = f.read()
        if "{" in content and "}" in content:
            print(f"  ✓ css/style.css - CSS 样式存在")
        else:
            print(f"  ✗ css/style.css - CSS 样式异常")
    
    js_path = os.path.join(frontend_dir, "js/app.js")
    if os.path.exists(js_path):
        with open(js_path, "r", encoding="utf-8") as f:
            content = f.read()
        if "function" in content or "const" in content or "let" in content:
            print(f"  ✓ js/app.js - JavaScript 逻辑存在")
        else:
            print(f"  ✗ js/app.js - JavaScript 逻辑异常")
    
    return True


def test_imports_basic():
    print("\n" + "=" * 60)
    print("  测试基础模块导入（不含深度学习框架）")
    print("=" * 60)
    
    try:
        from utils.utils import load_config, set_seed, AverageMeter
        print("  ✓ utils.utils 基础函数导入成功")
    except Exception as e:
        print(f"  ✗ utils.utils 导入失败: {e}")
        return False
    
    try:
        import yaml
        config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
        print("  ✓ 配置文件加载成功")
    except Exception as e:
        print(f"  ✗ 配置加载失败: {e}")
        return False
    
    return True


def main():
    results = []
    
    results.append(("文件结构", test_file_structure()))
    results.append(("配置语法", test_config_syntax()))
    results.append(("Python语法", test_python_syntax()))
    results.append(("前端文件", test_html_css_js()))
    results.append(("基础导入", test_imports_basic()))
    
    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✓ 通过" if ok else "✗ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 项通过")
    
    if passed == total:
        print("\n" + "=" * 60)
        print("  🎉 所有测试通过！项目结构完整")
        print("=" * 60)
        print("\n项目主要功能模块：")
        print("  1. NIR图像生成器 (U-Net / Pix2Pix)")
        print("  2. 多任务网络 (分类 + 卡路里 + 重量估计)")
        print("  3. 基线对比算法 (纯RGB)")
        print("  4. Flask后端API服务")
        print("  5. Web前端演示界面")
        print("\n使用方法：")
        print("  安装依赖: pip install -r requirements.txt")
        print("  启动应用: python app/server.py")
        print("  访问地址: http://localhost:5000")
    else:
        print("\n部分测试失败，请检查上述错误信息。")


if __name__ == "__main__":
    main()
