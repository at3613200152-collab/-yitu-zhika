import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    print("=" * 50)
    print("  测试模块导入")
    print("=" * 50)
    
    try:
        from utils.utils import load_config, set_seed
        print("✓ utils 模块导入成功")
    except Exception as e:
        print(f"✗ utils 模块导入失败: {e}")
        return False
    
    try:
        from data.dataset import HSIFoodIngrDataset, Nutrition5kDataset
        print("✓ data 模块导入成功")
    except Exception as e:
        print(f"✗ data 模块导入失败: {e}")
        return False
    
    try:
        from models.nir_generator import UNetGenerator, Pix2PixModel
        from models.multitask_net import MultiTaskResNet
        from models.baseline import BaselineResNet
        print("✓ models 模块导入成功")
    except Exception as e:
        print(f"✗ models 模块导入失败: {e}")
        return False
    
    try:
        from evaluation.metrics import calculate_psnr, calculate_ssim, calculate_mape, calculate_rmse
        print("✓ evaluation 模块导入成功")
    except Exception as e:
        print(f"✗ evaluation 模块导入失败: {e}")
        return False
    
    try:
        from inference.estimator import FoodCalorieEstimator
        print("✓ inference 模块导入成功")
    except Exception as e:
        print(f"✗ inference 模块导入失败: {e}")
        return False
    
    return True


def test_config():
    print("\n" + "=" * 50)
    print("  测试配置文件")
    print("=" * 50)
    
    try:
        from utils.utils import load_config
        config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
        print(f"✓ 配置文件加载成功")
        print(f"  - 项目名称: {config['project']['name']}")
        print(f"  - NIR生成器类型: {config['models']['nir_generator']['type']}")
        print(f"  - 多任务骨干网络: {config['models']['multitask']['backbone']}")
        return True
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return False


def test_dummy_data():
    print("\n" + "=" * 50)
    print("  测试数据集（自动生成模拟数据）")
    print("=" * 50)
    
    try:
        import torch
        from utils.utils import load_config
        from data.dataset import get_hsi_dataloaders, get_nutrition5k_dataloaders
        
        config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
        
        train_loader, val_loader = get_hsi_dataloaders(config)
        rgb, nir = next(iter(train_loader))
        print(f"✓ HSI数据集加载成功")
        print(f"  - 训练集: {len(train_loader.dataset)} 样本")
        print(f"  - 验证集: {len(val_loader.dataset)} 样本")
        print(f"  - RGB shape: {rgb.shape}")
        print(f"  - NIR shape: {nir.shape}")
        
        train_loader2, val_loader2 = get_nutrition5k_dataloaders(config, use_predicted_nir=False)
        inputs, labels = next(iter(train_loader2))
        print(f"\n✓ Nutrition5k数据集加载成功")
        print(f"  - 训练集: {len(train_loader2.dataset)} 样本")
        print(f"  - 验证集: {len(val_loader2.dataset)} 样本")
        print(f"  - 输入 shape: {inputs.shape}")
        print(f"  - 类别: {labels['class'].shape}")
        print(f"  - 卡路里: {labels['calories'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    print("\n" + "=" * 50)
    print("  测试模型构建")
    print("=" * 50)
    
    try:
        import torch
        from utils.utils import load_config
        from models.nir_generator import UNetGenerator, Pix2PixModel
        from models.multitask_net import MultiTaskResNet
        from models.baseline import BaselineResNet
        
        config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
        
        model_cfg = config["models"]["nir_generator"]
        nir_gen = UNetGenerator(
            in_channels=model_cfg["in_channels"],
            out_channels=model_cfg["out_channels"],
            base_channels=model_cfg["base_channels"],
            num_downs=model_cfg["num_downs"],
        )
        x = torch.randn(1, 3, 256, 256)
        y = nir_gen(x)
        print(f"✓ U-Net生成器构建成功")
        print(f"  - 输入: {x.shape}")
        print(f"  - 输出: {y.shape}")
        
        mt_model = MultiTaskResNet(config)
        x2 = torch.randn(1, 4, 224, 224)
        out = mt_model(x2)
        print(f"\n✓ 多任务网络构建成功")
        print(f"  - 输入: {x2.shape}")
        print(f"  - 分类输出: {out['cls_logits'].shape}")
        print(f"  - 卡路里输出: {out['calories'].shape}")
        print(f"  - 重量输出: {out['weight'].shape}")
        
        baseline_model = BaselineResNet(config)
        x3 = torch.randn(1, 3, 224, 224)
        out2 = baseline_model(x3)
        print(f"\n✓ 基线模型构建成功")
        print(f"  - 输入: {x3.shape}")
        print(f"  - 分类输出: {out2['cls_logits'].shape}")
        
        total_params_nir = sum(p.numel() for p in nir_gen.parameters())
        total_params_mt = sum(p.numel() for p in mt_model.parameters())
        print(f"\n  NIR生成器参数量: {total_params_nir / 1e6:.2f}M")
        print(f"  多任务网络参数量: {total_params_mt / 1e6:.2f}M")
        
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []
    
    results.append(test_imports())
    results.append(test_config())
    results.append(test_dummy_data())
    results.append(test_models())
    
    print("\n" + "=" * 50)
    print("  测试结果汇总")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if all(results):
        print("\n🎉 所有测试通过！项目可以正常运行。")
        print("\n下一步：")
        print("  1. 运行 start.bat 启动演示应用")
        print("  2. 或运行 python app/server.py")
        print("  3. 在浏览器中打开 http://localhost:5000")
    else:
        print("\n❌ 部分测试失败，请检查错误信息。")
    
    input("\n按回车键退出...")
