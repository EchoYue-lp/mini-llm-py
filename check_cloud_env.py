#!/usr/bin/env python3
"""
云平台环境检查脚本
在云平台上运行此脚本以验证环境配置是否正确
"""

import sys
import torch
import platform

def check_python_version():
    """检查Python版本"""
    print("=" * 60)
    print("检查Python版本...")
    print("=" * 60)
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 11:
        print("✅ Python版本正常（推荐3.11）")
        return True
    elif version.major == 3 and version.minor >= 10:
        print("⚠️  Python版本可用但建议升级到3.11")
        return True
    else:
        print("❌ Python版本过低，需要3.10+")
        return False

def check_pytorch():
    """检查PyTorch版本"""
    print("\n" + "=" * 60)
    print("检查PyTorch...")
    print("=" * 60)
    print(f"PyTorch版本: {torch.__version__}")

    version_parts = torch.__version__.split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1])

    if major >= 2:
        print("✅ PyTorch版本正常（推荐2.5.1+）")
        return True
    else:
        print("❌ PyTorch版本过低，需要2.0+")
        return False

def check_cuda():
    """检查CUDA"""
    print("\n" + "=" * 60)
    print("检查CUDA...")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")

    if cuda_available:
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            # 获取GPU显存信息
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  显存总量: {mem_total:.2f} GB")
            print(f"  已分配: {mem_allocated:.2f} GB")
            print(f"  已保留: {mem_reserved:.2f} GB")
        print("✅ CUDA正常（云平台推荐）")
        return True
    else:
        print("⚠️  CUDA不可用")
        # 检查是否有其他设备
        if torch.backends.mps.is_available():
            print("ℹ️  检测到MPS（Apple Silicon）")
            return True
        else:
            print("ℹ️  将使用CPU（训练会很慢）")
            return True

def check_dependencies():
    """检查其他依赖"""
    print("\n" + "=" * 60)
    print("检查其他依赖...")
    print("=" * 60)

    dependencies = {
        'transformers': '用于tokenizer',
        'tqdm': '进度条',
        'sacrebleu': '评估指标（可选）',
    }

    all_ok = True
    for module, desc in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {module:15s} - {desc}")
        except ImportError:
            print(f"❌ {module:15s} - 未安装（{desc}）")
            all_ok = False

    if not all_ok:
        print("\n提示：运行以下命令安装缺失依赖：")
        print("pip install -r requirements.txt")

    return all_ok

def test_device_computation():
    """测试当前可用设备上的基础计算。"""
    print("\n" + "=" * 60)
    print("测试设备计算...")
    print("=" * 60)

    try:
        device = "cuda" if torch.cuda.is_available() else \
                 "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"使用设备: {device}")

        # 创建简单张量并计算
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.matmul(x, y)

        print(f"矩阵乘法测试: {z.shape}")
        print("✅ 设备计算正常")
        return True
    except Exception as e:
        print(f"❌ 设备计算失败: {e}")
        return False

def check_model_files():
    """检查模型相关文件"""
    print("\n" + "=" * 60)
    print("检查项目文件...")
    print("=" * 60)

    import os

    required_dirs = [
        'models',
        'utils',
        'scripts',
        'finetuning',
        'inference',
        'evaluation',
        'tests',
    ]

    optional_dirs = [
        'data/wikitext2',
        'tokenization/gpt2',
    ]

    all_ok = True
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"✅ {dir_name:20s} - 存在")
        else:
            print(f"❌ {dir_name:20s} - 缺失（必需）")
            all_ok = False

    for dir_name in optional_dirs:
        if os.path.isdir(dir_name):
            print(f"✅ {dir_name:20s} - 存在")
        else:
            print(f"ℹ️  {dir_name:20s} - 缺失（需要运行数据准备脚本）")

    return all_ok

def run_smoke_tests():
    """Run the lightweight model smoke tests through pytest."""
    print("\n" + "=" * 60)
    print("运行模型 smoke tests...")
    print("=" * 60)

    try:
        import importlib.util
        import subprocess
        import sys

        if importlib.util.find_spec('pytest') is None:
            print("ℹ️  未安装 pytest，跳过 smoke tests")
            print("   开发环境可运行: python -m pip install -r requirements-dev.txt")
            return True

        result = subprocess.run(
            [
                sys.executable,
                '-m',
                'pytest',
                '-q',
                'tests/test_model_basics.py',
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print("✅ 模型 smoke tests 通过")
            return True
        else:
            print("❌ 测试失败")
            if result.stdout:
                print("标准输出:")
                print(result.stdout[:500])  # 只显示前500字符
            if result.stderr:
                print("错误输出:")
                print(result.stderr[:500])  # 只显示前500字符
            return False
    except Exception as e:
        print(f"⚠️  无法运行测试: {e}")
        print("提示: 可以手动运行: python -m pytest tests/test_model_basics.py")
        return False

def print_recommendations():
    """打印建议的训练配置"""
    print("\n" + "=" * 60)
    print("推荐配置")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️  未检测到CUDA，训练会很慢")
        print("推荐配置（CPU）:")
        print("  d_model=64, num_layers=2, batch_size=8")
        return

    # 获取GPU显存
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"检测到GPU显存: {total_memory:.2f} GB")
    print()

    if total_memory < 12:
        print("小显存配置（< 12GB）:")
        print("  d_model=128, num_layers=2, num_heads=4")
        print("  d_ff=512, batch_size=32, max_len=128")
    elif total_memory < 20:
        print("中等显存配置（12-20GB）:")
        print("  d_model=256, num_layers=4, num_heads=8")
        print("  d_ff=1024, batch_size=32, max_len=256")
    elif total_memory < 35:
        print("大显存配置（20-35GB）:")
        print("  d_model=512, num_layers=6, num_heads=8")
        print("  d_ff=2048, batch_size=64, max_len=512")
    else:
        print("超大显存配置（> 35GB）:")
        print("  d_model=768, num_layers=12, num_heads=12")
        print("  d_ff=3072, batch_size=128, max_len=512")

    print()
    print("提示：如果遇到OOM错误，请减小batch_size或d_model")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("云平台环境检查脚本")
    print("项目要求: PyTorch 2.0+ + Python 3.10+")
    print("=" * 60)

    checks = []

    # 运行所有检查
    checks.append(("Python版本", check_python_version()))
    checks.append(("PyTorch", check_pytorch()))
    checks.append(("CUDA", check_cuda()))
    checks.append(("依赖", check_dependencies()))
    checks.append(("设备计算", test_device_computation()))
    checks.append(("项目文件", check_model_files()))
    checks.append(("模型 smoke tests", run_smoke_tests()))

    # 总结
    print("\n" + "=" * 60)
    print("检查结果总结")
    print("=" * 60)

    for check_name, result in checks:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name:15s}: {status}")

    all_passed = all(result for _, result in checks)

    print()
    if all_passed:
        print("🎉 所有检查通过！环境配置正确，可以开始训练！")
        print()
        print("下一步操作请查看: README.md")
    else:
        print("⚠️  部分检查未通过，请根据上述提示修复问题")
        print()
        print("常见解决方案：")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 检查CUDA: 确认云平台GPU正确配置")
        print("3. 检查文件: 确认代码完整上传")

    # 打印推荐配置
    print_recommendations()

    print()
    print("学习路线请查看: docs/00-learning-path-and-code-map.md")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
