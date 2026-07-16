#!/usr/bin/env python3
"""
训练可视化工具 - 读取日志文件并生成可视化

使用方法：
1. 训练时将输出重定向到日志文件：
   python -m scripts.train_encoder_decoder 2>&1 | tee training.log

2. 在另一个终端运行此脚本：
   python -m scripts.visualize_training training.log

3. 或者使用 watch 实时更新：
   watch -n 5 python -m scripts.visualize_training training.log
"""

import re
import sys
import matplotlib.pyplot as plt
from pathlib import Path

def parse_log_file(log_file):
    """解析训练日志文件"""
    epochs = []
    train_losses = []
    val_losses = []
    learning_rates = []

    with open(log_file, 'r') as f:
        for line in f:
            # 匹配训练损失行: "Epoch 1 Train Loss: 5.1234, LR: 3.00e-04"
            train_match = re.search(r'Epoch (\d+) Train Loss: ([\d.]+), LR: ([\d.e+-]+)', line)
            if train_match:
                epoch = int(train_match.group(1))
                train_loss = float(train_match.group(2))
                lr = float(train_match.group(3))

                epochs.append(epoch)
                train_losses.append(train_loss)
                learning_rates.append(lr)

            # 匹配验证损失行: "Epoch 1 Val Loss: 4.9876"
            val_match = re.search(r'Epoch (\d+) Val Loss: ([\d.]+)', line)
            if val_match:
                val_loss = float(val_match.group(2))
                val_losses.append(val_loss)

    return epochs, train_losses, val_losses, learning_rates

def plot_training_progress(epochs, train_losses, val_losses, learning_rates, output_file='training_progress.png'):
    """生成训练进度图"""
    if not epochs:
        print("❌ 没有找到训练数据")
        return

    # 创建 2x1 子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 第一个图：损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 添加最佳点标注
    if val_losses:
        best_epoch = epochs[val_losses.index(min(val_losses))]
        best_val_loss = min(val_losses)
        ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best: Epoch {best_epoch}')
        ax1.plot(best_epoch, best_val_loss, 'g*', markersize=15)
        ax1.text(best_epoch, best_val_loss, f'  Best: {best_val_loss:.4f}', fontsize=10, verticalalignment='bottom')

    # 第二个图：学习率曲线
    ax2.plot(epochs, learning_rates, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # 使用对数坐标

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ 训练进度图已保存到: {output_file}")

    # 打印统计信息
    print(f"\n📊 训练统计:")
    print(f"  总 Epochs: {len(epochs)}")
    print(f"  当前 Epoch: {epochs[-1]}")
    print(f"  最新 Train Loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"  最新 Val Loss: {val_losses[-1]:.4f}")
        print(f"  最佳 Val Loss: {min(val_losses):.4f} (Epoch {epochs[val_losses.index(min(val_losses))]})")
    print(f"  当前学习率: {learning_rates[-1]:.2e}")

def main():
    if len(sys.argv) < 2:
        print("用法: python -m scripts.visualize_training <log_file>")
        print("\n示例:")
        print("  # 实时记录训练日志")
        print("  python -m scripts.train_encoder_decoder 2>&1 | tee training.log")
        print()
        print("  # 在另一个终端可视化")
        print("  python -m scripts.visualize_training training.log")
        print()
        print("  # 或使用 watch 自动更新（每5秒）")
        print("  watch -n 5 python -m scripts.visualize_training training.log")
        sys.exit(1)

    log_file = sys.argv[1]

    if not Path(log_file).exists():
        print(f"❌ 日志文件不存在: {log_file}")
        sys.exit(1)

    epochs, train_losses, val_losses, learning_rates = parse_log_file(log_file)

    if epochs:
        plot_training_progress(epochs, train_losses, val_losses, learning_rates)
    else:
        print(f"⚠️  在 {log_file} 中未找到训练数据")

if __name__ == "__main__":
    main()
