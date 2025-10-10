"""
恢复训练的辅助脚本
用于从保存的 checkpoint 继续训练
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
from utils.checkpoint_utils import get_model_info, infer_model_config_from_checkpoint


def resume_decoder_training(checkpoint_path, additional_epochs=10):
    """从 checkpoint 恢复 decoder-only 模型训练"""
    print(f"从 {checkpoint_path} 恢复训练...")
    checkpoint = torch.load(checkpoint_path)

    print(f"上次训练到 Epoch {checkpoint['epoch']}")
    print(f"最佳验证损失: {checkpoint['val_loss']:.4f}")

    # 这里可以继续训练，需要重新加载模型和优化器
    print(f"\n将继续训练 {additional_epochs} 个 epoch")
    print("注意: 需要在训练脚本中实现恢复训练功能")


def resume_encoder_decoder_training(checkpoint_path, additional_epochs=10):
    """从 checkpoint 恢复 encoder-decoder 模型训练"""
    print(f"从 {checkpoint_path} 恢复训练...")
    checkpoint = torch.load(checkpoint_path)

    print(f"上次训练到 Epoch {checkpoint['epoch']}")
    print(f"最佳验证损失: {checkpoint['val_loss']:.4f}")

    print(f"\n将继续训练 {additional_epochs} 个 epoch")
    print("注意: 需要在训练脚本中实现恢复训练功能")


def show_checkpoint_info(checkpoint_path):
    """显示 checkpoint 信息"""
    print(f"\n{'='*60}")
    print(f"Checkpoint 信息")
    print(f"{'='*60}")

    info = get_model_info(checkpoint_path)

    print(f"文件路径: {info['file_path']}")
    print(f"格式: {info['format']} ({'新格式' if info['format'] == 'new' else '旧格式（仅权重）'})")

    if 'epoch' in info:
        print(f"训练轮次: Epoch {info['epoch']}")
    if 'val_loss' in info:
        print(f"验证损失: {info['val_loss']:.4f}")

    print(f"模型参数总数: {info['num_parameters']:,}")
    print(f"包含优化器状态: {'✓' if info['has_optimizer'] else '✗'}")
    print(f"包含调度器状态: {'✓' if info['has_scheduler'] else '✗'}")

    print(f"\n{'='*60}")
    print("推断的模型配置:")
    print(f"{'='*60}")

    # 尝试推断配置（先假设是 decoder）
    try:
        config = infer_model_config_from_checkpoint(checkpoint_path, 'decoder')
        if config:
            print("模型类型: Decoder-Only")
            for key, value in config.items():
                print(f"  {key}: {value}")
    except:
        pass

    # 尝试 encoder-decoder
    try:
        config = infer_model_config_from_checkpoint(checkpoint_path, 'encoder_decoder')
        if config and 'src_vocab_size' in config:
            print("模型类型: Encoder-Decoder")
            for key, value in config.items():
                print(f"  {key}: {value}")
    except:
        pass

    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='查看或恢复训练 checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint 文件路径')
    parser.add_argument('--action', type=str, choices=['info', 'resume'], default='info',
                        help='操作类型: info (查看信息) 或 resume (恢复训练)')
    parser.add_argument('--model_type', type=str, choices=['decoder', 'encoder_decoder'],
                        help='模型类型 (恢复训练时必需)')
    parser.add_argument('--epochs', type=int, default=10, help='继续训练的 epoch 数')

    args = parser.parse_args()

    if args.action == 'info':
        show_checkpoint_info(args.checkpoint)
    elif args.action == 'resume':
        if not args.model_type:
            print("错误: 恢复训练需要指定 --model_type")
        elif args.model_type == 'decoder':
            resume_decoder_training(args.checkpoint, args.epochs)
        else:
            resume_encoder_decoder_training(args.checkpoint, args.epochs)
