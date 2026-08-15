"""
Checkpoint 工具函数
用于加载和保存模型检查点
"""

import random

import torch
from typing import Dict, Any, Tuple


def capture_rng_state() -> Dict[str, Any]:
    """Capture RNG state needed to continue data shuffling reproducibly."""
    state = {
        'python': random.getstate(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if 'python' in state:
        random.setstate(state['python'])
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def get_checkpoint_config(
    checkpoint_path: str,
    model_type: str = 'decoder',
    require_saved: bool = False,
) -> Dict[str, Any]:
    """Return the model config stored in a checkpoint.

    Weight-only checkpoints cannot reliably reveal ``num_heads`` because head
    count does not change the projection matrix shapes. Training resume
    therefore requires an explicitly saved config. Inference may infer the
    remaining dimensions, but callers must provide ``num_heads`` explicitly.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get('config'), dict):
        return dict(checkpoint['config'])
    if require_saved:
        raise ValueError(
            "Checkpoint 缺少 config，无法可靠恢复模型结构；"
            "请使用训练脚本保存的完整 checkpoint。"
        )
    return infer_model_config_from_checkpoint(checkpoint_path, model_type)


def get_checkpoint_training_config(checkpoint_path: str) -> Dict[str, Any]:
    """Return optional training metadata used to reconstruct the scheduler."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get('training_config'), dict):
        return dict(checkpoint['training_config'])
    return {}


def infer_model_config_from_checkpoint(checkpoint_path: str, model_type: str = 'decoder') -> Dict[str, Any]:
    """
    从 checkpoint 文件中推断模型配置

    Args:
        checkpoint_path: checkpoint 文件路径
        model_type: 模型类型 ('decoder' 或 'encoder_decoder')

    Returns:
        包含模型配置的字典
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False,
    )

    # 提取 state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    config = {}

    if model_type == 'decoder':
        # 推断 vocab_size
        if 'embed.weight' in state_dict:
            config['vocab_size'] = state_dict['embed.weight'].shape[0]

        # 推断 d_model
        if 'embed.weight' in state_dict:
            config['d_model'] = state_dict['embed.weight'].shape[1]

        # 推断 max_len (pe 的形状是 (1, max_len, d_model))
        if 'pos_enc.pe' in state_dict:
            config['max_len'] = state_dict['pos_enc.pe'].shape[1]

        # 推断 num_layers
        num_layers = 0
        while f'layers.{num_layers}.self_attn.q_linear.weight' in state_dict:
            num_layers += 1
        if num_layers > 0:
            config['num_layers'] = num_layers

        # 推断 d_ff
        if 'layers.0.ffn.linear1.weight' in state_dict:
            config['d_ff'] = state_dict['layers.0.ffn.linear1.weight'].shape[0]

    elif model_type == 'encoder_decoder':
        # 推断 src_vocab_size 和 tgt_vocab_size
        if 'src_embed.weight' in state_dict:
            config['src_vocab_size'] = state_dict['src_embed.weight'].shape[0]
        if 'tgt_embed.weight' in state_dict:
            config['tgt_vocab_size'] = state_dict['tgt_embed.weight'].shape[0]

        # 推断 d_model
        if 'src_embed.weight' in state_dict:
            config['d_model'] = state_dict['src_embed.weight'].shape[1]

        # 推断 max_len (pe 的形状是 (1, max_len, d_model))
        if 'src_pos_enc.pe' in state_dict:
            config['max_len'] = state_dict['src_pos_enc.pe'].shape[1]

        # 推断 num_layers
        num_encoder_layers = 0
        while f'encoder_layers.{num_encoder_layers}.self_attn.q_linear.weight' in state_dict:
            num_encoder_layers += 1
        if num_encoder_layers > 0:
            config['num_layers'] = num_encoder_layers

        # 推断 d_ff
        if 'encoder_layers.0.ffn.linear1.weight' in state_dict:
            config['d_ff'] = state_dict['encoder_layers.0.ffn.linear1.weight'].shape[0]

    return config


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_class,
    model_type: str = 'decoder',
    device: str = 'cpu',
    **override_config
) -> Tuple[Any, Dict[str, Any]]:
    """
    从 checkpoint 加载模型，自动推断配置

    Args:
        checkpoint_path: checkpoint 文件路径
        model_class: 模型类（DecoderOnlyModel 或 EncoderDecoderModel）
        model_type: 模型类型 ('decoder' 或 'encoder_decoder')
        device: 设备
        **override_config: 覆盖推断的配置

    Returns:
        (model, checkpoint_info) 元组
    """
    # 加载 checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # 提取 state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', None),
            'val_loss': checkpoint.get('val_loss', None),
        }
    else:
        state_dict = checkpoint
        checkpoint_info = {}

    # 优先使用保存的配置，如果没有则推断
    has_saved_config = isinstance(checkpoint, dict) and isinstance(
        checkpoint.get('config'), dict
    )
    if has_saved_config:
        config = dict(checkpoint['config'])
        print("使用 checkpoint 中保存的配置")
    else:
        print("Checkpoint 中没有保存配置，尝试从权重推断...")
        config = infer_model_config_from_checkpoint(checkpoint_path, model_type)

    # 覆盖配置
    config.update(override_config)
    if not has_saved_config and 'num_heads' not in config:
        raise ValueError(
            "Checkpoint 缺少 config，且 num_heads 无法从权重形状推断；"
            "请显式传入 num_heads，或使用包含 config 的完整 checkpoint。"
        )

    # 打印配置
    print("检测到的模型配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 创建模型
    model = model_class(**config)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # 打印 checkpoint 信息
    if checkpoint_info.get('epoch'):
        print(f"\n已加载最佳模型:")
        print(f"  Epoch: {checkpoint_info['epoch']}")
        if checkpoint_info.get('val_loss'):
            print(f"  Val Loss: {checkpoint_info['val_loss']:.4f}")

    return model, checkpoint_info


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    val_loss: float,
    save_path: str,
    scheduler=None,
    **extra_info
):
    """
    保存完整的训练 checkpoint

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        val_loss: 验证损失
        save_path: 保存路径
        scheduler: 学习率调度器（可选）
        **extra_info: 其他要保存的信息
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'rng_state': capture_rng_state(),
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # 添加额外信息
    checkpoint.update(extra_info)

    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint 已保存: {save_path}")


def load_checkpoint_for_training(
    checkpoint_path: str,
    model,
    optimizer,
    scheduler=None,
    scaler=None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    加载 checkpoint 用于恢复训练

    Args:
        checkpoint_path: checkpoint 文件路径
        model: 模型实例
        optimizer: 优化器实例
        scheduler: 学习率调度器实例（可选）
        scaler: AMP GradScaler 实例（可选）
        device: 设备

    Returns:
        包含训练信息的字典
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint 格式错误，应包含 dict")

    # 加载模型权重
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Checkpoint 缺少 model_state_dict，不能恢复训练")
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ 模型权重已加载")

    # 加载优化器状态
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ 优化器状态已加载")

    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✓ 调度器状态已加载")

    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print("✓ AMP Scaler 状态已加载")

    if isinstance(checkpoint.get('rng_state'), dict):
        restore_rng_state(checkpoint['rng_state'])
        print("✓ 随机数状态已加载")

    # 返回训练信息
    training_info = {
        'start_epoch': checkpoint.get('epoch', 0) + 1,
        'best_val_loss': checkpoint.get('val_loss', float('inf')),
    }

    print(f"\n恢复训练:")
    print(f"  从 Epoch {training_info['start_epoch']} 开始")
    print(f"  当前最佳验证损失: {training_info['best_val_loss']:.4f}")

    return training_info


def get_model_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    获取 checkpoint 的详细信息

    Args:
        checkpoint_path: checkpoint 文件路径

    Returns:
        包含模型信息的字典
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False,
    )

    info = {
        'file_path': checkpoint_path,
        'format': 'new' if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else 'old',
    }

    if isinstance(checkpoint, dict):
        if 'epoch' in checkpoint:
            info['epoch'] = checkpoint['epoch']
        if 'val_loss' in checkpoint:
            info['val_loss'] = checkpoint['val_loss']
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # 旧格式，整个 dict 就是 state_dict
            state_dict = checkpoint
        info['num_parameters'] = sum(p.numel() for p in state_dict.values())
        info['has_optimizer'] = 'optimizer_state_dict' in checkpoint
        info['has_scheduler'] = 'scheduler_state_dict' in checkpoint
    else:
        # checkpoint 本身就是 state_dict（极少见）
        state_dict = checkpoint
        info['num_parameters'] = sum(p.numel() for p in state_dict.values())
        info['has_optimizer'] = False
        info['has_scheduler'] = False

    return info
