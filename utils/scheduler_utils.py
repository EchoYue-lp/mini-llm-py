"""
学习率调度器工具
包含 warmup 和 cosine decay
"""

import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=0.5,
    last_epoch=-1
):
    """
    创建带 warmup 的 cosine decay 学习率调度器

    Args:
        optimizer: PyTorch 优化器
        num_warmup_steps: warmup 步数
        num_training_steps: 总训练步数
        num_cycles: cosine 周期数 (默认 0.5，即半个周期)
        last_epoch: 上一个 epoch 编号

    Returns:
        LambdaLR 调度器
    """

    def lr_lambda(current_step):
        # Warmup 阶段：线性增长
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay 阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    last_epoch=-1
):
    """
    创建带 warmup 的线性 decay 学习率调度器

    Args:
        optimizer: PyTorch 优化器
        num_warmup_steps: warmup 步数
        num_training_steps: 总训练步数
        last_epoch: 上一个 epoch 编号

    Returns:
        LambdaLR 调度器
    """

    def lr_lambda(current_step):
        # Warmup 阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # 线性 decay 阶段
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    last_epoch=-1
):
    """
    创建带 warmup 的恒定学习率调度器
    warmup 后保持恒定学习率

    Args:
        optimizer: PyTorch 优化器
        num_warmup_steps: warmup 步数
        last_epoch: 上一个 epoch 编号

    Returns:
        LambdaLR 调度器
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    lr_end=0.0,
    power=1.0,
    last_epoch=-1
):
    """
    创建带 warmup 的多项式 decay 学习率调度器

    Args:
        optimizer: PyTorch 优化器
        num_warmup_steps: warmup 步数
        num_training_steps: 总训练步数
        lr_end: 最终学习率
        power: 多项式的幂次
        last_epoch: 上一个 epoch 编号

    Returns:
        LambdaLR 调度器
    """

    lr_init = optimizer.defaults["lr"]
    assert lr_init > lr_end, f"lr_end ({lr_end}) 必须小于初始学习率 ({lr_init})"

    def lr_lambda(current_step):
        # Warmup 阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # 多项式 decay 阶段
        lr_range = lr_init - lr_end
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        pct_remaining = 1 - min(max(progress, 0.0), 1.0)
        return (lr_range * pct_remaining ** power + lr_end) / lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class WarmupLRScheduler:
    """
    封装的学习率调度器类，便于使用和可视化
    """

    def __init__(
        self,
        optimizer,
        scheduler_type='cosine',
        num_warmup_steps=None,
        num_training_steps=None,
        warmup_ratio=0.1,
        **kwargs
    ):
        """
        Args:
            optimizer: PyTorch 优化器
            scheduler_type: 调度器类型 ('cosine', 'linear', 'constant', 'polynomial')
            num_warmup_steps: warmup 步数 (如果为 None，使用 warmup_ratio 计算)
            num_training_steps: 总训练步数
            warmup_ratio: warmup 占总步数的比例 (当 num_warmup_steps 为 None 时使用)
            **kwargs: 其他调度器参数
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type

        # 计算 warmup 步数
        if num_warmup_steps is None:
            if num_training_steps is None:
                raise ValueError("必须指定 num_warmup_steps 或 (num_training_steps + warmup_ratio)")
            num_warmup_steps = int(num_training_steps * warmup_ratio)

        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        # 创建调度器
        if scheduler_type == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps, **kwargs
            )
        elif scheduler_type == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps, **kwargs
            )
        elif scheduler_type == 'constant':
            self.scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps, **kwargs
            )
        elif scheduler_type == 'polynomial':
            self.scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps, **kwargs
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")

    def step(self):
        """更新学习率"""
        self.scheduler.step()

    def get_last_lr(self):
        """获取当前学习率"""
        return self.scheduler.get_last_lr()

    def state_dict(self):
        """保存调度器状态"""
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.scheduler.load_state_dict(state_dict)
