"""
可视化学习率调度器
帮助理解不同调度器的行为
"""

import matplotlib.pyplot as plt
import torch
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.scheduler_utils import WarmupLRScheduler


def visualize_lr_schedule(
    num_training_steps=1000,
    warmup_ratio=0.1,
    lr=1e-3,
    scheduler_types=['cosine', 'linear', 'constant', 'polynomial']
):
    """
    可视化不同类型的学习率调度器

    Args:
        num_training_steps: 总训练步数
        warmup_ratio: warmup 比例
        lr: 初始学习率
        scheduler_types: 要可视化的调度器类型列表
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = ['blue', 'green', 'red', 'purple']

    for idx, scheduler_type in enumerate(scheduler_types):
        # 创建模型和优化器（用于测试）
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # 创建调度器
        scheduler = WarmupLRScheduler(
            optimizer,
            scheduler_type=scheduler_type,
            num_training_steps=num_training_steps,
            warmup_ratio=warmup_ratio
        )

        # 记录学习率
        lrs = []
        for step in range(num_training_steps):
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()

        # 绘图
        ax = axes[idx]
        ax.plot(range(num_training_steps), lrs, color=colors[idx], linewidth=2)
        ax.axvline(x=scheduler.num_warmup_steps, color='gray', linestyle='--', alpha=0.5, label='Warmup 结束')
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title(f'{scheduler_type.capitalize()} Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 添加信息文本
        info_text = f"Max LR: {max(lrs):.2e}\nMin LR: {min(lrs):.2e}\nWarmup: {scheduler.num_warmup_steps} steps"
        ax.text(0.95, 0.95, info_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    plt.tight_layout()
    plt.savefig('lr_schedules.png', dpi=150, bbox_inches='tight')
    print(f"学习率调度可视化已保存到: lr_schedules.png")
    plt.show()


def compare_warmup_ratios(
    num_training_steps=1000,
    warmup_ratios=[0.05, 0.1, 0.2, 0.3],
    lr=1e-3,
    scheduler_type='cosine'
):
    """
    比较不同 warmup 比例的效果

    Args:
        num_training_steps: 总训练步数
        warmup_ratios: warmup 比例列表
        lr: 初始学习率
        scheduler_type: 调度器类型
    """
    plt.figure(figsize=(10, 6))

    for warmup_ratio in warmup_ratios:
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        scheduler = WarmupLRScheduler(
            optimizer,
            scheduler_type=scheduler_type,
            num_training_steps=num_training_steps,
            warmup_ratio=warmup_ratio
        )

        lrs = []
        for step in range(num_training_steps):
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()

        plt.plot(range(num_training_steps), lrs, label=f'Warmup: {warmup_ratio:.0%}', linewidth=2)

    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title(f'{scheduler_type.capitalize()} Schedule - Different Warmup Ratios', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('warmup_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Warmup 比较图已保存到: warmup_comparison.png")
    plt.show()


if __name__ == "__main__":
    print("生成学习率调度可视化...")

    # 可视化不同调度器
    visualize_lr_schedule(
        num_training_steps=1000,
        warmup_ratio=0.1,
        lr=3e-4
    )

    # 比较不同 warmup 比例
    compare_warmup_ratios(
        num_training_steps=1000,
        warmup_ratios=[0.05, 0.1, 0.2, 0.3],
        lr=3e-4,
        scheduler_type='cosine'
    )
