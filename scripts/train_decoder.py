import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.transformer_models import DecoderOnlyModel
from utils.mask_utils import create_causal_mask, create_padding_mask, combine_masks
from utils.scheduler_utils import WarmupLRScheduler
from transformers import GPT2TokenizerFast
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def load_dataset_pt(pt_file):
    # 使用 weights_only=False 加载数据（数据来源可信）
    return torch.load(pt_file, weights_only=False)

def validate_dataset(data, max_len, dataset_name="dataset"):
    """
    验证数据集中的序列长度，报告统计信息

    Args:
        data: List[List[int]] - 数据集
        max_len: int - 最大允许长度
        dataset_name: str - 数据集名称（用于日志）
    """
    if not data:
        print(f"警告: {dataset_name} 为空")
        return

    lengths = [len(seq) for seq in data]
    min_len = min(lengths)
    max_seq_len = max(lengths)
    avg_len = sum(lengths) / len(lengths)

    # 统计超长序列
    over_limit = sum(1 for l in lengths if l > max_len)

    print(f"\n{dataset_name} 统计:")
    print(f"  样本数: {len(data)}")
    print(f"  长度范围: [{min_len}, {max_seq_len}]")
    print(f"  平均长度: {avg_len:.1f}")

    if over_limit > 0:
        print(f"  ⚠️  超过 max_len={max_len} 的序列: {over_limit}/{len(data)} ({over_limit/len(data)*100:.1f}%)")
        print(f"  → 这些序列将被自动截断")
    else:
        print(f"  ✓ 所有序列都在 max_len={max_len} 范围内")

def collate_fn(batch, pad_token_id=0, max_seq_len=1024):
    """
    动态 padding collate function
    将不同长度的序列 pad 到 batch 内最大长度

    Args:
        batch: List[List[int]] - 列表，每个元素是一个 token id 序列
        pad_token_id: padding token 的 id
        max_seq_len: 最大序列长度，超过此长度的序列会被截断（防御性编程）

    Returns:
        input_ids: (batch, max_len-1) - 输入序列（去掉最后一个 token）
        target_ids: (batch, max_len-1) - 目标序列（去掉第一个 token）
    """
    # 截断过长的序列（业界标准做法：防御性编程）
    batch = [x[:max_seq_len] if len(x) > max_seq_len else x for x in batch]

    # 找到 batch 内最大长度
    max_len = max(len(x) for x in batch)

    # 创建 padding 后的 tensor
    batch_tensor = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)

    # 填充每个序列
    for i, seq in enumerate(batch):
        seq_len = len(seq)
        batch_tensor[i, :seq_len] = torch.tensor(seq, dtype=torch.long)

    # input: 去掉最后一个 token，target: 去掉第一个 token
    return batch_tensor[:, :-1], batch_tensor[:, 1:]

def train_decoder_only(
    data_dir="data/wikitext2",
    tokenizer_dir="tokenization/gpt2",
    d_model=256,
    num_layers=4,
    num_heads=8,
    d_ff=1024,
    max_len=96,
    batch_size=64,
    epochs=100,
    lr=3e-4,
    warmup_ratio=0.1,
    use_scheduler=True,
    scheduler_type='cosine',
    max_grad_norm=1.0,
    device=None
):
    device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    train_data = load_dataset_pt(os.path.join(data_dir, "train_ids.pt"))
    val_data = load_dataset_pt(os.path.join(data_dir, "validation_ids.pt"))

    # 数据验证（业界最佳实践：训练前检查数据）
    print("\n" + "="*60)
    print("数据集验证")
    print("="*60)
    validate_dataset(train_data, max_len, "训练集")
    validate_dataset(val_data, max_len, "验证集")
    print("="*60)

    # 使用 lambda 传递 pad_token_id 和 max_len 参数
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id, max_len)
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id, max_len)
    )
    model = DecoderOnlyModel(vocab_size, d_model, num_layers, num_heads, d_ff, max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 添加 ignore_index 以忽略 padding token 的损失
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # 学习率调度器
    scheduler = None
    if use_scheduler:
        num_training_steps = len(train_loader) * epochs
        scheduler = WarmupLRScheduler(
            optimizer,
            scheduler_type=scheduler_type,
            num_training_steps=num_training_steps,
            warmup_ratio=warmup_ratio
        )
        print(f"使用 {scheduler_type} 学习率调度器")
        print(f"Warmup 步数: {scheduler.num_warmup_steps}")
        print(f"总训练步数: {num_training_steps}")

    # 保存最佳模型
    best_val_loss = float('inf')
    best_model_path = "decoder_only_best.pt"

    # TensorBoard 日志
    log_dir = f'runs/decoder_{d_model}d_{num_layers}L_{num_heads}H_bs{batch_size}'
    writer = SummaryWriter(log_dir=log_dir)
    print(f"\n📊 TensorBoard 日志目录: {log_dir}")
    print(f"💡 启动 TensorBoard 查看训练进度:")
    print(f"   tensorboard --logdir=runs --port=6006")
    print(f"   然后访问: http://localhost:6006\n")

    try:
        for epoch in range(1, epochs+1):
            model.train()
            total_loss = 0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch} - Train"):
                x, y = x.to(device), y.to(device)
                # 创建 mask：需要同时考虑因果关系和 padding
                causal_mask = create_causal_mask(x.size(1), device=device)
                padding_mask = create_padding_mask(x, pad_token_id=pad_token_id)
                mask = combine_masks(causal_mask, padding_mask)
                logits, _ = model(x, mask=mask)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

                # 更新学习率
                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} Train Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")

            # 记录训练指标到 TensorBoard
            writer.add_scalar('Loss/train', avg_loss, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"Epoch {epoch} - Val"):
                    x, y = x.to(device), y.to(device)
                    # 创建 mask：需要同时考虑因果关系和 padding
                    causal_mask = create_causal_mask(x.size(1), device=device)
                    padding_mask = create_padding_mask(x, pad_token_id=pad_token_id)
                    mask = combine_masks(causal_mask, padding_mask)
                    logits, _ = model(x, mask=mask)
                    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")

            # 记录验证指标到 TensorBoard
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            writer.add_scalars('Loss/train_vs_val', {
                'train': avg_loss,
                'validation': avg_val_loss
            }, epoch)

            # 如果当前验证损失更低，保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'config': {
                        'vocab_size': vocab_size,
                        'd_model': d_model,
                        'num_layers': num_layers,
                        'num_heads': num_heads,
                        'd_ff': d_ff,
                        'max_len': max_len,
                        'dropout': 0.1,
                    }
                }
                if scheduler is not None:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(checkpoint, best_model_path)
                print(f"✓ 最佳模型已保存: {best_model_path} (Val Loss: {avg_val_loss:.4f})")
            else:
                print(f"  当前最佳 Val Loss: {best_val_loss:.4f}")

        print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")
        writer.close()

    except KeyboardInterrupt:
        print("\n\n训练被用户中断！")
        # 保存当前状态
        interrupt_path = "decoder_only_interrupted.pt"
        checkpoint = {
            'epoch': epoch if 'epoch' in locals() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'config': {
                'vocab_size': vocab_size,
                'd_model': d_model,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'd_ff': d_ff,
                'max_len': max_len,
                'dropout': 0.1,
            }
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint, interrupt_path)
        print(f"✓ 中断时的模型已保存: {interrupt_path}")
        print(f"当前最佳验证损失: {best_val_loss:.4f}")
        writer.close()

    except Exception as e:
        print(f"\n\n训练过程中发生错误: {type(e).__name__}: {e}")
        # 保存当前状态以便调试
        error_path = "decoder_only_error.pt"
        checkpoint = {
            'epoch': epoch if 'epoch' in locals() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'config': {
                'vocab_size': vocab_size,
                'd_model': d_model,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'd_ff': d_ff,
                'max_len': max_len,
                'dropout': 0.1,
            }
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint, error_path)
        print(f"✓ 错误时的模型已保存: {error_path}")
        writer.close()
        raise  # 重新抛出异常以便查看完整堆栈跟踪

if __name__ == "__main__":
    train_decoder_only()
