import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.mask_utils import collate_fn_mt, create_padding_mask, create_causal_mask, combine_masks
from utils.scheduler_utils import WarmupLRScheduler
from models.transformer_models import EncoderDecoderModel
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

def collate_fn_with_padding(src_batch, tgt_batch, pad_token_id=0, max_seq_len=1024):
    """
    动态 padding collate function for encoder-decoder
    将不同长度的源和目标序列分别 pad 到各自 batch 内最大长度

    Args:
        src_batch: List[List[int]] - 源语言序列列表
        tgt_batch: List[List[int]] - 目标语言序列列表
        pad_token_id: padding token 的 id
        max_seq_len: 最大序列长度，超过此长度的序列会被截断（防御性编程）

    Returns:
        src_tensor: (batch, src_max_len) - padding 后的源序列
        tgt_tensor: (batch, tgt_max_len) - padding 后的目标序列
    """
    # 截断过长的序列（业界标准做法：防御性编程）
    src_batch = [x[:max_seq_len] if len(x) > max_seq_len else x for x in src_batch]
    tgt_batch = [x[:max_seq_len] if len(x) > max_seq_len else x for x in tgt_batch]

    # 找到各自的最大长度
    src_max_len = max(len(x) for x in src_batch)
    tgt_max_len = max(len(x) for x in tgt_batch)

    # 创建 padding 后的 tensor
    src_tensor = torch.full((len(src_batch), src_max_len), pad_token_id, dtype=torch.long)
    tgt_tensor = torch.full((len(tgt_batch), tgt_max_len), pad_token_id, dtype=torch.long)

    # 填充每个序列
    for i, seq in enumerate(src_batch):
        src_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    for i, seq in enumerate(tgt_batch):
        tgt_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    return src_tensor, tgt_tensor

def train_encoder_decoder(
    data_dir="data/iwslt2017",
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
    src_vocab_size = tgt_vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for split in ["train", "validation"]:
        src_data = load_dataset_pt(os.path.join(data_dir, f"{split}.en_ids.pt"))
        tgt_data = load_dataset_pt(os.path.join(data_dir, f"{split}.zh_ids.pt"))
        if split == "train":
            train_src, train_tgt = src_data, tgt_data
        else:
            val_src, val_tgt = src_data, tgt_data

    # 数据验证（业界最佳实践：训练前检查数据）
    print("\n" + "="*60)
    print("数据集验证")
    print("="*60)
    validate_dataset(train_src, max_len, "训练集 (源语言)")
    validate_dataset(train_tgt, max_len, "训练集 (目标语言)")
    validate_dataset(val_src, max_len, "验证集 (源语言)")
    validate_dataset(val_tgt, max_len, "验证集 (目标语言)")
    print("="*60)

    # 使用带 padding 的 collate_fn，传入 max_len 参数
    train_loader = DataLoader(
        list(zip(train_src, train_tgt)),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_with_padding(
            [x[0] for x in batch],
            [x[1] for x in batch],
            pad_token_id,
            max_len
        )
    )
    val_loader = DataLoader(
        list(zip(val_src, val_tgt)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_with_padding(
            [x[0] for x in batch],
            [x[1] for x in batch],
            pad_token_id,
            max_len
        )
    )
    model = EncoderDecoderModel(src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len).to(device)
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
    best_model_path = "encoder_decoder_best.pt"

    # TensorBoard 日志
    log_dir = f'runs/enc_dec_{d_model}d_{num_layers}L_{num_heads}H_bs{batch_size}'
    writer = SummaryWriter(log_dir=log_dir)
    print(f"\n📊 TensorBoard 日志目录: {log_dir}")
    print(f"💡 启动 TensorBoard 查看训练进度:")
    print(f"   tensorboard --logdir=runs --port=6006")
    print(f"   然后访问: http://localhost:6006\n")

    try:
        for epoch in range(1, epochs+1):
            model.train()
            total_loss = 0
            for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch} - Train"):
                src, tgt = src.to(device), tgt.to(device)
                # tgt: [B, L]，输入为[:-1]，目标为[1:]
                tgt_input = tgt[:, :-1]

                # 创建 mask：需要同时考虑因果关系和 padding
                src_mask = create_padding_mask(src, pad_token_id=pad_token_id)
                tgt_causal_mask = create_causal_mask(tgt_input.size(1), device=src.device)
                tgt_padding_mask = create_padding_mask(tgt_input, pad_token_id=pad_token_id)
                tgt_mask = combine_masks(tgt_causal_mask, tgt_padding_mask)
                cross_mask = create_padding_mask(src, pad_token_id=pad_token_id)

                logits, _ = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)
                loss = criterion(logits.view(-1, tgt_vocab_size), tgt[:, 1:].reshape(-1))
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
                for src, tgt in tqdm(val_loader, desc=f"Epoch {epoch} - Val"):
                    src, tgt = src.to(device), tgt.to(device)
                    tgt_input = tgt[:, :-1]

                    # 创建 mask：需要同时考虑因果关系和 padding
                    src_mask = create_padding_mask(src, pad_token_id=pad_token_id)
                    tgt_causal_mask = create_causal_mask(tgt_input.size(1), device=src.device)
                    tgt_padding_mask = create_padding_mask(tgt_input, pad_token_id=pad_token_id)
                    tgt_mask = combine_masks(tgt_causal_mask, tgt_padding_mask)
                    cross_mask = create_padding_mask(src, pad_token_id=pad_token_id)

                    logits, _ = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)
                    loss = criterion(logits.view(-1, tgt_vocab_size), tgt[:, 1:].reshape(-1))
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
                        'src_vocab_size': src_vocab_size,
                        'tgt_vocab_size': tgt_vocab_size,
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
        interrupt_path = "encoder_decoder_interrupted.pt"
        checkpoint = {
            'epoch': epoch if 'epoch' in locals() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'config': {
                'src_vocab_size': src_vocab_size,
                'tgt_vocab_size': tgt_vocab_size,
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
        error_path = "encoder_decoder_error.pt"
        checkpoint = {
            'epoch': epoch if 'epoch' in locals() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'config': {
                'src_vocab_size': src_vocab_size,
                'tgt_vocab_size': tgt_vocab_size,
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
    train_encoder_decoder(epochs=100)
