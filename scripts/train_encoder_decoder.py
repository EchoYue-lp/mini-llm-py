import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.mask_utils import collate_fn_mt, create_padding_mask, create_causal_mask, combine_masks
from utils.scheduler_utils import WarmupLRScheduler
from utils.translation_utils import beam_search_translate
from models.transformer_models import EncoderDecoderModel
from utils.sentencepiece_tokenizer import SentencePieceTokenizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def load_dataset_pt(pt_file):
    # 使用 weights_only=False 加载数据（数据来源可信）
    return torch.load(pt_file, weights_only=False)

def validate_dataset(data, max_len, dataset_name="dataset"):
    """验证数据集"""
    if not data:
        print(f"⚠ {dataset_name} 为空")
        return

    lengths = [len(seq) for seq in data]
    avg_len = sum(lengths) / len(lengths)
    max_seq_len = max(lengths)
    over_limit = sum(1 for l in lengths if l > max_len)

    status = f"✓" if over_limit == 0 else f"⚠ {over_limit} 条超长"
    print(f"{dataset_name}: {len(data):,} 条, 平均 {avg_len:.1f} tokens, 最大 {max_seq_len} [{status}]")

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

def test_translation_demo(model, tokenizer, demo_sentences, device):
    """在每个 epoch 结束时测试翻译效果"""
    model.eval()
    print("\n" + "─" * 60)
    print("Demo 翻译测试:")
    with torch.no_grad():
        for en_text in demo_sentences:
            try:
                src_ids = tokenizer.encode(en_text, add_special_tokens=False)
                zh_text = beam_search_translate(
                    model, src_ids, tokenizer,
                    beam_width=3, max_len=50, device=device
                )
                print(f"  EN: {en_text}")
                print(f"  ZH: {zh_text}")
            except Exception as e:
                print(f"  EN: {en_text}")
                print(f"  ZH: [翻译失败: {e}]")
    print("─" * 60)


def train_encoder_decoder(
    data_dir="data/iwslt2017",
    tokenizer_path="tokenization/sentencepiece_enzh.model",
    d_model=128,
    num_layers=6,
    num_heads=4,
    d_ff=512,
    max_len=96,
    batch_size=64,
    gradient_accumulation_steps=6,
    epochs=10,
    lr=2e-4,
    warmup_ratio=0.05,
    use_scheduler=True,
    scheduler_type='cosine',
    max_grad_norm=1.0,
    use_amp=True,
    gradient_checkpointing=False,
    device=None,
    demo_sentences=None
):
    device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SentencePieceTokenizer.from_pretrained(tokenizer_path)
    src_vocab_size = tgt_vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id

    # 默认 demo 句子
    if demo_sentences is None:
        demo_sentences = [
            "Hello, how are you?",
            "I love machine learning."
        ]

    for split in ["train", "validation"]:
        src_data = load_dataset_pt(os.path.join(data_dir, f"{split}.en_ids_sp.pt"))
        tgt_data = load_dataset_pt(os.path.join(data_dir, f"{split}.zh_ids_sp.pt"))
        if split == "train":
            train_src, train_tgt = src_data, tgt_data
        else:
            val_src, val_tgt = src_data, tgt_data

    # 数据验证
    validate_dataset(train_src, max_len, "训练集 (源)")
    validate_dataset(train_tgt, max_len, "训练集 (目标)")
    validate_dataset(val_src, max_len, "验证集 (源)")
    validate_dataset(val_tgt, max_len, "验证集 (目标)")

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

    # Gradient checkpointing (节省显存但会降低训练速度)
    if gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("✓ 启用 Gradient Checkpointing")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 添加 ignore_index 以忽略 padding token 的损失
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # 初始化 AMP Scaler
    scaler = None
    if use_amp and device == "cuda":
        scaler = torch.cuda.amp.GradScaler()
    elif use_amp:
        use_amp = False

    # 训练配置
    effective_batch_size = batch_size * gradient_accumulation_steps
    print(f"\n配置: batch={batch_size}×{gradient_accumulation_steps}={effective_batch_size}, AMP={'✓' if use_amp else '✗'}, GradCP={'✓' if gradient_checkpointing else '✗'}, device={device}")

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

    # 保存最佳模型
    best_val_loss = float('inf')
    best_model_path = "encoder_decoder_best.pt"

    # TensorBoard 日志
    amp_suffix = "_amp" if use_amp else ""
    log_dir = f'/hy-tmp/Net/logs/enc_dec_{d_model}d_{num_layers}L_{num_heads}H_bs{batch_size}x{gradient_accumulation_steps}{amp_suffix}'
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard: {log_dir}")

    try:
        for epoch in range(1, epochs+1):
            model.train()
            total_loss = 0
            optimizer.zero_grad()  # 在 epoch 开始时清零梯度

            for batch_idx, (src, tgt) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} - Train")):
                src, tgt = src.to(device), tgt.to(device)
                # tgt: [B, L]，输入为[:-1]，目标为[1:]
                tgt_input = tgt[:, :-1]

                # 创建 mask：需要同时考虑因果关系和 padding
                src_mask = create_padding_mask(src, pad_token_id=pad_token_id)
                tgt_causal_mask = create_causal_mask(tgt_input.size(1), device=src.device)
                tgt_padding_mask = create_padding_mask(tgt_input, pad_token_id=pad_token_id)
                tgt_mask = combine_masks(tgt_causal_mask, tgt_padding_mask)
                cross_mask = create_padding_mask(src, pad_token_id=pad_token_id)

                # 使用 AMP 进行前向和反向传播
                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits, _ = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)
                        loss = criterion(logits.view(-1, tgt_vocab_size), tgt[:, 1:].reshape(-1))
                        # 梯度累积：缩放 loss
                        loss = loss / gradient_accumulation_steps

                    scaler.scale(loss).backward()
                else:
                    logits, _ = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)
                    loss = criterion(logits.view(-1, tgt_vocab_size), tgt[:, 1:].reshape(-1))
                    # 梯度累积：缩放 loss
                    loss = loss / gradient_accumulation_steps
                    loss.backward()

                # 累积真实 loss（不缩放）用于日志记录
                total_loss += loss.item() * gradient_accumulation_steps

                # 每 gradient_accumulation_steps 步更新一次参数
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    if max_grad_norm > 0:
                        if use_amp and scaler is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()
                    else:
                        if use_amp and scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                    optimizer.zero_grad()

                    # 更新学习率（每次参数更新后）
                    if scheduler is not None:
                        scheduler.step()

            # 处理最后一个不完整的累积批次
            if len(train_loader) % gradient_accumulation_steps != 0:
                if max_grad_norm > 0:
                    if use_amp and scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                else:
                    if use_amp and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                optimizer.zero_grad()

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

                    # 使用 AMP 进行验证
                    if use_amp and scaler is not None:
                        with torch.cuda.amp.autocast():
                            logits, _ = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)
                            loss = criterion(logits.view(-1, tgt_vocab_size), tgt[:, 1:].reshape(-1))
                    else:
                        logits, _ = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)
                        loss = criterion(logits.view(-1, tgt_vocab_size), tgt[:, 1:].reshape(-1))

                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")

            # Demo 翻译测试
            test_translation_demo(model, tokenizer, demo_sentences, device)
            model.train()  # 切回训练模式

            # 记录验证指标到 TensorBoard
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            writer.add_scalars('Loss/train_vs_val', {
                'train': avg_loss,
                'validation': avg_val_loss
            }, epoch)

            # 保存最佳模型
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
                print(f"✓ 保存最佳模型 (Val Loss: {avg_val_loss:.4f})")

        print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")
        writer.close()

    except KeyboardInterrupt:
        print("\n中断！已保存到 encoder_decoder_interrupted.pt")
        torch.save({
            'epoch': epoch if 'epoch' in locals() else 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'config': {'src_vocab_size': src_vocab_size, 'tgt_vocab_size': tgt_vocab_size,
                      'd_model': d_model, 'num_layers': num_layers, 'num_heads': num_heads,
                      'd_ff': d_ff, 'max_len': max_len, 'dropout': 0.1}
        }, "encoder_decoder_interrupted.pt")
        writer.close()

    except Exception as e:
        print(f"\n错误: {type(e).__name__}: {e}")
        torch.save({
            'epoch': epoch if 'epoch' in locals() else 0,
            'model_state_dict': model.state_dict()
        }, "encoder_decoder_error.pt")
        writer.close()
        raise

if __name__ == "__main__":
    train_encoder_decoder(epochs=10)
