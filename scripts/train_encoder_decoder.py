import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.mask_utils import collate_fn_mt, create_padding_mask, create_causal_mask, combine_masks
from utils.scheduler_utils import WarmupLRScheduler
from models.transformer_models import EncoderDecoderModel
from transformers import GPT2TokenizerFast
from tqdm import tqdm

def load_dataset_pt(pt_file):
    return torch.load(pt_file)

def collate_fn(src_batch, tgt_batch):
    src_batch = [torch.tensor(x, dtype=torch.long) for x in src_batch]
    tgt_batch = [torch.tensor(x, dtype=torch.long) for x in tgt_batch]
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    return src_batch, tgt_batch

def train_encoder_decoder(
    data_dir="data/iwslt2017",
    tokenizer_dir="tokenization/gpt2",
    d_model=256,
    num_layers=4,
    num_heads=4,
    d_ff=1024,
    max_len=128,
    batch_size=32,
    epochs=3,
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
    for split in ["train", "validation"]:
        src_data = load_dataset_pt(os.path.join(data_dir, f"{split}.en_ids.pt"))
        tgt_data = load_dataset_pt(os.path.join(data_dir, f"{split}.zh_ids.pt"))
        if split == "train":
            train_src, train_tgt = src_data, tgt_data
        else:
            val_src, val_tgt = src_data, tgt_data
    train_loader = DataLoader(list(zip(train_src, train_tgt)), batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn_mt([x[0] for x in batch], [x[1] for x in batch]))
    val_loader = DataLoader(list(zip(val_src, val_tgt)), batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn_mt([x[0] for x in batch], [x[1] for x in batch]))
    model = EncoderDecoderModel(src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

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

    try:
        for epoch in range(1, epochs+1):
            model.train()
            total_loss = 0
            for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch} - Train"):
                src, tgt = src.to(device), tgt.to(device)
                # tgt: [B, L]，输入为[:-1]，目标为[1:]
                tgt_input = tgt[:, :-1]

                # 创建 mask：需要同时考虑因果关系和 padding
                src_mask = create_padding_mask(src)
                tgt_causal_mask = create_causal_mask(tgt_input.size(1), device=src.device)
                tgt_padding_mask = create_padding_mask(tgt_input)
                tgt_mask = combine_masks(tgt_causal_mask, tgt_padding_mask)
                cross_mask = create_padding_mask(src)

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
            # 验证
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for src, tgt in tqdm(val_loader, desc=f"Epoch {epoch} - Val"):
                    src, tgt = src.to(device), tgt.to(device)
                    tgt_input = tgt[:, :-1]

                    # 创建 mask：需要同时考虑因果关系和 padding
                    src_mask = create_padding_mask(src)
                    tgt_causal_mask = create_causal_mask(tgt_input.size(1), device=src.device)
                    tgt_padding_mask = create_padding_mask(tgt_input)
                    tgt_mask = combine_masks(tgt_causal_mask, tgt_padding_mask)
                    cross_mask = create_padding_mask(src)

                    logits, _ = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)
                    loss = criterion(logits.view(-1, tgt_vocab_size), tgt[:, 1:].reshape(-1))
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")

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
        raise  # 重新抛出异常以便查看完整堆栈跟踪

if __name__ == "__main__":
    train_encoder_decoder(epochs=100)
