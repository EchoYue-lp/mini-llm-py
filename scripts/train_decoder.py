import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.transformer_models import DecoderOnlyModel
from utils.mask_utils import create_causal_mask, create_padding_mask, combine_masks
from utils.scheduler_utils import WarmupLRScheduler
from transformers import GPT2TokenizerFast
from tqdm import tqdm

def load_dataset_pt(pt_file):
    return torch.load(pt_file)

def collate_fn(batch):
    # batch: List[List[int]]
    batch = [torch.tensor(x, dtype=torch.long) for x in batch]
    batch = torch.stack(batch)
    return batch[:, :-1], batch[:, 1:]  # input, target

def train_decoder_only(
    data_dir="data/wikitext2",
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
    vocab_size = tokenizer.vocab_size
    train_data = load_dataset_pt(os.path.join(data_dir, "train_ids.pt"))
    val_data = load_dataset_pt(os.path.join(data_dir, "validation_ids.pt"))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model = DecoderOnlyModel(vocab_size, d_model, num_layers, num_heads, d_ff, max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # 添加 ignore_index 以忽略 padding token 的损失
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
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
        raise  # 重新抛出异常以便查看完整堆栈跟踪

if __name__ == "__main__":
    train_decoder_only()
