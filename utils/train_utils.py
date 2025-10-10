import torch
import math
from .mask_utils import create_causal_mask, create_padding_mask

def compute_perplexity(loss):
    return math.exp(loss)

def evaluate_lm(model, data_loader, criterion, vocab_size, device, pad_token_id=0):
    """
    评估语言模型

    Args:
        model: 解码器模型
        data_loader: 数据加载器
        criterion: 损失函数（应该已经设置了 ignore_index）
        vocab_size: 词表大小
        device: 设备
        pad_token_id: padding token id（用于计算有效 token 数量）
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            # 为 decoder-only 模型创建 causal mask
            mask = create_causal_mask(x.size(1), device=device)
            logits, _ = model(x, mask=mask)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            # 只计算非 padding token 的数量
            num_tokens = (y != pad_token_id).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        ppl = compute_perplexity(avg_loss)
    else:
        avg_loss = float('inf')
        ppl = float('inf')

    return avg_loss, ppl

def evaluate_mt(model, data_loader, criterion, tgt_vocab_size, device, pad_token_id=0):
    """
    评估机器翻译模型

    Args:
        model: encoder-decoder 模型
        data_loader: 数据加载器
        criterion: 损失函数（应该已经设置了 ignore_index）
        tgt_vocab_size: 目标词表大小
        device: 设备
        pad_token_id: padding token id（用于计算有效 token 数量）
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            # 为 encoder-decoder 模型创建必要的 masks
            src_mask = create_padding_mask(src)
            tgt_mask = create_causal_mask(tgt[:, :-1].size(1), device=device)
            cross_mask = create_padding_mask(src)
            logits, _ = model(src, tgt[:, :-1], src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)
            loss = criterion(logits.view(-1, tgt_vocab_size), tgt[:, 1:].reshape(-1))
            # 只计算非 padding token 的数量
            num_tokens = (tgt[:, 1:] != pad_token_id).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        ppl = compute_perplexity(avg_loss)
    else:
        avg_loss = float('inf')
        ppl = float('inf')

    return avg_loss, ppl
