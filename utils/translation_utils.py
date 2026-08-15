"""
翻译生成工具
专门为 Encoder-Decoder 模型设计的生成策略
"""

from functools import wraps

import torch
import torch.nn.functional as F
from .mask_utils import create_padding_mask, create_causal_mask


def _inference_translation(function):
    @wraps(function)
    def wrapper(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                return function(model, *args, **kwargs)
        finally:
            model.train(was_training)

    return wrapper


def _validate_translation_request(model, src_ids, max_new_tokens):
    if src_ids is None or len(src_ids) == 0:
        raise ValueError("src_ids must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_len (generated token count) must be non-negative")
    if len(src_ids) > model.max_len:
        raise ValueError("source sequence exceeds model max_len")
    if 1 + max_new_tokens > model.max_len:
        raise ValueError("BOS plus generated tokens exceeds model max_len")


@_inference_translation
def beam_search_translate(model, src_ids, tokenizer, beam_width=5, max_len=50, device="cpu", length_penalty=0.6):
    """
    使用 Beam Search 进行翻译

    Args:
        model: EncoderDecoderModel 实例
        src_ids: 源语言 token ids (list)
        tokenizer: tokenizer
        beam_width: beam 宽度
        max_len: 新生成 token 数量（不包含初始 BOS）
        device: 设备
        length_penalty: 长度惩罚系数 (alpha)，越大越鼓励长句子
                       final_score = log_prob / (len ** length_penalty)

    Returns:
        翻译后的文本 (str)
    """
    _validate_translation_request(model, src_ids, max_len)
    if not 0 < beam_width <= model.tgt_vocab_size:
        raise ValueError("beam_width must be in [1, target vocab size]")
    if length_penalty < 0:
        raise ValueError("length_penalty must be non-negative")

    # 准备源序列
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, src_len)

    # 编码源序列 (只需要编码一次)
    with torch.no_grad():
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        src_mask = create_padding_mask(src, pad_token_id=pad_token_id)

        # 手动运行 encoder
        # 重要：应用 embedding scaling 以匹配训练时的行为
        src_emb = model.src_embed(src) * model.embed_scale
        src_emb = model.src_pos_enc(src_emb)
        memory = src_emb
        for layer in model.encoder_layers:
            memory, _ = layer(memory, self_mask=src_mask)
        # Pre-LN: Encoder 输出需要归一化
        memory = model.encoder_norm(memory)

    # 初始化 beams: [(序列, 累积log概率)]
    # GPT2 tokenizer 没有 BOS，使用 EOS 作为 BOS
    # 注意：即使 bos_id == eos_token_id（如使用 GPT2），下面的 len(seq) > 1 条件
    # 也能确保初始序列 [bos_id] 不会被立即判定为完成
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    beams = [(torch.tensor([bos_id], dtype=torch.long, device=device), 0.0)]

    completed = []  # 已完成的序列

    for step in range(max_len):
        candidates = []

        for seq, score in beams:
            # 如果序列已经生成了 EOS，放入 completed
            # 条件说明：
            # - len(seq) > 1: 确保至少生成了一个新 token（排除初始的 [bos_id]）
            #   即使 bos_id == eos_token_id，初始序列 len=1 也不满足此条件
            # - seq[-1] == eos_token_id: 最后一个 token 是 EOS
            if len(seq) > 1 and seq[-1].item() == tokenizer.eos_token_id:
                completed.append((seq, score))
                continue

            # 解码当前序列
            tgt = seq.unsqueeze(0)  # (1, cur_len)
            # 注意：生成时 tgt 序列逐步增长，没有 padding，所以只需要因果 mask
            tgt_mask = create_causal_mask(tgt.size(1), device=device)
            # cross_mask 处理源序列的 padding
            cross_mask = create_padding_mask(src, pad_token_id=pad_token_id)

            with torch.no_grad():
                # 手动运行 decoder
                # 重要：应用 embedding scaling 以匹配训练时的行为
                tgt_emb = model.tgt_embed(tgt) * model.embed_scale
                tgt_emb = model.tgt_pos_enc(tgt_emb)

                for layer in model.decoder_layers:
                    tgt_emb, _, _ = layer(tgt_emb, enc_out=memory, self_mask=tgt_mask, cross_mask=cross_mask)

                # Pre-LN: Decoder 输出需要归一化
                tgt_emb = model.decoder_norm(tgt_emb)
                logits = model.out_proj(tgt_emb)  # (1, cur_len, vocab_size)

            # 获取最后一个位置的 logits
            next_token_logits = logits[0, -1, :]  # (vocab_size,)
            log_probs = F.log_softmax(next_token_logits, dim=-1)

            # 获取 top-k 候选
            topk_log_probs, topk_ids = log_probs.topk(beam_width)

            for i in range(beam_width):
                next_token = topk_ids[i].unsqueeze(0)
                next_log_prob = topk_log_probs[i].item()

                new_seq = torch.cat([seq, next_token])
                new_score = score + next_log_prob

                candidates.append((new_seq, new_score))

        # 如果没有候选了，退出
        if not candidates:
            break

        # 选择 top beam_width 个候选（使用长度惩罚）
        def get_score(item):
            seq, score = item
            # 长度惩罚: score / (length ** alpha)
            return score / (len(seq) ** length_penalty)

        beams = sorted(candidates, key=get_score, reverse=True)[:beam_width]

        # 如果所有 beam 都完成了，提前退出
        if all(seq[-1].item() == tokenizer.eos_token_id for seq, _ in beams):
            completed.extend(beams)
            break

    # 合并未完成和已完成的序列
    all_sequences = completed + beams

    if not all_sequences:
        return ""

    # 选择得分最高的序列（使用长度惩罚）
    best_seq, best_score = max(all_sequences,
                               key=lambda x: x[1] / (len(x[0]) ** length_penalty))

    # 转换为 list 并去除 BOS/EOS
    output_ids = best_seq.tolist()
    # 去除 BOS（可能是 bos_token_id 或 eos_token_id）
    if output_ids and output_ids[0] == bos_id:
        output_ids = output_ids[1:]
    # 截断到 EOS
    if tokenizer.eos_token_id in output_ids:
        output_ids = output_ids[:output_ids.index(tokenizer.eos_token_id)]

    return tokenizer.decode(output_ids, skip_special_tokens=True)


@_inference_translation
def greedy_translate(model, src_ids, tokenizer, max_len=50, device="cpu"):
    """
    使用 Greedy Decoding 进行翻译

    Args:
        model: EncoderDecoderModel 实例
        src_ids: 源语言 token ids (list)
        tokenizer: tokenizer
        max_len: 新生成 token 数量（不包含初始 BOS）
        device: 设备

    Returns:
        翻译后的文本 (str)
    """
    _validate_translation_request(model, src_ids, max_len)

    # 获取起始 token
    # GPT2 tokenizer 没有 BOS，使用 EOS 作为 BOS（这是常见做法）
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, src_len)
    tgt = torch.tensor([bos_id], dtype=torch.long).unsqueeze(0).to(device)  # (1, 1)

    with torch.no_grad():
        for _ in range(max_len):
            src_mask = create_padding_mask(src, pad_token_id=pad_token_id)
            # 注意：生成时 tgt 序列逐步增长，没有 padding，所以只需要因果 mask
            tgt_mask = create_causal_mask(tgt.size(1), device=device)
            # cross_mask 处理源序列的 padding
            cross_mask = create_padding_mask(src, pad_token_id=pad_token_id)

            logits, _ = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    output_ids = tgt.squeeze(0).tolist()

    # 去除 BOS
    if output_ids and output_ids[0] == bos_id:
        output_ids = output_ids[1:]
    # 截断到 EOS
    if tokenizer.eos_token_id in output_ids:
        output_ids = output_ids[:output_ids.index(tokenizer.eos_token_id)]

    return tokenizer.decode(output_ids, skip_special_tokens=True)


@_inference_translation
def top_k_translate(model, src_ids, tokenizer, k=10, max_len=50, device="cpu", temperature=1.0):
    """
    使用 Top-K 采样进行翻译

    Args:
        model: EncoderDecoderModel 实例
        src_ids: 源语言 token ids (list)
        tokenizer: tokenizer
        k: top-k 参数
        max_len: 新生成 token 数量（不包含初始 BOS）
        device: 设备
        temperature: 温度参数，越大越随机

    Returns:
        翻译后的文本 (str)
    """
    _validate_translation_request(model, src_ids, max_len)
    if not 0 < k <= model.tgt_vocab_size:
        raise ValueError("k must be in [1, target vocab size]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    # 获取起始 token
    # GPT2 tokenizer 没有 BOS，使用 EOS 作为 BOS（这是常见做法）
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    tgt = torch.tensor([bos_id], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            src_mask = create_padding_mask(src, pad_token_id=pad_token_id)
            # 注意：生成时 tgt 序列逐步增长，没有 padding，所以只需要因果 mask
            tgt_mask = create_causal_mask(tgt.size(1), device=device)
            # cross_mask 处理源序列的 padding
            cross_mask = create_padding_mask(src, pad_token_id=pad_token_id)

            logits, _ = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)
            next_token_logits = logits[0, -1, :] / temperature

            # Top-k 过滤
            topk_probs, topk_ids = F.softmax(next_token_logits, dim=-1).topk(k)

            # 从 top-k 中采样
            next_token_idx = torch.multinomial(topk_probs, num_samples=1)
            next_token = topk_ids[next_token_idx].unsqueeze(0)

            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    output_ids = tgt.squeeze(0).tolist()

    # 去除 BOS
    if output_ids and output_ids[0] == bos_id:
        output_ids = output_ids[1:]
    # 截断到 EOS
    if tokenizer.eos_token_id in output_ids:
        output_ids = output_ids[:output_ids.index(tokenizer.eos_token_id)]

    return tokenizer.decode(output_ids, skip_special_tokens=True)


@_inference_translation
def top_p_translate(model, src_ids, tokenizer, p=0.9, max_len=50, device="cpu", temperature=1.0):
    """
    使用 Top-P (Nucleus) 采样进行翻译

    Args:
        model: EncoderDecoderModel 实例
        src_ids: 源语言 token ids (list)
        tokenizer: tokenizer
        p: nucleus 概率阈值
        max_len: 新生成 token 数量（不包含初始 BOS）
        device: 设备
        temperature: 温度参数

    Returns:
        翻译后的文本 (str)
    """
    _validate_translation_request(model, src_ids, max_len)
    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    # 获取起始 token
    # GPT2 tokenizer 没有 BOS，使用 EOS 作为 BOS（这是常见做法）
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    tgt = torch.tensor([bos_id], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            src_mask = create_padding_mask(src, pad_token_id=pad_token_id)
            # 注意：生成时 tgt 序列逐步增长，没有 padding，所以只需要因果 mask
            tgt_mask = create_causal_mask(tgt.size(1), device=device)
            # cross_mask 处理源序列的 padding
            cross_mask = create_padding_mask(src, pad_token_id=pad_token_id)

            logits, _ = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)
            next_token_logits = logits[0, -1, :] / temperature

            # Top-p 过滤
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            # 找到累积概率超过 p 的位置
            sorted_indices_to_remove = cumulative_probs > p
            # 保留第一个超过阈值的 token
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            # 获取保留的 indices
            indices_to_keep = sorted_indices[~sorted_indices_to_remove]
            probs_to_keep = probs[~sorted_indices_to_remove]

            # 从中采样
            next_token_idx = torch.multinomial(probs_to_keep, num_samples=1)
            next_token = indices_to_keep[next_token_idx].unsqueeze(0)

            tgt = torch.cat([tgt, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    output_ids = tgt.squeeze(0).tolist()

    # 去除 BOS
    if output_ids and output_ids[0] == bos_id:
        output_ids = output_ids[1:]
    # 截断到 EOS
    if tokenizer.eos_token_id in output_ids:
        output_ids = output_ids[:output_ids.index(tokenizer.eos_token_id)]

    return tokenizer.decode(output_ids, skip_special_tokens=True)
