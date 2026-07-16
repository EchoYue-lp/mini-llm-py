import torch
import torch.nn.functional as F
from .mask_utils import create_causal_mask, create_padding_mask, combine_masks


def top_p_candidates(probs, p):
    """Return normalized nucleus candidates, including the crossing token."""

    if not 0 < p <= 1:
        raise ValueError("p must be in (0, 1]")

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative_probs > p
    remove[1:] = remove[:-1].clone()
    remove[0] = False

    kept_probs = sorted_probs[~remove]
    kept_indices = sorted_indices[~remove]
    kept_probs = kept_probs / kept_probs.sum()
    return kept_probs, kept_indices

def beam_search_generate(model, input_ids, tokenizer, beam_width=3, max_len=50, device="cpu", length_penalty=0.6):
    """
    Beam Search 生成

    Args:
        model: 解码器模型
        input_ids: 输入 token ids
        tokenizer: tokenizer
        beam_width: beam 宽度
        max_len: 最大生成长度
        device: 设备
        length_penalty: 长度惩罚 (alpha)，score = log_prob / (len ** alpha)
    """
    model.eval()
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    sequences = [(input_ids, 0.0)]
    completed = []  # 存储已完成的序列

    for step in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            # 如果已经生成 EOS，放入 completed
            if seq[0, -1].item() == tokenizer.eos_token_id:
                completed.append((seq, score))
                continue

            # 创建 mask：同时考虑因果关系和 padding
            causal_mask = create_causal_mask(seq.size(1), device=device)
            padding_mask = create_padding_mask(seq, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
            mask = combine_masks(causal_mask, padding_mask)
            with torch.no_grad():
                logits, _ = model(seq, mask=mask)
            logits = logits[:, -1, :]
            probs = F.log_softmax(logits, dim=-1)
            topk_probs, topk_ids = probs.topk(beam_width)
            for k in range(beam_width):
                next_token = topk_ids[0, k].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, next_token], dim=1)
                new_score = score + topk_probs[0, k].item()
                all_candidates.append((new_seq, new_score))

        if not all_candidates:
            break

        # 使用长度惩罚选择 top-k
        sequences = sorted(all_candidates,
                         key=lambda x: x[1] / (x[0].size(1) ** length_penalty),
                         reverse=True)[:beam_width]

        # 如果所有 beam 都完成了，提前退出
        if all(seq[0, -1].item() == tokenizer.eos_token_id for seq, _ in sequences):
            completed.extend(sequences)
            break

    # 循环结束后，将未完成的序列也加入 completed
    for seq, score in sequences:
        if seq[0, -1].item() != tokenizer.eos_token_id:
            completed.append((seq, score))

    # 所有候选序列（已完成 + 未完成但达到 max_len）
    all_sequences = completed
    if not all_sequences:
        return ""

    # 选择得分最高的（应用长度惩罚）
    best_seq, best_score = max(all_sequences,
                               key=lambda x: x[1] / (x[0].size(1) ** length_penalty))
    best_seq = best_seq.squeeze(0).tolist()

    # 截断到 EOS（不包含 EOS）
    if tokenizer.eos_token_id in best_seq:
        eos_idx = best_seq.index(tokenizer.eos_token_id)
        best_seq = best_seq[:eos_idx]

    return tokenizer.decode(best_seq, skip_special_tokens=True)

def top_k_sampling(model, input_ids, tokenizer, k=10, max_len=50, device="cpu", temperature=1.0):
    """
    Top-K 采样生成

    Args:
        model: 解码器模型
        input_ids: 输入 token ids
        tokenizer: tokenizer
        k: top-k 参数
        max_len: 最大生成长度
        device: 设备
        temperature: 温度参数，越大越随机
    """
    model.eval()
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for _ in range(max_len):
        # 创建 mask：同时考虑因果关系和 padding
        causal_mask = create_causal_mask(input_ids.size(1), device=device)
        padding_mask = create_padding_mask(input_ids, pad_token_id=pad_token_id)
        mask = combine_masks(causal_mask, padding_mask)
        with torch.no_grad():
            logits, _ = model(input_ids, mask=mask)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_ids = probs.topk(k)

        # 使用 torch.multinomial 代替 random.choices
        next_token_idx = torch.multinomial(topk_probs[0], num_samples=1)
        next_token = topk_ids[0, next_token_idx].item()
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

        if next_token == tokenizer.eos_token_id:
            break

    output_ids = input_ids.squeeze(0).tolist()
    if tokenizer.eos_token_id in output_ids:
        eos_idx = output_ids.index(tokenizer.eos_token_id)
        output_ids = output_ids[:eos_idx]
    return tokenizer.decode(output_ids, skip_special_tokens=True)

def top_p_sampling(model, input_ids, tokenizer, p=0.9, max_len=50, device="cpu", temperature=1.0):
    """
    Top-P (Nucleus) 采样生成

    Args:
        model: 解码器模型
        input_ids: 输入 token ids
        tokenizer: tokenizer
        p: nucleus 概率阈值
        max_len: 最大生成长度
        device: 设备
        temperature: 温度参数，越大越随机
    """
    model.eval()
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for _ in range(max_len):
        # 创建 mask：同时考虑因果关系和 padding
        causal_mask = create_causal_mask(input_ids.size(1), device=device)
        padding_mask = create_padding_mask(input_ids, pad_token_id=pad_token_id)
        mask = combine_masks(causal_mask, padding_mask)
        with torch.no_grad():
            logits, _ = model(input_ids, mask=mask)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1).squeeze(0)
        sorted_probs, sorted_indices = top_p_candidates(probs, p)

        # 从 nucleus 集合中采样
        next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices[next_token_idx].item()
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

        if next_token == tokenizer.eos_token_id:
            break

    output_ids = input_ids.squeeze(0).tolist()
    if tokenizer.eos_token_id in output_ids:
        eos_idx = output_ids.index(tokenizer.eos_token_id)
        output_ids = output_ids[:eos_idx]
    return tokenizer.decode(output_ids, skip_special_tokens=True)
