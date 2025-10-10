import torch

def create_causal_mask(seq_len, device=None):
    """
    创建因果 mask (causal mask) 用于自回归生成

    Returns:
        mask: (1, 1, seq_len, seq_len)，下三角为1，上三角为0
              会在 MultiHeadAttention 中广播到 (batch, num_heads, seq_len, seq_len)
    """
    # 下三角为1，上三角为0
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    # 添加 batch 和 head 维度: (1, 1, seq_len, seq_len)
    return mask.unsqueeze(0).unsqueeze(0)

def create_padding_mask(seq, pad_token_id=0):
    """
    创建 padding mask，用于忽略 padding tokens

    Args:
        seq: (batch, seq_len) 输入序列
        pad_token_id: padding token 的 id

    Returns:
        mask: (batch, 1, 1, seq_len)，非 padding 位置为 True，padding 位置为 False
              会在 attention 中广播到 (batch, num_heads, seq_len, seq_len)
    """
    # (batch, seq_len) -> (batch, 1, 1, seq_len)
    return (seq != pad_token_id).unsqueeze(1).unsqueeze(2)

def combine_masks(causal_mask, padding_mask):
    """
    组合因果 mask 和 padding mask

    Args:
        causal_mask: (1, 1, seq_len, seq_len) 因果 mask，下三角为 True
        padding_mask: (batch, 1, 1, seq_len) padding mask，非 padding 位置为 True

    Returns:
        combined_mask: (batch, 1, seq_len, seq_len) 组合后的 mask
                      只有当位置既满足因果关系，又不是 padding 时才为 True

    Example:
        >>> causal = create_causal_mask(4)  # (1, 1, 4, 4)
        >>> seq = torch.tensor([[1, 2, 3, 0]])  # 最后一个是 padding
        >>> padding = create_padding_mask(seq)  # (1, 1, 1, 4)
        >>> mask = combine_masks(causal, padding)  # (1, 1, 4, 4)
    """
    # padding_mask: (batch, 1, 1, seq_len) -> (batch, 1, seq_len, seq_len)
    # 需要在 key 维度上扩展，表示每个 query 位置对所有 key 位置的 padding 状态
    batch_size = padding_mask.size(0)
    seq_len = padding_mask.size(-1)

    # 扩展 padding_mask 到 (batch, 1, seq_len, seq_len)
    # 每一行都是相同的 padding mask，表示哪些 key 位置是有效的
    padding_mask_expanded = padding_mask.expand(batch_size, 1, seq_len, seq_len)

    # causal_mask: (1, 1, seq_len, seq_len) 会自动广播到 (batch, 1, seq_len, seq_len)
    # 逐元素与操作：只有同时满足因果关系和非 padding 才为 True
    return causal_mask & padding_mask_expanded

def collate_fn_lm(batch, pad_token_id=0):
    # batch: List[List[int]]
    max_len = max(len(x) for x in batch)
    batch_tensor = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(batch):
        batch_tensor[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    # input, target
    return batch_tensor[:, :-1], batch_tensor[:, 1:]

def collate_fn_mt(src_batch, tgt_batch, pad_token_id=0):
    # src_batch, tgt_batch: List[List[int]]
    src_max = max(len(x) for x in src_batch)
    tgt_max = max(len(x) for x in tgt_batch)
    src_tensor = torch.full((len(src_batch), src_max), pad_token_id, dtype=torch.long)
    tgt_tensor = torch.full((len(tgt_batch), tgt_max), pad_token_id, dtype=torch.long)
    for i, x in enumerate(src_batch):
        src_tensor[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    for i, x in enumerate(tgt_batch):
        tgt_tensor[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return src_tensor, tgt_tensor
