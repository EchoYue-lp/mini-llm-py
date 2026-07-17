import torch

def create_causal_mask(seq_len, device=None):
    """
    创建因果 mask (causal mask) 用于自回归生成

    Returns:
        mask: (1, 1, seq_len, seq_len)，下三角为1，上三角为0
              会在 MultiHeadAttention 中广播到 (batch, num_heads, seq_len, seq_len)
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
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
    if seq.ndim != 2:
        raise ValueError("seq must have shape [B,T]")
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
    if causal_mask.dtype != torch.bool or padding_mask.dtype != torch.bool:
        raise TypeError("causal and padding masks must be boolean")
    if causal_mask.device != padding_mask.device:
        raise ValueError("causal and padding masks must be on the same device")
    if causal_mask.ndim != 4 or padding_mask.ndim != 4:
        raise ValueError("masks must have four dimensions")
    if causal_mask.size(-1) != padding_mask.size(-1):
        raise ValueError("causal and padding masks must use the same key length")
    try:
        torch.broadcast_shapes(causal_mask.shape, padding_mask.shape)
    except RuntimeError as error:
        raise ValueError("causal and padding masks are not broadcastable") from error
    return causal_mask & padding_mask

def collate_fn_lm(batch, pad_token_id=0):
    """
    语言模型的 collate function，支持动态 padding

    Args:
        batch: List[List[int]] - 序列列表
        pad_token_id: padding token 的 id

    Returns:
        input_ids: (batch, max_len-1) - 输入序列（去掉最后一个 token）
        target_ids: (batch, max_len-1) - 目标序列（去掉第一个 token）

    注意:
        - 这个函数将不同长度的序列 pad 到 batch 内最大长度
        - 使用时应配合 CrossEntropyLoss(ignore_index=pad_token_id)
        - 训练时应使用 combine_masks(causal_mask, padding_mask)
    """
    if not batch:
        raise ValueError("batch must contain at least one sequence")
    if any(len(sequence) < 2 for sequence in batch):
        raise ValueError("every LM sequence needs at least two tokens")
    # batch: List[List[int]]
    max_len = max(len(x) for x in batch)
    batch_tensor = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(batch):
        batch_tensor[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    # input, target
    return batch_tensor[:, :-1], batch_tensor[:, 1:]

def collate_fn_mt(src_batch, tgt_batch, pad_token_id=0):
    """
    机器翻译的 collate function，支持动态 padding

    Args:
        src_batch: List[List[int]] - 源语言序列列表
        tgt_batch: List[List[int]] - 目标语言序列列表
        pad_token_id: padding token 的 id

    Returns:
        src_tensor: (batch, src_max_len) - padding 后的源序列
        tgt_tensor: (batch, tgt_max_len) - padding 后的目标序列

    注意:
        - 源序列和目标序列分别 pad 到各自 batch 内的最大长度
        - 使用时应配合 CrossEntropyLoss(ignore_index=pad_token_id)
        - 训练时需要创建对应的 padding mask
    """
    if not src_batch or not tgt_batch:
        raise ValueError("source and target batches must be non-empty")
    if len(src_batch) != len(tgt_batch):
        raise ValueError("source and target batch sizes must match")
    if any(not sequence for sequence in (*src_batch, *tgt_batch)):
        raise ValueError("source and target sequences must be non-empty")
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
