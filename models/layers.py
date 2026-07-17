import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        if d_k <= 0:
            raise ValueError("d_k must be positive")
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, head, seq_len, d_k)
        if min(Q.ndim, K.ndim, V.ndim) < 3:
            raise ValueError("Q, K, and V must have shape [...,T,D]")
        if Q.size(-1) != self.d_k or K.size(-1) != self.d_k:
            raise ValueError("Q/K feature width must match configured d_k")
        if V.size(-1) != self.d_k:
            raise ValueError("this teaching implementation requires V width == d_k")
        if K.size(-2) != V.size(-2):
            raise ValueError("K and V sequence lengths must match")
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dtype != torch.bool:
                raise TypeError("mask must be boolean with True meaning visible")
            if mask.device != scores.device:
                raise ValueError("mask and scores must be on the same device")
            try:
                torch.broadcast_shapes(scores.shape, mask.shape)
            except RuntimeError as error:
                raise ValueError("mask is not broadcastable to attention scores") from error
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        # A fully masked row has no probability distribution; define it as zeros.
        attn = torch.nan_to_num(attn, nan=0.0)
        # 在 attention weights 上应用 dropout（训练时随机丢弃部分注意力连接）
        attn_dropped = self.dropout(attn)
        output = torch.matmul(attn_dropped, V)
        # 返回原始 attn（用于可视化），output 使用 dropout 后的结果
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(self.d_k, dropout)
        # 移除额外的 dropout，避免重复应用
        # attention weights 上已经应用了 dropout

    def forward(self, Q, K, V, mask=None):
        if any(tensor.ndim != 3 for tensor in (Q, K, V)):
            raise ValueError("Q, K, and V must have shape [B,T,D]")
        if any(tensor.size(-1) != self.d_model for tensor in (Q, K, V)):
            raise ValueError("Q, K, and V feature width must equal d_model")
        if Q.size(0) != K.size(0) or K.size(0) != V.size(0):
            raise ValueError("Q, K, and V batch sizes must match")
        if K.size(1) != V.size(1):
            raise ValueError("K and V sequence lengths must match")
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        # 线性变换并分头
        def transform(x):
            x_proj = x.reshape(x.size(0), x.size(1), self.num_heads, self.d_k)
            return x_proj.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, d_k)
        Q = transform(self.q_linear(Q))
        K = transform(self.k_linear(K))
        V = transform(self.v_linear(V))
        # mask 已经是 (1, 1, seq, seq) 或 (batch, 1, 1, seq) 格式，无需再 unsqueeze
        # 会自动广播到 (batch, num_heads, seq, seq)
        out, attn = self.attn(Q, K, V, mask)
        # 合并头: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.num_heads * self.d_k)
        # 只对最终输出应用线性变换，dropout 已在 attention weights 上应用
        return self.out_linear(out), attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive")
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.act(self.linear1(x))))

class PositionalEncoding(nn.Module):
    """
    标准位置编码实现
    参考: PyTorch 官方教程 https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        if d_model <= 0 or max_len <= 0:
            raise ValueError("d_model and max_len must be positive")
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # 计算 div_term = 1 / 10000^(2i/d_model)
        # 使用 exp 和 log 来避免数值溢出
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 应用 sin 到偶数索引 (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # 应用 cos 到奇数索引 (1, 3, 5, ...)
        # 这里自动处理了奇数 d_model 的情况：
        # - 如果 d_model 是偶数，pe[:, 1::2] 和 pe[:, 0::2] 形状相同
        # - 如果 d_model 是奇数，pe[:, 1::2] 会比 pe[:, 0::2] 少一列
        # 而 div_term 已经是正确的长度，所以不需要特殊处理
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])

        # 添加 batch 维度: (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"序列长度 {seq_len} 超过位置编码的最大长度 {self.pe.size(1)}")

        # 添加位置编码并应用 dropout
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class LayerNorm(nn.Module):
    """
    自定义 LayerNorm 实现（已弃用）

    注意：此实现仅用于教学目的。在实际应用中，请使用 PyTorch 官方的 nn.LayerNorm，
    因为它经过了高度优化，数值稳定性更好，并且支持混合精度训练。

    推荐用法：
        self.norm = nn.LayerNorm(d_model)  # 而不是 LayerNorm(d_model)
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        if d_model <= 0 or eps <= 0:
            raise ValueError("d_model and eps must be positive")
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return self.gamma * normalized + self.beta
