import torch
import torch.nn as nn
import math
from .layers import PositionalEncoding
from .decoder_encoder_layer import DecoderLayer, EncoderLayer

class DecoderOnlyModel(nn.Module):
    """
    Decoder-only Transformer with Pre-LN blocks and final normalization.

    Token ids use shape [B,T]; logits use [B,T,V]. Pre-LN leaves the residual
    stream after the final block unnormalized, so a final LayerNorm is applied
    before the vocabulary projection. It improves the gradient path but does
    not remove the need for an appropriate optimizer schedule.
    """
    def __init__(self, vocab_size, d_model=256, num_layers=4, num_heads=4, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        if min(vocab_size, d_model, num_layers, num_heads, d_ff, max_len) <= 0:
            raise ValueError("model dimensions, layer counts, and vocab size must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        # Pre-LN 架构需要最后的 LayerNorm，因为最后一层的输出没有经过归一化
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

        # Embedding 缩放因子（参考 Transformer 原论文）
        self.embed_scale = math.sqrt(d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        if x.ndim != 2:
            raise ValueError("decoder input ids must have shape [B,T]")
        if x.dtype != torch.long:
            raise TypeError("decoder input ids must use torch.long")
        if x.size(1) > self.max_len:
            raise ValueError("decoder sequence length exceeds max_len")
        # 应用 embedding 缩放以平衡位置编码的影响
        x = self.embed(x) * self.embed_scale
        x = self.pos_enc(x)
        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights, _ = layer(x, self_mask=mask)
            attn_weights_all.append(attn_weights)
        # Pre-LN: 最后需要一个归一化
        x = self.norm(x)
        logits = self.out_proj(x)
        return logits, attn_weights_all

class EncoderDecoderModel(nn.Module):
    """
    Encoder-decoder Transformer with Pre-LN blocks and final norms.

    Source ids are [B,Ts], target ids are [B,Tt], encoder memory is [B,Ts,D],
    and output logits are [B,Tt,Vt]. Source and target lengths are independent;
    cross-attention scores therefore use [B,H,Tt,Ts].
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_layers=4, num_heads=4, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        if min(
            src_vocab_size,
            tgt_vocab_size,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            max_len,
        ) <= 0:
            raise ValueError("model dimensions, layer counts, and vocab sizes must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.src_pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.tgt_pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        # Pre-LN: Encoder 输出需要归一化
        self.encoder_norm = nn.LayerNorm(d_model)
        # Pre-LN: Decoder 输出需要归一化
        self.decoder_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

        # Embedding 缩放因子（参考 Transformer 原论文）
        self.embed_scale = math.sqrt(d_model)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        # src: (batch, src_seq_len), tgt: (batch, tgt_seq_len)
        if src.ndim != 2 or tgt.ndim != 2:
            raise ValueError("source and target ids must have shape [B,T]")
        if src.dtype != torch.long or tgt.dtype != torch.long:
            raise TypeError("source and target ids must use torch.long")
        if src.size(0) != tgt.size(0):
            raise ValueError("source and target batch sizes must match")
        if src.size(1) > self.max_len or tgt.size(1) > self.max_len:
            raise ValueError("source or target sequence length exceeds max_len")
        # Encoder - 应用 embedding 缩放
        src_emb = self.src_embed(src) * self.embed_scale
        src_emb = self.src_pos_enc(src_emb)
        for layer in self.encoder_layers:
            src_emb, _ = layer(src_emb, self_mask=src_mask)
        # Pre-LN: Encoder 输出归一化
        memory = self.encoder_norm(src_emb)

        # Decoder - 应用 embedding 缩放
        tgt_emb = self.tgt_embed(tgt) * self.embed_scale
        tgt_emb = self.tgt_pos_enc(tgt_emb)
        attn_weights_all = []
        for layer in self.decoder_layers:
            tgt_emb, self_attn, cross_attn = layer(tgt_emb, enc_out=memory, self_mask=tgt_mask, cross_mask=cross_mask)
            attn_weights_all.append((self_attn, cross_attn))
        # Pre-LN: Decoder 输出归一化
        x = self.decoder_norm(tgt_emb)
        logits = self.out_proj(x)
        return logits, attn_weights_all
