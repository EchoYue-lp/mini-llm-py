import torch
import torch.nn as nn
import math
from .layers import PositionalEncoding
from .decoder_encoder_layer import DecoderLayer, EncoderLayer

class DecoderOnlyModel(nn.Module):
    """
    Decoder-Only Transformer with Pre-LN architecture (like GPT)

    Pre-LN 架构说明:
    - 每个子层前都进行 Layer Normalization
    - 最后需要一个 final LayerNorm，因为最后一层输出未归一化
    - 训练更稳定，不需要复杂的学习率调度
    """
    def __init__(self, vocab_size, d_model=256, num_layers=4, num_heads=4, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
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
    Encoder-Decoder Transformer with Pre-LN architecture

    Pre-LN 架构说明:
    - Encoder 和 Decoder 都使用 Pre-LN
    - Encoder 和 Decoder 最后都需要 LayerNorm
    - 训练更稳定，收敛更快
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_layers=4, num_heads=4, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
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
