import torch
import torch.nn as nn
from .layers import MultiHeadAttention, PositionwiseFeedForward

class DecoderLayer(nn.Module):
    """
    Decoder layer with Pre-LN self-attention, optional cross-attention, and FFN.

    Input/output shape is [B,T,D]. Pre-LN preserves an explicit residual
    identity path, which often improves deep-network optimization. It does not
    guarantee that warmup, gradient clipping, or other controls are unnecessary.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out=None, self_mask=None, cross_mask=None):
        # Pre-LN: Layer Norm -> Self-Attention -> Residual
        # Masked Multi-Head Self-Attention
        normed = self.norm1(x)
        attn_out, self_attn_weights = self.self_attn(normed, normed, normed, self_mask)
        x = x + self.dropout1(attn_out)

        cross_attn_weights = None
        if enc_out is not None:
            # Pre-LN: Layer Norm -> Cross-Attention -> Residual
            # Cross-Attention: Q=decoder, K/V=encoder
            normed = self.norm2(x)
            cross_out, cross_attn_weights = self.cross_attn(normed, enc_out, enc_out, cross_mask)
            x = x + self.dropout2(cross_out)

        # Pre-LN: Layer Norm -> FFN -> Residual
        # Feed Forward
        normed = self.norm3(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout3(ffn_out)

        return x, self_attn_weights, cross_attn_weights

class EncoderLayer(nn.Module):
    """
    Encoder layer with Pre-LN self-attention and a position-wise FFN.

    Input/output shape is [B,T,D]. Padding masks operate on key positions;
    padded query outputs are normally removed from the loss separately.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, self_mask=None):
        # Pre-LN: Layer Norm -> Self-Attention -> Residual
        # Multi-Head Self-Attention
        normed = self.norm1(x)
        attn_out, attn_weights = self.self_attn(normed, normed, normed, self_mask)
        x = x + self.dropout1(attn_out)

        # Pre-LN: Layer Norm -> FFN -> Residual
        # Feed Forward
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout2(ffn_out)

        return x, attn_weights
