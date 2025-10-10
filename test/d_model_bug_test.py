# 测试奇数 d_model
from models.layers import PositionalEncoding
import torch
# 测试奇数维度
pos_enc_odd = PositionalEncoding(d_model=257, max_len=100, dropout=0.1)
x = torch.randn(2, 10, 257)
output = pos_enc_odd(x)
print(f"奇数维度测试通过: {output.shape}")  # 应该输出 torch.Size([2, 10, 257])
# 测试偶数维度
pos_enc_even = PositionalEncoding(d_model=256, max_len=100, dropout=0.1)
x = torch.randn(2, 10, 256)
output = pos_enc_even(x)
print(f"偶数维度测试通过: {output.shape}")  # 应该输出 torch.Size([2, 10, 256])