# 云平台部署和训练指南

本指南帮助您在云平台（PyTorch 2.5.1.1 + CUDA 12.4.1 + Python 3.11）上部署和训练 Transformer 模型。

## 环境信息

- **PyTorch**: 2.5.1.1
- **CUDA**: 12.4.1
- **Python**: 3.11
- **预装**: PyTorch、torchvision、torchaudio

## 快速开始

### 1. 上传代码到云平台

将整个项目文件夹上传到云平台，或使用 git clone：

```bash
# 如果代码在 GitHub
git clone <your-repo-url>
cd llm
```

### 2. 安装依赖

云平台已预装 PyTorch，只需安装其他依赖：

```bash
pip install -r requirements.txt
```

### 3. 验证环境

运行测试脚本确保所有修复正常工作：

```bash
python test/test_fixes.py
```

期望输出：
```
============================================================
🎉 所有测试通过！代码修复成功！
============================================================
```

检查 CUDA 是否可用：

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')"
```

## 训练模型

### 训练 Decoder-Only 模型（如 GPT）

```bash
cd scripts
python train_decoder.py
```

主要参数（在代码中修改或通过命令行传递）：
- `d_model=256`: 模型维度
- `num_layers=4`: Transformer 层数
- `num_heads=4`: 注意力头数
- `d_ff=1024`: FFN 隐藏层维度
- `batch_size=32`: 批次大小
- `epochs=3`: 训练轮数
- `lr=3e-4`: 学习率
- `device='cuda'`: 自动检测，云平台上会使用 CUDA

### 训练 Encoder-Decoder 模型（如翻译）

```bash
cd scripts
python train_encoder_decoder.py
```

## 异常处理功能

代码已添加完善的异常处理：

### 1. 用户中断（Ctrl+C）
按 Ctrl+C 中断训练时：
- 自动保存当前模型到 `*_interrupted.pt`
- 显示最佳验证损失
- 可以使用 `resume_training.py` 恢复训练

### 2. 训练错误（OOM、CUDA Error 等）
遇到错误时：
- 自动保存当前模型到 `*_error.pt`
- 显示完整错误信息
- 便于调试和恢复

### 3. 检查点文件
训练过程中会生成以下文件：
- `decoder_only_best.pt` / `encoder_decoder_best.pt`: 最佳模型
- `*_interrupted.pt`: 中断时保存
- `*_error.pt`: 出错时保存

## 常见问题

### Q1: CUDA Out of Memory (OOM)

**解决方案**：
1. 减小 `batch_size`（如从 32 改为 16 或 8）
2. 减小 `d_model` 或 `num_layers`
3. 减小 `max_len`（序列长度）
4. 使用梯度累积：

```python
# 在训练循环中
accumulation_steps = 4
for i, (x, y) in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q2: 训练速度慢

**优化建议**：
1. 确认使用 GPU：`device='cuda'`
2. 使用混合精度训练（AMP）：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 训练循环中
with autocast():
    logits, _ = model(x, mask=mask)
    loss = criterion(logits.view(-1, vocab_size), y.view(-1))

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

3. 增大 `batch_size`（在内存允许的情况下）
4. 使用 DataLoader 的 `num_workers`：

```python
train_loader = DataLoader(..., num_workers=4, pin_memory=True)
```

### Q3: 如何监控训练进度

**方法 1**：使用 TensorBoard（推荐）

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')

# 训练循环中
writer.add_scalar('Loss/train', avg_loss, epoch)
writer.add_scalar('Loss/val', avg_val_loss, epoch)
writer.add_scalar('Learning_Rate', current_lr, epoch)
```

启动 TensorBoard：
```bash
tensorboard --logdir=runs
```

**方法 2**：保存训练日志

```bash
python train_decoder.py 2>&1 | tee training.log
```

### Q4: 如何恢复中断的训练

使用提供的恢复脚本：

```bash
python scripts/resume_training.py
```

或手动加载：

```python
from utils.checkpoint_utils import load_checkpoint_for_training

checkpoint_path = "decoder_only_interrupted.pt"
training_info = load_checkpoint_for_training(
    checkpoint_path, model, optimizer, scheduler, device='cuda'
)
start_epoch = training_info['start_epoch']
best_val_loss = training_info['best_val_loss']

# 继续训练
for epoch in range(start_epoch, epochs+1):
    # ... 训练代码
```

## 性能优化建议

### 云平台 GPU 配置建议

| GPU 类型 | batch_size | d_model | num_layers | 备注 |
|---------|-----------|---------|-----------|------|
| V100 16GB | 64 | 512 | 6 | 中等模型 |
| V100 32GB | 128 | 512 | 12 | 大模型 |
| A100 40GB | 256 | 768 | 12 | 超大模型 |
| T4 16GB | 32 | 256 | 4 | 小模型 |

### 推荐训练配置

**小模型（快速实验）**：
```python
d_model=128
num_layers=2
num_heads=4
d_ff=512
batch_size=64
max_len=128
```

**中等模型（标准训练）**：
```python
d_model=256
num_layers=4
num_heads=8
d_ff=1024
batch_size=32
max_len=256
```

**大模型（生产级）**：
```python
d_model=512
num_layers=6
num_heads=8
d_ff=2048
batch_size=16
max_len=512
```

## 重要修复说明

本次修复解决了以下严重问题：

### 1. ✅ 位置编码奇数维度问题
- **问题**：奇数 d_model 时位置编码计算错误
- **修复**：采用 PyTorch 官方标准实现
- **影响**：确保任意 d_model 都能正确工作

### 2. ✅ Embedding 缩放缺失
- **问题**：未按 Transformer 论文缩放 embedding
- **修复**：添加 `sqrt(d_model)` 缩放因子
- **影响**：提升训练稳定性和收敛速度

### 3. ✅ Top-P 采样边界条件
- **问题**：极端 p 值时可能出错
- **修复**：重写采样逻辑，保证至少保留 1 个 token
- **影响**：生成更稳定

### 4. ✅ Beam Search 序列处理
- **问题**：未完成序列处理不清晰
- **修复**：明确区分已完成和未完成序列
- **影响**：beam search 结果更准确

### 5. ✅ 训练异常处理
- **问题**：训练中断会丢失进度
- **修复**：添加 KeyboardInterrupt 和 Exception 处理
- **影响**：训练更稳健，可恢复

## 联系和支持

如遇问题，请检查：
1. 运行 `python test/test_fixes.py` 确认环境正确
2. 查看训练日志中的错误信息
3. 检查 GPU 内存使用：`nvidia-smi`
4. 确认数据文件路径正确

祝训练顺利！🚀
