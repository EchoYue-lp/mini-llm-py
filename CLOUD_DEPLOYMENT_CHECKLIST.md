# 云平台部署检查清单

**目标环境**: PyTorch 2.5.1.1 + CUDA 12.4.1 + Python 3.11

## 🚀 云平台部署步骤

### 第1步：上传代码
```bash
# 方式1：直接上传文件夹
# 将整个llm文件夹上传到云平台

# 方式2：使用git
git clone <your-repo-url>
cd llm
```

### 第2步：安装依赖
```bash
# 云平台已预装PyTorch 2.5.1.1，只需安装其他依赖
pip install -r requirements.txt
```

### 第3步：验证环境
```bash
# 运行测试确保所有修复正常
python test/test_fixes.py

# 检查CUDA是否可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}')"
```

期望输出：
```
CUDA: True
CUDA Version: 12.4
GPU Count: 1 (或更多)
```

### 第4步：准备数据
```bash
cd scripts

# 下载数据集
python download_datasets.py

# 预处理数据
python preprocess.py
```

### 第5步：开始训练

#### Decoder-Only模型（如GPT）
```bash
python train_decoder.py
```

#### Encoder-Decoder模型（如翻译）
```bash
python train_encoder_decoder.py
```

## ⚙️ 云平台优化建议

### 1. 根据GPU内存调整批次大小

| GPU类型 | 显存 | 推荐batch_size | 推荐d_model | 推荐num_layers |
|---------|------|---------------|-------------|---------------|
| T4      | 16GB | 32            | 256         | 4             |
| V100    | 16GB | 64            | 512         | 6             |
| V100    | 32GB | 128           | 512         | 12            |
| A100    | 40GB | 256           | 768         | 12            |

### 2. 修改训练参数
编辑训练脚本的参数（在文件末尾的`if __name__ == "__main__":`部分）：

**小模型（快速实验）**：
```python
train_decoder_only(
    d_model=128,
    num_layers=2,
    num_heads=4,
    d_ff=512,
    batch_size=64,
    max_len=128,
    epochs=3
)
```

**标准模型**：
```python
train_decoder_only(
    d_model=256,
    num_layers=4,
    num_heads=8,
    d_ff=1024,
    batch_size=32,
    max_len=256,
    epochs=10
)
```

**大模型**：
```python
train_decoder_only(
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    batch_size=16,
    max_len=512,
    epochs=20
)
```

### 3. 启用混合精度训练（可选，加速训练）
如果遇到显存不足或想加速训练，可以启用AMP（Automatic Mixed Precision）。

在训练脚本中添加：
```python
from torch.cuda.amp import autocast, GradScaler

# 在训练函数开始处添加
scaler = GradScaler()

# 修改训练循环
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    causal_mask = create_causal_mask(x.size(1), device=device)
    padding_mask = create_padding_mask(x, pad_token_id=pad_token_id)
    mask = combine_masks(causal_mask, padding_mask)

    optimizer.zero_grad()

    # 使用混合精度
    with autocast():
        logits, _ = model(x, mask=mask)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

    scaler.scale(loss).backward()

    if max_grad_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None:
        scheduler.step()
```

### 4. 监控训练（推荐）
```bash
# 方式1：实时查看输出并保存日志
python train_decoder.py 2>&1 | tee training.log

# 方式2：后台运行
nohup python train_decoder.py > training.log 2>&1 &

# 查看日志
tail -f training.log
```

### 5. 使用TensorBoard可视化（可选）
在训练脚本中添加：
```python
from torch.utils.tensorboard import SummaryWriter

# 在训练函数开始处
writer = SummaryWriter('runs/experiment_1')

# 在训练循环中记录
writer.add_scalar('Loss/train', avg_loss, epoch)
writer.add_scalar('Loss/val', avg_val_loss, epoch)
writer.add_scalar('Learning_Rate', current_lr, epoch)

# 训练结束时关闭
writer.close()
```

启动TensorBoard：
```bash
tensorboard --logdir=runs --port=6006
```

## 🛠️ 异常处理说明

### 1. 训练中断（Ctrl+C）
按Ctrl+C会触发KeyboardInterrupt：
- ✅ 自动保存模型到`decoder_only_interrupted.pt`或`encoder_decoder_interrupted.pt`
- ✅ 显示最佳验证损失
- ✅ 可使用`scripts/resume_training.py`恢复

### 2. 训练错误（OOM、CUDA Error等）
遇到异常会触发Exception处理：
- ✅ 自动保存模型到`decoder_only_error.pt`或`encoder_decoder_error.pt`
- ✅ 显示完整错误堆栈
- ✅ 可分析问题后继续训练

### 3. 恢复训练
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
```

## ⚠️ 常见问题

### Q1: CUDA Out of Memory
**解决方案**：
1. 减小`batch_size`（如32→16→8）
2. 减小`d_model`或`num_layers`
3. 减小`max_len`
4. 启用混合精度训练（见上文）
5. 使用梯度累积（在代码中实现）

### Q2: 训练速度慢
**检查**：
1. 确认使用GPU：检查输出中的设备信息
2. 增大batch_size（在显存允许的情况下）
3. 启用混合精度训练
4. 使用DataLoader的`num_workers`和`pin_memory`

### Q3: 验证损失不下降
**可能原因**：
1. 学习率过大或过小 - 尝试3e-4、1e-4、5e-5
2. 数据问题 - 检查数据预处理
3. 模型过拟合 - 增大dropout或减小模型
4. 需要更多训练轮次

### Q4: 如何检查GPU使用情况
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或一次性查看
nvidia-smi
```

## 📋 部署前检查清单

在云平台运行前，请确认：

- [ ] Python版本：3.11（云平台默认）
- [ ] PyTorch版本：2.5.1.1（云平台预装）
- [ ] CUDA版本：12.4.1（云平台预装）
- [ ] 依赖已安装：`pip install -r requirements.txt`
- [ ] 测试通过：`python test/test_fixes.py`
- [ ] CUDA可用：`torch.cuda.is_available() == True`
- [ ] 数据已准备：`data/wikitext2/`目录存在
- [ ] 批次大小合理：根据GPU显存调整
- [ ] 训练参数已配置：d_model、num_layers等

## 🎯 预期结果

### 训练输出示例
```
使用 cosine 学习率调度器
Warmup 步数: 500
总训练步数: 5000
Epoch 1 - Train: 100%|████████| 156/156 [02:30<00:00]
Epoch 1 Train Loss: 6.8234, LR: 2.50e-04
Epoch 1 - Val: 100%|████████| 39/39 [00:15<00:00]
Epoch 1 Val Loss: 6.2345
✓ 最佳模型已保存: decoder_only_best.pt (Val Loss: 6.2345)

Epoch 2 - Train: 100%|████████| 156/156 [02:28<00:00]
Epoch 2 Train Loss: 5.9123, LR: 2.00e-04
...
```

### 生成测试
```bash
cd scripts
python generate.py
```

应该能看到模型生成的文本（质量取决于训练程度）。

## 📞 支持

如果遇到问题：
1. 检查本清单中的常见问题
2. 运行`python test/test_fixes.py`确认环境
3. 查看完整错误堆栈
4. 检查`training.log`日志文件

## 🎉 总结

代码已经过全面修复和优化，完全兼容云平台环境：
- ✅ 5个严重问题已修复并测试通过
- ✅ 设备自动检测，优先使用CUDA
- ✅ 异常处理完善，训练可恢复
- ✅ 符合PyTorch官方标准实现
- ✅ 兼容Python 3.11和PyTorch 2.5.1.1

现在可以安全地在云平台上训练模型了！🚀
