# 12 LoRA 训练、Checkpoint 与过拟合

本文关注 LoRA 数学之外的完整训练闭环。

## 学习目标

读完后应能：

1. 构造与 next-token shift 对齐的 prompt/loss mask。
2. 解释梯度累积、optimizer state 和 resume training。
3. 区分 final、best、periodic checkpoint。
4. 根据 train/validation/task metrics 判断训练状态。

## Chat 数据与 Prompt Mask

每条记录：

```text
system -> user -> assistant
```

生成两份 token：

1. 完整对话，包含 assistant 标准答案。
2. Prompt，只包含 system/user 和 assistant 起始标记。

```text
[system][user][assistant prefix][assistant answer]
|---------- loss=0 -----------|---- loss=1 ----|
```

只监督 assistant 答案，避免模型浪费训练信号去复述 prompt。

## Padding Batch

```text
sample 1: [t1 t2 t3 t4]
sample 2: [t1 t2 PAD PAD]
```

每条样本需要记录：

- `prompt_length`。
- `real_sequence_length`。

Loss mask 同时满足：

```text
position >= prompt_length
position < real_sequence_length
```

## Next-Token Shift

```python
inputs = batch[:, :-1]
targets = batch[:, 1:]
logits = model(inputs)
```

逐 token loss：

```text
average_loss =
  sum(cross_entropy(logits, targets) * mask)
  / supervised_token_count
```

SFT 优化的是答案 token 概率，不直接优化工具准确率，任务指标必须单独评测。

## MLX 反向传播

```python
loss_and_grad = nn.value_and_grad(model, causal_lm_loss)
(loss, token_count), gradients = loss_and_grad(model, batch, lengths)
```

基座参数冻结后，gradients 只包含 LoRA A/B。

## 梯度累积

```text
effective_batch =
  batch_size * grad_accumulation_steps
```

累积到更新点后取平均，再调用 optimizer。累积 batch 与真实大 batch 不总是完全等价，
因为 dropout 和更新频率不同。

## 梯度范数与 Adam

梯度 L2 范数：

```text
sqrt(sum(g_i^2))
```

用于发现梯度消失、爆炸或学习率过高。Adam 只为可训练 LoRA 参数维护动量，这是 LoRA
节省训练内存的重要来源。

## 参数统计

当前教学实验：

| 实验 | 目标范围 | 可训练参数 | 比例 |
| --- | --- | ---: | ---: |
| 短训 | 最后 8 层 q/v | 327,680 | 0.055% |
| 长训 | 最后 16 层 q/v | 655,360 | 0.110% |

## Checkpoint 类型

```text
adapters.safetensors
best_adapters.safetensors
0000020_adapters.safetensors
training_history.json
training_summary.json
overfitting_analysis.json
```

| 文件 | 含义 |
| --- | --- |
| final adapter | 训练停止时的参数 |
| best adapter | 验证 loss 最低时的参数 |
| periodic adapter | 定期快照 |
| history | train/validation 曲线与系统指标 |
| analysis | 最佳点与最终点的差异 |

最后一步不一定最好。

## Train、Validation、Test

```text
train -> 计算梯度
valid -> 选择超参数、checkpoint、early stopping
test  -> 所有决策结束后的最终评测
```

根据 test 结果反复调整配置会造成数据泄漏。

## Early Stopping

```bash
python -m finetuning.train_lora_long --patience 5
```

表示连续 5 次验证没有改善时停止。

## 过拟合判断

当前长训示例：

- 最佳验证 loss：0.0680，第 110 步。
- 第 300 步验证 loss：0.0797。
- 第 300 步附近训练 loss：0.0058。

训练 loss 继续下降而验证 loss 回升，是典型过拟合信号。部署时应优先验证集最佳
checkpoint，并结合任务指标确认。

## Adapter 兼容

训练保存 `adapter_config.json`：

```json
{
  "fine_tune_type": "lora",
  "num_layers": 8,
  "lora_parameters": {
    "rank": 8,
    "scale": 2.0,
    "keys": ["self_attn.q_proj", "self_attn.v_proj"]
  }
}
```

MLX-LM 根据配置重建 LoRA 层，再加载 safetensors 中的 A/B。

## 对照源码

阅读顺序：

1. `ExperimentConfig`
2. `tokenize_chat_records`
3. `make_batch`
4. `causal_lm_loss`
5. `inject_lora`
6. `run_experiment`
7. `run_long_experiment`
8. `analyze_learning_curve`

## Prompt Mask 的具体例子

假设 tokenized 对话：

```text
0  [SYSTEM]
1  system text
2  [USER]
3  user text
4  [ASSISTANT]
5  {
6  "action"
7  ...
10 }
```

若 `prompt_length=5`，监督位置从 assistant 答案开始。Next-token shift 后，mask 的
索引需要与 targets 对齐，不能简单把原 token mask 原样复用。

典型 off-by-one：

```text
原序列位置:   0 1 2 3 4 5 6
input 位置:    0 1 2 3 4 5
target 位置:   1 2 3 4 5 6
```

Loss mask 应描述 target 位置，而不是未经 shift 的 input 位置。

## 为什么只监督 Assistant

若 system/user 也贡献 loss，模型会花大量容量学习复制输入格式。SFT 的目标通常是：

```text
给定 prompt，预测 assistant response
```

但不同训练框架的 chat template 和 mask 规则不同，必须检查实际 token 序列，不能只相信
配置名称 `mask_prompt=True`。

## Batch 平均的陷阱

不同样本有效答案 token 数不同。更合理的 loss：

```text
所有有效 token loss 之和
/
所有有效 token 数
```

若先对每条样本平均，再对 batch 平均，短答案和长答案权重相同，结果与 token-level
平均不同。应明确项目采用哪种口径。

## 梯度累积与 Loss 缩放

若累积 `K` 个大小相同的 microbatch：

```text
每次 loss / K
backward K 次
optimizer.step
```

项目的 MLX 实现选择累加梯度后除以 K。两种方法在理想条件下等价。

最后不足 K 个 microbatch 时，不能仍除以 K，否则梯度会偏小。项目要求 iteration 数能
被 accumulation steps 整除，以避免该边界。

## Optimizer State 为什么不能随便丢

Adam 保存：

```text
m_t: 一阶动量
v_t: 二阶动量
step: 当前更新步
```

只恢复 adapter 权重但重新创建 Adam，会改变后续更新轨迹。若目标是“继续训练”，需要
恢复 optimizer；若只是“从已有 adapter 开始新的实验”，可以重新创建并明确记录。

## Best 与 Final 的选择

Final 表示最后训练状态，Best 表示验证指标最优状态。可能出现：

```text
train loss: 一直下降
valid loss: 先下降后上升
task accuracy: 在不同 step 波动
```

选择 checkpoint 时建议优先级：

1. 任务主指标。
2. Validation loss。
3. 输出格式稳定性。
4. 推理成本。

不能只按 train loss 选择。

## Reproducibility

至少记录：

- 随机种子。
- 数据文件哈希或版本。
- 基座模型 revision。
- Tokenizer revision。
- MLX/MLX-LM 版本。
- 所有训练参数。
- 目标层列表。
- 硬件与 dtype。

相同 seed 不保证跨框架、跨硬件 bitwise 一致，但能减少不必要变量。

## 如何阅读训练记录

### Train Loss

是否能拟合训练数据。

### Validation Loss

对未参与梯度的数据是否改善。

### Gradient Norm

是否接近 0、突然爆炸或长期剧烈波动。

### Tokens/s

训练吞吐。必须同时记录 batch、长度和硬件。

### Peak Memory

确认参数设置能否稳定运行，不代表常驻内存或最终模型大小。

### Delta-W Norm

LoRA 更新是否仍接近 0，或变得异常大。

## 过拟合以外的异常

| 现象 | 可能原因 |
| --- | --- |
| Train loss 不下降 | Mask 错误、参数未解冻、学习率过低 |
| Loss 很快变 NaN | 学习率过高、数值不稳定、空 mask |
| Valid 明显优于 Train | Dropout、数据难度或统计口径不同 |
| Task 指标不升 | Loss 与业务字段不对齐、数据边界不足 |
| JSON 合法率下降 | 生成配置或结构化约束不足 |

## 常见错误

1. Prompt mask 与 next-token shift 错一位。
2. Padding token 被计入 loss。
3. 梯度累积忘记平均。
4. 保存 best adapter 后又被 final 覆盖。
5. 使用 test 集选择 checkpoint。
6. 只保存权重却声称可以精确 resume。
7. 训练历史没有记录配置和数据版本。

## 动手练习

1. 打印一条样本的 token、prompt_length 和 loss mask。
2. 手算一个变长 batch 的有效 token 数。
3. 比较 token-level 与 sample-level 平均 loss。
4. 用短 patience 运行长训，观察 early stopping。
5. 分别加载 best/final adapter 做固定测试集评测。

## 自测

1. Prompt mask 为什么要考虑 next-token shift？
2. Resume training 与从 adapter 开始新实验有什么区别？
3. 为什么 final checkpoint 不一定最好？
4. 为什么 validation loss 与任务准确率都要看？
5. 复现实验至少应记录哪些信息？
