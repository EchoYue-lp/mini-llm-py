# LoRA 微调：从公式到完整训练循环

本文配合以下代码阅读：

- `train_lora_short.py`：LoRA 数学、数据处理和短训练闭环。
- `train_lora_long.py`：显式长训练、early stopping 和过拟合分析。
- `tool_router.py`：加载基座模型和 LoRA adapter 进行推理。
- `evaluate.py`、`compare_models.py`：任务指标和逐样本评测。

当前实验使用 `Qwen/Qwen3-0.6B`、MLX 和 Mac M1 Pro 16GB。代码没有调用
`mlx_lm.lora` CLI；模型和 tokenizer 由 MLX-LM 加载，LoRA 层、batch、loss、
反向传播、优化器和 checkpoint 均在项目代码中明确实现。

## 1. LoRA 解决什么问题

全量微调会更新模型中的全部权重。对于一个线性层：

```text
y = xW
```

假设：

| 符号 | 形状 | 含义 |
| --- | --- | --- |
| `x` | `[batch, seq, in]` | 输入隐藏状态 |
| `W` | `[out, in]` | 预训练线性层权重 |
| `y` | `[batch, seq, out]` | 输出隐藏状态 |

全量微调直接学习一个与 `W` 同样大的更新矩阵。LoRA 假设任务所需更新可以用低秩
矩阵表达：

```text
Delta-W = (alpha / rank) * B^T @ A^T
```

在前向传播中写成：

```text
y = xW + (alpha / rank) * dropout(x) @ A @ B
```

维度为：

| 参数 | 形状 |
| --- | --- |
| `A` | `[in, rank]` |
| `B` | `[rank, out]` |
| `x @ A @ B` | `[batch, seq, out]` |

当 `rank` 远小于 `in/out` 时，训练参数从 `in * out` 降为：

```text
in * rank + rank * out
```

例如 `in=1024`、`out=1024`、`rank=8`：

```text
全量参数：1024 * 1024 = 1,048,576
LoRA参数：1024 * 8 + 8 * 1024 = 16,384
```

单层可训练参数约减少到原来的 1.56%。

## 2. 为什么冻结 W

代码先执行：

```python
model.freeze()
self.linear.freeze()
```

因此：

- 预训练权重 `W` 参与前向计算。
- 梯度不会更新 `W`。
- 优化器只接收 LoRA 的 `A` 和 `B`。
- 不同任务可以共享同一份基座模型，只保存各自 adapter。

冻结不代表模型不使用基座知识，而是保持基座知识不变，在旁路增加任务增量。

## 3. 为什么 A 随机、B 为零

初始化方式：

```python
A = random_uniform(...)
B = zeros(...)
```

训练开始时：

```text
A @ B = 0
```

所以：

```text
y = xW + 0 = xW
```

LoRA 注入不会在第0步破坏基座输出。实测的初始最大差异为 `0`。

如果 `A` 和 `B` 都初始化为零，两者第一步梯度可能无法形成有效的对称性破坏；如果
两者都随机初始化，模型在训练前就会产生随机偏移。因此常用“A随机、B为零”。

## 4. rank、alpha、scale 和 dropout

本项目默认：

```text
rank = 8
alpha = 16
scale = alpha / rank = 2
dropout = 0
```

### rank

- rank越大，表达能力和可训练参数越多。
- rank越小，显存和adapter文件越小。
- 入门实验可从4、8、16比较。

### alpha

`alpha/rank` 控制 LoRA 分支的整体强度。调整 rank 时保留 alpha/rank 的相对关系，
可以避免更新尺度变化过大。

### dropout

dropout只应用于LoRA分支，不修改基座分支。小数据过拟合时可尝试 `0.05` 或 `0.1`，
但应通过验证集决定，而不是凭感觉设置。

## 5. 为什么选择 q_proj 和 v_proj

Transformer注意力中常见投影：

```text
q_proj：生成Query
k_proj：生成Key
v_proj：生成Value
o_proj：输出投影
```

本项目默认：

```python
DEFAULT_TARGETS = ("q_proj", "v_proj")
```

这是一个参数量较小、常见的起点。代码扫描最后N个Transformer Block，将匹配的
`nn.Linear` 替换成 `EducationalLoRALinear`。

可以实验：

```bash
python train_lora_short.py --targets q_proj,v_proj
python train_lora_short.py --targets q_proj,k_proj,v_proj,o_proj
python train_lora_short.py --targets q_proj,v_proj,up_proj,down_proj
```

目标层越多不一定越好。需要同时比较：

- 可训练参数量；
- 峰值内存；
- 训练速度；
- 验证loss；
- 工具调用完整准确率。

## 6. Chat数据如何变成训练Token

每条数据包含：

```text
system -> user -> assistant
```

代码生成两份Token：

1. 完整对话Token：包含assistant标准答案。
2. Prompt Token：只包含system和user，并带assistant生成起始标记。

Prompt Token长度就是 `prompt_length`。

```text
[system tokens][user tokens][assistant prefix][assistant answer]
|------------- loss=0 -----------------|------ loss=1 ------|
```

这叫 prompt masking。若不mask，模型会浪费训练信号去复述system和user内容；本任务
真正需要学习的是assistant输出的意图、工具和参数JSON。

训练与推理都使用 `enable_thinking=False`，避免结构化JSON任务混入思考文本。

## 7. Padding batch如何构造

同一batch中的句子长度不同，需要补齐：

```text
样本1：[t1 t2 t3 t4]
样本2：[t1 t2 PAD PAD]
```

代码同时保存：

```text
[prompt_length, real_sequence_length]
```

loss mask结合这两个长度：

- `position >= prompt_length`：排除Prompt。
- `position < real_sequence_length`：排除Padding。因为next-token shift之后，有效目标
  位置是 `1..real_sequence_length-1`；若写成 `<=`，变长batch中较短样本的第一个
  padding token会被错误计入loss。

Padding只用于形成规则矩阵，不应贡献loss。

## 8. Next-token shift与因果交叉熵

语言模型用当前位置预测下一个Token：

```python
inputs = batch[:, :-1]
targets = batch[:, 1:]
logits = model(inputs)
```

示意：

```text
原Token： [BOS, 我, 要, 查, 订单]
inputs：  [BOS, 我, 要, 查]
targets： [我,  要, 查, 订单]
```

模型输出每个位置对整个词表的logits，交叉熵衡量正确下一个Token的概率：

```python
per_token_loss = cross_entropy(logits, targets)
average_loss = sum(per_token_loss * mask) / supervised_token_count
```

这是监督微调的核心目标。模型不是直接优化“意图准确率”，而是优化标准JSON答案的
Token概率。任务指标需要在独立评测阶段另外计算。

## 9. value_and_grad与反向传播

MLX代码：

```python
loss_and_grad = nn.value_and_grad(model, causal_lm_loss)
(loss, token_count), gradients = loss_and_grad(model, batch, lengths)
```

发生的步骤：

```text
Token -> 模型前向 -> logits -> masked CE loss
                                 |
                                 v
                    对LoRA A/B求偏导
```

因为基座参数已冻结，`gradients` 中只有可训练LoRA参数。

## 10. 梯度累积和有效batch

当显存不足以放大batch时，可以累积多个microbatch：

```text
effective_batch_size = batch_size * grad_accumulation_steps
```

代码先把梯度相加：

```python
accumulated_gradients += gradients
```

到更新点后取平均：

```python
accumulated_gradients /= grad_accumulation_steps
optimizer.update(model, accumulated_gradients)
```

示例：

```bash
python train_lora_short.py \
  --batch-size 1 \
  --grad-accumulation-steps 8
```

其有效batch接近8，但和真正batch=8并非在所有情况下完全等价，例如dropout随机性和
优化器更新频率不同。

## 11. 梯度范数和Adam

梯度L2范数：

```text
sqrt(sum(g_i^2))
```

它用于观察：

- 梯度是否接近0；
- 是否突然变得很大；
- 学习率是否可能过高；
- 训练是否出现不稳定。

Adam为每个参数维护一阶和二阶动量。LoRA减少了需要保存梯度和优化器状态的参数量，
这也是它节省训练内存的重要原因，不只是模型权重更少。

本项目使用常量学习率 `1e-4`，便于学习。更正式的实验应增加warmup和衰减策略。

## 12. 参数量如何统计

代码分别统计：

```text
total_parameters
trainable_parameters
trainable_ratio
```

当前实测：

| 实验 | 目标范围 | 可训练参数 | 比例 |
| --- | --- | ---: | ---: |
| 短训 | 最后8层q/v | 327,680 | 0.055% |
| 长训 | 最后16层q/v | 655,360 | 0.110% |

参数量计算可由单层公式验证：

```text
每个LoRA线性层 = in*rank + rank*out
总LoRA参数 = 所有目标线性层LoRA参数之和
```

## 13. final、best和定期checkpoint

训练输出：

```text
adapters.safetensors
best_adapters.safetensors
0000020_adapters.safetensors
0000040_adapters.safetensors
...
```

含义：

| 文件 | 含义 |
| --- | --- |
| `adapters.safetensors` | 训练停止时的最终参数 |
| `best_adapters.safetensors` | 验证loss最低时的参数 |
| `0000020_...` | 第20步定期快照 |

最后一步不一定最好。当前长训在第110步验证loss最低，训练到第300步后验证loss上升
17.28%，因此出现过拟合警告。

## 14. Train、Valid、Test必须隔离

三个集合职责：

```text
train：计算梯度，更新A/B
valid：选超参数、best checkpoint、early stopping
test：所有训练决策结束后，做最终一次客观评测
```

严禁根据test结果反复修改超参数，否则test实际上已经变成valid，最终指标会虚高。

`train_lora_long.py`支持：

```bash
python train_lora_long.py --patience 5
```

表示连续5次验证没有改善时停止。默认 `0` 表示关闭early stopping，方便完整观察
过拟合曲线。

## 15. Delta-W和fuse

训练时保存的是A和B，不需要复制完整W：

```text
Delta-W = scale * B^T @ A^T
```

推理有两种方式：

1. 动态adapter：运行时计算 `xW + xAB`。
2. 融合权重：提前计算 `W_fused = W + Delta-W`。

`fuse()`生成一个普通线性层。项目数学测试中：

```text
B=0时LoRA与基座最大差异：0
融合前后最大差异：约5.96e-08
```

微小差异来自浮点计算顺序。

融合后的优点是推理图更简单；缺点是失去快速切换adapter的能力，并需要保存完整融合
模型。

## 16. Adapter格式为什么能被tool_router加载

LoRA层采用与MLX-LM兼容的字段：

```text
linear
lora_a
lora_b
```

训练同时保存 `adapter_config.json`：

```json
{
  "fine_tune_type": "lora",
  "num_layers": 8,
  "lora_parameters": {
    "rank": 8,
    "scale": 2.0,
    "dropout": 0.0,
    "keys": ["self_attn.q_proj", "self_attn.v_proj"]
  }
}
```

因此 `tool_router.py` 可以执行：

```python
model, tokenizer = load(base_model, adapter_path=adapter_directory)
```

MLX-LM先根据配置重建LoRA层，再从`safetensors`加载A/B。

## 17. 训练记录

每个实验会生成：

```text
training_history.json
training_summary.json
overfitting_analysis.json  # 长训
```

记录内容：

- train loss；
- validation loss；
- gradient L2 norm；
- trained tokens；
- tokens/s；
- peak memory；
- best iteration；
- test loss和perplexity；
- 初始和最终Delta-W范数；
- 被替换的模块名称。

任务准确率由 `compare_models.py` 另外计算。训练loss和任务准确率是不同维度，二者都要看。

## 18. 短训与长训的区别

| 项目 | 短训 | 长训 |
| --- | ---: | ---: |
| 训练步数 | 40 | 300 |
| LoRA层数 | 最后8层 | 最后16层 |
| q/v投影数量 | 16 | 32 |
| 可训练参数 | 327,680 | 655,360 |
| 最佳验证loss | 0.1737，第20步 | 0.0680，第110步 |
| 最终完整准确率 | 20% | 80% |

当前测试集只有5条，上述百分比只能证明代码流程工作，不能代表生产能力。

## 19. 推荐阅读代码顺序

1. `ExperimentConfig`：先理解实验由哪些参数定义。
2. `EducationalLoRALinear`：理解公式、初始化、Delta-W和fuse。
3. `tokenize_chat_records`：理解Chat Template和prompt mask。
4. `make_batch`：理解padding和真实长度。
5. `causal_lm_loss`：理解next-token监督目标。
6. `inject_lora`：理解冻结、目标层和模块替换。
7. `run_experiment`：阅读短训完整循环。
8. `run_long_experiment`：阅读长训、early stopping和checkpoint。
9. `analyze_learning_curve`：理解训练loss与验证loss的差异。
10. `tool_router.py`：理解adapter如何重新加载。

## 20. 建议练习

按一次只改一个变量的方式实验：

1. rank：`4 / 8 / 16`。
2. target：`q,v` 与 `q,k,v,o`。
3. 层数：最后 `4 / 8 / 16 / 28` 层。
4. learning rate：`5e-5 / 1e-4 / 2e-4`。
5. dropout：`0 / 0.05 / 0.1`。
6. gradient accumulation：`1 / 4 / 8`。
7. early stopping patience：`0 / 3 / 5`。

每次记录：

```text
配置 -> 可训练参数 -> 峰值内存 -> 最佳验证loss
     -> 完整准确率 -> 缺参准确率 -> 逐样本错误
```

不要同时修改多个变量，否则无法判断收益来自哪里。
