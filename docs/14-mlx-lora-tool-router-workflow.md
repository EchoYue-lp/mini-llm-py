# 14 MLX LoRA 工具路由完整流程

本实验使用 Qwen3-0.6B、Apple MLX 和 48 条教学数据，覆盖从模型下载到三模型评测的
完整后训练链路。

## 学习目标

完成后应能：

1. 独立建立 MLX 环境并验证 Metal。
2. 从模型下载、数据生成走到短训和长训。
3. 加载 adapter 推理并运行固定测试集评测。
4. 根据错误信息定位模型、数据、目标层、内存和 adapter 问题。

## 环境

```bash
python -m venv .venv-mlx
source .venv-mlx/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mlx.txt
```

检查 Metal：

```bash
python -c "import mlx.core as mx; print(mx.default_device())"
```

已验证：

- Apple M1 Pro，16 GB 统一内存。
- Python 3.12.4。
- MLX-LM 0.31.3。

## 目录

```text
artifacts/models/Qwen3-0.6B/
artifacts/adapters/tool-router-short/
artifacts/adapters/tool-router-long/
artifacts/results/tool-router/
data/tool_router/
```

运行产物与源码分离，并由 Git 忽略。

## 1. 下载模型

```bash
python -m scripts.download_mlx_model
```

模型保存到 `artifacts/models/Qwen3-0.6B/`。

## 2. 基座推理

```bash
python -m inference.base_model \
  --prompt "用一句话解释什么是监督微调" \
  --max-tokens 80
```

这一步验证模型和 MLX 环境，但不代表工具路由 baseline。

## 3. 准备数据

```bash
python -m scripts.prepare_tool_router_data
python -m evaluation.validate_tool_router_data
```

生成：

```text
data/tool_router/train.jsonl
data/tool_router/valid.jsonl
data/tool_router/test.jsonl
```

## 4. 短训练

```bash
python -m finetuning.train_lora_short --dry-run
python -m finetuning.train_lora_short
```

默认配置：

```text
iterations = 40
num_layers = 8
targets = q_proj,v_proj
rank = 8
alpha = 16
learning_rate = 1e-4
```

输出到 `artifacts/adapters/tool-router-short/`。

## 5. Adapter 推理

```bash
python -m inference.tool_router \
  "查一下订单A1024到哪里了" \
  --adapter artifacts/adapters/tool-router-short
```

## 6. 基座与短训评测

```bash
python -m evaluation.tool_router \
  --label base \
  --output artifacts/results/tool-router/base.json

python -m evaluation.tool_router \
  --adapter artifacts/adapters/tool-router-short \
  --label short-lora \
  --output artifacts/results/tool-router/short-lora.json
```

## 7. 长训练

```bash
python -m finetuning.train_lora_long --dry-run
python -m finetuning.train_lora_long --iters 300 --num-layers 16
```

输出到 `artifacts/adapters/tool-router-long/`，包含 best/final adapter、训练历史与
过拟合分析。

## 8. 三模型对比

```bash
python -m evaluation.compare_tool_router_models
```

报告：

```text
artifacts/results/tool-router/comparison.md
```

## 教学实测

| 模型 | JSON 合法 | 完全正确 | Action | Intent | Tool | Arguments | Missing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 100% | 0% | 40% | 0% | 60% | 80% | 40% |
| Short LoRA | 80% | 20% | 40% | 60% | 40% | 60% | 60% |
| Long LoRA | 100% | 80% | 80% | 100% | 80% | 80% | 80% |

测试集只有 5 条，这些百分比只能证明流程工作。

## 资源记录

| 实验 | 可训练参数 | 峰值统一内存 | 最佳验证 loss |
| --- | ---: | ---: | ---: |
| Short | 327,680 | 约 1.67 GB | 0.1737 |
| Long | 655,360 | 约 1.77 GB | 0.0680 |

模型目录约 1.4 GB。建议为环境、模型、adapter 和报告预留 15-20 GB 磁盘。

## 进入真实实验前

1. 扩充到至少数百条经过审核的数据。
2. 建立每个意图 50-100 条的独立测试集。
3. 加入语义近似去重和数据版本管理。
4. 使用验证集最佳 checkpoint。
5. 增加 JSON Schema 或受约束解码。
6. 比较更大模型、GPU LoRA、全量微调、DPO 或 GRPO。

## 相关文档

- `11-lora-low-rank-adaptation.md`
- `12-lora-training-checkpoints-and-overfitting.md`
- `13-tool-routing-data-and-evaluation.md`

## 开始前检查

### 硬件

MLX 面向 Apple Silicon。确认：

```bash
uname -m
```

预期输出 `arm64`。

### Python

```bash
python --version
```

建议 Python 3.11 或 3.12。不要混用系统 Python、Conda Python 和不同虚拟环境中的 pip。

确认当前解释器：

```bash
which python
python -m pip --version
```

两条路径应指向 `.venv-mlx`。

### MLX

```bash
python -c "import mlx, mlx_lm; print('MLX ready')"
```

若导入失败，先修复环境，不要直接调训练参数。

## 为什么使用独立 MLX 环境

PyTorch 和 MLX 是两套独立运行时：

- PyTorch 环境支持 CPU/CUDA/MPS 和根项目测试。
- MLX 环境面向 Apple Silicon 后训练。

将二者强行安装到同一环境会增加版本冲突，也会让非 Mac 用户无法正常安装项目基础
依赖。共享的是源码、数据格式和评测思想，不是底层包。

## 每一步的输入与输出

| 步骤 | 输入 | 输出 |
| --- | --- | --- |
| 下载模型 | Hugging Face repo id | `artifacts/models/Qwen3-0.6B/` |
| 基座推理 | 模型 + prompt | 文本响应 |
| 准备数据 | 内置样本模板 | 三个 JSONL split |
| 数据校验 | JSONL | 记录数与 action 分布 |
| 短训练 | 模型 + train/valid/test | short adapter |
| Adapter 推理 | 模型 + adapter + 用户输入 | JSON 决策 |
| 单模型评测 | 模型/adapter + test | JSON report |
| 长训练 | 更长训练配置 | long adapter + analysis |
| 三模型比较 | 三份模型结果 | comparison.md |

清楚每一步的产物，可以避免在错误目录中寻找文件。

## Dry-Run 检查什么

```bash
python -m finetuning.train_lora_short --dry-run
```

Dry-run 应展示：

- 模型路径。
- 数据路径。
- Adapter 路径。
- Iterations。
- 目标层。
- Rank/alpha/scale。
- Batch 与梯度累积。

它不证明模型文件可加载，也不执行反向传播。真正 smoke test 至少需要少量 iterations。

## 推荐的 Smoke Test

完整 40/300 步前，先运行：

```bash
python -m finetuning.train_lora_short \
  --iters 2 \
  --num-layers 2 \
  --batch-size 1
```

确认：

1. 模型成功加载。
2. 找到目标 projection。
3. Trainable parameter 不为 0。
4. Loss 是有限数。
5. Adapter 文件能够保存。
6. Adapter 可以重新加载推理。

Smoke test 通过后再扩大训练。

## 常见报错排查

### Missing Model

```text
Missing artifacts/models/Qwen3-0.6B
```

运行：

```bash
python -m scripts.download_mlx_model
```

### Missing Data

```text
Missing data/tool_router/train.jsonl
```

运行：

```bash
python -m scripts.prepare_tool_router_data
python -m evaluation.validate_tool_router_data
```

### No Target Modules

可能原因：

- 模型结构与目标层名称不同。
- `--targets` 拼写错误。
- `num_layers` 范围不正确。

先打印模型模块名，不要盲目增加 target。

### Adapter Load Failure

检查：

- Base model 是否与训练时一致。
- `adapter_config.json` 是否存在。
- Rank、scale、num_layers 是否匹配。
- Safetensors key 是否对应目标模块。

### Loss 为 NaN

依次检查：

1. Loss mask 是否至少有一个有效 token。
2. 学习率是否过高。
3. 输入是否超过最大长度。
4. 数据中 assistant 答案是否为空。
5. Gradient norm 是否在 NaN 前异常增大。

### Metal Out of Memory

优先减少：

- Batch size。
- Sequence length。
- Adapted layers。
- Rank。

梯度累积可维持较大有效 batch，但不能减少单个 microbatch 的 activation。

## 目录清理

运行产物均在：

```text
artifacts/
data/tool_router/
```

清理前先确认需要保留：

- Best adapter。
- Adapter config。
- Training summary。
- Evaluation reports。

不要将基座模型、adapter 或数据误提交到 Git。

## 实验命名

默认 short/long 只适合教学。真实实验建议命名包含关键信息：

```text
tool-router-r8-lr1e-4-layers8-seed0
```

并在目录中保存完整 config，避免多个实验互相覆盖。

## 一次可靠实验的记录

```text
experiment name
git commit
base model revision
data version
random seed
target modules
rank / alpha / dropout
batch / accumulation
learning rate
best validation metric
test metrics
peak memory
tokens/s
```

这些信息比只保存一个 adapter 文件更重要。

## 从教学实验升级

建议按顺序扩展：

1. 先增加数据质量和边界覆盖。
2. 扩大 validation/test。
3. 使用 best checkpoint 与 early stopping。
4. 增加结构化约束。
5. 比较不同 rank、target、学习率。
6. 再尝试更大基座模型。
7. 最后考虑 DPO、GRPO 或全量微调。

训练方法越复杂，越需要稳定的数据与评测基础。

## 自测

1. Dry-run 能验证哪些内容，不能验证哪些内容？
2. 为什么 MLX 与 PyTorch 不共用一个 requirements 文件？
3. 两步 smoke test 应确认哪些产物？
4. Adapter 加载失败时最先检查什么？
5. 为什么先扩充测试集，再扩大模型？
