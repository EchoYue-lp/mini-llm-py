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

## 工作流不是命令列表，而是产物依赖图

完整依赖关系：

```text
model snapshot ------------------------------+
                                                |
raw example definitions -> train/valid/test ----+-> training
                                                |      |
chat template/tokenizer ------------------------+      v
                                                |   adapter + config + history
                                                |      |
fixed test + parser + metrics ------------------+------+
                                                       v
                                                  evaluation report
```

任何上游产物变化都会使下游结果失去可比性。例如只改 system prompt，不重新训练却直接比较
adapter，输入 token 边界已经变化；只改 parser，历史模型的 JSON valid 指标也会变化。

建议为每次实验保存 manifest：

```json
{
  "git_commit": "...",
  "base_model": "Qwen/Qwen3-0.6B",
  "base_revision": "...",
  "data_sha256": "...",
  "tokenizer_revision": "...",
  "mlx_lm_version": "0.31.3",
  "seed": 0,
  "adapter_config": "adapter_config.json"
}
```

## 模型 Snapshot 与 Revision

`snapshot_download()` 当前指定 repo id 和本地目录，没有固定 revision。因此不同日期重新下载
可能得到不同上游提交。教学流程可接受，正式实验应：

```python
snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    revision="full-commit-sha",
    local_dir=...,
)
```

并记录关键文件哈希。模型名称相同不保证权重、配置、tokenizer 与 chat template 完全相同。

离线运行前要确认本地目录包含 config、tokenizer 和权重分片，而不是只检查目录存在。

## Chat Template 必须贯穿三条路径

训练：

```python
tokenizer.apply_chat_template(
    messages,
    enable_thinking=False,
)
```

推理/评测：

```python
tokenizer.apply_chat_template(
    messages_without_answer,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

需要保持一致的不是函数名，而是最终 token 序列：

- 相同 system prompt。
- 相同 role 顺序。
- 相同 thinking 开关。
- 相同 assistant prefix。
- 相同 tokenizer files。

可以在 smoke test 中保存一条训练样本的 prompt token ids，再与推理模板生成的前缀逐 token
比较。模板漂移通常不会引发加载错误，只会表现为质量突然下降。

## MLX Lazy Execution 的阶段边界

MLX 表达式通常 lazy 执行。训练循环中的 `mx.eval(...)` 是真正的同步边界。理解这一点才能
正确解释：

- 为什么 Python 循环结束不代表设备计算已经结束。
- 为什么计时前后需要物化结果。
- 为什么打印 `.item()` 会隐式同步。
- 为什么 optimizer state 也要包含在 eval 中。

测量步骤耗时的基本结构：

```python
start = time.perf_counter()
loss, gradients = loss_and_grad(model, batch, lengths)
optimizer.update(model, gradients)
mx.eval(model.parameters(), optimizer.state, loss)
elapsed = time.perf_counter() - start
```

若把 `mx.eval` 放到计时区间外，记录的只是 graph 构建时间。

## 统一内存不等于无限内存

Apple Silicon CPU/GPU 共享统一内存，减少显式 host-device copy，但模型权重、activation、
optimizer state、临时 kernel workspace、Python 进程和系统应用仍竞争同一物理容量。

OOM 调参优先级取决于占用来源：

| 参数 | 主要影响 |
| --- | --- |
| Sequence length | Attention/activation，常是强影响项 |
| Microbatch size | Activation 近似线性增加 |
| Adapted layers | LoRA 参数、梯度和部分图状态 |
| Rank | LoRA 参数/optimizer 状态，通常小于 activation 影响 |
| Grad accumulation | 有效 batch 增大，但不降低单 microbatch 峰值 |

关闭其他高内存应用可能改善可用容量，但不应替代记录可复现的训练配置。

## Adapter 加载协议

加载过程不是“找到 safetensors 就相加”：

1. 加载与训练一致的 base model。
2. 读取 `adapter_config.json`。
3. 根据 `num_layers` 与 `keys` 重建 LoRA module。
4. 根据 rank/scale 创建 A/B shape。
5. 加载 safetensors trainable keys。
6. 执行推理并验证输出有限。

因此以下任一变化都会失败或静默错配：

- Base 模型层数/模块命名改变。
- Adapter key 使用绝对路径，加载端期待相对路径。
- Rank 与权重 shape 不一致。
- Scale 规则不同。
- 使用了同名但不同 revision 的模型。

Smoke test 应在保存后启动一个全新进程重新加载 adapter，而不是只用内存中的已训练 model
推理；后者无法验证磁盘产物完整性。

## Short 与 Long 实验的隔离

两次训练必须使用不同 adapter 目录。否则可能发生：

- Long final 覆盖 short final。
- `best_adapters.safetensors` 来自不同配置。
- comparison 报告加载错目录。
- 旧 periodic snapshot 混入新实验。

运行前应检查目标目录是否为空或 manifest 是否匹配。不要自动删除未知目录；使用包含配置
摘要和 seed 的新目录更安全。

## 阶段 Gate：每步通过什么才继续

| 阶段 | 必须通过的 Gate |
| --- | --- |
| 环境 | 正确解释器、MLX/MLX-LM import、Metal device |
| 模型 | 本地完整加载、tokenizer/template 可用、基础生成有限 |
| 数据 | Validator 全通过、split 数量/分布/哈希已记录 |
| 注入 | 替换模块数非零、trainable ratio 符合预期、Delta-W 初始为 0 |
| 训练 | supervised token 非零、loss/grad 有限、参数确实更新 |
| 保存 | config + safetensors + history 可读 |
| 重载 | 新进程能加载 adapter 并生成 |
| 评测 | 固定 test/parser/sampler，逐样本报告已保存 |

上一 Gate 未通过时不要进入下一阶段。比如 adapter 无法重载，继续跑 300 步只会产生更昂贵
但不可部署的产物。

## 端到端 Preflight 示例

```bash
python -m evaluation.validate_tool_router_data
python -m finetuning.train_lora_short --dry-run
python -m finetuning.train_lora_short --iters 2 --num-layers 2
python -m inference.tool_router \
  "查一下订单A1024到哪里了" \
  --adapter artifacts/adapters/tool-router-short
python -m evaluation.tool_router \
  --adapter artifacts/adapters/tool-router-short \
  --label smoke \
  --output artifacts/results/tool-router/smoke.json
```

检查 smoke report 不只看命令退出码，还要确认：

```text
samples > 0
raw responses non-empty
metrics keys complete
predictions count == samples
adapter path recorded correctly
```

## 性能测量的可比条件

Tokens/s 和 peak memory 只有在以下配置相同时才可比较：

- 模型与 adapter targets。
- Microbatch、accumulation、sequence length 分布。
- dtype 与 MLX/MLX-LM 版本。
- 报告窗口是否包含 validation/save。
- 是否经过 warm-up。
- `mx.eval` 同步位置。

短训练的前几步包含模型/kernel warm-up，不能直接与长训练稳定阶段比较。Peak memory 是进程
生命周期高水位，某次保存或评测临时分配也可能抬高数值。

## 失败恢复边界

当前 periodic adapter 只保存 LoRA 权重，不保存 Adam/RNG/迭代器状态，因此可以用于：

- 加载推理。
- 从该权重开始一个新的训练实验。
- 比较不同训练阶段。

不能保证从第 N 步“精确续跑”出与不中断训练相同的第 N+1 步。若要支持精确 resume，需要
额外保存 optimizer state、iteration、RNG、数据顺序和未完成 accumulation 状态，并验证恢复
后下一步 loss/gradient 与原运行一致。

## 最终交付清单

一次可审计实验至少保留：

```text
adapter_config.json
best_adapters.safetensors
adapters.safetensors
training_history.json
training_summary.json
evaluation JSON reports
comparison.md
experiment manifest with revisions/hashes
```

还应明确部署使用 best 还是 final。文件存在不代表有效，应在全新进程中完成 load + one
generation + schema validation。

## 本章调试不变量

1. 所有命令从仓库根目录运行，路径经 `project_paths` 解析。
2. Base revision、tokenizer、chat template 在训练/推理/评测完全一致。
3. 每个阶段有明确输入、输出和通过 Gate，不跳步猜测。
4. MLX 计时与状态读取前经过 `mx.eval` 同步。
5. Short/long/smoke 使用隔离目录和可追溯 manifest。
6. Adapter 保存后必须在新进程重载验证。
7. 教学百分比始终同时报告样本数，不作为生产结论。

## 自测

1. Dry-run 能验证哪些内容，不能验证哪些内容？
2. 为什么 MLX 与 PyTorch 不共用一个 requirements 文件？
3. 两步 smoke test 应确认哪些产物？
4. Adapter 加载失败时最先检查什么？
5. 为什么先扩充测试集，再扩大模型？
