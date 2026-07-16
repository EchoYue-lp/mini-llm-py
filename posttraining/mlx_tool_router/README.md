# LLM Post-training Lab

[English](README_EN.md) | 中文

> 迁移状态：本项目代码已直接迁入
> `mini-llm-py/posttraining/mlx_tool_router/`。该目录是唯一维护入口，不再与原
> `llm-posttrain-lab` 仓库同步。

这是一个面向初学者的 LLM 后训练学习仓库。第一阶段使用
`Qwen/Qwen3-0.6B` 和 Apple MLX，在 Mac M1 Pro 16GB 上完成模型下载、
基座推理、工具路由数据构造、数据校验、LoRA 短训练、LoRA 长训练，以及统一
测试集上的效果对比。

本仓库不绑定 Qwen 或 LoRA。后续可以复用同一套数据和评测流程，迁移到其他
模型、4B 量化模型、GPU LoRA、全量微调、DPO 或 GRPO。

LoRA公式、矩阵维度、数据mask、训练循环、checkpoint和参数实验详见
[docs/lora-finetuning.md](docs/lora-finetuning.md)。

> 当前数据集只有 48 条，测试集只有 5 条。它用于学习和验证工程流程，不能
> 代表生产效果，也不能据此判断模型的真实能力。

## 学习目标

1. 从 Hugging Face 下载模型到明确的本地目录。
2. 在微调前运行基座模型并建立 baseline。
3. 理解意图、槽位、工具选择、缺参追问和 OOS 数据。
4. 在训练前检查数据 Schema、标签和数据泄漏。
5. 完成 LoRA 短训练并验证训练链路。
6. 对比基座模型和短训练模型。
7. 完成 LoRA 长训练并观察过拟合。
8. 在同一测试集上对比基座、短训和长训模型。

## 项目结构

```text
posttraining/mlx_tool_router/
├── scripts/
│   ├── download_model.py       # 下载模型
│   └── prepare_demo_data.py    # 生成教学数据
├── inference.py                # 微调前的基座推理
├── validate_data.py            # 数据质量校验
├── train_lora_short.py         # 40 步 LoRA 短训练
├── train_lora_long.py          # 300 步 LoRA 长训练
├── tool_router.py              # 加载指定 adapter 推理
├── evaluate.py                 # 评测单个模型或 adapter
├── compare_models.py           # 三模型对比和 Markdown 报告
├── requirements.txt
├── README.md
└── README_EN.md
```

运行时会产生 `models/`、`data/`、`adapters/`、`results/` 和 `.venv/`。
这些目录已经加入 `.gitignore`，不会推送模型权重和本地实验产物。

## 0. 环境准备

已验证环境：

- Apple M1 Pro，16GB 统一内存
- macOS 26.5.2
- Python 3.12.4
- MLX-LM 0.31.3
- Hugging Face Hub 1.23.0

硬件分档：

| 使用范围 | 已验证配置 | 推荐配置 | 说明 |
| --- | --- | --- | --- |
| 数据生成、校验和报告读取 | M1 Pro，16GB 统一内存 | 16GB 统一内存 | 不加载模型时资源占用很低，但没有单独声明未实测的内存下限 |
| Qwen3-0.6B 基座推理、LoRA smoke test | M1 Pro，16GB 统一内存 | 16GB 统一内存 | `batch_size=2` 的 2-step Metal 回归峰值约 1.834GB；8GB 设备未验证 |
| 40/300 步完整教学实验 | M1 Pro，16GB 统一内存 | 16-24GB 统一内存 | 已有实验峰值约 1.7-1.8GB，但还要给 macOS、Python 和文件缓存留余量 |
| 后续量化 4B、多 adapter 或更大 batch | 尚未验证 | 24-32GB 统一内存，或改用 CUDA GPU | 不建议仅凭模型权重大小估算训练内存 |

磁盘至少预留 `5GB`，推荐 `15-20GB`，用于 1.4GB 基座模型、虚拟环境、数据、多个
checkpoint、adapter 和评测结果。这个仓库的预期运行环境是 Apple Silicon + Metal；
Intel Mac、普通 CPU 或非 macOS 平台不属于当前已验证路径。

```bash
cd /Users/ls/MyWork/code/python/mini-llm-py/posttraining/mlx_tool_router
/opt/anaconda3/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

要求 Python 3.10 或更高版本。不要使用 macOS 自带的 Python 3.9，否则可能解析
到较旧的 MLX-LM。

```bash
python --version
python -c "import importlib.metadata as m; print(m.version('mlx-lm'))"
python -c "import mlx.core as mx; print(mx.default_device())"
```

最后一条应输出类似 `Device(gpu, 0)`。

## 1. 下载模型

```bash
python scripts/download_model.py
```

脚本使用 `huggingface_hub.snapshot_download`，将官方
`Qwen/Qwen3-0.6B` 下载到：

```text
models/Qwen3-0.6B/
```

目录中应包含：

```text
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
```

本机实际目录约 1.4GB。模型放在项目目录而不是默认缓存中，便于理解文件结构、
离线运行和管理磁盘空间。不要把 `models/` 提交到普通 Git 仓库。

## 2. 运行微调前的模型

```bash
python inference.py
```

自定义问题：

```bash
python inference.py \
  --prompt "用一句话解释什么是监督微调" \
  --max-tokens 80
```

本机验证输出：

```text
监督微调是指在预训练模型的基础上，通过监督学习对模型进行微调，以适应特定任务的需求。
```

`inference.py` 使用 `enable_thinking=False`。工具路由任务需要短、稳定、机器可
解析的 JSON，不需要输出思考过程。

这一步只证明模型和 MLX 环境正常。真正的 baseline 要使用固定测试集运行
`evaluate.py`。

## 3. 准备和校验数据

```bash
python scripts/prepare_demo_data.py
```

固定随机种子后生成：

```text
data/train.jsonl   38 条
data/valid.jsonl    5 条
data/test.jsonl     5 条
```

每条记录采用 Chat 格式，assistant 内容是序列化的 JSON：

```json
{
  "action": "call_tool",
  "intent": "query_logistics",
  "tool": "logistics_query",
  "arguments": {"order_id": "A1024"},
  "missing_arguments": []
}
```

数据覆盖三类动作：

| 行为 | action | 示例 |
| --- | --- | --- |
| 调用工具 | `call_tool` | 有订单号时查询物流 |
| 缺参追问 | `ask_clarification` | 查天气但缺少城市 |
| 不调用工具 | `no_tool` | 闲聊、知识问题、OOS |

工具 Schema：

| 工具 | 参数 |
| --- | --- |
| `weather_query` | `city`, `date` |
| `logistics_query` | `order_id` |
| `order_cancel` | `order_id` |
| `refund_query` | `refund_id` |

训练前运行：

```bash
python validate_data.py
```

它检查 JSONL、Chat 角色、输出字段、action、工具参数、缺参一致性、`no_tool`
一致性，以及 train/valid/test 中的完全重复输入。真实项目还应增加语义近似去重。

## 4. LoRA 短训练

`train_lora_short.py` 不再调用 `mlx_lm.lora` 命令，而是直接实现教学版 LoRA：

1. `EducationalLoRALinear` 实现 `xW + (alpha/r) * xAB`。
2. `A` 随机初始化、`B` 初始化为零，初始增量严格为零。
3. `inject_lora` 冻结基座参数，只替换最后若干层的 `q_proj`、`v_proj`。
4. `tokenize_chat_records` 计算 prompt mask，只监督 assistant JSON。
5. `causal_lm_loss` 显式完成 token 位移、交叉熵和 padding mask。
6. `value_and_grad`、梯度累积和 Adam 更新都在训练循环中可见。
7. `delta_weight` 和 `fuse` 展示如何把低秩更新合并回普通线性层。
8. 代码保存 final、best 和定期 checkpoint，并记录 loss、梯度范数和显存。

默认公式和目标层：

```text
output = xW + (alpha / rank) * dropout(x) @ A @ B
rank = 8, alpha = 16, scale = 2
targets = q_proj,v_proj
```

```bash
python train_lora_short.py
```

只查看底层命令，不执行训练：

```bash
python train_lora_short.py --dry-run
```

覆盖参数：

```bash
python train_lora_short.py \
  --iters 40 \
  --num-layers 8 \
  --targets q_proj,v_proj \
  --batch-size 1 \
  --grad-accumulation-steps 1 \
  --learning-rate 1e-4
```

adapter 保存到 `adapters/tool-router-short/`。

本机实测：

| 项目 | 数值 |
| --- | ---: |
| 训练步数 | 40 |
| 训练层数 | 最后 8 层 |
| 可训练参数 | 327,680 / 596,049,920（0.055%） |
| LoRA投影层 | 16个（8层 x q/v） |
| 峰值统一内存 | 约 1.67GB |
| 最佳验证 loss | 0.1737（第20步） |
| 最终验证 loss | 0.2710（第40步） |
| 最终测试 loss | 0.2289 |

短训练的目的，是验证数据加载、反向传播、adapter 保存和 adapter 加载，而不是
获得生产质量。

```bash
python tool_router.py \
  "查一下订单A1024到哪里了" \
  --adapter adapters/tool-router-short
```

## 5. 微调前和短训练效果对比

```bash
python evaluate.py \
  --label base \
  --output results/base.json

python evaluate.py \
  --adapter adapters/tool-router-short \
  --label short-lora \
  --output results/short-lora.json
```

指标含义：

| 指标 | 含义 |
| --- | --- |
| JSON 合法率 | 输出能否被程序解析 |
| 整条完全正确率 | 五个字段全部正确 |
| 动作准确率 | 调工具、追问、不调用是否正确 |
| 意图准确率 | 业务意图是否正确 |
| 工具准确率 | 工具名称或不调用是否正确 |
| 参数准确率 | 参数名称和值是否全部正确 |
| 缺参判断准确率 | 缺少哪些参数是否判断正确 |

本次教学结果：

| 模型 | JSON 合法率 | 整条完全正确率 | 意图准确率 |
| --- | ---: | ---: | ---: |
| 基座 | 100% | 0% | 0% |
| 短训 LoRA | 80% | 20% | 60% |

短训提高了任务标签匹配，但产生了一条非法 JSON。这说明不能只看意图准确率。

## 6. LoRA 长训练

```bash
python train_lora_long.py
```

```bash
python train_lora_long.py --dry-run
python train_lora_long.py --iters 300 --num-layers 16
```

adapter 保存到 `adapters/tool-router-long/`。

本机实测：

| 项目 | 数值 |
| --- | ---: |
| 训练步数 | 300 |
| 训练层数 | 最后 16 层 |
| 可训练参数 | 655,360 / 596,049,920（0.110%） |
| LoRA投影层 | 32个（16层 x q/v） |
| 峰值统一内存 | 约 1.77GB |
| 最低验证 loss | 0.0680（第110步） |
| 第300步验证 loss | 0.0797 |
| 第300步附近训练 loss | 0.0058 |
| 最终测试 loss | 0.0642 |

训练 loss 接近 0，而验证 loss 在低点后回升，是过拟合信号。真实训练应根据验证
集选择最佳 checkpoint，而不是默认使用最后一个 checkpoint。

## 7. 微调前、短训练和长训练完整对比

```bash
python compare_models.py
```

终端显示百分比摘要，详细报告写入：

```text
results/comparison.md
```

本次结果：

| 模型 | JSON合法 | 完全正确 | 动作 | 意图 | 工具 | 参数 | 缺参 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 100% | 0% | 40% | 0% | 60% | 80% | 40% |
| short-lora | 80% | 20% | 40% | 60% | 40% | 60% | 60% |
| long-lora | 100% | 80% | 80% | 100% | 80% | 80% | 80% |

`results/comparison.md` 会逐条显示用户输入、期望 JSON、三种模型实际输出和是否
完全正确。新手应先读逐条报告，再看汇总分数，因为汇总指标不会告诉你具体错因。

这次实验说明：

- 基座能输出 JSON，但不知道本项目自定义标签的语义。
- 短训开始学习格式，但小数据和少步数会导致不稳定。
- 长训提升整条准确率，仍会错误补全参数或判断缺参。
- 更多训练步数不能替代更多高质量数据。

## 8. 还需要补充的学习阶段

1. 扩展到 500-2,000 条经过复核的训练数据。
2. 增加相似意图边界、否定、口语、错别字和 ASR 错误。
3. 增加 OOS、无需工具、危险工具、缺参和多意图样本。
4. 建立更大的人工测试集，每个意图至少 50-100 条。
5. 增加语义近似去重和数据版本管理。
6. 根据验证集选最佳 checkpoint，并实现 early stopping。
7. 增加延迟、峰值内存和 tokens/s 基准测试。
8. 使用 JSON Schema 或约束解码保证线上输出合法。
9. 使用同一测试集评测 Qwen3-4B 量化模型。
10. 在 GPU 上复用相同数据和评测代码进行 LoRA 或全量微调。

## 9. LoRA 和全量微调

本仓库先用 LoRA，是因为 M1 16GB 能快速完成整个学习闭环。LoRA 不是仓库的最终
范围。迁移到全量微调时，数据 Schema、数据切分、独立测试集、指标、错误分析和
随机种子应该保持不变；变化的是训练入口、优化器状态、显存需求和 checkpoint。

4B 模型的全量微调通常应放到大显存 GPU 或多卡环境，不建议在 M1 16GB 上强行运行。

## 10. 常见问题

### 重命名项目后虚拟环境失效

虚拟环境入口包含绝对路径，重命名后需要重建：

```bash
/opt/anaconda3/bin/python3 -m venv --clear .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### loss 很低但预测仍然错误

常见原因包括数据太少、分布差异、标签边界不清、提示词缺少工具 Schema，以及
模型记住训练样本但没有学会决策边界。应查看 `results/comparison.md`，不能只看
训练日志。
