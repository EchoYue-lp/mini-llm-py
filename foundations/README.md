# Foundations 前置课程

`foundations/` 是进入 `labs/` 之前的可运行课程。它不追求覆盖完整的高等数学、NumPy、
Pandas 或 PyTorch API，而是集中训练后续 Transformer 代码会反复使用的能力。

## 推荐顺序

| 顺序 | 模块 | 必须掌握 |
| --- | --- | --- |
| F00 | `f00_math_basics` | 向量、矩阵、点积、方差、Softmax、交叉熵、数值导数 |
| F01 | `f01_numpy_basics` | shape、dtype、切片、广播、矩阵乘法、reshape、transpose |
| F02 | `f02_embedding_geometry` | one-hot、查表、共现矩阵、低秩 SVD、上下文化 |
| F03 | `f03_pandas_basics` | JSONL、DataFrame、缺失值、筛选、groupby、merge |
| F04 | `f04_pytorch_tensors` | Tensor、device、Linear shape、多头重排、Mask |
| F05 | `f05_pytorch_autograd` | 计算图、梯度、VJP、残差导数、参数冻结 |
| F06 | `f06_pytorch_training` | `nn.Module`、DataLoader、Loss、Optimizer、train/eval |

## 运行命令

逐节运行：

```bash
python -m foundations.f00_math_basics
python -m foundations.f01_numpy_basics
python -m foundations.f02_embedding_geometry
python -m foundations.f03_pandas_basics
python -m foundations.f04_pytorch_tensors
python -m foundations.f05_pytorch_autograd
python -m foundations.f06_pytorch_training
```

一次运行全部课程：

```bash
python -m foundations
```

这些命令不下载数据，也不需要 checkpoint。F00 只依赖 Python 标准库，F01-F02 需要 NumPy，
F03 需要 Pandas，F04-F06 需要 PyTorch。

## F00 数学：把公式翻译成代码

对应模块：`foundations.f00_math_basics`

这一节不追求完整线性代数课程，而是建立后续代码必需的六个连接：

```text
向量点积       -> Attention 中一个 query-key score
矩阵乘法       -> Linear、Q/K/V projection、FFN
均值与方差     -> LayerNorm、RMSNorm、初始化尺度
稳定 Softmax   -> Attention weights、分类概率
交叉熵         -> next-token loss
数值导数       -> 梯度是局部变化率
```

必须能手算：

```text
[2,3] @ [3,4] -> [2,4]
dot([1,2], [3,4]) = 11
softmax(z) = softmax(z - max(z))
```

本节使用 Python list 手写小矩阵运算，是为了暴露求和发生在哪个维度。真正工程计算应使用
NumPy、PyTorch 或底层优化 kernel。

## F01 NumPy：建立张量 shape 直觉

对应模块：`foundations.f01_numpy_basics`

NumPy 是理解 PyTorch Tensor 的低成本入口。重点不是记 API 名称，而是明确每次操作是否：

- 改变 shape。
- 改变轴顺序。
- 复制数据或只创建 view。
- 沿某个轴归约。
- 通过 broadcasting 扩展逻辑 shape。

多头拆分示例：

```text
[B,T,D]
-> reshape [B,T,H,Dh]
-> transpose [B,H,T,Dh]
```

`reshape` 负责重新分组元素，`transpose` 负责交换轴语义，两者不能互相替代。广播也不是
随意匹配 shape：它从末尾维度开始比较，每一对维度必须相等或其中一个为 1。

## F02 Embedding Geometry：观察高维到低维

对应模块：`foundations.f02_embedding_geometry`

这一节把 01 文档中的抽象过程变成可观察实验：

```text
one-hot [V]
@ embedding table [V,D]
= lookup row [D]
```

随后从 `[V,V]` token 共现统计构造 PPMI 矩阵，再用截断 SVD 得到 `[V,D]` 表示。它展示了
低秩近似如何保留部分共享结构，也直接显示重建误差，强调降维是有损的。

最后用一个简单 context mixer 演示：同一个静态 token vector 加入不同邻居后，会形成不同
contextual state。真正 Transformer 使用多层 Attention/FFN 学习这种上下文化，而不是固定
求平均。

## F03 Pandas：检查数据，而不是训练模型

对应模块：`foundations.f03_pandas_basics`

需要掌握的最小 DataFrame 工作流：

```text
JSONL records
-> DataFrame
-> required-column validation
-> missing/empty row filtering
-> derived columns
-> groupby summary
-> merge metadata or evaluation results
```

这一节用 `word_count = text.str.split().str.len()` 演示派生列。它只是按空白统计英文词数，
不是模型 tokenizer 的 token 数；中文、子词和特殊 token 必须调用项目实际 tokenizer
统计。

Pandas 适合批量检查和聚合，不适合模型训练内循环。避免使用 Python `for` 或
`DataFrame.iterrows()` 逐行做本可向量化的运算。

## F04 PyTorch Tensor：从数组进入模型运行时

对应模块：`foundations.f04_pytorch_tensors`

PyTorch Tensor 在 NumPy shape 语义上增加了三个工程维度：

| 属性 | 典型问题 |
| --- | --- |
| `dtype` | FP32、FP16、BF16、Long、Bool 是否符合算子要求 |
| `device` | Tensor 和参数是否同时位于 CPU、CUDA 或 MPS |
| autograd 状态 | 当前运算是否需要构建反向图 |

必须理解：

- Token id 通常是 `torch.long`，不能直接作为浮点特征使用。
- Mask 通常用 `torch.bool`，并明确 `True` 是可见还是屏蔽。
- `nn.Linear(in,out).weight` 的 shape 是 `[out,in]`。
- `transpose` 后的 Tensor 可能不 contiguous，直接 `view` 前常需 `.contiguous()`。
- `torch.from_numpy` 通常共享 CPU 内存，`torch.tensor(array)` 通常复制数据。

## F05 Autograd：沿计算图追踪梯度

对应模块：`foundations.f05_pytorch_autograd`

反向传播不是“自动猜参数怎么改”，而是从标量 loss 出发，按链式法则计算梯度。

| 操作 | 含义 |
| --- | --- |
| `requires_grad=True` | 允许记录与该 Tensor 有关的可微运算 |
| `loss.backward()` | 从 loss 反向累积叶子 Tensor 的 `.grad` |
| `torch.autograd.grad` | 直接请求指定输入的梯度或 VJP |
| `.detach()` | 返回与原计算图断开的 Tensor view |
| `torch.no_grad()` | 在代码块内关闭梯度记录 |
| 参数 `requires_grad=False` | 不求该参数梯度，但仍可把梯度传给输入 |

残差例子 `y = x + x^2` 的导数是 `1 + 2x`。其中的 `1` 就是恒等路径贡献，后续理解
Pre-LN 时会再次出现。

## F06 PyTorch Training：闭合最小训练循环

对应模块：`foundations.f06_pytorch_training`

标准顺序：

```python
optimizer.zero_grad()
logits = model(features)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

每一步分别解决：清除旧梯度、执行前向、定义目标、计算梯度、更新参数。顺序写对还不够，
还需要确认：

- `DataLoader` 输出 shape 与模型输入一致。
- `CrossEntropyLoss` 接收 logits，不是 Softmax 后的概率。
- 标签 dtype 是 `torch.long`，值域落在类别范围内。
- 训练时调用 `model.train()`，评估时调用 `model.eval()`。
- 评估使用 `torch.no_grad()`，避免保存无用反向图。
- 观察 loss 是否下降，同时用任务指标检查模型是否真的学对。

本节使用一个线性可分的二维数据集。它的价值是验证训练闭环，不代表真实语言模型任务。

## 常见错误速查

| 现象 | 优先检查 |
| --- | --- |
| `matmul` shape error | 左矩阵最后一维与右矩阵倒数第二维 |
| 广播结果维度异常 | 从末尾开始逐维比较 shape |
| DataFrame 数量突然减少 | `dropna`、布尔筛选和 merge 类型 |
| Expected all tensors on same device | 输入、参数、mask 是否同 device |
| Expected Long but found Float | token id 或分类标签 dtype |
| `.grad is None` | `requires_grad`、detach/no_grad、优化器参数列表 |
| loss 不下降 | target 对齐、学习率、清零梯度、模型表达能力 |

## 学习方法

每个模块都包含三层内容：

1. 可复用的小函数，用于明确一个数学或 API 概念。
2. `run_demo()` 中的最小示例和断言。
3. 文件末尾的练习建议，用于主动修改 shape、dtype 或数据。

不要只看输出。运行前先写下预期 shape，运行后再解释每个轴、每次求和和每条梯度路径。
当断言失败时，先定位违反了哪个不变量，而不是直接删除断言。

## 为什么单独学习 Pandas

Pandas 通常不出现在模型 forward 中，但会频繁出现在数据准备、实验对比和错误样本分析中：

- 检查 JSONL 字段是否缺失。
- 比较 train/validation/test 的样本数量和长度分布。
- 按标签、来源或工具名分组统计。
- 将评测结果与原始样本、元数据合并。

训练内循环仍应使用 Tensor，而不是逐行操作 DataFrame。

## 进入 Labs 的达标标准

完成后应能不看答案完成以下操作：

1. 推导 `[B,T,D] @ [D,M] -> [B,T,M]`。
2. 解释广播为何不会自动复制出一个新数组。
3. 用 Pandas 找出空文本并按 split 统计平均词数，再说明它为何不等于 tokenizer token 数。
4. 在 NumPy 和 PyTorch 中把 `[B,T,D]` 拆成 `[B,H,T,Dh]` 再无损合并。
5. 解释 `.detach()`、`torch.no_grad()` 和 `requires_grad=False` 的区别。
6. 独立写出 `zero_grad -> forward -> loss -> backward -> step`。

达到这些标准后，再从 `labs.lab00_positional_encoding` 开始 Transformer 实验。
