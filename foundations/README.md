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

下面三个例子分别描述 shape、具体数值和数值稳定性。它们属于不同层级，不能只靠相似的
方括号记号混在一起记忆。

### 1. 矩阵乘法：这里的数字表示 shape

设矩阵 `A` 的 shape 是 `[2, 3]`，矩阵 `B` 的 shape 是 `[3, 4]`：

```text
shape(A) = [2, 3]    # A 有 2 行、3 列
shape(B) = [3, 4]    # B 有 3 行、4 列
shape(A @ B) = [2, 4]
```

这里的 `[2, 3]` 和 `[3, 4]` 是**矩阵尺寸**，不是矩阵中保存的具体数值。`@` 表示矩阵乘法。
它要求左矩阵的列数等于右矩阵的行数，也就是中间两个维度必须相等：

```text
左矩阵：[2, 3]
右矩阵：[3, 4]
内侧维度：3 == 3    # 决定能否相乘
外侧维度：2 和 4    # 决定输出 shape [2, 4]
```

相等的中间维度在结果中被归约掉，外侧的 `2` 和 `4` 留下来，所以输出 shape 是 `[2, 4]`。
输出矩阵中每个位置的值，都是 `A` 的一行与 `B` 的一列做点积得到的：

$$
(A B)_{i,k}=\sum_{j=1}^{3} A_{i,j}B_{j,k}
$$

### 2. 向量点积：这里的数字表示具体数值

下面的 `a` 和 `b` 都是长度为 2 的向量，方括号中写的是它们真正保存的数值：

```text
a = [1, 2]
b = [3, 4]

dot(a, b) = 1 * 3 + 2 * 4 = 11
```

点积把两个等长向量对应位置相乘后求和，输出是一个标量。Attention 中一个 query 与一个
key 的原始相关性分数，本质上就是这样的点积。

当 query/key 的维度是 `Dh` 时，点积包含 `Dh` 项求和，数值尺度会随维度增大。Scaled
Dot-Product Attention 会再除以 $\sqrt{Dh}$：

$$
\operatorname{score}(q,k)=\frac{q\cdot k}{\sqrt{Dh}}
$$

它不是为了让 shape 能对齐，而是为了控制送入 Softmax 的数值尺度，减少概率过早饱和。

### 3. Softmax：把一组分数转换成概率分布

模型通常先输出一组没有归一化的分数 $z=[z_1,z_2,\ldots,z_n]$，这些分数称为
**logits**。Softmax 对每个分数取指数，再除以所有指数值之和：

$$
\operatorname{softmax}(z_i)
=\frac{e^{z_i}}{\sum_{j=1}^{n}e^{z_j}}
$$

当类别数大于 `1` 且 logits 都是有限值时，Softmax 的每个输出都严格大于 $0$、小于 $1$，
并且所有值之和等于 $1$，因此可以把它解释为一个概率分布。Softmax 还会保持分数的大小顺序：
如果 $z_i>z_k$，那么对应的概率也一定满足 $p_i>p_k$。例如：

```text
logits       = [2, 1, 0]
softmax      = [0.6652, 0.2447, 0.0900]
probability sum = 1.0000
```

直接计算 $e^{z_i}$ 时，大 logits 可能造成浮点数溢出。工程实现会先从所有 logits 中减去
同一个最大值：

```text
z                   = [2, 1, 0]
z - max(z)          = [0, -1, -2]
softmax(z)          = softmax(z - max(z))
```

这样做不会改变概率，因为分子和分母都同时乘上了相同的常数 $e^{-\max(z)}$；但平移后的
最大输入变成了 `0`，所有指数项都不超过 $e^0=1$，从而显著降低数值上溢风险。这就是
**稳定 Softmax（stable Softmax）**。

### 4. 均值与方差：描述中心位置和波动尺度

对 $n$ 个数 $x_1,\ldots,x_n$，总体均值是：

$$
\mu=\frac{1}{n}\sum_{i=1}^{n}x_i
$$

总体方差衡量这些数偏离均值的平均平方距离：

$$
\operatorname{Var}(x)=\frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2
$$

例如 `[1, 2, 3]` 的均值是 `2`，相对均值的偏差是 `[-1, 0, 1]`，方差是：

```text
((-1)^2 + 0^2 + 1^2) / 3 = 2/3 ≈ 0.6667
```

均值描述数据整体位于哪里，方差描述数据分散得多宽。LayerNorm 会按特征轴减去均值并除以
标准差，RMSNorm 则使用均方根控制尺度但不减均值。这里使用分母 `n` 的总体方差；统计学中用于
无偏估计的样本方差可能使用 `n-1`，不要把两个定义混为一谈。

### 5. 交叉熵：正确答案的概率越低，惩罚越大

假设 Softmax 输出三个类别的概率：

```text
probabilities = [0.6652, 0.2447, 0.0900]
target class  = 0
```

单个 one-hot 标签的交叉熵，就是正确类别概率的负对数：

$$
L=-\log p_{\text{target}}
$$

上例中：

```text
loss = -log(0.6652) ≈ 0.4076
```

如果正确类别概率接近 `1`，loss 接近 `0`；如果正确类别概率接近 `0`，负对数会迅速增大。
训练就是不断调整参数，让真实下一个 token 的概率提高。PyTorch 的 `CrossEntropyLoss` 直接接收
logits，并在内部以稳定方式组合 `log_softmax` 和负对数似然，不应先手动调用 Softmax。

### 6. 数值导数：先把“梯度是局部变化率”变得可观察

导数描述输入发生很小变化时，输出变化有多快。中心有限差分用两次函数计算近似标量导数：

$$
f'(x)\approx\frac{f(x+\varepsilon)-f(x-\varepsilon)}{2\varepsilon}
$$

例如 $f(x)=x^2$，在 $x=3$ 附近取很小的 $\varepsilon$，数值结果会接近解析导数
$f'(3)=2\times3=6$。有限差分直观但需要重复执行 forward，并受到步长和浮点误差影响；PyTorch
Autograd 使用计算图和链式法则高效求导。后面的 F05 会进一步解释梯度如何沿图传播。

本节使用 Python list 手写小矩阵运算，是为了暴露求和发生在哪个维度。真正工程计算应使用
NumPy、PyTorch 或底层优化 kernel。

## F01 NumPy：建立张量 shape 直觉

对应模块：`foundations.f01_numpy_basics`

NumPy 是理解 PyTorch Tensor 的低成本入口。数组不仅包含数值，还包含 `shape` 和 `dtype`。
后续模型代码中的大多数错误，并不是公式完全写错，而是把某个轴的含义理解错了。

### 1. 先给每个 shape 符号一个含义

Transformer 文档经常使用下面这些缩写：

| 符号 | 含义 | 示例 |
| --- | --- | --- |
| `B` | batch size，一次处理多少条序列 | `2` |
| `T` | sequence length，每条序列有多少个 token | `3` |
| `D` | model dimension，每个 token 用多少个特征表示 | `4` |
| `H` | attention head 数量 | `2` |
| `Dh` | 每个 head 的特征维度，`Dh = D / H` | `2` |
| `M` | 某次线性投影的输出维度 | `5` |

例如 `hidden.shape == [2, 3, 4]` 表示：有 2 条序列，每条序列有 3 个 token，每个 token
由 4 个浮点数组成。数组一共保存 `2 * 3 * 4 = 24` 个数：

```text
hidden[b, t, d]
       |  |  |
       |  |  第 d 个特征
       |  第 t 个 token
       第 b 条序列
```

shape 只描述这些数如何组织，不代表数组中的具体数值。

### 2. 线性投影只改变最后一个特征轴

设输入 `X` 的 shape 是 `[B, T, D]`，权重 `W` 按 PyTorch 约定保存为 `[M, D]`。实际计算
需要使用 `W.T`，其 shape 是 `[D, M]`：

```text
X       : [B, T, D] = [2, 3, 4]
W       : [M, D]    = [5, 4]
W.T     : [D, M]    = [4, 5]
X @ W.T : [B, T, M] = [2, 3, 5]
```

矩阵乘法归约输入的 `D=4`，保留 batch 轴和 token 轴，再产生新的输出特征轴 `M=5`。
因此 Linear 不会把 2 条序列或 3 个 token 混在一起；它对每个 token 独立使用同一组权重。

### 3. `reshape` 与 `transpose` 做的不是同一件事

多头注意力需要把一个 `D` 维向量拆成 `H` 组，每组宽度是 `Dh = D / H`：

```text
[B, T, D]       = [2, 3, 4]
reshape
[B, T, H, Dh]   = [2, 3, 2, 2]
transpose(1, 2)
[B, H, T, Dh]   = [2, 2, 3, 2]
```

`reshape` 没有创造新数据，而是把原来的 4 个特征重新解释为 `2 heads * 2 features`。
`transpose` 则交换 `T` 和 `H` 两个轴的位置，让同一个 head 的所有 token 排在一起。二者都可能
不复制底层数据，但它们改变的是不同东西：前者改变分组方式，后者改变轴顺序。

合并多头时必须严格执行逆过程：

```text
[B, H, T, Dh]
-> transpose [B, T, H, Dh]
-> reshape   [B, T, D]
```

如果只 `reshape` 而不先把轴换回来，最终 shape 可能仍然正确，但 token 与 head 的数值顺序已经
混乱。这类错误通常不会立刻报错，却会让模型学不到正确结果。

### 4. Broadcasting 是逻辑扩展，不是随意复制

Broadcasting 允许不同 shape 的数组执行逐元素运算。最常见的场景是给 batch 中每个样本添加
同一个 bias，或用同一组 scale 缩放所有 token 的特征。如果没有广播，就需要先把较小数组
`repeat/tile` 成大数组，既啰嗦又浪费内存。

判断能否广播不靠猜测，只需要从右向左对齐 shape。每一对维度必须满足下面一个条件：

1. 两个尺寸相等。
2. 其中一个尺寸是 `1`。
3. 较短 shape 左侧没有对应维度时，把缺失维度视为 `1`。

最终结果在每个轴上取两个尺寸的较大值。只要有一对维度既不相等、也都不是 `1`，就不能广播。

**矩阵加行向量：**

```text
features : [3, 4]
row_bias :    [4]
右对齐后 : [1, 4]
结果      : [3, 4]
```

最右侧 `4 == 4`；缺失的左侧维度按 `1` 处理，所以 bias 可以沿 3 行共享。这正是
`nn.Linear` 给 batch 中每个样本添加同一个 bias 的 shape 关系。

**矩阵加列向量：**

```text
features    : [3, 4]
column_bias : [3, 1]
结果         : [3, 4]
```

左侧 `3 == 3`，右侧 `1` 可以扩展到 `4`。注意 `[3]` 默认按右侧对齐，它表示长度为 3 的
行向量语义，不能直接当作 `[3, 1]` 的列向量；需要显式 `reshape(3, 1)` 或 `[:, None]`。

**高维双向广播：**

```text
A          : [3, 1, 5]
B          :    [4, 1]
B 右对齐后 : [1, 4, 1]
结果        : [3, 4, 5]
```

从右向左检查：`5` 与 `1` 得到 `5`，`1` 与 `4` 得到 `4`，`3` 与补出的 `1` 得到 `3`。
这里不是只有 B 被扩展，A 和 B 分别在不同轴上逻辑扩展。

Transformer 中也会反复使用同一规则。例如长度为 `D=4` 的向量缩放 hidden state 的四个特征：

```text
hidden        : [B, T, D] = [2, 3, 4]
feature_scale : [D]       =       [4]
逻辑对齐       :             [1, 1, 4]
结果           : [B, T, D] = [2, 3, 4]
```

`[4]` 被解释成 `[1, 1, 4]`，表示每条序列、每个 token 都使用同一组四维缩放系数。NumPy
和 PyTorch 不需要先真实复制出 6 份 `feature_scale`；显式的 `broadcast_to/expand` view 可以
用 stride `0` 表达重复读取。随后执行乘法或加法时，**运算结果**通常仍会分配自己的输出内存，
所以“广播不复制小操作数”不等于“整个运算零内存分配”。

广播最危险的地方是它可能让错误 shape 合法运行：

```python
a = np.zeros((3,))    # [3]
b = np.zeros((3, 1))  # [3, 1]
result = a - b
print(result.shape)   # [3, 3]，不是预期的 [3]
```

`a` 被右对齐为 `[1, 3]`，与 `b=[3, 1]` 双向广播成 `[3, 3]`。代码不报错，但可能让 loss
和后续归约悄悄改变含义。调试时应在关键边界写 `assert tensor.shape == expected_shape`，并使用
`reshape`、`squeeze`、`unsqueeze` 或 `None` 索引明确表达行轴、列轴和 batch 轴。NumPy 还可以
用 `np.broadcast_shapes(shape_a, shape_b)` 单独检查预期输出 shape。

### 5. `axis` 决定沿哪个方向归约

对于 logits shape `[B, T, V]`，`softmax(axis=-1)` 表示固定某条序列和某个 token，只沿词表
轴 `V` 归一化。输出 shape 不变，但每个 `[V]` 切片的概率和变为 `1`。如果误写成 token 轴，
代码仍可能运行，却把不同位置的分数归一化到了一起。

运行本节时，不要只看最终 shape。还要逐步回答：每个轴表示什么、哪一个轴被拆分、哪一个轴
被交换、哪一个轴被矩阵乘法或 Softmax 归约。

## F02 Embedding Geometry：观察高维到低维

对应模块：`foundations.f02_embedding_geometry`

这一节把 01 文档中的“token 如何进入连续向量空间”变成可观察实验。先区分三个阶段：

```text
原始文本 --Tokenizer--> token id --Embedding lookup--> token vector
"苹果"                  42                         [D]
```

Tokenizer 负责把字符或子词映射成整数 id；Embedding 不直接阅读文字，它只根据 id 从参数表中
取出一行。模型真正处理的是后面的浮点向量。

### 1. `V` 和 `D` 分别表示什么

设：

- `V` 是 vocabulary size，即词表中 token 的数量。
- `D` 是 embedding dimension，即每个 token 向量包含多少个可训练浮点数。

如果 `V=50_000`、`D=512`，Embedding 参数表的 shape 就是 `[50_000, 512]`。表中有
50,000 行，每个 token id 对应唯一一行，每行有 512 个浮点数。

概念上，可以先把 token id 写成一个长度为 `V` 的 one-hot 向量，再乘 Embedding 表：

```text
one_hot(token_id) : [V]
embedding_table   : [V, D]
结果               : [D]

[V] @ [V, D] -> [D]
```

one-hot 中只有 token id 对应的位置是 `1`，其余位置都是 `0`，所以矩阵乘法最终只会留下参数表
的对应行：

```python
one_hot @ embedding_table == embedding_table[token_id]
```

工程实现直接查表，不会真的构造巨大的 one-hot 向量。one-hot 只是帮助理解“为什么按 id 取出
一行”等价于一次矩阵乘法。

### 2. 512 维不是把所有文字硬塞进 512 个格子

one-hot 的第 42 维可以明确表示“这是 id 42”，但不同 token 的 one-hot 两两正交，无法表达
语义关系。Embedding 的 512 个坐标也通常没有“第 1 维是水果、第 2 维是时态”这种人工定义。
它们是一组由训练共同决定的连续坐标。

Embedding 表开始时通常是小随机数。模型做 next-token prediction 时，loss 的梯度会不断更新
用到的行以及后续网络参数。为了在大量上下文中做出正确预测，模型会逐渐学到一套有用的几何
组织：上下文或功能相似的 token 往往产生更相近、或能被后续层以相似方式使用的表示。

因此，从 `V` 维 one-hot 到 `D` 维向量并不是某个固定压缩公式保证“信息绝不丢失”，而是一种
可训练参数化：

- one-hot 负责无歧义地标识 token 身份。
- `[V, D]` 参数表为每个身份提供一个可学习的起始向量。
- Attention 和 FFN 再根据上下文把静态向量变成动态表示。
- `D=512` 只是容量、计算量和显存之间的架构选择，不是自然语言唯一正确的维度。

如果 `D` 太小，模型难以同时编码任务需要的多种关系；如果 `D` 很大，参数量和矩阵乘法成本会
明显增加。是否“足够”最终要由模型规模、数据、任务和实验结果共同判断。

### 3. 为什么示例还使用共现矩阵、PPMI 和 SVD

本节代码额外实现了一条可观察的统计学习路径，它不是现代 Transformer 训练 Embedding 的实际
流程，而是用来展示“共享上下文可以形成低维几何结构”：

```text
语料中的邻居计数
-> 共现矩阵 [V, V]
-> PPMI：突出比随机情况更常一起出现的 token 对
-> 截断 SVD：只保留最重要的 D 个方向
-> 低维表示 [V, D]
```

共现矩阵第 `i` 行描述 token `i` 周围出现过哪些 token。原始 `[V, V]` 行很宽，截断 SVD 用
一个秩不超过 `D` 的矩阵近似它，把共享模式压缩进 `[V, D]`。代码同时计算重建误差，是为了
明确：降维保留主要结构，但通常是有损近似。

### 4. 静态 Embedding 不等于最终上下文表示

查表得到的只是 token 的初始静态向量。同一个 token id 每次都会查到同一行，但它进入不同
句子后，会通过 Attention 与周围 token 交换信息。例如“bank”在金融语境和河岸语境中起始
向量相同，经过多层 Transformer 后的 hidden state 应当不同。

示例代码用“当前向量加邻居平均值”演示这种变化，只是为了让输入相同、上下文不同、输出不同
这件事可见。真正 Transformer 使用训练得到的 Attention、残差连接和 FFN，不是固定求平均。

## F03 Pandas：检查数据，而不是训练模型

对应模块：`foundations.f03_pandas_basics`

JSONL 是一种常见数据格式：每一行都是一个完整 JSON 对象。DataFrame 则是带列名的二维表格，
每一行通常对应一条训练或评测样本，每一列对应一个字段。

```json
{"text":"predict next token","split":"train","label":"lm"}
{"text":"inspect padding mask","split":"train","label":null}
```

读入 DataFrame 后，不要立刻开始训练。需要掌握的最小检查工作流是：

```text
JSONL records
-> DataFrame
-> required-column validation
-> missing/empty row filtering
-> derived columns
-> groupby summary
-> merge metadata or evaluation results
```

### 1. 缺失值、空字符串和缺失列不是同一种问题

- 缺失列：整个 DataFrame 都没有 `text` 或 `split` 字段，说明数据 schema 不符合约定，应立即报错。
- 缺失值：某行 `text` 是 JSON `null`，Pandas 通常表示为 `NaN`，可用 `dropna` 检查或删除。
- 空字符串：字段存在，但内容是 `""` 或只有空格；需要先 `str.strip()` 再判断。

如果只处理 `NaN` 而忽略空字符串，空样本仍会进入 tokenizer；如果静默接受缺失列，后面的错误
通常会出现在离数据源很远的训练代码中，更难定位。

### 2. 派生列是为了检查数据，不是改变原始语义

这一节使用下面的向量化表达式增加 `word_count`：

```python
frame["word_count"] = frame["text"].str.split().str.len()
```

它对整列执行字符串操作，不需要手写逐行循环。但它只是按空白统计英文词数，不是模型 tokenizer
的 token 数。中文没有天然空格，BPE/SentencePiece 还会把词拆成子词，特殊 token 也有自己的
规则；需要精确 token 长度时，必须调用项目实际 tokenizer。

### 3. `groupby` 回答分组统计问题

假设清洗后的数据包含 `split`、`text`、`word_count`，那么按 `split` 分组可以回答：训练集和
验证集各有多少条样本、平均长度是否差异过大。

```text
split       examples    average_words
train       2           3.0
validation  1           4.0
```

`groupby` 不只是为了生成漂亮表格。它常用于发现数据切分失衡、某个来源异常偏长、某类工具样本
数量过少等训练前问题。

### 4. `merge` 前先明确行之间是什么关系

示例把多条样本与来源元数据按 `source_id` 合并：样本表中一个来源可以出现多次，来源表中每个
`source_id` 应只出现一次，所以关系是 `many_to_one`。

```text
examples: many rows with source_id=1
sources : one metadata row with source_id=1
```

使用 `validate="many_to_one"` 可以在来源表意外出现重复 id 时立即报错。否则 merge 可能把一条
样本复制成多条，悄悄改变数据量。合并后还要检查右表未匹配的行是否产生了空值。

Pandas 适合批量检查和聚合，不适合模型训练内循环。优先使用列运算、布尔筛选和聚合；避免用
Python `for` 或 `DataFrame.iterrows()` 逐行执行本可向量化的运算。

## F04 PyTorch Tensor：从数组进入模型运行时

对应模块：`foundations.f04_pytorch_tensors`

PyTorch Tensor 可以理解为“带运行时状态的多维数组”。除了实际数值，阅读一个 Tensor 时至少
要同时检查四件事：

| 属性 | 它回答的问题 | 常见错误 |
| --- | --- | --- |
| `shape` | 每个轴分别表示什么 | token 轴与 head 轴颠倒 |
| `dtype` | 每个元素用什么类型保存 | Embedding 收到 Float id |
| `device` | 数据位于 CPU、CUDA 还是 MPS | 输入与参数不在同一设备 |
| autograd 状态 | 是否记录反向传播所需操作 | 意外 detach 导致没有梯度 |

### 1. 同样是数字，id、特征和 mask 需要不同 dtype

```text
token_ids : torch.long     # 离散索引，例如 [12, 98, 7]
hidden    : torch.float32  # 可微的连续特征
visible   : torch.bool     # True/False 逻辑条件
```

Embedding 用 token id 做行索引，所以通常要求整数 `torch.long`。Embedding 查表后的 hidden
state 才是浮点 Tensor，可以参与 Linear、Attention 和梯度计算。Mask 表达是否允许某个位置
参与计算，用 `torch.bool` 能让约定更明确，也避免把 `0/1` 数值误当作权重。

### 2. Tensor 和模型参数必须位于同一 device

CPU、CUDA GPU 和 Apple MPS 有各自的内存空间。模型在 CUDA 上时，输入和参与运算的 mask 也
必须移动到 CUDA：

```python
device = torch.device("cuda")
model = model.to(device)
token_ids = token_ids.to(device)
visible = visible.to(device)
```

`.to(device)` 返回位于目标设备的 Tensor；如果没有把返回值保存下来，原变量可能仍留在 CPU。
不要在 `forward` 中反复把大 Tensor 跨设备搬运，这会产生同步和传输开销。

### 3. `nn.Linear` 的存储 shape 与计算方向

`nn.Linear(in_features=D, out_features=M)` 的权重保存为 `[M, D]`，即每个输出特征对应一行
权重。对输入 `X [..., D]` 的等价计算是：

```text
weight       : [M, D]
weight.T     : [D, M]
X @ weight.T : [..., M]
```

PyTorch 的 `nn.Linear` 会自动完成转置语义和 bias 广播。看到源码中的 `weight` 是 `[out, in]`
时，不要误以为输入需要写成 `[out]`。

### 4. Mask 必须同时说明 shape 和布尔语义

Attention scores 常见 shape 是 `[B, H, Tq, Tk]`：最后一维 `Tk` 表示当前 query 可以查看的
key 位置。本节约定 `visible=True` 表示允许关注，`False` 表示屏蔽：

```python
masked_scores = scores.masked_fill(~visible, float("-inf"))
probabilities = torch.softmax(masked_scores, dim=-1)
```

以长度为 3 的 causal mask 为例：

```text
[[True,  False, False],   # token 0 只能看 token 0
 [True,  True,  False],   # token 1 可以看 token 0..1
 [True,  True,  True ]]   # token 2 可以看 token 0..2
```

被屏蔽位置变成 `-inf`，经过 Softmax 后概率为 `0`。Mask 可以依靠 broadcasting 从
`[1, 1, T, T]` 扩展到 `[B, H, T, T]`，但必须确认它在哪些轴上共享。

如果某一整行全部被屏蔽，普通 Softmax 会遇到 `0/0` 并产生 `NaN`。本节的
`masked_softmax` 明确把这种行定义为全零；真实模型也应从数据和 mask 设计上确认全屏蔽行是否
本来就不该出现，不能只把 `NaN` 隐藏掉。

### 5. `transpose` 后为什么常见 `.contiguous()`

物理内存可以看作一段线性 storage。Tensor 的多维结构主要由三类元数据解释：

- `shape`：每个轴有多少个逻辑元素。
- `stride`：沿某个轴前进一步，要在 storage 中跨过多少个元素。
- `storage_offset`：Tensor 的第一个逻辑元素从 storage 的哪个位置开始。

PyTorch 的 stride 以“元素个数”为单位；NumPy 的 `.strides` 通常以“字节数”为单位，这一点在
对照两套输出时不要混淆。

设一个连续的 `[3, 4]` PyTorch Tensor 保存 12 个元素：

```text
逻辑 shape : [3, 4]
物理 storage: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
stride      : [4, 1]
```

忽略 `storage_offset` 时，`A[i, j]` 的位置是：

$$
\operatorname{offset}(i,j)=i\times4+j\times1
$$

例如 `A[2, 1]` 的 offset 是 `2*4 + 1 = 9`，对应 storage 中从 0 开始编号的第 9 个位置，也就是
数值 `10`。

**Transpose：只交换元数据即可形成 view。**

```text
A             shape=[3, 4] stride=[4, 1]
A.transpose   shape=[4, 3] stride=[1, 4]
```

原始数据没有移动。转置后沿新行轴移动一步只跨 1 个元素，沿新列轴移动一步跨 4 个元素。由于
逻辑相邻元素不再按常规行优先顺序连续排列，这个 view 通常是 non-contiguous。

**Slice：修改 offset、shape 或 stride。**

`A[:, ::2]` 每隔一列取一个值，可以表示为 shape `[3, 2]`、stride `[4, 2]`。它仍引用原 storage，
不需要把选中的 6 个元素立即复制到新数组。

**Broadcast/expand：用 stride `0` 重复读取同一位置。**

将 `[4]` 的向量显式 `expand` 为 `[3, 4]` 时，可以得到 stride `[0, 1]`：

$$
\operatorname{offset}(i,j)=i\times0+j\times1=j
$$

无论行索引 `i` 是多少，同一列都读取原向量的同一个元素。这解释了为什么扩展 view 不需要保存
三份数据。也正因为多个逻辑位置指向同一物理位置，不应对 expanded view 做会让同一位置收到
多个不同写入值的原地修改；NumPy 的 `broadcast_to` 通常直接返回只读 view，PyTorch 对许多
重叠内存写入也会拒绝或要求先 `clone()`。

`view` 要求目标 shape 能按照当前 stride 关系直接解释。它并非简单地“所有 non-contiguous
Tensor 一律不可用”，但转置后直接合并不相邻轴通常无法满足这个条件。多头合并的明确写法是：


```python
heads.transpose(1, 2).contiguous().view(B, T, H * Dh)
```

`.contiguous()` 会按**当前逻辑顺序**分配新 storage 并复制数值。一个 `[4, 3]` 的连续结果 stride
应是 `[3, 1]`，不再是转置 view 的 `[1, 4]`。`reshape` 会优先返回 view，无法满足时可能自动
复制，因此写起来方便，但也更难仅凭代码判断是否发生了内存分配。

连续布局通常更有利于 CPU cache line 和 GPU 合并访存，但复制本身也有成本。不要习惯性在每次
transpose 后调用 `.contiguous()`；只有后续算子要求、需要合并特定轴，或性能分析证明有收益时
再转换布局。

### 6. 从 NumPy 创建 Tensor 时要知道是否共享内存

`torch.from_numpy(array)` 通常与 CPU NumPy 数组共享同一块内存，修改一方可能影响另一方；
`torch.tensor(array)` 通常创建副本。共享内存能避免复制，但如果调用方不知道别处会原地修改，
也会造成隐蔽的数据污染。

## F05 Autograd：沿计算图追踪梯度

对应模块：`foundations.f05_pytorch_autograd`

反向传播不是“自动猜参数怎么改”，而是先记录 forward 中发生的可微运算，再从标量 loss 出发，
按链式法则计算 loss 对各个叶子 Tensor 的局部变化率。

### 1. 最小计算图：从一个标量函数开始

```text
x = 3
y = x^2 + 2x + 1 = 16
dy/dx = 2x + 2 = 8
```

对应 PyTorch：

```python
x = torch.tensor(3.0, requires_grad=True)
y = x.square() + 2 * x + 1
y.backward()
print(x.grad)  # tensor(8.)
```

`requires_grad=True` 表示从 `x` 出发的可微操作需要被记录。`y.backward()` 从输出沿计算图反向
应用链式法则，最终把 $\partial y/\partial x$ 累积到叶子 Tensor 的 `x.grad`。

### 2. `.grad` 默认是累积，不是覆盖

连续两次调用 `backward()` 时，新梯度会加到已有 `.grad` 上。这对把多个 loss 的贡献相加很有
用，但训练时通常希望每个 batch 独立计算梯度，所以必须在下一次 backward 前调用：

```python
optimizer.zero_grad()
```

如果忘记清零，优化器使用的将是当前 batch 与之前 batch 的梯度之和，训练行为就不再是代码表面
表达的普通 mini-batch 梯度下降。

### 3. 为什么会出现 VJP，而不是总构造完整 Jacobian

当函数把向量 $x\in\mathbb{R}^{n}$ 映射为向量 $y\in\mathbb{R}^{m}$ 时，所有偏导数组成
Jacobian $J\in\mathbb{R}^{m\times n}$。真实网络的 `m`、`n` 极大，显式构造完整 Jacobian
既慢又占内存。

反向传播接收上游梯度向量 $v$，直接计算 vector-Jacobian product（VJP）$v^T J$。标量 loss
可以看作 `m=1` 的特殊情况，上游梯度就是 `1`。这也是非标量输出调用 `backward` 时需要提供
`gradient`，或使用 `torch.autograd.grad(..., grad_outputs=v)` 的原因。

### 4. 残差连接为什么提供一条直接梯度路径

对逐元素函数：

$$
y=x+x^2
$$

其导数是：

$$
\frac{dy}{dx}=1+2x
$$

其中 `2x` 来自非线性分支，`1` 来自 `y=x+...` 的恒等残差路径。即使非线性分支的局部导数很
小，梯度仍有一项可以直接传回输入。后续理解深层 Transformer 的残差连接和 Pre-LN 时会反复
用到这个结构。

### 5. 三种“不要为这里求梯度”不是同一件事

| 操作 | 作用范围 | 典型用途 |
| --- | --- | --- |
| `tensor.detach()` | 返回与此前计算图断开的 Tensor | 截断某条特定数据路径 |
| `with torch.no_grad()` | 代码块内不记录新的反向图 | 推理、评估、手动更新参数 |
| `parameter.requires_grad_(False)` | 不为指定参数保存梯度 | 冻结 backbone、只训练 LoRA |

冻结 Linear 权重并不等于阻断整个计算。若输入仍需要梯度，Autograd 仍可使用冻结权重计算 loss
对输入的梯度，只是 `weight.grad` 保持为 `None`。而在中间激活上调用 `.detach()` 会切断该激活
之前的梯度路径，影响更上游的参数。

## F06 PyTorch Training：闭合最小训练循环

对应模块：`foundations.f06_pytorch_training`

训练循环把前面所有概念闭合起来：DataLoader 提供 batch，模型产生 logits，loss 衡量预测误差，
Autograd 计算梯度，optimizer 再用梯度修改参数。

### 1. 先看清一批数据的 shape 合同

本节使用二维点的二分类任务。设 batch size 是 32：

```text
features : [B, 2] = [32, 2]    float32
labels   : [B]    = [32]       long，取值只能是 0 或 1
logits   : [B, 2] = [32, 2]    每个样本对两个类别的未归一化分数
loss     : []                   标量
```

模型 `nn.Linear(2, 2)` 对每个二维样本输出两个 logits。标签不是 one-hot `[32, 2]`，而是类别
索引 `[32]`。这正是 `nn.CrossEntropyLoss` 的常用输入约定。

### 2. 标准训练顺序中每一步都改变了状态

```python
optimizer.zero_grad()
logits = model(features)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

| 步骤 | 发生的事情 | 忘记或写错的后果 |
| --- | --- | --- |
| `zero_grad()` | 清除参数上一个 batch 累积的梯度 | 多个 batch 的梯度意外相加 |
| `model(features)` | 执行 forward，构建本次计算图 | shape/device 错误通常在这里暴露 |
| `criterion(logits, labels)` | 把预测与目标压缩成标量 loss | 标签错位会训练错误目标 |
| `loss.backward()` | 计算 loss 对可训练参数的梯度 | 只算梯度，尚未修改参数 |
| `optimizer.step()` | 按 SGD/AdamW 规则更新参数 | 没有它，loss 不会因训练而改善 |

### 3. `CrossEntropyLoss` 要接收 logits

分类交叉熵在数值稳定的实现中已经组合了 `log_softmax` 和 negative log-likelihood。调用前不要
再手动做 Softmax：

```python
# 正确
loss = criterion(logits, labels)

# 不要这样做
loss = criterion(torch.softmax(logits, dim=-1), labels)
```

手动传入概率会改变损失函数的数学含义，并失去内部稳定的 log-sum-exp 计算。对于 decoder-only
语言模型，类别数从这里的 `2` 变成词表大小 `V`，shape 通常是 logits `[B*T, V]`、labels
`[B*T]`，但合同完全相同。

### 4. Batch、epoch 和平均 loss

- sample：一条数据。
- batch：一次 forward/backward 使用的一组样本。
- epoch：训练集中的样本大致都被访问一次。

最后一个 batch 可能小于配置的 batch size，因此计算 epoch 平均 loss 时，本节先累加
`loss.item() * 当前批样本数`，最后再除以总样本数。直接平均每个 batch 的 loss 会让较小的最后
一批获得与完整 batch 相同的权重。

### 5. `train/eval` 与 `no_grad` 解决不同问题

```python
model.train()              # Dropout/BatchNorm 使用训练行为
model.eval()               # Dropout/BatchNorm 使用评估行为
with torch.no_grad():      # 不构建反向图，减少评估内存
    logits = model(inputs)
```

`model.eval()` 不会自动关闭梯度，`torch.no_grad()` 也不会自动切换 Dropout 等模块状态。可靠的评估
通常需要两者同时使用。

本节使用线性可分的二维数据集，并在同一数据集上报告 accuracy。它的价值是验证训练闭环，不代表
真实语言模型任务，也不能演示泛化能力。真实实验必须保留未参与参数更新的 validation/test 数据，
同时检查 loss、任务指标、过拟合和数据泄漏。

## 常见错误速查

| 现象 | 优先检查 |
| --- | --- |
| `matmul` shape error | 左矩阵最后一维与右矩阵倒数第二维 |
| 广播结果维度异常 | 从末尾开始逐维比较 shape |
| Softmax 概率沿错误方向求和 | `dim/axis` 是否指向类别或 key 轴 |
| DataFrame 数量突然减少 | `dropna`、布尔筛选和 merge 类型 |
| DataFrame merge 后数量变多 | join key 是否重复，是否设置关系校验 |
| Expected all tensors on same device | 输入、参数、mask 是否同 device |
| Expected Long but found Float | token id 或分类标签 dtype |
| `view` 提示 stride 不兼容 | `transpose` 后是否需要 `contiguous` 或 `reshape` |
| Attention 出现 `NaN` | 是否存在全屏蔽行、极端 logits 或错误 mask 语义 |
| `.grad is None` | `requires_grad`、detach/no_grad、优化器参数列表 |
| loss 不下降 | target 对齐、是否传入原始 logits、学习率、清零梯度、模型表达能力 |

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
2. 解释广播为什么不需要先把较小操作数物理复制到目标 shape，同时说明运算结果仍可能分配新数组。
3. 解释文本、token id、one-hot、Embedding 表和 contextual hidden state 之间的关系。
4. 用 Pandas 找出空文本并按 split 统计平均词数，再说明它为何不等于 tokenizer token 数。
5. 在 NumPy 和 PyTorch 中把 `[B,T,D]` 拆成 `[B,H,T,Dh]` 再无损合并。
6. 给定 causal mask，明确 `True/False` 的语义，并说明 Softmax 沿哪个轴执行。
7. 解释 `.detach()`、`torch.no_grad()` 和 `requires_grad=False` 的区别。
8. 独立写出 `zero_grad -> forward -> loss -> backward -> step`，并说出每一步改变的状态。

达到这些标准后，再从 `labs.lab00_positional_encoding` 开始 Transformer 实验。
