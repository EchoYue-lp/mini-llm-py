# 00 学习路线、代码地图与工程数学基础

根目录 `README.md` 负责环境、命令和完整项目说明。本目录只解释模型原理与实验。

## 推荐顺序

| 顺序 | 文档 | 对应实验 |
| --- | --- | --- |
| F00-F06 | 数学、NumPy、Embedding、Pandas 与 PyTorch 前置课程 | `foundations/` |
| 00 | 学习路线、代码地图与工程数学基础 | 全部 |
| 01 | Tokenizer、Embedding 与 Logits | 训练脚本 |
| 02 | 位置编码与 RoPE | `lab00` |
| 03 | Scaled Attention 与 Mask | `lab01` |
| 04 | Multi-Head Attention | `lab02` |
| 05 | FFN、残差与 Pre-LN Block | `lab03` |
| 06 | Encoder-Decoder 与翻译训练 | `lab04` |
| 07 | Decoder-Only、Loss 与生成 | `lab05` |
| 08 | KV Cache、MHA、MQA 与 GQA | `lab06`、`lab10` |
| 09 | RMSNorm、RoPE 与 SwiGLU | `lab07` |
| 10 | MoE Router、Capacity 与专家 | `lab08`、`lab11` |
| 11 | LoRA 低秩适配原理 | `lab09` |
| 12 | LoRA 训练、Checkpoint 与过拟合 | MLX 训练代码 |
| 13 | 工具路由数据与评测 | 数据与评测代码 |
| 14 | MLX LoRA 完整实验 | 完整后训练流程 |

## 前置课程命令

进入 Transformer Labs 前，先完成：

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

详细说明见 `foundations/README.md`。

## Labs 实验命令

```bash
python -m labs.lab00_positional_encoding
python -m labs.lab01_attention_basics
python -m labs.lab02_multi_head_attention
python -m labs.lab03_pre_ln_block
python -m labs.lab04_tiny_copy_task --steps 400
python -m labs.lab05_tiny_language_model --steps 100
python -m labs.lab06_kv_cache
python -m labs.lab07_modern_blocks
python -m labs.lab08_moe_routing
python -m labs.lab09_lora_linear
python -m labs.lab10_mha_mqa_gqa
python -m labs.lab11_moe_variants
```

不要只运行命令看 shape；每个实验的观察量、断言与建议修改项见 `labs/README.md`。

## 源码阅读顺序

```text
models/layers.py
  -> models/decoder_encoder_layer.py
  -> models/transformer_models.py
  -> utils/mask_utils.py
  -> scripts/train_decoder.py
  -> scripts/train_encoder_decoder.py
  -> utils/generation_utils.py
  -> utils/translation_utils.py
  -> finetuning/train_lora_short.py
  -> evaluation/tool_router.py
```

## 本节的定位

这里的“最低数学”不是只记住几个 shape 和 API，而是恢复一套能直接解释代码的推理工具。
读完本节，应能完成下面五件事：

1. 从 shape 和索引公式判断一次张量运算到底混合了哪些维度。
2. 解释 Q/K/V、Attention score 和 LoRA 低秩分支各自在做什么线性映射。
3. 推导 `sqrt(Dh)` 缩放和稳定 Softmax 的数值原因。
4. 从自回归最大似然推导 next-token shift 与交叉熵代码。
5. 用链式法则解释残差连接、Pre-LN 和参数冻结时的梯度路径。

后续专题会继续展开这些概念。本节负责建立共同语言，使数学公式、PyTorch shape、训练
现象和调试动作能够互相对应。

| 工程问题 | 本节数学工具 | 后续专题 |
| --- | --- | --- |
| Linear、Q/K/V 和 LoRA 的 shape | 索引求和、线性映射、矩阵秩 | 01、04、11 |
| Attention 为什么缩放 | 点积的均值与方差 | 03、04 |
| Mask、NaN 和混合精度 | 稳定 Softmax、logit 平移不变性 | 03 |
| Loss 与 target 为什么错一位 | 条件概率、最大似然、交叉熵 | 06、07 |
| 深层 Block 为什么能训练 | 链式法则、Jacobian、残差路径 | 05、09 |

## 阅读 Transformer 代码的四条线

每读一个函数，同时追踪四条线：

| 线索 | 要问的问题 | 常见错误 |
| --- | --- | --- |
| 数据线 | 输入和输出各轴是什么语义 | 只看总元素数，不看轴顺序 |
| 可见性线 | 当前 query 能读取哪些 key | causal/padding mask 语义写反 |
| 参数线 | 哪些映射有参数，哪些参数可训练 | 冻结遗漏、LoRA 注入错层 |
| 目标线 | 每个 logit 对应哪个 target token | shift 两次、没有 shift、PAD 进 loss |

其中 shape 只是起点。两个张量 shape 相同，不代表语义相同：

```text
Q:      [B, H, T, Dh]
K:      [B, H, T, Dh]
V:      [B, H, T, Dh]
hidden: [B, T, H, Dh]
```

它们可能拥有相同元素数量，但轴顺序和使用方式不同。每次 `view`、`reshape`、
`transpose`、`permute` 之后，都应重新写出轴名。

### 先写轴名，再写数字

不要只写：

```text
[2, 4, 8, 16]
```

应写成：

```text
[B=2, H=4, T=8, Dh=16]
```

这样才能判断：

- `transpose(-2, -1)` 是否真的在交换 `T` 与 `Dh`。
- Softmax 是否沿 key 轴 `Tk` 归一化。
- Padding mask 是否屏蔽 key，而不是错误地屏蔽 head。
- 合并多头后是否恢复到 `[B,T,D]`，其中 `D = H * Dh`。

### 用索引公式确认“谁和谁相乘”

矩阵写法很紧凑，但索引写法更适合调试。单头 Attention score 的一个元素是：

$$
S_{b,i,j} = \frac{1}{\sqrt{D_h}}
\sum_{d=1}^{D_h} Q_{b,i,d} K_{b,j,d}
$$

这个式子明确说明：

- `i` 是 query 位置。
- `j` 是 key 位置。
- `d` 被求和消失。
- 结果保留 query-key 二维关系，所以 score shape 是 `[B,Tq,Tk]`。

看到复杂 `einsum` 或 `matmul` 时，先把其中一个输出元素写成求和，通常比盯着整条表达式
更快找到错位。

## 工程数学一：线性映射、特征混合与低秩更新

### PyTorch Linear 的真实 shape 约定

为了便于手算，数学里常把一个 token 写成行向量：

```text
x: [D]
W: [D, M]
x @ W: [M]
```

但 `nn.Linear(D, M)` 内部保存的是：

```text
linear.weight: [M, D]
linear.bias:   [M]
```

它实际计算：

```python
y = x @ linear.weight.T + linear.bias
```

对批量序列：

```text
x: [B, T, D]
W: [M, D]
y: [B, T, M]
```

逐元素写成：

$$
y_{b,t,m} = b_m + \sum_{d=1}^{D} x_{b,t,d} W_{m,d}
$$

求和发生在输入特征轴 `D`，`B` 和 `T` 都只是批量轴，线性层不会在不同 token 之间
交换信息。Attention 负责 token 间混合，普通 FFN Linear 负责单个 token 内的特征混合。

严格说，带 bias 的 `nn.Linear` 是仿射映射，不是纯线性映射。工程文档通常仍简称为
Linear，但推导齐次性时要记得 bias 的存在。

### Q/K/V 是学习出来的表示，不是固定字段

Self-Attention 中：

```text
Q = X Wq
K = X Wk
V = X Wv
```

三组参数让同一个 hidden state 在三种用途下被重新表示：

- Q/K 的点积决定匹配分数。
- Softmax 将匹配分数变成混合权重。
- V 提供真正被汇总的内容。

把 `Wq` 称为“投影”是一种工程口语。它不一定是线性代数里满足 `P^2=P` 的投影矩阵；
矩形矩阵也不一定是“基底变换”，因为严格的基底变换通常要求方阵且可逆。更准确的说法
是：它是一个学习到的线性映射，把输入特征变成适合当前计算的表示。

### 多头不等于正交子空间

`D` 个投影通道被 reshape 成 `H` 个 head：

```text
[B,T,D] -> [B,T,H,Dh] -> [B,H,T,Dh]
```

不同 head 有机会学习不同关系，但标准 MHA 没有约束各 head 的参数或输出必须正交。多头的
直接作用是让模型并行维护多组 Q/K/V 匹配与 Value 混合；“每个头一定对应一种独立语义”
只是可能出现的经验现象，不是架构保证。

### 矩阵秩描述映射自由度

对矩阵 `W`，秩 `rank(W)` 可以理解为它能产生多少个线性独立的输出方向。若：

```text
A: [D, r]
B: [r, M]
Delta = A @ B: [D, M]
```

则：

```text
rank(Delta) <= r
```

因为所有更新都必须经过中间的 `r` 维空间。这正是 LoRA 的约束：冻结原权重 `W`，只学习
低秩更新 `Delta-W`，而不是把原权重实时分解成 A/B。

本项目 PyTorch Lab 的约定是：

```text
x: [*, in]
A: [in, r]
B: [r, out]
x @ A @ B: [*, out]
```

由于 `nn.Linear.weight` 保存为 `[out,in]`，融合时需要转置：

```python
delta_weight = scale * (A @ B).T
fused.weight = base.weight + delta_weight
```

`B` 初始化为零时，初始 `A @ B = 0`，所以注入 LoRA 不改变基座输出。这里能严格推出的
是“更新矩阵的秩不超过 `r`”；下游任务只需要低维适配是 LoRA 的有效假设和经验依据，
不是对所有任务都成立的数学定理。

### 线性映射的调试不变量

遇到线性层报错或训练异常，依次检查：

1. 输入最后一维是否等于 `in_features`。
2. 代码使用的是数学约定 `[in,out]`，还是 PyTorch 权重约定 `[out,in]`。
3. `reshape` 是否只重排元素，没有意外改变 token/head 语义。
4. 残差相加前两边是否完全同 shape。
5. LoRA A/B 的乘法顺序和 fuse 时的转置是否一致。

## 工程数学二：点积、方差与 Scaled Attention

### 点积在比较什么

单个 query 与 key 的 score：

$$
s = q \cdot k = \sum_{d=1}^{D_h} q_d k_d
$$

它同时受方向和向量长度影响，不是纯余弦相似度。Q/K 投影、归一化方式和训练过程共同
决定 score 的尺度与语义。

矩阵形式：

```text
Q:   [B,H,Tq,Dh]
K^T: [B,H,Dh,Tk]
S:   [B,H,Tq,Tk]
```

`S[b,h,i,j]` 表示第 `b` 个样本、第 `h` 个 head 中，query 位置 `i` 对 key 位置 `j`
的匹配分数。

### 为什么除以 `sqrt(Dh)`

用一个简化初始化模型推导。假设每个 `q_d`、`k_d`：

- 相互独立。
- 均值为 0。
- 方差为 1。

则每一项 `q_d k_d` 的均值为 0、方差为 1。独立项相加后：

$$
\operatorname{Var}\left(\sum_{d=1}^{D_h} q_d k_d\right) = D_h
$$

所以点积的标准差约为 `sqrt(Dh)`。缩放后：

$$
\operatorname{Var}\left(\frac{q\cdot k}{\sqrt{D_h}}\right) \approx 1
$$

这让不同 head dimension 下的 score 初始尺度更可控，Softmax 不会仅仅因为 `Dh` 变大
就变得过尖。

这个推导依赖独立、零均值、单位方差等近似条件。训练后 Q/K 不再满足这些理想假设，
`sqrt(Dh)` 仍作为稳定的尺度基准，而不是精确保证 score 方差永远等于 1。

### 为什么过尖的 Attention 会难优化

Softmax 的 Jacobian 为：

$$
\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)
$$

当注意力权重接近 one-hot 时，大多数 `p_i` 接近 0 或 1，很多导数会很小，模型较难连续
调整“应该把权重从哪个 key 挪到哪个 key”。这里说的是 Attention 内部 Softmax 的
梯度路径。

不要把它和输出层 `softmax + cross entropy` 完全混为一谈。对输出 logits，二者合并后
有简洁梯度 `p - y`；模型若非常自信但预测错误，正确类别仍可收到很大的纠正信号。

## 工程数学三：稳定 Softmax、Mask 与温度

### Softmax 只关心相对差值

$$
\operatorname{softmax}(z_i) =
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

对所有 logit 同时加常数 `c`，结果不变：

$$
\frac{e^{z_i+c}}{\sum_j e^{z_j+c}}
= \frac{e^c e^{z_i}}{e^c \sum_j e^{z_j}}
= \operatorname{softmax}(z_i)
$$

因此可以选择 `c = -max(z)`：

$$
\operatorname{softmax}(z_i) =
\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}, \quad m=\max_j z_j
$$

此时最大的指数输入为 0，其余都小于等于 0，避免直接计算巨大正指数。PyTorch 的
`torch.softmax` 和 `F.cross_entropy` 已在底层采用稳定实现，不要为了“手动稳定”先把
概率算出来再传给交叉熵。

### dtype 会改变风险边界

在 `float16` 中，直接计算 `exp(z)` 时，`z` 稍大于 11 就可能上溢。`bfloat16` 与
`float32` 有更大的指数范围，不能简单套用同一阈值，但仍有舍入、下溢和极端 score
问题。减最大值是跨 dtype 的基本做法，混合精度和 fused kernel 还可能在内部使用更高
精度完成归约。

### Mask 为什么必须在 Softmax 前

对禁止位置加入 `-inf`：

```python
scores = scores.masked_fill(~mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
```

数学上 `exp(-inf)=0`，禁止位置自然得到零概率，允许位置仍归一化为 1。若先 Softmax 再
把某些概率改成零，剩余概率之和不再是 1，除非额外归一化。

Softmax 的归一化轴必须是 key 轴：

```text
scores: [B,H,Tq,Tk]
softmax(dim=-1)  # 对每个 query 的所有 key 归一化
```

### Fully masked row 为什么产生 NaN

若一整行都被屏蔽：

```text
[-inf, -inf, -inf]
```

稳定实现需要计算 `z - max(z)`，但这里出现 `-inf - (-inf)`，结果未定义；从概率角度看，
“没有任何允许事件的分布”本来也无法归一化。因此正确策略优先是避免为有效 query 构造
全屏蔽行，或显式规定这种行的输出语义，而不是只在末尾静默吞掉 NaN。

本项目 `models/layers.py` 会把 Attention 中出现的 NaN 权重置零，作为全 padding 情况的
防护；调试时仍应回查 mask 是否符合预期。

### Temperature 是 logit 尺度控制

$$
p_i(\tau) = \operatorname{softmax}(z_i / \tau)
$$

- `tau < 1`：放大 logit 差异，分布更尖。
- `tau = 1`：保持原分布。
- `tau > 1`：缩小 logit 差异，分布更平。

Attention 的 `1/sqrt(Dh)` 和生成时 temperature 都在控制 Softmax 输入尺度，但目的不同：
前者稳定网络内部计算，后者在推理时调整采样分布。

## 工程数学四：Log、最大似然与 Next-Token Loss

### 为什么概率乘积要变成对数和

自回归模型把一个序列的联合概率分解为：

$$
P(x_1,\ldots,x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

最大化整个训练集上的概率乘积数值很不稳定，而且不方便求导。取对数后：

$$
\log P(x_1,\ldots,x_T)
= \sum_{t=1}^{T} \log P(x_t \mid x_{<t})
$$

训练最小化负对数似然：

$$
\mathcal{L}_{NLL}
= -\sum_t \log P(x_t \mid x_{<t})
$$

Log 把连乘变成求和，也让很小的概率不必直接以接近零的浮点数相乘。

### 交叉熵直接作用于 logits

某位置 logits 为 `z`，正确类别为 `y`。单 token 交叉熵可写成：

$$
\mathcal{L}(z,y)
= \log\sum_j e^{z_j} - z_y
$$

这就是稳定的 `logsumexp - correct_logit` 形式。PyTorch 用法：

```python
loss = F.cross_entropy(logits, labels)
```

不要先做：

```python
probabilities = torch.softmax(logits, dim=-1)
loss = F.cross_entropy(probabilities, labels)  # 错误 API 语义
```

`F.cross_entropy` 期望未归一化 logits，并在内部组合稳定的 `log_softmax + NLL`。

### 与 KL 散度的关系

真实分布 `q` 与模型分布 `p` 的交叉熵满足：

$$
H(q,p) = H(q) + D_{KL}(q\|p)
$$

监督标签固定时，`H(q)` 与模型参数无关，所以最小化交叉熵等价于最小化
`D_KL(q||p)`。普通分类中 `q` 常是 one-hot；使用 label smoothing 或软标签时，`q`
不再是严格 one-hot，但上面的分解仍成立。

### Shift 的本质是对齐条件与目标

序列：

```text
[BOS, x1, x2, x3, EOS]
```

应构造成：

```text
model input: [BOS, x1, x2, x3]
target:      [x1,  x2, x3, EOS]
```

位置 `t` 的 hidden state 只能读取 `<=t` 的输入，却要预测右移一位的 target。这就是
next-token learning，不是“拿当前位置答案预测当前位置答案”。

工程中有两种等价布局：

```python
# 布局 A：先对完整序列前向，再切 logits/labels
shift_logits = logits[:, :-1, :]
shift_labels = token_ids[:, 1:]

# 布局 B：数据层已产生错位后的 inputs/labels
inputs = token_ids[:, :-1]
labels = token_ids[:, 1:]
logits = model(inputs)
```

两种只能选一种。若数据层已经 shift，训练层再次 `[:-1]` / `[1:]`，就会错两格。

本项目 Decoder-Only 使用布局 B：`utils/mask_utils.py::collate_fn_lm` 先返回错位后的 `x/y`，
`scripts/train_decoder.py` 直接计算两者的交叉熵。Encoder-Decoder 训练则在
`scripts/train_encoder_decoder.py` 中显式使用 `tgt[:, :-1]` 和 `tgt[:, 1:]`。

### Flatten 只合并样本，不应打乱对应关系

模型输出与标签：

```text
logits: [B,T,V]
labels: [B,T]
```

交叉熵常改成：

```python
loss = F.cross_entropy(
    logits.reshape(B * T, V),
    labels.reshape(B * T),
    ignore_index=pad_token_id,
)
```

这里把 `B` 和 `T` 合并成 token 样本轴，词表轴 `V` 必须保留。`reshape(-1, V)` 若用了
错误的 `V`，可能不会立刻报出直观错误，因此词表大小应来自当前 tokenizer/model config。

### PAD mask 与 loss ignore 解决不同问题

- Attention mask：阻止有效 token 读取 PAD key。
- `ignore_index`：阻止 PAD target 贡献 loss。

只做前者，PAD 位置仍会训练输出层；只做后者，有效 token 仍可能把 PAD 当作上下文。两者
不能互相替代。

## 工程数学五：计算图、梯度与 Pre-LN

### 梯度和 Jacobian 不完全是一回事

若 loss 是标量、参数 `theta` 是向量，则：

$$
\nabla_{\theta} L
$$

是与参数同 shape 的梯度。若一个向量函数 `y=f(x)` 对向量输入求导，得到的是 Jacobian：

$$
J_f[i,j] = \frac{\partial y_i}{\partial x_j}
$$

反向传播使用链式法则计算 vector-Jacobian product，通常不会显式构造巨大的完整 Jacobian。
这也是 autograd 能处理大型网络的关键。

### 链式法则就是反向传播

若：

```text
x -> h=f(x) -> y=g(h) -> loss
```

则：

$$
\frac{\partial L}{\partial x}
= \frac{\partial L}{\partial y}
  \frac{\partial y}{\partial h}
  \frac{\partial h}{\partial x}
$$

PyTorch 训练循环：

```python
optimizer.zero_grad()
logits = model(inputs)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

- Forward 构建计算图并保存反向需要的中间量。
- `backward()` 沿图累积叶子参数的 `.grad`。
- `step()` 由优化器根据 `.grad` 更新参数。
- 默认梯度会累积，所以常规训练必须在合适时机清零。

### 冻结参数不等于切断整条计算图

若 LoRA 中基座 `W.requires_grad=False`：

- 不会计算和保存 `W.grad`。
- `W` 仍参与 forward。
- 梯度仍可通过 `x @ W` 传回输入 `x` 和更早层。
- A/B 若 `requires_grad=True`，仍会收到梯度。

真正切断历史通常来自 `.detach()`、在 `torch.no_grad()` 中执行，或把值转成与计算图无关
的新张量。调试“为什么没梯度”时，应区分参数冻结与图被截断。

### 残差连接提供显式恒等项

一层残差：

$$
x_{l+1} = x_l + F(x_l)
$$

对输入的 Jacobian：

$$
\frac{\partial x_{l+1}}{\partial x_l}
= I + J_F(x_l)
$$

`I` 表示梯度有一条不经过子层内部变换的路径。这能显著改善深层优化，但不能表述为梯度
“绝对无损”：多层网络的总 Jacobian 仍包含许多矩阵乘积，优化还受初始化、归一化、
激活、精度和 loss 尺度影响。

### Pre-LN 与 Post-LN 的梯度路径差异

Post-LN：

$$
x_{l+1} = N(x_l + F(x_l))
$$

其局部 Jacobian：

$$
J_{post} = J_N \left(I + J_F\right)
$$

主残差路径也必须经过 Norm 的 Jacobian。

Pre-LN：

$$
x_{l+1} = x_l + F(N(x_l))
$$

其局部 Jacobian：

$$
J_{pre} = I + J_F J_N
$$

这里有一个显式的恒等分支绕过 Norm 和子层，所以深层训练通常更稳定。这也是本项目
`DecoderLayer` / `EncoderLayer` 先 Norm、再执行子层、最后残差相加的原因。

Pre-LN 不是“任何模型都必须使用”的定理，也不保证完全不需要 warmup、梯度裁剪或其他
稳定策略。它改变了优化性质和残差流；具体模型仍可能使用 Post-LN、Sandwich Norm、
DeepNorm 或其他变体。

### 为什么 Pre-LN stack 末尾还需要 final norm

Pre-LN 每个子层归一化的是送入 `F` 的分支，残差更新后的主干不会自动成为归一化输出：

```text
x -> x + F(Norm(x))
```

堆叠结束后，通常再做一次：

```text
hidden = final_norm(hidden)
logits = output_projection(hidden)
```

本项目 `models/transformer_models.py` 的 Decoder-Only、Encoder 和 Decoder 末端都显式
包含 final `nn.LayerNorm`。

### 梯度调试清单

训练不动、出现 NaN 或参数没有更新时，按顺序检查：

1. `loss.requires_grad` 是否为 `True`，`loss.grad_fn` 是否存在。
2. 目标参数 `requires_grad` 是否符合预期。
3. `backward()` 后关键参数 `.grad` 是 `None`、全零、有限值还是 NaN/Inf。
4. 是否在 forward 中误用了 `.detach()`、`.item()`、`no_grad()` 或破坏图的 inplace 操作。
5. 优化器是否真的包含目标参数，`step()` 是否执行。
6. 梯度累积时 loss 缩放、清零和 scheduler step 的频率是否正确。
7. Mixed precision 下是否正确使用 loss scaling，并在裁剪前 unscale。

## 常见说法的严格版本

| 常见简化说法 | 更准确的理解 |
| --- | --- |
| `Wq` 是基底变换 | 它是学习到的线性/仿射映射；只有满足额外条件时才是严格基底变换 |
| 多头学习正交子空间 | 多头提供多组表示与混合，标准 MHA 不强制正交 |
| score 大就一定导致输出层梯度消失 | Attention Softmax 可能饱和；输出 Softmax 与 CE 合并后的梯度需单独分析 |
| `exp(z)` 在所有低精度里都在 `z>11` 溢出 | 该量级主要适用于 FP16；BF16 指数范围更接近 FP32 |
| 残差让梯度无损传播 | 残差提供恒等项并改善传播，但不保证整个深网 Jacobian 恒等 |
| 现代 LLM 全部必须用 Pre-LN | Pre-LN 很常见且易优化，但仍存在多种有效归一化架构 |
| LoRA 说明原权重本身低秩 | LoRA 约束的是任务更新 `Delta-W`，不是原权重 `W` |

这些限定不是咬文嚼字。它们会直接影响你如何解释训练曲线、选择断点、判断一个 shape 正确
但语义错误的实现，以及是否把经验规律误当成架构保证。

## 最小可运行验证

下面的片段把本节几条数学结论变成可执行检查：

```python
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)

# 1. nn.Linear 的权重是 [out, in]
x = torch.randn(2, 3, 4)
linear = torch.nn.Linear(4, 5)
assert torch.allclose(linear(x), x @ linear.weight.T + linear.bias)

# 2. Stable softmax 与平移前相同
z = torch.tensor([1000.0, 999.0, 998.0])
stable = torch.softmax(z - z.max(), dim=-1)
assert torch.allclose(torch.softmax(z, dim=-1), stable)

# 3. Cross entropy 直接接收 logits
labels = torch.tensor([0])
loss = F.cross_entropy(z.unsqueeze(0), labels)
manual = torch.logsumexp(z, dim=0) - z[0]
assert torch.allclose(loss, manual)

# 4. Scaled score 保持 shape，并沿 key 轴归一化
q = torch.randn(2, 4, 3, 8)  # [B,H,T,Dh]
k = torch.randn(2, 4, 3, 8)
scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
weights = torch.softmax(scores, dim=-1)
assert scores.shape == (2, 4, 3, 3)
assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 3))

# 5. A @ B 的秩不会超过中间维 r
A = torch.randn(6, 2)
B = torch.randn(2, 5)
assert torch.linalg.matrix_rank(A @ B) <= 2

# 6. residual 的局部导数包含 identity 项
x = torch.randn(4, requires_grad=True)
y = x + x.square()
y.sum().backward()
assert torch.allclose(x.grad, 1 + 2 * x.detach())
```

仓库内更完整的对应实验：

```bash
python -m labs.lab01_attention_basics
python -m labs.lab03_pre_ln_block
python -m labs.lab05_tiny_language_model --steps 100
python -m labs.lab09_lora_linear
```

## 每个阶段的达标标准

| 阶段 | 至少应能回答 |
| --- | --- |
| Tokenizer | 文本为什么必须变成 id，PAD 为什么不能参与 loss |
| Embedding | `[B,T]` 如何变成 `[B,T,D]` |
| Attention | `QK^T` 为什么得到 token-to-token 权重 |
| Mask | 某个 token 当前允许看到哪些位置 |
| Block | Attention、FFN、残差和 Norm 各自负责什么 |
| Loss | 第 `t` 个位置到底预测哪个 token |
| Generation | 训练并行、推理逐 token 的区别 |
| KV Cache | 哪些历史计算可以复用 |
| LoRA | 为什么只训练少量矩阵也能适配任务 |
| Evaluation | 为什么训练 loss 低不等于任务正确 |

## 推荐的学习循环

每学习一个主题，按相同流程操作：

1. 先阅读对应文档，不急着运行完整训练。
2. 在纸上写出一个最小 shape 例子。
3. 运行对应 lab，确认输出与推导一致。
4. 修改一个变量，例如 head 数或序列长度。
5. 故意制造一个错误，阅读异常信息。
6. 回到完整源码，找到同一个概念的工程实现。

这种方式比连续阅读大量文章更容易形成长期记忆。

## 建议维护一张 Shape 表

例如 Decoder-Only：

```text
input ids       [B, T]
embedding       [B, T, D]
Q/K/V           [B, H, T, Dh]
attention score [B, H, T, T]
block output    [B, T, D]
logits          [B, T, V]
labels          [B, T]
```

每次调试先更新这张表。若实际 shape 与预期不同，优先修复 shape，而不是继续猜训练参数。

## 自测

完成本项目后，应能不看代码回答：

1. 为什么 causal mask 不能只在 loss 中处理？
2. 为什么 Transformer block 输入输出通常都是 `[B,T,D]`？
3. 为什么训练能一次计算所有位置，而生成必须逐 token？
4. 为什么 KV Cache 保存 K/V 而不是只保存最终 logits？
5. 为什么 LoRA 的 B 通常初始化为零？
6. 为什么工具路由必须看完全正确率，而不能只看语言模型 loss？
