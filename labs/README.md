# Transformer Labs 使用指南

`labs/` 不是完整模型训练脚本，而是一组单变量实验。每个 Lab 应回答一个明确问题，并用
shape、数值、梯度或等价性断言把结论固定下来。

运行所有 Lab 前，先完成 `foundations/`。所有命令从仓库根目录执行。

## 推荐顺序

| Lab | 核心问题 | 关键观察 |
| --- | --- | --- |
| 00 Position | 模型如何获得顺序 | sin/cos 平移、learned 参数、RoPE 范数 |
| 01 Attention | score 如何变成信息混合 | `sqrt(Dh)`、熵、causal 权重 |
| 02 MHA | 一个 D 维表示如何拆成多头 | stride、contiguous、参数量 |
| 03 Pre-LN | 残差为何改善梯度 | 零分支时 output/gradient 恒等 |
| 04 Copy Task | Encoder-Decoder 闭环是否正确 | BOS/EOS、shift、mask、loss |
| 05 Tiny LM | Next-token 目标是否对齐 | `log(V)` 基线、规律生成 |
| 06 KV Cache | 缓存与完整前向是否等价 | 每步 cache 长度、最大误差 |
| 07 Modern Block | RMSNorm/RoPE/SwiGLU 改了什么 | 精度、offset、参数预算 |
| 08 MoE Routing | Top-K 如何分派 token | 权重和、熵、count、router 梯度 |
| 09 LoRA | 低秩分支如何启动和融合 | A/B 首步梯度、rank、fuse |
| 10 MHA/MQA/GQA | KV 共享节省了什么 | 参数量、compact cache、展开语义 |
| 11 MoE Variants | Dense/Sparse/Shared 如何取舍 | 激活 expert、权重质量、expert 梯度 |

## Lab 00：位置表示

运行：

```bash
python -m labs.lab00_positional_encoding
```

应解释：

- 正弦位置是固定函数，learned position 是参数表。
- 一对 `[sin(pw), cos(pw)]` 平移位置等价于二维旋转。
- RoPE 改变 Q/K 相位但保持每对向量范数。

建议修改：位置长度、奇数 `d_model`、RoPE offset。故意把 sin/cos 赋值长度写错，观察奇数
维度错误。

## Lab 01：Scaled Attention

```bash
python -m labs.lab01_attention_basics
```

新增统计会显示 `Dh=64` 时 raw dot-product 标准差约为 8，除以 `sqrt(64)` 后约为 1；未缩放
Softmax 熵更低、更尖锐。

不要只检查 output shape，还应检查：

```text
valid row sum == 1
future causal weights == 0
softmax axis == key axis
```

## Lab 02：Multi-Head Shape

```bash
python -m labs.lab02_multi_head_attention
```

`transpose` 后 head Tensor 通常不 contiguous。Lab 打印 stride，并验证合并前需要恢复逻辑
顺序。固定 D 时增加 head 数不会增加 Q/K/V/O 总参数量。

建议尝试 `D=12,H=3/4/5`，分别解释合法 shape 和失败原因。

## Lab 03：Pre-LN 与 Residual

```bash
python -m labs.lab03_pre_ln_block
```

Lab 将所有 residual branch 参数置零，此时：

```text
output == input
d output.sum() / d input == 1
```

这是残差恒等项的直接实验，不是只比较两个随机网络的梯度大小。随机梯度范数受初始化和
loss 影响，不能单次比较就下普遍结论。

## Lab 04：Tiny Copy Task

```bash
python -m labs.lab04_tiny_copy_task --steps 400
```

Copy Task 用确定目标隔离训练管线问题。Lab 断言 decoder input 与 labels 只错开一位，并打印
随机均匀预测的交叉熵基线 `log(V)`。

诊断顺序：先确认一个 batch 能过拟合，再增加 padding、不同 source/target 长度和 Beam
Search。

## Lab 05：Tiny Decoder-Only LM

```bash
python -m labs.lab05_tiny_language_model --steps 100
```

数据规律是：

```text
next_token = (current_token + 1) % vocab_size
```

因此可以逐元素验证 labels，而不是只观察 loss 下降。若生成错误，分别检查 shift、causal
mask、最后位置 logits 和 train/eval 状态。

## Lab 06：KV Cache

```bash
python -m labs.lab06_kv_cache
```

Lab 验证每一步 cache length 恰好增加 1，并比较：

```text
无缓存重复投影 token 数: T(T+1)/2
缓存后投影 token 数:     T
```

这不表示 decode 变成常数成本；当前 Q 仍需读取全部历史 K/V。Lab 使用 `torch.cat` 是教学
写法，生产实现应预分配或分页。

## Lab 07：现代 Block 组件

```bash
python -m labs.lab07_modern_blocks
```

Lab 中 RMSNorm 用 FP32 计算统计量再转回输入 dtype；RoPE 支持 `position_offset`，可模拟 KV
Cache decode；SwiGLU 输出参数量与经典 FFN 等预算 hidden size 会同时打印。

建议用 FP16/BF16 输入测试极大值，并比较 RMSNorm 与 LayerNorm 的输出均值。

## Lab 08：Top-K MoE

```bash
python -m labs.lab08_moe_routing
```

除了 expert assignment，Lab 还输出 router entropy、Top-1 count、balance loss 和 router
gradient norm。均匀路由下本项目 balance loss 的参考值是 1，不是 0。

把 router bias 人为推向一个 expert，观察 entropy、count 和 balance loss 如何变化。

## Lab 09：LoRA Linear

```bash
python -m labs.lab09_lora_linear
```

Lab 验证：

```text
B=0 -> initial adapter output == base output
first step -> grad(A)=0, grad(B)>0
rank(Delta-W) <= configured rank
eval dynamic adapter ~= fused Linear
```

Optimizer 只接收 `requires_grad=True` 的 A/B 参数，不把冻结基座混入参数集合。

## Lab 10：MHA、MQA 与 GQA

```bash
python -m labs.lab10_mha_mqa_gqa
```

Lab 同时打印参数量和 cache elements/token。`repeat_interleave` 只是复用普通 Attention 公式的
逻辑展开；真正 cache 保持 `[B,Hkv,T,Dh]` compact 布局。

比较结果应满足：

```text
MHA > GQA > MQA
```

这里比较的是 K/V projection 参数与 cache，不代表全模型所有参数同比例下降。

## Lab 11：MoE 变体

```bash
python -m labs.lab11_moe_variants
```

每 token 激活 expert：

```text
Dense:  4
Sparse: 2 routed
Shared: 1 shared + 2 routed
```

普通 Sparse Top-K 权重重新归一化为 1；Shared 变体保留 routed 原始概率质量再加 shared
output。Lab 还验证只有实际收到 token 的 expert 才产生 expert 参数梯度。

## 阅读输出的统一方法

每个 Lab 至少回答：

1. 输入/输出及中间 shape 是什么。
2. 哪个轴发生求和、广播或重排。
3. 哪个数值不变量应成立。
4. 哪些参数收到梯度。
5. 教学实现与生产实现的差距是什么。

只看到脚本退出码为 0 不代表理解完成。修改一个变量、预测结果、再运行验证，才是这些 Lab
的使用方式。
