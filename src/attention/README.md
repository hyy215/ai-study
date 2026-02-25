
# Attention Mechanism in TypeScript

这是一个从零开始（From Scratch）实现的 **Attention** 机制的 TypeScript 版本。

该实现基于 PyTorch 版本的逻辑，通过手写矩阵运算（如矩阵乘法、Softmax 等）来模拟深度学习框架的底层行为，旨在帮助理解 Attention 机制的数学原理。

## 目录结构

*   `selfAttention.ts`: **Self-Attention** 的核心实现，输入序列关注自身。
*   `maskedSelfAttention.ts`: **Masked Self-Attention** 实现，用于 Decoder，确保只能看到过去的信息（Look-ahead Mask）。
*   `crossAttention.ts`: **Encoder-Decoder Attention** (Cross Attention) 实现，Decoder 查询 Encoder 的信息。
*   `multiHeadAttention.ts`: **Multi-Head Attention** 实现，并行计算多个注意力头，捕捉不同子空间的特征。
*   `tensorMath.ts`: 基础数学工具库，模拟 PyTorch 的 `matmul`, `softmax`, `transpose`, `masked_fill`, `tril`, `concat` 等张量操作。

## 如何运行

确保你已经安装了项目依赖：

```bash
npm install
```

### 1. 运行 Self-Attention

```bash
npx tsx src/attention/selfAttention.ts
```

### 2. 运行 Masked Self-Attention

Masked Attention 通常用于 Decoder，确保当前 Token 只能关注到自身和之前的 Token（因果遮蔽）。

```bash
npx tsx src/attention/maskedSelfAttention.ts
```

### 3. 运行 Cross Attention (Encoder-Decoder Attention)

Cross Attention 用于 Decoder 关注 Encoder 的输出，Query 来自 Decoder，Key/Value 来自 Encoder。

```bash
npx tsx src/attention/crossAttention.ts
```

### 4. 运行 Multi-Head Attention

Multi-Head Attention 将 Embedding 维度分割成多个头，分别计算注意力，最后拼接结果。

```bash
npx tsx src/attention/multiHeadAttention.ts
```

## 核心概念

### Scaled Dot-Product Attention

标准公式：

$$
Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

1.  **Q (Query)**: 查询向量
2.  **K (Key)**: 键向量
3.  **V (Value)**: 值向量
4.  **Scaling**: 除以 $\sqrt{d_k}$ 以防止梯度消失
5.  **Masking (可选)**: 在 Softmax 之前，将不需要关注的位置（如未来的 token）设置为 $-\infty$，使其概率趋近于 0。
6.  **Softmax**: 将分数转换为概率分布

### Attention 类型对比

| 类型 | Query 来源 | Key/Value 来源 | 用途 |
| :--- | :--- | :--- | :--- |
| **Self-Attention** | 输入序列 $X$ | 输入序列 $X$ | Encoder 提取特征 |
| **Masked Self-Attention** | 输入序列 $X$ | 输入序列 $X$ (带 Mask) | Decoder 生成序列 (自回归) |
| **Cross-Attention** | Decoder 输入 | Encoder 输出 | Decoder 获取源序列信息 |
| **Multi-Head Attention** | 任意来源 | 任意来源 | 扩展注意力机制，捕捉多维度特征 |

## 参考资料

本项目代码实现参考了 DeepLearning.AI 的课程：
[Attention in Transformers: Concepts and Code in PyTorch](https://learn.deeplearning.ai/courses/attention-in-transformers-concepts-and-code-in-pytorch/information)
