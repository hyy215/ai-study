import chalk from 'chalk';
import { TensorMath, type Matrix } from './tensorMath.js';

// --- Self-Attention 核心实现 ---

class SelfAttention {
    d_model: number;
    // 模拟线性层的权重 (在实际训练中这些是可学习的参数)
    w_q: Matrix;
    w_k: Matrix;
    w_v: Matrix;

    constructor(d_model: number) {
        this.d_model = d_model;
        // 初始化权重矩阵 (d_model x d_model)
        // 注意：为了简化，这里省略了 bias
        this.w_q = TensorMath.rand(d_model, d_model);
        this.w_k = TensorMath.rand(d_model, d_model);
        this.w_v = TensorMath.rand(d_model, d_model);
    }

    /**
     * 前向传播
     * @param x 输入序列 (Batch Size 假设为 1, Shape: [seq_len, d_model])
     */
    forward(x: Matrix): { output: Matrix, attentionWeights: Matrix } {
        // 1. 生成 Query, Key, Value
        // Q = X * W_Q
        const q = TensorMath.matmul(x, this.w_q);
        // K = X * W_K
        const k = TensorMath.matmul(x, this.w_k);
        // V = X * W_V
        const v = TensorMath.matmul(x, this.w_v);

        // 2. 计算注意力分数 (Scores)
        // Scores = Q * K^T
        const k_transposed = TensorMath.transpose(k);
        let scores = TensorMath.matmul(q, k_transposed);

        // 3. 缩放 (Scaling)
        // Scores = Scores / sqrt(d_k)
        const d_k = this.d_model; // 这里简化假设 d_k = d_model
        scores = TensorMath.scale(scores, Math.sqrt(d_k));

        // 4. Softmax 归一化
        const attentionWeights = TensorMath.softmax(scores);

        // 5. 加权求和
        // Output = Weights * V
        const output = TensorMath.matmul(attentionWeights, v);

        return { output, attentionWeights };
    }
}

// --- 运行演示 ---

function main() {
    console.log(chalk.blue("=== Self-Attention TypeScript Implementation ===\n"));

    // 假设输入序列长度为 3，嵌入维度为 4
    // 例如: ["I", "love", "AI"]
    const seq_len = 3;
    const d_model = 4;

    console.log(chalk.yellow(`Input Sequence Length: ${seq_len}`));
    console.log(chalk.yellow(`Embedding Dimension (d_model): ${d_model}\n`));

    // 1. 创建模拟输入 (Batch Size = 1)
    const input = TensorMath.rand(seq_len, d_model);
    console.log("Input Matrix (X):");
    console.table(input);

    // 2. 初始化 Self-Attention 层
    const selfAttention = new SelfAttention(d_model);

    // 3. 执行前向传播
    const result = selfAttention.forward(input);

    console.log("\nAttention Weights (Softmax(QK^T / sqrt(d_k))):");
    // 保留 4 位小数以便查看
    const formattedWeights = result.attentionWeights.map(row => 
        row.map(val => Number(val.toFixed(4)))
    );
    console.table(formattedWeights);

    console.log("\nOutput Matrix (Weighted Sum of V):");
    const formattedOutput = result.output.map(row => 
        row.map(val => Number(val.toFixed(4)))
    );
    console.table(formattedOutput);
}

main();
