import chalk from 'chalk';
import { TensorMath, type Matrix } from './tensorMath.js';

// --- Masked Self-Attention 核心实现 ---

class MaskedSelfAttention {
    d_model: number;
    w_q: Matrix;
    w_k: Matrix;
    w_v: Matrix;

    constructor(d_model: number) {
        this.d_model = d_model;
        // 初始化权重矩阵
        this.w_q = TensorMath.rand(d_model, d_model);
        this.w_k = TensorMath.rand(d_model, d_model);
        this.w_v = TensorMath.rand(d_model, d_model);
    }

    /**
     * 前向传播 (带 Mask)
     * @param x 输入序列 [seq_len, d_model]
     * @param mask (可选) 外部传入的 Mask，如果不传则默认使用 Look-ahead Mask (下三角)
     */
    forward(x: Matrix, mask?: Matrix): { output: Matrix, attentionWeights: Matrix } {
        const seq_len = x.length;

        // 1. 生成 Query, Key, Value
        const q = TensorMath.matmul(x, this.w_q);
        const k = TensorMath.matmul(x, this.w_k);
        const v = TensorMath.matmul(x, this.w_v);

        // 2. 计算注意力分数 (Scores): Q * K^T
        let scores = TensorMath.matmul(q, TensorMath.transpose(k));

        // 3. 缩放 (Scaling)
        scores = TensorMath.scale(scores, Math.sqrt(this.d_model));

        // 4. 应用掩码 (Masking)
        // 如果没有传入 mask，默认使用 Look-ahead Mask (下三角矩阵)
        // 这种 Mask 常用于 Decoder，防止看到未来的 token
        if (!mask) {
            mask = TensorMath.tril(seq_len);
        }
        
        // 将 Mask 为 0 的位置填充为 -1e9 (负无穷)
        // 这样 Softmax 后这些位置的概率就会趋近于 0
        scores = TensorMath.maskedFill(scores, mask, -1e9);

        // 5. Softmax 归一化
        const attentionWeights = TensorMath.softmax(scores);

        // 6. 加权求和: Weights * V
        const output = TensorMath.matmul(attentionWeights, v);

        return { output, attentionWeights };
    }
}

// --- 运行演示 ---

function main() {
    console.log(chalk.blue("=== Masked Self-Attention TypeScript Implementation ===\n"));

    const seq_len = 4;
    const d_model = 4;

    console.log(chalk.yellow(`Sequence Length: ${seq_len}`));
    console.log(chalk.yellow(`Embedding Dimension: ${d_model}\n`));

    // 1. 创建模拟输入
    const input = TensorMath.rand(seq_len, d_model);
    console.log("Input Matrix (X):");
    console.table(input);

    // 2. 初始化 Masked Self-Attention 层
    const maskedAttention = new MaskedSelfAttention(d_model);

    // 3. 执行前向传播
    // 这里我们不传 mask，让它内部自动生成 Look-ahead Mask
    const result = maskedAttention.forward(input);

    console.log("\nLook-ahead Mask (自动生成):");
    const mask = TensorMath.tril(seq_len);
    console.table(mask);

    console.log("\nAttention Weights (After Masking & Softmax):");
    // 注意观察：这里应该是一个下三角矩阵，右上角应该全为 0
    const formattedWeights = result.attentionWeights.map(row => 
        row.map(val => Number(val.toFixed(4)))
    );
    console.table(formattedWeights);

    console.log("\nOutput Matrix:");
    const formattedOutput = result.output.map(row => 
        row.map(val => Number(val.toFixed(4)))
    );
    console.table(formattedOutput);
}

main();
