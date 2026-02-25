
import chalk from 'chalk';
import { TensorMath, type Matrix } from './tensorMath.js';

// --- Multi-Head Attention (Standard Implementation) ---
// 特点：将 Embedding 维度分割成多个头，分别计算注意力
// d_head = d_model / n_head
// 这是 Transformer (Attention Is All You Need) 的标准做法

class MultiHeadAttention {
    d_model: number;
    n_head: number;
    d_head: number;

    // 线性层权重
    // 在标准实现中，我们使用单个大矩阵进行投影，然后通过 reshape/split 分头
    // W_q: [d_model, d_model]
    w_q: Matrix;
    w_k: Matrix;
    w_v: Matrix;
    w_o: Matrix; // 输出线性层

    constructor(d_model: number, n_head: number) {
        if (d_model % n_head !== 0) {
            throw new Error(`d_model (${d_model}) must be divisible by n_head (${n_head})`);
        }

        this.d_model = d_model;
        this.n_head = n_head;
        this.d_head = d_model / n_head; // 关键：维度切分

        // 初始化权重矩阵
        // 投影矩阵的维度都是 [d_model, d_model]
        this.w_q = TensorMath.rand(d_model, d_model);
        this.w_k = TensorMath.rand(d_model, d_model);
        this.w_v = TensorMath.rand(d_model, d_model);
        this.w_o = TensorMath.rand(d_model, d_model);
    }

    /**
     * 将投影后的矩阵切分为多个头 (Split Heads)
     * 模拟操作: [batch, seq_len, d_model] -> [batch, seq_len, n_head, d_head] -> transpose -> [batch, n_head, seq_len, d_head]
     * 在这里我们简化为返回一个矩阵数组: Matrix[]，其中每个 Matrix 是 [seq_len, d_head]
     */
    private splitHeads(x: Matrix): Matrix[] {
        const seq_len = x.length;
        const heads: Matrix[] = [];

        for (let h = 0; h < this.n_head; h++) {
            // 为每个头创建一个矩阵 [seq_len, d_head]
            const head: Matrix = [];
            for (let i = 0; i < seq_len; i++) {
                // 截取当前头对应的列范围
                const start = h * this.d_head;
                const end = start + this.d_head;
                // 注意：这里使用了非空断言，因为我们确保 x 的维度是正确的
                head.push(x[i]!.slice(start, end));
            }
            heads.push(head);
        }
        return heads;
    }

    /**
     * 拼接多个头的结果 (Concat Heads)
     * 模拟操作: [batch, n_head, seq_len, d_head] -> transpose -> [batch, seq_len, n_head, d_head] -> reshape -> [batch, seq_len, d_model]
     * Input: n_head x [seq_len, d_head]
     * Output: [seq_len, d_model]
     */
    private concatHeads(heads: Matrix[]): Matrix {
        // 实际上 TensorMath.concat 就是做这个事情：在最后一个维度上拼接
        return TensorMath.concat(heads);
    }

    /**
     * 单个头的 Scaled Dot-Product Attention
     */
    private attention(q: Matrix, k: Matrix, v: Matrix, mask?: Matrix): Matrix {
        // 1. Scores = Q * K^T
        let scores = TensorMath.matmul(q, TensorMath.transpose(k));
        
        // 2. Scale
        scores = TensorMath.scale(scores, Math.sqrt(this.d_head));

        // 3. Mask (如果有)
        if (mask) {
            scores = TensorMath.maskedFill(scores, mask, -1e9);
        }

        // 4. Softmax
        const weights = TensorMath.softmax(scores);

        // 5. Output = Weights * V
        return TensorMath.matmul(weights, v);
    }

    /**
     * 前向传播
     * @param x 输入序列 [seq_len, d_model]
     * @param mask (可选)
     */
    forward(x: Matrix, mask?: Matrix): { output: Matrix, concatOutput: Matrix } {
        // 1. 线性投影 (Linear Projections)
        // 将输入 x 分别投影到 Q, K, V 空间
        // Output: [seq_len, d_model]
        const q_full = TensorMath.matmul(x, this.w_q);
        const k_full = TensorMath.matmul(x, this.w_k);
        const v_full = TensorMath.matmul(x, this.w_v);

        // 2. 分头 (Split Heads)
        // 将 d_model 维度切分为 n_head 个 d_head
        // Output: n_head x [seq_len, d_head]
        const q_heads = this.splitHeads(q_full);
        const k_heads = this.splitHeads(k_full);
        const v_heads = this.splitHeads(v_full);

        // 3. 并行计算每个头的 Attention (Scaled Dot-Product Attention)
        const attentionOutputs: Matrix[] = [];
        for (let i = 0; i < this.n_head; i++) {
            const headOutput = this.attention(q_heads[i]!, k_heads[i]!, v_heads[i]!, mask);
            attentionOutputs.push(headOutput);
        }

        // 4. 拼接 (Concat)
        // 将所有头的输出拼接回 [seq_len, d_model]
        const concatOutput = this.concatHeads(attentionOutputs);

        // 5. 最终线性变换 (Output Linear)
        // 混合各个头的信息
        // Output: [seq_len, d_model]
        const output = TensorMath.matmul(concatOutput, this.w_o);

        return { output, concatOutput };
    }
}

// --- 运行演示 ---

function main() {
    console.log(chalk.blue("=== Multi-Head Attention (Standard Split) Implementation ===\n"));

    const seq_len = 3;
    const d_model = 8; // 必须能被 n_head 整除
    const n_head = 2;
    const d_head = d_model / n_head; // 4

    console.log(chalk.yellow(`Sequence Length: ${seq_len}`));
    console.log(chalk.yellow(`Embedding Dimension (d_model): ${d_model}`));
    console.log(chalk.yellow(`Number of Heads (n_head): ${n_head}`));
    console.log(chalk.yellow(`Head Dimension (d_head): ${d_head} (d_model / n_head)`));
    console.log(chalk.yellow(`Concat Dimension: ${n_head * d_head} (Same as d_model)\n`));

    // 1. 创建模拟输入
    const input = TensorMath.rand(seq_len, d_model);
    console.log("Input Matrix (X):");
    // 格式化输出，保留4位小数
    const formattedInput = input.map(row => row.map(val => Number(val.toFixed(4))));
    console.table(formattedInput);

    // 2. 初始化 Multi-Head Attention 层
    const multiHeadAttention = new MultiHeadAttention(d_model, n_head);

    // 3. 执行前向传播
    const result = multiHeadAttention.forward(input);

    console.log("\nConcatenated Heads Output (Before Final Linear):");
    // 形状应该是 [seq_len, d_model]
    const formattedConcat = result.concatOutput.map(row => 
        row.map(val => Number(val.toFixed(4)))
    );
    console.table(formattedConcat);

    console.log("\nMulti-Head Attention Output (After Final Linear):");
    // 形状应该恢复为 [seq_len, d_model]
    const formattedOutput = result.output.map(row => 
        row.map(val => Number(val.toFixed(4)))
    );
    console.table(formattedOutput);
}

main();
