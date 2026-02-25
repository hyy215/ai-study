import chalk from 'chalk';
import { TensorMath, type Matrix } from './tensorMath.js';

// --- Encoder-Decoder Attention (Cross Attention) 核心实现 ---

class CrossAttention {
    d_model: number;
    // 线性层权重
    w_q: Matrix; // 用于处理 Decoder 的输入
    w_k: Matrix; // 用于处理 Encoder 的输出
    w_v: Matrix; // 用于处理 Encoder 的输出

    constructor(d_model: number) {
        this.d_model = d_model;
        // 初始化权重矩阵
        this.w_q = TensorMath.rand(d_model, d_model);
        this.w_k = TensorMath.rand(d_model, d_model);
        this.w_v = TensorMath.rand(d_model, d_model);
    }

    /**
     * 前向传播
     * @param decoderInput 来自 Decoder 上一层的输出 [decoder_seq_len, d_model]
     * @param encoderOutput 来自 Encoder 最后一层的输出 [encoder_seq_len, d_model]
     * @param mask (可选) 针对 Padding 的 Mask，通常是对 encoderOutput 进行 mask
     */
    forward(
        decoderInput: Matrix, 
        encoderOutput: Matrix, 
        mask?: Matrix
    ): { output: Matrix, attentionWeights: Matrix } {
        
        // 1. 生成 Query, Key, Value
        // Query 来自 Decoder
        const q = TensorMath.matmul(decoderInput, this.w_q);
        
        // Key 和 Value 来自 Encoder
        const k = TensorMath.matmul(encoderOutput, this.w_k);
        const v = TensorMath.matmul(encoderOutput, this.w_v);

        // 2. 计算注意力分数 (Scores): Q * K^T
        // 结果形状: [decoder_seq_len, encoder_seq_len]
        let scores = TensorMath.matmul(q, TensorMath.transpose(k));

        // 3. 缩放 (Scaling)
        scores = TensorMath.scale(scores, Math.sqrt(this.d_model));

        // 4. 应用掩码 (Masking)
        // Cross Attention 通常只需要 Padding Mask (掩盖 Encoder 中无效的 Padding Token)
        if (mask) {
            scores = TensorMath.maskedFill(scores, mask, -1e9);
        }

        // 5. Softmax 归一化
        // 对每一行(每个 decoder token)计算其对所有 encoder token 的关注度
        const attentionWeights = TensorMath.softmax(scores);

        // 6. 加权求和: Weights * V
        // 结果形状: [decoder_seq_len, d_model]
        const output = TensorMath.matmul(attentionWeights, v);

        return { output, attentionWeights };
    }
}

// --- 运行演示 ---

function main() {
    console.log(chalk.blue("=== Encoder-Decoder Attention (Cross Attention) TypeScript Implementation ===\n"));

    const d_model = 4;
    const encoder_seq_len = 5; // 例如：源句子 "I love AI very much"
    const decoder_seq_len = 3; // 例如：目标句子 "我 爱 AI"

    console.log(chalk.yellow(`Encoder Sequence Length: ${encoder_seq_len}`));
    console.log(chalk.yellow(`Decoder Sequence Length: ${decoder_seq_len}`));
    console.log(chalk.yellow(`Embedding Dimension: ${d_model}\n`));

    // 1. 创建模拟输入
    // Encoder Output (Memory)
    const encoderOutput = TensorMath.rand(encoder_seq_len, d_model);
    // Decoder Input (Query)
    const decoderInput = TensorMath.rand(decoder_seq_len, d_model);

    console.log("Encoder Output (Key/Value Source):");
    console.table(encoderOutput);
    console.log("\nDecoder Input (Query Source):");
    console.table(decoderInput);

    // 2. 初始化 Cross Attention 层
    const crossAttention = new CrossAttention(d_model);

    // 3. 执行前向传播
    const result = crossAttention.forward(decoderInput, encoderOutput);

    console.log("\nAttention Weights (Decoder tokens attending to Encoder tokens):");
    // 形状应该是 [decoder_seq_len, encoder_seq_len]
    // 每一行代表一个 Decoder token 对所有 Encoder token 的注意力分布
    const formattedWeights = result.attentionWeights.map(row => 
        row.map(val => Number(val.toFixed(4)))
    );
    console.table(formattedWeights);

    console.log("\nOutput Matrix (Context Vectors for Decoder):");
    // 形状应该是 [decoder_seq_len, d_model]
    const formattedOutput = result.output.map(row => 
        row.map(val => Number(val.toFixed(4)))
    );
    console.table(formattedOutput);
}

main();
