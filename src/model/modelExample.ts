import { AutoModelForCausalLM, AutoTokenizer, pipeline } from '@huggingface/transformers';
import chalk from 'chalk';

/**
 * 模型示例 CLI 工具
 * 
 * 此脚本演示了 Transformer decoder-only 模型架构
 * 以及手动进行下一个 token 的预测，灵感来自
 * "How Transformer LLMs Work" 课程的第 6 课。
 */

async function main() {
    console.log(chalk.cyan.bold('\n--- 模型示例 (TypeScript 版) ---\n'));

    // 1. 设置 - 加载模型和分词器
    // 使用 distilgpt2，因为它是此环境下最可靠的小型模型
    const modelId = 'Xenova/distilgpt2';
    
    console.log(chalk.gray(`正在加载模型和分词器 (${modelId})...`));
    const tokenizer = await AutoTokenizer.from_pretrained(modelId);
    const model = await AutoModelForCausalLM.from_pretrained(modelId, {
        dtype: 'fp32', 
    });
    console.log(chalk.green('✔ 模型和分词器加载成功！\n'));

    // 2. 生成文本响应 (使用 Pipeline)
    console.log(chalk.blue.bold('步骤 1: 使用 pipeline 生成文本'));
    const generator = await pipeline('text-generation', modelId, {
        dtype: 'fp32',
    });
    
    const prompt = "France: Paris; Germany: Berlin; Italy: Rome; Spain:";
    console.log(chalk.white(`提示词: "${prompt}"`));
    
    console.log(chalk.gray('正在生成...'));
    const output = await generator(prompt, {
        max_new_tokens: 1,
        do_sample: false,
    });
    
    // @ts-ignore
    console.log(chalk.yellow(`Pipeline 输出: "${output[0].generated_text}"\n`));

    // 3. 探索模型架构 (概念性)
    console.log(chalk.blue.bold('步骤 2: 探索模型架构'));
    // 在 Transformers.js 中，我们可以查看一些配置信息
    const config = model.config as any;
    console.log(chalk.white(`词汇表大小: ${config.vocab_size}`));
    console.log(chalk.white(`隐藏层大小 (嵌入维度): ${config.n_embd || config.hidden_size}`));
    console.log(chalk.white(`层数: ${config.n_layer || config.num_hidden_layers}\n`));

    // 4. 手动单 Token 预测
    console.log(chalk.blue.bold('步骤 3: 手动预测下一个 Token'));
    
    // a. 对输入提示词进行 Token 化
    console.log(chalk.gray('正在对提示词进行 Token 化...'));
    const inputs = await tokenizer(prompt);
    console.log(chalk.white(`输入 ID: [${inputs.input_ids.data}]`));
    
    // b. 模型推理
    console.log(chalk.gray('正在运行模型推理...'));
    // 将所有输入 (input_ids, attention_mask 等) 传递给模型
    const result = await model(inputs);
    const logits = result.logits; // 形状: [batch, sequence_length, vocab_size]
    
    console.log(chalk.white(`Logits 形状: [${logits.dims}]`));

    // c. 获取最后一个 token 的 logits
    // logits.data 是一个 Float32Array
    // 我们想要序列维度中的最后一个向量
    const [, seqLen, vocabSize] = logits.dims;
    const lastTokenLogits = logits.data.slice((seqLen - 1) * vocabSize, seqLen * vocabSize);
    
    // d. 找到概率最高的 token ID (Argmax)
    let maxVal = -Infinity;
    let tokenId = -1;
    for (let i = 0; i < lastTokenLogits.length; i++) {
        if (lastTokenLogits[i] > maxVal) {
            maxVal = lastTokenLogits[i];
            tokenId = i;
        }
    }
    
    console.log(chalk.white(`预测的 Token ID: ${tokenId}`));

    // e. 解码预测的 token ID
    const predictedToken = tokenizer.decode([tokenId]);
    console.log(chalk.green.bold(`\n预测的下一个 Token: "${predictedToken}"`));
    
    console.log(chalk.cyan.bold('\n--- 第 6 课示例结束 ---\n'));
}

main().catch(err => {
    console.error(chalk.red('发生错误:'), err);
});
