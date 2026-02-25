import { AutoTokenizer } from '@huggingface/transformers';
import chalk from 'chalk';
import * as readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import { encoding_for_model } from 'tiktoken';

/**
 * 分词器 CLI 工具
 * 
 * 此脚本允许在 @huggingface/transformers 和 tiktoken 之间进行选择
 * 以对输入文本进行分词，并用交替颜色可视化 Token。
 */

// 定义一个颜色调色板以循环进行可视化 (柔和/浅色主题)
const colors = [
    chalk.bgHex('#FFB3BA').black, // 柔和粉
    chalk.bgHex('#FFDFBA').black, // 柔和橙
    chalk.bgHex('#FFFFBA').black, // 柔和黄
    chalk.bgHex('#BAFFC9').black, // 柔和绿
    chalk.bgHex('#BAE1FF').black, // 柔和蓝
    chalk.bgHex('#E6B3FF').black, // 柔和紫
];

interface TokenizerStrategy {
    name: string;
    init(): Promise<void>;
    tokenize(text: string): Promise<{ tokens: string[], ids: any[] }>;
}

class HuggingFaceTokenizer implements TokenizerStrategy {
    name = '@huggingface/transformers (Xenova/gpt2)';
    private tokenizer: any;
    private byteDecoder: Record<string, number> | null = null;

    async init() {
        console.log(chalk.gray('正在加载分词器 (Xenova/gpt2)...'));
        this.tokenizer = await AutoTokenizer.from_pretrained('Xenova/gpt2');
        
        // 尝试获取 byte_decoder 以进行正确的字节级解码
        if (this.tokenizer.decoder && this.tokenizer.decoder.byte_decoder) {
            this.byteDecoder = this.tokenizer.decoder.byte_decoder;
        }
        
        console.log(chalk.green('✔ 分词器加载成功！'));
    }

    async tokenize(text: string) {
        const { input_ids } = await this.tokenizer(text);
        // 转换为 any 以处理不同的 TypedArray 返回类型
        const ids = Array.from(input_ids.data as any);
        
        const tokens: string[] = [];
        
        // 如果我们有 byte_decoder，我们可以进行正确的流式解码
        if (this.byteDecoder) {
            // @ts-ignore
            const tokenStrings = this.tokenizer.model.convert_ids_to_tokens(ids);
            const decoder = new TextDecoder("utf-8");
            
            for (const tokenStr of tokenStrings) {
                const bytes: number[] = [];
                for (const char of tokenStr) {
                    // @ts-ignore
                    const byteVal = this.byteDecoder[char];
                    if (byteVal !== undefined) {
                        bytes.push(Number(byteVal));
                    } else {
                        // 回退
                        bytes.push(char.codePointAt(0) || 0);
                    }
                }
                const decodedStr = decoder.decode(new Uint8Array(bytes), { stream: true });
                tokens.push(decodedStr);
            }
            
             // 刷新
            const last = decoder.decode();
            if (last) {
                tokens.push(last);
                // 我们没有刷新部分的 ID，但这对于可视化来说没问题
                // 或者如果我们想要严格的 1:1，我们可以将其追加到最后一个 token
                // 目前我们就直接 push 它，但我们需要调整 ids 数组长度或处理不匹配的情况
                if (ids.length < tokens.length) {
                    ids.push(-1);
                }
            }
        } else {
            // 回退到标准解码 (可能会分割字符)
            for (let i = 0; i < ids.length; i++) {
                const id = ids[i];
                const tokenStr = this.tokenizer.decode([id as any]);
                tokens.push(tokenStr);
            }
        }
        
        return { tokens, ids };
    }
}

class TiktokenTokenizer implements TokenizerStrategy {
    name = 'tiktoken (gpt-4o)';
    private enc: any;

    async init() {
        // tiktoken 实际上不需要异步初始化，但为了保持接口一致
        // 我们使用 gpt-4o 作为 tiktoken 示例的默认值
        try {
            this.enc = encoding_for_model("gpt-4o");
            console.log(chalk.green('✔ Tiktoken 编码器 (gpt-4o) 加载成功！'));
        } catch (e) {
             console.error(chalk.red('加载 tiktoken 模型失败'), e);
             throw e;
        }
    }

    async tokenize(text: string) {
        const encoded = this.enc.encode(text);
        // encoded 是一个 Uint32Array (或类似类型)，转换为普通数组
        const ids = Array.from(encoded);
        
        const tokens: string[] = [];
        const decoder = new TextDecoder();

        for (const id of ids) {
             // tiktoken 的 decode 返回 Uint8Array
             const tokenBytes = this.enc.decode([id as number]); 
             // 将 Uint8Array 转换为字符串
             const tokenStr = decoder.decode(tokenBytes);
             tokens.push(tokenStr);
        }
        
        return { tokens, ids };
    }
}

async function main() {
    const rl = readline.createInterface({ input, output });

    try {
        console.log(chalk.cyan('选择分词器库:'));
        console.log('1. @huggingface/transformers (GPT-2)');
        console.log('2. tiktoken (GPT-4o)');
        
        const choice = await rl.question(chalk.bold('选择 (1 或 2) > '));
        
        let strategy: TokenizerStrategy;
        
        if (choice.trim() === '2') {
            strategy = new TiktokenTokenizer();
        } else {
            strategy = new HuggingFaceTokenizer();
        }

        await strategy.init();

        console.log(chalk.cyan(`\n正在使用 ${strategy.name}`));
        console.log(chalk.cyan('输入一个句子以查看它是如何被分词的。输入 "exit" 或按 Ctrl+C 退出。\n'));

        while (true) {
            const answer = await rl.question(chalk.bold('输入 > '));
           
            if (answer.trim().toLowerCase() === 'exit') {
                break;
            }

            if (!answer.trim()) {
                continue;
            }
            console.log(answer);
            const { tokens, ids } = await strategy.tokenize(answer);

            let visualOutput = '';

            for (let i = 0; i < tokens.length; i++) {
                const tokenStr = tokens[i];
                // 循环应用颜色
                const colorFn = colors[i % colors.length];
                if (colorFn) {
                    visualOutput += colorFn(tokenStr);
                }
            }

            console.log('\n' + chalk.bold('Tokens:'));
            console.log(visualOutput);
            
            console.log('\n' + chalk.bold('Token 列表:'));
            console.log(tokens.map((t, i) => `${i + 1}. ${JSON.stringify(t)} (ID: ${ids[i]})`).join('\n'));
            
            console.log(chalk.gray('\n--------------------------------------------------\n'));
        }

    } catch (error) {
        console.error(chalk.red('发生错误:'), error);
    } finally {
        rl.close();
        process.exit(0);
    }
}

main();
